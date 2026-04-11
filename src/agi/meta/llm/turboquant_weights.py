# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
TurboQuant model weight compression via SVD + PolarQuant.

Compresses transformer weight matrices by factorizing with truncated SVD,
then applying TurboQuant (PolarQuant + bit-packing) to the resulting
factors.  The combination achieves higher compression than GGUF Q4
while preserving model quality.

Pipeline::

    W (d_out x d_in, fp16)
      -> SVD:  U (d_out x k)  S (k,)  Vt (k x d_in)
      -> TQ3:  compress(U), compress(Vt), keep S as fp32
      -> Store CompressedWeight

Inference (factored matmul, avoids materializing full W)::

    x (batch, d_in)
      -> h = x @ Vt.T      (batch, k)
      -> h = h * S          (batch, k)
      -> out = h @ U.T      (batch, d_out)

Theory:
    - Eckart-Young-Mirsky theorem: truncated SVD is the optimal rank-k
      approximation in both Frobenius and spectral norms.
    - Zandieh et al. (ICLR 2026): random rotation decorrelates
      dimensions for near-optimal scalar quantization.
    - The SVD factors U and Vt have orthonormal columns/rows, making
      their entries approximately i.i.d. Gaussian after rotation --
      ideal for the Lloyd-Max codebook.

Reuses :class:`TurboQuantKV` from ``turboquant_kv.py`` for the
rotation + quantization + bit-packing pipeline (shape-agnostic).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from agi.meta.llm.turboquant_kv import CompressedKV, TurboQuantKV

logger = logging.getLogger(__name__)

try:
    import scipy.linalg

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

_DEFAULT_SKIP_PATTERNS: List[str] = [
    r".*embed.*",
    r".*lm_head.*",
    r".*norm.*",
    r".*bias$",
]


@dataclass
class WeightCompressionConfig:
    """Configuration for SVD + TurboQuant weight compression.

    Attributes:
        energy_threshold: Fraction of Frobenius-norm energy to retain
            when selecting the SVD truncation rank (0.0-1.0).
        bits: TurboQuant bit width for factor quantization (2, 3, or 4).
        min_rank: Floor on SVD rank per layer.
        max_rank_ratio: Ceiling as fraction of min(d_out, d_in).
        pack_factors: Bit-pack quantized factors for max compression.
        skip_patterns: Regex patterns for layer names to skip.
        min_matrix_elements: Minimum weight elements to bother compressing.
        seed: Random seed for TurboQuant rotation matrices.
    """

    energy_threshold: float = 0.95
    bits: int = 3
    min_rank: int = 16
    max_rank_ratio: float = 0.5
    pack_factors: bool = True
    skip_patterns: List[str] = field(
        default_factory=lambda: list(_DEFAULT_SKIP_PATTERNS)
    )
    min_matrix_elements: int = 4096
    seed: int = 42


# ------------------------------------------------------------------ #
# Compressed data structures                                          #
# ------------------------------------------------------------------ #


@dataclass
class CompressedWeight:
    """A single compressed weight matrix (SVD factors + TurboQuant).

    Stores the truncated SVD factors U_k, S_k, Vt_k where U and Vt
    are optionally TurboQuant-compressed.
    """

    U_compressed: Union[CompressedKV, np.ndarray]
    S: np.ndarray  # (k,) singular values, always fp32
    Vt_compressed: Union[CompressedKV, np.ndarray]
    rank: int
    original_shape: Tuple[int, int]
    original_dtype: np.dtype
    energy_retained: float
    quantized: bool  # True if TQ was applied to factors

    # Shapes needed for decompression reshape
    U_shape: Tuple[int, int] = (0, 0)  # (d_out, k)
    Vt_shape: Tuple[int, int] = (0, 0)  # (k, d_in)

    def nbytes(self) -> int:
        """Total bytes of the compressed representation."""
        s_bytes = self.S.nbytes
        if self.quantized:
            u_bytes = self.U_compressed.nbytes()
            vt_bytes = self.Vt_compressed.nbytes()
        else:
            u_bytes = self.U_compressed.nbytes
            vt_bytes = self.Vt_compressed.nbytes
        return u_bytes + s_bytes + vt_bytes

    def original_nbytes(self) -> int:
        """Bytes of the original uncompressed weight."""
        return int(np.prod(self.original_shape)) * self.original_dtype.itemsize

    def compression_ratio(self) -> float:
        """Ratio of original size to compressed size."""
        return self.original_nbytes() / max(self.nbytes(), 1)


@dataclass
class CompressedModel:
    """An entire model with compressed weight matrices."""

    weights: Dict[str, CompressedWeight]
    uncompressed: Dict[str, np.ndarray]
    config: WeightCompressionConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_original_bytes(self) -> int:
        """Total bytes if all weights were uncompressed."""
        total = sum(cw.original_nbytes() for cw in self.weights.values())
        total += sum(v.nbytes for v in self.uncompressed.values())
        return total

    def total_compressed_bytes(self) -> int:
        """Total bytes of the compressed model."""
        total = sum(cw.nbytes() for cw in self.weights.values())
        total += sum(v.nbytes for v in self.uncompressed.values())
        return total

    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        return self.total_original_bytes() / max(self.total_compressed_bytes(), 1)

    def summary(self) -> Dict[str, Any]:
        """Human-readable compression summary."""
        return {
            "compressed_layers": len(self.weights),
            "uncompressed_layers": len(self.uncompressed),
            "original_mb": self.total_original_bytes() / (1024 * 1024),
            "compressed_mb": self.total_compressed_bytes() / (1024 * 1024),
            "ratio": self.compression_ratio(),
            "savings_pct": (1 - 1 / self.compression_ratio()) * 100,
        }


# ------------------------------------------------------------------ #
# Main engine                                                         #
# ------------------------------------------------------------------ #


class TurboQuantWeights:
    """SVD + TurboQuant weight compression engine.

    Compresses weight matrices by truncated SVD factorization followed
    by TurboQuant quantization of the SVD factors.

    Args:
        config: Compression configuration.
    """

    def __init__(self, config: Optional[WeightCompressionConfig] = None) -> None:
        self.config = config or WeightCompressionConfig()
        self._compiled_skip = [re.compile(p) for p in self.config.skip_patterns]

    # ------------------------------------------------------------------ #
    # Rank selection                                                      #
    # ------------------------------------------------------------------ #

    def select_rank(
        self,
        singular_values: np.ndarray,
        energy_threshold: Optional[float] = None,
    ) -> int:
        """Select SVD truncation rank from cumulative energy.

        Args:
            singular_values: Singular values in descending order.
            energy_threshold: Override config energy threshold.

        Returns:
            Rank k such that sum(S[:k]^2) / sum(S^2) >= threshold.
        """
        threshold = energy_threshold or self.config.energy_threshold
        S2 = singular_values.astype(np.float64) ** 2
        total = S2.sum()
        if total < 1e-30:
            return self.config.min_rank

        cumulative = np.cumsum(S2) / total
        k = int(np.searchsorted(cumulative, threshold)) + 1

        max_rank = int(self.config.max_rank_ratio * len(singular_values))
        k = max(self.config.min_rank, min(k, max_rank))
        return k

    # ------------------------------------------------------------------ #
    # Single weight compression                                           #
    # ------------------------------------------------------------------ #

    def compress_weight(
        self,
        weight: np.ndarray,
        name: str = "",
    ) -> CompressedWeight:
        """Compress a single weight matrix via SVD + TurboQuant.

        Args:
            weight: 2-D weight matrix (d_out, d_in).
            name: Layer name (for logging).

        Returns:
            CompressedWeight with quantized SVD factors.
        """
        if weight.ndim != 2:
            raise ValueError(f"Expected 2-D weight, got shape {weight.shape}")

        d_out, d_in = weight.shape
        original_dtype = weight.dtype
        W = weight.astype(np.float32)

        # --- SVD ---
        if _HAS_SCIPY:
            U, S, Vt = scipy.linalg.svd(W, full_matrices=False, lapack_driver="gesdd")
        else:
            U, S, Vt = np.linalg.svd(W, full_matrices=False)

        # --- Rank selection ---
        k = self.select_rank(S)
        U_k = np.ascontiguousarray(U[:, :k])  # (d_out, k)
        S_k = S[:k].astype(np.float32)
        Vt_k = np.ascontiguousarray(Vt[:k, :])  # (k, d_in)

        energy = float(np.sum(S_k**2) / max(np.sum(S**2), 1e-30))

        # --- TurboQuant on factors ---
        tq_u = TurboQuantKV(
            head_dim=k,
            n_heads=1,
            bits=self.config.bits,
            use_gpu=False,
            seed=self.config.seed,
        )
        tq_vt = TurboQuantKV(
            head_dim=d_in,
            n_heads=1,
            bits=self.config.bits,
            use_gpu=False,
            seed=self.config.seed + 1,
        )

        # Reshape to (1, 1, n_vectors, dim) for TQ API
        U_4d = U_k.reshape(1, 1, d_out, k)
        Vt_4d = Vt_k.reshape(1, 1, k, d_in)

        U_comp = tq_u.compress(U_4d, packed=self.config.pack_factors)
        Vt_comp = tq_vt.compress(Vt_4d, packed=self.config.pack_factors)

        cw = CompressedWeight(
            U_compressed=U_comp,
            S=S_k,
            Vt_compressed=Vt_comp,
            rank=k,
            original_shape=(d_out, d_in),
            original_dtype=np.dtype(original_dtype),
            energy_retained=energy,
            quantized=True,
            U_shape=(d_out, k),
            Vt_shape=(k, d_in),
        )

        logger.info(
            "[tq-weights] %s: (%d, %d) -> rank %d (%.1f%% energy), "
            "%.1fx compression",
            name or "weight",
            d_out,
            d_in,
            k,
            energy * 100,
            cw.compression_ratio(),
        )

        return cw

    # ------------------------------------------------------------------ #
    # Decompression                                                       #
    # ------------------------------------------------------------------ #

    def decompress_weight(self, cw: CompressedWeight) -> np.ndarray:
        """Reconstruct an approximate weight matrix from compressed form.

        Args:
            cw: Compressed weight from :meth:`compress_weight`.

        Returns:
            Reconstructed weight matrix (d_out, d_in) in float32.
        """
        U_k, Vt_k = self._decompress_factors(cw)
        W_approx = (U_k * cw.S[np.newaxis, :]) @ Vt_k
        return W_approx

    def compressed_linear(
        self,
        x: np.ndarray,
        cw: CompressedWeight,
    ) -> np.ndarray:
        """Compute a linear transform using factored matmul.

        Equivalent to ``x @ W.T`` but avoids materializing the full
        weight matrix, reducing peak memory.

        Args:
            x: Input tensor (..., d_in).
            cw: Compressed weight with shape (d_out, d_in).

        Returns:
            Output tensor (..., d_out).
        """
        U_k, Vt_k = self._decompress_factors(cw)
        # x @ W.T = x @ (U @ diag(S) @ Vt).T = x @ Vt.T @ diag(S) @ U.T
        h = x @ Vt_k.T  # (..., k)
        h = h * cw.S  # (..., k) — broadcast multiply
        return h @ U_k.T  # (..., d_out)

    def _decompress_factors(
        self, cw: CompressedWeight
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompress U and Vt factors from a CompressedWeight."""
        if cw.quantized:
            tq_u = TurboQuantKV(
                head_dim=cw.rank,
                n_heads=1,
                bits=self.config.bits,
                use_gpu=False,
                seed=self.config.seed,
            )
            tq_vt = TurboQuantKV(
                head_dim=cw.original_shape[1],
                n_heads=1,
                bits=self.config.bits,
                use_gpu=False,
                seed=self.config.seed + 1,
            )
            U_k = tq_u.decompress(cw.U_compressed).reshape(cw.U_shape)
            Vt_k = tq_vt.decompress(cw.Vt_compressed).reshape(cw.Vt_shape)
        else:
            U_k = cw.U_compressed.reshape(cw.U_shape)
            Vt_k = cw.Vt_compressed.reshape(cw.Vt_shape)
        return U_k, Vt_k

    # ------------------------------------------------------------------ #
    # Model-level compression                                             #
    # ------------------------------------------------------------------ #

    def _should_skip(self, name: str, tensor: np.ndarray) -> bool:
        """Check if a layer should be skipped."""
        if tensor.ndim != 2:
            return True
        if tensor.size < self.config.min_matrix_elements:
            return True
        for pat in self._compiled_skip:
            if pat.match(name):
                return True
        return False

    def compress_state_dict(
        self,
        state_dict: Dict[str, np.ndarray],
    ) -> CompressedModel:
        """Compress all eligible weight matrices in a state dict.

        Args:
            state_dict: Mapping of layer names to weight arrays.
                Values can be numpy arrays or (if torch is available)
                torch tensors which will be converted automatically.

        Returns:
            CompressedModel with compressed and uncompressed layers.
        """
        weights: Dict[str, CompressedWeight] = {}
        uncompressed: Dict[str, np.ndarray] = {}

        for name, tensor in state_dict.items():
            # Convert torch tensors if present
            arr = self._to_numpy(tensor)

            if self._should_skip(name, arr):
                uncompressed[name] = arr
                logger.debug("[tq-weights] Skipping %s (%s)", name, arr.shape)
                continue

            weights[name] = self.compress_weight(arr, name=name)

        model = CompressedModel(
            weights=weights,
            uncompressed=uncompressed,
            config=self.config,
        )

        summary = model.summary()
        logger.info(
            "[tq-weights] Model compressed: %d layers, %.1f MB -> %.1f MB (%.1fx)",
            summary["compressed_layers"],
            summary["original_mb"],
            summary["compressed_mb"],
            summary["ratio"],
        )

        return model

    # ------------------------------------------------------------------ #
    # PyTorch model patching                                              #
    # ------------------------------------------------------------------ #

    def patch_torch_model(
        self,
        model: Any,
        compressed: CompressedModel,
    ) -> Any:
        """Replace nn.Linear layers with CompressedLinear modules.

        Args:
            model: A torch.nn.Module.
            compressed: CompressedModel from :meth:`compress_state_dict`.

        Returns:
            The patched model (modified in place).
        """
        import torch  # noqa: F811 — lazy import

        for name, cw in compressed.weights.items():
            # Navigate to parent module
            parts = name.replace(".weight", "").split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            getattr(parent, attr)  # verify exists

            bias = None
            bias_name = name.replace(".weight", ".bias")
            if bias_name in compressed.uncompressed:
                bias = torch.from_numpy(compressed.uncompressed[bias_name])

            new_module = CompressedLinear(cw, self, bias=bias)
            setattr(parent, attr, new_module)
            logger.debug("[tq-weights] Patched %s", name)

        return model

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #

    def save(self, compressed: CompressedModel, path: Union[str, Path]) -> None:
        """Save a compressed model to disk.

        Args:
            compressed: CompressedModel to save.
            path: Output file path (.npz).
        """
        import json

        arrays: Dict[str, np.ndarray] = {}
        meta: Dict[str, Any] = {
            "config": {
                "energy_threshold": compressed.config.energy_threshold,
                "bits": compressed.config.bits,
                "min_rank": compressed.config.min_rank,
                "max_rank_ratio": compressed.config.max_rank_ratio,
                "pack_factors": compressed.config.pack_factors,
                "seed": compressed.config.seed,
            },
            "compressed_layers": {},
            "uncompressed_layers": list(compressed.uncompressed.keys()),
        }

        for name, cw in compressed.weights.items():
            prefix = f"c_{name}"
            arrays[f"{prefix}_S"] = cw.S
            if cw.quantized:
                arrays[f"{prefix}_U_indices"] = cw.U_compressed.indices
                arrays[f"{prefix}_U_norms"] = cw.U_compressed.norms
                arrays[f"{prefix}_Vt_indices"] = cw.Vt_compressed.indices
                arrays[f"{prefix}_Vt_norms"] = cw.Vt_compressed.norms
            else:
                arrays[f"{prefix}_U"] = cw.U_compressed
                arrays[f"{prefix}_Vt"] = cw.Vt_compressed

            meta["compressed_layers"][name] = {
                "rank": cw.rank,
                "original_shape": list(cw.original_shape),
                "original_dtype": str(cw.original_dtype),
                "energy_retained": cw.energy_retained,
                "quantized": cw.quantized,
                "U_shape": list(cw.U_shape),
                "Vt_shape": list(cw.Vt_shape),
                "bits": cw.U_compressed.bits if cw.quantized else 0,
                "packed": cw.U_compressed.packed if cw.quantized else False,
                "U_n_values": cw.U_compressed.n_values if cw.quantized else 0,
                "U_idx_shape": list(cw.U_compressed.shape) if cw.quantized else [],
                "Vt_n_values": cw.Vt_compressed.n_values if cw.quantized else 0,
                "Vt_idx_shape": list(cw.Vt_compressed.shape) if cw.quantized else [],
            }

        for name, arr in compressed.uncompressed.items():
            arrays[f"u_{name}"] = arr

        arrays["__metadata__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )

        np.savez_compressed(str(path), **arrays)
        logger.info("[tq-weights] Saved to %s", path)

    def load(self, path: Union[str, Path]) -> CompressedModel:
        """Load a compressed model from disk.

        Args:
            path: Path to .npz file saved by :meth:`save`.

        Returns:
            CompressedModel.
        """
        import json

        data = np.load(str(path), allow_pickle=False)
        meta = json.loads(bytes(data["__metadata__"]))

        cfg = WeightCompressionConfig(**meta["config"])

        weights: Dict[str, CompressedWeight] = {}
        for name, layer_meta in meta["compressed_layers"].items():
            prefix = f"c_{name}"
            S = data[f"{prefix}_S"]

            if layer_meta["quantized"]:
                U_comp = CompressedKV(
                    indices=data[f"{prefix}_U_indices"],
                    norms=data[f"{prefix}_U_norms"],
                    bits=layer_meta["bits"],
                    original_dtype=np.dtype(np.float32),
                    packed=layer_meta["packed"],
                    n_values=layer_meta["U_n_values"],
                    shape=tuple(layer_meta["U_idx_shape"]),
                )
                Vt_comp = CompressedKV(
                    indices=data[f"{prefix}_Vt_indices"],
                    norms=data[f"{prefix}_Vt_norms"],
                    bits=layer_meta["bits"],
                    original_dtype=np.dtype(np.float32),
                    packed=layer_meta["packed"],
                    n_values=layer_meta["Vt_n_values"],
                    shape=tuple(layer_meta["Vt_idx_shape"]),
                )
            else:
                U_comp = data[f"{prefix}_U"]
                Vt_comp = data[f"{prefix}_Vt"]

            weights[name] = CompressedWeight(
                U_compressed=U_comp,
                S=S,
                Vt_compressed=Vt_comp,
                rank=layer_meta["rank"],
                original_shape=tuple(layer_meta["original_shape"]),
                original_dtype=np.dtype(layer_meta["original_dtype"]),
                energy_retained=layer_meta["energy_retained"],
                quantized=layer_meta["quantized"],
                U_shape=tuple(layer_meta["U_shape"]),
                Vt_shape=tuple(layer_meta["Vt_shape"]),
            )

        uncompressed: Dict[str, np.ndarray] = {}
        for name in meta["uncompressed_layers"]:
            uncompressed[name] = data[f"u_{name}"]

        return CompressedModel(
            weights=weights,
            uncompressed=uncompressed,
            config=cfg,
        )

    # ------------------------------------------------------------------ #
    # Static estimation                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_compression(
        d_out: int,
        d_in: int,
        rank: int,
        bits: int = 3,
        original_dtype: str = "float16",
    ) -> Dict[str, float]:
        """Estimate compression ratio without running SVD.

        Args:
            d_out: Output dimension.
            d_in: Input dimension.
            rank: SVD truncation rank.
            bits: TurboQuant bit width.
            original_dtype: Original weight dtype.

        Returns:
            Dict with original_bytes, compressed_bytes, ratio, savings_pct.
        """
        elem_size = 2 if original_dtype == "float16" else 4
        original = d_out * d_in * elem_size

        # S: always fp32
        s_bytes = rank * 4

        # U factors: d_out vectors of dim k
        u_bits = d_out * rank * bits
        u_packed = (u_bits + 7) // 8
        u_norms = d_out * 4  # fp32 per vector

        # Vt factors: k vectors of dim d_in
        vt_bits = rank * d_in * bits
        vt_packed = (vt_bits + 7) // 8
        vt_norms = rank * 4

        compressed = s_bytes + u_packed + u_norms + vt_packed + vt_norms

        return {
            "original_bytes": original,
            "compressed_bytes": compressed,
            "ratio": original / max(compressed, 1),
            "savings_pct": (1 - compressed / original) * 100,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        """Convert a tensor to numpy, handling torch tensors."""
        if isinstance(tensor, np.ndarray):
            return tensor
        # torch.Tensor
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)


# ------------------------------------------------------------------ #
# PyTorch nn.Module for compressed inference                          #
# ------------------------------------------------------------------ #


class CompressedLinear:
    """Drop-in replacement for nn.Linear using compressed weights.

    Decompresses SVD factors on each forward pass and performs
    factored matmul to avoid materializing the full weight matrix.

    Note: This is a plain class, not an nn.Module, to avoid requiring
    torch at import time. Use :meth:`TurboQuantWeights.patch_torch_model`
    which wraps this in an actual Module.
    """

    def __init__(
        self,
        compressed_weight: CompressedWeight,
        engine: TurboQuantWeights,
        bias: Any = None,
    ) -> None:
        self.cw = compressed_weight
        self.engine = engine
        self._bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.engine.compressed_linear(x, self.cw)
        if self._bias is not None:
            bias = self.engine._to_numpy(self._bias)
            out = out + bias
        return out
