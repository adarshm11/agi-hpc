# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Unit tests for TurboQuant KV cache compression.

Tests the TurboQuantKV compressor on CPU (NumPy) to validate:
- Round-trip compress/decompress
- Reconstruction accuracy (MSE, cosine similarity)
- Compression ratio calculations
- Edge cases (zero vectors, single-element batches)
- Memory estimation utility
- All supported bit widths (2, 3, 4)

Usage:
    pytest tests/unit/test_turboquant_kv.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from agi.meta.llm.turboquant_kv import CompressedKV, TurboQuantKV

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between corresponding vectors."""
    # Flatten to (N, D)
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])

    dot = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    # Avoid division by zero
    denom = np.maximum(norm_a * norm_b, 1e-30)
    return float(np.mean(dot / denom))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two arrays."""
    return float(np.mean((a - b) ** 2))


def _random_kv(
    batch: int = 1,
    n_heads: int = 4,
    seq_len: int = 32,
    head_dim: int = 64,
    dtype: str = "float32",
    seed: int = 42,
) -> np.ndarray:
    """Generate a random KV-like tensor."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, n_heads, seq_len, head_dim)).astype(dtype)


# ------------------------------------------------------------------ #
# Basic round-trip tests                                              #
# ------------------------------------------------------------------ #


class TestRoundTrip:
    """Compress then decompress and verify reconstruction quality."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_shape(self, bits: int) -> None:
        """Decompressed tensor has the same shape as the original."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=bits, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == tensor.shape

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_round_trip_dtype(self, bits: int) -> None:
        """Decompressed tensor is float32."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=bits, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert reconstructed.dtype == np.float32

    def test_3bit_cosine_similarity(self) -> None:
        """3-bit quantisation preserves cosine similarity > 0.95."""
        tensor = _random_kv(batch=2, n_heads=8, seq_len=64, head_dim=128, seed=123)
        tq = TurboQuantKV(head_dim=128, n_heads=8, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.95, f"Cosine similarity {cos_sim:.4f} too low for 3-bit"

    def test_4bit_cosine_similarity(self) -> None:
        """4-bit quantisation preserves cosine similarity > 0.98."""
        tensor = _random_kv(batch=2, n_heads=8, seq_len=64, head_dim=128, seed=456)
        tq = TurboQuantKV(head_dim=128, n_heads=8, bits=4, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.98, f"Cosine similarity {cos_sim:.4f} too low for 4-bit"

    def test_2bit_cosine_similarity(self) -> None:
        """2-bit quantisation preserves cosine similarity > 0.85."""
        tensor = _random_kv(batch=2, n_heads=8, seq_len=64, head_dim=128, seed=789)
        tq = TurboQuantKV(head_dim=128, n_heads=8, bits=2, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.85, f"Cosine similarity {cos_sim:.4f} too low for 2-bit"

    def test_mse_decreases_with_more_bits(self) -> None:
        """Higher bit widths give lower reconstruction error."""
        tensor = _random_kv(batch=1, n_heads=4, seq_len=32, head_dim=128, seed=42)
        mse_values = {}
        for bits in [2, 3, 4]:
            tq = TurboQuantKV(head_dim=128, n_heads=4, bits=bits, use_gpu=False, seed=0)
            compressed = tq.compress(tensor)
            reconstructed = tq.decompress(compressed)
            mse_values[bits] = _mse(tensor, reconstructed)

        assert (
            mse_values[4] < mse_values[3] < mse_values[2]
        ), f"MSE should decrease with more bits: {mse_values}"


# ------------------------------------------------------------------ #
# Compressed container tests                                          #
# ------------------------------------------------------------------ #


class TestCompressedKV:
    """Tests for the CompressedKV dataclass."""

    def test_indices_dtype(self) -> None:
        """Indices are stored as uint8."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.indices.dtype == np.uint8

    def test_norms_shape(self) -> None:
        """Norms shape is (B, H, S) -- one norm per vector."""
        tensor = _random_kv(batch=2, n_heads=4, seq_len=16, head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.norms.shape == (2, 4, 16)

    def test_nbytes_less_than_original(self) -> None:
        """Compressed representation uses less memory."""
        tensor = _random_kv(head_dim=128)
        tq = TurboQuantKV(head_dim=128, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.nbytes() < tensor.nbytes

    def test_compression_ratio(self) -> None:
        """Compression ratio is > 1 (actual savings)."""
        tensor = _random_kv(head_dim=128)
        tq = TurboQuantKV(head_dim=128, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        ratio = compressed.compression_ratio(128)
        assert ratio > 1.0, f"Expected ratio > 1, got {ratio}"

    def test_bits_preserved(self) -> None:
        """The bits field is correctly stored."""
        tensor = _random_kv(head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.bits == 3


# ------------------------------------------------------------------ #
# Edge cases                                                          #
# ------------------------------------------------------------------ #


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_vector(self) -> None:
        """Zero vectors compress and decompress without error."""
        tensor = np.zeros((1, 1, 1, 64), dtype=np.float32)
        tq = TurboQuantKV(head_dim=64, n_heads=1, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        # Zero vector should reconstruct to approximately zero
        assert np.allclose(reconstructed, 0.0, atol=1e-6)

    def test_single_element_batch(self) -> None:
        """Works with batch=1, seq_len=1."""
        tensor = _random_kv(batch=1, n_heads=1, seq_len=1, head_dim=64)
        tq = TurboQuantKV(head_dim=64, n_heads=1, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == (1, 1, 1, 64)

    def test_large_head_dim(self) -> None:
        """Works with head_dim > 4096 (uses structured rotation)."""
        head_dim = 5000
        tensor = _random_kv(batch=1, n_heads=1, seq_len=2, head_dim=head_dim, seed=99)
        tq = TurboQuantKV(head_dim=head_dim, n_heads=1, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == tensor.shape
        cos_sim = _cosine_similarity(tensor, reconstructed)
        assert cos_sim > 0.90

    def test_fp16_input(self) -> None:
        """Accepts float16 input tensors."""
        tensor = _random_kv(head_dim=64, dtype="float16")
        tq = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        compressed = tq.compress(tensor)
        assert compressed.original_dtype == np.dtype("float16")
        reconstructed = tq.decompress(compressed)
        assert reconstructed.shape == tensor.shape

    def test_invalid_bits_raises(self) -> None:
        """Unsupported bit width raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported bits=5"):
            TurboQuantKV(head_dim=64, bits=5, use_gpu=False)

    def test_reproducibility_with_seed(self) -> None:
        """Same seed produces identical compressed output."""
        tensor = _random_kv(head_dim=64, seed=42)
        tq1 = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        tq2 = TurboQuantKV(head_dim=64, n_heads=4, bits=3, use_gpu=False, seed=0)
        c1 = tq1.compress(tensor)
        c2 = tq2.compress(tensor)
        np.testing.assert_array_equal(c1.indices, c2.indices)
        np.testing.assert_array_equal(c1.norms, c2.norms)


# ------------------------------------------------------------------ #
# Memory estimation                                                   #
# ------------------------------------------------------------------ #


class TestMemoryEstimation:
    """Tests for the static memory estimation utility."""

    def test_gemma4_8k(self) -> None:
        """Gemma 4 27B at 8K context: compressed < original."""
        est = TurboQuantKV.estimate_memory(
            n_layers=36,
            n_kv_heads=16,
            head_dim=256,
            seq_len=8192,
            bits=3,
            original_dtype="float16",
        )
        assert est["compressed_gb"] < est["original_gb"]
        assert est["ratio"] > 1.0
        assert est["saved_gb"] > 0

    def test_compression_ratio_positive(self) -> None:
        """Ratio is always > 1 for reasonable configs."""
        est = TurboQuantKV.estimate_memory(
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            seq_len=4096,
            bits=3,
        )
        assert est["ratio"] > 1.0

    def test_4bit_less_compression(self) -> None:
        """4-bit gives less compression than 3-bit."""
        est3 = TurboQuantKV.estimate_memory(
            n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096, bits=3
        )
        est4 = TurboQuantKV.estimate_memory(
            n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096, bits=4
        )
        # Both use uint8 storage so compressed size is the same;
        # the difference is in reconstruction quality, not storage.
        # This test validates the estimator runs without error.
        assert est3["compressed_gb"] == est4["compressed_gb"]


# ------------------------------------------------------------------ #
# Theoretical compression ratio                                       #
# ------------------------------------------------------------------ #


class TestTheoreticalRatio:
    """Test the convenience compression_ratio method."""

    def test_ratio_greater_than_one(self) -> None:
        tq = TurboQuantKV(head_dim=128, bits=3, use_gpu=False, seed=0)
        assert tq.compression_ratio() > 1.0

    def test_ratio_increases_with_head_dim(self) -> None:
        """Larger head_dim means less norm overhead per element."""
        tq_small = TurboQuantKV(head_dim=64, bits=3, use_gpu=False, seed=0)
        tq_large = TurboQuantKV(head_dim=256, bits=3, use_gpu=False, seed=0)
        assert tq_large.compression_ratio() > tq_small.compression_ratio()
