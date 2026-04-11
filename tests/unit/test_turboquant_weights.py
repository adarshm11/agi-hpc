# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for TurboQuant model weight compression (SVD + PolarQuant).

Tests compression quality, rank selection, factored matmul correctness,
model-level compression, serialization, and edge cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from agi.meta.llm.turboquant_weights import (
    CompressedModel,
    TurboQuantWeights,
    WeightCompressionConfig,
)

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if norm < 1e-30:
        return 1.0
    return float(dot / norm)


def _frobenius_relative_error(original: np.ndarray, approx: np.ndarray) -> float:
    """Relative Frobenius norm error: ||W - W_hat||_F / ||W||_F."""
    orig_norm = np.linalg.norm(original)
    if orig_norm < 1e-30:
        return 0.0
    return float(np.linalg.norm(original - approx) / orig_norm)


def _random_weight(d_out: int, d_in: int, seed: int = 42) -> np.ndarray:
    """Generate a random weight matrix with realistic singular value decay."""
    rng = np.random.default_rng(seed)
    # Create a matrix with decaying singular values (typical for trained models)
    U = np.linalg.qr(rng.standard_normal((d_out, min(d_out, d_in))))[0]
    Vt = np.linalg.qr(rng.standard_normal((d_in, min(d_out, d_in))))[0].T
    # Zipf-like decay: s_i ~ 1/i^0.5
    k = min(d_out, d_in)
    S = 1.0 / np.sqrt(np.arange(1, k + 1))
    W = (U[:, :k] * S[np.newaxis, :]) @ Vt[:k, :]
    return W.astype(np.float32)


def _low_rank_weight(d_out: int, d_in: int, rank: int, seed: int = 42) -> np.ndarray:
    """Generate an exactly rank-r weight matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d_out, rank)).astype(np.float32)
    B = rng.standard_normal((rank, d_in)).astype(np.float32)
    return A @ B


# ------------------------------------------------------------------ #
# Test: Rank Selection                                                #
# ------------------------------------------------------------------ #


class TestRankSelection:
    """Tests for SVD rank selection via energy threshold."""

    def test_energy_95_percent(self) -> None:
        config = WeightCompressionConfig(min_rank=1, max_rank_ratio=1.0)
        engine = TurboQuantWeights(config)
        S = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1], dtype=np.float32)
        k = engine.select_rank(S, energy_threshold=0.95)
        # Cumulative energy: 100/130.26, 125/130.26, 129/130.26, ...
        assert k >= 2
        assert k <= 4

    def test_energy_99_percent_higher_rank(self) -> None:
        engine = TurboQuantWeights()
        S = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1], dtype=np.float32)
        k95 = engine.select_rank(S, energy_threshold=0.95)
        k99 = engine.select_rank(S, energy_threshold=0.99)
        assert k99 >= k95

    def test_min_rank_floor(self) -> None:
        config = WeightCompressionConfig(min_rank=32)
        engine = TurboQuantWeights(config)
        # Even if energy is captured in 1 component, floor applies
        S = np.array([100.0] + [0.001] * 63, dtype=np.float32)
        k = engine.select_rank(S, energy_threshold=0.99)
        assert k >= 32

    def test_max_rank_ceiling(self) -> None:
        config = WeightCompressionConfig(max_rank_ratio=0.25)
        engine = TurboQuantWeights(config)
        S = np.ones(100, dtype=np.float32)  # uniform — needs all for 99%
        k = engine.select_rank(S, energy_threshold=0.99)
        assert k <= 25  # 0.25 * 100

    def test_monotonic_with_threshold(self) -> None:
        engine = TurboQuantWeights()
        S = np.array([10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2], dtype=np.float32)
        ranks = [
            engine.select_rank(S, energy_threshold=t) for t in [0.8, 0.9, 0.95, 0.99]
        ]
        for i in range(len(ranks) - 1):
            assert ranks[i + 1] >= ranks[i]


# ------------------------------------------------------------------ #
# Test: Weight Compression Round-Trip                                 #
# ------------------------------------------------------------------ #


class TestWeightCompression:
    """Tests for SVD + TurboQuant weight compression."""

    def test_round_trip_shape(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(256, 256)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        assert W_hat.shape == W.shape

    def test_round_trip_dtype(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(128, 128)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        assert W_hat.dtype == np.float32

    def test_identity_matrix(self) -> None:
        """Identity has flat spectrum so rank truncation hurts it.
        With max_rank_ratio=0.5, we keep half the rank — expect moderate quality."""
        config = WeightCompressionConfig(energy_threshold=0.999, bits=4)
        engine = TurboQuantWeights(config)
        W = np.eye(64, dtype=np.float32)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        cos = _cosine_similarity(W, W_hat)
        # Identity's uniform spectrum means 50% rank keeps 50% energy
        assert cos > 0.60, f"Identity cosine {cos:.4f} too low"
        assert W_hat.shape == W.shape

    def test_low_rank_matrix_exact(self) -> None:
        """A rank-32 matrix compressed at rank>=32 should reconstruct well."""
        config = WeightCompressionConfig(energy_threshold=0.999, bits=4, min_rank=32)
        engine = TurboQuantWeights(config)
        W = _low_rank_weight(128, 128, rank=32)
        cw = engine.compress_weight(W)
        assert cw.rank >= 32
        W_hat = engine.decompress_weight(cw)
        cos = _cosine_similarity(W, W_hat)
        assert cos > 0.90, f"Low-rank cosine {cos:.4f} too low"

    def test_3bit_cosine_similarity(self) -> None:
        config = WeightCompressionConfig(energy_threshold=0.95, bits=3)
        engine = TurboQuantWeights(config)
        W = _random_weight(512, 512)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        cos = _cosine_similarity(W, W_hat)
        assert cos > 0.85, f"3-bit cosine {cos:.4f} too low"

    def test_4bit_cosine_similarity(self) -> None:
        config = WeightCompressionConfig(energy_threshold=0.95, bits=4)
        engine = TurboQuantWeights(config)
        W = _random_weight(512, 512)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        cos = _cosine_similarity(W, W_hat)
        assert cos > 0.90, f"4-bit cosine {cos:.4f} too low"

    def test_compression_ratio_positive(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(512, 512)
        cw = engine.compress_weight(W)
        assert cw.compression_ratio() > 1.0

    def test_rectangular_tall(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(1024, 256)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        assert W_hat.shape == (1024, 256)
        assert _cosine_similarity(W, W_hat) > 0.80

    def test_rectangular_wide(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(256, 1024)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        assert W_hat.shape == (256, 1024)
        assert _cosine_similarity(W, W_hat) > 0.80

    def test_fp16_input(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(256, 256).astype(np.float16)
        cw = engine.compress_weight(W)
        assert cw.original_dtype == np.float16
        W_hat = engine.decompress_weight(cw)
        assert W_hat.shape == W.shape


# ------------------------------------------------------------------ #
# Test: Factored Matmul                                               #
# ------------------------------------------------------------------ #


class TestFactoredMatmul:
    """Tests for factored matmul (inference path)."""

    def test_matches_full_reconstruction(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(256, 128)
        cw = engine.compress_weight(W)
        x = np.random.default_rng(99).standard_normal((4, 128)).astype(np.float32)

        W_hat = engine.decompress_weight(cw)
        expected = x @ W_hat.T
        actual = engine.compressed_linear(x, cw)

        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_output_shape(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(512, 256)
        cw = engine.compress_weight(W)
        x = np.random.default_rng(99).standard_normal((8, 256)).astype(np.float32)
        out = engine.compressed_linear(x, cw)
        assert out.shape == (8, 512)

    def test_single_vector(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(128, 64)
        cw = engine.compress_weight(W)
        x = np.random.default_rng(99).standard_normal((1, 64)).astype(np.float32)
        out = engine.compressed_linear(x, cw)
        assert out.shape == (1, 128)


# ------------------------------------------------------------------ #
# Test: Model-Level Compression                                       #
# ------------------------------------------------------------------ #


class TestModelCompression:
    """Tests for compressing a full state dict."""

    def _make_state_dict(self) -> dict:
        rng = np.random.default_rng(42)
        return {
            "layer.0.self_attn.q_proj.weight": rng.standard_normal((256, 256)).astype(
                np.float32
            ),
            "layer.0.self_attn.k_proj.weight": rng.standard_normal((256, 256)).astype(
                np.float32
            ),
            "layer.0.mlp.gate_proj.weight": rng.standard_normal((512, 256)).astype(
                np.float32
            ),
            "embed_tokens.weight": rng.standard_normal((1000, 256)).astype(np.float32),
            "layer.0.input_layernorm.weight": rng.standard_normal(256).astype(
                np.float32
            ),
            "lm_head.weight": rng.standard_normal((1000, 256)).astype(np.float32),
            "layer.0.self_attn.q_proj.bias": rng.standard_normal(256).astype(
                np.float32
            ),
        }

    def test_compress_state_dict(self) -> None:
        engine = TurboQuantWeights()
        sd = self._make_state_dict()
        cm = engine.compress_state_dict(sd)
        assert isinstance(cm, CompressedModel)
        assert len(cm.weights) > 0
        assert cm.compression_ratio() > 1.0

    def test_skip_embeddings(self) -> None:
        engine = TurboQuantWeights()
        sd = self._make_state_dict()
        cm = engine.compress_state_dict(sd)
        assert "embed_tokens.weight" in cm.uncompressed
        assert "embed_tokens.weight" not in cm.weights

    def test_skip_lm_head(self) -> None:
        engine = TurboQuantWeights()
        sd = self._make_state_dict()
        cm = engine.compress_state_dict(sd)
        assert "lm_head.weight" in cm.uncompressed

    def test_skip_1d_weights(self) -> None:
        engine = TurboQuantWeights()
        sd = self._make_state_dict()
        cm = engine.compress_state_dict(sd)
        assert "layer.0.input_layernorm.weight" in cm.uncompressed
        assert "layer.0.self_attn.q_proj.bias" in cm.uncompressed

    def test_memory_accounting(self) -> None:
        engine = TurboQuantWeights()
        sd = self._make_state_dict()
        cm = engine.compress_state_dict(sd)
        assert cm.total_compressed_bytes() < cm.total_original_bytes()
        summary = cm.summary()
        assert summary["ratio"] > 1.0
        assert summary["savings_pct"] > 0


# ------------------------------------------------------------------ #
# Test: CompressedLinear                                              #
# ------------------------------------------------------------------ #


class TestCompressedLinearNumpy:
    """Tests for the NumPy-only compressed linear."""

    def test_forward_shape(self) -> None:
        from agi.meta.llm.turboquant_weights import CompressedLinearNumpy

        engine = TurboQuantWeights()
        W = _random_weight(128, 64)
        cw = engine.compress_weight(W)
        linear = CompressedLinearNumpy(cw, engine)
        x = np.random.default_rng(99).standard_normal((4, 64)).astype(np.float32)
        out = linear(x)
        assert out.shape == (4, 128)

    def test_with_bias(self) -> None:
        from agi.meta.llm.turboquant_weights import CompressedLinearNumpy

        engine = TurboQuantWeights()
        W = _random_weight(128, 64)
        cw = engine.compress_weight(W)
        bias = np.ones(128, dtype=np.float32)
        linear = CompressedLinearNumpy(cw, engine, bias=bias)
        x = np.zeros((1, 64), dtype=np.float32)
        out = linear(x)
        # With zero input, output should be close to bias
        np.testing.assert_allclose(out, bias.reshape(1, -1), atol=0.5)

    def test_matches_engine_output(self) -> None:
        from agi.meta.llm.turboquant_weights import CompressedLinearNumpy

        engine = TurboQuantWeights()
        W = _random_weight(256, 128)
        cw = engine.compress_weight(W)
        linear = CompressedLinearNumpy(cw, engine)
        x = np.random.default_rng(99).standard_normal((2, 128)).astype(np.float32)
        np.testing.assert_array_equal(linear(x), engine.compressed_linear(x, cw))


# ------------------------------------------------------------------ #
# Test: Edge Cases                                                    #
# ------------------------------------------------------------------ #


class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_zero_matrix(self) -> None:
        config = WeightCompressionConfig(min_rank=16)
        engine = TurboQuantWeights(config)
        W = np.zeros((64, 64), dtype=np.float32)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        assert np.allclose(W_hat, 0.0, atol=0.1)

    def test_rank_one_matrix(self) -> None:
        engine = TurboQuantWeights()
        rng = np.random.default_rng(42)
        a = rng.standard_normal((128, 1)).astype(np.float32)
        b = rng.standard_normal((1, 64)).astype(np.float32)
        W = a @ b
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        cos = _cosine_similarity(W, W_hat)
        assert cos > 0.80, f"Rank-1 cosine {cos:.4f} too low"

    def test_small_matrix(self) -> None:
        config = WeightCompressionConfig(min_rank=2, min_matrix_elements=1)
        engine = TurboQuantWeights(config)
        W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        cw = engine.compress_weight(W)
        W_hat = engine.decompress_weight(cw)
        assert W_hat.shape == (3, 2)

    def test_deterministic_with_seed(self) -> None:
        config = WeightCompressionConfig(seed=123)
        engine = TurboQuantWeights(config)
        W = _random_weight(128, 128)
        cw1 = engine.compress_weight(W)
        cw2 = engine.compress_weight(W)
        W1 = engine.decompress_weight(cw1)
        W2 = engine.decompress_weight(cw2)
        np.testing.assert_array_equal(W1, W2)

    def test_rejects_1d_input(self) -> None:
        engine = TurboQuantWeights()
        with pytest.raises(ValueError, match="2-D"):
            engine.compress_weight(np.zeros(100, dtype=np.float32))


# ------------------------------------------------------------------ #
# Test: Serialization                                                 #
# ------------------------------------------------------------------ #


class TestSerialization:
    """Tests for save/load round-trip."""

    def test_save_load_roundtrip(self) -> None:
        engine = TurboQuantWeights()
        rng = np.random.default_rng(42)
        sd = {
            "attn.q_proj.weight": rng.standard_normal((256, 256)).astype(np.float32),
            "attn.k_proj.weight": rng.standard_normal((256, 256)).astype(np.float32),
            "norm.weight": rng.standard_normal(256).astype(np.float32),
        }
        cm = engine.compress_state_dict(sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.npz"
            engine.save(cm, path)
            cm2 = engine.load(path)

        assert set(cm2.weights.keys()) == set(cm.weights.keys())
        assert set(cm2.uncompressed.keys()) == set(cm.uncompressed.keys())

        for name in cm.weights:
            W1 = engine.decompress_weight(cm.weights[name])
            W2 = engine.decompress_weight(cm2.weights[name])
            np.testing.assert_allclose(W1, W2, atol=1e-5)

    def test_compressed_model_nbytes(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(256, 256)
        sd = {"proj.weight": W}
        cm = engine.compress_state_dict(sd)
        assert cm.total_compressed_bytes() > 0
        assert cm.total_compressed_bytes() < cm.total_original_bytes()


# ------------------------------------------------------------------ #
# Test: Static Estimation                                             #
# ------------------------------------------------------------------ #


class TestEstimation:
    """Tests for static compression estimation."""

    def test_estimate_basic(self) -> None:
        est = TurboQuantWeights.estimate_compression(4096, 4096, rank=512, bits=3)
        assert est["ratio"] > 5.0
        assert est["savings_pct"] > 80.0

    def test_estimate_vs_q4(self) -> None:
        # TQ3 + SVD should beat Q4 for appropriately chosen rank
        est = TurboQuantWeights.estimate_compression(4096, 4096, rank=512, bits=3)
        q4_bytes = 4096 * 4096 * 4 / 8  # 4-bit = 0.5 bytes per element
        assert est["compressed_bytes"] < q4_bytes

    def test_higher_bits_less_compression(self) -> None:
        est3 = TurboQuantWeights.estimate_compression(1024, 1024, rank=128, bits=3)
        est4 = TurboQuantWeights.estimate_compression(1024, 1024, rank=128, bits=4)
        assert est3["ratio"] > est4["ratio"]


# ------------------------------------------------------------------ #
# Test: Cached Decompression                                          #
# ------------------------------------------------------------------ #


class TestCachedDecompression:
    """Tests that TurboQuantKV instances are cached (no repeat QR)."""

    def test_cache_populated(self) -> None:
        engine = TurboQuantWeights()
        W = _random_weight(128, 128)
        engine.compress_weight(W)
        assert len(engine._tq_cache) > 0

    def test_same_dims_reuse_instance(self) -> None:
        engine = TurboQuantWeights()
        W1 = _random_weight(128, 128, seed=1)
        W2 = _random_weight(128, 128, seed=2)
        engine.compress_weight(W1)
        cache_before = dict(engine._tq_cache)
        engine.compress_weight(W2)
        # Same dimensions → same cached instances
        for key in cache_before:
            assert engine._tq_cache[key] is cache_before[key]

    def test_different_dims_different_instances(self) -> None:
        engine = TurboQuantWeights()
        W1 = _random_weight(128, 128)
        W2 = _random_weight(256, 256)  # different d_in → different TQ
        engine.compress_weight(W1)
        engine.compress_weight(W2)
        assert len(engine._tq_cache) >= 2


# ------------------------------------------------------------------ #
# Test: Precompute Factors                                            #
# ------------------------------------------------------------------ #


class TestPrecomputeFactors:
    """Tests for precompute_factors (trade memory for speed)."""

    def test_precomputed_matches_on_the_fly(self) -> None:
        engine = TurboQuantWeights()
        sd = {"proj.weight": _random_weight(256, 256)}
        cm = engine.compress_state_dict(sd)
        factors = engine.precompute_factors(cm)

        # Precomputed factors should match on-the-fly decompression
        for name, (U_pre, Vt_pre) in factors.items():
            U_otf, Vt_otf = engine._decompress_factors(cm.weights[name])
            np.testing.assert_array_equal(U_pre, U_otf)
            np.testing.assert_array_equal(Vt_pre, Vt_otf)

    def test_precomputed_all_layers(self) -> None:
        engine = TurboQuantWeights()
        rng = np.random.default_rng(42)
        sd = {
            "a.weight": rng.standard_normal((128, 128)).astype(np.float32),
            "b.weight": rng.standard_normal((256, 128)).astype(np.float32),
            "norm.weight": rng.standard_normal(128).astype(np.float32),
        }
        cm = engine.compress_state_dict(sd)
        factors = engine.precompute_factors(cm)
        assert set(factors.keys()) == set(cm.weights.keys())


# ------------------------------------------------------------------ #
# Test: CompressedLinear as nn.Module                                 #
# ------------------------------------------------------------------ #


class TestCompressedLinearModule:
    """Tests that CompressedLinear works as a torch.nn.Module."""

    @pytest.fixture()
    def _setup(self):
        import torch

        engine = TurboQuantWeights()
        W = _random_weight(128, 64)
        cw = engine.compress_weight(W)
        return engine, cw, torch

    def test_is_nn_module(self, _setup) -> None:
        import torch.nn as nn

        from agi.meta.llm.turboquant_weights import _make_compressed_linear

        engine, cw, torch = _setup
        module = _make_compressed_linear(cw, engine)
        assert isinstance(module, nn.Module)

    def test_forward_shape(self, _setup) -> None:
        from agi.meta.llm.turboquant_weights import _make_compressed_linear

        engine, cw, torch = _setup
        module = _make_compressed_linear(cw, engine)
        x = torch.randn(4, 64)
        out = module(x)
        assert out.shape == (4, 128)

    def test_no_gradient_through_weights(self, _setup) -> None:
        from agi.meta.llm.turboquant_weights import _make_compressed_linear

        engine, cw, torch = _setup
        module = _make_compressed_linear(cw, engine)
        x = torch.randn(2, 64, requires_grad=True)
        out = module(x)
        # Output should be differentiable w.r.t. input
        out.sum().backward()
        assert x.grad is not None
        # But the module should have no trainable parameters
        assert sum(p.numel() for p in module.parameters()) == 0

    def test_with_bias(self, _setup) -> None:
        from agi.meta.llm.turboquant_weights import _make_compressed_linear

        engine, cw, torch = _setup
        bias = torch.ones(128)
        module = _make_compressed_linear(cw, engine, bias=bias)
        x = torch.zeros(1, 64)
        out = module(x)
        # Zero input + bias → output ≈ bias
        np.testing.assert_allclose(
            out.detach().numpy(), bias.numpy().reshape(1, -1), atol=0.5
        )
