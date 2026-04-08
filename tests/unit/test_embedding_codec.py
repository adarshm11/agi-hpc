# AGI-HPC Project
# Tests for embedding codec and shared embedding service.

from __future__ import annotations

import json

import numpy as np
import pytest

from agi.common.embedding_codec import EmbeddingCodec


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def sample_embedding(rng: np.random.Generator) -> np.ndarray:
    """Single 1024-dim normalized embedding."""
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


class TestEmbeddingCodecPassthrough:
    """Test codec in passthrough mode (no turboquant_pro dependency needed)."""

    def test_passthrough_compress(self, sample_embedding: np.ndarray) -> None:
        codec = EmbeddingCodec.__new__(EmbeddingCodec)
        codec.dim = 1024
        codec.bits = 3
        codec._pca_data = None
        codec._tq = None
        codec._tq_pca = None
        codec._mode = "passthrough"

        result = codec.compress(sample_embedding)
        assert "v" in result
        assert len(result["v"]) == 1024

    def test_passthrough_decompress_list(self) -> None:
        codec = EmbeddingCodec.__new__(EmbeddingCodec)
        codec.dim = 1024
        codec.bits = 3
        codec._pca_data = None
        codec._tq = None
        codec._tq_pca = None
        codec._mode = "passthrough"

        raw = [0.1] * 1024
        result = codec.decompress(raw)
        assert result.shape == (1024,)
        assert result.dtype == np.float32

    def test_passthrough_roundtrip(self, sample_embedding: np.ndarray) -> None:
        codec = EmbeddingCodec.__new__(EmbeddingCodec)
        codec.dim = 1024
        codec.bits = 3
        codec._pca_data = None
        codec._tq = None
        codec._tq_pca = None
        codec._mode = "passthrough"

        compressed = codec.compress(sample_embedding)
        decompressed = codec.decompress(compressed)
        np.testing.assert_allclose(sample_embedding, decompressed, atol=1e-5)


class TestEmbeddingCodecTQ:
    """Test codec with TurboQuant (requires turboquant_pro)."""

    @pytest.fixture()
    def codec(self) -> EmbeddingCodec:
        try:
            # Force tq_only mode by using non-existent PCA path
            return EmbeddingCodec(dim=1024, bits=3, pca_path="/nonexistent")
        except RuntimeError:
            pytest.skip("turboquant_pro not installed")

    def test_tq_compress_format(
        self, codec: EmbeddingCodec, sample_embedding: np.ndarray
    ) -> None:
        result = codec.compress(sample_embedding)
        assert "_tq" in result
        assert result["_tq"] == "raw"
        assert "b" in result  # base85-encoded bytes
        assert "n" in result  # norm
        assert "d" in result  # dim
        assert "bits" in result

    def test_tq_roundtrip(
        self, codec: EmbeddingCodec, sample_embedding: np.ndarray
    ) -> None:
        compressed = codec.compress(sample_embedding)
        decompressed = codec.decompress(compressed)

        assert decompressed.shape == (1024,)
        cos = float(
            np.dot(sample_embedding, decompressed)
            / (np.linalg.norm(sample_embedding) * np.linalg.norm(decompressed))
        )
        assert cos > 0.95, f"Cosine {cos} too low"

    def test_tq_json_serializable(
        self, codec: EmbeddingCodec, sample_embedding: np.ndarray
    ) -> None:
        compressed = codec.compress(sample_embedding)
        # Must be JSON-serializable for Event payload
        json_str = json.dumps(compressed)
        parsed = json.loads(json_str)
        decompressed = codec.decompress(parsed)
        assert decompressed.shape == (1024,)

    def test_tq_size_reduction(
        self, codec: EmbeddingCodec, sample_embedding: np.ndarray
    ) -> None:
        stats = codec.payload_size(sample_embedding)
        assert stats["ratio"] > 3.0, f"Expected >3x compression, got {stats['ratio']}"

    def test_tq_decompress_raw_list(self, codec: EmbeddingCodec) -> None:
        """Codec should handle raw float lists (legacy format)."""
        raw = [0.1] * 1024
        result = codec.decompress(raw)
        assert result.shape == (1024,)

    def test_repr(self, codec: EmbeddingCodec) -> None:
        r = repr(codec)
        assert "tq_only" in r
        assert "1024" in r
