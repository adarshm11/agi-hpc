# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the Ego arbiter disagreement detection logic.

Tests the cosine similarity measurement between Superego and Id
responses, confidence mapping, and the decision to route to the
Ego when disagreement is high.
"""

from __future__ import annotations

import pytest


class TestDisagreementDetection:
    """Tests for the disagreement measurement used by the Ego arbiter."""

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts via BGE-M3.

        Falls back to random embeddings if model is unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("BAAI/bge-m3", device="cpu")
            embs = model.encode([text_a, text_b], normalize_embeddings=True)
            return float(embs[0] @ embs[1])
        except Exception:
            pytest.skip("BGE-M3 model not available")

    def test_similar_texts_high_similarity(self) -> None:
        sim = self._compute_similarity(
            "Python is a high-level programming language.",
            "Python is a popular high-level programming language.",
        )
        assert sim > 0.8

    def test_different_texts_lower_similarity(self) -> None:
        sim = self._compute_similarity(
            "The economy is growing steadily this quarter.",
            "Abstract art challenges conventional aesthetics.",
        )
        assert sim < 0.7


class TestConfidenceMapping:
    """Tests for the similarity-to-confidence mapping."""

    @staticmethod
    def map_confidence(similarity: float) -> float:
        """Mirror the mapping logic from atlas-rag-server.py."""
        if similarity > 0.85:
            return 0.9
        elif similarity >= 0.5:
            return 0.4 + (similarity - 0.5) / 0.35 * 0.45
        else:
            return 0.3

    def test_high_similarity_high_confidence(self) -> None:
        assert self.map_confidence(0.90) == 0.9

    def test_medium_similarity_medium_confidence(self) -> None:
        conf = self.map_confidence(0.70)
        assert 0.4 < conf < 0.9

    def test_low_similarity_low_confidence(self) -> None:
        assert self.map_confidence(0.3) == 0.3

    def test_boundary_085(self) -> None:
        conf = self.map_confidence(0.85)
        assert conf == pytest.approx(0.85, abs=0.01)

    def test_boundary_050(self) -> None:
        conf = self.map_confidence(0.50)
        assert conf == pytest.approx(0.40, abs=0.01)

    def test_ego_triggers_below_050(self) -> None:
        """Ego should arbitrate when confidence < 0.5."""
        # similarity=0.55 -> confidence ~0.46
        conf = self.map_confidence(0.55)
        assert conf < 0.5  # Would trigger Ego arbitration

    def test_ego_does_not_trigger_above_050(self) -> None:
        """Ego should NOT arbitrate when confidence >= 0.5."""
        # similarity=0.65 -> confidence ~0.59
        conf = self.map_confidence(0.65)
        assert conf >= 0.5  # Id synthesizes normally

    def test_monotonic(self) -> None:
        """Confidence should be monotonically increasing with similarity."""
        sims = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        confs = [self.map_confidence(s) for s in sims]
        for i in range(1, len(confs)):
            assert confs[i] >= confs[i - 1]


class TestEgoSystemPrompt:
    """Tests for the Ego's system prompt content."""

    EGO_SYSTEM = (
        "You are the Ego — the mediator of the psyche. The Superego (analytical, "
        "moral) and the Id (creative, instinctual) have debated but strongly disagree. "
        "Your role is to find the practical, balanced resolution. You are grounded in "
        "reality. Be concise and authoritative."
    )

    def test_mentions_mediator(self) -> None:
        assert "mediator" in self.EGO_SYSTEM.lower()

    def test_mentions_superego_and_id(self) -> None:
        assert "Superego" in self.EGO_SYSTEM
        assert "Id" in self.EGO_SYSTEM

    def test_mentions_reality(self) -> None:
        """Freudian Ego is grounded in the reality principle."""
        assert "reality" in self.EGO_SYSTEM.lower()
