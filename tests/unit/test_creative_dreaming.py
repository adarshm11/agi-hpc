# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for creative dream generation (Stage 4).

Tests diverse triplet selection, novelty scoring against existing
wiki, coherence evaluation, and dream insight persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agi.dreaming.consolidator import (
    ConsolidatorConfig,
    DreamInsight,
    MemoryConsolidator,
)


class FakeEpisode:
    """Minimal episode stub for testing."""

    def __init__(self, id: str, desc: str, text: str):
        self.id = id
        self.task_description = desc
        self.text = text
        self.messages = []
        self.success = True
        self.insights = []
        self.consolidated = False
        self.timestamp = "2026-04-07"


def _make_episodes(n: int = 12):
    topics = [
        ("quantum", "Quantum computing basics"),
        ("ethics", "AI ethics and fairness"),
        ("climate", "Climate change mitigation"),
        ("medical", "Medical triage decisions"),
        ("coding", "Python debugging techniques"),
        ("space", "Mars colonization challenges"),
    ]
    episodes = []
    for i in range(n):
        t = topics[i % len(topics)]
        episodes.append(
            FakeEpisode(
                id=f"ep-{i:03d}",
                desc=t[1],
                text=f"Detailed content about {t[0]} topic {i}. "
                f"This covers {t[1]} in depth with examples.",
            )
        )
    return episodes


class TestDiverseTripletSelection:
    """Tests for _select_diverse_triplets."""

    def test_returns_triplets(self) -> None:
        config = ConsolidatorConfig()
        consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
        consolidator.config = config
        consolidator.wiki_dir = "/tmp/test-wiki"

        episodes = _make_episodes(12)
        triplets = consolidator._select_diverse_triplets(episodes, n=3)

        assert len(triplets) == 3
        for triplet in triplets:
            assert len(triplet) == 3

    def test_triplets_are_diverse(self) -> None:
        config = ConsolidatorConfig()
        consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
        consolidator.config = config
        consolidator.wiki_dir = "/tmp/test-wiki"

        episodes = _make_episodes(12)
        triplets = consolidator._select_diverse_triplets(episodes, n=2)

        # Each triplet should have episodes from different topics
        for triplet in triplets:
            descs = [ep.task_description for ep in triplet]
            # At least 2 of 3 should be different topics
            assert len(set(descs)) >= 2

    def test_too_few_episodes(self) -> None:
        config = ConsolidatorConfig()
        consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
        consolidator.config = config
        consolidator.wiki_dir = "/tmp/test-wiki"

        episodes = _make_episodes(4)
        triplets = consolidator._select_diverse_triplets(episodes, n=1)

        assert len(triplets) == 0


class TestNoveltyScoring:
    """Tests for _score_novelty."""

    @pytest.mark.asyncio
    async def test_novel_insight_high_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConsolidatorConfig(wiki_dir=tmpdir)
            consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
            consolidator.config = config
            consolidator.wiki_dir = tmpdir

            # No existing articles = high novelty
            score = await consolidator._score_novelty(
                "A completely new insight about quantum jazz fusion"
            )
            assert score > 0.7

    @pytest.mark.asyncio
    async def test_redundant_insight_low_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an existing article with overlapping content
            article = Path(tmpdir) / "dream-existing.md"
            article.write_text(
                "quantum computing uses qubits for superposition "
                "entanglement gates circuits processors algorithms "
                "error correction decoherence measurements",
                encoding="utf-8",
            )

            config = ConsolidatorConfig(wiki_dir=tmpdir)
            consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
            consolidator.config = config
            consolidator.wiki_dir = tmpdir

            score = await consolidator._score_novelty(
                "quantum computing uses qubits for superposition "
                "entanglement gates circuits processors"
            )
            assert score < 0.5

    @pytest.mark.asyncio
    async def test_empty_wiki_moderate_novelty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConsolidatorConfig(wiki_dir=tmpdir)
            consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
            consolidator.config = config
            consolidator.wiki_dir = tmpdir

            score = await consolidator._score_novelty("any insight text")
            assert 0.7 <= score <= 0.9


class TestCoherenceScoring:
    """Tests for _score_coherence."""

    @pytest.mark.asyncio
    async def test_parses_numeric_score(self) -> None:
        config = ConsolidatorConfig()
        consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
        consolidator.config = config
        consolidator.wiki_dir = "/tmp/test-wiki"

        with patch(
            "agi.dreaming.consolidator._llm_generate",
            new_callable=AsyncMock,
            return_value="8",
        ):
            score = await consolidator._score_coherence(
                "A clear insight connecting two domains",
                _make_episodes(3),
            )
            assert score == 0.8

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self) -> None:
        config = ConsolidatorConfig()
        consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
        consolidator.config = config
        consolidator.wiki_dir = "/tmp/test-wiki"

        with patch(
            "agi.dreaming.consolidator._llm_generate",
            new_callable=AsyncMock,
            side_effect=Exception("LLM offline"),
        ):
            score = await consolidator._score_coherence(
                "Some insight", _make_episodes(3)
            )
            assert score == 0.5  # Fallback


class TestDreamInsightPersistence:
    """Tests for _save_dream_insight."""

    def test_saves_to_wiki(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConsolidatorConfig(wiki_dir=tmpdir)
            consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
            consolidator.config = config
            consolidator.wiki_dir = tmpdir

            insight = DreamInsight(
                text="Quantum error correction patterns mirror "
                "biological DNA repair mechanisms",
                source_episodes=["ep-001", "ep-042"],
                novelty_score=0.85,
                coherence_score=0.75,
            )

            consolidator._save_dream_insight(insight)

            articles = list(Path(tmpdir).glob("dream-insight-*.md"))
            assert len(articles) == 1

            content = articles[0].read_text(encoding="utf-8")
            assert "Quantum error correction" in content
            assert "Novelty: 0.85" in content
            assert "ep-001" in content

    def test_slug_from_insight_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConsolidatorConfig(wiki_dir=tmpdir)
            consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
            consolidator.config = config
            consolidator.wiki_dir = tmpdir

            insight = DreamInsight(
                text="Neural networks resemble ant colonies",
                source_episodes=["ep-001"],
                novelty_score=0.9,
                coherence_score=0.8,
            )

            consolidator._save_dream_insight(insight)

            articles = list(Path(tmpdir).glob("dream-insight-*.md"))
            assert len(articles) == 1
            assert "neural" in articles[0].name.lower()


class TestDreamInsightDataclass:
    """Tests for the DreamInsight structure."""

    def test_creation(self) -> None:
        di = DreamInsight(
            text="Test insight",
            source_episodes=["a", "b", "c"],
            novelty_score=0.7,
            coherence_score=0.8,
        )
        assert di.novelty_score == 0.7
        assert len(di.source_episodes) == 3
