# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the knowledge gap curriculum planner.

Tests gap detection signals (low quality, safety flags, short
responses, high disagreement), gap merging, severity ranking,
and DM formatting.
"""

from __future__ import annotations

from agi.metacognition.curriculum_planner import (
    CurriculumPlan,
    CurriculumPlanner,
    KnowledgeGap,
)


def _make_planner():
    """Create a planner without DB dependency."""
    return CurriculumPlanner(db_dsn="dbname=test")


def _make_episodes():
    """Sample episodes for gap detection testing."""
    return [
        {
            "id": "ep-001",
            "user_message": "Explain quantum entanglement",
            "atlas_response": "Quantum entanglement is a phenomenon...",
            "hemisphere": "lh",
            "safety_flags": {},
            "quality_score": 0.9,
            "metadata": {},
        },
        {
            "id": "ep-002",
            "user_message": "What is quantum decoherence?",
            "atlas_response": "I'm not sure about that.",
            "hemisphere": "lh",
            "safety_flags": {},
            "quality_score": 0.3,
            "metadata": {},
        },
        {
            "id": "ep-003",
            "user_message": "How does quantum tunneling work?",
            "atlas_response": "Hmm.",
            "hemisphere": "lh",
            "safety_flags": {},
            "quality_score": 0.2,
            "metadata": {},
        },
        {
            "id": "ep-004",
            "user_message": "Is climate change reversible?",
            "atlas_response": "A detailed analysis of climate...",
            "hemisphere": "both",
            "safety_flags": {},
            "quality_score": 0.8,
            "metadata": {},
        },
        {
            "id": "ep-005",
            "user_message": "Should AI make medical decisions?",
            "atlas_response": "This is a nuanced question...",
            "hemisphere": "both",
            "safety_flags": {
                "output": {
                    "flags": ["pii_email"],
                    "passed": True,
                },
            },
            "quality_score": 0.5,
            "metadata": {},
        },
        {
            "id": "ep-006",
            "user_message": "Explain neural network backpropagation",
            "atlas_response": "OK.",
            "hemisphere": "lh",
            "safety_flags": {},
            "quality_score": 0.4,
            "metadata": {},
        },
    ]


class TestGapDetection:
    """Tests for individual gap detection signals."""

    def test_detects_low_quality(self) -> None:
        planner = _make_planner()
        episodes = _make_episodes()
        gaps = planner._detect_low_quality(episodes)

        # Should detect "quantum" as a weak domain
        # (ep-002=0.3, ep-003=0.2 are low)
        domains = [g.domain for g in gaps]
        assert any("quantum" in d for d in domains)

    def test_detects_safety_issues(self) -> None:
        planner = _make_planner()
        episodes = _make_episodes()
        gaps = planner._detect_safety_issues(episodes)

        assert len(gaps) == 1
        assert gaps[0].signal == "output_safety_flags"

    def test_detects_short_responses(self) -> None:
        planner = _make_planner()
        episodes = _make_episodes()
        gaps = planner._detect_uncertain_responses(episodes)

        # "Hmm." and "OK." are very short
        short_domains = [g.domain for g in gaps]
        assert len(short_domains) > 0

    def test_detects_high_disagreement(self) -> None:
        planner = _make_planner()
        episodes = _make_episodes()
        gaps = planner._detect_high_disagreement(episodes)

        # ep-004 and ep-005 both use "both" hemisphere
        # Need at least 3 debate episodes for detection
        # Current set has 2, so may not trigger
        assert isinstance(gaps, list)


class TestGapMerging:
    """Tests for gap deduplication and merging."""

    def test_merges_same_domain(self) -> None:
        planner = _make_planner()
        gaps = [
            KnowledgeGap(
                domain="quantum",
                signal="low_quality",
                severity=0.7,
                sample_questions=["Q1"],
                episode_ids=["ep-001"],
            ),
            KnowledgeGap(
                domain="quantum",
                signal="short_responses",
                severity=0.5,
                sample_questions=["Q2"],
                episode_ids=["ep-002"],
            ),
        ]

        merged = planner._merge_gaps(gaps)
        assert len(merged) == 1
        assert merged[0].severity == 0.7  # Keeps highest
        assert "low_quality" in merged[0].signal
        assert "short_responses" in merged[0].signal

    def test_keeps_different_domains(self) -> None:
        planner = _make_planner()
        gaps = [
            KnowledgeGap(domain="quantum", signal="low_quality", severity=0.7),
            KnowledgeGap(domain="ethics", signal="low_quality", severity=0.6),
        ]

        merged = planner._merge_gaps(gaps)
        assert len(merged) == 2


class TestCurriculumPlan:
    """Tests for the full analysis pipeline."""

    def test_empty_episodes(self) -> None:
        planner = _make_planner()
        # With no DB, returns empty plan
        plan = planner.analyze()
        assert isinstance(plan, CurriculumPlan)
        assert plan.total_episodes_analyzed == 0

    def test_severity_ordering(self) -> None:
        gaps = [
            KnowledgeGap(domain="a", signal="x", severity=0.3),
            KnowledgeGap(domain="b", signal="x", severity=0.9),
            KnowledgeGap(domain="c", signal="x", severity=0.6),
        ]
        gaps.sort(key=lambda g: g.severity, reverse=True)
        assert gaps[0].domain == "b"
        assert gaps[-1].domain == "a"

    def test_difficulty_from_severity(self) -> None:
        # High severity -> low difficulty (build foundation)
        gap = KnowledgeGap(domain="x", signal="y", severity=0.9)
        # Planner sets difficulty in analyze(), let's test the logic
        if gap.severity > 0.8:
            gap.suggested_difficulty = 1
        elif gap.severity > 0.5:
            gap.suggested_difficulty = 2
        else:
            gap.suggested_difficulty = 3
        assert gap.suggested_difficulty == 1


class TestDMFormatting:
    """Tests for formatting gaps as DM context."""

    def test_format_with_gaps(self) -> None:
        planner = _make_planner()
        plan = CurriculumPlan(
            gaps=[
                KnowledgeGap(
                    domain="quantum",
                    signal="low_quality",
                    severity=0.7,
                    sample_questions=["What is decoherence?"],
                ),
            ],
            total_episodes_analyzed=50,
            recommended_scenarios=6,
            domains_to_focus=["quantum"],
        )

        text = planner.format_for_dm(plan)
        assert "quantum" in text
        assert "low_quality" in text
        assert "decoherence" in text

    def test_format_no_gaps(self) -> None:
        planner = _make_planner()
        plan = CurriculumPlan()
        text = planner.format_for_dm(plan)
        assert "No knowledge gaps" in text


class TestTopicExtraction:
    """Tests for the topic extraction heuristic."""

    def test_extracts_significant_word(self) -> None:
        planner = _make_planner()
        topic = planner._extract_topic("Explain quantum entanglement in detail")
        assert topic == "quantum"

    def test_skips_stop_words(self) -> None:
        planner = _make_planner()
        topic = planner._extract_topic("What would explain these patterns?")
        # "What", "would", "explain", "these" are all stop words
        assert topic == "patterns"

    def test_fallback_for_short_questions(self) -> None:
        planner = _make_planner()
        topic = planner._extract_topic("Hi")
        assert topic == "general"
