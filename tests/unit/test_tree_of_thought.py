# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for Tree-of-Thought multi-path reasoning.

Tests branch generation, parsing, evaluation, selection,
synthesis, and the full reasoning pipeline with mocked LLMs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agi.reasoning.tree_of_thought import (
    ReasoningBranch,
    TreeOfThought,
    TreeResult,
)


@pytest.fixture()
def tot():
    """Create a TreeOfThought with mock URLs."""
    return TreeOfThought(
        superego_url="http://mock:8080",
        id_url="http://mock:8082",
        ego_url="http://mock:8084",
    )


def _mock_response(content):
    mock = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock


class TestBranchParsing:
    """Tests for parsing LLM output into branches."""

    def test_parses_three_branches(self, tot) -> None:
        raw = (
            "APPROACH 1: Logical Analysis\n"
            "ANSWER: The answer is 42 based on formal logic.\n"
            "CONFIDENCE: 8\n\n"
            "APPROACH 2: Historical Precedent\n"
            "ANSWER: History shows the answer is 42.\n"
            "CONFIDENCE: 7\n\n"
            "APPROACH 3: Evidence-Based\n"
            "ANSWER: Studies confirm the answer is 42.\n"
            "CONFIDENCE: 9\n"
        )
        branches = tot._parse_branches(raw, "superego")
        assert len(branches) == 3
        assert branches[0].approach == "Logical Analysis"
        assert branches[0].self_score == 8.0
        assert branches[2].self_score == 9.0

    def test_fallback_unparsed(self, tot) -> None:
        raw = "Just a plain answer without formatting."
        branches = tot._parse_branches(raw, "id")
        assert len(branches) == 1
        assert branches[0].approach == "unparsed"

    def test_handles_error_response(self, tot) -> None:
        raw = "(error: connection refused)"
        branches = tot._parse_branches(raw, "superego")
        assert len(branches) == 0

    def test_truncates_long_content(self, tot) -> None:
        raw = "APPROACH 1: Test\n" "ANSWER: " + "x" * 1000 + "\n" "CONFIDENCE: 5\n"
        branches = tot._parse_branches(raw, "id")
        assert len(branches) == 1
        assert len(branches[0].content) <= 500


class TestBranchSelection:
    """Tests for selecting top branches."""

    def test_selects_highest_scored(self, tot) -> None:
        branches = [
            ReasoningBranch("superego", "a", "...", ego_score=8),
            ReasoningBranch("superego", "b", "...", ego_score=5),
            ReasoningBranch("id", "c", "...", ego_score=9),
            ReasoningBranch("id", "d", "...", ego_score=6),
        ]
        selected = tot._select_top(branches, top_n=2)
        assert len(selected) == 2
        # Best from each hemisphere
        assert any(b.hemisphere == "superego" for b in selected)
        assert any(b.hemisphere == "id" for b in selected)

    def test_ensures_both_hemispheres(self, tot) -> None:
        branches = [
            ReasoningBranch("superego", "a", "...", ego_score=10),
            ReasoningBranch("superego", "b", "...", ego_score=9),
            ReasoningBranch("id", "c", "...", ego_score=3),
        ]
        selected = tot._select_top(branches, top_n=2)
        # Should include the Id branch despite low score
        assert any(b.hemisphere == "id" for b in selected)

    def test_handles_empty(self, tot) -> None:
        selected = tot._select_top([], top_n=2)
        assert selected == []


class TestEvaluation:
    """Tests for Ego evaluation of branches."""

    def test_parses_ego_scores(self, tot) -> None:
        branches = [
            ReasoningBranch("superego", "a", "answer1"),
            ReasoningBranch("superego", "b", "answer2"),
            ReasoningBranch("id", "c", "answer3"),
        ]

        with patch("agi.reasoning.tree_of_thought.requests.post") as mock_post:
            mock_post.return_value = _mock_response("B1: 8\nB2: 6\nB3: 9")
            tot._evaluate_branches("test query", branches)

        assert branches[0].ego_score == 8.0
        assert branches[1].ego_score == 6.0
        assert branches[2].ego_score == 9.0


class TestDebateLog:
    """Tests for the UI debate log formatting."""

    def test_includes_all_branches(self, tot) -> None:
        branches = [
            ReasoningBranch(
                "superego",
                "logic",
                "answer1",
                self_score=8,
                ego_score=7,
            ),
            ReasoningBranch(
                "id",
                "intuition",
                "answer2",
                self_score=6,
                ego_score=9,
            ),
        ]
        selected = [branches[1]]

        log = tot._format_debate_log(branches, selected)
        assert "Superego" in log
        assert "Id" in log
        assert "logic" in log
        assert "intuition" in log
        assert "★" in log  # Star for selected


class TestFullPipeline:
    """Tests for the complete reason() pipeline."""

    def test_produces_result(self, tot) -> None:
        # Mock all LLM calls
        responses = [
            # Superego branches
            _mock_response(
                "APPROACH 1: Logic\n"
                "ANSWER: The logical answer is X.\n"
                "CONFIDENCE: 8\n\n"
                "APPROACH 2: Rules\n"
                "ANSWER: The rules say Y.\n"
                "CONFIDENCE: 7\n"
            ),
            # Id branches
            _mock_response(
                "APPROACH 1: Gut feeling\n"
                "ANSWER: It feels like Z.\n"
                "CONFIDENCE: 6\n\n"
                "APPROACH 2: Analogy\n"
                "ANSWER: Like cooking, it's W.\n"
                "CONFIDENCE: 7\n"
            ),
            # Ego evaluation
            _mock_response("B1: 8\nB2: 7\nB3: 6\nB4: 9"),
            # Ego synthesis
            _mock_response("The answer combines logic and intuition..."),
        ]

        with patch("agi.reasoning.tree_of_thought.requests.post") as mock_post:
            mock_post.side_effect = responses
            result = tot.reason("What is the meaning of life?")

        assert isinstance(result, TreeResult)
        assert result.method == "tree_of_thought"
        assert result.total_branches >= 2
        assert len(result.selected_branches) >= 1
        assert result.synthesis != ""
        assert result.latency_s >= 0
        assert "Reasoning Branches" in result.debate_log


class TestTreeResultDataclass:
    """Tests for the TreeResult structure."""

    def test_creation(self) -> None:
        r = TreeResult(query="test")
        assert r.method == "tree_of_thought"
        assert r.total_branches == 0
        assert r.branches == []

    def test_with_branches(self) -> None:
        r = TreeResult(
            query="test",
            branches=[
                ReasoningBranch("superego", "a", "content"),
            ],
            total_branches=1,
            synthesis="The answer is...",
            latency_s=5.0,
        )
        assert r.total_branches == 1
        assert r.synthesis.startswith("The answer")
