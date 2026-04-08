# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the Executive Function module.

Tests all five core functions: mode selection, goal tracking,
task decomposition, working memory, and inhibition control.
"""

from __future__ import annotations

from agi.metacognition.executive_function import (
    ExecutiveDecision,
    ExecutiveFunction,
)


class TestComplexityEstimation:
    """Tests for query complexity scoring."""

    def test_simple_factual(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("What is the capital of France?")
        assert d.complexity <= 2

    def test_moderate_explanation(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("How does quantum entanglement work?")
        assert d.complexity >= 2

    def test_complex_comparison(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "Compare and contrast utilitarian and "
            "deontological ethics, evaluating the "
            "trade-offs of each approach"
        )
        assert d.complexity >= 3

    def test_deep_ethical(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "Is it ethical to create AI systems that can "
            "suffer? What are the moral implications and "
            "competing values at stake?"
        )
        assert d.complexity >= 4

    def test_multi_question(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "What is X? How does it relate to Y? " "And should we use Z instead?"
        )
        assert d.complexity >= 3


class TestModeSelection:
    """Tests for reasoning mode selection."""

    def test_simple_gets_single(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("What year was Python created?")
        assert d.mode == "single"

    def test_complex_gets_debate(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "Analyze the relationship between privacy " "and security in AI systems"
        )
        assert d.mode in ("debate", "tot")

    def test_deep_gets_tot(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "From multiple perspectives, evaluate the "
            "ethical dilemma of autonomous weapons. "
            "Consider competing values and moral frameworks."
        )
        assert d.mode == "tot"


class TestTaskDecomposition:
    """Tests for multi-part query decomposition."""

    def test_numbered_list(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "1. What is machine learning?\n"
            "2. How does it differ from deep learning?\n"
            "3. What are the main applications?"
        )
        assert len(d.sub_queries) >= 3

    def test_multiple_questions(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "What is the speed of light? "
            "How was it first measured? "
            "Why can't anything go faster?"
        )
        assert len(d.sub_queries) >= 2

    def test_single_question_not_decomposed(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("What is the meaning of life?")
        assert len(d.sub_queries) <= 1

    def test_and_then_separator(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "First explain quantum computing and then "
            "compare it to classical computing"
        )
        assert len(d.sub_queries) >= 2


class TestInhibition:
    """Tests for inhibition control (ask for clarification)."""

    def test_very_short_query_inhibited(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("hi")
        assert d.inhibit

    def test_dangling_reference_inhibited(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("What about it")
        assert d.inhibit

    def test_clear_query_not_inhibited(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("Explain the theory of relativity")
        assert not d.inhibit

    def test_same_as_before_no_history(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("Do the same as before")
        assert d.inhibit


class TestGoalTracking:
    """Tests for cross-turn goal detection."""

    def test_no_history_no_goal(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("What is Python?")
        assert not d.goal_continuation

    def test_explicit_continuation(self) -> None:
        history = [
            {"role": "user", "content": "Explain quantum computing"},
            {"role": "assistant", "content": "Quantum computing uses qubits..."},
        ]
        ef = ExecutiveFunction(session_history=history)
        ef._turn_count = 1
        d = ef.decide("Building on that, what about quantum error correction?")
        assert d.goal_continuation

    def test_topic_overlap_detected(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Explain quantum entanglement superposition and decoherence",
            },
            {"role": "assistant", "content": "Entanglement is..."},
        ]
        ef = ExecutiveFunction(session_history=history)
        ef._turn_count = 1
        d = ef.decide(
            "How does quantum entanglement and superposition "
            "relate to decoherence in practice?"
        )
        assert d.goal_continuation


class TestContextStrategy:
    """Tests for context retrieval strategy selection."""

    def test_simple_gets_minimal(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide("What is 2+2?")
        assert d.context_strategy == "minimal"

    def test_episodic_for_references(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "You told me about quantum computing last time. " "Can you elaborate?"
        )
        assert d.context_strategy == "episodic_recent"

    def test_deep_search_for_research(self) -> None:
        ef = ExecutiveFunction()
        d = ef.decide(
            "What does the research say about the "
            "effectiveness of transformer architectures?"
        )
        assert d.context_strategy == "semantic_deep"


class TestDecisionDataclass:
    """Tests for ExecutiveDecision structure."""

    def test_defaults(self) -> None:
        d = ExecutiveDecision()
        assert d.mode == "single"
        assert d.complexity == 1
        assert d.sub_queries == []
        assert not d.inhibit
        assert d.context_strategy == "default"
        assert not d.goal_continuation
