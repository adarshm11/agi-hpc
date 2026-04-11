# AGI-HPC Temporal Cognition Tests
from __future__ import annotations

import pytest

from agi.metacognition.temporal import TemporalCognition


@pytest.fixture
def tc():
    return TemporalCognition(window_size=10)


class TestBoredom:
    def test_no_queries(self, tc):
        assert tc.state.boredom_score == 0.0

    def test_single_query(self, tc):
        tc.observe_query("What is PCA?")
        assert tc.state.boredom_score == 0.0

    def test_repeated_queries_increase_boredom(self, tc):
        for _ in range(5):
            tc.observe_query("What is PCA?")
        assert tc.state.boredom_score > 0.5

    def test_diverse_queries_low_boredom(self, tc):
        queries = [
            "What is PCA?",
            "How does Docker work?",
            "Explain quantum computing",
            "Best pizza in SF",
            "Python async patterns",
        ]
        for q in queries:
            tc.observe_query(q)
        assert tc.state.boredom_score < 0.3

    def test_boredom_bounded(self, tc):
        for _ in range(20):
            tc.observe_query("same exact query")
        assert tc.state.boredom_score <= 1.0


class TestPatience:
    def test_simple_query(self, tc):
        p = tc.get_patience(1)
        assert p == pytest.approx(1.4)

    def test_complex_query(self, tc):
        p = tc.get_patience(5)
        assert p == pytest.approx(3.0)

    def test_scales_with_complexity(self, tc):
        p1 = tc.get_patience(1)
        p3 = tc.get_patience(3)
        p5 = tc.get_patience(5)
        assert p1 < p3 < p5


class TestPacing:
    def test_slow_queries_no_pace(self, tc):
        tc.observe_query("query 1")
        assert not tc.state.should_pace

    def test_state_accessible(self, tc):
        state = tc.observe_query("test")
        assert state.boredom_score >= 0.0
        assert state.queries_per_minute >= 0.0
