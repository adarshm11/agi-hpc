# AGI-HPC Cascade Detector Tests
from __future__ import annotations

import pytest

from agi.metacognition.cascade import CascadeDetector


@pytest.fixture
def detector():
    return CascadeDetector()


class TestHealthySystem:
    def test_all_healthy(self, detector):
        health = {name: "healthy" for name in detector._graph}
        result = detector.evaluate(health)
        assert all(h.status == "healthy" for h in result)
        assert detector.get_degraded() == []


class TestSingleFailure:
    def test_leaf_failure_no_cascade(self, detector):
        health = {name: "healthy" for name in detector._graph}
        health["temporal"] = "failed"
        detector.evaluate(health)
        degraded = detector.get_degraded()
        assert "temporal" in degraded
        # temporal has no dependents, so only itself
        assert len(degraded) == 1

    def test_safety_failure_cascades_to_lh_rh(self, detector):
        health = {name: "healthy" for name in detector._graph}
        health["safety_gateway"] = "failed"
        detector.evaluate(health)
        degraded = detector.get_degraded()
        assert "safety_gateway" in degraded
        assert "lh" in degraded
        assert "rh" in degraded

    def test_semantic_failure_cascades_deeply(self, detector):
        health = {name: "healthy" for name in detector._graph}
        health["semantic_memory"] = "failed"
        detector.evaluate(health)
        degraded = detector.get_degraded()
        assert "semantic_memory" in degraded
        assert "knowledge_graph" in degraded
        assert "research_loop" in degraded
        assert "executive_function" in degraded


class TestMultipleFailures:
    def test_two_independent_failures(self, detector):
        health = {name: "healthy" for name in detector._graph}
        health["temporal"] = "failed"
        health["procedural_memory"] = "failed"
        detector.evaluate(health)
        degraded = detector.get_degraded()
        assert len(degraded) == 2

    def test_cascading_chain(self, detector):
        health = {name: "healthy" for name in detector._graph}
        health["safety_gateway"] = "failed"
        detector.evaluate(health)
        degraded = detector.get_degraded()
        # safety -> lh, rh -> ego (cascade chain)
        assert "ego" in degraded


class TestCascadeChains:
    def test_identifies_root_cause(self, detector):
        health = {name: "healthy" for name in detector._graph}
        health["safety_gateway"] = "failed"
        detector.evaluate(health)
        chains = detector.get_cascade_chains()
        assert "safety_gateway" in chains
        assert "lh" in chains["safety_gateway"]
        assert "rh" in chains["safety_gateway"]

    def test_no_chains_when_healthy(self, detector):
        health = {name: "healthy" for name in detector._graph}
        detector.evaluate(health)
        assert detector.get_cascade_chains() == {}


class TestDegradedStatus:
    def test_cascaded_is_degraded_not_failed(self, detector):
        health = {name: "healthy" for name in detector._graph}
        health["safety_gateway"] = "failed"
        result = detector.evaluate(health)
        lh = next(h for h in result if h.name == "lh")
        assert lh.status == "degraded"
        assert not lh.direct_failure
        assert lh.cascaded_from == "safety_gateway"


class TestCustomGraph:
    def test_simple_chain(self):
        graph = {"a": [], "b": ["a"], "c": ["b"]}
        detector = CascadeDetector(graph)
        health = {"a": "failed", "b": "healthy", "c": "healthy"}
        detector.evaluate(health)
        degraded = detector.get_degraded()
        assert "a" in degraded
        assert "b" in degraded
        assert "c" in degraded
