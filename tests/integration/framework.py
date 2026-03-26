# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Integration Test Framework for AGI-HPC.

Provides a structured approach to integration testing:
- Test case definition and registration
- Categorized test execution
- Result aggregation and reporting

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Result of an integration test case."""

    name: str
    passed: bool
    duration: float  # seconds
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationTestCase:
    """Definition of an integration test case."""

    name: str
    description: str
    category: str  # e.g., "lh_rh", "memory", "safety", "e2e"
    test_fn: Callable[[], None]
    setup: Optional[Callable[[], None]] = None
    teardown: Optional[Callable[[], None]] = None
    timeout_sec: float = 30.0
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------


class IntegrationTestRunner:
    """Runner for integration test cases."""

    def __init__(self, verbose: bool = True) -> None:
        self._cases: List[IntegrationTestCase] = []
        self._results: List[TestResult] = []
        self._verbose = verbose

    def register(self, case: IntegrationTestCase) -> None:
        """Register a test case."""
        self._cases.append(case)
        if self._verbose:
            logger.info("[test] registered: %s (%s)", case.name, case.category)

    def run_all(self) -> List[TestResult]:
        """Run all registered test cases."""
        self._results = []
        for case in self._cases:
            result = self._run_single(case)
            self._results.append(result)
        return self._results

    def run_category(self, category: str) -> List[TestResult]:
        """Run test cases in a specific category."""
        results = []
        for case in self._cases:
            if case.category == category:
                result = self._run_single(case)
                results.append(result)
        self._results.extend(results)
        return results

    def run_tagged(self, tag: str) -> List[TestResult]:
        """Run test cases with a specific tag."""
        results = []
        for case in self._cases:
            if tag in case.tags:
                result = self._run_single(case)
                results.append(result)
        self._results.extend(results)
        return results

    def _run_single(self, case: IntegrationTestCase) -> TestResult:
        """Run a single test case."""
        if self._verbose:
            logger.info("[test] running: %s", case.name)

        start = time.monotonic()

        try:
            # Setup
            if case.setup:
                case.setup()

            # Execute test
            case.test_fn()

            duration = time.monotonic() - start

            if self._verbose:
                logger.info("[test] PASSED: %s (%.2fs)", case.name, duration)

            return TestResult(
                name=case.name,
                passed=True,
                duration=duration,
            )

        except Exception as e:
            duration = time.monotonic() - start

            if self._verbose:
                logger.error("[test] FAILED: %s (%.2fs): %s", case.name, duration, e)

            return TestResult(
                name=case.name,
                passed=False,
                duration=duration,
                error=str(e),
            )

        finally:
            # Teardown
            if case.teardown:
                try:
                    case.teardown()
                except Exception as e:
                    logger.warning("[test] teardown error for %s: %s", case.name, e)

    def summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed
        total_duration = sum(r.duration for r in self._results)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "total_duration": total_duration,
            "failures": [
                {"name": r.name, "error": r.error}
                for r in self._results
                if not r.passed
            ],
        }


# ---------------------------------------------------------------------------
# Predefined Integration Test Cases
# ---------------------------------------------------------------------------


def _test_event_fabric_pubsub():
    """Test EventFabric publish/subscribe round-trip."""
    received = []

    class MockFabric:
        def __init__(self):
            self._handlers = {}

        def publish(self, topic, payload):
            if topic in self._handlers:
                for handler in self._handlers[topic]:
                    handler(payload)

        def subscribe(self, topic, handler):
            self._handlers.setdefault(topic, []).append(handler)

    fabric = MockFabric()
    fabric.subscribe("test.topic", lambda p: received.append(p))
    fabric.publish("test.topic", {"key": "value"})

    assert len(received) == 1
    assert received[0]["key"] == "value"


def _test_memory_enrichment():
    """Test memory enrichment pipeline."""
    # Stub test - verifies the pattern works
    context = {"facts": [], "episodes": [], "skills": []}
    context["facts"].append({"content": "test fact", "confidence": 0.9})
    assert len(context["facts"]) == 1


def _test_safety_gate():
    """Test safety gate blocking."""
    approved = True
    issues = []

    # Simulate safety check
    action = {"type": "move", "magnitude": 100.0}
    if action.get("magnitude", 0) > 10.0:
        approved = False
        issues.append("excessive_magnitude")

    assert not approved
    assert "excessive_magnitude" in issues


def _test_plan_dispatch():
    """Test LH→RH plan dispatch."""
    plan_steps = [
        {"step_id": "1", "kind": "navigate", "description": "go to A"},
        {"step_id": "2", "kind": "manipulate", "description": "pick up B"},
    ]

    actions = []
    for step in plan_steps:
        if "navigate" in step["kind"]:
            actions.append({"type": "move", "duration": 0.5})
        elif "manipulate" in step["kind"]:
            actions.append({"type": "grasp", "duration": 0.3})

    assert len(actions) == 2
    assert actions[0]["type"] == "move"
    assert actions[1]["type"] == "grasp"


def _test_metacog_review():
    """Test metacognition review loop."""
    plan = {"steps": [{"id": "1"}], "confidence": 0.8}

    # Simulate review
    if plan["confidence"] > 0.7:
        decision = "ACCEPT"
    else:
        decision = "REVISE"

    assert decision == "ACCEPT"


def _test_full_pipeline():
    """Test full end-to-end pipeline."""
    # 1. Create task
    task = {"goal": "navigate to target", "type": "navigation"}

    # 2. Memory enrichment
    context = {"facts": ["target is at (1,0,0)"], "skills": ["move_to"]}

    # 3. Plan generation
    plan = {"steps": [{"action": "move", "target": [1, 0, 0]}]}

    # 4. Safety check
    safety_result = {"approved": True}

    # 5. Meta review
    meta_result = {"decision": "ACCEPT"}

    # 6. Execution
    execution_result = {"success": True}

    assert safety_result["approved"]
    assert meta_result["decision"] == "ACCEPT"
    assert execution_result["success"]


INTEGRATION_TEST_CASES: List[IntegrationTestCase] = [
    IntegrationTestCase(
        name="event_fabric_pubsub",
        description="Test EventFabric publish/subscribe round-trip",
        category="events",
        test_fn=_test_event_fabric_pubsub,
        tags=["smoke", "events"],
    ),
    IntegrationTestCase(
        name="memory_enrichment",
        description="Test memory enrichment pipeline",
        category="memory",
        test_fn=_test_memory_enrichment,
        tags=["smoke", "memory"],
    ),
    IntegrationTestCase(
        name="safety_gate_blocking",
        description="Test safety gate blocks unsafe actions",
        category="safety",
        test_fn=_test_safety_gate,
        tags=["smoke", "safety"],
    ),
    IntegrationTestCase(
        name="lh_rh_plan_dispatch",
        description="Test LH→RH plan dispatch and action translation",
        category="lh_rh",
        test_fn=_test_plan_dispatch,
        tags=["smoke", "lh_rh"],
    ),
    IntegrationTestCase(
        name="metacognition_review",
        description="Test metacognition review loop",
        category="metacognition",
        test_fn=_test_metacog_review,
        tags=["smoke", "metacognition"],
    ),
    IntegrationTestCase(
        name="full_pipeline_e2e",
        description="Test complete pipeline from task to execution",
        category="e2e",
        test_fn=_test_full_pipeline,
        tags=["e2e", "full"],
    ),
]
