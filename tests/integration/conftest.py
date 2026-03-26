# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Shared fixtures for integration tests."""

import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_event_fabric():
    """Mock EventFabric that captures published events."""

    class MockFabric:
        def __init__(self):
            self.published_events = []
            self.subscriptions = {}

        def publish(self, topic: str, payload: dict):
            self.published_events.append((topic, payload))

        def subscribe(self, topic: str, handler):
            self.subscriptions[topic] = handler

        def close(self):
            pass

    return MockFabric()


@pytest.fixture
def mock_lh_planner():
    """Mock LH Planner that returns canned plans."""
    planner = MagicMock()
    planner.plan.return_value = MagicMock(
        plan_id="test-plan-001",
        steps=[
            MagicMock(
                step_id="step-1",
                kind="navigate",
                description="Navigate to target",
            ),
            MagicMock(
                step_id="step-2",
                kind="manipulate",
                description="Pick up object",
            ),
        ],
    )
    return planner


@pytest.fixture
def mock_rh_control():
    """Mock RH ControlService."""
    control = MagicMock()
    control.translate_step.return_value = [
        {"type": "move", "target": "[1.0, 0.0, 0.0]", "magnitude": 0.5, "duration": 0.5}
    ]
    control.execute_actions = AsyncMock(
        return_value=[
            MagicMock(success=True, observation=None, reward=1.0, done=False, info={})
        ]
    )
    return control


@pytest.fixture
def mock_safety_gateway():
    """Mock SafetyGateway that can approve or reject."""
    gateway = MagicMock()
    gateway.check.return_value = MagicMock(approved=True, issues=[], verdict="APPROVED")
    return gateway


@pytest.fixture
def chaos_config():
    """ChaosConfig for testing."""
    from tests.integration.chaos import ChaosConfig

    return ChaosConfig(
        enabled=True,
        failure_rate=0.5,
        latency_ms=50.0,
        seed=42,
    )
