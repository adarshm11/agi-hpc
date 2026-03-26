# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.rh.control.primitives module.

Tests the real source API: ExecutionContext, PrimitiveResult,
ReachPrimitive, GraspPrimitive, PlacePrimitive, NavigationPrimitive,
PrimitiveLibrary, and the MotorPrimitive protocol.
"""

import numpy as np
import pytest

from agi.rh.control.primitives import (
    ExecutionContext,
    GraspPrimitive,
    MotorPrimitive,
    NavigationPrimitive,
    PlacePrimitive,
    PrimitiveLibrary,
    PrimitiveResult,
    ReachPrimitive,
)


class TestExecutionContext:
    """Tests for ExecutionContext dataclass."""

    def test_default_fields(self):
        ctx = ExecutionContext()
        assert ctx.robot_state == {}
        assert ctx.world_state == {}
        assert ctx.parameters == {}
        assert ctx.timestamp == 0.0
        assert ctx.step_count == 0

    def test_custom_fields(self):
        ctx = ExecutionContext(
            robot_state={"position": [1.0, 2.0, 3.0], "gripper": 0.5},
            world_state={"objects": ["box"]},
            parameters={"target": [1.0, 0.0, 0.0]},
            timestamp=1.5,
            step_count=10,
        )
        assert ctx.robot_state["position"] == [1.0, 2.0, 3.0]
        assert ctx.parameters["target"] == [1.0, 0.0, 0.0]
        assert ctx.step_count == 10

    def test_get_position_default(self):
        ctx = ExecutionContext()
        pos = ctx.get_position()
        np.testing.assert_array_equal(pos, [0.0, 0.0, 0.0])

    def test_get_position_with_state(self):
        ctx = ExecutionContext(robot_state={"position": [1.0, 2.0, 3.0]})
        pos = ctx.get_position()
        np.testing.assert_array_almost_equal(pos, [1.0, 2.0, 3.0])

    def test_get_gripper_state_default(self):
        ctx = ExecutionContext()
        assert ctx.get_gripper_state() == 0.0

    def test_get_gripper_state_with_value(self):
        ctx = ExecutionContext(robot_state={"gripper": 0.75})
        assert ctx.get_gripper_state() == pytest.approx(0.75)


class TestPrimitiveResult:
    """Tests for PrimitiveResult dataclass."""

    def test_required_fields(self):
        r = PrimitiveResult(success=True, primitive_name="reach")
        assert r.success is True
        assert r.primitive_name == "reach"
        assert r.duration == 0.0
        assert r.error is None
        assert r.actions_generated == 0

    def test_failure_result(self):
        r = PrimitiveResult(
            success=False,
            primitive_name="grasp",
            error="Preconditions not met",
        )
        assert r.success is False
        assert r.error == "Preconditions not met"

    def test_full_result(self):
        r = PrimitiveResult(
            success=True,
            primitive_name="place",
            duration=0.05,
            actions_generated=4,
            final_state={"position": [1, 0, 0]},
            metadata={"smoothed": True},
        )
        assert r.actions_generated == 4
        assert r.final_state["position"] == [1, 0, 0]
        assert r.metadata["smoothed"] is True


class TestReachPrimitive:
    """Tests for ReachPrimitive."""

    def test_name(self):
        prim = ReachPrimitive()
        assert prim.name == "reach"

    def test_can_execute_with_target(self):
        prim = ReachPrimitive()
        ctx = ExecutionContext(parameters={"target": [1.0, 0.0, 0.0]})
        assert prim.can_execute(ctx) is True

    def test_can_execute_without_target(self):
        prim = ReachPrimitive()
        ctx = ExecutionContext()
        assert prim.can_execute(ctx) is False

    def test_generate_actions_produces_moves(self):
        prim = ReachPrimitive(speed=0.5, tolerance=0.01)
        ctx = ExecutionContext(
            robot_state={"position": [0.0, 0.0, 0.0]},
            parameters={"target": [1.0, 0.0, 0.0]},
        )
        actions = prim.generate_actions(ctx)
        assert len(actions) > 0
        assert all(a["type"] == "move" for a in actions)

    def test_generate_actions_at_target_returns_empty(self):
        prim = ReachPrimitive(tolerance=0.01)
        ctx = ExecutionContext(
            robot_state={"position": [1.0, 0.0, 0.0]},
            parameters={"target": [1.0, 0.0, 0.0]},
        )
        actions = prim.generate_actions(ctx)
        assert actions == []

    def test_execute_returns_result(self):
        prim = ReachPrimitive()
        ctx = ExecutionContext(
            robot_state={"position": [0.0, 0.0, 0.0]},
            parameters={"target": [0.5, 0.0, 0.0]},
        )
        result = prim.execute(ctx)
        assert isinstance(result, PrimitiveResult)
        assert result.success is True
        assert result.primitive_name == "reach"
        assert result.actions_generated > 0


class TestGraspPrimitive:
    """Tests for GraspPrimitive."""

    def test_name(self):
        prim = GraspPrimitive()
        assert prim.name == "grasp"

    def test_can_execute_gripper_open(self):
        prim = GraspPrimitive()
        ctx = ExecutionContext(robot_state={"gripper": 0.5})
        assert prim.can_execute(ctx) is True

    def test_can_execute_gripper_closed(self):
        prim = GraspPrimitive()
        ctx = ExecutionContext(robot_state={"gripper": 0.0})
        assert prim.can_execute(ctx) is False

    def test_generate_actions(self):
        prim = GraspPrimitive(force=0.8)
        ctx = ExecutionContext(robot_state={"gripper": 1.0})
        actions = prim.generate_actions(ctx)
        assert len(actions) == 2
        assert actions[0]["type"] == "grasp"
        assert actions[1]["type"] == "hold"

    def test_execute_returns_success(self):
        prim = GraspPrimitive()
        ctx = ExecutionContext(robot_state={"gripper": 1.0})
        result = prim.execute(ctx)
        assert result.success is True
        assert result.primitive_name == "grasp"


class TestPlacePrimitive:
    """Tests for PlacePrimitive."""

    def test_name(self):
        prim = PlacePrimitive()
        assert prim.name == "place"

    def test_can_execute_with_target(self):
        prim = PlacePrimitive()
        ctx = ExecutionContext(parameters={"target": [1.0, 0.0, 0.0]})
        assert prim.can_execute(ctx) is True

    def test_can_execute_without_target(self):
        prim = PlacePrimitive()
        ctx = ExecutionContext()
        assert prim.can_execute(ctx) is False

    def test_generate_actions(self):
        prim = PlacePrimitive(release_height=0.05)
        ctx = ExecutionContext(parameters={"target": [1.0, 0.0, 0.0]})
        actions = prim.generate_actions(ctx)
        assert len(actions) == 4
        assert actions[0]["type"] == "move"
        assert actions[2]["type"] == "release"


class TestNavigationPrimitive:
    """Tests for NavigationPrimitive."""

    def test_name(self):
        prim = NavigationPrimitive()
        assert prim.name == "navigate"

    def test_can_execute_with_goal(self):
        prim = NavigationPrimitive()
        ctx = ExecutionContext(parameters={"goal": [5.0, 5.0, 0.0]})
        assert prim.can_execute(ctx) is True

    def test_can_execute_without_goal(self):
        prim = NavigationPrimitive()
        ctx = ExecutionContext()
        assert prim.can_execute(ctx) is False

    def test_generate_actions_distant_goal(self):
        prim = NavigationPrimitive(max_speed=1.0)
        ctx = ExecutionContext(
            robot_state={"position": [0.0, 0.0, 0.0]},
            parameters={"goal": [3.0, 4.0, 0.0]},
        )
        actions = prim.generate_actions(ctx)
        assert len(actions) == 1
        assert actions[0]["type"] == "navigate"
        assert actions[0]["speed"] == 1.0

    def test_generate_actions_near_goal_returns_empty(self):
        prim = NavigationPrimitive()
        ctx = ExecutionContext(
            robot_state={"position": [0.0, 0.0, 0.0]},
            parameters={"goal": [0.05, 0.0, 0.0]},
        )
        actions = prim.generate_actions(ctx)
        assert actions == []


class TestPrimitiveLibrary:
    """Tests for PrimitiveLibrary."""

    def test_defaults_registered(self):
        lib = PrimitiveLibrary()
        names = lib.list_primitives()
        assert "reach" in names
        assert "grasp" in names
        assert "place" in names
        assert "navigate" in names

    def test_get_existing(self):
        lib = PrimitiveLibrary()
        prim = lib.get("reach")
        assert prim is not None
        assert prim.name == "reach"

    def test_get_missing(self):
        lib = PrimitiveLibrary()
        assert lib.get("nonexistent") is None

    def test_register_custom(self):
        lib = PrimitiveLibrary()
        custom = ReachPrimitive(speed=2.0)
        lib.register(custom)
        retrieved = lib.get("reach")
        assert retrieved is custom

    def test_execute_success(self):
        lib = PrimitiveLibrary()
        ctx = ExecutionContext(
            robot_state={"position": [0.0, 0.0, 0.0]},
            parameters={"target": [1.0, 0.0, 0.0]},
        )
        result = lib.execute("reach", ctx)
        assert result.success is True
        assert result.primitive_name == "reach"

    def test_execute_unknown_primitive(self):
        lib = PrimitiveLibrary()
        ctx = ExecutionContext()
        result = lib.execute("unknown", ctx)
        assert result.success is False
        assert "Unknown primitive" in result.error

    def test_execute_preconditions_not_met(self):
        lib = PrimitiveLibrary()
        ctx = ExecutionContext()  # no "target" in parameters
        result = lib.execute("reach", ctx)
        assert result.success is False
        assert result.error == "Preconditions not met"


class TestMotorPrimitiveProtocol:
    """Tests for MotorPrimitive protocol (runtime_checkable)."""

    def test_reach_satisfies_protocol(self):
        assert isinstance(ReachPrimitive(), MotorPrimitive)

    def test_grasp_satisfies_protocol(self):
        assert isinstance(GraspPrimitive(), MotorPrimitive)

    def test_place_satisfies_protocol(self):
        assert isinstance(PlacePrimitive(), MotorPrimitive)

    def test_navigate_satisfies_protocol(self):
        assert isinstance(NavigationPrimitive(), MotorPrimitive)
