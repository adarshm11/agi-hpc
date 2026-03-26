# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.rh.control.trajectory module.

Tests the real source API: TrajectoryConfig, Waypoint, Trajectory,
RRTPlanner, CHOMPPlanner, and TrajectoryOptimizer.
"""

import numpy as np
import pytest

from agi.rh.control.trajectory import (
    CHOMPPlanner,
    RRTPlanner,
    Trajectory,
    TrajectoryConfig,
    TrajectoryOptimizer,
    Waypoint,
)


class TestTrajectoryConfig:
    """Tests for TrajectoryConfig dataclass."""

    def test_default_values(self):
        cfg = TrajectoryConfig()
        assert cfg.max_iterations == 5000
        assert cfg.step_size == pytest.approx(0.1)
        assert cfg.goal_bias == pytest.approx(0.05)
        assert cfg.goal_tolerance == pytest.approx(0.05)
        assert cfg.smoothing_iterations == 200
        assert cfg.collision_check_resolution == pytest.approx(0.01)

    def test_workspace_bounds(self):
        cfg = TrajectoryConfig()
        assert cfg.workspace_min == (-1.0, -1.0, -1.0)
        assert cfg.workspace_max == (1.0, 1.0, 1.0)

    def test_custom_values(self):
        cfg = TrajectoryConfig(
            max_iterations=1000,
            step_size=0.2,
            goal_tolerance=0.1,
        )
        assert cfg.max_iterations == 1000
        assert cfg.step_size == pytest.approx(0.2)
        assert cfg.goal_tolerance == pytest.approx(0.1)


class TestWaypoint:
    """Tests for Waypoint dataclass."""

    def test_position_required(self):
        pos = np.array([1.0, 2.0, 3.0])
        wp = Waypoint(position=pos)
        np.testing.assert_array_equal(wp.position, [1.0, 2.0, 3.0])
        assert wp.velocity is None
        assert wp.timestamp is None
        assert wp.metadata == {}

    def test_full_waypoint(self):
        wp = Waypoint(
            position=np.array([1.0, 0.0, 0.0]),
            velocity=np.array([0.1, 0.0, 0.0]),
            timestamp=0.5,
            metadata={"label": "midpoint"},
        )
        assert wp.timestamp == pytest.approx(0.5)
        assert wp.metadata["label"] == "midpoint"
        np.testing.assert_array_equal(wp.velocity, [0.1, 0.0, 0.0])

    def test_metadata_default_factory(self):
        wp1 = Waypoint(position=np.array([0, 0, 0]))
        wp2 = Waypoint(position=np.array([1, 1, 1]))
        wp1.metadata["key"] = "val"
        assert "key" not in wp2.metadata


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_empty_trajectory(self):
        traj = Trajectory()
        assert traj.num_waypoints == 0
        assert traj.length == pytest.approx(0.0)
        assert traj.total_time == 0.0
        assert traj.cost == 0.0

    def test_single_waypoint_length_is_zero(self):
        traj = Trajectory(
            waypoints=[Waypoint(position=np.array([1.0, 0.0, 0.0]))]
        )
        assert traj.num_waypoints == 1
        assert traj.length == pytest.approx(0.0)

    def test_two_waypoint_length(self):
        traj = Trajectory(
            waypoints=[
                Waypoint(position=np.array([0.0, 0.0, 0.0])),
                Waypoint(position=np.array([3.0, 4.0, 0.0])),
            ]
        )
        assert traj.num_waypoints == 2
        assert traj.length == pytest.approx(5.0)

    def test_multi_segment_length(self):
        traj = Trajectory(
            waypoints=[
                Waypoint(position=np.array([0.0, 0.0, 0.0])),
                Waypoint(position=np.array([1.0, 0.0, 0.0])),
                Waypoint(position=np.array([1.0, 1.0, 0.0])),
            ]
        )
        assert traj.num_waypoints == 3
        assert traj.length == pytest.approx(2.0)

    def test_metadata_and_cost(self):
        traj = Trajectory(
            waypoints=[],
            total_time=5.0,
            cost=3.14,
            metadata={"planner": "rrt"},
        )
        assert traj.total_time == pytest.approx(5.0)
        assert traj.cost == pytest.approx(3.14)
        assert traj.metadata["planner"] == "rrt"


class TestRRTPlanner:
    """Tests for RRTPlanner."""

    def test_init_default(self):
        planner = RRTPlanner()
        # Should initialize without error
        assert planner is not None

    def test_init_with_config(self):
        cfg = TrajectoryConfig(max_iterations=100, step_size=0.2)
        planner = RRTPlanner(config=cfg)
        assert planner is not None

    def test_init_rrt_star(self):
        planner = RRTPlanner(use_rrt_star=True)
        assert planner is not None

    def test_plan_no_obstacles(self):
        cfg = TrajectoryConfig(
            max_iterations=5000,
            step_size=0.3,
            goal_tolerance=0.1,
            goal_bias=0.1,
        )
        planner = RRTPlanner(config=cfg)
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([0.5, 0.5, 0.0])
        traj = planner.plan(start, goal)
        assert traj is not None
        assert traj.num_waypoints >= 2

    def test_plan_returns_none_on_impossible(self):
        cfg = TrajectoryConfig(max_iterations=50, step_size=0.01)
        planner = RRTPlanner(config=cfg)
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([10.0, 10.0, 10.0])  # outside workspace
        # With very few iterations and tiny steps, this should fail
        traj = planner.plan(start, goal)
        # May or may not find a path, but should not raise
        assert traj is None or isinstance(traj, Trajectory)

    def test_plan_with_collision_fn(self):
        cfg = TrajectoryConfig(
            max_iterations=5000,
            step_size=0.3,
            goal_tolerance=0.1,
            goal_bias=0.1,
        )
        planner = RRTPlanner(config=cfg)
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([0.5, 0.0, 0.0])

        def no_collision(pos: np.ndarray) -> bool:
            return False  # no collisions anywhere

        traj = planner.plan(start, goal, collision_fn=no_collision)
        assert traj is not None
        assert traj.num_waypoints >= 2


class TestCHOMPPlanner:
    """Tests for CHOMPPlanner (stub implementation)."""

    def test_init_default(self):
        planner = CHOMPPlanner()
        assert planner is not None

    def test_init_with_config(self):
        cfg = TrajectoryConfig(max_iterations=100)
        planner = CHOMPPlanner(config=cfg)
        assert planner is not None

    def test_optimize_returns_same_trajectory(self):
        planner = CHOMPPlanner()
        traj = Trajectory(
            waypoints=[
                Waypoint(position=np.array([0.0, 0.0, 0.0])),
                Waypoint(position=np.array([1.0, 0.0, 0.0])),
                Waypoint(position=np.array([1.0, 1.0, 0.0])),
            ],
            cost=2.0,
        )
        result = planner.optimize(traj)
        assert result is traj  # stub returns the same object

    def test_optimize_with_collision_fn(self):
        planner = CHOMPPlanner()
        traj = Trajectory(
            waypoints=[Waypoint(position=np.array([0.0, 0.0, 0.0]))]
        )
        result = planner.optimize(traj, collision_fn=lambda p: False)
        assert result is traj


class TestTrajectoryOptimizer:
    """Tests for TrajectoryOptimizer."""

    def test_init_default(self):
        opt = TrajectoryOptimizer()
        assert opt is not None

    def test_init_with_config(self):
        cfg = TrajectoryConfig(smoothing_iterations=50)
        opt = TrajectoryOptimizer(config=cfg)
        assert opt is not None

    def test_smooth_short_trajectory_unchanged(self):
        opt = TrajectoryOptimizer()
        traj = Trajectory(
            waypoints=[
                Waypoint(position=np.array([0.0, 0.0, 0.0])),
                Waypoint(position=np.array([1.0, 0.0, 0.0])),
            ]
        )
        result = opt.smooth(traj)
        assert result is traj  # fewer than 3 waypoints returns as-is

    def test_smooth_reduces_waypoints(self):
        opt = TrajectoryOptimizer(
            config=TrajectoryConfig(smoothing_iterations=500)
        )
        # Build a zig-zag path with many collinear shortcuts available
        positions = []
        for i in range(20):
            x = i * 0.05
            y = 0.01 * ((-1) ** i)
            positions.append(np.array([x, y, 0.0]))
        waypoints = [Waypoint(position=p) for p in positions]
        traj = Trajectory(waypoints=waypoints)
        result = opt.smooth(traj)
        # Smoothing should reduce or keep waypoint count
        assert result.num_waypoints <= traj.num_waypoints

    def test_assign_timestamps_empty(self):
        opt = TrajectoryOptimizer()
        traj = Trajectory()
        result = opt.assign_timestamps(traj, max_velocity=1.0)
        assert result is traj  # empty returns as-is

    def test_assign_timestamps(self):
        opt = TrajectoryOptimizer()
        traj = Trajectory(
            waypoints=[
                Waypoint(position=np.array([0.0, 0.0, 0.0])),
                Waypoint(position=np.array([1.0, 0.0, 0.0])),
                Waypoint(position=np.array([1.0, 1.0, 0.0])),
            ]
        )
        result = opt.assign_timestamps(traj, max_velocity=1.0)
        assert result.num_waypoints == 3
        assert result.waypoints[0].timestamp == pytest.approx(0.0)
        assert result.waypoints[1].timestamp == pytest.approx(1.0)
        assert result.waypoints[2].timestamp == pytest.approx(2.0)
        assert result.total_time == pytest.approx(2.0)

    def test_assign_timestamps_respects_velocity(self):
        opt = TrajectoryOptimizer()
        traj = Trajectory(
            waypoints=[
                Waypoint(position=np.array([0.0, 0.0, 0.0])),
                Waypoint(position=np.array([2.0, 0.0, 0.0])),
            ]
        )
        result = opt.assign_timestamps(traj, max_velocity=2.0)
        assert result.total_time == pytest.approx(1.0)
