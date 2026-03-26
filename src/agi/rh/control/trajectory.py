# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Trajectory Planning and Optimization for AGI-HPC.

Provides path planning and trajectory optimization for the RH control
subsystem:
- RRT/RRT* sampling-based path planning
- CHOMP covariant Hamiltonian optimization (stub)
- Trajectory smoothing and optimization

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory planners and optimizers."""

    max_iterations: int = 5000
    step_size: float = 0.1
    goal_bias: float = 0.05
    goal_tolerance: float = 0.05
    smoothing_iterations: int = 200
    collision_check_resolution: float = 0.01
    workspace_min: Tuple[float, ...] = (-1.0, -1.0, -1.0)
    workspace_max: Tuple[float, ...] = (1.0, 1.0, 1.0)


@dataclass
class Waypoint:
    """A single point along a trajectory."""

    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """An ordered sequence of waypoints forming a motion trajectory."""

    waypoints: List[Waypoint] = field(default_factory=list)
    total_time: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        """Compute the total Euclidean path length."""
        if len(self.waypoints) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.waypoints)):
            total += float(
                np.linalg.norm(
                    self.waypoints[i].position - self.waypoints[i - 1].position
                )
            )
        return total

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)


class _RRTNode:
    __slots__ = ("position", "parent", "cost")

    def __init__(
        self,
        position: np.ndarray,
        parent: Optional["_RRTNode"] = None,
        cost: float = 0.0,
    ) -> None:
        self.position = position
        self.parent = parent
        self.cost = cost


class RRTPlanner:
    """Rapidly-exploring Random Tree (RRT / RRT*) path planner."""

    def __init__(
        self, config: Optional[TrajectoryConfig] = None, use_rrt_star: bool = False
    ) -> None:
        self._config = config or TrajectoryConfig()
        self._use_rrt_star = use_rrt_star
        logger.info(
            "[rh][trajectory] RRTPlanner initialized rrt_star=%s max_iter=%d",
            self._use_rrt_star,
            self._config.max_iterations,
        )

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        collision_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> Optional[Trajectory]:
        """Plan a collision-free path from start to goal."""
        cfg = self._config
        dim = len(start)
        ws_min = np.array(cfg.workspace_min[:dim])
        ws_max = np.array(cfg.workspace_max[:dim])
        root = _RRTNode(position=np.array(start, dtype=np.float64))
        nodes: List[_RRTNode] = [root]

        def _is_free(pos: np.ndarray) -> bool:
            return collision_fn is None or not collision_fn(pos)

        for iteration in range(cfg.max_iterations):
            if random.random() < cfg.goal_bias:
                sample = np.array(goal, dtype=np.float64)
            else:
                sample = np.random.uniform(ws_min, ws_max)
            nearest = min(
                nodes, key=lambda n: float(np.linalg.norm(n.position - sample))
            )
            direction = sample - nearest.position
            dist = float(np.linalg.norm(direction))
            if dist < 1e-12:
                continue
            direction = direction / dist
            step = min(cfg.step_size, dist)
            new_pos = nearest.position + direction * step
            if not _is_free(new_pos):
                continue
            if not self._edge_collision_free(
                nearest.position, new_pos, collision_fn, cfg
            ):
                continue
            new_cost = nearest.cost + float(np.linalg.norm(new_pos - nearest.position))
            new_node = _RRTNode(position=new_pos, parent=nearest, cost=new_cost)
            if self._use_rrt_star:
                self._rewire(nodes, new_node, collision_fn, cfg)
            nodes.append(new_node)
            if float(np.linalg.norm(new_pos - goal)) < cfg.goal_tolerance:
                logger.info(
                    "[rh][trajectory] RRT found path in %d iterations", iteration + 1
                )
                return self._extract_trajectory(new_node, goal)
        logger.warning(
            "[rh][trajectory] RRT failed to find path in %d iterations",
            cfg.max_iterations,
        )
        return None

    @staticmethod
    def _edge_collision_free(
        from_pos: np.ndarray,
        to_pos: np.ndarray,
        collision_fn: Optional[Callable[[np.ndarray], bool]],
        cfg: TrajectoryConfig,
    ) -> bool:
        if collision_fn is None:
            return True
        dist = float(np.linalg.norm(to_pos - from_pos))
        steps = max(2, int(math.ceil(dist / cfg.collision_check_resolution)))
        for i in range(1, steps):
            t = i / steps
            point = from_pos + t * (to_pos - from_pos)
            if collision_fn(point):
                return False
        return True

    @staticmethod
    def _rewire(
        nodes: List[_RRTNode],
        new_node: _RRTNode,
        collision_fn: Optional[Callable[[np.ndarray], bool]],
        cfg: TrajectoryConfig,
    ) -> None:
        radius = cfg.step_size * 2.0
        for node in nodes:
            dist = float(np.linalg.norm(node.position - new_node.position))
            if dist > radius or dist < 1e-12:
                continue
            candidate_cost = new_node.cost + dist
            if candidate_cost < node.cost:
                if RRTPlanner._edge_collision_free(
                    new_node.position, node.position, collision_fn, cfg
                ):
                    node.parent = new_node
                    node.cost = candidate_cost

    @staticmethod
    def _extract_trajectory(node: _RRTNode, goal: np.ndarray) -> Trajectory:
        path: List[np.ndarray] = []
        current: Optional[_RRTNode] = node
        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        waypoints = [Waypoint(position=p) for p in path]
        return Trajectory(waypoints=waypoints, cost=node.cost)


class CHOMPPlanner:
    """Covariant Hamiltonian Optimization for Motion Planning (stub)."""

    def __init__(self, config: Optional[TrajectoryConfig] = None) -> None:
        self._config = config or TrajectoryConfig()
        logger.info("[rh][trajectory] CHOMPPlanner initialized (stub)")

    def optimize(
        self,
        initial_trajectory: Trajectory,
        collision_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> Trajectory:
        """Optimize a trajectory using CHOMP. Currently returns unchanged."""
        logger.warning("[rh][trajectory] CHOMPPlanner.optimize is a stub")
        return initial_trajectory


class TrajectoryOptimizer:
    """Smooth and optimize trajectories via shortcut smoothing."""

    def __init__(self, config: Optional[TrajectoryConfig] = None) -> None:
        self._config = config or TrajectoryConfig()
        logger.info(
            "[rh][trajectory] TrajectoryOptimizer initialized smoothing_iters=%d",
            self._config.smoothing_iterations,
        )

    def smooth(
        self,
        trajectory: Trajectory,
        collision_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> Trajectory:
        """Smooth a trajectory using random shortcutting."""
        if trajectory.num_waypoints < 3:
            return trajectory
        cfg = self._config
        positions = [wp.position.copy() for wp in trajectory.waypoints]
        improved = 0
        for _ in range(cfg.smoothing_iterations):
            if len(positions) < 3:
                break
            i = random.randint(0, len(positions) - 2)
            j = random.randint(i + 1, len(positions) - 1)
            if j - i <= 1:
                continue
            if RRTPlanner._edge_collision_free(
                positions[i], positions[j], collision_fn, cfg
            ):
                positions = positions[: i + 1] + positions[j:]
                improved += 1
        waypoints = [Waypoint(position=p) for p in positions]
        result = Trajectory(waypoints=waypoints)
        result.cost = result.length
        logger.info(
            "[rh][trajectory] smoothing complete: %d -> %d waypoints (%d shortcuts)",
            trajectory.num_waypoints,
            result.num_waypoints,
            improved,
        )
        return result

    def assign_timestamps(
        self, trajectory: Trajectory, max_velocity: float = 1.0
    ) -> Trajectory:
        """Assign timestamps to waypoints based on a maximum velocity."""
        if trajectory.num_waypoints == 0:
            return trajectory
        t = 0.0
        waypoints: List[Waypoint] = []
        for i, wp in enumerate(trajectory.waypoints):
            if i > 0:
                dist = float(
                    np.linalg.norm(wp.position - trajectory.waypoints[i - 1].position)
                )
                t += dist / max(max_velocity, 1e-6)
            waypoints.append(
                Waypoint(
                    position=wp.position.copy(),
                    velocity=wp.velocity,
                    timestamp=t,
                    metadata=dict(wp.metadata),
                )
            )
        result = Trajectory(
            waypoints=waypoints,
            total_time=t,
            cost=trajectory.cost,
            metadata=dict(trajectory.metadata),
        )
        logger.debug("[rh][trajectory] assigned timestamps total_time=%.3fs", t)
        return result
