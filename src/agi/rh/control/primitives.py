# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Motor Primitives for RH Advanced Control.

Provides a protocol-based system for defining and executing
motor primitives (reach, grasp, place, navigate):
- MotorPrimitive protocol for primitive interface
- ExecutionContext for runtime state
- PrimitiveResult for outcomes
- Concrete primitives: Reach, Grasp, Place, Navigation
- PrimitiveLibrary for registration and lookup

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class ExecutionContext:
    """Runtime context for primitive execution."""

    robot_state: Dict[str, Any] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    step_count: int = 0

    def get_position(self) -> np.ndarray:
        """Get current end-effector position."""
        return np.array(
            self.robot_state.get("position", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )

    def get_gripper_state(self) -> float:
        """Get gripper opening (0=closed, 1=open)."""
        return float(self.robot_state.get("gripper", 0.0))


@dataclass
class PrimitiveResult:
    """Result from executing a motor primitive."""

    success: bool
    primitive_name: str
    duration: float = 0.0
    error: Optional[str] = None
    actions_generated: int = 0
    final_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Motor Primitive Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MotorPrimitive(Protocol):
    """Protocol for motor primitives."""

    @property
    def name(self) -> str:
        """Return the primitive name."""
        ...

    def can_execute(self, context: ExecutionContext) -> bool:
        """Check if the primitive can execute in the given context."""
        ...

    def generate_actions(
        self,
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        """Generate action sequence for this primitive."""
        ...

    def execute(self, context: ExecutionContext) -> PrimitiveResult:
        """Execute the primitive and return result."""
        ...


# ---------------------------------------------------------------------------
# Concrete Primitives
# ---------------------------------------------------------------------------


class ReachPrimitive:
    """Motor primitive for reaching a target position."""

    def __init__(
        self,
        speed: float = 0.5,
        tolerance: float = 0.01,
    ) -> None:
        self._speed = speed
        self._tolerance = tolerance

    @property
    def name(self) -> str:
        return "reach"

    def can_execute(self, context: ExecutionContext) -> bool:
        return "target" in context.parameters

    def generate_actions(
        self,
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        target = np.array(context.parameters.get("target", [0, 0, 0]), dtype=np.float64)
        current = context.get_position()
        direction = target - current
        distance = float(np.linalg.norm(direction))

        if distance < self._tolerance:
            return []

        _normalized = direction / distance if distance > 0 else direction  # noqa: F841
        steps = max(1, int(distance / (self._speed * 0.01)))

        actions = []
        for i in range(steps):
            t = (i + 1) / steps
            waypoint = current + direction * t
            actions.append(
                {
                    "type": "move",
                    "target": waypoint.tolist(),
                    "magnitude": self._speed,
                    "duration": 0.01,
                }
            )
        return actions

    def execute(self, context: ExecutionContext) -> PrimitiveResult:
        start = time.monotonic()
        actions = self.generate_actions(context)
        duration = time.monotonic() - start

        return PrimitiveResult(
            success=len(actions) > 0 or self._at_target(context),
            primitive_name=self.name,
            duration=duration,
            actions_generated=len(actions),
        )

    def _at_target(self, context: ExecutionContext) -> bool:
        target = np.array(context.parameters.get("target", [0, 0, 0]), dtype=np.float64)
        current = context.get_position()
        return float(np.linalg.norm(target - current)) < self._tolerance


class GraspPrimitive:
    """Motor primitive for grasping an object."""

    def __init__(self, force: float = 0.8) -> None:
        self._force = force

    @property
    def name(self) -> str:
        return "grasp"

    def can_execute(self, context: ExecutionContext) -> bool:
        return context.get_gripper_state() > 0.1  # Must be at least partially open

    def generate_actions(
        self,
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        return [
            {"type": "grasp", "force": self._force, "duration": 0.3},
            {"type": "hold", "force": self._force * 0.5, "duration": 0.1},
        ]

    def execute(self, context: ExecutionContext) -> PrimitiveResult:
        start = time.monotonic()
        actions = self.generate_actions(context)
        duration = time.monotonic() - start

        return PrimitiveResult(
            success=True,
            primitive_name=self.name,
            duration=duration,
            actions_generated=len(actions),
        )


class PlacePrimitive:
    """Motor primitive for placing an object."""

    def __init__(self, release_height: float = 0.05) -> None:
        self._release_height = release_height

    @property
    def name(self) -> str:
        return "place"

    def can_execute(self, context: ExecutionContext) -> bool:
        return "target" in context.parameters

    def generate_actions(
        self,
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        target = context.parameters.get("target", [0, 0, 0])
        approach = list(target)
        approach[2] += self._release_height

        return [
            {"type": "move", "target": approach, "magnitude": 0.3, "duration": 0.5},
            {"type": "move", "target": target, "magnitude": 0.1, "duration": 0.3},
            {"type": "release", "magnitude": 0.0, "duration": 0.2},
            {"type": "move", "target": approach, "magnitude": 0.3, "duration": 0.3},
        ]

    def execute(self, context: ExecutionContext) -> PrimitiveResult:
        start = time.monotonic()
        actions = self.generate_actions(context)
        duration = time.monotonic() - start

        return PrimitiveResult(
            success=True,
            primitive_name=self.name,
            duration=duration,
            actions_generated=len(actions),
        )


class NavigationPrimitive:
    """Motor primitive for mobile base navigation."""

    def __init__(self, max_speed: float = 1.0) -> None:
        self._max_speed = max_speed

    @property
    def name(self) -> str:
        return "navigate"

    def can_execute(self, context: ExecutionContext) -> bool:
        return "goal" in context.parameters

    def generate_actions(
        self,
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        goal = np.array(context.parameters.get("goal", [0, 0, 0]), dtype=np.float64)
        current = context.get_position()
        direction = goal - current
        distance = float(np.linalg.norm(direction))

        if distance < 0.1:
            return []

        duration = distance / self._max_speed
        return [
            {
                "type": "navigate",
                "goal": goal.tolist(),
                "speed": self._max_speed,
                "duration": duration,
            }
        ]

    def execute(self, context: ExecutionContext) -> PrimitiveResult:
        start = time.monotonic()
        actions = self.generate_actions(context)
        duration = time.monotonic() - start

        return PrimitiveResult(
            success=True,
            primitive_name=self.name,
            duration=duration,
            actions_generated=len(actions),
        )


# ---------------------------------------------------------------------------
# Primitive Library
# ---------------------------------------------------------------------------


class PrimitiveLibrary:
    """Registry and lookup for motor primitives."""

    def __init__(self) -> None:
        self._primitives: Dict[str, MotorPrimitive] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in primitives."""
        self.register(ReachPrimitive())
        self.register(GraspPrimitive())
        self.register(PlacePrimitive())
        self.register(NavigationPrimitive())

    def register(self, primitive: MotorPrimitive) -> None:
        """Register a motor primitive."""
        self._primitives[primitive.name] = primitive
        logger.debug("[RH][Primitives] registered: %s", primitive.name)

    def get(self, name: str) -> Optional[MotorPrimitive]:
        """Get a primitive by name."""
        return self._primitives.get(name)

    def list_primitives(self) -> List[str]:
        """List all registered primitive names."""
        return list(self._primitives.keys())

    def execute(
        self,
        name: str,
        context: ExecutionContext,
    ) -> PrimitiveResult:
        """Execute a named primitive."""
        primitive = self.get(name)
        if primitive is None:
            return PrimitiveResult(
                success=False,
                primitive_name=name,
                error=f"Unknown primitive: {name}",
            )

        if not primitive.can_execute(context):
            return PrimitiveResult(
                success=False,
                primitive_name=name,
                error="Preconditions not met",
            )

        return primitive.execute(context)
