# AGI-HPC Cross-Subsystem Failure Cascades
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
"""Detect and propagate subsystem failures through dependency graph.

When one subsystem degrades, dependent subsystems are marked as
degraded rather than letting errors propagate silently.

Source: The Synthetic Mind, Ch 13 — Integration & Failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def _cascade_label(h: SubsystemHealth) -> str:
    if h.direct_failure:
        return "direct"
    return f"from:{h.cascaded_from}"


# Which subsystems depend on which others
DEFAULT_DEPENDENCY_GRAPH: dict[str, list[str]] = {
    "semantic_memory": [],
    "episodic_memory": [],
    "procedural_memory": [],
    "knowledge_graph": ["semantic_memory"],
    "research_loop": ["knowledge_graph", "semantic_memory"],
    "executive_function": ["semantic_memory", "episodic_memory"],
    "safety_gateway": [],
    "lh": ["safety_gateway"],
    "rh": ["safety_gateway"],
    "ego": ["lh", "rh"],
    "temporal": [],
}


@dataclass
class SubsystemHealth:
    """Health status of one subsystem."""

    name: str
    status: str  # healthy, degraded, failed
    direct_failure: bool = False
    cascaded_from: Optional[str] = None


class CascadeDetector:
    """Detect failure cascades through a dependency graph."""

    def __init__(self, graph: Optional[dict[str, list[str]]] = None) -> None:
        self._graph = graph or DEFAULT_DEPENDENCY_GRAPH
        self._last_health: list[SubsystemHealth] = []

    def evaluate(self, health: dict[str, str]) -> list[SubsystemHealth]:
        """Propagate failures through dependency graph.

        Args:
            health: map of subsystem name to status
                    ("healthy", "degraded", "failed")

        Returns:
            Full health state with cascade information.
        """
        result: dict[str, SubsystemHealth] = {}

        # First pass: record direct statuses
        for name in self._graph:
            status = health.get(name, "healthy")
            direct = status != "healthy"
            result[name] = SubsystemHealth(
                name=name,
                status=status,
                direct_failure=direct,
            )

        # Second pass: propagate failures through dependencies
        changed = True
        iterations = 0
        while changed and iterations < 20:
            changed = False
            iterations += 1
            for name, deps in self._graph.items():
                if result[name].status != "healthy":
                    continue
                for dep in deps:
                    dep_health = result.get(dep)
                    if dep_health and dep_health.status in (
                        "degraded",
                        "failed",
                    ):
                        result[name] = SubsystemHealth(
                            name=name,
                            status="degraded",
                            direct_failure=False,
                            cascaded_from=dep,
                        )
                        changed = True
                        break

        self._last_health = list(result.values())

        degraded = [h for h in self._last_health if h.status != "healthy"]
        if degraded:
            logger.warning(
                "Cascade: %d subsystems affected: %s",
                len(degraded),
                ", ".join(f"{h.name}({_cascade_label(h)})" for h in degraded),
            )

        return self._last_health

    def get_degraded(self) -> list[str]:
        """Return names of currently degraded subsystems."""
        return [h.name for h in self._last_health if h.status != "healthy"]

    def get_cascade_chains(self) -> dict[str, list[str]]:
        """Return which direct failures caused which cascades."""
        chains: dict[str, list[str]] = {}
        for h in self._last_health:
            if h.cascaded_from and not h.direct_failure:
                chains.setdefault(h.cascaded_from, []).append(h.name)
        return chains
