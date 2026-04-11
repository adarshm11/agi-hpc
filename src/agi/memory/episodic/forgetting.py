# AGI-HPC Ebbinghaus Forgetting Curves
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
"""Time-based memory decay for episodic memory.

Episodes lose retrieval weight over time unless reinforced
by recall or high quality scores. Based on Ebbinghaus (1885):
  retention = e^(-t/S)
where S = stability, which increases with each recall.

Source: The Synthetic Mind, Ch 4 — Memory That Forgets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ForgettingConfig:
    """Parameters for the forgetting curve."""

    base_half_life_hours: float = 24.0
    quality_multiplier: float = 2.0
    recall_boost: float = 1.5
    prune_threshold: float = 0.1


def compute_stability(
    quality_score: float,
    recall_count: int,
    config: ForgettingConfig,
) -> float:
    """Compute stability S for an episode.

    S = base_half_life * (1 + quality * quality_multiplier) * recall_boost^recalls
    Higher quality and more recalls = slower forgetting.
    """
    base = config.base_half_life_hours
    quality_factor = 1.0 + quality_score * config.quality_multiplier
    recall_factor = config.recall_boost**recall_count
    return base * quality_factor * recall_factor


def compute_retention(
    age_hours: float,
    stability: float,
) -> float:
    """Compute retention using Ebbinghaus curve.

    retention = e^(-t/S) where t = age in hours, S = stability.
    """
    if stability <= 0:
        return 0.0
    return math.exp(-age_hours / stability)


def episode_retention(
    episode_timestamp: datetime,
    quality_score: float,
    metadata: dict[str, Any],
    now: datetime,
    config: ForgettingConfig,
) -> float:
    """Compute current retention for an episode."""
    age = (now - episode_timestamp).total_seconds() / 3600.0
    recall_count = metadata.get("recall_count", 0)
    stability = compute_stability(quality_score, recall_count, config)
    return compute_retention(age, stability)


def is_pruneable(
    episode_timestamp: datetime,
    quality_score: float,
    metadata: dict[str, Any],
    now: datetime,
    config: ForgettingConfig,
) -> bool:
    """Check if an episode has decayed below the prune threshold."""
    retention = episode_retention(
        episode_timestamp, quality_score, metadata, now, config
    )
    return retention < config.prune_threshold


def boost_recall(metadata: dict[str, Any]) -> dict[str, Any]:
    """Boost stability when an episode is recalled.

    Called when the episodic memory system retrieves this episode
    in response to a query. Increases recall_count in metadata.
    """
    updated = dict(metadata)
    updated["recall_count"] = updated.get("recall_count", 0) + 1
    return updated
