# AGI-HPC Temporal Cognition
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
"""Temporal awareness: boredom detection, patience, and pacing.

Source: The Synthetic Mind, Ch 12 — Temporal Cognition.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class TemporalState:
    """Current temporal awareness state."""

    boredom_score: float = 0.0
    patience_factor: float = 1.0
    queries_per_minute: float = 0.0
    should_pace: bool = False


class TemporalCognition:
    """Track temporal patterns in user interaction."""

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._recent_queries: deque[str] = deque(maxlen=window_size)
        self._query_times: deque[float] = deque(maxlen=window_size)
        self._state = TemporalState()

    def observe_query(self, query: str) -> TemporalState:
        """Record a new query and update temporal state."""
        now = time.time()
        self._recent_queries.append(query.lower().strip())
        self._query_times.append(now)

        self._state.boredom_score = self._compute_boredom()
        self._state.queries_per_minute = self._compute_rate()
        self._state.should_pace = self._state.queries_per_minute > 10
        return self._state

    def get_patience(self, complexity: int) -> float:
        """Return patience factor based on query complexity.

        Higher complexity = more patience (take more time to think).
        Range: 1.0 (simple) to 3.0 (very complex).
        """
        self._state.patience_factor = 1.0 + min(complexity, 5) * 0.4
        return self._state.patience_factor

    @property
    def state(self) -> TemporalState:
        return self._state

    def _compute_boredom(self) -> float:
        """Boredom increases when recent queries are too similar."""
        if len(self._recent_queries) < 2:
            return 0.0

        queries = list(self._recent_queries)
        similarities = []
        latest_words = set(queries[-1].split())
        if not latest_words:
            return 0.0

        for prev in queries[:-1]:
            prev_words = set(prev.split())
            if not prev_words:
                continue
            overlap = len(latest_words & prev_words)
            union = len(latest_words | prev_words)
            if union > 0:
                similarities.append(overlap / union)

        if not similarities:
            return 0.0

        avg_sim = sum(similarities) / len(similarities)
        return min(avg_sim, 1.0)

    def _compute_rate(self) -> float:
        """Queries per minute based on recent timestamps."""
        if len(self._query_times) < 2:
            return 0.0
        span = self._query_times[-1] - self._query_times[0]
        if span <= 0:
            return 0.0
        return (len(self._query_times) - 1) / (span / 60.0)
