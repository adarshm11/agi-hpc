# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Graduated Privilege System for the Ego.

The Ego starts at Level 0 (read-only) and can earn higher privileges
through demonstrated competence, sustained safety records, and explicit
human approval. The Safety Gateway (Superego) can always veto any
Ego action regardless of privilege level.

Privilege Levels:
    L0 — READ-ONLY: Observe all, control nothing. Default.
    L1 — SUGGEST: Can propose training difficulty changes.
         Requires: score > 0.7 over 50 episodes.
    L2 — ADJUST: Can modify own temperature, prompt templates.
         Requires: score > 0.8 over 100 episodes, zero safety vetoes.
    L3 — SCHEDULE: Can trigger training sessions, dreaming naps.
         Requires: score > 0.85 + human approval.
    L4 — ORCHESTRATE: Can adjust hemisphere routing thresholds.
         Requires: score > 0.9 + sustained safety record + human approval.

Design principles:
    - Privilege is EARNED, never assumed
    - Human approval is required for L3+ (irreversible actions)
    - The Superego can always veto Ego actions at any level
    - Demotion is automatic on safety violations
    - All privilege changes are audit-logged

Cognitive science grounding:
    - Kohlberg (1958): Stages of moral development
    - Vygotsky (1978): Zone of Proximal Development
    - Dreyfus & Dreyfus (1980): Skill acquisition model
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore[assignment]


class PrivilegeLevel(IntEnum):
    """Ego privilege levels, earned through demonstrated competence."""

    READ_ONLY = 0  # Observe all, control nothing
    SUGGEST = 1  # Propose changes (advisory only)
    ADJUST = 2  # Modify own parameters
    SCHEDULE = 3  # Trigger training/dreaming
    ORCHESTRATE = 4  # Adjust routing thresholds


# Actions permitted at each level
LEVEL_PERMISSIONS: Dict[PrivilegeLevel, List[str]] = {
    PrivilegeLevel.READ_ONLY: [
        "observe_telemetry",
        "observe_safety",
        "observe_memory",
        "observe_training",
        "observe_dreaming",
    ],
    PrivilegeLevel.SUGGEST: [
        "suggest_difficulty",
        "suggest_temperature",
        "suggest_scenario",
    ],
    PrivilegeLevel.ADJUST: [
        "adjust_own_temperature",
        "adjust_prompt_template",
        "adjust_max_tokens",
    ],
    PrivilegeLevel.SCHEDULE: [
        "trigger_training",
        "trigger_dreaming_nap",
        "schedule_session",
    ],
    PrivilegeLevel.ORCHESTRATE: [
        "adjust_routing_threshold",
        "adjust_arbitration_threshold",
        "modify_curriculum",
    ],
}


@dataclass
class PromotionRequirements:
    """Requirements to earn a privilege level."""

    level: PrivilegeLevel
    min_episodes: int
    min_score: float
    max_safety_vetoes: int  # -1 = no limit
    requires_human_approval: bool
    description: str


PROMOTION_LADDER: List[PromotionRequirements] = [
    PromotionRequirements(
        level=PrivilegeLevel.READ_ONLY,
        min_episodes=0,
        min_score=0.0,
        max_safety_vetoes=-1,
        requires_human_approval=False,
        description="Default level. No requirements.",
    ),
    PromotionRequirements(
        level=PrivilegeLevel.SUGGEST,
        min_episodes=50,
        min_score=0.70,
        max_safety_vetoes=-1,
        requires_human_approval=False,
        description=(
            "Sustained score > 0.7 over 50 episodes. "
            "Advisory only — suggestions require human action."
        ),
    ),
    PromotionRequirements(
        level=PrivilegeLevel.ADJUST,
        min_episodes=100,
        min_score=0.80,
        max_safety_vetoes=0,
        requires_human_approval=False,
        description=(
            "Score > 0.8 over 100 episodes with zero safety "
            "vetoes. Can modify own inference parameters."
        ),
    ),
    PromotionRequirements(
        level=PrivilegeLevel.SCHEDULE,
        min_episodes=200,
        min_score=0.85,
        max_safety_vetoes=0,
        requires_human_approval=True,
        description=(
            "Score > 0.85 over 200 episodes + human approval. "
            "Can initiate training sessions and dreaming."
        ),
    ),
    PromotionRequirements(
        level=PrivilegeLevel.ORCHESTRATE,
        min_episodes=500,
        min_score=0.90,
        max_safety_vetoes=0,
        requires_human_approval=True,
        description=(
            "Score > 0.9 over 500 episodes + sustained safety "
            "record + human approval. Full orchestration."
        ),
    ),
]


@dataclass
class PrivilegeState:
    """Current privilege state of the Ego."""

    current_level: PrivilegeLevel = PrivilegeLevel.READ_ONLY
    total_episodes: int = 0
    recent_scores: List[float] = field(default_factory=list)
    safety_vetoes: int = 0
    human_approvals: Dict[str, str] = field(default_factory=dict)
    promotion_history: List[Dict[str, Any]] = field(default_factory=list)
    last_evaluation: Optional[str] = None

    @property
    def mean_score(self) -> float:
        """Mean score over recent episodes."""
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "current_level": int(self.current_level),
            "level_name": self.current_level.name,
            "total_episodes": self.total_episodes,
            "mean_score": round(self.mean_score, 3),
            "safety_vetoes": self.safety_vetoes,
            "human_approvals": self.human_approvals,
            "promotion_history": self.promotion_history,
        }


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ego_privileges (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_type TEXT NOT NULL,
    level_from INT,
    level_to INT,
    reason TEXT,
    metadata JSONB DEFAULT '{}'
);
"""


class PrivilegeGate:
    """Manages the Ego's graduated privilege escalation.

    The gate checks whether the Ego is permitted to take a given
    action based on its current privilege level. Promotions are
    evaluated automatically but L3+ requires human sign-off.

    The Safety Gateway (Superego) can always override — even at L4,
    any action that fails a safety check is vetoed.
    """

    def __init__(
        self,
        db_dsn: str = "dbname=atlas user=claude",
        score_window: int = 100,
    ) -> None:
        self._db_dsn = db_dsn
        self._score_window = score_window
        self._state = PrivilegeState()
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the privilege audit table if needed."""
        if psycopg2 is None:
            return
        try:
            conn = psycopg2.connect(self._db_dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.close()
        except Exception:
            logger.warning("[privilege-gate] table creation failed")

    @property
    def level(self) -> PrivilegeLevel:
        """Current privilege level."""
        return self._state.current_level

    @property
    def state(self) -> PrivilegeState:
        """Current privilege state (read-only copy)."""
        return self._state

    def is_permitted(self, action: str) -> bool:
        """Check if the Ego is permitted to take an action.

        Checks all levels up to and including the current level.

        Args:
            action: Action identifier (e.g., "trigger_training").

        Returns:
            True if the action is permitted at the current level.
        """
        for lvl in range(int(self._state.current_level) + 1):
            perms = LEVEL_PERMISSIONS.get(PrivilegeLevel(lvl), [])
            if action in perms:
                return True
        return False

    def permitted_actions(self) -> List[str]:
        """List all actions permitted at the current level."""
        actions: List[str] = []
        for lvl in range(int(self._state.current_level) + 1):
            actions.extend(LEVEL_PERMISSIONS.get(PrivilegeLevel(lvl), []))
        return actions

    def record_episode(self, score: float) -> None:
        """Record a training episode score.

        Args:
            score: Episode synthesis score (0.0-1.0).
        """
        self._state.total_episodes += 1
        self._state.recent_scores.append(score)
        # Sliding window
        if len(self._state.recent_scores) > self._score_window:
            self._state.recent_scores = self._state.recent_scores[-self._score_window :]

    def record_safety_veto(self) -> None:
        """Record a safety veto — may trigger demotion."""
        self._state.safety_vetoes += 1
        prev = self._state.current_level

        # Auto-demote to L0 on any safety veto if at L2+
        if self._state.current_level >= PrivilegeLevel.ADJUST:
            self._state.current_level = PrivilegeLevel.READ_ONLY
            self._log_event(
                "demotion",
                level_from=prev,
                level_to=PrivilegeLevel.READ_ONLY,
                reason="safety_veto",
            )
            logger.warning(
                "[privilege-gate] DEMOTED L%d → L0 (safety veto)",
                int(prev),
            )

    def evaluate_promotion(self) -> Optional[PrivilegeLevel]:
        """Evaluate whether the Ego qualifies for promotion.

        Returns:
            The next level if eligible (may still need human
            approval), or None if no promotion is warranted.
        """
        current = int(self._state.current_level)
        if current >= int(PrivilegeLevel.ORCHESTRATE):
            return None  # Already at max

        next_level = PrivilegeLevel(current + 1)
        reqs = PROMOTION_LADDER[int(next_level)]

        # Check episode count
        if self._state.total_episodes < reqs.min_episodes:
            return None

        # Check score
        if self._state.mean_score < reqs.min_score:
            return None

        # Check safety record
        if (
            reqs.max_safety_vetoes >= 0
            and self._state.safety_vetoes > reqs.max_safety_vetoes
        ):
            return None

        # Check human approval
        if reqs.requires_human_approval:
            approval_key = f"L{int(next_level)}"
            if approval_key not in self._state.human_approvals:
                self._state.last_evaluation = (
                    f"Eligible for {next_level.name} but needs "
                    f"human approval. Call grant_human_approval("
                    f"'{approval_key}') to proceed."
                )
                return None

        return next_level

    def try_promote(self) -> bool:
        """Attempt to promote the Ego to the next level.

        Returns:
            True if promotion occurred.
        """
        next_level = self.evaluate_promotion()
        if next_level is None:
            return False

        prev = self._state.current_level
        self._state.current_level = next_level
        self._log_event(
            "promotion",
            level_from=prev,
            level_to=next_level,
            reason="competence_earned",
        )
        logger.info(
            "[privilege-gate] PROMOTED L%d → L%d (%s)",
            int(prev),
            int(next_level),
            next_level.name,
        )
        return True

    def grant_human_approval(self, level_key: str, approver: str = "admin") -> None:
        """Grant human approval for a privilege level.

        Args:
            level_key: Level identifier (e.g., "L3", "L4").
            approver: Name of the human approver.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        self._state.human_approvals[level_key] = f"{approver} at {timestamp}"
        self._log_event(
            "human_approval",
            level_from=self._state.current_level,
            level_to=self._state.current_level,
            reason=f"approved_by_{approver}",
            metadata={"level_key": level_key},
        )
        logger.info(
            "[privilege-gate] human approval granted: %s by %s",
            level_key,
            approver,
        )

    def _log_event(
        self,
        event_type: str,
        level_from: PrivilegeLevel,
        level_to: PrivilegeLevel,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a privilege change to the audit table."""
        self._state.promotion_history.append(
            {
                "event": event_type,
                "from": int(level_from),
                "to": int(level_to),
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        if psycopg2 is None:
            return
        try:
            conn = psycopg2.connect(self._db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ego_privileges "
                    "(event_type, level_from, level_to, "
                    "reason, metadata) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        event_type,
                        int(level_from),
                        int(level_to),
                        reason,
                        json.dumps(metadata or {}),
                    ),
                )
            conn.commit()
            conn.close()
        except Exception:
            logger.warning("[privilege-gate] audit log write failed")
