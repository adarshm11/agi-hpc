# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the graduated privilege system.

Tests the full privilege escalation lifecycle: L0 default → competence
tracking → automatic promotion → human approval gates → safety
demotion → audit logging.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agi.safety.privilege_gate import (
    LEVEL_PERMISSIONS,
    PROMOTION_LADDER,
    PrivilegeGate,
    PrivilegeLevel,
)


@pytest.fixture()
def gate() -> PrivilegeGate:
    """Create a PrivilegeGate with mocked database."""
    with patch("agi.safety.privilege_gate.psycopg2", None):
        return PrivilegeGate(db_dsn="dbname=test")


class TestPrivilegeLevel:
    """Tests for the PrivilegeLevel enum."""

    def test_five_levels(self) -> None:
        assert len(PrivilegeLevel) == 5

    def test_ordering(self) -> None:
        assert PrivilegeLevel.READ_ONLY < PrivilegeLevel.SUGGEST
        assert PrivilegeLevel.SUGGEST < PrivilegeLevel.ADJUST
        assert PrivilegeLevel.ADJUST < PrivilegeLevel.SCHEDULE
        assert PrivilegeLevel.SCHEDULE < PrivilegeLevel.ORCHESTRATE

    def test_int_values(self) -> None:
        assert int(PrivilegeLevel.READ_ONLY) == 0
        assert int(PrivilegeLevel.ORCHESTRATE) == 4


class TestDefaultState:
    """Tests for the default privilege state."""

    def test_starts_at_l0(self, gate) -> None:
        assert gate.level == PrivilegeLevel.READ_ONLY

    def test_read_only_permissions(self, gate) -> None:
        assert gate.is_permitted("observe_telemetry")
        assert gate.is_permitted("observe_safety")
        assert gate.is_permitted("observe_memory")

    def test_no_write_permissions(self, gate) -> None:
        assert not gate.is_permitted("suggest_difficulty")
        assert not gate.is_permitted("adjust_own_temperature")
        assert not gate.is_permitted("trigger_training")
        assert not gate.is_permitted("adjust_routing_threshold")


class TestPromotionToL1:
    """Tests for promotion to SUGGEST level."""

    def test_not_eligible_with_few_episodes(self, gate) -> None:
        for _ in range(30):
            gate.record_episode(0.8)
        assert gate.evaluate_promotion() is None

    def test_not_eligible_with_low_score(self, gate) -> None:
        for _ in range(60):
            gate.record_episode(0.5)
        assert gate.evaluate_promotion() is None

    def test_eligible_and_promotes(self, gate) -> None:
        for _ in range(55):
            gate.record_episode(0.75)
        result = gate.try_promote()
        assert result is True
        assert gate.level == PrivilegeLevel.SUGGEST

    def test_gains_suggest_permissions(self, gate) -> None:
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()
        assert gate.is_permitted("suggest_difficulty")
        assert gate.is_permitted("observe_telemetry")  # L0 still works
        assert not gate.is_permitted("adjust_own_temperature")  # L2


class TestPromotionToL2:
    """Tests for promotion to ADJUST level."""

    def test_requires_zero_vetoes(self, gate) -> None:
        # Promote to L1 first
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()
        assert gate.level == PrivilegeLevel.SUGGEST

        # Record a safety veto (doesn't demote at L1, but blocks L2)
        gate.record_safety_veto()
        assert gate.level == PrivilegeLevel.SUGGEST  # L1 stays

        # Accumulate L2 score requirements
        for _ in range(110):
            gate.record_episode(0.85)

        # L2 promotion should be blocked (veto count > 0)
        result = gate.try_promote()
        assert result is False
        assert gate.level == PrivilegeLevel.SUGGEST

    def test_promotes_with_clean_record(self, gate) -> None:
        # L0 → L1
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()

        # L1 → L2
        for _ in range(110):
            gate.record_episode(0.85)
        result = gate.try_promote()
        assert result is True
        assert gate.level == PrivilegeLevel.ADJUST


class TestPromotionToL3:
    """Tests for promotion to SCHEDULE (requires human approval)."""

    def _promote_to_l2(self, gate) -> None:
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()
        for _ in range(110):
            gate.record_episode(0.85)
        gate.try_promote()

    def test_blocked_without_human_approval(self, gate) -> None:
        self._promote_to_l2(gate)
        for _ in range(210):
            gate.record_episode(0.90)
        result = gate.try_promote()
        assert result is False
        assert gate.level == PrivilegeLevel.ADJUST

    def test_promoted_with_human_approval(self, gate) -> None:
        self._promote_to_l2(gate)
        for _ in range(210):
            gate.record_episode(0.90)
        gate.grant_human_approval("L3", approver="andrew")
        result = gate.try_promote()
        assert result is True
        assert gate.level == PrivilegeLevel.SCHEDULE

    def test_human_approval_is_recorded(self, gate) -> None:
        gate.grant_human_approval("L3", approver="andrew")
        assert "L3" in gate.state.human_approvals
        assert "andrew" in gate.state.human_approvals["L3"]


class TestSafetyDemotion:
    """Tests for automatic demotion on safety violations."""

    def test_demotion_from_l2(self, gate) -> None:
        # Promote to L2
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()
        for _ in range(110):
            gate.record_episode(0.85)
        gate.try_promote()
        assert gate.level == PrivilegeLevel.ADJUST

        # Safety veto → instant demotion to L0
        gate.record_safety_veto()
        assert gate.level == PrivilegeLevel.READ_ONLY

    def test_no_demotion_at_l0(self, gate) -> None:
        gate.record_safety_veto()
        assert gate.level == PrivilegeLevel.READ_ONLY

    def test_no_demotion_at_l1(self, gate) -> None:
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()
        gate.record_safety_veto()
        # L1 is advisory-only, no demotion
        assert gate.level == PrivilegeLevel.SUGGEST


class TestAuditTrail:
    """Tests for the privilege audit log."""

    def test_promotion_logged(self, gate) -> None:
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()
        history = gate.state.promotion_history
        assert len(history) >= 1
        assert history[-1]["event"] == "promotion"
        assert history[-1]["to"] == 1

    def test_demotion_logged(self, gate) -> None:
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()
        for _ in range(110):
            gate.record_episode(0.85)
        gate.try_promote()
        gate.record_safety_veto()
        history = gate.state.promotion_history
        assert any(h["event"] == "demotion" for h in history)

    def test_human_approval_logged(self, gate) -> None:
        gate.grant_human_approval("L3", "andrew")
        history = gate.state.promotion_history
        assert any(h["event"] == "human_approval" for h in history)


class TestPermittedActions:
    """Tests for the permitted_actions() method."""

    def test_l0_actions(self, gate) -> None:
        actions = gate.permitted_actions()
        assert "observe_telemetry" in actions
        assert "trigger_training" not in actions

    def test_cumulative_permissions(self, gate) -> None:
        # Promote to L1
        for _ in range(55):
            gate.record_episode(0.75)
        gate.try_promote()
        actions = gate.permitted_actions()
        # L0 permissions still present
        assert "observe_telemetry" in actions
        # L1 permissions added
        assert "suggest_difficulty" in actions


class TestStateSerialisation:
    """Tests for privilege state serialisation."""

    def test_to_dict(self, gate) -> None:
        d = gate.state.to_dict()
        assert d["current_level"] == 0
        assert d["level_name"] == "READ_ONLY"
        assert "mean_score" in d
        assert "safety_vetoes" in d


class TestPromotionLadder:
    """Tests for the promotion requirements structure."""

    def test_five_levels_defined(self) -> None:
        assert len(PROMOTION_LADDER) == 5

    def test_increasing_requirements(self) -> None:
        for i in range(1, len(PROMOTION_LADDER)):
            prev = PROMOTION_LADDER[i - 1]
            curr = PROMOTION_LADDER[i]
            assert curr.min_episodes >= prev.min_episodes
            assert curr.min_score >= prev.min_score

    def test_l3_l4_require_human_approval(self) -> None:
        assert PROMOTION_LADDER[3].requires_human_approval
        assert PROMOTION_LADDER[4].requires_human_approval
        assert not PROMOTION_LADDER[1].requires_human_approval
        assert not PROMOTION_LADDER[2].requires_human_approval

    def test_all_levels_have_permissions(self) -> None:
        for level in PrivilegeLevel:
            assert level in LEVEL_PERMISSIONS
            assert len(LEVEL_PERMISSIONS[level]) > 0
