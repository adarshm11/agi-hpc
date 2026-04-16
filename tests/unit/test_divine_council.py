# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the Divine Council static data and dataclasses.

The behavior-level tests that previously lived here — score parsing,
flag extraction, deliberation flow, tally logic — have been superseded
by :mod:`tests.reasoning.test_divine_council`, which exercises the
same paths against the post-2026-04-13 `CouncilBackend` transport
abstraction via ``_ScriptedBackend``. Tests in *this* module are kept
intentionally narrow: member definitions, model name, and pure-data
verdict / log formatting that doesn't need a running council.

Removed (now covered in tests/reasoning/):
    - TestMaxTokens           → covered by TestDeliberate in reasoning/
    - TestDeliberation        → TestDeliberate + TestTallyConsensusLogic
    - TestScoreParsing        → TestScoreExtraction
    - TestEthicalFlags        → TestFlagExtraction / TestSeverityExtraction
    - TestSingleServerArch    → obsolete (URL-based transport replaced
                                 by pluggable CouncilBackend)
    - TestVerdict.test_handles_error_gracefully
                              → TestDeliberate.test_partial_backend_outage
"""

from __future__ import annotations

from agi.reasoning.divine_council import (
    COUNCIL_MEMBERS,
    MODEL_NAME,
    CouncilVerdict,
    CouncilVote,
)


class TestCouncilMembers:
    """Tests for council member definitions."""

    def test_seven_members(self) -> None:
        assert len(COUNCIL_MEMBERS) == 7

    def test_required_members(self) -> None:
        expected = {
            "judge",
            "advocate",
            "synthesizer",
            "ethicist",
            "historian",
            "futurist",
            "pragmatist",
        }
        assert set(COUNCIL_MEMBERS.keys()) == expected

    def test_each_has_system_prompt(self) -> None:
        for _mid, info in COUNCIL_MEMBERS.items():
            assert "system_prompt" in info
            assert len(info["system_prompt"]) > 50

    def test_enriched_prompts_have_mission(self) -> None:
        for mid, info in COUNCIL_MEMBERS.items():
            assert "MISSION" in info["system_prompt"], f"{mid} missing MISSION"

    def test_enriched_prompts_have_rules(self) -> None:
        for mid, info in COUNCIL_MEMBERS.items():
            assert "RULES" in info["system_prompt"], f"{mid} missing RULES"

    def test_enriched_prompts_have_success_metrics(self) -> None:
        for mid, info in COUNCIL_MEMBERS.items():
            assert (
                "SUCCESS METRICS" in info["system_prompt"]
            ), f"{mid} missing SUCCESS METRICS"

    def test_each_has_color(self) -> None:
        for mid, info in COUNCIL_MEMBERS.items():
            assert "color" in info, f"{mid} missing color"
            assert info["color"].startswith("#"), f"{mid} color not hex"


class TestModelName:
    """Tests for model name constant."""

    def test_model_name_exists(self) -> None:
        assert MODEL_NAME is not None

    def test_model_name_is_26b(self) -> None:
        assert "26B" in MODEL_NAME


class TestFormatLog:
    """Tests for the UI debate log formatting."""

    def test_includes_all_members(self) -> None:
        verdict = CouncilVerdict(
            consensus=True,
            synthesis="The answer is...",
            votes={
                "judge": CouncilVote("judge", "Good. 8/10", 8, latency_s=5),
                "advocate": CouncilVote("advocate", "I challenge...", 5, latency_s=4),
                "synthesizer": CouncilVote("synthesizer", "Merged...", 7, latency_s=6),
                "ethicist": CouncilVote("ethicist", "No concerns.", 8, latency_s=3),
                "historian": CouncilVote(
                    "historian", "Past precedent...", 7, latency_s=4
                ),
                "futurist": CouncilVote("futurist", "Long-term ok.", 7, latency_s=5),
                "pragmatist": CouncilVote("pragmatist", "Feasible.", 8, latency_s=3),
            },
            approval_count=6,
            challenge_count=1,
        )
        log = verdict.format_log()
        for name in [
            "Judge",
            "Advocate",
            "Synthesizer",
            "Ethicist",
            "Historian",
            "Futurist",
            "Pragmatist",
        ]:
            assert name in log, f"{name} missing from log"
        assert "Yes" in log and "approve" in log


class TestVerdictDataclass:
    """Tests for the CouncilVerdict structure."""

    def test_defaults(self) -> None:
        v = CouncilVerdict(consensus=False, synthesis="")
        assert v.method == "divine_council"
        assert v.approval_count == 0
        assert v.votes == {}
