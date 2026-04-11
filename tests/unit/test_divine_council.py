# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the Divine Council multi-agent deliberation.

Tests parallel deliberation, vote aggregation, consensus detection,
ethical flagging, and synthesis.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agi.reasoning.divine_council import (
    COUNCIL_MEMBERS,
    MODEL_NAME,
    CouncilVerdict,
    CouncilVote,
    DivineCouncil,
)


def _mock_response(content):
    mock = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock


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


class TestMaxTokens:
    """Tests for increased max_tokens (512 for 26B-A4B)."""

    def test_call_uses_512_tokens(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as m:
            m.return_value = _mock_response("Score: 8/10")
            council._call_member("judge", "test")
        call_args = m.call_args
        body = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert body["max_tokens"] == 512


class TestDeliberation:
    """Tests for the full deliberation pipeline."""

    def test_produces_verdict(self) -> None:
        council = DivineCouncil()
        responses = [
            _mock_response("The response is accurate. Score: 8/10"),
            _mock_response("I challenge this — the reasoning is weak."),
            _mock_response("Combining both views, the answer is..."),
            _mock_response("No ethical concerns. The response is fair."),
            _mock_response("We tried something similar before. 7/10"),
            _mock_response("Long-term this looks sustainable. 7/10"),
            _mock_response("Feasible with current resources. 8/10"),
        ]

        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.side_effect = responses
            verdict = council.deliberate(
                "Is AI safe?",
                superego_response="AI needs regulation.",
                id_response="AI empowers creativity.",
            )

        assert isinstance(verdict, CouncilVerdict)
        assert verdict.method == "divine_council"
        assert len(verdict.votes) == 7
        assert verdict.synthesis != ""
        assert verdict.total_latency_s >= 0

    def test_consensus_when_approved(self) -> None:
        council = DivineCouncil()
        responses = [
            _mock_response("Excellent answer. 9/10"),
            _mock_response("Minor quibble but overall good."),
            _mock_response("Well-integrated synthesis."),
            _mock_response("Ethically sound. No concerns."),
            _mock_response("Consistent with past successes. 8/10"),
            _mock_response("Good long-term outlook. 8/10"),
            _mock_response("Practical and achievable. 9/10"),
        ]

        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.side_effect = responses
            verdict = council.deliberate("Test query")

        assert verdict.consensus is True
        assert verdict.approval_count >= 4

    def test_no_consensus_with_ethical_flag(self) -> None:
        council = DivineCouncil()
        responses = [
            _mock_response("Good. 8/10"),
            _mock_response("Acceptable."),
            _mock_response("Synthesis looks good."),
            _mock_response("CONCERN: potential bias detected. Harm risk."),
            _mock_response("No prior precedent for this. 7/10"),
            _mock_response("Consequences seem manageable. 7/10"),
            _mock_response("Feasible approach. 8/10"),
        ]

        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.side_effect = responses
            verdict = council.deliberate("Test query")

        assert len(verdict.ethical_flags) > 0
        assert verdict.consensus is False


class TestScoreParsing:
    """Tests for score extraction from responses."""

    def test_parses_score(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response("Score: 8/10. Good answer.")
            vote = council._call_member("judge", "test")

        assert vote.score == 8.0

    def test_default_score_without_number(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response("This is a reasonable answer.")
            vote = council._call_member("judge", "test")

        assert vote.score == 5.0


class TestEthicalFlags:
    """Tests for ethicist flag detection."""

    def test_detects_bias(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                "There is potential bias in the response."
            )
            vote = council._call_member("ethicist", "test")

        assert "bias" in vote.flags

    def test_detects_harm(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                "This could cause harm to vulnerable groups."
            )
            vote = council._call_member("ethicist", "test")

        assert "harm" in vote.flags

    def test_clean_passes(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response("Ethically sound. No issues found.")
            vote = council._call_member("ethicist", "test")

        assert len(vote.flags) == 0


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

    def test_handles_error_gracefully(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.side_effect = Exception("all down")
            verdict = council.deliberate("test")

        # Should still produce a verdict, just with error messages
        assert isinstance(verdict, CouncilVerdict)
        assert len(verdict.votes) == 7


class TestSingleServerArchitecture:
    """Tests that all members share a single llama-server endpoint."""

    def test_default_urls_all_same(self) -> None:
        council = DivineCouncil()
        urls = set(council._urls.values())
        assert len(urls) == 1
        assert urls.pop() == "http://localhost:8084"

    def test_all_members_have_url(self) -> None:
        council = DivineCouncil()
        for mid in COUNCIL_MEMBERS:
            assert mid in council._urls

    def test_custom_base_url(self) -> None:
        council = DivineCouncil(base_url="http://gpu-box:9090")
        for mid in COUNCIL_MEMBERS:
            assert council._urls[mid] == "http://gpu-box:9090"

    def test_member_urls_override(self) -> None:
        custom = {"judge": "http://a:1", "advocate": "http://b:2"}
        council = DivineCouncil(member_urls=custom)
        assert council._urls == custom
