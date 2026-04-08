# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the Dungeon Master training service.

Tests scenario generation (ErisML Pantheon + LLM-generated),
debate orchestration, and episode storage — all with mocked LLM
endpoints and database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agi.training.dungeon_master import (
    DMConfig,
    DebateResult,
    DungeonMaster,
    ETHICAL_DOMAINS,
    NOVEL_SCENARIO_PROMPTS,
    SessionResult,
    TrainingScenario,
)


@pytest.fixture()
def dm():
    """Create a DungeonMaster with mocked dependencies."""
    config = DMConfig(
        ego_url="http://mock-ego:8084",
        superego_url="http://mock-superego:8080",
        id_url="http://mock-id:8082",
    )
    with patch("agi.training.dungeon_master.EpisodicMemory", None):
        return DungeonMaster(config)


@pytest.fixture()
def mock_llm_response():
    """Create a mock requests.post response."""

    def _make_response(content: str):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        return mock_resp

    return _make_response


class TestDMConfig:
    """Tests for DMConfig defaults."""

    def test_defaults(self) -> None:
        cfg = DMConfig()
        assert cfg.ego_url == "http://localhost:8084"
        assert cfg.superego_url == "http://localhost:8080"
        assert cfg.id_url == "http://localhost:8082"
        assert cfg.episodes_per_session == 20
        assert cfg.timeout == 300

    def test_custom_urls(self) -> None:
        cfg = DMConfig(ego_url="http://custom:9999")
        assert cfg.ego_url == "http://custom:9999"


class TestEthicalDomains:
    """Tests for the ethical domain coverage."""

    def test_eight_domains_defined(self) -> None:
        assert len(ETHICAL_DOMAINS) == 8

    def test_domains_match_erisml(self) -> None:
        expected = {
            "Consequences",
            "Rights & Duties",
            "Justice & Fairness",
            "Autonomy & Agency",
            "Privacy & Data Governance",
            "Societal & Environmental",
            "Procedural & Legitimacy",
            "Epistemic Status",
        }
        assert set(ETHICAL_DOMAINS) == expected

    def test_novel_prompts_cover_all_domains(self) -> None:
        prompt_domains = {d for d, _ in NOVEL_SCENARIO_PROMPTS}
        assert prompt_domains == set(ETHICAL_DOMAINS)


class TestTrainingScenario:
    """Tests for the TrainingScenario dataclass."""

    def test_creation(self) -> None:
        s = TrainingScenario(
            scenario_id="test-001",
            domain="Consequences",
            title="Test Scenario",
            narrative="A hospital must decide...",
            options=["Option A", "Option B"],
            difficulty=2,
        )
        assert s.scenario_id == "test-001"
        assert s.domain == "Consequences"
        assert s.source == "erisml"

    def test_default_source(self) -> None:
        s = TrainingScenario(
            scenario_id="x",
            domain="x",
            title="x",
            narrative="x",
            options=[],
            difficulty=1,
        )
        assert s.source == "erisml"


class TestScenarioGeneration:
    """Tests for scenario generation."""

    def test_llm_scenario_generation(self, dm, mock_llm_response) -> None:
        with patch("agi.training.dungeon_master.requests.post") as mock_post:
            mock_post.return_value = mock_llm_response(
                "A hospital faces a difficult choice about a scarce treatment..."
            )
            scenario = dm._scenario_from_llm(difficulty=2)

        assert isinstance(scenario, TrainingScenario)
        assert scenario.source == "llm_generated"
        assert scenario.difficulty == 2
        assert scenario.domain in {d for d, _ in NOVEL_SCENARIO_PROMPTS}

    def test_generate_scenario_without_erisml(self, dm, mock_llm_response) -> None:
        with (
            patch("agi.training.dungeon_master._PANTHEON_CASES", []),
            patch("agi.training.dungeon_master.ERISML_AVAILABLE", False),
            patch("agi.training.dungeon_master.requests.post") as mock_post,
        ):
            mock_post.return_value = mock_llm_response("A dilemma unfolds...")
            scenario = dm.generate_scenario(difficulty=3)

        assert isinstance(scenario, TrainingScenario)
        assert scenario.source == "llm_generated"


class TestDebateOrchestration:
    """Tests for the debate flow."""

    def test_run_debate(self, dm, mock_llm_response) -> None:
        scenario = TrainingScenario(
            scenario_id="test-debate",
            domain="Consequences",
            title="Test",
            narrative="A difficult choice must be made...",
            options=["Option A", "Option B", "Option C"],
            difficulty=2,
        )

        responses = [
            # Superego response
            mock_llm_response("From a deontological perspective, we must consider..."),
            # Id response
            mock_llm_response("The human cost is what matters most here..."),
            # Ego evaluation
            mock_llm_response(
                '{"superego_score": 7, "id_score": 8, "synthesis_score": 7, '
                '"feedback": "Both could better integrate opposing views."}'
            ),
        ]

        with patch("agi.training.dungeon_master.requests.post") as mock_post:
            mock_post.side_effect = responses
            result = dm.run_debate(scenario)

        assert isinstance(result, DebateResult)
        assert result.scenario_id == "test-debate"
        assert result.superego_response != ""
        assert result.id_response != ""
        assert 0.0 <= result.synthesis_score <= 1.0
        assert result.latency_s >= 0

    def test_debate_handles_malformed_eval(self, dm, mock_llm_response) -> None:
        scenario = TrainingScenario(
            scenario_id="test-malformed",
            domain="Ethics",
            title="Test",
            narrative="...",
            options=[],
            difficulty=1,
        )

        responses = [
            mock_llm_response("Superego says..."),
            mock_llm_response("Id says..."),
            mock_llm_response("This is not valid JSON at all"),
        ]

        with patch("agi.training.dungeon_master.requests.post") as mock_post:
            mock_post.side_effect = responses
            result = dm.run_debate(scenario)

        # Should still return a result with default score
        assert isinstance(result, DebateResult)
        assert result.synthesis_score == 0.5  # Default fallback


class TestSessionResult:
    """Tests for the SessionResult dataclass."""

    def test_creation(self) -> None:
        sr = SessionResult(
            session_id="test",
            episodes=10,
            mean_synthesis_score=0.72,
            mean_domain_score=0.68,
            domains_covered=["Consequences", "Rights & Duties"],
            duration_s=120.0,
            results=[],
        )
        assert sr.episodes == 10
        assert sr.mean_synthesis_score == 0.72
