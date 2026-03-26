# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""
Integration tests for Safety <-> Episodic Memory subsystem.
Sprint 6 Phase 3.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from agi.memory.unified import UnifiedMemoryService, UnifiedMemoryConfig


@pytest.fixture
def safety_memory_config():
    return UnifiedMemoryConfig(
        semantic_addr="localhost:50053",
        episodic_addr="localhost:50052",
        procedural_addr="localhost:50054",
        timeout_sec=2.0,
        default_max_episodes=10,
        enable_caching=False,
    )


@pytest.fixture
def safety_episodic_client():
    """Episodic client that returns safety-relevant episodes."""
    client = MagicMock()

    async def mock_search(situation_description, task_type="", max_results=5, **kwargs):
        result = MagicMock()
        ep = MagicMock()
        ep.episode_id = "safety_ep_001"
        ep.task_description = f"Safety event: {situation_description[:30]}"
        ep.task_type = task_type or "safety"
        ep.scenario_id = "safety_scenario"
        ep.similarity = 0.9
        ep.insights = ["Reduce speed near obstacles", "Check clearance before turn"]
        ep.outcome = MagicMock(success=False, completion_percentage=0.3)
        ep.HasField = lambda f: f == "outcome"
        result.episodes = [ep]
        return result

    client.search = AsyncMock(side_effect=mock_search)
    return client


class TestSafetyVerdictStorage:
    """Test that safety verdicts are stored in episodic memory."""

    @pytest.mark.asyncio
    async def test_safety_episode_retrieval(
        self, safety_memory_config, safety_episodic_client
    ):
        service = UnifiedMemoryService(
            config=safety_memory_config,
            episodic_client=safety_episodic_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="Emergency stop triggered near obstacle",
            task_type="safety",
            include_semantic=False,
            include_procedural=False,
        )
        assert len(ctx.episodes) >= 1
        ep = ctx.episodes[0]
        assert "Safety event" in ep.get("task_description", "")
        assert ep.get("success") is False

    @pytest.mark.asyncio
    async def test_safety_insights_preserved(
        self, safety_memory_config, safety_episodic_client
    ):
        service = UnifiedMemoryService(
            config=safety_memory_config,
            episodic_client=safety_episodic_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="Near-miss collision event",
            task_type="safety",
            include_semantic=False,
            include_procedural=False,
        )
        assert len(ctx.episodes) >= 1
        ep = ctx.episodes[0]
        assert len(ep.get("insights", [])) >= 1


class TestSafetyAuditTrail:
    """Test audit trail functionality for safety events."""

    @pytest.mark.asyncio
    async def test_prompt_includes_safety_episodes(
        self, safety_memory_config, safety_episodic_client
    ):
        service = UnifiedMemoryService(
            config=safety_memory_config,
            episodic_client=safety_episodic_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="Check safety history for corridor B",
            task_type="safety",
            include_semantic=False,
            include_procedural=False,
        )
        prompt = ctx.to_prompt_context()
        assert "SIMILAR PAST EPISODES" in prompt
        assert "failed" in prompt

    @pytest.mark.asyncio
    async def test_episodic_failure_graceful(self, safety_memory_config):
        failing = MagicMock()
        failing.search = AsyncMock(side_effect=Exception("episodic unavailable"))
        service = UnifiedMemoryService(
            config=safety_memory_config, episodic_client=failing
        )
        ctx = await service.enrich_planning_context(
            task_description="safety query",
            task_type="safety",
            include_semantic=False,
            include_procedural=False,
        )
        assert ctx.episodes == []
        assert ctx.has_context is False


class TestEpisodicRetrievalForSafety:
    """Test episodic retrieval for safety-related queries."""

    @pytest.mark.asyncio
    async def test_safety_query_calls_episodic(
        self, safety_memory_config, safety_episodic_client
    ):
        service = UnifiedMemoryService(
            config=safety_memory_config,
            episodic_client=safety_episodic_client,
        )
        await service.enrich_planning_context(
            task_description="Obstacle detected",
            task_type="safety",
            include_semantic=False,
            include_procedural=False,
        )
        safety_episodic_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_safety_episodes_aggregated(self, safety_memory_config):
        client = MagicMock()

        async def multi_search(**kwargs):
            result = MagicMock()
            episodes = []
            for i in range(3):
                ep = MagicMock()
                ep.episode_id = f"safety_ep_{i:03d}"
                ep.task_description = f"Safety event {i}"
                ep.task_type = "safety"
                ep.scenario_id = "test"
                ep.similarity = 0.8 - i * 0.1
                ep.insights = [f"Insight {i}"]
                ep.outcome = MagicMock(
                    success=i > 0, completion_percentage=0.5 + i * 0.2
                )
                ep.HasField = lambda f: f == "outcome"
                episodes.append(ep)
            result.episodes = episodes
            return result

        client.search = AsyncMock(side_effect=multi_search)
        service = UnifiedMemoryService(
            config=safety_memory_config,
            episodic_client=client,
        )
        ctx = await service.enrich_planning_context(
            task_description="Review all safety events",
            task_type="safety",
            include_semantic=False,
            include_procedural=False,
        )
        assert len(ctx.episodes) == 3
