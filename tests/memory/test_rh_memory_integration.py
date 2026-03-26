# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""
Integration tests for RH <-> Memory subsystem.
Sprint 6 Phase 3.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from agi.memory.unified import UnifiedMemoryService, UnifiedMemoryConfig


@pytest.fixture
def rh_memory_config():
    return UnifiedMemoryConfig(
        semantic_addr="localhost:50053",
        episodic_addr="localhost:50052",
        procedural_addr="localhost:50054",
        timeout_sec=2.0,
        default_max_facts=5,
        default_max_skills=10,
        enable_caching=False,
    )


class TestRHWorldStateQueries:
    """Test RH querying memory for world state information."""

    @pytest.mark.asyncio
    async def test_world_state_from_semantic(
        self, rh_memory_config, mock_semantic_client
    ):
        service = UnifiedMemoryService(
            config=rh_memory_config,
            semantic_client=mock_semantic_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="Query world state for obstacle positions",
            task_type="perception",
            include_episodic=False,
            include_procedural=False,
        )
        assert len(ctx.facts) >= 1

    @pytest.mark.asyncio
    async def test_world_state_empty_on_failure(self, rh_memory_config):
        failing = MagicMock()
        failing.search = AsyncMock(side_effect=Exception("unavailable"))
        service = UnifiedMemoryService(config=rh_memory_config, semantic_client=failing)
        ctx = await service.enrich_planning_context(
            task_description="query",
            include_episodic=False,
            include_procedural=False,
        )
        assert ctx.facts == []


class TestRHSkillRetrieval:
    """Test RH retrieving skills from procedural memory."""

    @pytest.mark.asyncio
    async def test_skill_retrieval(self, rh_memory_config, mock_procedural_client):
        service = UnifiedMemoryService(
            config=rh_memory_config,
            procedural_client=mock_procedural_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="Pick up object from table",
            task_type="manipulation",
            include_semantic=False,
            include_episodic=False,
        )
        assert len(ctx.skills) >= 1

    @pytest.mark.asyncio
    async def test_skill_proficiency_in_context(
        self, rh_memory_config, mock_procedural_client
    ):
        service = UnifiedMemoryService(
            config=rh_memory_config,
            procedural_client=mock_procedural_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="Grasp and place",
            task_type="manipulation",
            include_semantic=False,
            include_episodic=False,
        )
        prompt = ctx.to_prompt_context()
        assert "proficiency:" in prompt


class TestRHProceduralMemoryForManipulation:
    """Test procedural memory used for manipulation tasks."""

    @pytest.mark.asyncio
    async def test_manipulation_uses_procedural_and_semantic(
        self,
        rh_memory_config,
        mock_semantic_client,
        mock_procedural_client,
    ):
        service = UnifiedMemoryService(
            config=rh_memory_config,
            semantic_client=mock_semantic_client,
            procedural_client=mock_procedural_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="Assemble component A with B",
            task_type="manipulation",
            include_episodic=False,
        )
        assert len(ctx.facts) >= 1
        assert len(ctx.skills) >= 1

    @pytest.mark.asyncio
    async def test_procedural_failure_still_returns_semantic(
        self, rh_memory_config, mock_semantic_client
    ):
        failing = MagicMock()
        failing.search = AsyncMock(side_effect=Exception("fail"))
        service = UnifiedMemoryService(
            config=rh_memory_config,
            semantic_client=mock_semantic_client,
            procedural_client=failing,
        )
        ctx = await service.enrich_planning_context(
            task_description="Test partial",
            task_type="manipulation",
            include_episodic=False,
        )
        assert len(ctx.facts) >= 1
        assert ctx.skills == []
