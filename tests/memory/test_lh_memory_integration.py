# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""
Integration tests for LH <-> Memory subsystem.
Sprint 6 Phase 3.
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock
from agi.memory.unified import UnifiedMemoryService, UnifiedMemoryConfig


@pytest.fixture
def memory_config():
    return UnifiedMemoryConfig(
        semantic_addr="localhost:50053",
        episodic_addr="localhost:50052",
        procedural_addr="localhost:50054",
        timeout_sec=2.0,
        default_max_facts=5,
        default_max_episodes=3,
        default_max_skills=5,
        enable_caching=True,
    )


@pytest.fixture
def memory_service_with_clients(
    memory_config,
    mock_semantic_client,
    mock_episodic_client,
    mock_procedural_client,
):
    return UnifiedMemoryService(
        config=memory_config,
        semantic_client=mock_semantic_client,
        episodic_client=mock_episodic_client,
        procedural_client=mock_procedural_client,
    )


class TestLHMemoryEnrichment:
    """Test LH planning context enrichment via UnifiedMemoryService."""

    @pytest.mark.asyncio
    async def test_enrichment_populates_all_context(self, memory_service_with_clients):
        ctx = await memory_service_with_clients.enrich_planning_context(
            task_description="Navigate to charging station",
            task_type="navigation",
            scenario_id="warehouse-v2",
        )
        assert ctx.has_context is True
        assert len(ctx.facts) >= 1 and len(ctx.episodes) >= 1 and len(ctx.skills) >= 1

    @pytest.mark.asyncio
    async def test_enrichment_returns_prompt_context_string(
        self, memory_service_with_clients
    ):
        ctx = await memory_service_with_clients.enrich_planning_context(
            task_description="Pick up the red cube",
            task_type="manipulation",
        )
        prompt = ctx.to_prompt_context()
        assert isinstance(prompt, str)
        assert "RELEVANT FACTS" in prompt
        assert "SIMILAR PAST EPISODES" in prompt
        assert "AVAILABLE SKILLS" in prompt

    @pytest.mark.asyncio
    async def test_empty_context_when_all_excluded(self, memory_config):
        service = UnifiedMemoryService(config=memory_config)
        ctx = await service.enrich_planning_context(
            task_description="test",
            include_semantic=False,
            include_episodic=False,
            include_procedural=False,
        )
        assert ctx.facts == [] and ctx.episodes == [] and ctx.skills == []

    @pytest.mark.asyncio
    async def test_partial_include_flags(self, memory_service_with_clients):
        ctx = await memory_service_with_clients.enrich_planning_context(
            task_description="Navigate corridor",
            task_type="navigation",
            include_episodic=False,
        )
        assert len(ctx.facts) >= 1
        assert ctx.episodes == []
        assert len(ctx.skills) >= 1


class TestLHMemoryParallelQueries:
    """Verify parallel execution of memory queries."""

    @pytest.mark.asyncio
    async def test_parallel_queries_execute_concurrently(self, memory_config):
        call_times = []

        async def slow_search(**kwargs):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            result = MagicMock()
            result.facts = []
            return result

        async def slow_episodic_search(**kwargs):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            result = MagicMock()
            result.episodes = []
            return result

        async def slow_procedural_search(**kwargs):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            result = MagicMock()
            result.skills = []
            return result

        sem = MagicMock()
        sem.search = AsyncMock(side_effect=slow_search)
        ep = MagicMock()
        ep.search = AsyncMock(side_effect=slow_episodic_search)
        proc = MagicMock()
        proc.search = AsyncMock(side_effect=slow_procedural_search)

        service = UnifiedMemoryService(
            config=memory_config,
            semantic_client=sem,
            episodic_client=ep,
            procedural_client=proc,
        )
        start = time.monotonic()
        await service.enrich_planning_context(
            task_description="test parallel",
            task_type="test",
        )
        elapsed = time.monotonic() - start
        # If sequential >= 0.15s; parallel should be ~0.05s
        assert elapsed < 0.12


class TestLHMemoryTimeout:
    """Test timeout and partial failure handling."""

    @pytest.mark.asyncio
    async def test_semantic_failure_returns_partial(
        self,
        memory_config,
        mock_episodic_client,
        mock_procedural_client,
    ):
        failing_semantic = MagicMock()
        failing_semantic.search = AsyncMock(side_effect=Exception("semantic down"))
        service = UnifiedMemoryService(
            config=memory_config,
            semantic_client=failing_semantic,
            episodic_client=mock_episodic_client,
            procedural_client=mock_procedural_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="test timeout",
            task_type="navigation",
        )
        assert ctx.facts == []
        assert len(ctx.episodes) >= 1
        assert len(ctx.skills) >= 1

    @pytest.mark.asyncio
    async def test_episodic_failure_returns_partial(
        self,
        memory_config,
        mock_semantic_client,
        mock_procedural_client,
    ):
        failing_episodic = MagicMock()
        failing_episodic.search = AsyncMock(side_effect=Exception("episodic down"))
        service = UnifiedMemoryService(
            config=memory_config,
            semantic_client=mock_semantic_client,
            episodic_client=failing_episodic,
            procedural_client=mock_procedural_client,
        )
        ctx = await service.enrich_planning_context(
            task_description="test timeout",
            task_type="navigation",
        )
        assert len(ctx.facts) >= 1
        assert ctx.episodes == []
        assert len(ctx.skills) >= 1

    @pytest.mark.asyncio
    async def test_all_failures_returns_empty(self, memory_config):
        sem = MagicMock()
        sem.search = AsyncMock(side_effect=Exception("fail"))
        ep = MagicMock()
        ep.search = AsyncMock(side_effect=Exception("fail"))
        proc = MagicMock()
        proc.search = AsyncMock(side_effect=Exception("fail"))
        service = UnifiedMemoryService(
            config=memory_config,
            semantic_client=sem,
            episodic_client=ep,
            procedural_client=proc,
        )
        ctx = await service.enrich_planning_context(
            task_description="total failure",
            task_type="test",
        )
        assert ctx.facts == [] and ctx.episodes == [] and ctx.skills == []
        assert ctx.has_context is False


class TestLHMemoryCaching:
    """Test caching behaviour of UnifiedMemoryService."""

    @pytest.mark.asyncio
    async def test_cache_returns_same_context(self, memory_service_with_clients):
        ctx1 = await memory_service_with_clients.enrich_planning_context(
            task_description="cache test",
            task_type="nav",
            scenario_id="s1",
        )
        ctx2 = await memory_service_with_clients.enrich_planning_context(
            task_description="cache test",
            task_type="nav",
            scenario_id="s1",
        )
        assert ctx1 is ctx2

    @pytest.mark.asyncio
    async def test_different_keys_not_cached(self, memory_service_with_clients):
        ctx1 = await memory_service_with_clients.enrich_planning_context(
            task_description="task A",
            task_type="nav",
        )
        ctx2 = await memory_service_with_clients.enrich_planning_context(
            task_description="task B",
            task_type="nav",
        )
        assert ctx1 is not ctx2

    @pytest.mark.asyncio
    async def test_caching_disabled(
        self,
        mock_semantic_client,
        mock_episodic_client,
        mock_procedural_client,
    ):
        config = UnifiedMemoryConfig(enable_caching=False)
        service = UnifiedMemoryService(
            config=config,
            semantic_client=mock_semantic_client,
            episodic_client=mock_episodic_client,
            procedural_client=mock_procedural_client,
        )
        ctx1 = await service.enrich_planning_context(
            task_description="no cache",
            task_type="test",
        )
        ctx2 = await service.enrich_planning_context(
            task_description="no cache",
            task_type="test",
        )
        assert ctx1 is not ctx2


class TestLHMemoryResultAggregation:
    """Test result aggregation and prompt formatting."""

    @pytest.mark.asyncio
    async def test_prompt_context_format(self, memory_service_with_clients):
        ctx = await memory_service_with_clients.enrich_planning_context(
            task_description="Navigate to waypoint B",
            task_type="navigation",
        )
        prompt = ctx.to_prompt_context()
        assert "confidence:" in prompt
        assert "proficiency:" in prompt

    @pytest.mark.asyncio
    async def test_empty_context_prompt_is_empty(self, memory_config):
        service = UnifiedMemoryService(config=memory_config)
        ctx = await service.enrich_planning_context(
            task_description="x",
            include_semantic=False,
            include_episodic=False,
            include_procedural=False,
        )
        assert ctx.to_prompt_context() == ""

    @pytest.mark.asyncio
    async def test_stub_mode_returns_placeholder(self):
        service = UnifiedMemoryService(config=UnifiedMemoryConfig())
        ctx = await service.enrich_planning_context(
            task_description="stub test",
            task_type="general",
        )
        assert ctx.has_context is True
        assert any(
            "Placeholder" in f.get("content", "") or "stub" in f.get("source", "")
            for f in ctx.facts
        )
