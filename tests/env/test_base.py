# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""
Unit tests for environment base module.
Sprint 6 Phase 3.
"""

import numpy as np
import pytest
from agi.env.base import (
    Space,
    SpaceType,
    EnvironmentSpec,
    StepResult,
    ResetResult,
    Environment,
    EnvironmentRegistry,
    create_env,
    register_env,
)
from agi.env.backends import MockEnvironment, MockConfig


class TestSpace:
    """Tests for Space class."""

    def test_box_space_creation(self, box_space):
        assert box_space.space_type == SpaceType.BOX
        assert box_space.shape == (4,)
        assert box_space.low is not None
        assert box_space.high is not None

    def test_discrete_space_creation(self, discrete_space):
        assert discrete_space.space_type == SpaceType.DISCRETE
        assert discrete_space.n == 5

    def test_multi_discrete_space(self):
        space = Space.multi_discrete([3, 4, 5])
        assert space.space_type == SpaceType.MULTI_DISCRETE
        assert space.nvec == [3, 4, 5]
        assert space.shape == (3,)

    def test_dict_space(self, box_space, discrete_space):
        space = Space.dict_space({"obs": box_space, "act": discrete_space})
        assert space.space_type == SpaceType.DICT
        assert "obs" in space.spaces

    def test_box_sample(self, box_space):
        sample = box_space.sample()
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (4,)

    def test_discrete_sample(self, discrete_space):
        sample = discrete_space.sample()
        assert 0 <= sample < 5

    def test_box_contains(self, box_space):
        valid = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert box_space.contains(valid)
        invalid = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert not box_space.contains(invalid)

    def test_discrete_contains(self, discrete_space):
        assert discrete_space.contains(3) is True
        assert discrete_space.contains(5) is False
        assert discrete_space.contains(-1) is False


class TestEnvironmentSpec:
    """Tests for EnvironmentSpec."""

    def test_spec_creation(self, box_space):
        spec = EnvironmentSpec(
            name="test_env",
            observation_space=box_space,
            action_space=Space.discrete(3),
            max_episode_steps=500,
        )
        assert spec.name == "test_env"
        assert spec.max_episode_steps == 500
        assert "version" in spec.metadata

    def test_spec_default_metadata(self, box_space):
        spec = EnvironmentSpec(
            name="x", observation_space=box_space, action_space=box_space
        )
        assert spec.metadata["version"] == "1.0.0"


class TestStepResult:
    """Tests for StepResult."""

    def test_done_property(self):
        r1 = StepResult(observation=None, reward=0.0, terminated=True, truncated=False)
        assert r1.done is True
        r2 = StepResult(observation=None, reward=0.0, terminated=False, truncated=True)
        assert r2.done is True
        r3 = StepResult(observation=None, reward=0.0, terminated=False, truncated=False)
        assert r3.done is False


class TestMockEnvironment:
    """Tests for MockEnvironment."""

    @pytest.mark.asyncio
    async def test_reset(self, mock_env):
        result = await mock_env.reset()
        assert isinstance(result, ResetResult)
        assert result.observation is not None

    @pytest.mark.asyncio
    async def test_step(self, mock_env):
        await mock_env.reset()
        action = mock_env.action_space.sample()
        result = await mock_env.step(action)
        assert isinstance(result, StepResult)
        assert isinstance(result.reward, float)

    @pytest.mark.asyncio
    async def test_close(self, mock_env):
        await mock_env.reset()
        await mock_env.close()
        with pytest.raises(RuntimeError):
            await mock_env.step(np.zeros(4))

    @pytest.mark.asyncio
    async def test_observe(self, mock_env):
        await mock_env.reset()
        obs = await mock_env.observe()
        assert obs is not None

    @pytest.mark.asyncio
    async def test_seed_reproducibility(self, mock_env):
        mock_env.seed(42)
        r1 = await mock_env.reset(seed=42)
        mock_env2 = MockEnvironment()
        mock_env2.seed(42)
        r2 = await mock_env2.reset(seed=42)
        np.testing.assert_array_equal(r1.observation, r2.observation)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with MockEnvironment() as env:
            result = await env.reset()
            assert result.observation is not None

    @pytest.mark.asyncio
    async def test_max_episode_truncation(self):
        config = MockConfig(max_episode_steps=3, terminate_prob=0.0)
        env = MockEnvironment(config=config)
        await env.reset()
        for _ in range(3):
            result = await env.step(np.zeros(4))
        assert result.truncated is True


class TestEnvironmentRegistry:
    """Tests for EnvironmentRegistry."""

    def test_mock_registered(self):
        assert "mock" in EnvironmentRegistry.list()
        assert "mock:simple" in EnvironmentRegistry.list()

    def test_create_mock(self):
        env = create_env("mock")
        assert isinstance(env, MockEnvironment)

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError):
            create_env("nonexistent:env")
