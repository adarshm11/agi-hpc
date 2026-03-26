# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""
Unit tests for MuJoCo environment fallback mode.
Sprint 6 Phase 3.
"""

import numpy as np
import pytest
from agi.env.backends import MuJoCoEnvironment, MuJoCoConfig
from agi.env.base import StepResult, ResetResult


class TestMuJoCoFallback:
    """Test MuJoCo environment in fallback mode (no mujoco installed)."""

    @pytest.fixture
    def mujoco_env(self):
        return MuJoCoEnvironment()

    def test_spec_exists(self, mujoco_env):
        spec = mujoco_env.spec
        assert spec is not None
        assert spec.observation_space is not None
        assert spec.action_space is not None

    @pytest.mark.asyncio
    async def test_reset_returns_observation(self, mujoco_env):
        result = await mujoco_env.reset()
        assert isinstance(result, ResetResult)
        assert result.observation is not None

    @pytest.mark.asyncio
    async def test_step_returns_result(self, mujoco_env):
        await mujoco_env.reset()
        action = np.zeros(6)
        result = await mujoco_env.step(action)
        assert isinstance(result, StepResult)

    @pytest.mark.asyncio
    async def test_close(self, mujoco_env):
        await mujoco_env.reset()
        await mujoco_env.close()

    @pytest.mark.asyncio
    async def test_render_returns_none(self, mujoco_env):
        result = await mujoco_env.render()
        assert result is None

    def test_custom_config(self):
        config = MuJoCoConfig(frame_skip=10, width=640, height=480)
        env = MuJoCoEnvironment(config=config)
        assert env._config.frame_skip == 10
