# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""
Unit tests for PyBullet environment fallback mode.
Sprint 6 Phase 3.
"""

import numpy as np
import pytest
from agi.env.backends import PyBulletEnvironment, PyBulletConfig
from agi.env.base import StepResult, ResetResult


class TestPyBulletFallback:
    """Test PyBullet environment in fallback mode (no pybullet installed)."""

    @pytest.fixture
    def pybullet_env(self):
        return PyBulletEnvironment()

    def test_spec_exists(self, pybullet_env):
        spec = pybullet_env.spec
        assert spec is not None
        assert spec.observation_space is not None
        assert spec.action_space is not None

    @pytest.mark.asyncio
    async def test_reset_returns_observation(self, pybullet_env):
        result = await pybullet_env.reset()
        assert isinstance(result, ResetResult)
        assert result.observation is not None

    @pytest.mark.asyncio
    async def test_step_returns_result(self, pybullet_env):
        await pybullet_env.reset()
        action = np.zeros(7)
        result = await pybullet_env.step(action)
        assert isinstance(result, StepResult)

    @pytest.mark.asyncio
    async def test_close(self, pybullet_env):
        await pybullet_env.reset()
        await pybullet_env.close()

    @pytest.mark.asyncio
    async def test_render_returns_none(self, pybullet_env):
        result = await pybullet_env.render()
        assert result is None

    def test_custom_config(self):
        config = PyBulletConfig(use_gui=False, gravity=(0, 0, -10.0))
        env = PyBulletEnvironment(config=config)
        assert env._config.use_gui is False
