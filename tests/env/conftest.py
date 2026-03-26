# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Shared fixtures for environment tests."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from agi.env.base import Space, EnvironmentSpec
from agi.env.backends import MockEnvironment, MockConfig


@pytest.fixture
def mock_env():
    """Create a basic mock environment."""
    return MockEnvironment()


@pytest.fixture
def mock_config():
    """Create a mock environment config."""
    return MockConfig(obs_dim=10, action_dim=4, max_episode_steps=100)


@pytest.fixture
def configured_mock_env(mock_config):
    """Create a configured mock environment."""
    return MockEnvironment(config=mock_config)


@pytest.fixture
def box_space():
    """Create a Box space for testing."""
    return Space.box(low=-1.0, high=1.0, shape=(4,))


@pytest.fixture
def discrete_space():
    """Create a Discrete space for testing."""
    return Space.discrete(n=5)
