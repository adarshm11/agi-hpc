# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for Episodic Memory store_from_chat() convenience method
and integration patterns used by the RAG server.

Tests use mocked PostgreSQL connections since the unit test environment
does not have a running database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agi.memory.episodic.store import (
    Episode,
    EpisodicMemory,
    EpisodicMemoryConfig,
)


@pytest.fixture()
def memory():
    """Create an EpisodicMemory with mocked PostgreSQL."""
    mock_pg = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_pg.connect.return_value = mock_conn

    with patch("agi.memory.episodic.store.psycopg2", mock_pg):
        mem = EpisodicMemory(
            EpisodicMemoryConfig(
                db_dsn="dbname=test",
                auto_create_table=True,
            )
        )
        yield mem


class TestStoreFromChat:
    """Tests for the store_from_chat convenience method."""

    def test_basic_store(self, memory: EpisodicMemory) -> None:
        episode_id = memory.store_from_chat(
            session_id="sess-1",
            user_msg="What is Python?",
            response="Python is a programming language.",
            hemisphere="lh",
        )
        assert episode_id  # UUID string
        assert len(episode_id) == 36  # UUID format

    def test_store_with_safety_results(self, memory: EpisodicMemory) -> None:
        safety_input = {
            "passed": True,
            "score": 0.95,
            "flags": [],
            "gate": "input",
        }
        safety_output = {
            "passed": True,
            "score": 0.90,
            "flags": [],
            "gate": "output",
        }
        episode_id = memory.store_from_chat(
            session_id="sess-2",
            user_msg="Explain quantum computing",
            response="Quantum computing uses qubits...",
            hemisphere="both",
            safety_input=safety_input,
            safety_output=safety_output,
        )
        assert episode_id

    def test_store_with_embedding(self, memory: EpisodicMemory) -> None:
        import numpy as np

        mock_embed = MagicMock(return_value=np.random.randn(1024).astype(np.float32))
        episode_id = memory.store_from_chat(
            session_id="sess-3",
            user_msg="Hello world",
            response="Greetings!",
            hemisphere="lh",
            embed_fn=mock_embed,
        )
        assert episode_id
        mock_embed.assert_called_once_with("Hello world")

    def test_store_with_failed_embedding(self, memory: EpisodicMemory) -> None:
        mock_embed = MagicMock(side_effect=RuntimeError("model not loaded"))
        episode_id = memory.store_from_chat(
            session_id="sess-4",
            user_msg="Test query",
            response="Test response",
            embed_fn=mock_embed,
        )
        # Should still store, just without embedding
        assert episode_id

    def test_store_with_no_safety(self, memory: EpisodicMemory) -> None:
        episode_id = memory.store_from_chat(
            session_id="sess-5",
            user_msg="Simple question",
            response="Simple answer",
        )
        assert episode_id


class TestEpisodeDataclass:
    """Tests for the Episode data transfer object."""

    def test_to_dict(self) -> None:
        ep = Episode(
            id="test-uuid",
            session_id="sess-1",
            user_message="Hello",
            atlas_response="Hi there",
            hemisphere="lh",
            quality_score=0.8,
        )
        d = ep.to_dict()
        assert d["id"] == "test-uuid"
        assert d["session_id"] == "sess-1"
        assert d["user_message"] == "Hello"
        assert d["hemisphere"] == "lh"
        assert d["quality_score"] == 0.8

    def test_default_values(self) -> None:
        ep = Episode()
        assert ep.hemisphere == "lh"
        assert ep.quality_score == 0.0
        assert ep.safety_flags == {}
        assert ep.metadata == {}
        assert ep.embedding is None


class TestEpisodicMemoryConfig:
    """Tests for EpisodicMemoryConfig defaults."""

    def test_defaults(self) -> None:
        cfg = EpisodicMemoryConfig()
        assert cfg.db_dsn == "dbname=atlas user=claude"
        assert cfg.table_name == "episodes"
        assert cfg.auto_create_table is True

    def test_custom_dsn(self) -> None:
        cfg = EpisodicMemoryConfig(db_dsn="dbname=test user=test")
        assert cfg.db_dsn == "dbname=test user=test"
