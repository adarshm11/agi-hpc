# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the daily training + nap cycle infrastructure.

Tests the DungeonMaster CLI argument parsing, session orchestration,
and systemd timer configuration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from agi.training.dungeon_master import DMConfig, DungeonMaster, SessionResult


class TestDailyTrainingShellScript:
    """Tests for daily_training_session.sh existence and structure."""

    def test_script_exists(self) -> None:
        script = Path("scripts/daily_training_session.sh")
        assert script.exists()

    def test_script_has_shebang(self) -> None:
        script = Path("scripts/daily_training_session.sh")
        first_line = script.read_text().split("\n")[0]
        assert first_line.startswith("#!/bin/bash")

    def test_script_has_phases(self) -> None:
        content = Path("scripts/daily_training_session.sh").read_text()
        assert "Phase 1" in content
        assert "Phase 2" in content
        assert "DM Training" in content
        assert "Dreaming Nap" in content

    def test_script_checks_ego_health(self) -> None:
        content = Path("scripts/daily_training_session.sh").read_text()
        assert "8084" in content  # Ego port

    def test_script_supports_skip_nap(self) -> None:
        content = Path("scripts/daily_training_session.sh").read_text()
        assert "--skip-nap" in content


class TestDailyBackupScript:
    """Tests for daily_backup.sh existence and structure."""

    def test_script_exists(self) -> None:
        assert Path("scripts/daily_backup.sh").exists()

    def test_script_backs_up_wiki(self) -> None:
        content = Path("scripts/daily_backup.sh").read_text()
        assert "wiki" in content.lower()

    def test_script_backs_up_postgres(self) -> None:
        content = Path("scripts/daily_backup.sh").read_text()
        assert "pg_dump" in content

    def test_script_has_retention_policy(self) -> None:
        content = Path("scripts/daily_backup.sh").read_text()
        assert "30" in content  # 30 day retention
        assert "12" in content  # 12 week retention


class TestSystemdTimers:
    """Tests for systemd timer configuration files."""

    def test_training_service_exists(self) -> None:
        assert Path("deploy/systemd/atlas-training.service").exists()

    def test_training_timer_exists(self) -> None:
        assert Path("deploy/systemd/atlas-training.timer").exists()

    def test_backup_service_exists(self) -> None:
        assert Path("deploy/systemd/atlas-backup.service").exists()

    def test_backup_timer_exists(self) -> None:
        assert Path("deploy/systemd/atlas-backup.timer").exists()

    def test_training_timer_schedule(self) -> None:
        content = Path("deploy/systemd/atlas-training.timer").read_text()
        assert "10:00:00" in content  # 10 AM UTC

    def test_backup_timer_schedule(self) -> None:
        content = Path("deploy/systemd/atlas-backup.timer").read_text()
        assert "04:00:00" in content  # 4 AM UTC

    def test_timers_are_persistent(self) -> None:
        for timer in ["atlas-training.timer", "atlas-backup.timer"]:
            content = Path(f"deploy/systemd/{timer}").read_text()
            assert "Persistent=true" in content

    def test_dm_service_exists(self) -> None:
        assert Path("deploy/systemd/atlas-ego.service").exists()

    def test_dm_service_runs_on_cpu(self) -> None:
        content = Path("deploy/systemd/atlas-ego.service").read_text()
        assert "--n-gpu-layers 0" in content
        assert "8084" in content


class TestDMSessionIntegration:
    """Tests for the DM session runner."""

    def test_session_with_mocked_llm(self) -> None:
        config = DMConfig(episodes_per_session=2)

        with patch("agi.training.dungeon_master.EpisodicMemory", None):
            dm = DungeonMaster(config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"superego_score": 7, "id_score": 6, '
                        '"synthesis_score": 7, "feedback": "Good."}'
                    }
                }
            ]
        }

        with (
            patch("agi.training.dungeon_master._PANTHEON_CASES", []),
            patch("agi.training.dungeon_master.ERISML_AVAILABLE", False),
            patch("agi.training.dungeon_master.requests.post") as mock_post,
        ):
            mock_post.return_value = mock_resp
            session = dm.run_session(episodes=2, difficulty=1)

        assert session.episodes == 2
        assert 0.0 <= session.mean_synthesis_score <= 1.0
        assert session.duration_s >= 0
        assert len(session.domains_covered) > 0
        assert isinstance(session, SessionResult)


class TestNATSEventPublishing:
    """Tests for NATS event integration in DM training."""

    def test_publish_event_method_exists(self) -> None:
        config = DMConfig(episodes_per_session=1)
        with patch("agi.training.dungeon_master.EpisodicMemory", None):
            dm = DungeonMaster(config)
        assert hasattr(dm, "_publish_event")

    def test_publish_event_noop_without_nats(self) -> None:
        config = DMConfig(episodes_per_session=1)
        with patch("agi.training.dungeon_master.EpisodicMemory", None):
            dm = DungeonMaster(config)
        dm._nats = None
        # Should not raise
        dm._publish_event("test.topic", {"key": "value"})

    def test_session_publishes_events(self) -> None:
        config = DMConfig(episodes_per_session=1)
        with patch("agi.training.dungeon_master.EpisodicMemory", None):
            dm = DungeonMaster(config)

        published = []
        dm._nats = MagicMock()
        dm._nats.publish = MagicMock(return_value=None)

        # Mock async publish

        async def mock_pub(topic, payload):
            published.append((topic, payload))

        dm._nats.publish = mock_pub

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"superego_score": 7, "id_score": 6, '
                        '"synthesis_score": 7, "feedback": "Good."}'
                    }
                }
            ]
        }

        with (
            patch("agi.training.dungeon_master._PANTHEON_CASES", []),
            patch("agi.training.dungeon_master.ERISML_AVAILABLE", False),
            patch("agi.training.dungeon_master.requests.post") as mock_post,
        ):
            mock_post.return_value = mock_resp
            dm.run_session(episodes=1, difficulty=1)

        # Should have published: session_start, episode_complete, session_complete
        topics = [p[0] for p in published]
        assert "agi.training.progress" in topics
        assert "agi.training.result" in topics


class TestDMConfigNATS:
    """Tests for DM config NATS URL."""

    def test_default_nats_url(self) -> None:
        config = DMConfig()
        assert config.nats_url == "nats://localhost:4222"

    def test_custom_nats_url(self) -> None:
        config = DMConfig(nats_url="nats://custom:4222")
        assert config.nats_url == "nats://custom:4222"
