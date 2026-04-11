# AGI-HPC Ebbinghaus Forgetting Tests
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agi.memory.episodic.forgetting import (
    ForgettingConfig,
    boost_recall,
    compute_retention,
    compute_stability,
    episode_retention,
    is_pruneable,
)


@pytest.fixture
def config():
    return ForgettingConfig()


class TestStability:
    def test_base_stability(self, config):
        s = compute_stability(0.0, 0, config)
        assert s == config.base_half_life_hours

    def test_quality_increases_stability(self, config):
        s_low = compute_stability(0.0, 0, config)
        s_high = compute_stability(1.0, 0, config)
        assert s_high > s_low

    def test_recall_increases_stability(self, config):
        s0 = compute_stability(0.5, 0, config)
        s3 = compute_stability(0.5, 3, config)
        assert s3 > s0


class TestRetention:
    def test_fresh_is_one(self):
        assert compute_retention(0.0, 24.0) == 1.0

    def test_decays_over_time(self):
        r1 = compute_retention(1.0, 24.0)
        r24 = compute_retention(24.0, 24.0)
        r72 = compute_retention(72.0, 24.0)
        assert 0 < r72 < r24 < r1 < 1.0

    def test_higher_stability_slower_decay(self):
        r_low = compute_retention(24.0, 12.0)
        r_high = compute_retention(24.0, 48.0)
        assert r_high > r_low

    def test_zero_stability(self):
        assert compute_retention(1.0, 0.0) == 0.0


class TestEpisodeRetention:
    def test_recent_episode(self, config):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(minutes=5)
        r = episode_retention(ts, 0.5, {}, now, config)
        assert r > 0.9

    def test_old_episode(self, config):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=7)
        r = episode_retention(ts, 0.5, {}, now, config)
        assert r < 0.5

    def test_recalled_episode_decays_slower(self, config):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=3)
        r_no_recall = episode_retention(ts, 0.5, {}, now, config)
        r_recalled = episode_retention(ts, 0.5, {"recall_count": 5}, now, config)
        assert r_recalled > r_no_recall


class TestPruneable:
    def test_fresh_not_pruneable(self, config):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(minutes=1)
        assert not is_pruneable(ts, 0.5, {}, now, config)

    def test_old_is_pruneable(self, config):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=30)
        assert is_pruneable(ts, 0.0, {}, now, config)

    def test_high_quality_survives_longer(self, config):
        now = datetime.now(timezone.utc)
        ts = now - timedelta(days=10)
        low_q = is_pruneable(ts, 0.0, {}, now, config)
        high_q = is_pruneable(ts, 1.0, {}, now, config)
        # low quality pruned, high quality might survive
        assert low_q or not high_q


class TestBoostRecall:
    def test_increments_count(self):
        meta = {"recall_count": 2}
        updated = boost_recall(meta)
        assert updated["recall_count"] == 3

    def test_starts_at_zero(self):
        updated = boost_recall({})
        assert updated["recall_count"] == 1

    def test_preserves_other_fields(self):
        meta = {"recall_count": 1, "other": "data"}
        updated = boost_recall(meta)
        assert updated["other"] == "data"
