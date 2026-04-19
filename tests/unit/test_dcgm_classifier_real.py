"""Integration-ish tests against real GV100 traces captured on Atlas.

Fixtures in ``tests/fixtures/dcgm_profiles/`` are actual
``scripts/collect_gpu_power_trace.py`` output from Atlas GPU 1
(Quadro GV100) on 2026-04-19. They cement the classifier's
behaviour against lived hardware, not just synthetic curves.

If the classifier's default thresholds need to be tuned as new
cards or workloads appear, these tests will catch regressions.
"""

from __future__ import annotations

import json
from pathlib import Path

from agi.safety.dcgm_classifier import classify_trace, profile_matches_compute_claim

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dcgm_profiles"


def _load(name: str) -> list[dict]:
    p = FIXTURES / f"{name}.jsonl"
    return json.loads(p.read_text())["samples"]


def test_real_idle_trace_classified_as_idle():
    """12s idle GV100 trace (~27-28 W, 0% util)."""
    r = classify_trace(_load("idle_baseline"))
    assert r.profile == "idle"
    assert r.confidence >= 0.5
    assert not profile_matches_compute_claim(r.profile)
    # Sanity: baseline is near the well-known GV100 idle draw
    assert 20 < r.features.avg_power_w < 40


def test_real_active_cupy_trace_classified_as_active_sustained():
    """12s cupy matmul — sustained 200-240 W, 90%+ util."""
    r = classify_trace(_load("active_cupy"))
    assert r.profile == "active_sustained"
    assert r.confidence >= 0.5
    assert profile_matches_compute_claim(r.profile)
    assert r.features.peak_power_w > 150
    assert r.features.sustained_pct > 50


def test_real_cached_replay_trace_classified_as_idle_not_active():
    """The attestation-critical test: a 'cached replay' (sleep, no work)
    MUST NOT be classified as active_* — that would let a malicious
    caller claim compute happened when it didn't."""
    r = classify_trace(_load("cached_replay"))
    assert r.profile == "idle"
    assert not profile_matches_compute_claim(r.profile)
    # The classifier must flag this as NOT a compute-claim match
    # even though returning True would be benign (just over-trusting)
    # — security matters.


def test_real_traces_separate_cleanly():
    """Active and idle avg_power should be well-separated on real
    hardware — if they overlap, our thresholds need retuning."""
    idle = classify_trace(_load("idle_baseline")).features
    active = classify_trace(_load("active_cupy")).features
    replay = classify_trace(_load("cached_replay")).features
    # Active should be ~7x idle's avg_power
    assert active.avg_power_w > 5 * idle.avg_power_w
    # Replay looks like idle, not like active
    assert abs(replay.avg_power_w - idle.avg_power_w) < 10
    assert active.avg_power_w - replay.avg_power_w > 100
