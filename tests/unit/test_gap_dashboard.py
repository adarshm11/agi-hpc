"""Unit tests for the Phase 5 Gap Mapping dashboard plumbing.

Covers ``graph.summary(source_filter=...)`` and
``dissatisfaction_events.recent_events()`` — the two queries the
``/api/ukg/status`` endpoint consumes.

Spec §5.1 + AC7 — dashboard ranking is based on node-level
signal_count and last_signal_at fields, never on evidence[] scans.
"""

from __future__ import annotations

from pathlib import Path

from agi.knowledge.graph import summary, upsert_node
from agi.metacognition.dissatisfaction_events import (
    append_event,
    recent_events,
)


def _seed_dissatisfaction(
    path: Path, *, topic_key: str, signal_count: int, last_ts: int
) -> None:
    """Seed a dissatisfaction-sourced gap node with aggregate fields set."""
    upsert_node(
        id=f"gap_{topic_key}",
        type="gap",
        topic=topic_key.replace("-", " "),
        title=f"[gap] {topic_key}",
        source="dissatisfaction",
        evidence=[f"event:evt-{topic_key}"],
        now=last_ts,
        extra={
            "signal_count": signal_count,
            "first_signal_at": last_ts - 3600,
            "last_signal_at": last_ts,
        },
        path=path,
    )


def _base_event(**overrides) -> dict:
    rec = {
        "event_id": "evt-base",
        "conversation_id": "c-base",
        "topic": "base",
        "topic_key": "base",
        "signal_turns": [1],
        "rationale": "test",
        "score": 0.85,
        "ts": 100,
        "detector_model": "qwen3",
        "detector_version": "gap-det-0.1.0",
    }
    rec.update(overrides)
    return rec


# ── graph.summary(source_filter=...) ────────────────────────────


def test_summary_without_source_filter_unchanged(tmp_path: Path):
    """Baseline: no source_filter means no extra key — spec §5.1 says
    base counters are unaffected by the filter."""
    p = tmp_path / "g.jsonl"
    _seed_dissatisfaction(p, topic_key="k1", signal_count=3, last_ts=200)
    s = summary(path=p)
    assert "top_dissatisfaction_topics" not in s
    # Base counters still correct
    assert s["by_type"]["gap"] == 1
    assert s["total"] == 1


def test_summary_source_filter_adds_ranked_list(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    _seed_dissatisfaction(p, topic_key="low", signal_count=1, last_ts=100)
    _seed_dissatisfaction(p, topic_key="high", signal_count=5, last_ts=200)
    _seed_dissatisfaction(p, topic_key="mid", signal_count=3, last_ts=300)

    s = summary(path=p, source_filter="dissatisfaction")
    ranked = s["top_dissatisfaction_topics"]
    # Ordered by signal_count desc (primary), last_signal_at desc (tiebreaker)
    assert [r["topic_key"] for r in ranked] == ["high", "mid", "low"]
    assert ranked[0]["signal_count"] == 5
    assert ranked[0]["last_signal_at"] == 200


def test_summary_source_filter_ignores_other_sources(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    _seed_dissatisfaction(p, topic_key="diss", signal_count=2, last_ts=100)
    # Add a help-queue gap that should NOT appear in the filtered list
    upsert_node(
        id="gap_help",
        type="gap",
        topic="help topic",
        title="[gap] help",
        source="help_queue",
        path=p,
    )
    s = summary(path=p, source_filter="dissatisfaction")
    ranked = s["top_dissatisfaction_topics"]
    assert [r["topic_key"] for r in ranked] == ["diss"]


def test_summary_source_filter_tiebreaker_on_last_signal_at(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    # Same signal_count; differ on last_signal_at — newest wins
    _seed_dissatisfaction(p, topic_key="older", signal_count=3, last_ts=100)
    _seed_dissatisfaction(p, topic_key="newer", signal_count=3, last_ts=200)
    s = summary(path=p, source_filter="dissatisfaction")
    ranked = s["top_dissatisfaction_topics"]
    assert [r["topic_key"] for r in ranked] == ["newer", "older"]


def test_summary_source_filter_node_without_aggregates_still_sorts(tmp_path: Path):
    """A legacy dissatisfaction node written before Phase 3 added the
    aggregate fields: we should not crash, and it should rank last
    (signal_count defaults to 0)."""
    p = tmp_path / "g.jsonl"
    _seed_dissatisfaction(p, topic_key="modern", signal_count=2, last_ts=200)
    # Legacy node: no extras
    upsert_node(
        id="gap_legacy",
        type="gap",
        topic="legacy",
        title="[gap] legacy",
        source="dissatisfaction",
        path=p,
    )
    s = summary(path=p, source_filter="dissatisfaction")
    ranked = s["top_dissatisfaction_topics"]
    assert ranked[0]["topic_key"] == "modern"
    # legacy is present but at the bottom
    assert "legacy" in {r["topic_key"] for r in ranked}
    legacy = next(r for r in ranked if r["topic_key"] == "legacy")
    assert legacy["signal_count"] == 0


def test_summary_source_filter_respects_top_topics_limit(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    for i in range(10):
        _seed_dissatisfaction(
            p, topic_key=f"k{i:02d}", signal_count=10 - i, last_ts=100 + i
        )
    s = summary(path=p, source_filter="dissatisfaction", top_topics=3)
    assert len(s["top_dissatisfaction_topics"]) == 3


def test_summary_source_filter_empty_graph(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    s = summary(path=p, source_filter="dissatisfaction")
    assert s["top_dissatisfaction_topics"] == []


def test_summary_key_generalizes_to_any_source(tmp_path: Path):
    """source_filter is generic. Try another source name."""
    p = tmp_path / "g.jsonl"
    upsert_node(
        id="gap_q1",
        type="gap",
        topic="q1",
        title="[gap] q1",
        source="help_queue",
        path=p,
        extra={"signal_count": 4, "last_signal_at": 100},
    )
    s = summary(path=p, source_filter="help_queue")
    assert "top_help_queue_topics" in s
    assert s["top_help_queue_topics"][0]["topic_key"] == "q1"


# ── recent_events(n) ────────────────────────────────────────────


def test_recent_events_empty_file(tmp_path: Path):
    assert recent_events(5, path=tmp_path / "nope.jsonl") == []


def test_recent_events_orders_newest_first(tmp_path: Path):
    p = tmp_path / "e.jsonl"
    append_event(_base_event(event_id="e1", conversation_id="c1", ts=100), path=p)
    append_event(_base_event(event_id="e2", conversation_id="c2", ts=300), path=p)
    append_event(_base_event(event_id="e3", conversation_id="c3", ts=200), path=p)
    got = recent_events(5, path=p)
    assert [e["event_id"] for e in got] == ["e2", "e3", "e1"]


def test_recent_events_respects_limit(tmp_path: Path):
    p = tmp_path / "e.jsonl"
    for i in range(10):
        append_event(
            _base_event(event_id=f"e{i}", conversation_id=f"c{i}", ts=100 + i),
            path=p,
        )
    got = recent_events(3, path=p)
    assert len(got) == 3
    assert [e["event_id"] for e in got] == ["e9", "e8", "e7"]


def test_recent_events_zero_limit_returns_empty(tmp_path: Path):
    p = tmp_path / "e.jsonl"
    append_event(_base_event(), path=p)
    assert recent_events(0, path=p) == []
