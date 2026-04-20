"""Unit tests for agi.metacognition.dissatisfaction_events (Gap Mapping Phase 2).

Covers schema validation, event construction from a detector signal,
append/read round-trips, dedup gate (one event per conversation), the
tolerant-reader behavior, and helper queries used by the aggregator.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from agi.metacognition.dissatisfaction_events import (
    REQUIRED_FIELDS,
    EventValidationError,
    append_event,
    conversation_has_event,
    event_exists,
    events_for_topic_key,
    iter_events,
    make_event,
    new_event_id,
    validate_event,
)

# ── fixtures / helpers ───────────────────────────────────────────


@dataclass
class _StubSignal:
    verdict: str = "unsatisfied"
    topic: str = "why is matrix rank not factorization"
    signal_turns: list[int] = None  # type: ignore[assignment]
    rationale: str = "user corrected assistant twice"
    score: float = 0.85
    detector_model: str = "qwen3"
    detector_version: str = "gap-det-0.1.0"

    def __post_init__(self) -> None:
        if self.signal_turns is None:
            self.signal_turns = [2, 4]


def _base_event(**overrides) -> dict:
    now = int(time.time())
    rec = {
        "event_id": "conv-abc123-sig",
        "conversation_id": "abc123",
        "topic": "why is matrix rank not factorization",
        "topic_key": "why-is-matrix-rank-not-factorization",
        "signal_turns": [2, 4],
        "rationale": "user corrected assistant twice",
        "score": 0.85,
        "ts": now,
        "detector_model": "qwen3",
        "detector_version": "gap-det-0.1.0",
    }
    rec.update(overrides)
    return rec


# ── validation ───────────────────────────────────────────────────


def test_valid_event_passes():
    validate_event(_base_event())


def test_missing_required_field_rejected():
    for field in REQUIRED_FIELDS:
        rec = _base_event()
        del rec[field]
        with pytest.raises(EventValidationError, match=field):
            validate_event(rec)


def test_non_dict_rejected():
    with pytest.raises(EventValidationError, match="must be a dict"):
        validate_event("not a dict")  # type: ignore[arg-type]


def test_empty_identity_fields_rejected():
    for f in (
        "event_id",
        "conversation_id",
        "topic_key",
        "detector_model",
        "detector_version",
    ):
        with pytest.raises(EventValidationError, match=f):
            validate_event(_base_event(**{f: ""}))


def test_event_id_unsafe_chars_rejected():
    with pytest.raises(EventValidationError, match="event_id"):
        validate_event(_base_event(event_id="has spaces and $"))


def test_signal_turns_must_be_ints():
    with pytest.raises(EventValidationError, match="signal_turns"):
        validate_event(_base_event(signal_turns=[1, "two", 3]))
    with pytest.raises(EventValidationError, match="signal_turns"):
        validate_event(_base_event(signal_turns="not a list"))


def test_score_type_and_range():
    with pytest.raises(EventValidationError, match="score"):
        validate_event(_base_event(score="high"))
    with pytest.raises(EventValidationError, match="range"):
        validate_event(_base_event(score=1.5))
    with pytest.raises(EventValidationError, match="range"):
        validate_event(_base_event(score=-0.1))
    # Boundaries OK
    validate_event(_base_event(score=0.0))
    validate_event(_base_event(score=1.0))


def test_ts_must_be_positive_int():
    with pytest.raises(EventValidationError, match="ts"):
        validate_event(_base_event(ts=0))
    with pytest.raises(EventValidationError, match="ts"):
        validate_event(_base_event(ts=-1))
    with pytest.raises(EventValidationError, match="ts"):
        validate_event(_base_event(ts="now"))


# ── make_event ───────────────────────────────────────────────────


def test_make_event_from_signal_produces_valid_record():
    sig = _StubSignal()
    ev = make_event(
        signal=sig,
        conversation_id="abc123",
        topic_key="matrix-rank",
        now=1_700_000_000,
    )
    assert ev["event_id"] == "conv-abc123-sig"
    assert ev["conversation_id"] == "abc123"
    assert ev["topic"] == sig.topic
    assert ev["topic_key"] == "matrix-rank"
    assert ev["signal_turns"] == [2, 4]
    assert ev["score"] == 0.85
    assert ev["detector_model"] == "qwen3"
    assert ev["detector_version"] == "gap-det-0.1.0"
    assert ev["ts"] == 1_700_000_000
    validate_event(ev)  # must pass its own validator


def test_make_event_explicit_id_wins():
    sig = _StubSignal()
    ev = make_event(
        signal=sig,
        conversation_id="abc",
        topic_key="x",
        event_id="custom-handle",
    )
    assert ev["event_id"] == "custom-handle"


def test_make_event_rejects_missing_context():
    sig = _StubSignal()
    with pytest.raises(EventValidationError, match="conversation_id"):
        make_event(signal=sig, conversation_id="", topic_key="x")
    with pytest.raises(EventValidationError, match="topic_key"):
        make_event(signal=sig, conversation_id="abc", topic_key="")


def test_new_event_id_uuid_like():
    a, b = new_event_id(), new_event_id()
    assert a != b
    assert a.startswith("evt-")
    assert len(a) > len("evt-")


# ── append / read roundtrip ─────────────────────────────────────


def test_append_and_iter_roundtrip(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    ev = _base_event()
    assert append_event(ev, path=p) is True
    events = list(iter_events(path=p))
    assert len(events) == 1
    assert events[0]["event_id"] == ev["event_id"]


def test_append_validates_and_raises(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    with pytest.raises(EventValidationError):
        append_event(_base_event(score=2.0), path=p)
    assert not p.exists() or p.read_text() == ""


def test_iter_events_missing_file_is_empty(tmp_path: Path):
    assert list(iter_events(path=tmp_path / "nope.jsonl")) == []


def test_iter_events_since_filter(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    append_event(_base_event(event_id="e1", conversation_id="c1", ts=100), path=p)
    append_event(_base_event(event_id="e2", conversation_id="c2", ts=200), path=p)
    append_event(_base_event(event_id="e3", conversation_id="c3", ts=300), path=p)
    got = [e["event_id"] for e in iter_events(since=200, path=p)]
    assert got == ["e2", "e3"]


def test_iter_events_tolerates_corrupt_lines(tmp_path: Path, caplog):
    p = tmp_path / "events.jsonl"
    append_event(_base_event(event_id="e1", conversation_id="c1"), path=p)
    with open(p, "a") as f:
        f.write("not-json\n")
        f.write(
            json.dumps({"event_id": "bad", "conversation_id": "c-bad"}) + "\n"
        )  # missing fields
    append_event(_base_event(event_id="e2", conversation_id="c2"), path=p)

    caplog.set_level(logging.WARNING, logger="metacognition.dissatisfaction_events")
    events = list(iter_events(path=p))
    assert [e["event_id"] for e in events] == ["e1", "e2"]
    assert sum("invalid_event_skipped" in m for m in caplog.messages) >= 2


# ── dedup gate ─────────────────────────────────────────────────


def test_dedup_rejects_second_event_same_conversation(tmp_path: Path, caplog):
    p = tmp_path / "events.jsonl"
    assert (
        append_event(_base_event(event_id="e1", conversation_id="c1"), path=p) is True
    )

    caplog.set_level(logging.WARNING, logger="metacognition.dissatisfaction_events")
    assert (
        append_event(
            _base_event(
                event_id="e2-diff", conversation_id="c1", topic_key="different"
            ),
            path=p,
        )
        is False
    )
    assert any("event_dedup_rejected" in m for m in caplog.messages)

    # File still has exactly one event
    events = list(iter_events(path=p))
    assert len(events) == 1
    assert events[0]["event_id"] == "e1"


def test_dedup_allows_different_conversations(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    assert (
        append_event(_base_event(event_id="e1", conversation_id="c1"), path=p) is True
    )
    assert (
        append_event(_base_event(event_id="e2", conversation_id="c2"), path=p) is True
    )
    events = list(iter_events(path=p))
    assert len(events) == 2


# ── helpers ─────────────────────────────────────────────────────


def test_event_exists(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    append_event(_base_event(event_id="target-id", conversation_id="c1"), path=p)
    assert event_exists("target-id", path=p) is True
    assert event_exists("missing", path=p) is False
    assert event_exists("", path=p) is False


def test_conversation_has_event(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    append_event(_base_event(conversation_id="alice"), path=p)
    assert conversation_has_event("alice", path=p) is True
    assert conversation_has_event("bob", path=p) is False


def test_events_for_topic_key_ordered_by_ts(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    append_event(
        _base_event(event_id="a", conversation_id="c1", topic_key="k", ts=300), path=p
    )
    append_event(
        _base_event(event_id="b", conversation_id="c2", topic_key="k", ts=100), path=p
    )
    append_event(
        _base_event(event_id="c", conversation_id="c3", topic_key="other", ts=200),
        path=p,
    )
    hits = events_for_topic_key("k", path=p)
    assert [h["event_id"] for h in hits] == ["b", "a"]  # sorted ascending


def test_events_for_topic_key_empty_returns_empty(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    assert events_for_topic_key("k", path=p) == []
    assert events_for_topic_key("", path=p) == []


# ── path override ──────────────────────────────────────────────


def test_env_override_default_path(monkeypatch, tmp_path: Path):
    """DISSATISFACTION_EVENTS_PATH controls the default. Set it before
    the module caches DEFAULT_PATH in tests that rely on the default."""
    monkeypatch.setenv("DISSATISFACTION_EVENTS_PATH", str(tmp_path / "custom.jsonl"))
    # Module's DEFAULT_PATH is evaluated at import time, so just patch it.
    import agi.metacognition.dissatisfaction_events as ev_mod

    monkeypatch.setattr(ev_mod, "DEFAULT_PATH", tmp_path / "custom.jsonl")
    append_event(_base_event())  # no explicit path — uses DEFAULT_PATH
    assert (tmp_path / "custom.jsonl").exists()
