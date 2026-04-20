# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Phase 2 — sidecar event log for dissatisfaction signals.

Append-only JSONL at ``/archive/neurogolf/dissatisfaction_events.jsonl``
holding the raw audit trail that the UKG aggregate nodes point at via
``evidence: ["event:<event_id>", ...]``.

Same atomicity discipline as the UKG graph log: O_APPEND for
incremental writes, full-file rewrite for the (rare) repair path.

Per spec §1.3 + §4 the "one event per conversation" invariant is
enforced here — the aggregator's dedup story is `append_event`
rejecting a second record for the same ``conversation_id``. We do the
O(n) scan on write for v1 because event volume is low; a cached
seen-set can replace it when throughput warrants.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("metacognition.dissatisfaction_events")

DEFAULT_PATH = Path(
    os.environ.get(
        "DISSATISFACTION_EVENTS_PATH",
        "/archive/neurogolf/dissatisfaction_events.jsonl",
    )
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "event_id",
    "conversation_id",
    "topic",
    "topic_key",
    "signal_turns",
    "rationale",
    "score",
    "ts",
    "detector_model",
    "detector_version",
)

_EVENT_ID_SAFE = re.compile(r"^[A-Za-z0-9_\-:.]+$")


# ── validation ───────────────────────────────────────────────────


class EventValidationError(ValueError):
    """Raised when an event record violates schema invariants."""


def validate_event(event: Any) -> None:
    """Raise ``EventValidationError`` iff ``event`` violates v1 invariants.

    Does NOT check uniqueness / dedup — that's a write-time concern.
    """
    if not isinstance(event, dict):
        raise EventValidationError(f"event must be a dict, got {type(event).__name__}")
    for f in REQUIRED_FIELDS:
        if f not in event:
            raise EventValidationError(f"missing required field: {f!r}")

    for f in (
        "event_id",
        "conversation_id",
        "topic",
        "topic_key",
        "rationale",
        "detector_model",
        "detector_version",
    ):
        v = event[f]
        if not isinstance(v, str):
            raise EventValidationError(
                f"{f!r} must be a string, got {type(v).__name__}"
            )

    # Non-empty for identity fields
    for f in (
        "event_id",
        "conversation_id",
        "topic_key",
        "detector_model",
        "detector_version",
    ):
        if not event[f]:
            raise EventValidationError(f"{f!r} must be non-empty")

    if not _EVENT_ID_SAFE.match(event["event_id"]):
        raise EventValidationError(
            f"event_id contains unsafe characters: {event['event_id']!r}"
        )

    st = event["signal_turns"]
    if not isinstance(st, list) or not all(isinstance(i, int) for i in st):
        raise EventValidationError("signal_turns must be a list of ints")

    score = event["score"]
    if not isinstance(score, (int, float)):
        raise EventValidationError("score must be a number")
    if not (0.0 <= float(score) <= 1.0):
        raise EventValidationError(f"score out of range 0..1: {score}")

    ts = event["ts"]
    if not isinstance(ts, int) or ts <= 0:
        raise EventValidationError("ts must be a positive int (unix seconds)")


# ── event construction ───────────────────────────────────────────


def make_event(
    *,
    signal: Any,
    conversation_id: str,
    topic_key: str,
    event_id: str | None = None,
    now: int | None = None,
) -> dict:
    """Build an event record from a ``ConversationSignal`` + context.

    ``topic_key`` is supplied by the aggregator (which owns topic
    normalization per spec §1.4); the detector's signal carries only
    ``topic`` free-text.

    ``event_id`` defaults to a deterministic ``conv-<id>-sig`` handle
    so a retry for the same conversation is easy to spot. Callers may
    supply an explicit id (e.g. a UUID) if they want uniqueness even
    across conversation retries.
    """
    if not conversation_id:
        raise EventValidationError("conversation_id is required")
    if not topic_key:
        raise EventValidationError("topic_key is required")

    ts = int(now if now is not None else time.time())
    eid = event_id or f"conv-{conversation_id}-sig"

    record = {
        "event_id": eid,
        "conversation_id": conversation_id,
        "topic": getattr(signal, "topic", "") or "",
        "topic_key": topic_key,
        "signal_turns": list(getattr(signal, "signal_turns", []) or []),
        "rationale": getattr(signal, "rationale", "") or "",
        "score": float(getattr(signal, "score", 0.0) or 0.0),
        "ts": ts,
        "detector_model": getattr(signal, "detector_model", "") or "",
        "detector_version": getattr(signal, "detector_version", "") or "",
    }
    validate_event(record)
    return record


def new_event_id() -> str:
    """Random UUIDv4 handle for callers that want uniqueness independent
    of ``conversation_id``."""
    return f"evt-{uuid.uuid4().hex[:12]}"


# ── writer ───────────────────────────────────────────────────────


def _path(path: Path | None) -> Path:
    return Path(path) if path is not None else DEFAULT_PATH


def append_event(event: dict, *, path: Path | None = None) -> bool:
    """Validate, dedup-check, and append one event as a JSONL line.

    Returns:
      True  — event was appended.
      False — event was rejected by the conversation-id dedup gate.

    Raises ``EventValidationError`` on schema violation — a schema bug
    should be visible, not silently accepted, and the caller can then
    choose to log-and-continue rather than crash.
    """
    validate_event(event)
    target = _path(path)

    if conversation_has_event(event["conversation_id"], path=target):
        log.warning(
            "event_dedup_rejected: conversation_id=%s already has an event",
            event["conversation_id"],
        )
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(",", ":"), sort_keys=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return True


# ── readers ──────────────────────────────────────────────────────


def iter_events(
    *,
    since: int | None = None,
    path: Path | None = None,
) -> Iterator[dict]:
    """Yield every valid event in file order, optionally filtered by ``ts``.

    Invalid lines are skipped with a warning so a single corrupt record
    doesn't poison later readers. ``since`` is inclusive-from: events
    with ``ts >= since`` are yielded.
    """
    target = _path(path)
    if not target.exists():
        return
    with open(target, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception as e:
                log.warning(
                    "invalid_event_skipped: line=%d parse_error=%r",
                    lineno,
                    str(e)[:200],
                )
                continue
            try:
                validate_event(rec)
            except EventValidationError as e:
                log.warning(
                    "invalid_event_skipped: line=%d validation=%r id=%r",
                    lineno,
                    str(e)[:200],
                    (rec or {}).get("event_id"),
                )
                continue
            if since is not None and rec["ts"] < since:
                continue
            yield rec


def event_exists(event_id: str, *, path: Path | None = None) -> bool:
    """True iff any event with this ``event_id`` has been recorded."""
    if not event_id:
        return False
    return any(e.get("event_id") == event_id for e in iter_events(path=path))


def conversation_has_event(conversation_id: str, *, path: Path | None = None) -> bool:
    """True iff any event with this ``conversation_id`` has been recorded.

    This is the dedup gate enforced by ``append_event``. Exposed for
    the aggregator so it can make the same check before doing work
    (e.g. loading the graph) that would be wasted on a duplicate.
    """
    if not conversation_id:
        return False
    return any(
        e.get("conversation_id") == conversation_id for e in iter_events(path=path)
    )


def recent_events(n: int = 5, *, path: Path | None = None) -> list[dict]:
    """Return the ``n`` most recent events, newest first.

    Used by the dashboard's "Recent dissatisfaction events" row. For
    v1 volumes, reading the whole sidecar and sorting is fine; if the
    log grows large we can tail-seek in a later phase.
    """
    if n <= 0:
        return []
    evs = list(iter_events(path=path))
    evs.sort(key=lambda e: e.get("ts", 0), reverse=True)
    return evs[:n]


def events_for_topic_key(topic_key: str, *, path: Path | None = None) -> list[dict]:
    """Return all events whose ``topic_key`` matches — ordered by ``ts``.

    Used by the aggregator to rebuild a node's aggregate counters
    (``signal_count``, ``first_signal_at``, ``last_signal_at``) when
    an upsert needs them fresh. Also useful for the dashboard's
    "recent dissatisfaction events per topic" view.
    """
    if not topic_key:
        return []
    hits = [e for e in iter_events(path=path) if e.get("topic_key") == topic_key]
    hits.sort(key=lambda e: e.get("ts", 0))
    return hits
