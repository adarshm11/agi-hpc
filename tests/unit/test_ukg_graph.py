"""Unit tests for agi.knowledge.graph — Unified Knowledge Graph v1 invariants.

Covers the recommended-first-tests list from the spec plus:
- normalization round-trips
- evidence union ordering
- unknown-source warning
- context eligibility trust gate
- query_nodes filters + sort
- compact preserves latest-per-id
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from agi.knowledge.graph import (
    SCHEMA_VERSION,
    ValidationError,
    append_node,
    compact,
    context_reader_mode,
    get_node,
    is_context_eligible,
    iter_records,
    load_latest,
    merge_evidence,
    normalize_tags,
    normalize_topic_key,
    query_nodes,
    summary,
    upsert_node,
    validate_record,
)


def _base_record(**overrides) -> dict:
    """Minimal valid filled+verified record for tests to mutate."""
    now = int(time.time())
    rec = {
        "schema_version": SCHEMA_VERSION,
        "id": "sensei_task_167",
        "type": "filled",
        "status": "active",
        "topic": "symmetry completion",
        "topic_key": "symmetry-completion",
        "tags": ["arc", "transformation"],
        "title": "Count distinct colors in grid",
        "body_ref": "wiki/sensei_task_167.md",
        "verified": True,
        "verified_at": now,
        "source": "primer",
        "created_at": now,
        "last_touched_at": now,
        "evidence": ["conv:abc123", "help:t167"],
    }
    rec.update(overrides)
    return rec


# ── normalization ─────────────────────────────────────────────────


def test_normalize_topic_key_basic():
    assert normalize_topic_key("Symmetry Completion") == "symmetry-completion"
    assert (
        normalize_topic_key("  count  distinct__colors!  ") == "count-distinct-colors"
    )
    assert normalize_topic_key("") == ""


def test_normalize_tags_dedupes_preserving_order():
    assert normalize_tags(["ARC", "arc", "Transformation", "arc"]) == [
        "arc",
        "transformation",
    ]
    # non-strings dropped, whitespace trimmed, underscores → hyphens
    assert normalize_tags(["foo_bar", " baz ", None, 7, "foo-bar"]) == [  # type: ignore
        "foo-bar",
        "baz",
    ]


def test_merge_evidence_first_seen_stable():
    old = ["conv:a", "help:t1"]
    new = ["help:t1", "wiki:x", "conv:b"]
    assert merge_evidence(old, new) == ["conv:a", "help:t1", "wiki:x", "conv:b"]


# ── validation ────────────────────────────────────────────────────


def test_valid_filled_record_passes():
    validate_record(_base_record())


def test_valid_gap_record_passes():
    validate_record(
        _base_record(
            id="gap_symmetry_completion",
            type="gap",
            verified=False,
            verified_at=None,
            body_ref=None,
        )
    )


def test_verified_requires_filled():
    with pytest.raises(ValidationError, match="type='filled'"):
        validate_record(_base_record(type="gap", body_ref=None))


def test_verified_false_requires_null_verified_at():
    with pytest.raises(ValidationError, match="verified_at=null"):
        validate_record(_base_record(verified=False))


def test_filled_requires_body_ref():
    with pytest.raises(ValidationError, match="body_ref"):
        validate_record(_base_record(body_ref=None))


def test_bad_schema_version_rejected():
    with pytest.raises(ValidationError, match="schema_version"):
        validate_record(_base_record(schema_version=0))


def test_bad_type_enum_rejected():
    with pytest.raises(ValidationError, match="type must be one of"):
        validate_record(_base_record(type="unknown"))


def test_bad_status_enum_rejected():
    with pytest.raises(ValidationError, match="status must be one of"):
        validate_record(_base_record(status="live"))


def test_created_after_last_touched_rejected():
    now = int(time.time())
    with pytest.raises(ValidationError, match="created_at must be <="):
        validate_record(_base_record(created_at=now + 10, last_touched_at=now))


def test_tags_must_be_unique():
    with pytest.raises(ValidationError, match="unique"):
        validate_record(_base_record(tags=["arc", "arc"]))


def test_unknown_source_warns_but_accepted(caplog):
    caplog.set_level(logging.WARNING, logger="knowledge.graph")
    rec = _base_record(source="erebus_internal_daemon")
    validate_record(rec)
    assert any("unknown_source" in m for m in caplog.messages)


def test_empty_source_rejected():
    with pytest.raises(ValidationError, match="source"):
        validate_record(_base_record(source=""))


# ── append / read / materialize ───────────────────────────────────


def test_append_and_reload_roundtrip(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    append_node(_base_record(), path=p)
    latest = load_latest(path=p)
    assert latest["sensei_task_167"]["topic"] == "symmetry completion"


def test_append_validates(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    with pytest.raises(ValidationError):
        append_node(_base_record(type="invalid"), path=p)
    assert not p.exists() or p.read_text() == ""


def test_later_last_touched_wins(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    t0 = int(time.time())
    append_node(_base_record(last_touched_at=t0, title="old"), path=p)
    append_node(_base_record(last_touched_at=t0 + 10, title="new"), path=p)
    got = get_node("sensei_task_167", path=p)
    assert got is not None and got["title"] == "new"


def test_tie_on_timestamp_later_line_wins(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    t0 = int(time.time())
    append_node(_base_record(last_touched_at=t0, title="first"), path=p)
    append_node(_base_record(last_touched_at=t0, title="second"), path=p)
    got = get_node("sensei_task_167", path=p)
    assert got is not None and got["title"] == "second"


def test_invalid_line_skipped_with_warning(tmp_path: Path, caplog):
    p = tmp_path / "graph.jsonl"
    append_node(_base_record(), path=p)
    with open(p, "a") as f:
        f.write("not-json\n")
        f.write(json.dumps({"schema_version": 1, "id": "x"}) + "\n")
    caplog.set_level(logging.WARNING, logger="knowledge.graph")
    recs = list(iter_records(path=p))
    assert len(recs) == 1
    assert any("invalid_record_skipped" in m for m in caplog.messages)


# ── promotion / upsert ────────────────────────────────────────────


def test_gap_to_filled_promotion_preserves_created_at(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    t0 = int(time.time())
    upsert_node(
        id="sensei_task_167",
        type="gap",
        topic="symmetry completion",
        title="open: task 167",
        source="help_queue",
        evidence=["help:t167"],
        now=t0,
        path=p,
    )
    upsert_node(
        id="sensei_task_167",
        type="filled",
        topic="symmetry completion",
        title="Count distinct colors in grid",
        body_ref="wiki/sensei_task_167.md",
        verified=True,
        source="primer",
        evidence=["conv:abc"],
        now=t0 + 3600,
        path=p,
    )
    got = get_node("sensei_task_167", path=p)
    assert got is not None
    assert got["type"] == "filled"
    assert got["verified"] is True
    assert got["created_at"] == t0
    assert got["last_touched_at"] == t0 + 3600
    assert got["evidence"] == ["help:t167", "conv:abc"]
    assert got["verified_at"] == t0 + 3600


def test_filled_to_gap_transition_forbidden(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    upsert_node(
        id="n1",
        type="filled",
        topic="t",
        title="a",
        body_ref="wiki/n1.md",
        verified=True,
        source="primer",
        path=p,
    )
    with pytest.raises(ValidationError, match="filled→"):
        upsert_node(
            id="n1",
            type="gap",
            topic="t",
            title="a",
            source="manual",
            path=p,
        )


def test_upsert_preserves_verified_at_across_updates(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    t0 = int(time.time())
    upsert_node(
        id="n1",
        type="filled",
        topic="t",
        title="a",
        body_ref="wiki/n1.md",
        verified=True,
        source="primer",
        now=t0,
        path=p,
    )
    upsert_node(
        id="n1",
        type="filled",
        topic="t",
        title="a-refined",
        body_ref="wiki/n1.md",
        verified=True,
        source="primer",
        now=t0 + 100,
        path=p,
    )
    got = get_node("n1", path=p)
    assert got["verified_at"] == t0
    assert got["last_touched_at"] == t0 + 100


# ── context eligibility ──────────────────────────────────────────


def test_is_context_eligible_happy_path(tmp_path: Path):
    body = tmp_path / "wiki" / "sensei_task_167.md"
    body.parent.mkdir()
    body.write_text("# note\n")
    node = _base_record(body_ref="sensei_task_167.md")
    assert is_context_eligible(node, wiki_root=tmp_path / "wiki") is True


def test_is_context_eligible_rejects_gap():
    node = _base_record(type="gap", body_ref=None, verified=False, verified_at=None)
    assert is_context_eligible(node) is False


def test_is_context_eligible_rejects_unverified():
    node = _base_record(verified=False, verified_at=None)
    assert is_context_eligible(node) is False


def test_is_context_eligible_rejects_archived():
    node = _base_record(status="archived")
    assert is_context_eligible(node) is False


def test_is_context_eligible_warns_on_missing_body(tmp_path: Path, caplog):
    node = _base_record(body_ref=str(tmp_path / "nope.md"))
    caplog.set_level(logging.WARNING, logger="knowledge.graph")
    assert is_context_eligible(node) is False
    assert any("filled_excluded_missing_body" in m for m in caplog.messages)


# ── query ─────────────────────────────────────────────────────────


def test_query_nodes_filters_and_sorts(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    t0 = int(time.time())
    upsert_node(
        id="n1",
        type="gap",
        topic="a",
        title="gap a",
        source="help_queue",
        now=t0,
        path=p,
    )
    upsert_node(
        id="n2",
        type="filled",
        topic="a",
        title="a",
        body_ref="wiki/n2.md",
        verified=True,
        source="primer",
        now=t0 + 1,
        path=p,
    )
    upsert_node(
        id="n3",
        type="filled",
        topic="b",
        title="b",
        body_ref="wiki/n3.md",
        verified=True,
        source="primer",
        now=t0 + 2,
        path=p,
    )
    got = query_nodes(type="filled", topic_key="a", path=p)
    assert [n["id"] for n in got] == ["n2"]
    got = query_nodes(type="filled", limit=1, path=p)
    assert got[0]["id"] == "n3"
    got = query_nodes(type="filled", sort="last_touched_asc", path=p)
    assert [n["id"] for n in got] == ["n2", "n3"]


def test_query_unknown_sort_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="sort must be"):
        query_nodes(sort="random", path=tmp_path / "g.jsonl")


# ── compaction ────────────────────────────────────────────────────


def test_compact_preserves_materialized_view(tmp_path: Path):
    p = tmp_path / "graph.jsonl"
    t0 = int(time.time())
    for i in range(5):
        upsert_node(
            id="n1",
            type="filled",
            topic="t",
            title=f"title-{i}",
            body_ref="wiki/n1.md",
            verified=True,
            source="primer",
            now=t0 + i,
            path=p,
        )
    upsert_node(
        id="n2",
        type="gap",
        topic="t",
        title="gap",
        source="help_queue",
        now=t0 + 10,
        path=p,
    )
    pre = load_latest(path=p)
    n_compacted = compact(path=p)
    assert n_compacted == 2
    post = load_latest(path=p)
    assert pre == post
    assert len([ln for ln in p.read_text().splitlines() if ln.strip()]) == 2


# ── config flag ──────────────────────────────────────────────────


def test_context_reader_mode_default_wiki(monkeypatch):
    monkeypatch.delenv("EREBUS_CONTEXT_READER", raising=False)
    assert context_reader_mode() == "wiki"


def test_context_reader_mode_env_override(monkeypatch):
    monkeypatch.setenv("EREBUS_CONTEXT_READER", "graph")
    assert context_reader_mode() == "graph"


def test_context_reader_mode_invalid_env_falls_back(monkeypatch, caplog):
    monkeypatch.setenv("EREBUS_CONTEXT_READER", "hybrid")
    caplog.set_level(logging.WARNING, logger="knowledge.graph")
    assert context_reader_mode() == "wiki"
    assert any("EREBUS_CONTEXT_READER" in m for m in caplog.messages)


# ── summary (Phase 5 dashboard aggregation) ──────────────────────


def _seed_summary_graph(p: Path, t0: int) -> None:
    """Helper: seed a graph with a mix of filled/gap/stub nodes and topics."""
    # Topic A: 2 filled, 1 gap → fill_rate 2/3
    upsert_node(
        id="n_a1",
        type="filled",
        topic="topic A",
        title="A one",
        body_ref="wiki/a1.md",
        verified=True,
        source="primer",
        now=t0,
        path=p,
    )
    upsert_node(
        id="n_a2",
        type="filled",
        topic="topic A",
        title="A two",
        body_ref="wiki/a2.md",
        verified=True,
        source="primer",
        now=t0 + 10,
        path=p,
    )
    upsert_node(
        id="n_a3",
        type="gap",
        topic="topic A",
        title="A three (gap)",
        source="help_queue",
        now=t0 + 5,
        path=p,
    )
    # Topic B: 3 gaps, 0 filled — zero fill-rate, should rank high by gap count
    for i in range(3):
        upsert_node(
            id=f"n_b{i}",
            type="gap",
            topic="topic B",
            title=f"B gap {i}",
            source="help_queue",
            now=t0 + 20 + i,
            path=p,
        )
    # Topic C: 1 filled, 0 gap — should not appear in top_topics_by_gap
    upsert_node(
        id="n_c1",
        type="filled",
        topic="topic C",
        title="C one",
        body_ref="wiki/c1.md",
        verified=True,
        source="primer",
        now=t0 + 30,
        path=p,
    )


def test_summary_counts_types_and_statuses(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    t0 = int(time.time())
    _seed_summary_graph(p, t0)
    s = summary(path=p)
    assert s["total"] == 7
    assert s["by_type"] == {"filled": 3, "gap": 4, "stub": 0}
    assert s["by_status"] == {"active": 7, "archived": 0}


def test_summary_fill_rate(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    t0 = int(time.time())
    _seed_summary_graph(p, t0)
    s = summary(path=p)
    # 3 filled / (3 filled + 4 gap) = 3/7 ≈ 0.429
    assert abs(s["fill_rate"] - 3 / 7) < 0.001


def test_summary_top_topics_by_gap_excludes_zero_gap(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    t0 = int(time.time())
    _seed_summary_graph(p, t0)
    s = summary(path=p)
    tops = s["top_topics_by_gap"]
    topic_keys = [r["topic_key"] for r in tops]
    # Topic B (3 gaps) first, Topic A (1 gap) second, Topic C excluded
    assert topic_keys == ["topic-b", "topic-a"]
    assert tops[0]["gaps"] == 3
    assert tops[0]["filled"] == 0
    assert tops[0]["fill_rate"] == 0.0
    assert tops[1]["gaps"] == 1
    assert tops[1]["filled"] == 2
    assert abs(tops[1]["fill_rate"] - 2 / 3) < 0.001


def test_summary_recent_fills_sorted_by_verified_at_desc(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    t0 = int(time.time())
    _seed_summary_graph(p, t0)
    s = summary(path=p)
    recent = s["recent_fills"]
    # Three filled nodes: C (t0+30), A2 (t0+10), A1 (t0)
    assert [r["id"] for r in recent] == ["n_c1", "n_a2", "n_a1"]
    assert recent[0]["title"] == "C one"
    # All recent fills should report verified_at
    assert all(r["verified_at"] is not None for r in recent)


def test_summary_top_topics_limit(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    t0 = int(time.time())
    # 12 topics with at least one gap each
    for i in range(12):
        upsert_node(
            id=f"g{i}",
            type="gap",
            topic=f"topic {i:02d}",
            title=f"gap {i}",
            source="help_queue",
            now=t0 + i,
            path=p,
        )
    s = summary(path=p, top_topics=5)
    assert len(s["top_topics_by_gap"]) == 5


def test_summary_recent_fills_limit(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    t0 = int(time.time())
    for i in range(12):
        upsert_node(
            id=f"f{i}",
            type="filled",
            topic="t",
            title=f"fill {i}",
            body_ref=f"wiki/f{i}.md",
            verified=True,
            source="primer",
            now=t0 + i,
            path=p,
        )
    s = summary(path=p, recent_fills=5)
    assert len(s["recent_fills"]) == 5


def test_summary_empty_graph(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    s = summary(path=p)
    assert s["total"] == 0
    assert s["by_type"] == {"filled": 0, "gap": 0, "stub": 0}
    assert s["by_status"] == {"active": 0, "archived": 0}
    assert s["fill_rate"] == 0.0
    assert s["top_topics_by_gap"] == []
    assert s["recent_fills"] == []
