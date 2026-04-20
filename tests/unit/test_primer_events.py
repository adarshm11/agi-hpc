"""Unit tests for agi.primer.events (append + aggregate + tail)."""

from __future__ import annotations

import json
from pathlib import Path

from agi.primer.events import aggregate, append, bucket_edges, tail_lines


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_append_writes_one_line_per_call(tmp_path: Path):
    p = tmp_path / "events.jsonl"
    append(
        task=167,
        expert="kimi",
        ok=True,
        latency_s=42.1,
        verify_pass=True,
        published=True,
        path=p,
    )
    append(
        task=168,
        expert="glm-4.7",
        ok=False,
        latency_s=300.0,
        verify_pass=False,
        published=False,
        error="timeout after 300.0s",
        path=p,
    )
    lines = _lines(p)
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["task"] == 167
    assert r0["expert"] == "kimi"
    assert r0["ok"] is True
    assert r0["verify_pass"] is True
    assert r0["published"] is True
    r1 = json.loads(lines[1])
    assert r1["ok"] is False
    assert r1["error"].startswith("timeout")


def test_aggregate_counts_and_buckets():
    buckets = bucket_edges()
    assert buckets[0] == "0-30s"
    assert buckets[-1].startswith("≥300s")
    raw = [
        json.dumps(r)
        for r in [
            # kimi: 2 pass (fast, mid), 1 fail (slow), 1 error
            {
                "ts": 1,
                "task": 1,
                "expert": "kimi",
                "ok": True,
                "latency_s": 15.0,
                "verify_pass": True,
                "published": True,
            },
            {
                "ts": 2,
                "task": 2,
                "expert": "kimi",
                "ok": True,
                "latency_s": 45.0,
                "verify_pass": True,
                "published": False,
            },
            {
                "ts": 3,
                "task": 3,
                "expert": "kimi",
                "ok": True,
                "latency_s": 200.0,
                "verify_pass": False,
                "published": False,
            },
            {
                "ts": 4,
                "task": 4,
                "expert": "kimi",
                "ok": False,
                "latency_s": 300.0,
                "verify_pass": False,
                "published": False,
                "error": "timeout",
            },
            # glm: 1 pass, 1 fail
            {
                "ts": 5,
                "task": 5,
                "expert": "glm-4.7",
                "ok": True,
                "latency_s": 90.0,
                "verify_pass": True,
                "published": True,
            },
            {
                "ts": 6,
                "task": 6,
                "expert": "glm-4.7",
                "ok": True,
                "latency_s": 90.0,
                "verify_pass": False,
                "published": False,
            },
        ]
    ]
    agg = aggregate(raw)
    k = agg["per_expert"]["kimi"]
    assert k["calls"] == 4
    assert k["verify_pass"] == 2
    assert k["verify_fail"] == 1
    assert k["errors"] == 1
    # buckets: 0-30 | 30-60 | 60-120 | 120-240 | 240-300 | overflow
    # kimi lands in 0-30, 30-60, 120-240, overflow (from the error)
    assert k["latency_buckets"] == [1, 1, 0, 1, 0, 1]
    g = agg["per_expert"]["glm-4.7"]
    assert g["calls"] == 2
    assert g["latency_buckets"][2] == 2  # both at 90s → 60-120 bucket
    assert agg["published"] == 2
    assert agg["bucket_labels"] == buckets


def test_aggregate_skips_malformed_lines():
    raw = [
        "not json",
        "",
        json.dumps(
            {
                "ts": 1,
                "task": 1,
                "expert": "kimi",
                "ok": True,
                "latency_s": 10.0,
                "verify_pass": True,
                "published": False,
            }
        ),
    ]
    agg = aggregate(raw)
    assert agg["per_expert"]["kimi"]["calls"] == 1


def test_tail_lines_returns_empty_for_missing_file(tmp_path: Path):
    assert tail_lines(tmp_path / "nope.jsonl") == []


def test_tail_lines_returns_last_n(tmp_path: Path):
    p = tmp_path / "e.jsonl"
    with open(p, "w") as f:
        for i in range(1000):
            f.write(f'{{"i":{i}}}\n')
    got = tail_lines(p, max_lines=10)
    assert len(got) <= 10
    # the last line in the file should be the last returned
    last = json.loads(got[-1])
    assert last["i"] == 999
