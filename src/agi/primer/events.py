# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Append-only JSONL event log for Primer calls.

One line per expert response per tick. The dashboard reads the tail
to aggregate tasks-taught count, per-expert verify rate, and per-expert
latency histograms — data that survives a Primer restart (unlike the
in-memory ``HealthTracker`` rolling window).

Schema (one JSON object per line):

    {
      "ts":          float,   # unix seconds
      "task":        int,     # ARC task number
      "expert":      str,     # vMOE expert name (e.g. "kimi")
      "ok":          bool,    # call returned without error/timeout
      "latency_s":   float,
      "verify_pass": bool,    # code passed all train examples (false if !ok)
      "published":   bool,    # this response became sensei_task_NNN.md
      "error":       str      # empty unless !ok
    }

Writes use O_APPEND (atomic up to PIPE_BUF for small lines on POSIX) so
concurrent writers would interleave cleanly — though the Primer daemon is
single-process in practice. No rotation yet; event rate is ~tens per hour,
so size stays in the low-MB range for months.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("primer.events")

EVENTS_PATH = Path(
    os.environ.get("PRIMER_EVENTS_PATH", "/archive/neurogolf/primer_events.jsonl")
)


def append(
    *,
    task: int,
    expert: str,
    ok: bool,
    latency_s: float,
    verify_pass: bool,
    published: bool,
    error: str = "",
    path: Path | None = None,
) -> None:
    """Append one event. Swallows write errors — telemetry must never
    crash the Primer loop."""
    target = path or EVENTS_PATH
    record = {
        "ts": round(time.time(), 3),
        "task": int(task),
        "expert": expert,
        "ok": bool(ok),
        "latency_s": round(float(latency_s), 2),
        "verify_pass": bool(verify_pass),
        "published": bool(published),
        "error": error[:200],
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as e:  # noqa: BLE001
        log.warning("events append failed: %s", e)


# ── aggregation (used by telemetry_server) ───────────────────────

# Latency bucket upper bounds in seconds (last bucket = "≥300s / timeout").
_BUCKETS = (30.0, 60.0, 120.0, 240.0, 300.0)


def _bucket_index(latency_s: float, ok: bool) -> int:
    """Return the histogram bucket index for a latency value.

    Index 0..len(_BUCKETS)-1 are "< upper_bound[i]"; the final index
    (len(_BUCKETS)) is the overflow/timeout bucket. Failed calls land
    in the overflow bucket since their latency is usually the timeout
    deadline and semantically "this expert was unusable."
    """
    if not ok:
        return len(_BUCKETS)
    for i, ub in enumerate(_BUCKETS):
        if latency_s < ub:
            return i
    return len(_BUCKETS)


def bucket_edges() -> list[str]:
    """Human-readable labels for the histogram buckets (one per bucket)."""
    labels: list[str] = []
    prev = 0.0
    for ub in _BUCKETS:
        labels.append(f"{int(prev)}-{int(ub)}s")
        prev = ub
    labels.append(f"≥{int(_BUCKETS[-1])}s/err")
    return labels


def _empty_expert_stats() -> dict[str, Any]:
    return {
        "calls": 0,
        "verify_pass": 0,
        "verify_fail": 0,
        "errors": 0,
        "latency_buckets": [0] * (len(_BUCKETS) + 1),
    }


def aggregate(lines: list[str]) -> dict[str, Any]:
    """Aggregate a list of JSONL event strings into per-expert stats.

    Returns ``{"per_expert": {name: stats}, "published": int}`` where
    ``stats`` has ``calls``, ``verify_pass``, ``verify_fail``, ``errors``,
    and ``latency_buckets`` (list of ints, one per bucket).
    """
    per_expert: dict[str, dict[str, Any]] = {}
    published = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        name = ev.get("expert") or "?"
        stats = per_expert.setdefault(name, _empty_expert_stats())
        stats["calls"] += 1
        if ev.get("ok"):
            if ev.get("verify_pass"):
                stats["verify_pass"] += 1
            else:
                stats["verify_fail"] += 1
        else:
            stats["errors"] += 1
        idx = _bucket_index(float(ev.get("latency_s") or 0), bool(ev.get("ok")))
        stats["latency_buckets"][idx] += 1
        if ev.get("published"):
            published += 1
    return {
        "per_expert": per_expert,
        "published": published,
        "bucket_labels": bucket_edges(),
    }


def tail_lines(path: Path, max_lines: int = 2000) -> list[str]:
    """Return the last ``max_lines`` lines of ``path`` (or empty list)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Read the last ~chunk bytes and split; event lines are ~120
            # bytes so 2000 lines ≈ 240 KB. Cap at 1 MB to be safe.
            chunk = min(size, max(64 * 1024, max_lines * 200))
            f.seek(size - chunk)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        out = text.splitlines()
        if len(out) > max_lines:
            out = out[-max_lines:]
        # If we didn't start at byte 0, the first line may be partial.
        if len(data) < size and out:
            out = out[1:]
        return out
    except FileNotFoundError:
        return []
    except Exception:
        return []
