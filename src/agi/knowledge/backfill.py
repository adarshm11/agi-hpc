# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Phase 2 — wiki → UKG backfill.

Walks a wiki directory, parses verified sensei-note frontmatter, and
emits one ``filled`` node per note into the graph. Idempotent: a second
run on the same wiki + graph is a no-op unless ``force=True``.

Verification invariant comes from ``agi.common.sensei_note.is_verified``
— a note without a ``verified_by:`` frontmatter field is a draft and is
NEVER imported (matches the Primer's loader-side safety rule).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from agi.common.sensei_note import is_verified

from .graph import (
    get_node,
    normalize_tags,
    normalize_topic_key,
    upsert_node,
)

log = logging.getLogger("knowledge.backfill")

# Tags that describe the output *class* rather than the topic. Topic
# selection prefers a non-generic tag (the "family").
_GENERIC_TAGS: frozenset[str] = frozenset(
    {
        "arc",
        "primer",
        "classification",
        "transformation",
        "extraction",
        "expansion",
        "sensei",
    }
)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TAGS_RE = re.compile(r"^\s*tags\s*:\s*\[(.*?)\]\s*$", re.MULTILINE)
_TASK_RE = re.compile(r"^\s*task\s*:\s*(\d+)\s*$", re.MULTILINE)
_WRITTEN_AT_RE = re.compile(r"^\s*written_at\s*:\s*(\S+)\s*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class NoteMeta:
    task: int | None
    tags: list[str]
    written_at: str | None
    title: str | None

    @property
    def topic(self) -> str:
        """Human-readable topic: first non-generic tag, fallback to first, fallback 'general'."""
        for t in self.tags:
            if t not in _GENERIC_TAGS:
                return t.replace("-", " ")
        if self.tags:
            return self.tags[0].replace("-", " ")
        return "general"


@dataclass
class BackfillReport:
    imported: int = 0
    skipped_unverified: int = 0
    skipped_already_present: int = 0
    failed: int = 0
    imported_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.imported_ids is None:
            self.imported_ids = []

    def summary(self) -> str:
        return (
            f"imported={self.imported} "
            f"skipped_unverified={self.skipped_unverified} "
            f"skipped_already_present={self.skipped_already_present} "
            f"failed={self.failed}"
        )


# ── parsing ──────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> dict[str, str] | None:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    return {"_raw": m.group(1), "_body": text[m.end() :]}


def _parse_tags(raw_fm: str) -> list[str]:
    m = _TAGS_RE.search(raw_fm)
    if not m:
        return []
    inner = m.group(1)
    parts = [p.strip().strip("'\"") for p in inner.split(",")]
    return normalize_tags(parts)


def _parse_meta(text: str) -> NoteMeta | None:
    fm = _parse_frontmatter(text)
    if fm is None:
        return None
    raw = fm["_raw"]
    body = fm["_body"]

    task_m = _TASK_RE.search(raw)
    task = int(task_m.group(1)) if task_m else None

    tags = _parse_tags(raw)

    written_m = _WRITTEN_AT_RE.search(raw)
    written_at = written_m.group(1).strip().strip("'\"") if written_m else None

    title_m = _TITLE_RE.search(body)
    title = title_m.group(1).strip() if title_m else None

    return NoteMeta(task=task, tags=tags, written_at=written_at, title=title)


def _to_unix_seconds(date_str: str | None, fallback_mtime: float) -> int:
    """Parse ``written_at: 2026-04-19`` → unix seconds. Fallback on mtime.

    The Primer writes ISO dates without time, which parse as midnight UTC.
    That's fine for ordering — nodes from the same day tie-break by
    file-order per the graph's tie rule.
    """
    if date_str:
        try:
            # Try date-only first (YYYY-MM-DD), then full ISO
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_str):
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            else:
                dt = datetime.fromisoformat(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            pass
    return int(fallback_mtime)


# ── one-note import ──────────────────────────────────────────────


def _import_note(
    path: Path,
    *,
    wiki_root: Path,
    graph_path: Path | None,
    force: bool,
    dry_run: bool,
    report: BackfillReport,
) -> None:
    node_id = path.stem  # e.g. sensei_task_167

    # Read + verify gate — identical to the Primer's loader-side safety rule.
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("backfill_read_failed: path=%s error=%r", path, str(e)[:200])
        report.failed += 1
        return

    if not is_verified(text):
        log.info("backfill_skip_unverified: %s", path.name)
        report.skipped_unverified += 1
        return

    meta = _parse_meta(text)
    if meta is None:
        log.warning("backfill_skip_no_frontmatter: %s", path.name)
        report.failed += 1
        return

    if not force:
        existing = get_node(node_id, path=graph_path)
        if existing is not None and existing.get("type") == "filled":
            report.skipped_already_present += 1
            return

    # Assemble node fields
    topic = meta.topic
    topic_key = normalize_topic_key(topic)
    tags = meta.tags
    # body_ref is stored relative to wiki_root so it survives Atlas ↔ dev moves.
    try:
        rel = path.relative_to(wiki_root)
        body_ref = str(rel).replace("\\", "/")
    except ValueError:
        body_ref = str(path).replace("\\", "/")
    title = meta.title or f"Task {meta.task:03d}" if meta.task else path.stem
    ts = _to_unix_seconds(meta.written_at, path.stat().st_mtime)

    if dry_run:
        log.info(
            "backfill_would_import: id=%s topic=%s tags=%s body_ref=%s ts=%d",
            node_id,
            topic_key,
            tags,
            body_ref,
            ts,
        )
        report.imported += 1
        if report.imported_ids is not None:
            report.imported_ids.append(node_id)
        return

    upsert_node(
        id=node_id,
        type="filled",
        status="active",
        topic=topic,
        topic_key=topic_key,
        tags=tags,
        title=title,
        body_ref=body_ref,
        verified=True,
        source="backfill",
        evidence=[f"wiki:{path.name}"],
        now=ts,
        path=graph_path,
    )
    report.imported += 1
    if report.imported_ids is not None:
        report.imported_ids.append(node_id)
    log.info("backfill_imported: id=%s topic=%s", node_id, topic_key)


# ── public entrypoint ────────────────────────────────────────────


def backfill_wiki(
    wiki_dir: Path | str,
    *,
    graph_path: Path | None = None,
    pattern: str = "sensei_*.md",
    force: bool = False,
    dry_run: bool = False,
) -> BackfillReport:
    """Walk ``wiki_dir`` and upsert one ``filled`` node per verified note.

    Idempotent: a node is skipped if one with the same id and type
    ``filled`` already exists, unless ``force=True``. ``dry_run=True``
    reports what would be imported without touching the graph.

    Unverified notes (no ``verified_by`` in frontmatter) are always
    skipped — matching the Primer's loader-side safety invariant.
    """
    root = Path(wiki_dir)
    if not root.is_dir():
        log.warning("backfill_wiki: wiki_dir=%s is not a directory", root)
        return BackfillReport()

    report = BackfillReport()
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        _import_note(
            path,
            wiki_root=root,
            graph_path=graph_path,
            force=force,
            dry_run=dry_run,
            report=report,
        )
    return report


def iter_sensei_notes(wiki_dir: Path, pattern: str = "sensei_*.md") -> Iterable[Path]:
    """Convenience: iterate sensei-style notes in a wiki dir."""
    root = Path(wiki_dir)
    if not root.is_dir():
        return
    yield from sorted(root.glob(pattern))
