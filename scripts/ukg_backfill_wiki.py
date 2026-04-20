#!/usr/bin/env python3
"""Backfill the Unified Knowledge Graph from the existing sensei wiki.

Walks ``--wiki-dir`` for ``sensei_*.md`` files, parses their frontmatter,
and upserts one ``filled`` node per verified note into the graph JSONL.
Idempotent — a second run on the same inputs is a no-op unless
``--force`` is supplied.

Usage:
    python3 scripts/ukg_backfill_wiki.py
    python3 scripts/ukg_backfill_wiki.py --wiki-dir ./wiki --dry-run
    python3 scripts/ukg_backfill_wiki.py --graph /tmp/g.jsonl --force
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running as a script from the repo root: ``python3 scripts/ukg_...``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agi.knowledge.backfill import backfill_wiki  # noqa: E402


def _default_wiki_dir() -> Path:
    """Resolve wiki dir: env → repo-local ./wiki → Atlas default."""
    env = os.environ.get("EREBUS_WIKI_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent.parent / "wiki"
    if here.is_dir():
        return here
    return Path("/home/claude/agi-hpc/wiki")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wiki-dir",
        type=Path,
        default=_default_wiki_dir(),
        help="Directory containing sensei_*.md notes",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Graph JSONL path (defaults to KNOWLEDGE_GRAPH_PATH env or /archive/neurogolf/knowledge_graph.jsonl)",
    )
    parser.add_argument(
        "--pattern",
        default="sensei_*.md",
        help="Glob pattern inside wiki-dir (default: sensei_*.md)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upsert even if a filled node with the same id already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be imported without modifying the graph",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-note import log lines",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    report = backfill_wiki(
        args.wiki_dir,
        graph_path=args.graph,
        pattern=args.pattern,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(report.summary())
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
