#!/usr/bin/env python3
"""
Demo: Dreaming Consolidation Cycle

Shows the 5-stage biological memory consolidation pipeline:
1. Episodic Replay — fetch unconsolidated episodes
2. Topic Clustering — group by subject via LLM
3. Certainty Assessment — score facts (A-D grades)
4. Creative Dreaming — recombine fragments for insights
5. Housekeeping — update wiki index, mark consolidated

Requires: PostgreSQL + LLM server running on Atlas.
For local testing without services, use --dry-run.

Usage:
    python scripts/demo_dream_cycle.py              # full cycle on Atlas
    python scripts/demo_dream_cycle.py --dry-run     # show structure only
"""

from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, "src")


def demo_dry_run() -> None:
    """Show the dreaming pipeline structure without live services."""
    print("=" * 60)
    print("Atlas Dreaming Demo (Dry Run)")
    print("=" * 60)

    print("""
The 5-stage biological memory consolidation pipeline:

  Stage 1: EPISODIC REPLAY (Wilson & McNaughton, 1994)
    Fetch unconsolidated episodes from PostgreSQL.
    Each episode = one user conversation with safety flags.

  Stage 2: TOPIC CLUSTERING
    LLM groups episodes by subject matter.
    Output: topic clusters with keywords.

  Stage 3: CERTAINTY ASSESSMENT (Bayesian Brain)
    Extract discrete facts from episodes.
    Score each: certainty (0-1) + confidence (0-1).
    Grade articles: A (established) / B (probable) /
                    C (unverified) / D (contradicted).

  Stage 4: CREATIVE DREAMING (REM sleep analogue)
    Select 3 diverse episodic fragments.
    Recombine to find cross-domain analogies.
    Output: DreamInsight with novelty + coherence scores.

  Stage 5: HOUSEKEEPING
    Update wiki index.md with all articles.
    Mark episodes as consolidated in database.
    Apply forgetting curve to stale memories.

Wiki articles become the AGI's life story:
  - Provenance: which conversations sparked this knowledge
  - Certainty grades: how confident the system is
  - Dream insights: creative connections discovered during sleep
""")

    print("To run a live cycle on Atlas:")
    print("  python scripts/demo_dream_cycle.py")
    print("=" * 60)


def demo_live() -> None:
    """Run a live dreaming cycle on Atlas."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    print("=" * 60)
    print("Atlas Dreaming Demo (Live Cycle)")
    print("=" * 60)

    from agi.dreaming.consolidator import (
        ConsolidatorConfig,
        MemoryConsolidator,
    )

    config = ConsolidatorConfig()
    consolidator = MemoryConsolidator(config)

    print(f"Wiki dir: {config.wiki_dir}")
    print(f"LLM URL: {config.llm_url}")
    print(f"Max episodes: {config.max_episodes_per_cycle}")
    print()

    result = asyncio.run(consolidator.run_cycle())

    print()
    print("=" * 60)
    print("Dreaming Cycle Complete")
    print("=" * 60)
    print(f"  Episodes processed:  {result.episodes_processed}")
    print(f"  Clusters found:      {result.clusters_found}")
    print(f"  Articles created:    {result.articles_created}")
    print(f"  Articles updated:    {result.articles_updated}")
    print(f"  Dream insights:      {result.dream_insights}")
    print(f"  Errors:              {len(result.errors)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atlas Dreaming Consolidation Demo")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pipeline structure without live services",
    )
    args = parser.parse_args()

    if args.dry_run:
        demo_dry_run()
    else:
        demo_live()
