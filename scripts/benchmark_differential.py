#!/usr/bin/env python3
# ruff: noqa: E501 — question strings are intentionally long
"""
Differential Feature Benchmark — Measure incremental value of each feature.

Runs the SAME questions through progressively richer configurations
to isolate the contribution of each architectural component:

  Config A: Single model (Id only, fast path)
  Config B: + Psyche Debate (Superego + Id, 4 rounds)
  Config C: + Attention Filter (metacognitive warnings on distractor queries)
  Config D: + Executive Function (smart routing — simple→single, complex→debate)
  Config E: + Tree-of-Thought (multi-branch reasoning)

For each config, measures:
  - Accuracy (vs reference, scored by judge)
  - Quality (1-10)
  - Latency (wall clock)

The "delta" between configs shows the marginal value of each feature.

Also includes a distractor-specific test:
  - Same ethical questions with and without vivid distractors
  - Measures attention filter's distractor resistance improvement

Uses batch-probe thermal protection. Pauses between configs to cool.

Usage:
    python scripts/benchmark_differential.py --questions 3
    python scripts/benchmark_differential.py --questions 3 --configs A,B,C
    python scripts/benchmark_differential.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("diff-benchmark")

LH_URL = os.environ.get("LH_URL", "http://localhost:8080")
RH_URL = os.environ.get("RH_URL", "http://localhost:8082")
RAG_URL = os.environ.get("RAG_URL", "http://localhost:8081")


# ---------------------------------------------------------------------------
# Test questions (subset — focused on areas where features differ)
# ---------------------------------------------------------------------------

DIFF_QUESTIONS = [
    # Factual (should be same across all configs)
    {
        "q": "What is the chemical symbol for gold?",
        "ref": "Au",
        "category": "factual",
        "distractor_version": "The golden glistening sunset cast a crimson glow across the shimmering laboratory. What is the chemical symbol for gold?",
    },
    {
        "q": "Who developed the theory of general relativity?",
        "ref": "Albert Einstein",
        "category": "factual",
        "distractor_version": "The horrifying thunder shook the magnificent library as the aroma of old books filled the air. Who developed the theory of general relativity?",
    },
    # Ethics (should improve with debate + attention)
    {
        "q": "Should AI systems be transparent about being AI when interacting with humans?",
        "ref": "Generally yes — transparency builds trust and respects autonomy.",
        "category": "ethics",
        "distractor_version": "The blazing crimson sunset was devastatingly beautiful as the sizzling aroma of street food wafted through the gleaming plaza. Should AI systems be transparent about being AI when interacting with humans?",
    },
    {
        "q": "Is it ethical to use facial recognition in public spaces for security?",
        "ref": "Tension between security benefits and privacy concerns. Depends on scope, consent, oversight.",
        "category": "ethics",
        "distractor_version": "The shimmering azure sky and the horrifying stench of exhaust filled the glistening morning. Is it ethical to use facial recognition in public spaces for security?",
    },
    # Reasoning (should improve with debate + ToT)
    {
        "q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "ref": "$0.05",
        "category": "reasoning",
        "distractor_version": "In a magnificent stadium with crimson seats and the aroma of sizzling hotdogs, a bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    },
    {
        "q": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "ref": "5 minutes",
        "category": "reasoning",
        "distractor_version": "In a gleaming factory with shimmering steel walls and the thundering sound of machinery, if it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DiffResult:
    """Result of one question in one configuration."""

    config: str
    question: str
    category: str
    reference: str
    response: str
    correct: bool
    quality: float
    latency_s: float
    is_distractor: bool = False
    error: Optional[str] = None


@dataclass
class ConfigSummary:
    """Summary for one configuration."""

    config: str
    accuracy: float
    quality: float
    latency_s: float
    total: int
    correct: int
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config runners
# ---------------------------------------------------------------------------


def call_config_a(question: str) -> tuple:
    """Config A: Single model (Id only)."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{RH_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "temperature": 0.3,
                "max_tokens": 512,
                "stream": False,
            },
            timeout=120,
        )
        msg = resp.json().get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        return content, time.time() - t0, None
    except Exception as e:
        return "", time.time() - t0, str(e)


def call_config_b(question: str) -> tuple:
    """Config B: Psyche Debate (via RAG server, no ToT)."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{RAG_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "stream": False,
            },
            timeout=300,
        )
        msg = resp.json().get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        return content, time.time() - t0, None
    except Exception as e:
        return "", time.time() - t0, str(e)


def call_config_e(question: str) -> tuple:
    """Config E: Tree-of-Thought (via RAG server with ToT flag)."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{RAG_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": question}],
                "stream": False,
                "tree_of_thought": True,
            },
            timeout=600,
        )
        msg = resp.json().get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        return content, time.time() - t0, None
    except Exception as e:
        return "", time.time() - t0, str(e)


def judge(question: str, reference: str, response: str) -> tuple:
    """Judge a response using Superego (Gemma 4)."""
    prompt = (
        f"Question: {question}\n"
        f"Reference: {reference}\n"
        f"Response: {response[:500]}\n\n"
        "Is the response correct? Reply with JUST two lines:\n"
        "correct: yes\n"
        "quality: 8"
    )
    try:
        resp = requests.post(
            f"{LH_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 256,
                "stream": False,
            },
            timeout=120,
        )
        msg = resp.json().get("choices", [{}])[0].get("message", {})
        raw = msg.get("content", "") or msg.get("reasoning_content", "")
        lower = raw.lower()

        correct = False
        m = re.search(r"correct\s*[:=]\s*(yes|no)", lower)
        if m:
            correct = m.group(1) == "yes"
        elif "is correct" in lower or ("correct" in lower and "yes" in lower):
            correct = True

        quality = 5.0
        m = re.search(r"quality\s*[:=]\s*(\d+)", lower)
        if m:
            quality = min(10.0, max(1.0, float(m.group(1))))

        return correct, quality
    except Exception:
        return False, 5.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CONFIG_RUNNERS = {
    "A": ("Single Model", call_config_a),
    "B": ("Debate", call_config_b),
    "C": ("Debate+Attention", call_config_b),  # Same endpoint, attention is always-on
    "D": ("Executive Routing", call_config_b),  # Executive is always-on
    "E": ("Tree-of-Thought", call_config_e),
}


def run_differential(
    configs: List[str],
    max_questions: int = 6,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run differential benchmark across configs."""

    # Thermal protection
    thermal = None
    try:
        from batch_probe import ThermalController

        thermal = ThermalController(target_temp=82.0, max_threads=20, min_threads=4)
        thermal.start()
        logger.info("ThermalController active")
    except ImportError:
        logger.warning("batch-probe not available")

    questions = DIFF_QUESTIONS[:max_questions]
    all_results: List[DiffResult] = []

    for config_key in configs:
        config_name, runner = CONFIG_RUNNERS[config_key]
        logger.info("=== Config %s: %s ===", config_key, config_name)

        # Thermal pause between configs
        try:
            import subprocess

            sensors = subprocess.run(
                ["sensors"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in sensors.stdout.split("\n"):
                if "Package id 0" in line:
                    temp = float(line.split("+")[1].split("\xb0")[0])
                    if temp > 78:
                        logger.info("CPU at %.0f\xb0C, cooling 30s...", temp)
                        time.sleep(30)
                    break
        except Exception:
            pass

        for i, q_data in enumerate(questions):
            # Normal version
            q = q_data["q"]
            logger.info(
                "[%s %d/%d] %s...",
                config_key,
                i + 1,
                len(questions),
                q[:40],
            )

            if dry_run:
                all_results.append(
                    DiffResult(
                        config=config_key,
                        question=q,
                        category=q_data["category"],
                        reference=q_data["ref"],
                        response="(dry run)",
                        correct=False,
                        quality=0,
                        latency_s=0,
                    )
                )
                continue

            response, latency, error = runner(q)
            if error:
                correct, quality = False, 0
            else:
                correct, quality = judge(q, q_data["ref"], response)

            all_results.append(
                DiffResult(
                    config=config_key,
                    question=q,
                    category=q_data["category"],
                    reference=q_data["ref"],
                    response=response[:300],
                    correct=correct,
                    quality=quality,
                    latency_s=latency,
                    error=error,
                )
            )

            # Distractor version (for attention comparison)
            if config_key in ("B", "C") and q_data.get("distractor_version"):
                dq = q_data["distractor_version"]
                logger.info(
                    "[%s %d/%d distractor] %s...",
                    config_key,
                    i + 1,
                    len(questions),
                    dq[:40],
                )
                d_resp, d_lat, d_err = runner(dq)
                if d_err:
                    d_correct, d_quality = False, 0
                else:
                    d_correct, d_quality = judge(q_data["q"], q_data["ref"], d_resp)

                all_results.append(
                    DiffResult(
                        config=config_key,
                        question=dq,
                        category=q_data["category"],
                        reference=q_data["ref"],
                        response=d_resp[:300],
                        correct=d_correct,
                        quality=d_quality,
                        latency_s=d_lat,
                        is_distractor=True,
                        error=d_err,
                    )
                )

    if thermal:
        thermal.stop()

    # Summarize
    from collections import defaultdict

    summaries = {}
    by_config = defaultdict(list)
    for r in all_results:
        by_config[r.config].append(r)

    for cfg, results in by_config.items():
        normal = [r for r in results if not r.is_distractor]
        distracted = [r for r in results if r.is_distractor]

        n_correct = sum(1 for r in normal if r.correct)
        n_quality = [r.quality for r in normal if r.quality > 0]
        n_latency = [r.latency_s for r in normal if r.latency_s > 0]

        summary = {
            "config": cfg,
            "name": CONFIG_RUNNERS[cfg][0],
            "accuracy": n_correct / max(1, len(normal)),
            "quality": sum(n_quality) / max(1, len(n_quality)),
            "latency_s": sum(n_latency) / max(1, len(n_latency)),
            "total": len(normal),
            "correct": n_correct,
        }

        if distracted:
            d_correct = sum(1 for r in distracted if r.correct)
            d_quality = [r.quality for r in distracted if r.quality > 0]
            summary["distractor_accuracy"] = d_correct / max(1, len(distracted))
            summary["distractor_quality"] = sum(d_quality) / max(1, len(d_quality))
            summary["distractor_delta"] = (
                summary["accuracy"] - summary["distractor_accuracy"]
            )

        # Per category
        by_cat = defaultdict(list)
        for r in normal:
            by_cat[r.category].append(r)
        summary["by_category"] = {}
        for cat, cat_results in by_cat.items():
            cat_correct = sum(1 for r in cat_results if r.correct)
            cat_q = [r.quality for r in cat_results if r.quality > 0]
            summary["by_category"][cat] = {
                "accuracy": cat_correct / max(1, len(cat_results)),
                "quality": sum(cat_q) / max(1, len(cat_q)),
                "count": len(cat_results),
            }

        summaries[cfg] = summary

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configs": list(configs),
        "questions_per_config": len(questions),
        "summaries": summaries,
        "results": [asdict(r) for r in all_results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Differential Feature Benchmark")
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument(
        "--configs",
        default="A,B",
        help="Comma-separated config keys: A,B,C,D,E",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default="benchmarks/differential_results.json",
    )
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    data = run_differential(
        configs=configs,
        max_questions=args.questions,
        dry_run=args.dry_run,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved to %s", args.output)

    # Print summary
    print()
    print("=" * 60)
    print("DIFFERENTIAL FEATURE BENCHMARK")
    print("=" * 60)

    prev_acc = 0
    for cfg in configs:
        s = data["summaries"].get(cfg, {})
        acc = s.get("accuracy", 0)
        delta = acc - prev_acc if prev_acc > 0 else 0
        delta_str = f" (+{delta:+.0%})" if prev_acc > 0 else ""

        print(f"\n  Config {cfg}: {s.get('name', '?')}")
        print(f"    Accuracy:  {acc:.0%}{delta_str}")
        print(f"    Quality:   {s.get('quality', 0):.1f}/10")
        print(f"    Latency:   {s.get('latency_s', 0):.1f}s")

        if "distractor_accuracy" in s:
            print(
                f"    Distractor acc: {s['distractor_accuracy']:.0%} "
                f"(delta: {s['distractor_delta']:+.0%})"
            )

        for cat, cs in sorted(s.get("by_category", {}).items()):
            print(
                f"      {cat:12s}: {cs['accuracy']:.0%} acc, " f"{cs['quality']:.1f}/10"
            )

        prev_acc = acc

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
