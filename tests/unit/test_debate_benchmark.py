# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the debate validation benchmark.

Tests question coverage, category balance, scoring,
summarization, and dry-run mode.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Import benchmark data directly
sys.path.insert(0, "scripts")
from benchmark_debate import (
    QUESTIONS,
    BenchmarkResult,
    summarize,
)


class TestQuestionCoverage:
    """Tests for benchmark question set."""

    def test_five_categories(self) -> None:
        assert set(QUESTIONS.keys()) == {
            "factual",
            "reasoning",
            "ethics",
            "creative",
            "code",
        }

    def test_twenty_per_category(self) -> None:
        for cat, questions in QUESTIONS.items():
            assert len(questions) >= 20, f"{cat} has only {len(questions)} questions"

    def test_all_have_question_and_reference(self) -> None:
        for cat, questions in QUESTIONS.items():
            for q in questions:
                assert "q" in q, f"Missing 'q' in {cat}"
                assert "ref" in q, f"Missing 'ref' in {cat}"
                assert len(q["q"]) > 10, f"Question too short in {cat}"
                assert len(q["ref"]) >= 1, f"Reference missing in {cat}"

    def test_total_question_count(self) -> None:
        total = sum(len(qs) for qs in QUESTIONS.values())
        assert total >= 100  # At least 100 questions


class TestSummarization:
    """Tests for result summarization."""

    def _make_results(self) -> list:
        return [
            BenchmarkResult(
                question="Q1",
                category="factual",
                reference="A1",
                mode="single",
                response="R1",
                quality_score=8.0,
                correct=True,
                latency_s=1.5,
            ),
            BenchmarkResult(
                question="Q2",
                category="factual",
                reference="A2",
                mode="single",
                response="R2",
                quality_score=6.0,
                correct=False,
                latency_s=2.0,
            ),
            BenchmarkResult(
                question="Q1",
                category="factual",
                reference="A1",
                mode="debate",
                response="R3",
                quality_score=9.0,
                correct=True,
                latency_s=60.0,
            ),
            BenchmarkResult(
                question="Q2",
                category="factual",
                reference="A2",
                mode="debate",
                response="R4",
                quality_score=7.0,
                correct=True,
                latency_s=55.0,
            ),
        ]

    def test_computes_accuracy(self) -> None:
        results = self._make_results()
        summary = summarize(results)
        assert summary["single"]["accuracy"] == 0.5
        assert summary["debate"]["accuracy"] == 1.0

    def test_computes_quality(self) -> None:
        results = self._make_results()
        summary = summarize(results)
        assert summary["single"]["mean_quality"] == 7.0
        assert summary["debate"]["mean_quality"] == 8.0

    def test_computes_latency(self) -> None:
        results = self._make_results()
        summary = summarize(results)
        assert summary["single"]["mean_latency_s"] == 1.75
        assert summary["debate"]["mean_latency_s"] == 57.5

    def test_by_category(self) -> None:
        results = self._make_results()
        summary = summarize(results)
        assert "factual" in summary["single"]["by_category"]


class TestDryRun:
    """Tests for dry-run mode."""

    def test_dry_run_executes(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_debate.py",
                "--dry-run",
                "--questions",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "DEBATE VALIDATION BENCHMARK" in result.stdout


class TestBenchmarkScript:
    """Tests for the benchmark script file."""

    def test_script_exists(self) -> None:
        assert Path("scripts/benchmark_debate.py").exists()

    def test_has_all_modes(self) -> None:
        content = Path("scripts/benchmark_debate.py").read_text()
        assert "single" in content
        assert "debate" in content
        assert "arbiter" in content
