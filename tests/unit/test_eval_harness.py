"""Unit tests for evals.harness.

Use stubbed generator_fn — no network calls. Verifies:
- portfolio loading
- outcome taxonomy (pass / partial / fail / error)
- report JSON structure
- CLI-entry points remain importable
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.harness import (
    HARNESS_VERSION,
    CaseResult,
    Portfolio,
    PortfolioEntry,
    load_portfolio,
    run_harness,
)


FIXTURES = Path(__file__).resolve().parents[1].parent / "evals" / "fixtures"


# ── portfolio loading ─────────────────────────────────────────────


def test_load_portfolio_from_yaml(tmp_path):
    yaml_body = """
name: mini
description: just one
tasks:
  - id: 20
    difficulty: easy
  - id: 56
    difficulty: medium
    note: symmetry classifier
"""
    p = tmp_path / "mini.yaml"
    p.write_text(yaml_body)
    portfolio = load_portfolio(p)
    assert portfolio.name == "mini"
    assert len(portfolio.tasks) == 2
    assert portfolio.tasks[0].id == 20
    assert portfolio.tasks[1].difficulty == "medium"
    assert portfolio.tasks[1].note == "symmetry classifier"


def test_load_portfolio_empty_tasks(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("name: empty\ndescription: none\n")
    portfolio = load_portfolio(p)
    assert portfolio.tasks == []


# ── harness with stubbed generator ────────────────────────────────


def _make_portfolio() -> Portfolio:
    return Portfolio(
        name="fixtures",
        description="fixture tasks",
        tasks=[
            PortfolioEntry(id=1, difficulty="easy"),  # identity.json
            PortfolioEntry(id=2, difficulty="medium"),  # swap_01.json
        ],
    )


def _fixture_task_dir(tmp_path):
    """Copy the fixtures into a task*.json layout the harness expects."""
    fixtures_dir = tmp_path / "tasks"
    fixtures_dir.mkdir()
    (fixtures_dir / "task001.json").write_text(
        (FIXTURES / "task_identity.json").read_text()
    )
    (fixtures_dir / "task002.json").write_text(
        (FIXTURES / "task_swap_01.json").read_text()
    )
    return fixtures_dir


def test_harness_pass_outcome_on_correct_code(tmp_path):
    task_dir = _fixture_task_dir(tmp_path)
    results_dir = tmp_path / "results"
    portfolio = _make_portfolio()

    def gen(task):
        # Identity for task 1 (passes), bogus for task 2 (fails)
        if task["train"][0]["input"] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]:
            return "def transform(g):\n    return g"
        return "def transform(g):\n    return [[99]]"

    report = run_harness(
        portfolio,
        generator_fn=gen,
        task_dir=task_dir,
        results_dir=results_dir,
        expert_name="stub",
    )
    assert report["harness_version"] == HARNESS_VERSION
    assert report["portfolio"] == "fixtures"
    assert report["expert"] == "stub"
    outcomes = {c["task_id"]: c["outcome"] for c in report["cases"]}
    assert outcomes[1] == "pass"
    assert outcomes[2] == "fail"
    assert report["summary"]["pass_count"] == 1
    assert report["summary"]["fail_count"] == 1
    assert report["summary"]["pass_rate"] == 0.5


def test_harness_error_outcome_on_generator_exception(tmp_path):
    task_dir = _fixture_task_dir(tmp_path)
    portfolio = _make_portfolio()

    def gen(_task):
        raise RuntimeError("LLM unavailable")

    report = run_harness(
        portfolio,
        generator_fn=gen,
        task_dir=task_dir,
        results_dir=tmp_path / "r",
        expert_name="stub",
    )
    assert all(c["outcome"] == "error" for c in report["cases"])
    assert report["summary"]["error_count"] == 2


def test_harness_error_outcome_on_missing_task_json(tmp_path):
    portfolio = Portfolio(
        name="p",
        description="",
        tasks=[PortfolioEntry(id=999, difficulty="easy")],
    )

    def gen(_task):
        return "def transform(g):\n    return g"

    report = run_harness(
        portfolio,
        generator_fn=gen,
        task_dir=tmp_path,  # empty — task999.json doesn't exist
        results_dir=tmp_path / "r",
        expert_name="stub",
    )
    assert report["cases"][0]["outcome"] == "error"
    assert "not found" in report["cases"][0]["diagnostic"]


def test_harness_error_outcome_on_no_code_extractable(tmp_path):
    task_dir = _fixture_task_dir(tmp_path)
    portfolio = Portfolio(
        name="p", description="", tasks=[PortfolioEntry(id=1, difficulty="easy")]
    )

    def gen(_task):
        return "I can't solve this."  # no def transform

    report = run_harness(
        portfolio,
        generator_fn=gen,
        task_dir=task_dir,
        results_dir=tmp_path / "r",
        expert_name="stub",
    )
    assert report["cases"][0]["outcome"] == "error"
    assert "no code extracted" in report["cases"][0]["diagnostic"]


def test_harness_writes_result_file(tmp_path):
    task_dir = _fixture_task_dir(tmp_path)
    results_dir = tmp_path / "results"
    portfolio = Portfolio(
        name="one", description="", tasks=[PortfolioEntry(id=1, difficulty="easy")]
    )

    def gen(_task):
        return "def transform(g):\n    return g"

    run_harness(
        portfolio,
        generator_fn=gen,
        task_dir=task_dir,
        results_dir=results_dir,
        expert_name="stub",
    )
    files = list(results_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["portfolio"] == "one"
    assert payload["expert"] == "stub"
    assert payload["cases"][0]["task_id"] == 1
    assert payload["cases"][0]["outcome"] == "pass"


def test_harness_skips_file_write_when_requested(tmp_path):
    task_dir = _fixture_task_dir(tmp_path)
    results_dir = tmp_path / "results"
    portfolio = Portfolio(
        name="nf", description="", tasks=[PortfolioEntry(id=1, difficulty="easy")]
    )

    def gen(_task):
        return "def transform(g):\n    return g"

    run_harness(
        portfolio,
        generator_fn=gen,
        task_dir=task_dir,
        results_dir=results_dir,
        expert_name="stub",
        write_file=False,
    )
    assert not results_dir.exists() or list(results_dir.glob("*.json")) == []


def test_case_result_roundtrips_as_dict():
    c = CaseResult(
        task_id=42,
        outcome="pass",
        latency_s=1.23,
        score_correct=3,
        score_total=3,
    )
    from dataclasses import asdict

    d = asdict(c)
    assert d["task_id"] == 42
    assert d["outcome"] == "pass"
