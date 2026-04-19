"""Evaluation harness driver.

Given a portfolio YAML and an expert name, runs each task through a
``generator_fn`` (by default: vMOE call with a standard direct prompt),
validates the returned code against ``task.train``, and writes a result
JSON.

The harness is a pure orchestrator — it does not touch Atlas memory,
does not publish to the wiki, and does not talk to systemd. Reproducible
by construction: fixed seed, temperature=0, git SHA stamped in output.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger("evals.harness")

HARNESS_VERSION = "0.1"

_DEFAULT_TASK_DIR = Path("/archive/neurogolf")
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── prompt template (a simple "direct" strategy) ──────────────────


_DIRECT_PROMPT = """You are solving an ARC-AGI puzzle. Below is a set of input/output grid examples. Find the transformation rule and write a Python function `transform(grid: list[list[int]]) -> list[list[int]]` that implements it.

Hard requirements:
- Use only numpy and the Python stdlib.
- Deterministic: no randomness.
- Your function must match every training example exactly.

Training examples:

```json
{examples}
```

Output format: a Python code block with ONLY the ``def transform`` function. No explanation.
"""


def _build_direct_messages(task: dict) -> list[dict[str, Any]]:
    """Assemble a chat-completion message list for the direct strategy."""
    examples = json.dumps(task.get("train", []), indent=2)
    user = _DIRECT_PROMPT.format(examples=examples)
    return [{"role": "user", "content": user}]


# ── data classes ──────────────────────────────────────────────────


@dataclass
class PortfolioEntry:
    id: int
    difficulty: str = "unknown"
    note: str = ""


@dataclass
class Portfolio:
    name: str
    description: str
    tasks: list[PortfolioEntry]

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            tasks=[
                PortfolioEntry(
                    id=int(t["id"]),
                    difficulty=t.get("difficulty", "unknown"),
                    note=t.get("note", ""),
                )
                for t in data.get("tasks", [])
            ],
        )


@dataclass
class CaseResult:
    task_id: int
    outcome: str  # pass | partial | fail | timeout | error
    latency_s: float
    score_correct: int = 0
    score_total: int = 0
    diagnostic: str = ""
    code_hash: str = ""
    expert: str = ""
    strategy: str = ""


@dataclass
class HarnessReport:
    harness_version: str
    timestamp: str
    git_sha: str
    portfolio: str
    expert: str
    strategy: str
    seed: int
    temperature: float
    cases: list[CaseResult]
    summary: dict = field(default_factory=dict)


# ── portfolio loading ─────────────────────────────────────────────


def load_portfolio(path: str | Path) -> Portfolio:
    """Parse a YAML portfolio file. Accepts either yaml or json payload."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # project already depends on pyyaml

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)  # yaml is a superset; try json as fallback
    return Portfolio.from_dict(data or {})


def load_task(task_dir: Path, task_id: int) -> dict | None:
    path = task_dir / f"task{task_id:03d}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── the generator_fn default (vMOE expert call) ───────────────────


def make_default_generator(expert_name: str, temperature: float = 0.0, max_tokens: int = 8000) -> Callable[[dict], Awaitable[str]]:
    """Return an async generator_fn that asks one vMOE expert per task."""

    async def _gen(task: dict) -> str:
        # Lazy import so unit tests can run without openai installed
        from agi.primer.vmoe import default_experts, vMOE

        moe = vMOE(experts=default_experts())
        messages = _build_direct_messages(task)
        resp = await moe.call(
            expert_name,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not resp.ok:
            raise RuntimeError(f"expert {expert_name} failed: {resp.error}")
        return resp.content

    return _gen


# ── single-case runner ────────────────────────────────────────────


def _score_one(
    task_id: int,
    task: dict,
    generator_fn: Callable[[dict], Awaitable[str] | str],
    *,
    expert: str,
    strategy: str,
) -> CaseResult:
    """Run one task through generator_fn, validate, return CaseResult."""
    # Lazy imports keep the module importable even when running unit
    # tests with primer.validator stubbed or openai absent.
    from agi.primer.validator import extract_code, validate

    t0 = time.time()
    try:
        result = generator_fn(task)
        if asyncio.iscoroutine(result):
            content = asyncio.get_event_loop().run_until_complete(result)
        else:
            content = result
    except Exception as e:  # noqa: BLE001
        return CaseResult(
            task_id=task_id,
            outcome="error",
            latency_s=time.time() - t0,
            diagnostic=f"{type(e).__name__}: {e}",
            expert=expert,
            strategy=strategy,
        )

    code = extract_code(content or "")
    if not code:
        return CaseResult(
            task_id=task_id,
            outcome="error",
            latency_s=time.time() - t0,
            diagnostic="no code extracted",
            expert=expert,
            strategy=strategy,
        )
    vr = validate(code, task)
    code_hash = hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
    per_example = vr.per_example or []
    correct = sum(1 for e in per_example if e.get("correct"))
    total = len(per_example)
    if vr.all_pass:
        outcome = "pass"
    elif correct > 0:
        outcome = "partial"
    else:
        outcome = "fail"
    return CaseResult(
        task_id=task_id,
        outcome=outcome,
        latency_s=time.time() - t0,
        score_correct=correct,
        score_total=total,
        diagnostic=vr.diagnostic[:400],
        code_hash=code_hash,
        expert=expert,
        strategy=strategy,
    )


# ── orchestrator ──────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        repo = Path(__file__).resolve().parents[1]
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return r.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def run_harness(
    portfolio: Portfolio,
    *,
    expert_name: str = "kimi",
    strategy: str = "direct",
    generator_fn: Callable[[dict], Awaitable[str] | str] | None = None,
    task_dir: Path | None = None,
    results_dir: Path | None = None,
    seed: int = 0,
    temperature: float = 0.0,
    write_file: bool = True,
) -> dict:
    """Run every task in ``portfolio`` through ``generator_fn`` and record
    results to a report JSON. Returns the report dict.

    If ``generator_fn`` is None, the default (vMOE expert call) is used —
    tests should inject a stub instead."""
    task_dir = task_dir or _DEFAULT_TASK_DIR
    results_dir = results_dir or _DEFAULT_RESULTS_DIR
    if generator_fn is None:
        generator_fn = make_default_generator(expert_name, temperature=temperature)

    cases: list[CaseResult] = []
    for entry in portfolio.tasks:
        task = load_task(task_dir, entry.id)
        if task is None:
            cases.append(
                CaseResult(
                    task_id=entry.id,
                    outcome="error",
                    latency_s=0.0,
                    diagnostic=f"task JSON not found at {task_dir}/task{entry.id:03d}.json",
                    expert=expert_name,
                    strategy=strategy,
                )
            )
            continue
        cases.append(
            _score_one(
                entry.id,
                task,
                generator_fn,
                expert=expert_name,
                strategy=strategy,
            )
        )

    counts = {"pass": 0, "partial": 0, "fail": 0, "timeout": 0, "error": 0}
    for c in cases:
        counts[c.outcome] = counts.get(c.outcome, 0) + 1
    total = len(cases)
    summary = {
        "pass_count": counts["pass"],
        "partial_count": counts["partial"],
        "fail_count": counts["fail"],
        "timeout_count": counts["timeout"],
        "error_count": counts["error"],
        "total": total,
        "pass_rate": (counts["pass"] / total) if total else 0.0,
    }
    report = HarnessReport(
        harness_version=HARNESS_VERSION,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_sha=_git_sha(),
        portfolio=portfolio.name,
        expert=expert_name,
        strategy=strategy,
        seed=seed,
        temperature=temperature,
        cases=cases,
        summary=summary,
    )
    out = {
        **asdict(report),
        "cases": [asdict(c) for c in cases],
    }
    if write_file:
        results_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"{report.timestamp.replace(':', '-')}_{portfolio.name}_{expert_name}.json"
        )
        (results_dir / fname).write_text(json.dumps(out, indent=2))
        log.info("wrote %s", results_dir / fname)
    return out


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Atlas evaluation harness")
    ap.add_argument("--portfolio", required=True, help="Path to portfolio YAML")
    ap.add_argument("--expert", default="kimi", help="vMOE expert name")
    ap.add_argument(
        "--strategy", default="direct", help="Prompt strategy (informational only)"
    )
    ap.add_argument(
        "--task-dir",
        default=str(_DEFAULT_TASK_DIR),
        help="Directory containing task*.json files",
    )
    ap.add_argument(
        "--results-dir",
        default=str(_DEFAULT_RESULTS_DIR),
        help="Where to write the JSON report",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    portfolio = load_portfolio(args.portfolio)
    report = run_harness(
        portfolio,
        expert_name=args.expert,
        strategy=args.strategy,
        task_dir=Path(args.task_dir),
        results_dir=Path(args.results_dir),
        seed=args.seed,
        temperature=args.temperature,
    )
    s = report["summary"]
    print(
        f"{portfolio.name}: pass={s['pass_count']}/{s['total']} "
        f"({s['pass_rate']:.1%}) partial={s['partial_count']} "
        f"fail={s['fail_count']} error={s['error_count']}"
    )
    sys.exit(0 if s["pass_rate"] == 1.0 else 1)


if __name__ == "__main__":
    main()
