# Evaluation Harness

A lightweight, repeatable driver for offline evaluation of Atlas's problem-solving components. Produces JSON reports that are diffable across commits.

Resolves issue #39 — "Build a basic evaluation harness for offline and synthetic testing."

## Design principles

- **Does NOT mutate any Atlas state.** Runs read-only against task fixtures + LLM services. Results write to `evals/results/` (gitignored); the harness never touches `arc_scientist_memory.json` or publishes wiki notes.
- **Reproducible.** Fixed seed, `temperature=0` by default, all inputs pinned in the portfolio YAML, git SHA stamped on every result JSON.
- **Dependency-injectable.** The harness accepts a `generator_fn(task) -> str` callable. Default impl calls a vMOE expert; tests pass a stub.
- **Reuses the Primer validator.** Candidate code is scored against `task.train` via `agi.primer.validator.validate()` — same gate the Primer uses before publishing. An "evaluation pass" means the same thing in both places.

## Layout

    evals/
    ├── README.md               # this file
    ├── __init__.py             # exports run_harness() for programmatic use
    ├── harness.py              # the driver
    ├── portfolios/
    │   └── arc_smoke.yaml      # minimal task set for CI
    ├── fixtures/               # synthetic task JSONs for tests
    └── results/                # JSON output per run (gitignored)

## Usage

### CLI

```bash
# Run the default smoke portfolio against Kimi
python -m evals.harness \
    --portfolio evals/portfolios/arc_smoke.yaml \
    --expert kimi

# Run against a specific strategy
python -m evals.harness \
    --portfolio evals/portfolios/arc_smoke.yaml \
    --expert qwen3 \
    --strategy diagnostic
```

### Programmatic (for notebooks, CI, comparisons)

```python
from evals.harness import run_harness, load_portfolio

portfolio = load_portfolio("evals/portfolios/arc_smoke.yaml")
report = run_harness(portfolio, expert_name="kimi")

print(f"Pass rate: {report['summary']['pass_rate']:.1%}")
for case in report['cases']:
    print(f"  task{case['task_id']:03d}: {case['outcome']}")
```

## Portfolio YAML schema

```yaml
name: arc_smoke
description: 5-task smoke portfolio — fast feedback for CI
tasks:
  - id: 20
    difficulty: easy          # easy / medium / hard — informational only
    note: solved by Primer
  - id: 56
    difficulty: medium
    note: symmetry-classifier family
  # ...
```

The portfolio is a deliberately-short list — a **smoke test**, not a full benchmark. A full-sweep portfolio would be generated from the memory file.

## Result JSON schema

```jsonc
{
  "harness_version": "0.1",
  "timestamp": "2026-04-19T18:45:00Z",
  "git_sha": "484736a",
  "portfolio": "arc_smoke",
  "expert": "kimi",
  "strategy": "direct",
  "seed": 0,
  "temperature": 0.0,
  "cases": [
    {
      "task_id": 20,
      "outcome": "pass" | "partial" | "fail" | "timeout" | "error",
      "score": {"correct": 2, "total": 2},
      "latency_s": 142.1,
      "diagnostic": "",
      "code_hash": "abc123..."
    }
  ],
  "summary": {
    "pass_count": 3,
    "partial_count": 1,
    "fail_count": 1,
    "total": 5,
    "pass_rate": 0.6
  }
}
```

## Outcome taxonomy

| outcome | meaning |
|---|---|
| `pass` | `validator.all_pass == True` — code ran and matched every train example |
| `partial` | Code ran but matched some examples, not all (exists for diagnostics; not a publish-worthy state) |
| `fail` | Code ran but matched zero examples |
| `timeout` | Subprocess or LLM call exceeded wall-clock budget |
| `error` | Other — compile error, malformed response, no `def transform` found |

## Running in CI

A portfolio like `evals/portfolios/arc_smoke.yaml` can be driven by a new CI workflow:

```yaml
- name: Smoke eval
  run: python -m evals.harness --portfolio evals/portfolios/arc_smoke.yaml --expert qwen3
```

The workflow compares `pass_rate` against a baseline stored in `evals/baselines/`; if pass rate drops more than N%, fail the build. Not yet wired; see the followup issue.

## What's NOT included (by design)

- **No end-to-end "did chat feel good" evaluation** — that's subjective and needs humans.
- **No cognitive-architecture decision evaluation** — that's issue #41, separate harness.
- **No cost aggregation** — NRP is free; tokens-per-solve lives in the memory file for the Scientist, not in this harness.
- **No wiki-note generation** — the harness reads but never writes the wiki.
