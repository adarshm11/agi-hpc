# SLOs, KPIs, and Core Success Metrics

**Resolves:** #37
**Companion:** [`METRICS_INVENTORY.md`](METRICS_INVENTORY.md) (what we measure today), [`METRICS_CONTRIBUTOR_GUIDE.md`](METRICS_CONTRIBUTOR_GUIDE.md) (how to add a metric).

This document proposes a first set of Service Level Objectives (SLOs) and Key Performance Indicators (KPIs) for Atlas AI. SLOs define "what good looks like" for operations; KPIs define progress on the research program. They aren't enforced yet — this is the target state.

---

## Design principles

1. **Prefer a small set of high-signal metrics over a large set of low-signal ones.** The dashboard is the contract; anything worth measuring belongs in a panel or an API endpoint.
2. **Measure what drives decisions.** An SLO whose breach wouldn't change anyone's behaviour isn't useful.
3. **Budget-based, not threshold-based.** "99.5% of requests ≤ X seconds over a rolling 30-day window" beats "no single request > Y." Error budgets let you plan risk, not fight false alarms.
4. **Cognitive architecture has multiple levels of evaluation.** "Did the service respond" (infra SLO) is a different question from "did Erebus solve more puzzles this week" (research KPI). Keep them separate; don't let one proxy the other.

---

## 1. Infrastructure SLOs (operator-facing)

| SLO | Target | Measurement | Breach signal |
|---|---|---|---|
| **Telemetry availability** | 99.5% of HTTP requests to `/api/*` return 2xx over 30-day rolling | Count 2xx / total, bucketed per-minute in VictoriaMetrics | > 1 min of continuous 5xx, or 30-day budget burn |
| **Telemetry latency** | p90 of `/api/erebus/status` and `/api/primer/status` ≤ 500 ms | Measure request duration per endpoint; aggregate in VictoriaMetrics | p90 > 500 ms sustained 5 min |
| **Dashboard render** | 99% of `dashboard-render.yaml` CI runs pass over 7 days | GitHub Actions history; the workflow runs every 30 min | ≥ 3 consecutive failures |
| **Deploy drift** | 0 commits/month where live SHA lags main HEAD > 10 min outside a known deploy window | `deploy-smoke.yaml` alerts; manual audit of the Actions log | Any SHA-mismatch alert > 10 min |
| **NATS leaf uptime** | Leaf-connection state is `is_spoke=true, rtt<100ms` 99.5% of 30-day window | `/leafz` scrape in telemetry | Disconnected > 5 min |
| **Atlas systemd health** | All units in `atlas.target` show `active (running)` 99% of 30-day window | `systemctl list-units --state=failed \| wc -l` | Any unit in `failed` > 10 min |

These are **operator SLOs**: if the service is up and reachable, the operator has done their job. They don't speak to whether Atlas is *good at what it does*, only whether it's *available*.

### Why these six

- Telemetry is load-bearing for the whole dashboard and all observability. If it's down, we're blind.
- Dashboard-render catches the class of failures that silently ate 16 commits (see [`CHANGELOG.md`](CHANGELOG.md) §2026-04-19 deploy-drift post-mortem). Mechanical guard against a recurring failure mode.
- Deploy drift is the other half of that post-mortem — a monitor asserting the deployed state matches intent.
- NATS leaf carries cortex ↔ subcortical coordination. Without it, everything degrades to local-only mode.
- Systemd health is the last-resort canary for "everything on Atlas is fine."

### Missing from the list (consciously)

- Chat latency. Bound by NRP ELLM variance; we can't SLO someone else's managed service. We *can* SLO our cascading-fallback *logic*: "chat handler returns an answer or a clear error within N seconds" — worth adding once the cascade lands in Phase 3.
- RAG accuracy. Not operator-scope; belongs in research KPIs.
- Primer publish rate. Varies with task queue shape, not a reliability metric.

---

## 2. Erebus / ARC Scientist KPIs (research-facing)

| KPI | Measurement | Notes |
|---|---|---|
| **Competition coverage** | `total_solves / 400` from `arc_scientist_memory.json` | The headline number. 100 / 400 (25%) as of 2026-04-19. Target: 150 / 400 (37.5%) by end of 2026. |
| **Per-cycle solve rate** | Solves in this cycle / attempts in this cycle, parsed from the `Progress: N solved in M attempts` log line | Efficiency proxy. Steady ~20% during healthy ensemble windows. |
| **Strategy win-rates** | For each strategy in `strategies` dict: `successes / attempts` | Shifts over time as prompts evolve. `primitives_guided` at 0/118 flagged for removal. |
| **Error-type mix** | Counts of `reasoning / execution / perception / specification` per recent N attempts | Shifts indicate where current attempt-cost is going. Most tasks get "reasoning" or "perception" as the dominant tag. |
| **Mentor-preamble lift** | Solve-rate delta on tasks with vs. without a `sensei_task_NNN.md` | Requires A/B infrastructure we don't have yet. Long-term project KPI. |
| **Primer verification rate** | Verified notes published / candidates attempted, per Primer tick | Signal on ensemble quality. Today: 1 verified / 3 attempted per tick = 33%. |
| **Primer expert efficiency** | Per-expert `(successes / total_calls)` from `primer_health.json` history | Informs which experts deserve more / less budget. |

### Why these seven

- Competition coverage is the unambiguous "are we winning" number. Everything else is a leading indicator.
- Per-cycle rate separates "Scientist is still learning on new tasks" from "Scientist has already solved the easy ones and is grinding the long tail."
- Strategy win-rates feed into the Thompson-sampling weights; visible decay of a strategy is decision-ready signal.
- Error-type mix tells us which kind of help is needed. If `perception` errors dominate, vision burst; if `reasoning`, more meta-articles.
- Preamble lift, when we can measure it, is THE KPI — it's the answer to "does the whole sensei mechanism work?"
- Primer metrics complete the picture for the teaching-layer experiment.

---

## 3. Cognitive architecture KPIs (long-horizon research)

These are the research-program KPIs (distinct from operational SLOs). They don't breach; they trend. Tracked across sessions, not per-minute.

| KPI | Why it matters | Current state |
|---|---|---|
| **Dual-hemisphere debate improvement over single-model** | Validates the central architectural claim | Experiment not yet run. See Phase 0 in [`AGI_ROADMAP.md`](AGI_ROADMAP.md). |
| **Metacognition calibration** | System's self-reported confidence tracks actual correctness | Disagreement metric exists; not yet correlated with ground truth. |
| **Memory retrieval quality** | RAG improves factual accuracy by ≥20% (hypothesis) | Experiment pending. |
| **Wiki growth rate** | New verified sensei notes per week | 10+ hand-written + 1 auto (Primer's first) as of 2026-04-19. Expected to accelerate as Primer runs 24/7. |
| **Auto- vs. human-sensei correctness parity** | Fraction of Primer-generated notes that later get human-corrected | Target: ≤ 5% correction rate. Incident count today: 0 (Primer has only published 1 note). |
| **Fine-tuning adapter adoption** | Whether dream-cycle LoRA adapters measurably improve ego quality | Blocked on self-hosted ego pod (Phase 2). |
| **Solve-rate vs. wiki-coverage correlation** | Does wiki-covered task family have higher solve rate than un-covered? | Would need explicit family tagging on wiki articles to measure cleanly. |

These are "does the thesis hold" metrics. They're answered through experiments, not continuous monitoring.

---

## 4. Cost / efficiency metrics

NRP is free (shared managed service); no direct dollar cost. But throughput-per-token matters for capacity planning:

| Metric | Measurement | Why |
|---|---|---|
| **Input tokens per Primer tick** | Sum of `usage.prompt_tokens` across all expert calls in a tick | Context-size drift detection; caught the 196K-token bug in the first Primer smoke |
| **Output tokens per verified note** | `usage.completion_tokens` of the winning ensemble response | Typical 2-5K for a sensei note; outlier > 15K suggests runaway CoT |
| **Wall-time per Primer tick** | Time from `tick start` log to `tick complete` | Degrades during NRP load; health tracker should compensate |
| **Tokens per solve (ARC Scientist)** | `usage.completion_tokens` summed over all attempts on a task that ultimately solved | Long-term efficiency measure; improves as sensei coverage grows |

These are tracked in each service's logs; VictoriaMetrics could aggregate if we bothered wiring them. Not urgent.

---

## 5. Error budget policy

**Budget calculation** — for each infra SLO, monthly error budget = `(1 - SLO) × 30 days`:

- 99.5% SLO → 3.6 hours/month of allowable downtime or degradation
- 99.0% SLO → 7.2 hours/month
- 99.9% SLO → 43.2 minutes/month

**Budget enforcement** (proposed, not yet automated):

1. When an SLO burns 50% of its monthly budget with > 15 days remaining → operator is notified (GitHub issue auto-opened by CI).
2. When an SLO burns 100% of its budget → feature freeze on anything that touches that service until the budget window rolls over.
3. After a rollover, budget resets. Any recurring breach pattern over multiple months → revise the SLO target or fix the root cause.

We're nowhere near the automation for this yet. Current state: eyeball `gh run list` and dashboard periodically.

---

## 6. What "done" looks like

- Every SLO in §1 has a visible panel on the dashboard showing actual-vs-target.
- Every KPI in §2 is exposed via `/api/erebus/status` and historizable via VictoriaMetrics.
- `METRICS_CONTRIBUTOR_GUIDE.md` describes how a contributor adds a new metric without polluting the namespace.
- Post-incident reviews can reference specific SLO breaches rather than "it felt slow."
- Research experiments (Phase 0 of the roadmap) produce numbers against §3 KPIs.

None of §5 or much of §1 is automated today. This doc describes the target; [`METRICS_INVENTORY.md`](METRICS_INVENTORY.md) §8 describes the gap.

---

## 7. Out of scope (explicitly)

- **Model accuracy benchmarks (MMLU, HumanEval, etc.)** — these evaluate the *underlying LLMs*, not Atlas. NRP's model choice drives those numbers; we observe, we don't own.
- **Safety incident counts (ErisML violations, etc.)** — separate SLO family, documented in the safety subsystem docs.
- **Per-user / per-tenant metrics** — Atlas is single-tenant today. Multi-tenancy isn't on the roadmap.
- **Compliance audits (SOC 2, etc.)** — academic / research context, not production commercial.

---

## Appendix: proposed SLI → SLO derivations

For the audit-minded, the measurement formulas behind each SLO:

**Telemetry availability (99.5%, 30 days):**

    SLI = sum_over(30d) { count(2xx) } / sum_over(30d) { count(all) }
    SLO threshold: SLI ≥ 0.995
    Monthly budget: (1 - 0.995) × 30d × 24h = 3.6 hours

**Telemetry latency (p90 ≤ 500 ms, per-minute):**

    SLI = histogram_quantile(0.90, sum(rate(http_duration_seconds_bucket{path=~"/api/erebus/status|/api/primer/status"}[1m])) by (le))
    SLO threshold: SLI ≤ 0.5s for each 1-min bucket
    Window: rolling 30 days; tolerate breach if < 5% of 1-min buckets exceed

**Dashboard render (99% pass rate, 7 days):**

    SLI = count(runs where status='success', last 7d) / count(all runs, last 7d)
    SLO threshold: SLI ≥ 0.99
    Cron cadence: every 30 min → ~336 runs/week

These are the formulas a Grafana dashboard / alert rule would encode when we build them.
