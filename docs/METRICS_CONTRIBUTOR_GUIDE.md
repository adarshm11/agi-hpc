# Metrics & Evaluation Contributor Guide

**Resolves:** #42
**Audience:** anyone adding observability to an Atlas service or writing an evaluation script.
**Companions:** [`METRICS_INVENTORY.md`](METRICS_INVENTORY.md) (what exists), [`SLOS_AND_KPIS.md`](SLOS_AND_KPIS.md) (what we target).

This guide is how to add metrics and logs *correctly* — without breaking existing panels, polluting the namespace, or shipping observability that nobody can consume.

---

## Decision tree — does my change need a metric?

```
Am I adding a new service, endpoint, or long-running loop?
├── Yes → you need at least a heartbeat + a basic counter. See §2.
│
Am I changing an existing service's behaviour in a way operators might
need to debug later?
├── Yes → structured log the key events. See §3. Consider if it deserves
│         a metric (§2) or a separate panel.
│
Am I adding a one-off script or tool?
├── No metrics required, but log to stdout/stderr cleanly. See §5.
│
Am I adding a new subsystem (like the Primer)?
└── Yes → you likely need: heartbeat, a status JSON file, a /api/ endpoint
          exposing it, a synthetic NATS-topology node, and a dashboard
          card. See §6 for the full checklist.
```

---

## 1. Metric naming conventions

### 1.1 Namespace

All metrics that will eventually go into VictoriaMetrics or a Prometheus exporter follow this pattern:

    atlas_<subsystem>_<object>_<unit>

Examples:

    atlas_telemetry_http_request_duration_seconds
    atlas_primer_verify_duration_seconds
    atlas_primer_notes_published_total
    atlas_scientist_solves_total
    atlas_nats_leaf_rtt_seconds
    atlas_ego_chat_tokens_out_total

### 1.2 Suffix rules (Prometheus convention, since we use VictoriaMetrics)

- `_total` — monotonic counters. Only goes up.
- `_seconds`, `_bytes`, `_count` — gauges with a unit.
- `_bucket`, `_count`, `_sum` — emitted automatically by a histogram.
- `_info` — labels-only gauge with constant value 1 (for metadata like `version`).

### 1.3 Labels

- Low cardinality only. Never `user_id`, `task_id`, `commit_sha` as a label (put those in logs).
- Common acceptable labels: `subsystem`, `expert`, `strategy`, `endpoint`, `status`.
- Avoid unbounded label values. If a label can take > 100 distinct values, it's a tag for logs, not a metric dimension.

### 1.4 Don't invent new prefixes

`atlas_*` is the agreed prefix for this project. Don't ship `mymetric_*` or `primer_total` — they won't show up in the dashboards that filter on the `atlas_` prefix.

---

## 2. Adding a metric

We don't run a Prometheus Python client in-process today; metrics are scraped by `telemetry_server.py` from live OS state + in-memory dicts. Two shapes:

### 2.1 System-derived metrics (OS state)

For anything visible to `/proc`, `nvidia-smi`, `ss`, `df`, etc. — add a scrape in `scripts/telemetry_server.py`:

```python
def _get_my_metric():
    try:
        out = _run(["my-tool", "--json"])
        data = json.loads(out)
        return {"my_metric_value": data.get("value", 0)}
    except Exception:
        return {"my_metric_value": 0}
```

Wire it into `get_cached_telemetry()` so `/api/telemetry` returns it. Add a panel to `schematic.html` if appropriate.

### 2.2 In-process metrics (service-internal)

For state that only the process knows — e.g. the Primer's health tracker, the Scientist's solve counts — use the pattern already in place:

1. Maintain the metric as an instance attribute (like `vMOE.health` uses `HealthTracker`).
2. Periodically write a JSON snapshot to `/archive/neurogolf/<subsystem>_<name>.json`.
3. `telemetry_server.py` reads the file in its `/api/<subsystem>/status` handler.
4. The dashboard polls the endpoint and renders.

Why file-based? It's simple, survives service restarts, doesn't require a shared process / IPC, and is trivially debuggable (`cat` the file). VictoriaMetrics can scrape the JSON-exposing endpoints directly.

### 2.3 Structured logging

For events (not metrics — events have one-off payloads and aren't aggregated over time windows):

```python
log.info(
    "primer.publish",
    extra={
        "task_num": task_num,
        "expert": r.expert,
        "latency_s": r.latency_s,
        "verified": True,
    },
)
```

Today we log in plain-text format and grep. We should move to JSON logging (see §3.2) but haven't yet — the contribution bar is "log enough fields that grep works on the plain-text form."

---

## 3. Log field conventions

Until we move to JSON logging, the convention for human-readable log lines is:

    <subsystem>.<event>: <human summary> (k=v k=v ...)

Examples:

    primer.publish: task020 via qwen3 (latency_s=142.1 family=symmetry-completion)
    scientist.solve: task056 via diagnostic+kimi (score=46/46 attempt=11/400)
    ego.chat_fallback: kimi timeout → glm-4.7 (elapsed_s=61.2 reason=NRP-load)

### 3.1 Field conventions (applied to both plain-text k=v and future JSON)

| Field | Semantics | Examples |
|---|---|---|
| `subsystem` | Which top-level module | `primer`, `scientist`, `ego`, `telemetry` |
| `event` | Short verb for what happened | `publish`, `solve`, `fail`, `restart`, `timeout` |
| `task_num` | ARC task number if relevant | `20`, `056`, `381` |
| `expert` | vMOE expert name | `kimi`, `glm-4.7`, `qwen3` |
| `strategy` | arc_scientist strategy | `direct`, `diagnostic`, `example_chain` |
| `latency_s` | Duration of this event, seconds | `142.1` |
| `score` | `correct/total` for an ARC attempt | `46/46`, `0/1` |
| `reason` | Short cause for failures | `timeout`, `no-code`, `mismatch`, `NRP-load` |

Don't invent new field names casually. If your event genuinely doesn't fit existing fields, document the addition in the relevant service's module docstring.

### 3.2 Future — JSON logging

When we migrate to `python-json-logger`, the above conventions become machine-consumable directly. The plain-text format is chosen to be a subset of what JSON logs will carry — migration is opt-in per service and existing logs stay grep-compatible.

---

## 4. Adding a new API endpoint for observability

If you're exposing a new dashboard signal:

### 4.1 Endpoint naming

    /api/<subsystem>/<noun>

Where `<noun>` is a concrete thing:
- `/api/primer/status` — process liveness + health state
- `/api/erebus/status` — scientist progress
- `/api/version` — current git SHA + mtime

### 4.2 Response shape

Always JSON. Always a single top-level object (not a bare array). Always include enough context that a reader without prior knowledge can interpret:

```json
{
  "running": true,
  "tasks_touched": 3,
  "last_touched_task": 46,
  "last_touched_age_s": 142,
  "expert_health": {
    "kimi": { "healthy": false, "degraded_until_s": 1200, ... }
  }
}
```

Don't return `true` or `[1,2,3]` as the top-level payload. Callers benefit from named fields.

### 4.3 Caching

If the backend scrape is expensive, cache in `telemetry_server.py`'s dict-based cache with a short TTL (5-30 s typical). Example: `_ui_version_cache` caches `git rev-parse` for 15 s.

### 4.4 CORS / auth

Telemetry endpoints are behind Caddy + OAuth2. No additional auth needed per-endpoint; if you add a write endpoint (POST), think carefully — `/api/erebus/result` is the only write today and it's behind the same oauth layer.

---

## 5. One-off scripts / tools

For scripts under `scripts/`:

### 5.1 Log to stdout with a clear level prefix

```python
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("my-tool")
log.info("starting")
```

### 5.2 Exit non-zero on failure

Tools consumed by CI or systemd use the exit code to decide next steps. Don't `print("FAIL")` and exit 0.

### 5.3 If the tool produces artifacts

- Write to `/archive/neurogolf/<tool-name>/` (not `/tmp` — doesn't survive reboots) — ephemeral but retained.
- Use a timestamped filename: `artifact_2026-04-19T15:30Z.json`.
- Log the path of every artifact you write, so log consumers can find it.

---

## 6. New-subsystem checklist

When adding a substantial new subsystem (size of "The Primer" or larger):

- [ ] Module under `src/agi/<name>/` with `__init__.py` exporting the public surface.
- [ ] Unit tests under `tests/unit/test_<name>_*.py` that don't require network.
- [ ] Systemd unit at `deploy/systemd/atlas-<name>.service` if it's long-running.
- [ ] Env-file support via `EnvironmentFile=-/home/claude/.<name>.env` (optional `-` prefix).
- [ ] Status file written by the service to `/archive/neurogolf/<name>_<thing>.json`.
- [ ] Status endpoint `/api/<name>/status` in `telemetry_server.py`.
- [ ] Frontend poller `poll<Name>()` in `schematic.html` with `setInterval`.
- [ ] Synthetic node in the NATS topology panel (see the pattern in `schematic.html` line ~2050).
- [ ] Entry in [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) §"Core components."
- [ ] Design doc at `docs/<NAME>.md` if the subsystem has non-obvious internals.
- [ ] CHANGELOG.md entry under the current date.

The Primer is the reference implementation of the whole checklist. `src/agi/primer/`, `tests/unit/test_primer_*.py`, `deploy/systemd/atlas-primer.service`, `/api/primer/status`, `pollPrimer()`, synthetic node "The Primer," [`THE_PRIMER.md`](THE_PRIMER.md). Copy that pattern.

---

## 7. Evaluation artifacts

If you're building evaluation infra (benchmarks, test harnesses, ARC oracles):

### 7.1 Repeatability

- Store inputs and expected outputs as JSON / YAML in `benchmarks/<name>/fixtures/`.
- Emit results as JSON, not just stdout. Results JSON should include: timestamp, git SHA, input hash, pass/fail per case, aggregates.
- Include the code version + model version (if calling an LLM) in every result record.

### 7.2 Determinism

- Fix random seeds at the top of every benchmark script (`random.seed(...)`, `numpy.random.seed(...)`, `torch.manual_seed(...)`).
- For LLM calls, set `temperature=0` unless you're explicitly measuring variance.
- Note non-determinism explicitly in the result record if it applies (e.g. NRP managed endpoint is shared and has variance that isn't controlled by our seed).

### 7.3 Scoring

- One score per case, plus a rollup. Don't invent new aggregation metrics — use mean / p50 / p90 / p95 / p99.
- If a benchmark has multiple dimensions (accuracy, latency, tokens), emit all three; don't collapse them early.

### 7.4 Where to put evaluation code

- `benchmarks/<name>/` — standalone benchmark driver + fixtures.
- `tests/integration/test_<name>_eval.py` — if it runs as part of CI.
- `tests/unit/` — only for pure-logic tests, no network.

---

## 8. What to avoid

- **Unstructured `print` statements left in service code.** Convert to `log.info(...)` with a level.
- **Metrics with unbounded cardinality.** `task_num` as a label will blow up VictoriaMetrics indexes.
- **Endpoints that return different shapes on success vs error.** Always the same top-level object; error cases have known error fields (e.g. `{error: "..."}`) not arbitrary types.
- **Writing to `/tmp/` for anything you want to survive a reboot.** Use `/archive/`.
- **Hand-rolling a pyproject parser or a JSON parser.** Use stdlib.
- **Adding a new service without a status endpoint.** If it's worth running, it's worth monitoring.
- **Depending on "the chat handler" or "the dashboard poll" as implicit test oracles.** They'll change and break your check; write a dedicated unit test.
- **Silent failure modes.** If a scraper's subprocess fails, return a clearly-flagged empty payload (`{"error": "..."}`) rather than `{}` — consumers need to know the difference between "data is zero" and "we couldn't collect."

---

## 9. Review checklist (for PRs adding metrics or evaluation)

- [ ] Follows the `atlas_<subsystem>_<object>_<unit>` naming (if metric) or `<subsystem>.<event>` format (if log)
- [ ] Label cardinality bounded (if metric)
- [ ] Exposed via a status endpoint, status file, or log — at least one
- [ ] Unit test covers the scrape / emit path
- [ ] Dashboard panel added if the metric is user-visible
- [ ] Entry in [`METRICS_INVENTORY.md`](METRICS_INVENTORY.md) updated
- [ ] If new SLO/KPI, entry in [`SLOS_AND_KPIS.md`](SLOS_AND_KPIS.md) updated
- [ ] `schematic.html` changes verified via hard-refresh + local render

---

## 10. When in doubt

Ask: "what would I look at during an incident?" If the thing you're adding wouldn't help an operator diagnose a problem at 3am, it's probably not worth the observability surface. Metrics and logs are a tax on the people reading them; spend the tax on signals that actually drive decisions.
