# Knowledge Gap Mapping v1 — Implementation Specification

**Status:** draft for review (authored 2026-04-19 after the UKG v1 spec was locked).
**Purpose:** define a shippable v1 for detecting, storing, clustering, and prioritizing "things Atlas has been asked about but answered poorly," using the Unified Knowledge Graph as the storage substrate.

This spec mirrors the structure of `unified_knowledge_graph_v1_spec.docx` and is subject to the same lock-before-code review cadence.

## Design intent

Atlas today accumulates verified knowledge (sensei notes) and open tasks (help queue). It does **not** record conversations where the user expressed dissatisfaction, asked for clarification, or corrected Atlas. Roadmap item 2.3 ("Knowledge Gap Mapping") calls for surfacing those signals so the dreaming subsystem can prioritize consolidation around weak areas and a future curiosity module can autonomously seek information in those gaps.

**Design principle:** reuse the UKG. A dissatisfaction-derived gap is a `type=gap` node with `source="dissatisfaction"`. The graph is already shaped to hold it. What we are adding is a **producer** (the detector) and **consumers** (clustering, dreaming prioritizer, dashboard), not a second store.

**Load-bearing invariants (carried over from the UKG spec):**

- Only `filled ∧ verified ∧ active` nodes are teaching context. Dissatisfaction gaps are never fed back to a generator as truth.
- Append-only JSONL with full-state snapshots per write.
- The centralized trust gate (`is_context_eligible`) is the single place that decides what is safe for a generator to consume.

## 1. Scope and decisions

Items marked **D** are decisions I propose; they are open for your review and lock.

1. **D — Detector runs post-conversation, not inline.** End-of-conversation hook reads the final N turns, classifies the outcome, and emits zero or one gap node per conversation. No inline token-level cost during chat.
2. **D — Classification is three-way, not binary.** `{satisfied, neutral, unsatisfied}`. Only `unsatisfied` triggers a gap emit. `neutral` is explicitly left alone — the silence of "no strong signal" is different from "user was happy."
3. **D — One gap per conversation, keyed by conversation id.** Avoids exploding the graph when a single hostile conversation has multiple clarification requests. The gap's `evidence` field carries each individual signal (`conv:abc123:turn-4`, `conv:abc123:turn-7`).
4. **D — Topic extraction is free-text, clustered periodically.** Matches the UKG's `topic` + `topic_key` model. Detector writes whatever string best describes the topic; a separate nightly clustering job collapses near-duplicates (Levenshtein or embedding-similarity). Controlled vocab is explicitly rejected (UKG lesson #2).
5. **D — Detector is a small LLM call, not a classifier heuristic.** The same ego/superego/council stack is already warm. A lightweight prompt to one of them is cheaper than building and maintaining a dedicated classifier, and it can extract the topic in the same call.
6. **D — Gaps persist indefinitely; staleness is computed, not destructive.** A `last_signal_at` timestamp on each node drives a `staleness_score` in the dashboard. Old gaps fade visually but are not deleted — deletion loses signal (UKG lesson #4).
7. **D — Atlas-originated conversations only for v1.** The Erebus chat handler (`_erebus_chat` in `scripts/telemetry_server.py`) is the one in-scope source. Other producers (a future Primer `/ask` endpoint, Claude-Code sessions, Jupyter history) are out of scope until v2 — the detector module is written with a stable `classify_conversation(turns, …)` signature so new sources plug in without a protocol change.

## 2. Detector contract

A single function with a stable signature. Implementations can be swapped out later (LLM → fine-tuned classifier → rules) without touching the graph.

```python
def classify_conversation(
    *,
    conversation_id: str,
    turns: list[dict],   # [{role: "user"|"assistant", content: str, ts: float}]
    model_used: str,     # ego backend at the time — observability only
) -> ConversationSignal | None:
    ...
```

`ConversationSignal` returns:

| Field | Type | Purpose |
|---|---|---|
| `verdict` | `"satisfied" | "neutral" | "unsatisfied"` | three-way classification |
| `topic` | `str` | human-readable topic description |
| `topic_key` | `str` | derived via graph.normalize_topic_key |
| `signal_turns` | `list[int]` | indices of turns that carried the dissatisfaction signal |
| `rationale` | `str` | short free-text explanation (stored for audit, not re-fed to generator) |
| `score` | `float` | 0..1 strength of the verdict; 1.0 for overt dissatisfaction |

Returning `None` means "inconclusive — do not emit a gap." Use this liberally; false-negatives are fine, false-positives pollute the graph.

### Detector prompt sketch

```
System: You are a conversation auditor for Atlas AI. Classify whether
the user was satisfied with Atlas's answer.

Rules:
- "satisfied" requires evidence: thanks, next-topic transition, no corrections.
- "unsatisfied" requires evidence: user correction, clarification request,
  repeated question, explicit dissatisfaction.
- "neutral" is the default. When in doubt, return neutral.
- Extract a topic if and only if the verdict is "unsatisfied."

User: <last 10 turns of the conversation>

Return: {"verdict": "...", "topic": "...", "signal_turns": [...], "rationale": "..."}
```

## 3. Gap emitter

Thin wrapper over `agi.knowledge.graph.upsert_node`:

```python
def emit_gap(sig: ConversationSignal, *, graph_path: Path | None = None) -> None:
    if sig.verdict != "unsatisfied":
        return
    evidence = [f"conv:{sig.conversation_id}:turn-{i}" for i in sig.signal_turns]
    upsert_node(
        id=f"gap_{sig.topic_key}",   # shared across conversations with same topic
        type="gap",
        status="active",
        topic=sig.topic,
        topic_key=sig.topic_key,
        tags=["dissatisfaction", *_auto_tags(sig)],
        title=f"[gap] {sig.topic}",
        body_ref=None,
        verified=False,
        source="dissatisfaction",
        evidence=evidence,
        path=graph_path,
    )
```

Unlike help-queue-sourced gaps (one per task number), dissatisfaction gaps are **keyed by topic**. Each new conversation signaling the same topic adds to the node's `evidence[]` rather than creating a new node, so a frequently-asked-but-poorly-answered topic accumulates weight naturally.

## 4. Clustering

A nightly job reduces near-duplicate topic keys:

```python
def cluster_topics(path: Path | None = None, *, threshold: float = 0.85) -> list[Cluster]:
    """Return equivalence classes of topic_keys that should be merged."""
```

v1 uses character-level Jaro-Winkler distance on the topic_key strings. v2 may upgrade to sentence-embedding similarity from the existing PCA-384 index, but only if Jaro-Winkler proves insufficient in practice.

Clusters are reported, not auto-merged. The dashboard shows proposed merges; an operator confirms each one. This keeps the graph's history clean and reversible.

## 5. Consumers

### 5.1 Dashboard (extension of Phase 5 card)

The existing "Knowledge Graph" panel gains two new rows:

- **Top dissatisfaction topics** — a filtered view of `top_topics_by_gap` restricted to `source=="dissatisfaction"`, sorted by `evidence[]` length (weight).
- **Recent dissatisfaction signals** — last 5 events, each one a line with conversation id + topic + age.

Implementation: extend `graph.summary` to take an optional `source_filter` kwarg and return a parallel `top_dissatisfaction_topics` list.

### 5.2 Dreaming prioritizer

The dreaming subsystem currently consolidates memory without awareness of where Atlas is weak. The v1 integration:

```python
def dreaming_priority(path: Path | None = None, *, top_n: int = 5) -> list[str]:
    """Return the top-N topic_keys to rehearse during the next dream cycle."""
```

Ranks `type=gap ∧ source=dissatisfaction` nodes by `len(evidence) * recency_weight(last_signal_at)`. Dreaming reads this list, retrieves relevant sensei notes via the UKG's topic index, and generates synthesis prompts for the next consolidation round.

### 5.3 Curiosity (Phase 4, explicitly out of scope here)

Reserved for a future spec. The curiosity module will query the same gap list and autonomously seek information (web search, paper retrieval, tool use) to close the gap. Not in v1.

## 6. Delivery phases for v1

| Phase | Component | Deliverable |
|---|---|---|
| 1 | Detector module | `src/agi/metacognition/dissatisfaction.py` + unit tests against recorded conversations. |
| 2 | Gap emitter | `src/agi/metacognition/gap_emitter.py` wired to UKG; idempotent re-runs. |
| 3 | Conversation hook | Hook the detector into `_erebus_chat` in `scripts/telemetry_server.py` so it runs at end-of-turn. |
| 4 | Clustering | `src/agi/knowledge/clustering.py` + nightly cron job + dashboard review surface. |
| 5 | Dashboard rows | Extend `graph.summary` + `schematic.html` UKG panel. |
| 6 | Dreaming priority | `dreaming_priority()` + integration into the existing dream scheduler. |

Each phase is a separate commit, tested in isolation, green on CI before the next.

## 7. Out of scope for v1

- Non-Atlas conversation sources (Claude-Code sessions, Jupyter, shell history).
- Auto-merge of topic clusters (human-in-the-loop only for v1).
- Curiosity module (Phase 4 of the roadmap, separate spec).
- Sentiment embedding models (Jaro-Winkler for clustering suffices; upgrade gated on data).
- Re-classifying historical conversations (v1 is forward-looking only).

## 8. Acceptance criteria

1. A fresh conversation that ends in user dissatisfaction produces exactly one `gap` node with `source="dissatisfaction"`.
2. A second conversation on the same topic appends to the existing node's `evidence[]` rather than creating a duplicate.
3. A satisfied or neutral conversation produces zero gap nodes.
4. The nightly clustering job reports proposed merges without mutating the graph, and operator-confirmed merges produce valid node lineage.
5. The dashboard surfaces top dissatisfaction topics ordered by weighted evidence count.
6. The dreaming priority list is stable across two dreamless polls (no noise in ordering) and responds to a fresh dissatisfaction signal by re-prioritizing within one tick.
7. No dissatisfaction-sourced gap is ever returned by `is_context_eligible()` — the existing trust gate holds.

## 9. Open questions for review

1. **Detector model choice.** Ego (GLM-4.7) is flagship but expensive per call. Qwen3 is faster and cheaper. Propose: default to Qwen3, with an `EREBUS_DETECTOR_MODEL` env override. OK?
2. **Conversation sampling.** Should we classify every conversation, or sample (e.g., 1 in 5)? Lean: classify every conversation — cost is bounded by chat volume, which is not high, and missing a real signal is worse than the small compute overhead.
3. **Storage format for the raw rationale.** Option A: stored in `evidence[]` as free text (cheap, but evidence[] was designed for provenance handles, not content). Option B: a parallel `dissatisfaction_events.jsonl` keyed by conversation id that the graph references via `evidence[]`. Lean: Option B — respects the UKG node-as-lightweight-index principle.
4. **Gap node id scheme.** `gap_<topic_key>` shares nodes across conversations. Alternative: `gap_conv_<conversation_id>` makes each conversation its own node. Lean: topic-keyed — conversations-as-their-own-nodes would miss the "frequently asked" signal that makes clustering useful.
5. **Dashboard card or panel extension?** Lean: extension of existing UKG panel, not a new card. Keeps the dashboard from fragmenting.

## 10. Recommended first tests (implementation sequence)

1. Classify a known-satisfied conversation → `verdict=="satisfied"`, no emit.
2. Classify a known-unsatisfied conversation → gap emitted with correct topic.
3. Two unsatisfied conversations on the same topic → one node with two evidence entries.
4. Two unsatisfied conversations on different topics → two distinct nodes.
5. Verify the emitted gap is excluded from `is_context_eligible()` — no leak path.
6. Clustering: three near-duplicate topic keys collapse to one proposed merge.
7. Dreaming priority on an empty graph returns `[]` without error.
