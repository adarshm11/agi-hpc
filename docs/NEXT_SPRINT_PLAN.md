# Next Sprint Plan — Cognitive Architecture Improvements

## Sources
- The Synthetic Mind (joshferrer1/the-synthetic-library)
- Karpathy LLM Wiki pattern
- Home23 cognitive engine

## 1. Ebbinghaus Forgetting Curves for Episodic Memory
**Source:** Synthetic Mind Ch 4 — Memory That Forgets
**What:** Add time-based decay to episodic memory. Episodes lose retrieval
weight over time unless reinforced by recall or high quality scores.
**Where:** `src/agi/memory/episodic/store.py`
**Formula:** `retention = e^(-t/S)` where S = stability (increases with
each successful recall). High quality_score episodes decay slower.
**Impact:** Dreaming/consolidation becomes meaningful — promote surviving
episodes to semantic memory, prune decayed ones.

## 2. Curiosity Drive for Research Loop
**Source:** Synthetic Mind Ch 5 — Motivation & Drives
**What:** Wire the research loop as an intrinsic curiosity drive that
accumulates "information hunger" over time. The longer a gap persists
without resolution, the higher its priority becomes.
**Where:** `src/agi/metacognition/research_loop.py`
**Design:** ResearchGoal.priority increases with age. Cooldown decreases
when many gaps are detected. System becomes "more curious" when idle.

## 3. WikiPage Compiled Knowledge
**Source:** Karpathy LLM Wiki
**What:** Maintain synthesized markdown pages per topic in the knowledge
graph. Each ingestion updates relevant pages, maintaining cross-references
and flagging contradictions. Pages are the "compounding artifact."
**Where:** `src/agi/memory/knowledge/wiki.py` (new)
**Design:** WikiPage dataclass with topic, content, last_updated,
references. KnowledgeGraph.compile_page(topic) regenerates from entities
and relationships. Lint checks for stale pages.

## 4. GOAP Planner for Executive Function
**Source:** Synthetic Mind Ch 7 — Decision-Making
**What:** Replace the simple complexity heuristic in ExecutiveFunction
with a Goal-Oriented Action Planning (GOAP) system that can decompose
complex queries into sub-goals with preconditions.
**Where:** `src/agi/metacognition/executive_function.py`
**Design:** Goal = desired state. Actions = available tools/queries.
Planner finds action sequence satisfying preconditions.

## 5. Temporal Cognition / Pacing
**Source:** Synthetic Mind Ch 12
**What:** Add boredom detection (repeated similar queries), patience
modeling (longer responses for complex questions), and behavioral
pacing (don't overwhelm the user).
**Where:** `src/agi/metacognition/monitor.py` or new `temporal.py`

## 6. Cross-Subsystem Failure Cascades
**Source:** Synthetic Mind Ch 13 — Integration & Failure
**What:** When one subsystem degrades (e.g., semantic memory slow),
detect and gracefully degrade dependent subsystems rather than
propagating errors.
**Where:** `src/agi/metacognition/consistency_checker.py`
**Design:** Dependency graph between subsystems. When a health check
fails, mark downstream dependencies as degraded. Executive function
adjusts routing accordingly.

## 7. Google's Recommended Sampling Params
**Source:** r/LocalLLaMA Gemma 4 troubleshooting
**What:** Test Google's recommended `temp=1.0, top_p=0.95, top_k=64`
for council/creative tasks instead of our current temp=0.1.
**Where:** `atlas-rag-server.py`, council agent configs
**Note:** Keep temp=0.1 for factual/triage tasks.

## Priority Order
1. WikiPage compiled knowledge (extends current sprint work)
2. Ebbinghaus forgetting curves (high impact, moderate effort)
3. Curiosity drive (small change to research loop)
4. Temporal cognition (new but small)
5. Cross-subsystem cascades (extends existing checker)
6. Google sampling params (config change, needs testing)
7. GOAP planner (large effort, save for later)
