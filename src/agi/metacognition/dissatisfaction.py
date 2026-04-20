# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Phase 1 — conversation dissatisfaction detector.

Post-conversation classifier for the Knowledge Gap Mapping v1 subsystem.
Reads the tail of a conversation, decides whether the user was
``satisfied`` / ``neutral`` / ``unsatisfied``, and — if unsatisfied —
extracts a free-text topic for the aggregator to normalize and upsert
into the Unified Knowledge Graph.

Spec: ``docs/KNOWLEDGE_GAP_MAPPING_v1_spec.md``. Settled decisions that
shape this module:

- Topic normalization happens in the aggregator, not here. The detector
  returns free-text ``topic`` only; ``topic_key`` is computed once by
  the aggregator so canonicalization lives in exactly one place.
- ``None`` return means the detector failed (parse error, LLM failure,
  invalid verdict). The ``verdict == "neutral"`` return means the
  detector ran successfully and found no strong signal — distinct.
- Every event records ``detector_model`` + ``detector_version`` so
  drift and A/B comparisons are tractable without digging into logs.

This module deliberately has no hard dependency on ``openai`` or
``httpx``: the default LLM path uses ``urllib.request`` so the detector
can run inside the telemetry server without pulling extra deps. Tests
inject a stub via the ``llm_call`` kwarg.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Protocol

log = logging.getLogger("metacognition.dissatisfaction")

# ── constants ────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen3"
"""Fast + cheap + already warm on NRP — chosen per spec §1.5."""

DETECTOR_VERSION = "gap-det-0.1.0"
"""Bumped when the prompt or parsing logic changes in a way that
could shift classifier behavior. Stored on every event."""

MAX_TURNS_IN_PROMPT = 10
"""Cap on the number of conversation turns fed to the detector."""

_ELLM_URL = os.environ.get(
    "NRP_LLM_URL", "https://ellm.nrp-nautilus.io/v1/chat/completions"
)

_VERDICTS: frozenset[str] = frozenset({"satisfied", "neutral", "unsatisfied"})


@dataclass
class ConversationSignal:
    """Detector output. See spec §2 for field semantics."""

    verdict: str
    """One of ``satisfied`` / ``neutral`` / ``unsatisfied``."""

    topic: str
    """Free-text topic. Empty unless ``verdict == "unsatisfied"``.
    The aggregator derives ``topic_key`` from this via
    ``graph.normalize_topic_key``."""

    signal_turns: list[int]
    """Zero-based indices of conversation turns the detector flagged
    as carrying the signal. Empty for satisfied/neutral."""

    rationale: str
    """Short free-text audit string. Stored in the sidecar event log
    for observability only — NEVER re-fed to a generator as truth."""

    score: float
    """0.0..1.0 confidence. The aggregator's emit threshold is 0.7."""

    detector_model: str
    """Model id used for this classification."""

    detector_version: str
    """Module version at classification time."""


# ── prompt construction ──────────────────────────────────────────


_SYSTEM_PROMPT = """You are a conversation auditor for Atlas AI. Your job is to classify whether the user was satisfied with Atlas's answer in the recent conversation.

Rules:
- "satisfied" requires evidence: thanks, next-topic transition, no corrections.
- "unsatisfied" requires evidence: user correction, clarification request, repeated question, explicit dissatisfaction.
- "neutral" is the default. When in doubt, return neutral.
- Extract a short, specific topic string ONLY if the verdict is "unsatisfied". Leave topic empty otherwise.

Respond with a single JSON object and nothing else. Keys:
  "verdict":      one of "satisfied", "neutral", "unsatisfied"
  "topic":        short free-text topic, empty string if not unsatisfied
  "signal_turns": list of 0-based integer turn indices that carried the signal
  "rationale":    one short sentence explaining your decision
  "score":        float 0..1 confidence in your verdict
"""


def _format_turns(turns: list[dict], *, max_turns: int) -> str:
    """Render the tail of the conversation for the user-message body."""
    tail = turns[-max_turns:]
    offset = max(0, len(turns) - max_turns)
    lines: list[str] = []
    for i, t in enumerate(tail, start=offset):
        role = (t.get("role") or "?").strip() or "?"
        content = (t.get("content") or "").strip()
        if len(content) > 800:
            content = content[:797] + "..."
        lines.append(f"[{i}] {role}: {content}")
    return "\n".join(lines)


def _build_messages(turns: list[dict], *, max_turns: int) -> list[dict]:
    transcript = _format_turns(turns, max_turns=max_turns)
    user = (
        "Conversation transcript (indices are absolute, not tail-relative):\n\n"
        f"{transcript}\n\n"
        "Return a JSON object per the system rules."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ── response parsing ─────────────────────────────────────────────


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_GREEDY_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> dict | None:
    """Extract a JSON object from the detector's reply.

    Tolerant of prefix/suffix chatter, markdown fences, and the
    occasional trailing reasoning block. Returns ``None`` if nothing
    parseable is found — caller logs and emits nothing.
    """
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = _FENCE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = _GREEDY_JSON_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


# ── LLM transport (injectable for tests) ─────────────────────────


class LLMCallable(Protocol):
    """Minimal shape the detector needs from an LLM client."""

    def __call__(self, *, model: str, messages: list[dict]) -> str: ...


def _default_llm_call(*, model: str, messages: list[dict]) -> str:
    """Default LLM transport: NRP ellm via urllib (no extra deps).

    Raises on network or HTTP error; ``classify_conversation`` catches
    and returns ``None`` so a detector failure never interrupts the
    chat handler that called it.
    """
    import urllib.error
    import urllib.request

    token = os.environ.get("NRP_LLM_TOKEN", "")
    if not token:
        raise RuntimeError("NRP_LLM_TOKEN unset")
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 400,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _ELLM_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8", errors="replace"))
    content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
    return content or ""


# ── public entrypoint ────────────────────────────────────────────


def classify_conversation(
    *,
    conversation_id: str,
    turns: list[dict],
    ego_model: str,
    llm_call: Callable[..., str] | None = None,
    detector_model: str | None = None,
    max_turns: int = MAX_TURNS_IN_PROMPT,
) -> ConversationSignal | None:
    """Classify the outcome of a conversation.

    Returns ``None`` on any failure (empty turns, LLM error, parse
    failure, invalid verdict). A non-None return means the detector
    ran successfully — callers distinguish "no strong signal" via
    ``result.verdict == "neutral"``.

    ``conversation_id`` is not used by the classifier itself but is
    accepted here so callers can pass a single context object through
    to the downstream aggregator.

    ``ego_model`` records which backend served the conversation.
    Observability only; does not affect classifier behavior.
    """
    if not turns:
        return None

    model = detector_model or os.environ.get("EREBUS_DETECTOR_MODEL", DEFAULT_MODEL)
    caller = llm_call or _default_llm_call
    messages = _build_messages(turns, max_turns=max_turns)

    try:
        raw = caller(model=model, messages=messages)
    except Exception as e:  # noqa: BLE001 — any transport failure is a None return
        log.warning(
            "detector_llm_failed: conv=%s model=%s err=%r",
            conversation_id,
            model,
            str(e)[:200],
        )
        return None

    parsed = _parse_response(raw)
    if not isinstance(parsed, dict):
        log.warning(
            "detector_parse_failed: conv=%s raw_peek=%r",
            conversation_id,
            (raw or "")[:200],
        )
        return None

    verdict = parsed.get("verdict")
    if verdict not in _VERDICTS:
        log.warning(
            "detector_bad_verdict: conv=%s verdict=%r", conversation_id, verdict
        )
        return None

    topic_raw = parsed.get("topic") or ""
    topic = topic_raw.strip() if isinstance(topic_raw, str) else str(topic_raw).strip()

    signal_raw = parsed.get("signal_turns") or []
    if isinstance(signal_raw, list):
        signal_turns = [int(i) for i in signal_raw if isinstance(i, (int, float))]
    else:
        signal_turns = []

    rationale_raw = parsed.get("rationale") or ""
    rationale = (
        rationale_raw.strip()
        if isinstance(rationale_raw, str)
        else str(rationale_raw).strip()
    )

    try:
        score = float(parsed.get("score") or 0.0)
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))

    return ConversationSignal(
        verdict=verdict,
        topic=topic,
        signal_turns=signal_turns,
        rationale=rationale,
        score=score,
        detector_model=model,
        detector_version=DETECTOR_VERSION,
    )
