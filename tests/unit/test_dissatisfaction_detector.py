"""Unit tests for agi.metacognition.dissatisfaction (Gap Mapping Phase 1).

Covers the detector's parsing and dispatch logic. The LLM transport is
injected via the ``llm_call`` kwarg so no network traffic happens.

Spec: docs/KNOWLEDGE_GAP_MAPPING_v1_spec.md, §2 and §10.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from agi.metacognition.dissatisfaction import (
    DEFAULT_MODEL,
    DETECTOR_VERSION,
    MAX_TURNS_IN_PROMPT,
    ConversationSignal,
    _build_messages,
    _format_turns,
    _parse_response,
    classify_conversation,
)

# ── helpers ──────────────────────────────────────────────────────


def _stub_llm(reply: str) -> Callable[..., str]:
    """Build an llm_call stub that returns a fixed reply."""
    captured: dict = {}

    def _call(*, model: str, messages: list[dict]) -> str:
        captured["model"] = model
        captured["messages"] = messages
        return reply

    _call.captured = captured  # type: ignore[attr-defined]
    return _call


def _raising_llm(exc: Exception) -> Callable[..., str]:
    def _call(*, model: str, messages: list[dict]) -> str:
        raise exc

    return _call


def _turns_satisfied() -> list[dict]:
    return [
        {"role": "user", "content": "what is the rank of a matrix?"},
        {
            "role": "assistant",
            "content": "Rank is the dimension of the column space.",
        },
        {"role": "user", "content": "thanks, makes sense"},
    ]


def _turns_unsatisfied() -> list[dict]:
    return [
        {"role": "user", "content": "what is the rank of a matrix?"},
        {"role": "assistant", "content": "Rank is the number of rows."},
        {"role": "user", "content": "no, that's not quite right — try again?"},
        {"role": "assistant", "content": "It's the size of the matrix."},
        {"role": "user", "content": "still wrong. rank is about independent columns."},
    ]


# ── prompt formatting ───────────────────────────────────────────


def test_format_turns_tail_respects_max():
    turns = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
    rendered = _format_turns(turns, max_turns=5)
    assert "[15] user: msg 15" in rendered
    assert "[19] user: msg 19" in rendered
    assert "msg 14" not in rendered


def test_format_turns_truncates_long_content():
    turns = [{"role": "user", "content": "x" * 1500}]
    rendered = _format_turns(turns, max_turns=10)
    # Truncation keeps the line readable
    assert "..." in rendered
    assert len(rendered) < 1500


def test_build_messages_has_system_and_user():
    msgs = _build_messages(_turns_satisfied(), max_turns=10)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "thanks" in msgs[1]["content"]


# ── response parsing ────────────────────────────────────────────


def test_parse_response_plain_json():
    out = _parse_response('{"verdict": "satisfied", "score": 0.9}')
    assert out == {"verdict": "satisfied", "score": 0.9}


def test_parse_response_markdown_fence():
    raw = 'here you go:\n```json\n{"verdict": "neutral"}\n```\nhope that helps'
    assert _parse_response(raw) == {"verdict": "neutral"}


def test_parse_response_greedy_fallback():
    raw = '<think>let me reason</think>\n{"verdict": "unsatisfied", "topic": "x"}\n</final>'
    out = _parse_response(raw)
    assert out and out["verdict"] == "unsatisfied"


def test_parse_response_empty_returns_none():
    assert _parse_response("") is None
    assert _parse_response(None) is None  # type: ignore[arg-type]


def test_parse_response_malformed_returns_none():
    assert _parse_response("this is not json at all") is None


# ── classify_conversation: happy paths ──────────────────────────


def test_satisfied_conversation_returns_satisfied():
    llm = _stub_llm(
        json.dumps(
            {
                "verdict": "satisfied",
                "topic": "",
                "signal_turns": [],
                "rationale": "user thanked and moved on",
                "score": 0.9,
            }
        )
    )
    sig = classify_conversation(
        conversation_id="c1",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is not None
    assert sig.verdict == "satisfied"
    assert sig.topic == ""
    assert sig.signal_turns == []
    assert sig.score == 0.9
    assert sig.detector_model == DEFAULT_MODEL
    assert sig.detector_version == DETECTOR_VERSION


def test_unsatisfied_returns_topic_and_signal_turns():
    llm = _stub_llm(
        json.dumps(
            {
                "verdict": "unsatisfied",
                "topic": "matrix rank — rows vs columns",
                "signal_turns": [2, 4],
                "rationale": "user corrected assistant twice",
                "score": 0.85,
            }
        )
    )
    sig = classify_conversation(
        conversation_id="c2",
        turns=_turns_unsatisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is not None
    assert sig.verdict == "unsatisfied"
    assert sig.topic == "matrix rank — rows vs columns"
    assert sig.signal_turns == [2, 4]
    assert sig.score == 0.85


def test_neutral_is_distinct_from_none():
    """Neutral means 'detector ran, no strong signal' — distinct from
    None which means 'detector failed.' Both produce no emit but the
    aggregator / dashboard treat them differently."""
    llm = _stub_llm(
        json.dumps(
            {
                "verdict": "neutral",
                "topic": "",
                "signal_turns": [],
                "rationale": "brief exchange, no explicit signal",
                "score": 0.5,
            }
        )
    )
    sig = classify_conversation(
        conversation_id="c3",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is not None
    assert sig.verdict == "neutral"


# ── failure paths (all must return None) ────────────────────────


def test_empty_turns_returns_none():
    sig = classify_conversation(
        conversation_id="c4",
        turns=[],
        ego_model="glm-4.7",
        llm_call=_stub_llm("irrelevant"),
    )
    assert sig is None


def test_llm_exception_returns_none(caplog):
    caplog.set_level(logging.WARNING, logger="metacognition.dissatisfaction")
    sig = classify_conversation(
        conversation_id="c5",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=_raising_llm(RuntimeError("timeout")),
    )
    assert sig is None
    assert any("detector_llm_failed" in m for m in caplog.messages)


def test_malformed_json_returns_none(caplog):
    caplog.set_level(logging.WARNING, logger="metacognition.dissatisfaction")
    sig = classify_conversation(
        conversation_id="c6",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=_stub_llm("absolutely not json"),
    )
    assert sig is None
    assert any("detector_parse_failed" in m for m in caplog.messages)


def test_bad_verdict_returns_none(caplog):
    caplog.set_level(logging.WARNING, logger="metacognition.dissatisfaction")
    llm = _stub_llm(json.dumps({"verdict": "confused", "score": 0.9}))
    sig = classify_conversation(
        conversation_id="c7",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is None
    assert any("detector_bad_verdict" in m for m in caplog.messages)


# ── coercion and clamping ───────────────────────────────────────


def test_score_clamped_to_unit_interval():
    # Model returns an absurd confidence; detector clamps rather than
    # rejecting — the verdict is still usable, just score-capped.
    llm = _stub_llm(json.dumps({"verdict": "unsatisfied", "topic": "x", "score": 42.0}))
    sig = classify_conversation(
        conversation_id="c8",
        turns=_turns_unsatisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is not None
    assert sig.score == 1.0

    llm2 = _stub_llm(
        json.dumps({"verdict": "unsatisfied", "topic": "x", "score": -0.3})
    )
    sig2 = classify_conversation(
        conversation_id="c8b",
        turns=_turns_unsatisfied(),
        ego_model="glm-4.7",
        llm_call=llm2,
    )
    assert sig2 is not None
    assert sig2.score == 0.0


def test_non_int_signal_turns_filtered():
    llm = _stub_llm(
        json.dumps(
            {
                "verdict": "unsatisfied",
                "topic": "x",
                "signal_turns": [1, "two", 3, None, 4.5],
                "score": 0.8,
            }
        )
    )
    sig = classify_conversation(
        conversation_id="c9",
        turns=_turns_unsatisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is not None
    assert sig.signal_turns == [1, 3, 4]  # 4.5 is float, coerced to int


def test_missing_optional_fields_have_safe_defaults():
    llm = _stub_llm(json.dumps({"verdict": "neutral"}))
    sig = classify_conversation(
        conversation_id="c10",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is not None
    assert sig.topic == ""
    assert sig.signal_turns == []
    assert sig.rationale == ""
    assert sig.score == 0.0


# ── env / detector override ─────────────────────────────────────


def test_env_override_detector_model(monkeypatch):
    monkeypatch.setenv("EREBUS_DETECTOR_MODEL", "glm-4.7")
    llm = _stub_llm(json.dumps({"verdict": "satisfied", "score": 1.0}))
    sig = classify_conversation(
        conversation_id="c11",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is not None
    assert sig.detector_model == "glm-4.7"
    assert llm.captured["model"] == "glm-4.7"  # type: ignore[attr-defined]


def test_explicit_detector_model_wins_over_env(monkeypatch):
    monkeypatch.setenv("EREBUS_DETECTOR_MODEL", "env-model")
    llm = _stub_llm(json.dumps({"verdict": "satisfied", "score": 1.0}))
    sig = classify_conversation(
        conversation_id="c12",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
        detector_model="explicit-model",
    )
    assert sig is not None
    assert sig.detector_model == "explicit-model"


def test_max_turns_caps_prompt_size():
    turns = [{"role": "user", "content": f"turn {i}"} for i in range(30)]
    llm = _stub_llm(json.dumps({"verdict": "neutral"}))
    classify_conversation(
        conversation_id="c13",
        turns=turns,
        ego_model="glm-4.7",
        llm_call=llm,
        max_turns=5,
    )
    user_msg = llm.captured["messages"][1]["content"]  # type: ignore[attr-defined]
    assert "turn 25" in user_msg
    assert "turn 29" in user_msg
    assert "turn 24" not in user_msg


def test_detector_version_is_recorded():
    llm = _stub_llm(json.dumps({"verdict": "satisfied", "score": 0.9}))
    sig = classify_conversation(
        conversation_id="c14",
        turns=_turns_satisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert sig is not None
    assert sig.detector_version == DETECTOR_VERSION
    # Version string has a shape we can sanity-check
    assert "gap-det" in sig.detector_version


# ── signal-exchange contract ────────────────────────────────────


def test_signal_is_dataclass_with_all_fields():
    llm = _stub_llm(
        json.dumps(
            {
                "verdict": "unsatisfied",
                "topic": "x",
                "signal_turns": [1],
                "rationale": "y",
                "score": 0.8,
            }
        )
    )
    sig = classify_conversation(
        conversation_id="c15",
        turns=_turns_unsatisfied(),
        ego_model="glm-4.7",
        llm_call=llm,
    )
    assert isinstance(sig, ConversationSignal)
    # aggregator depends on exactly these fields existing
    assert hasattr(sig, "verdict")
    assert hasattr(sig, "topic")
    assert hasattr(sig, "signal_turns")
    assert hasattr(sig, "rationale")
    assert hasattr(sig, "score")
    assert hasattr(sig, "detector_model")
    assert hasattr(sig, "detector_version")


def test_default_max_turns_matches_spec_constant():
    """Spec §2 sets max_turns=10 in the detector contract."""
    assert MAX_TURNS_IN_PROMPT == 10
