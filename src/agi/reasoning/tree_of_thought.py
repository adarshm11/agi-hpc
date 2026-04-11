# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Tree-of-Thought Debate — Multi-path reasoning for the Freudian psyche.

Instead of each hemisphere producing one answer, each generates
multiple reasoning branches (different approaches to the question).
The Ego evaluates which branches are strongest, then the best
branches are debated and synthesized.

Standard debate flow:
    Superego -> 1 answer -> Id -> 1 answer -> challenge -> synthesize

Tree-of-Thought flow:
    Superego -> 3 branches -> Id -> 3 branches ->
    Ego evaluates all 6 -> top 2 debated -> synthesize

Cognitive science grounding:
    - Tree of Thought (Yao et al., 2023): deliberate search over
      reasoning paths with self-evaluation
    - Dual Process Theory (Kahneman, 2011): System 2 explores
      multiple paths before committing
    - Divergent thinking (Guilford, 1967): generate multiple
      solutions before converging

Usage:
    from agi.reasoning.tree_of_thought import TreeOfThought
    tot = TreeOfThought(superego_url, id_url, ego_url)
    result = tot.reason(query, context, n_branches=3)
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


@dataclass
class ReasoningBranch:
    """A single reasoning branch from one hemisphere.

    Attributes:
        hemisphere: Which model produced this ("superego" or "id").
        approach: Brief label for the reasoning strategy.
        content: The full reasoning text.
        self_score: Self-assessed quality (1-10).
        ego_score: Ego's evaluation (1-10, set later).
    """

    hemisphere: str
    approach: str
    content: str
    self_score: float = 5.0
    ego_score: float = 0.0


@dataclass
class TreeResult:
    """Result of a Tree-of-Thought debate.

    Attributes:
        query: The original user question.
        branches: All generated reasoning branches.
        selected_branches: The top branches chosen by the Ego.
        synthesis: Final synthesized answer.
        debate_log: Full debate transcript for UI display.
        total_branches: How many branches were generated.
        latency_s: Total wall clock time.
        method: "tree_of_thought" for identification.
    """

    query: str
    branches: List[ReasoningBranch] = field(default_factory=list)
    selected_branches: List[ReasoningBranch] = field(default_factory=list)
    synthesis: str = ""
    debate_log: str = ""
    total_branches: int = 0
    latency_s: float = 0.0
    method: str = "tree_of_thought"


BRANCH_PROMPT = """\
You are {role}. The user asks: {query}

Generate {n} DIFFERENT approaches to answering this question.
Each approach should use a distinct reasoning strategy:
- Approach 1: {strategy1}
- Approach 2: {strategy2}
- Approach 3: {strategy3}

For each approach, write a concise answer (2-4 sentences).
Rate your own confidence in each approach (1-10).

Respond in this exact format:
APPROACH 1: [strategy name]
ANSWER: [your answer]
CONFIDENCE: [1-10]

APPROACH 2: [strategy name]
ANSWER: [your answer]
CONFIDENCE: [1-10]

APPROACH 3: [strategy name]
ANSWER: [your answer]
CONFIDENCE: [1-10]"""

SUPEREGO_STRATEGIES = (
    "Apply formal logical analysis or established frameworks",
    "Consider rules, duties, and precedent",
    "Use evidence-based reasoning with citations",
)

ID_STRATEGIES = (
    "Think from the gut — what feels right?",
    "Use creative analogy from a different domain",
    "Consider the human impact and emotional dimension",
)

EVALUATE_PROMPT = """\
You are the Ego — the impartial evaluator. Rate each reasoning \
branch for quality.

Question: {query}

{branches_text}

For each branch, score 1-10 on:
- Accuracy: Is the reasoning correct?
- Depth: Does it go beyond the obvious?
- Usefulness: Would this help the user?

Respond with ONLY the scores in this format:
B1: [score]
B2: [score]
B3: [score]
B4: [score]
B5: [score]
B6: [score]"""


class TreeOfThought:
    """Multi-path reasoning engine for the Freudian psyche.

    Generates multiple reasoning branches from each hemisphere,
    evaluates them via the Ego (or Divine Council), and synthesizes
    the best into a final answer.

    When a DivineCouncil instance is provided, branch evaluation
    uses 4-agent deliberation (Judge/Advocate/Synthesizer/Ethicist)
    instead of a single Ego call. This produces richer scoring,
    adversarial challenge of weak branches, and ethical review.
    """

    def __init__(
        self,
        superego_url: str = "http://localhost:8080",
        id_url: str = "http://localhost:8082",
        ego_url: str = "http://localhost:8084",
        timeout: int = 300,
        council: object = None,
    ) -> None:
        self._superego_url = superego_url
        self._id_url = id_url
        self._ego_url = ego_url
        self._timeout = timeout
        self._council = council  # Optional DivineCouncil instance

    def _call_llm(
        self,
        url: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        max_tokens: int = 512,
    ) -> str:
        """Call an LLM endpoint."""
        try:
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                timeout=self._timeout,
            )
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content if content else "(no response)"
        except Exception as e:
            return f"(error: {e})"

    def reason(
        self,
        query: str,
        context: str = "",
        n_branches: int = 3,
    ) -> TreeResult:
        """Run Tree-of-Thought reasoning.

        Args:
            query: The user's question.
            context: Optional RAG context.
            n_branches: Branches per hemisphere (default 3).

        Returns:
            TreeResult with all branches, evaluations,
            and final synthesis.
        """
        t0 = time.monotonic()

        # Step 1: Generate branches in parallel (both hemispheres)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            superego_future = ex.submit(
                self._generate_branches,
                "superego",
                query,
                context,
                n_branches,
            )
            id_future = ex.submit(
                self._generate_branches,
                "id",
                query,
                context,
                n_branches,
            )
            superego_branches = superego_future.result()
            id_branches = id_future.result()

        all_branches = superego_branches + id_branches

        logger.info(
            "[tot] Generated %d branches (%d superego, %d id)",
            len(all_branches),
            len(superego_branches),
            len(id_branches),
        )

        # Step 2: Evaluate all branches
        council_verdict = None
        if self._council is not None:
            # Divine Council evaluation: 4-agent deliberation
            council_verdict = self._council_evaluate(query, all_branches)
        else:
            # Simple Ego evaluation (single LLM call)
            self._evaluate_branches(query, all_branches)

        # Step 3: Select top branches (best from each hemisphere)
        selected = self._select_top(all_branches, top_n=2)

        logger.info(
            "[tot] Selected top %d branches: %s",
            len(selected),
            [f"{b.hemisphere}:{b.approach}={b.ego_score:.0f}" for b in selected],
        )

        # Step 4: Synthesize from best branches
        # Always use the full-size Ego for synthesis — the council
        # E4B Synthesizer is too small to produce quality answers.
        # Council is used for evaluation/scoring only.
        synthesis = self._synthesize(query, selected)

        # Build debate log for UI
        debate_log = self._format_debate_log(all_branches, selected, council_verdict)

        elapsed = time.monotonic() - t0

        return TreeResult(
            query=query,
            branches=all_branches,
            selected_branches=selected,
            synthesis=synthesis,
            debate_log=debate_log,
            total_branches=len(all_branches),
            latency_s=elapsed,
        )

    def _generate_branches(
        self,
        hemisphere: str,
        query: str,
        context: str,
        n: int,
    ) -> List[ReasoningBranch]:
        """Generate n reasoning branches from one hemisphere."""
        if hemisphere == "superego":
            url = self._superego_url
            role = (
                "the Superego — analytical, precise, moral. "
                "You apply rigorous frameworks and evidence."
            )
            strategies = SUPEREGO_STRATEGIES
            temp = 0.4
        else:
            url = self._id_url
            role = (
                "the Id — creative, instinctual, empathetic. "
                "You think with intuition and emotion."
            )
            strategies = ID_STRATEGIES
            temp = 0.7

        prompt = BRANCH_PROMPT.format(
            role=role,
            query=query,
            n=n,
            strategy1=strategies[0],
            strategy2=strategies[1],
            strategy3=strategies[2],
        )

        if context:
            prompt = f"Context:\n{context[:500]}\n\n{prompt}"

        raw = self._call_llm(
            url,
            [{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=800,
        )

        return self._parse_branches(raw, hemisphere)

    def _parse_branches(self, raw: str, hemisphere: str) -> List[ReasoningBranch]:
        """Parse LLM output into ReasoningBranch objects."""
        branches: List[ReasoningBranch] = []

        # Split by APPROACH markers
        parts = re.split(r"APPROACH\s+\d+\s*:", raw, flags=re.IGNORECASE)

        for part in parts[1:]:  # Skip text before first APPROACH
            approach = ""
            answer = ""
            confidence = 5.0

            lines = part.strip().split("\n")
            if lines:
                approach = lines[0].strip().strip("[]")

            # Find ANSWER: section
            answer_match = re.search(
                r"ANSWER:\s*(.+?)(?=CONFIDENCE:|$)",
                part,
                re.IGNORECASE | re.DOTALL,
            )
            if answer_match:
                answer = answer_match.group(1).strip()

            # Find CONFIDENCE: score
            conf_match = re.search(r"CONFIDENCE:\s*(\d+)", part, re.IGNORECASE)
            if conf_match:
                confidence = min(10.0, max(1.0, float(conf_match.group(1))))

            if answer:
                branches.append(
                    ReasoningBranch(
                        hemisphere=hemisphere,
                        approach=approach[:50] or f"{hemisphere}_branch",
                        content=answer[:500],
                        self_score=confidence,
                    )
                )

        # Fallback: if parsing failed, treat entire text as one branch
        if not branches and raw and "(error" not in raw:
            branches.append(
                ReasoningBranch(
                    hemisphere=hemisphere,
                    approach="unparsed",
                    content=raw[:500],
                    self_score=5.0,
                )
            )

        return branches

    def _council_evaluate(
        self,
        query: str,
        branches: List[ReasoningBranch],
    ):
        """Have the Divine Council evaluate branches.

        The council receives a formatted brief with all branches
        and deliberates in parallel. The Judge's scores are mapped
        back to branch ego_scores. The Advocate's challenges flag
        weak reasoning. The Ethicist checks for safety concerns.
        The Synthesizer produces the final answer.

        Returns:
            CouncilVerdict from the council's deliberation.
        """
        # Format branches as the "response" for the council
        branches_text = ""
        for i, b in enumerate(branches):
            branches_text += (
                f"Branch {i+1} ({b.hemisphere}, {b.approach}):\n"
                f"{b.content[:300]}\n\n"
            )

        verdict = self._council.deliberate(
            query=query,
            superego_response=(
                "Superego branches:\n"
                + "\n".join(
                    f"- {b.approach}: {b.content[:200]}"
                    for b in branches
                    if b.hemisphere == "superego"
                )
            ),
            id_response=(
                "Id branches:\n"
                + "\n".join(
                    f"- {b.approach}: {b.content[:200]}"
                    for b in branches
                    if b.hemisphere == "id"
                )
            ),
            context=branches_text,
        )

        # Map Judge's score to all branches (uniform for now)
        judge_vote = verdict.votes.get("judge")
        if judge_vote and judge_vote.score > 0:
            # Parse per-branch scores from Judge if present
            import re as _re

            for i, b in enumerate(branches):
                match = _re.search(
                    rf"[Bb](?:ranch)?\s*{i+1}\s*[:=]\s*(\d+)",
                    judge_vote.response,
                )
                if match:
                    b.ego_score = min(10.0, float(match.group(1)))
                else:
                    b.ego_score = judge_vote.score

        # Advocate challenges lower scores for weak branches
        advocate_vote = verdict.votes.get("advocate")
        if advocate_vote:
            lower = advocate_vote.response.lower()
            for b in branches:
                # If advocate specifically calls out this approach
                if b.approach.lower() in lower:
                    b.ego_score = max(1.0, b.ego_score - 2.0)

        # Ethicist flags reduce scores
        ethicist_vote = verdict.votes.get("ethicist")
        if ethicist_vote and ethicist_vote.flags:
            for b in branches:
                b.ego_score = max(1.0, b.ego_score - 1.0)

        logger.info(
            "[tot] Council evaluated %d branches: "
            "consensus=%s, %d approve, %d challenge",
            len(branches),
            verdict.consensus,
            verdict.approval_count,
            verdict.challenge_count,
        )

        return verdict

    def _evaluate_branches(
        self,
        query: str,
        branches: List[ReasoningBranch],
    ) -> None:
        """Have the Ego evaluate all branches."""
        branches_text = ""
        for i, b in enumerate(branches):
            branches_text += (
                f"B{i+1} ({b.hemisphere}, {b.approach}):\n" f"{b.content[:200]}\n\n"
            )

        prompt = EVALUATE_PROMPT.format(query=query, branches_text=branches_text)

        raw = self._call_llm(
            self._ego_url,
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100,
        )

        # Parse scores
        scores = re.findall(r"B(\d+)\s*:\s*(\d+)", raw, re.IGNORECASE)
        for idx_str, score_str in scores:
            idx = int(idx_str) - 1
            if 0 <= idx < len(branches):
                branches[idx].ego_score = min(10.0, max(1.0, float(score_str)))

    def _select_top(
        self,
        branches: List[ReasoningBranch],
        top_n: int = 2,
    ) -> List[ReasoningBranch]:
        """Select top branches by Ego score.

        Ensures at least one from each hemisphere if possible.
        """
        if not branches:
            return []

        sorted_branches = sorted(branches, key=lambda b: b.ego_score, reverse=True)

        selected: List[ReasoningBranch] = []
        hemispheres_seen: set = set()

        # First pass: pick best from each hemisphere
        for b in sorted_branches:
            if b.hemisphere not in hemispheres_seen:
                selected.append(b)
                hemispheres_seen.add(b.hemisphere)
                if len(selected) >= top_n:
                    break

        # Second pass: fill remaining slots with best overall
        if len(selected) < top_n:
            for b in sorted_branches:
                if b not in selected:
                    selected.append(b)
                    if len(selected) >= top_n:
                        break

        return selected

    def _synthesize(
        self,
        query: str,
        selected: List[ReasoningBranch],
    ) -> str:
        """Synthesize the best branches into a final answer."""
        if not selected:
            return "(no branches to synthesize)"

        branches_text = "\n\n".join(
            f"[{b.hemisphere} — {b.approach}, score={b.ego_score:.0f}]:\n"
            f"{b.content}"
            for b in selected
        )

        prompt = (
            f"The user asked: {query}\n\n"
            f"The strongest reasoning paths:\n\n"
            f"{branches_text}\n\n"
            "Synthesize these into a clear, direct answer. "
            "Integrate the strongest points naturally. "
            "Don't reference the branches or scores — "
            "just answer the question as if you thought "
            "of it yourself."
        )

        # Use the Ego for synthesis (balanced mediator)
        return self._call_llm(
            self._ego_url,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )

    def _format_debate_log(
        self,
        all_branches: List[ReasoningBranch],
        selected: List[ReasoningBranch],
        council_verdict=None,
    ) -> str:
        """Format the debate for the UI collapsible."""
        lines: List[str] = []

        lines.append("### Reasoning Branches\n")
        for b in all_branches:
            star = " ★" if b in selected else ""
            lines.append(
                f"**{b.hemisphere.title()} — {b.approach}** "
                f"(self={b.self_score:.0f}, "
                f"ego={b.ego_score:.0f}){star}\n"
                f"{b.content}\n"
            )

        lines.append("\n### Selected for Synthesis\n")
        for b in selected:
            lines.append(
                f"- {b.hemisphere.title()}: {b.approach} " f"(score={b.ego_score:.0f})"
            )

        # Include council deliberation details if available
        if council_verdict is not None:
            lines.append("\n### Divine Council Deliberation\n")
            lines.append(council_verdict.format_log())

        return "\n".join(lines)
