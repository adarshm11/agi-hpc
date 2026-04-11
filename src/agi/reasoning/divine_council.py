# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Divine Council — Multi-agent Ego deliberation.

Replaces the single Ego mediator with a council of specialized
sub-agents that deliberate in parallel and reach consensus.  All
council members share a **single llama-server** process running
Gemma 4 26B-A4B MoE (26B total params, 4B active per token) with
``--parallel 8``.  The model loads once (~15 GB RAM); each parallel
slot adds ~300 MB of KV cache — total footprint ~18 GB.

Council Members (7):
    Judge      — Impartial evaluator. Scores correctness, logic.
    Advocate   — Devil's advocate. Challenges consensus, finds flaws.
    Synthesizer— Integration expert. Merges perspectives, resolves tension.
    Ethicist   — Moral compass. Checks alignment, fairness, safety.
    Historian  — Precedent tracker. References prior decisions, patterns.
    Futurist   — Consequence mapper. Second-order effects, long-term impact.
    Pragmatist — Feasibility assessor. Resources, constraints, viability.

Cognitive science grounding:
    - Minsky (1986): Society of Mind — intelligence as many agents
    - Mercier & Sperber (2011): Argumentative Theory of Reasoning —
      reasoning evolved for social deliberation, not solo thinking
    - Surowiecki (2004): Wisdom of Crowds — diverse independent
      judgments aggregate better than individual expert judgment
    - Schank (1982): Dynamic Memory — case-based reasoning from prior
      experience (Historian)
    - Gilbert & Wilson (2007): Prospection — mental simulation of
      future states to guide present decisions (Futurist)
    - Simon (1955): Bounded Rationality — satisficing under real-world
      resource constraints (Pragmatist)

The council improves on a single Ego because:
    1. Diversity: Seven lenses catch more problems than one
    2. Adversarial: The Advocate prevents groupthink
    3. Temporal: Historian + Futurist provide past-present-future coverage
    4. Grounded: The Pragmatist vetoes beautiful theories that can't ship
    5. Parallel: All members deliberate simultaneously via --parallel 8
    6. Efficient: One process, one model load, ~75% less RAM than 4 servers

Usage:
    council = DivineCouncil()
    verdict = council.deliberate(query, superego_response, id_response)
    # verdict.consensus = True
    # verdict.synthesis = "The balanced answer is..."
    # verdict.votes = {"judge": "approve", "advocate": "challenge", ...}
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Council member definitions
# ---------------------------------------------------------------------------

MODEL_NAME = "Gemma 4 26B-A4B"

COUNCIL_MEMBERS = {
    "judge": {
        "name": "Judge",
        "system_prompt": (
            "You are the Judge on the Divine Council.\n\n"
            "MISSION: Impartial evaluation of reasoning quality. "
            "You are the calibration standard for the council.\n\n"
            "RULES:\n"
            "1. Assess accuracy, logical coherence, and completeness.\n"
            "2. Score each response on a scale of 1-10.\n"
            "3. Provide a brief justification for each score.\n"
            "4. Note any factual errors or logical fallacies.\n"
            "5. Do not advocate — only evaluate.\n\n"
            "SUCCESS METRICS: Your scores correlate with ground "
            "truth. Higher-scored responses are objectively better."
        ),
        "color": "#4a9eff",
    },
    "advocate": {
        "name": "Advocate",
        "system_prompt": (
            "You are the Devil's Advocate on the Divine Council.\n\n"
            "MISSION: Prevent groupthink through rigorous challenge. "
            "You are the immune system of the council.\n\n"
            "RULES:\n"
            "1. Find flaws, unstated assumptions, and edge cases.\n"
            "2. Challenge even when consensus seems strong.\n"
            "3. Propose specific counterarguments, not vague doubt.\n"
            "4. If everyone agrees, you must disagree and say why.\n"
            "5. Be rigorous but constructive — strengthen, "
            "don't obstruct.\n\n"
            "SUCCESS METRICS: Your challenges expose real "
            "weaknesses. The final answer is stronger because "
            "of your objections."
        ),
        "color": "#f87171",
    },
    "synthesizer": {
        "name": "Synthesizer",
        "system_prompt": (
            "You are the Synthesizer on the Divine Council.\n\n"
            "MISSION: Integration and resolution. Produce an answer "
            "that is better than any individual perspective.\n\n"
            "RULES:\n"
            "1. Take the strongest elements from each perspective.\n"
            "2. Resolve tensions rather than picking sides.\n"
            "3. Address the Advocate's challenges directly.\n"
            "4. Your synthesis must answer the original question.\n"
            "5. Be concise and authoritative — no hedging.\n\n"
            "SUCCESS METRICS: Your synthesis is preferred over "
            "any individual response. Users find it clear, "
            "complete, and balanced."
        ),
        "color": "#4ade80",
    },
    "ethicist": {
        "name": "Ethicist",
        "system_prompt": (
            "You are the Ethicist on the Divine Council.\n\n"
            "MISSION: Moral evaluation grounded in specific "
            "frameworks. You are the conscience of the council.\n\n"
            "RULES:\n"
            "1. Check for fairness, potential harm, and bias.\n"
            "2. Reference specific moral frameworks when flagging.\n"
            "3. If ethically sound, confirm concisely.\n"
            "4. Flag concerns with severity: minor, moderate, "
            "serious.\n"
            "5. Consider impact on vulnerable populations.\n\n"
            "SUCCESS METRICS: No harmful content passes your "
            "review. Your flags cite specific principles, not "
            "vague unease."
        ),
        "color": "#f59e0b",
    },
    "historian": {
        "name": "Historian",
        "system_prompt": (
            "You are the Historian on the Divine Council.\n\n"
            "MISSION: Case-based reasoning from prior experience. "
            "You are the institutional memory of the council.\n\n"
            "RULES:\n"
            "1. Identify precedents — similar past decisions and "
            "their outcomes.\n"
            "2. Note patterns: what worked before and what failed.\n"
            "3. Flag if the current proposal repeats a known mistake.\n"
            "4. Cite the specific prior case when making a claim.\n"
            "5. If no precedent exists, say so — novelty is useful "
            "information.\n\n"
            "SUCCESS METRICS: Your precedent analysis prevents "
            "repeated mistakes and surfaces proven approaches. "
            "The council benefits from accumulated experience."
        ),
        "color": "#a78bfa",
    },
    "futurist": {
        "name": "Futurist",
        "system_prompt": (
            "You are the Futurist on the Divine Council.\n\n"
            "MISSION: Prospective reasoning — second-order effects "
            "and long-term consequences. You are the early warning "
            "system of the council.\n\n"
            "RULES:\n"
            "1. Trace consequences forward: if we do X, then Y, "
            "then Z.\n"
            "2. Identify unintended side effects and cascading "
            "impacts.\n"
            "3. Consider how the decision ages — is it still good "
            "in a week? A month?\n"
            "4. Flag irreversible commitments that deserve extra "
            "scrutiny.\n"
            "5. Distinguish likely consequences from speculative "
            "ones.\n\n"
            "SUCCESS METRICS: Your foresight catches downstream "
            "problems before they materialize. Decisions made with "
            "your input have fewer surprises."
        ),
        "color": "#06b6d4",
    },
    "pragmatist": {
        "name": "Pragmatist",
        "system_prompt": (
            "You are the Pragmatist on the Divine Council.\n\n"
            "MISSION: Feasibility assessment under real-world "
            "constraints. You are the reality check of the council.\n\n"
            "RULES:\n"
            "1. Evaluate resource requirements: time, compute, "
            "data, human effort.\n"
            "2. Identify the simplest path that achieves the goal.\n"
            "3. Flag over-engineered proposals — good enough now "
            "beats perfect later.\n"
            "4. Consider operational burden: who maintains this?\n"
            "5. If infeasible, propose a viable alternative.\n\n"
            "SUCCESS METRICS: Your assessments are calibrated — "
            "feasible plans succeed, flagged plans would have "
            "failed. The council ships more because of your "
            "grounding."
        ),
        "color": "#ec4899",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CouncilVote:
    """One council member's deliberation output."""

    member: str
    response: str
    score: float  # 1-10 for judge, 0-1 for others
    flags: List[str] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class CouncilVerdict:
    """The council's collective verdict."""

    consensus: bool
    synthesis: str
    votes: Dict[str, CouncilVote] = field(default_factory=dict)
    approval_count: int = 0
    challenge_count: int = 0
    ethical_flags: List[str] = field(default_factory=list)
    total_latency_s: float = 0.0
    method: str = "divine_council"

    def format_log(self) -> str:
        """Format for the UI collapsible."""
        lines = ["### Divine Council Deliberation\n"]
        for member_id, vote in self.votes.items():
            info = COUNCIL_MEMBERS.get(member_id, {})
            name = info.get("name", member_id)
            lines.append(
                f"**{name}** " f"({vote.latency_s:.1f}s):\n" f"{vote.response[:300]}\n"
            )
        lines.append(
            f"\n**Consensus:** "
            f"{'Yes' if self.consensus else 'No'} "
            f"({self.approval_count} approve, "
            f"{self.challenge_count} challenge)"
        )
        if self.ethical_flags:
            lines.append(f"**Ethical flags:** {', '.join(self.ethical_flags)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Divine Council
# ---------------------------------------------------------------------------


class DivineCouncil:
    """Multi-agent Ego deliberation engine.

    Runs all council members in parallel and aggregates their
    judgments into a consensus verdict with synthesized response.
    """

    # Default base URL for the single shared llama-server.
    DEFAULT_URL = "http://localhost:8084"

    def __init__(
        self,
        member_urls: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        if member_urls is not None:
            self._urls = member_urls
        else:
            url = base_url or self.DEFAULT_URL
            self._urls = {mid: url for mid in COUNCIL_MEMBERS}
        self._timeout = timeout

    def _call_member(
        self,
        member_id: str,
        prompt: str,
    ) -> CouncilVote:
        """Call one council member."""
        url = self._urls.get(member_id, "http://localhost:8084")
        info = COUNCIL_MEMBERS.get(member_id, {})
        system = info.get("system_prompt", "You are a council member.")

        t0 = time.monotonic()
        try:
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 512,
                    "stream": False,
                },
                timeout=self._timeout,
            )
            data = resp.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "") or msg.get("reasoning_content", "")

            # Extract score if present
            score = 5.0
            score_match = re.search(r"(\d+)/10", content)
            if score_match:
                score = min(10.0, float(score_match.group(1)))

            # Extract flags
            flags = []
            if member_id == "ethicist":
                lc = content.lower()
                for fw in ["bias", "harm", "unfair", "unsafe"]:
                    if fw in lc:
                        flags.append(fw)
                if "concern" in lc and "no concern" not in lc:
                    flags.append("concern")

            elapsed = time.monotonic() - t0
            return CouncilVote(
                member=member_id,
                response=content[:500],
                score=score,
                flags=flags,
                latency_s=elapsed,
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning("[council] %s failed: %s", member_id, e)
            return CouncilVote(
                member=member_id,
                response=f"(error: {e})",
                score=5.0,
                latency_s=elapsed,
            )

    def deliberate(
        self,
        query: str,
        superego_response: str = "",
        id_response: str = "",
        context: str = "",
    ) -> CouncilVerdict:
        """Run full council deliberation.

        All members receive the same brief and deliberate in
        parallel. The Synthesizer produces the final answer,
        informed by Judge scores and Advocate challenges.

        Args:
            query: The user's original question.
            superego_response: Superego's analytical response.
            id_response: Id's creative response.
            context: Optional additional context.

        Returns:
            CouncilVerdict with synthesis and vote details.
        """
        t0 = time.monotonic()

        # Build the brief for all council members
        brief = f"The user asked: {query}\n\n"
        if superego_response:
            brief += (
                f"Superego (analytical) responded:\n" f"{superego_response[:400]}\n\n"
            )
        if id_response:
            brief += f"Id (creative) responded:\n" f"{id_response[:400]}\n\n"
        brief += (
            "Based on these responses, provide your assessment "
            "according to your role on the council."
        )

        # All members deliberate in parallel (single llama-server handles
        # concurrency via --parallel 8; ThreadPoolExecutor just fires requests)
        votes: Dict[str, CouncilVote] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(COUNCIL_MEMBERS),
        ) as ex:
            futures = {
                ex.submit(self._call_member, mid, brief): mid for mid in COUNCIL_MEMBERS
            }
            for future in concurrent.futures.as_completed(futures):
                mid = futures[future]
                votes[mid] = future.result()

        # Count approvals vs challenges
        # Advocate always challenges. Others approve unless
        # they give an explicitly low score (<4) or have flags.
        approval = 0
        challenge = 0
        for mid, vote in votes.items():
            if mid == "advocate":
                challenge += 1
            elif vote.flags:
                challenge += 1
            elif vote.score < 4:
                challenge += 1
            else:
                approval += 1

        # Ethical flags from ethicist
        ethical_flags = votes.get(
            "ethicist", CouncilVote(member="ethicist", response="", score=5)
        ).flags

        # Use Synthesizer's response as the final answer
        synthesis = votes.get(
            "synthesizer", CouncilVote(member="synthesizer", response="", score=5)
        ).response

        # If Synthesizer failed, fall back to highest-scored response
        if not synthesis or "(error" in synthesis:
            best = max(
                votes.values(),
                key=lambda v: v.score,
            )
            synthesis = best.response

        elapsed = time.monotonic() - t0
        # Majority of non-advocate members must approve, with no ethical flags.
        # With 7 members (advocate always challenges), max approval = 6.
        consensus = approval >= 4 and len(ethical_flags) == 0

        verdict = CouncilVerdict(
            consensus=consensus,
            synthesis=synthesis,
            votes=votes,
            approval_count=approval,
            challenge_count=challenge,
            ethical_flags=ethical_flags,
            total_latency_s=elapsed,
        )

        logger.info(
            "[council] Deliberation: %d approve, %d challenge, " "consensus=%s, %.1fs",
            approval,
            challenge,
            consensus,
            elapsed,
        )

        return verdict
