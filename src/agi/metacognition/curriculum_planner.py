# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Knowledge Gap Curriculum Planner — Metacognitive Self-Improvement.

Analyzes episodic memory to detect knowledge gaps and automatically
generates targeted training scenarios for the DM. Closes the loop:

    Poor response -> Gap detected -> DM trains on gap -> Dream ->
    Wiki updated -> Better future responses

Gap detection signals:
    1. Low quality scores in episodic memory
    2. Safety flags on responses (output gate caught something)
    3. Short responses (may indicate uncertainty/hedging)
    4. Topic clusters with declining quality over time
    5. Questions that triggered Ego arbitration (high disagreement)

Cognitive science grounding:
    - Metacognitive monitoring (Flavell, 1979): knowing what you don't know
    - Zone of Proximal Development (Vygotsky, 1978): train just beyond current ability
    - Desirable difficulties (Bjork, 1994): optimal challenge improves learning
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore[assignment]


@dataclass
class KnowledgeGap:
    """A detected knowledge gap.

    Attributes:
        domain: Topic area (e.g., "ethics", "quantum physics").
        signal: What triggered the detection.
        severity: 0.0-1.0 (higher = bigger gap).
        sample_questions: Example questions that exposed the gap.
        episode_ids: Episodes that contributed to detection.
        suggested_difficulty: Recommended training difficulty.
    """

    domain: str
    signal: str
    severity: float
    sample_questions: List[str] = field(default_factory=list)
    episode_ids: List[str] = field(default_factory=list)
    suggested_difficulty: int = 2


@dataclass
class CurriculumPlan:
    """A training plan generated from detected gaps.

    Attributes:
        gaps: Detected knowledge gaps, sorted by severity.
        total_episodes_analyzed: How many episodes were scanned.
        recommended_scenarios: Number of training scenarios.
        domains_to_focus: Priority domains for next training session.
    """

    gaps: List[KnowledgeGap] = field(default_factory=list)
    total_episodes_analyzed: int = 0
    recommended_scenarios: int = 0
    domains_to_focus: List[str] = field(default_factory=list)


class CurriculumPlanner:
    """Analyzes episodic memory to detect gaps and plan training.

    Reads from PostgreSQL episodes table (read-only analysis),
    then generates a CurriculumPlan that the DungeonMaster can
    use to create targeted training scenarios.
    """

    def __init__(
        self,
        db_dsn: str = "dbname=atlas user=claude",
        lookback_episodes: int = 100,
        quality_threshold: float = 0.6,
        min_gap_severity: float = 0.3,
    ) -> None:
        self._db_dsn = db_dsn
        self._lookback = lookback_episodes
        self._quality_threshold = quality_threshold
        self._min_severity = min_gap_severity

    def analyze(self) -> CurriculumPlan:
        """Analyze recent episodes and detect knowledge gaps.

        Returns:
            CurriculumPlan with detected gaps sorted by severity.
        """
        episodes = self._fetch_recent_episodes()
        if not episodes:
            logger.info("[curriculum] No episodes to analyze")
            return CurriculumPlan()

        gaps: List[KnowledgeGap] = []

        # Signal 1: Low quality scores
        gaps.extend(self._detect_low_quality(episodes))

        # Signal 2: Safety-flagged responses
        gaps.extend(self._detect_safety_issues(episodes))

        # Signal 3: Short/hedging responses
        gaps.extend(self._detect_uncertain_responses(episodes))

        # Signal 4: High-disagreement episodes (Ego arbitrated)
        gaps.extend(self._detect_high_disagreement(episodes))

        # Filter by minimum severity and deduplicate
        gaps = [g for g in gaps if g.severity >= self._min_severity]
        gaps = self._merge_gaps(gaps)
        gaps.sort(key=lambda g: g.severity, reverse=True)

        # Set difficulty based on severity
        for gap in gaps:
            if gap.severity > 0.8:
                gap.suggested_difficulty = 1  # Start easy to build foundation
            elif gap.severity > 0.5:
                gap.suggested_difficulty = 2
            else:
                gap.suggested_difficulty = 3  # Minor gap, challenge more

        plan = CurriculumPlan(
            gaps=gaps,
            total_episodes_analyzed=len(episodes),
            recommended_scenarios=min(len(gaps) * 3, 20),
            domains_to_focus=[g.domain for g in gaps[:5]],
        )

        logger.info(
            "[curriculum] Analyzed %d episodes, found %d gaps. " "Focus domains: %s",
            len(episodes),
            len(gaps),
            plan.domains_to_focus,
        )

        return plan

    def _fetch_recent_episodes(self) -> List[Dict[str, Any]]:
        """Fetch recent episodes from PostgreSQL."""
        if psycopg2 is None:
            return []

        try:
            conn = psycopg2.connect(self._db_dsn)
            episodes = []
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, user_message, atlas_response, "
                    "hemisphere, safety_flags, quality_score, "
                    "metadata "
                    "FROM episodes "
                    "ORDER BY timestamp DESC "
                    "LIMIT %s",
                    (self._lookback,),
                )
                for row in cur.fetchall():
                    episodes.append(
                        {
                            "id": str(row[0]),
                            "user_message": row[1] or "",
                            "atlas_response": row[2] or "",
                            "hemisphere": row[3] or "lh",
                            "safety_flags": row[4] or {},
                            "quality_score": float(row[5] or 0),
                            "metadata": row[6] or {},
                        }
                    )
            conn.close()
            return episodes
        except Exception as e:
            logger.warning("[curriculum] DB fetch failed: %s", e)
            return []

    def _detect_low_quality(self, episodes: List[Dict[str, Any]]) -> List[KnowledgeGap]:
        """Detect topics where quality scores are consistently low."""
        # Group by rough topic (first significant word of question)
        topic_scores: Dict[str, List[Dict]] = defaultdict(list)
        for ep in episodes:
            topic = self._extract_topic(ep["user_message"])
            topic_scores[topic].append(ep)

        gaps = []
        for topic, eps in topic_scores.items():
            scores = [e["quality_score"] for e in eps if e["quality_score"] > 0]
            if not scores:
                continue

            mean_score = sum(scores) / len(scores)
            if mean_score < self._quality_threshold:
                severity = 1.0 - mean_score
                gaps.append(
                    KnowledgeGap(
                        domain=topic,
                        signal="low_quality",
                        severity=severity,
                        sample_questions=[e["user_message"][:100] for e in eps[:3]],
                        episode_ids=[e["id"] for e in eps[:5]],
                    )
                )

        return gaps

    def _detect_safety_issues(
        self, episodes: List[Dict[str, Any]]
    ) -> List[KnowledgeGap]:
        """Detect topics that trigger safety flags on output."""
        gaps = []
        flagged = []

        for ep in episodes:
            safety = ep.get("safety_flags", {})
            output_flags = (
                safety.get("output", {}).get("flags", [])
                if isinstance(safety, dict)
                else []
            )
            if output_flags:
                flagged.append(ep)

        if flagged:
            gaps.append(
                KnowledgeGap(
                    domain="safety_sensitive_topics",
                    signal="output_safety_flags",
                    severity=min(1.0, len(flagged) / 10.0),
                    sample_questions=[e["user_message"][:100] for e in flagged[:3]],
                    episode_ids=[e["id"] for e in flagged[:5]],
                )
            )

        return gaps

    def _detect_uncertain_responses(
        self, episodes: List[Dict[str, Any]]
    ) -> List[KnowledgeGap]:
        """Detect topics where responses are unusually short (hedging)."""
        gaps = []
        topic_lengths: Dict[str, List[Dict]] = defaultdict(list)

        for ep in episodes:
            response_len = len(ep["atlas_response"])
            topic = self._extract_topic(ep["user_message"])
            topic_lengths[topic].append(
                {
                    **ep,
                    "response_len": response_len,
                }
            )

        # Find topics with mean response length < 200 chars
        # (suggests hedging or uncertainty)
        for topic, eps in topic_lengths.items():
            lengths = [e["response_len"] for e in eps]
            mean_len = sum(lengths) / len(lengths)

            if mean_len < 200 and len(eps) >= 2:
                gaps.append(
                    KnowledgeGap(
                        domain=topic,
                        signal="short_responses",
                        severity=min(0.8, max(0.2, 1.0 - mean_len / 500)),
                        sample_questions=[e["user_message"][:100] for e in eps[:3]],
                        episode_ids=[e["id"] for e in eps[:5]],
                    )
                )

        return gaps

    def _detect_high_disagreement(
        self, episodes: List[Dict[str, Any]]
    ) -> List[KnowledgeGap]:
        """Detect topics where Superego and Id frequently disagree."""
        gaps = []
        debate_eps = [e for e in episodes if e["hemisphere"] == "both"]

        if len(debate_eps) >= 3:
            topic_debates: Dict[str, int] = defaultdict(int)

            for ep in debate_eps:
                topic = self._extract_topic(ep["user_message"])
                topic_debates[topic] += 1

            for topic, count in topic_debates.items():
                if count >= 2:
                    gaps.append(
                        KnowledgeGap(
                            domain=topic,
                            signal="high_disagreement",
                            severity=min(0.7, count / 10.0),
                            sample_questions=[
                                e["user_message"][:100]
                                for e in debate_eps
                                if self._extract_topic(e["user_message"]) == topic
                            ][:3],
                            episode_ids=[
                                e["id"]
                                for e in debate_eps
                                if self._extract_topic(e["user_message"]) == topic
                            ][:5],
                        )
                    )

        return gaps

    def _extract_topic(self, question: str) -> str:
        """Extract a rough topic label from a question.

        Uses the first significant word (>4 chars) as a
        simple topic bucketing strategy.
        """
        words = [
            w.lower().strip("?.,!:;")
            for w in question.split()
            if len(w) > 4
            and w.lower()
            not in {
                "what",
                "which",
                "where",
                "would",
                "could",
                "should",
                "about",
                "there",
                "their",
                "these",
                "those",
                "please",
                "explain",
            }
        ]
        return words[0] if words else "general"

    def _merge_gaps(self, gaps: List[KnowledgeGap]) -> List[KnowledgeGap]:
        """Merge gaps with the same domain, keeping highest severity."""
        by_domain: Dict[str, KnowledgeGap] = {}
        for gap in gaps:
            key = gap.domain
            if key in by_domain:
                existing = by_domain[key]
                if gap.severity > existing.severity:
                    existing.severity = gap.severity
                existing.signal += f"+{gap.signal}"
                existing.sample_questions.extend(gap.sample_questions)
                existing.episode_ids.extend(gap.episode_ids)
            else:
                by_domain[key] = gap

        # Deduplicate episode_ids and sample_questions
        for gap in by_domain.values():
            gap.episode_ids = list(dict.fromkeys(gap.episode_ids))[:10]
            gap.sample_questions = list(dict.fromkeys(gap.sample_questions))[:5]

        return list(by_domain.values())

    def format_for_dm(self, plan: CurriculumPlan) -> str:
        """Format a curriculum plan as context for the DM.

        Returns a string that can be injected into the DM's
        scenario generation prompt.
        """
        if not plan.gaps:
            return "No knowledge gaps detected. Training broadly."

        lines = [
            "Knowledge gaps detected (train on these):",
        ]
        for gap in plan.gaps[:5]:
            lines.append(
                f"  - {gap.domain} (severity={gap.severity:.2f}, "
                f"signal={gap.signal})"
            )
            if gap.sample_questions:
                lines.append(f"    Example: {gap.sample_questions[0]}")
        lines.append(
            f"Recommended: {plan.recommended_scenarios} scenarios "
            f"at difficulty {plan.gaps[0].suggested_difficulty}"
        )
        return "\n".join(lines)
