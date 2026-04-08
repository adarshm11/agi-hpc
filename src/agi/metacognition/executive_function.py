# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Executive Function — Unified cognitive control for Atlas AI.

The prefrontal cortex of the architecture. Manages goal-directed
behavior, selects reasoning strategies, tracks context across turns,
decomposes complex tasks, and inhibits premature responses.

Five core functions (Miyake et al., 2000; Diamond, 2013):

1. MODE SELECTOR (Shifting)
   Classifies query complexity and selects the appropriate
   reasoning mode: System 1 (single model, fast), System 2
   (debate), or Tree-of-Thought (multi-branch).

2. GOAL TRACKER (Updating)
   Maintains user intent across conversation turns. Detects
   when the user is building toward a larger goal vs asking
   independent questions.

3. TASK DECOMPOSER (Planning)
   Breaks complex multi-part queries into sub-queries that
   can be answered independently and then composed.

4. WORKING MEMORY MANAGER (Updating)
   Selects which episodic and semantic context is most relevant
   for the current query, not just top-K similarity.

5. INHIBITION CONTROL (Inhibition)
   Detects when the system lacks sufficient information and
   should ask a clarifying question instead of guessing.

Freudian mapping:
   Executive function IS the Ego's reality principle — the
   capacity to delay impulse (Id), consult conscience (Superego),
   plan ahead, and act deliberately rather than reactively.

Usage:
    ef = ExecutiveFunction()
    decision = ef.decide(query, session_history)
    # decision.mode = "tot"
    # decision.sub_queries = ["What is X?", "How does X relate to Y?"]
    # decision.inhibit = False
    # decision.context_strategy = "episodic_recent"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------


@dataclass
class ExecutiveDecision:
    """The executive function's decision about how to process a query.

    Attributes:
        mode: Reasoning mode to use.
        complexity: Estimated complexity (1-5).
        sub_queries: Decomposed sub-queries (if complex).
        inhibit: Whether to ask for clarification instead.
        inhibit_reason: Why clarification is needed.
        context_strategy: How to retrieve context.
        goal_continuation: Whether this continues a prior goal.
        goal_summary: Summary of the ongoing goal if any.
    """

    mode: str = "single"  # "single", "debate", "tot"
    complexity: int = 1  # 1-5
    sub_queries: List[str] = field(default_factory=list)
    inhibit: bool = False
    inhibit_reason: str = ""
    context_strategy: str = "default"
    goal_continuation: bool = False
    goal_summary: str = ""


# ---------------------------------------------------------------------------
# Complexity signals
# ---------------------------------------------------------------------------

# Keywords suggesting simple factual queries (System 1)
SIMPLE_SIGNALS = {
    "what is",
    "who is",
    "when did",
    "where is",
    "define",
    "what does",
    "how many",
    "what year",
    "capital of",
    "symbol for",
    "formula for",
    "true or false",
    "yes or no",
}

# Keywords suggesting complex reasoning (System 2 debate)
COMPLEX_SIGNALS = {
    "compare",
    "contrast",
    "analyze",
    "evaluate",
    "pros and cons",
    "trade-off",
    "should",
    "explain why",
    "what caused",
    "how does",
    "implications of",
    "relationship between",
    "advantages and disadvantages",
}

# Keywords suggesting deep ethical/philosophical (Tree-of-Thought)
DEEP_SIGNALS = {
    "ethical",
    "moral",
    "dilemma",
    "philosophy",
    "should we",
    "is it right",
    "fairness",
    "what if",
    "imagine",
    "hypothetical",
    "paradox",
    "tension between",
    "competing values",
    "multiple perspectives",
    "from every angle",
}

# Keywords suggesting multi-part queries (decomposition needed)
MULTI_PART_SIGNALS = {
    " and then ",
    " also ",
    " additionally ",
    "first.*then",
    "compare.*and.*recommend",
    "list.*and.*explain",
    "both.*and",
    r"\d+\.",
    r"\d+\)",  # Numbered items
}

# Keywords suggesting ambiguity (inhibition needed)
AMBIGUOUS_SIGNALS = {
    "it",
    "that",
    "this thing",
    "the one",
    "you know what i mean",
    "the usual",
    "same as before",
    "like last time",
}


# ---------------------------------------------------------------------------
# Executive Function
# ---------------------------------------------------------------------------


class ExecutiveFunction:
    """Unified cognitive control for Atlas AI.

    Analyzes each query to determine the optimal processing
    strategy before committing resources. This prevents wasting
    a 60-second debate on "What is 2+2?" and ensures complex
    ethical questions get the full Tree-of-Thought treatment.
    """

    def __init__(
        self,
        session_history: Optional[List[Dict]] = None,
        enable_inhibition: bool = True,
        enable_decomposition: bool = True,
    ) -> None:
        self._history: List[Dict] = session_history or []
        self._enable_inhibition = enable_inhibition
        self._enable_decomposition = enable_decomposition
        self._current_goal: Optional[str] = None
        self._turn_count = 0

    def decide(
        self,
        query: str,
        session_history: Optional[List[Dict]] = None,
    ) -> ExecutiveDecision:
        """Make an executive decision about how to process a query.

        This is the main entry point. Analyzes the query and
        returns a decision about mode, decomposition, inhibition,
        and context strategy.

        Args:
            query: The user's current query.
            session_history: Optional conversation history.

        Returns:
            ExecutiveDecision with processing instructions.
        """
        if session_history is not None:
            self._history = session_history
        self._turn_count += 1

        decision = ExecutiveDecision()

        # Step 1: Estimate complexity
        decision.complexity = self._estimate_complexity(query)

        # Step 2: Select reasoning mode based on complexity
        decision.mode = self._select_mode(query, decision.complexity)

        # Step 3: Check for multi-part queries (decomposition)
        if self._enable_decomposition:
            sub_queries = self._decompose(query)
            if len(sub_queries) > 1:
                decision.sub_queries = sub_queries

        # Step 4: Check for ambiguity (inhibition)
        if self._enable_inhibition:
            inhibit, reason = self._check_inhibition(query)
            if inhibit:
                decision.inhibit = True
                decision.inhibit_reason = reason

        # Step 5: Detect goal continuation
        goal_cont, goal_summary = self._track_goal(query)
        decision.goal_continuation = goal_cont
        decision.goal_summary = goal_summary

        # Step 6: Select context retrieval strategy
        decision.context_strategy = self._select_context(query, decision)

        logger.info(
            "[executive] query=%s... -> mode=%s complexity=%d "
            "subs=%d inhibit=%s goal=%s",
            query[:30],
            decision.mode,
            decision.complexity,
            len(decision.sub_queries),
            decision.inhibit,
            decision.goal_continuation,
        )

        return decision

    # ------------------------------------------------------------------
    # Core functions
    # ------------------------------------------------------------------

    def _estimate_complexity(self, query: str) -> int:
        """Estimate query complexity on a 1-5 scale.

        1: Simple factual (what is X?)
        2: Moderate lookup (explain X)
        3: Analysis needed (compare X and Y)
        4: Deep reasoning (ethical dilemma)
        5: Multi-step with synthesis (compare, then recommend)
        """
        lower = query.lower()
        score = 2  # Default moderate

        # Simple signals reduce complexity
        simple_hits = sum(1 for s in SIMPLE_SIGNALS if s in lower)
        if simple_hits >= 2:
            score = 1
        elif simple_hits == 1:
            score = max(1, score - 1)

        # Complex signals increase
        complex_hits = sum(1 for s in COMPLEX_SIGNALS if s in lower)
        score += min(2, complex_hits)

        # Deep signals push to 4+
        deep_hits = sum(1 for s in DEEP_SIGNALS if s in lower)
        if deep_hits >= 2:
            score = max(score, 4)
        elif deep_hits == 1:
            score = max(score, 3)

        # Multi-part queries are 4+
        multi_hits = sum(1 for s in MULTI_PART_SIGNALS if re.search(s, lower))
        if multi_hits >= 1:
            score = max(score, 4)

        # Long queries tend to be more complex
        word_count = len(query.split())
        if word_count > 50:
            score = max(score, 3)
        elif word_count > 100:
            score = max(score, 4)

        # Question marks: multiple = multi-part
        q_count = query.count("?")
        if q_count >= 3:
            score = max(score, 4)
        elif q_count == 2:
            score = max(score, 3)

        return min(5, max(1, score))

    def _select_mode(self, query: str, complexity: int) -> str:
        """Select reasoning mode based on complexity.

        1-2: System 1 (single model, fast)
        3:   System 2 (debate between Superego and Id)
        4-5: Tree-of-Thought (multi-branch exploration)
        """
        if complexity <= 2:
            return "single"
        elif complexity == 3:
            return "debate"
        else:
            return "tot"

    def _decompose(self, query: str) -> List[str]:
        """Decompose a multi-part query into sub-queries.

        Looks for explicit separators (and, then, also,
        numbered lists) and splits into independent questions.
        """
        # Check for numbered items: "1. ... 2. ... 3. ..."
        numbered = re.findall(
            r"(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)",
            query,
            re.DOTALL,
        )
        if len(numbered) >= 2:
            return [q.strip() for q in numbered if q.strip()]

        # Check for multiple questions separated by "?"
        questions = [
            q.strip() + "?"
            for q in query.split("?")
            if q.strip() and len(q.strip()) > 10
        ]
        if len(questions) >= 2:
            return questions

        # Check for "X and then Y" or "X, also Y"
        for sep in [" and then ", " then ", ". Also, ", ". Additionally, "]:
            if sep.lower() in query.lower():
                parts = re.split(re.escape(sep), query, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    return [p.strip() for p in parts if p.strip()]

        return [query]  # No decomposition needed

    def _check_inhibition(self, query: str) -> tuple:
        """Check if the query is too ambiguous to answer well.

        Returns (should_inhibit, reason).
        """
        lower = query.lower().strip()

        # Very short queries might be ambiguous
        if len(lower) < 5:
            return True, "Query is very short. Could you elaborate?"

        # Dangling references without history
        if not self._history:
            for signal in ["it", "that", "this", "them"]:
                # Check if pronoun is used without clear referent
                if (
                    lower.startswith(f"what about {signal}")
                    or lower.startswith(f"tell me more about {signal}")
                    or lower == signal
                ):
                    return True, (
                        f"I'm not sure what '{signal}' refers to. "
                        "Could you be more specific?"
                    )

        # "Same as before" without history
        if not self._history and any(
            s in lower for s in ["same as before", "like last time", "again"]
        ):
            return True, (
                "I don't have context from a previous conversation. "
                "Could you restate your request?"
            )

        return False, ""

    def _track_goal(self, query: str) -> tuple:
        """Detect if this query continues an ongoing goal.

        Returns (is_continuation, goal_summary).
        """
        if not self._history:
            return False, ""

        # Check if the query references something from history
        lower = query.lower()
        continuation_signals = [
            "building on",
            "following up",
            "continuing",
            "next step",
            "what about",
            "and also",
            "related to that",
            "in addition",
        ]

        is_cont = any(s in lower for s in continuation_signals)

        if is_cont:
            if self._current_goal:
                return True, self._current_goal
            # First continuation detected — set goal from history
            self._current_goal = "Continuing from previous turn"
            return True, self._current_goal

        # Try to detect implicit continuation from topic overlap
        if self._history:
            last_query = ""
            for msg in reversed(self._history):
                if msg.get("role") == "user":
                    last_query = msg.get("content", "").lower()
                    break

            if last_query:
                # Simple word overlap check
                prev_words = set(w for w in last_query.split() if len(w) > 4)
                curr_words = set(w for w in lower.split() if len(w) > 4)
                overlap = len(prev_words & curr_words)
                if overlap >= 3:
                    goal = "Continuing topic from previous turn"
                    self._current_goal = goal
                    return True, goal

        return False, ""

    def _select_context(
        self,
        query: str,
        decision: ExecutiveDecision,
    ) -> str:
        """Select the optimal context retrieval strategy.

        Options:
        - "default": Standard RAG (wiki + pgvector + FTS)
        - "episodic_recent": Prioritize recent conversation history
        - "semantic_deep": Deep semantic search with more results
        - "wiki_focused": Prioritize dream-consolidated wiki
        - "minimal": Just the query, no RAG (for simple factual)
        """
        lower = query.lower()

        # Research/evidence queries always need deep search
        if any(w in lower for w in ["research", "paper", "study", "evidence"]):
            return "semantic_deep"

        if decision.complexity <= 1:
            return "minimal"

        if decision.goal_continuation:
            return "episodic_recent"

        if any(
            w in lower
            for w in [
                "we discussed",
                "you told me",
                "remember when",
                "last time",
            ]
        ):
            return "episodic_recent"

        if decision.complexity >= 4:
            return "semantic_deep"

        return "default"
