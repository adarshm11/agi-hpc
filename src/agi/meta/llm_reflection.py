# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
LLM-Based Metacognitive Reflection for AGI-HPC.

Uses language models for deep plan critique, explanation generation,
and creative problem-solving.

Implements Sprint 6 requirements:
- Plan critique with structured prompts
- Human-readable explanation generation
- Creative alternative approach suggestion
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CRITIQUE_PROMPT = (
    "Analyze the following plan for logical consistency, completeness, "
    "and potential issues. For each issue found, provide a severity "
    "(low/medium/high), a description, and a suggestion for fixing it. "
    "Rate the overall plan quality from 0.0 to 1.0.\n\n"
    "Plan:\n{plan}\n"
)

EXPLANATION_PROMPT = (
    "Explain the following plan in a way that is understandable by "
    "a {audience} audience. Summarize the goal, key steps, and expected "
    "outcome.\n\n"
    "Plan:\n{plan}\n"
)

CREATIVE_PROMPT = (
    "Given the following plan and constraints, suggest alternative "
    "approaches that may be more efficient, robust, or creative. "
    "Return each alternative as a structured suggestion.\n\n"
    "Plan:\n{plan}\n"
    "Constraints:\n{constraints}\n"
)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class ReflectionConfig:
    """Configuration for the LLM reflector."""

    model_name: str = "gpt-4"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: str = "You are a metacognitive reviewer for an AGI planning system."


@dataclass
class PlanCritique:
    """Structured result from plan critique."""

    issues: List[str] = field(default_factory=list)
    score: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM Reflector
# ---------------------------------------------------------------------------


class LLMReflector:
    """LLM-based metacognitive reflector.

    Provides plan critique, explanation generation, and creative
    alternative suggestion using a language model backend.  When no
    LLM client is provided, returns sensible stub responses so that
    downstream code can operate without an active LLM connection.
    """

    def __init__(
        self,
        config: Optional[ReflectionConfig] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.config = config or ReflectionConfig()
        self._client = llm_client
        logger.info(
            "[meta][reflection] initialized model=%s",
            self.config.model_name,
        )

    # ------------------------------------------------------------------
    # Critique
    # ------------------------------------------------------------------

    async def critique_plan(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> PlanCritique:
        """Critique a plan for issues and improvements.

        Args:
            plan: Plan dictionary to critique.
            context: Optional additional context.

        Returns:
            PlanCritique with issues, score, and suggestions.
        """
        if not self._client:
            logger.debug("[meta][reflection] no client, returning stub critique")
            return PlanCritique(score=0.5)

        prompt = CRITIQUE_PROMPT.format(plan=json.dumps(plan, indent=2))
        if context:
            prompt += f"\nContext:\n{json.dumps(context, indent=2)}\n"

        try:
            response = self._client.complete(
                prompt,
                system=self.config.system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return self._parse_critique(response.content)
        except Exception as e:
            logger.error("[meta][reflection] critique failed: %s", e)
            return PlanCritique(issues=[str(e)], score=0.0)

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    async def explain(
        self,
        plan: Dict[str, Any],
        audience: str = "developer",
    ) -> str:
        """Generate a human-readable explanation of a plan.

        Args:
            plan: Plan dictionary to explain.
            audience: Target audience (e.g. "developer", "non-technical").

        Returns:
            Explanation string.
        """
        if not self._client:
            steps = plan.get("steps", [])
            return f"Plan with {len(steps)} step(s) for a {audience} audience."

        prompt = EXPLANATION_PROMPT.format(
            plan=json.dumps(plan, indent=2),
            audience=audience,
        )

        try:
            response = self._client.complete(
                prompt,
                system=self.config.system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.content
        except Exception as e:
            logger.error("[meta][reflection] explanation failed: %s", e)
            return f"Explanation unavailable: {e}"

    # ------------------------------------------------------------------
    # Creative alternatives
    # ------------------------------------------------------------------

    async def suggest_alternatives(
        self,
        plan: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Suggest alternative approaches for a plan.

        Args:
            plan: Plan dictionary.
            constraints: Optional constraints to respect.

        Returns:
            List of alternative suggestion dictionaries.
        """
        if not self._client:
            return []

        prompt = CREATIVE_PROMPT.format(
            plan=json.dumps(plan, indent=2),
            constraints=json.dumps(constraints or {}, indent=2),
        )

        try:
            response = self._client.complete(
                prompt,
                system=self.config.system_prompt,
                temperature=min(1.0, self.config.temperature + 0.2),
                max_tokens=self.config.max_tokens,
            )
            parsed = json.loads(response.content)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception as e:
            logger.error("[meta][reflection] alternatives failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_critique(content: str) -> PlanCritique:
        """Parse LLM response into PlanCritique."""
        try:
            data = json.loads(content)
            return PlanCritique(
                issues=data.get("issues", []),
                score=float(data.get("score", 0.5)),
                suggestions=data.get("suggestions", []),
                raw_response=content,
            )
        except (json.JSONDecodeError, ValueError):
            return PlanCritique(
                score=0.5,
                raw_response=content,
            )
