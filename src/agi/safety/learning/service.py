# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Safety Learning Service for AGI-HPC.

Implements online learning for safety rule weights using Bayesian-inspired
weight updates based on action outcome feedback. The SafetyLearner monitors
how well each safety rule predicts actual outcomes and adjusts rule priorities
accordingly:

- False negatives (missed dangers) increase rule weight
- False positives (unnecessary blocks) slightly decrease rule weight
- Performance metrics are tracked per rule for anomaly detection

Sprint 6 implementation.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agi.safety.rules.engine import SafetyRuleEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class OutcomeFeedback:
    """Feedback from action outcome.

    Captures whether a safety prediction matched reality, including
    which rules were violated and any unintended side effects.

    Attributes:
        action_id: Unique identifier for the action that was evaluated.
        predicted_safe: Whether the safety system predicted the action was safe.
        actual_safe: Whether the action was actually safe in practice.
        violated_rules: List of rule IDs that were violated by the action.
        unintended_effects: List of descriptions of unintended side effects.
        severity: Severity of the outcome on a 0-1 scale (0 = benign, 1 = catastrophic).
    """

    action_id: str
    predicted_safe: bool
    actual_safe: bool
    violated_rules: list[str] = field(default_factory=list)
    unintended_effects: list[str] = field(default_factory=list)
    severity: float = 0.0

    def __post_init__(self) -> None:
        """Validate severity is in range [0, 1]."""
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"Severity must be between 0 and 1, got {self.severity}")


@dataclass
class RuleStats:
    """Statistics for a safety rule.

    Tracks confusion matrix counts for evaluating rule prediction accuracy.

    Attributes:
        total_triggers: Total number of times this rule was involved in an evaluation.
        true_positives: Rule correctly flagged an unsafe action.
        false_positives: Rule incorrectly flagged a safe action as unsafe.
        true_negatives: Rule correctly allowed a safe action.
        false_negatives: Rule failed to flag an unsafe action.
    """

    total_triggers: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP). Returns 0.0 if no positives."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN). Returns 0.0 if no actual positives."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def f1_score(self) -> float:
        """Calculate F1 score: harmonic mean of precision and recall."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy: (TP + TN) / total. Returns 0.0 if no samples."""
        total = (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total


@dataclass
class SafetyLearnerConfig:
    """Configuration for the SafetyLearner.

    Attributes:
        learning_rate: Base learning rate for weight adjustments (default 0.01).
        min_samples: Minimum number of samples before reporting performance
            metrics or making weight adjustments (default 10).
        anomaly_threshold: Threshold for flagging anomalous rule performance.
            Rules with false negative rate above this are flagged (default 0.5).
        max_weight_delta: Maximum absolute weight change per update (default 0.1).
    """

    learning_rate: float = 0.01
    min_samples: int = 10
    anomaly_threshold: float = 0.5
    max_weight_delta: float = 0.1


# ---------------------------------------------------------------------------
# Safety Learner
# ---------------------------------------------------------------------------


class SafetyLearner:
    """Online learning for safety rule weights.

    Uses action outcome feedback to update rule weights in the SafetyRuleEngine.
    Rules that miss dangerous actions (false negatives) have their weight
    increased, while rules that unnecessarily block safe actions (false
    positives) have their weight slightly decreased.

    The learner also tracks per-rule performance metrics and can detect
    anomalous patterns such as sudden increases in false negatives or
    persistently low-precision rules.

    Args:
        rule_engine: The SafetyRuleEngine instance whose rule weights
            will be adjusted based on learned outcomes.
        learning_rate: Base learning rate for Bayesian weight updates.
        min_samples: Minimum number of outcome samples before adjusting
            weights or reporting metrics for a rule.
    """

    def __init__(
        self,
        rule_engine: SafetyRuleEngine,
        learning_rate: float = 0.01,
        min_samples: int = 10,
    ) -> None:
        self._rule_engine = rule_engine
        self._config = SafetyLearnerConfig(
            learning_rate=learning_rate,
            min_samples=min_samples,
        )
        self._rule_stats: dict[str, RuleStats] = {}
        self._recent_outcomes: deque[OutcomeFeedback] = deque(maxlen=1000)
        self._total_outcomes: int = 0
        logger.info(
            "SafetyLearner initialized with learning_rate=%.4f, min_samples=%d",
            learning_rate,
            min_samples,
        )

    @classmethod
    def from_config(
        cls,
        rule_engine: SafetyRuleEngine,
        config: SafetyLearnerConfig,
    ) -> SafetyLearner:
        """Create a SafetyLearner from a config dataclass.

        Args:
            rule_engine: The SafetyRuleEngine instance.
            config: Configuration dataclass.

        Returns:
            Configured SafetyLearner instance.
        """
        learner = cls(
            rule_engine=rule_engine,
            learning_rate=config.learning_rate,
            min_samples=config.min_samples,
        )
        learner._config = config
        return learner

    # ------------------------------------------------------------------
    # Outcome Recording
    # ------------------------------------------------------------------

    def record_outcome(self, feedback: OutcomeFeedback) -> None:
        """Record an action outcome and update rule weights accordingly.

        Classification logic:
        - If the action was predicted safe but was actually unsafe (false negative),
          the violated rules' weights are increased proportional to severity.
        - If the action was predicted unsafe but was actually safe (false positive),
          the violated rules' weights are slightly decreased.
        - True positives and true negatives update statistics without weight changes.

        Args:
            feedback: Outcome feedback for a completed action.
        """
        self._recent_outcomes.append(feedback)
        self._total_outcomes += 1

        if feedback.predicted_safe and feedback.actual_safe:
            # True negative: predicted safe, was safe
            self._record_true_negative(feedback)
        elif not feedback.predicted_safe and not feedback.actual_safe:
            # True positive: predicted unsafe, was unsafe
            self._record_true_positive(feedback)
        elif feedback.predicted_safe and not feedback.actual_safe:
            # False negative: predicted safe but was actually unsafe
            self._record_false_negative(feedback)
        elif not feedback.predicted_safe and feedback.actual_safe:
            # False positive: predicted unsafe but was actually safe
            self._record_false_positive(feedback)

        logger.debug(
            "Recorded outcome for action %s: predicted_safe=%s, actual_safe=%s",
            feedback.action_id,
            feedback.predicted_safe,
            feedback.actual_safe,
        )

    def _record_true_negative(self, feedback: OutcomeFeedback) -> None:
        """Record a true negative outcome (predicted safe, was safe).

        No weight adjustment needed - the system correctly identified safety.

        Args:
            feedback: The outcome feedback.
        """
        for rule_id in feedback.violated_rules:
            stats = self._get_or_create_stats(rule_id)
            stats.true_negatives += 1
            stats.total_triggers += 1

    def _record_true_positive(self, feedback: OutcomeFeedback) -> None:
        """Record a true positive outcome (predicted unsafe, was unsafe).

        No weight adjustment needed - the rules correctly caught the danger.

        Args:
            feedback: The outcome feedback.
        """
        for rule_id in feedback.violated_rules:
            stats = self._get_or_create_stats(rule_id)
            stats.true_positives += 1
            stats.total_triggers += 1

    def _record_false_negative(self, feedback: OutcomeFeedback) -> None:
        """Record a false negative (predicted safe but was actually unsafe).

        This is the most dangerous case: the safety system missed a real danger.
        Increase the weight of violated rules proportional to severity so they
        are more likely to trigger in future evaluations.

        Args:
            feedback: The outcome feedback.
        """
        for rule_id in feedback.violated_rules:
            stats = self._get_or_create_stats(rule_id)
            stats.false_negatives += 1
            stats.total_triggers += 1

            # Increase weight: larger increase for higher severity
            # Scale by learning rate and severity
            delta = self._config.learning_rate * (1.0 + feedback.severity)
            delta = min(delta, self._config.max_weight_delta)
            self._adjust_rule_weight(rule_id, delta)

            logger.warning(
                "False negative for rule %s (severity=%.2f): increasing weight by %.4f",
                rule_id,
                feedback.severity,
                delta,
            )

    def _record_false_positive(self, feedback: OutcomeFeedback) -> None:
        """Record a false positive (predicted unsafe but was actually safe).

        Slightly decrease rule weight since the rule was overly cautious.
        The decrease is smaller than false negative increases to maintain
        a safety-conservative bias.

        Args:
            feedback: The outcome feedback.
        """
        for rule_id in feedback.violated_rules:
            stats = self._get_or_create_stats(rule_id)
            stats.false_positives += 1
            stats.total_triggers += 1

            # Decrease weight: smaller decrease than false negative increase
            # to maintain conservative safety bias (factor of 0.5)
            delta = -self._config.learning_rate * 0.5
            delta = max(delta, -self._config.max_weight_delta)
            self._adjust_rule_weight(rule_id, delta)

            logger.info(
                "False positive for rule %s: decreasing weight by %.4f",
                rule_id,
                abs(delta),
            )

    # ------------------------------------------------------------------
    # Weight Adjustment
    # ------------------------------------------------------------------

    def _adjust_rule_weight(self, rule_id: str, delta: float) -> None:
        """Adjust a rule's priority/weight, clamped to [1, 1000].

        Looks up the rule in the engine and modifies its priority field
        directly. The delta is added to the current priority.

        Args:
            rule_id: The ID of the rule to adjust.
            delta: The weight change to apply. Positive increases priority,
                negative decreases it.
        """
        rule = self._rule_engine.get_rule(rule_id)
        if rule is None:
            logger.warning("Cannot adjust weight for unknown rule: %s", rule_id)
            return

        old_priority = rule.priority
        new_priority = old_priority + delta * old_priority
        new_priority = max(1, min(1000, int(round(new_priority))))
        rule.priority = new_priority

        logger.debug(
            "Adjusted rule %s priority: %d -> %d (delta=%.4f)",
            rule_id,
            old_priority,
            new_priority,
            delta,
        )

    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------

    def get_rule_performance(self) -> dict[str, dict[str, Any]]:
        """Get performance metrics for rules with enough samples.

        Returns metrics only for rules that have accumulated at least
        min_samples outcomes, to avoid noisy statistics.

        Returns:
            Dictionary mapping rule_id to a dict with performance metrics.
        """
        results: dict[str, dict[str, Any]] = {}
        for rule_id, stats in self._rule_stats.items():
            if stats.total_triggers < self._config.min_samples:
                continue
            results[rule_id] = {
                "precision": stats.precision,
                "recall": stats.recall,
                "f1_score": stats.f1_score,
                "accuracy": stats.accuracy,
                "total_triggers": stats.total_triggers,
                "true_positives": stats.true_positives,
                "false_positives": stats.false_positives,
                "true_negatives": stats.true_negatives,
                "false_negatives": stats.false_negatives,
            }
        return results

    # ------------------------------------------------------------------
    # Anomaly Detection
    # ------------------------------------------------------------------

    def detect_anomalies(self, recent_window: int = 100) -> list[str]:
        """Detect anomalous safety patterns from recent outcomes.

        Checks for two types of anomalies:
        1. Sudden increase in false negatives in the recent window
           (false negative rate exceeds anomaly_threshold).
        2. Rules with persistently low precision (below anomaly_threshold)
           that have enough samples to be statistically meaningful.

        Args:
            recent_window: Number of recent outcomes to analyze for
                sudden pattern changes.

        Returns:
            List of human-readable anomaly descriptions.
        """
        anomalies: list[str] = []

        # Check 1: Recent false negative rate
        recent = list(self._recent_outcomes)[-recent_window:]
        if recent:
            false_negatives_count = sum(
                1 for fb in recent if fb.predicted_safe and not fb.actual_safe
            )
            fn_rate = false_negatives_count / len(recent)
            if fn_rate > self._config.anomaly_threshold:
                anomalies.append(
                    f"High false negative rate in last {len(recent)} outcomes: "
                    f"{fn_rate:.2%} (threshold: "
                    f"{self._config.anomaly_threshold:.2%})"
                )

        # Check 2: Low precision rules with enough samples
        for rule_id, stats in self._rule_stats.items():
            if stats.total_triggers < self._config.min_samples:
                continue

            if (
                stats.precision < self._config.anomaly_threshold
                and (stats.true_positives + stats.false_positives) > 0
            ):
                anomalies.append(
                    f"Rule {rule_id} has low precision: "
                    f"{stats.precision:.2%} over {stats.total_triggers} "
                    f"evaluations"
                )

            # Check for high false negative rate per rule
            total_actual_positive = stats.true_positives + stats.false_negatives
            if total_actual_positive > 0:
                fn_rate_rule = stats.false_negatives / total_actual_positive
                if (
                    fn_rate_rule > self._config.anomaly_threshold
                    and stats.total_triggers >= self._config.min_samples
                ):
                    anomalies.append(
                        f"Rule {rule_id} has high false negative rate: "
                        f"{fn_rate_rule:.2%} over {total_actual_positive} "
                        f"actual positive cases"
                    )

        if anomalies:
            logger.warning(
                "Detected %d safety anomalies: %s",
                len(anomalies),
                "; ".join(anomalies),
            )

        return anomalies

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_or_create_stats(self, rule_id: str) -> RuleStats:
        """Get or create a RuleStats entry for a given rule.

        Args:
            rule_id: The rule ID to look up.

        Returns:
            The RuleStats instance for this rule, creating a new one if
            it does not yet exist.
        """
        if rule_id not in self._rule_stats:
            self._rule_stats[rule_id] = RuleStats()
        return self._rule_stats[rule_id]

    @property
    def total_outcomes(self) -> int:
        """Return total number of outcomes recorded."""
        return self._total_outcomes

    @property
    def rule_stats(self) -> dict[str, RuleStats]:
        """Return a copy of the current rule statistics."""
        return dict(self._rule_stats)

    def reset_stats(self) -> None:
        """Reset all rule statistics and recent outcomes."""
        self._rule_stats.clear()
        self._recent_outcomes.clear()
        self._total_outcomes = 0
        logger.info("SafetyLearner statistics reset")
