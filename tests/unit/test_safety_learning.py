# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.safety.learning.service module."""

import pytest
from unittest.mock import MagicMock

from agi.safety.learning.service import (
    OutcomeFeedback,
    RuleStats,
    SafetyLearner,
    SafetyLearnerConfig,
)
from agi.safety.rules.engine import SafetyRule, SafetyRuleEngine


def _make_engine(rule_id="rule_1", priority=100):
    engine = SafetyRuleEngine()
    rule = MagicMock(spec=SafetyRule)
    rule.id = rule_id
    rule.priority = priority
    engine._rules[rule_id] = rule
    return engine, rule


class TestSafetyLearnerConfig:
    def test_default_config(self):
        cfg = SafetyLearnerConfig()
        assert cfg.learning_rate == 0.01
        assert cfg.min_samples == 10
        assert cfg.anomaly_threshold == 0.5
        assert cfg.max_weight_delta == 0.1

    def test_custom_config(self):
        cfg = SafetyLearnerConfig(
            learning_rate=0.05,
            min_samples=20,
            anomaly_threshold=0.3,
            max_weight_delta=0.2,
        )
        assert cfg.learning_rate == 0.05
        assert cfg.min_samples == 20


class TestOutcomeFeedback:
    def test_basic(self):
        fb = OutcomeFeedback(
            action_id="a1",
            predicted_safe=True,
            actual_safe=True,
        )
        assert fb.action_id == "a1"
        assert fb.severity == 0.0
        assert fb.violated_rules == []

    def test_severity_bounds(self):
        fb = OutcomeFeedback(
            action_id="a2",
            predicted_safe=True,
            actual_safe=False,
            severity=0.8,
            violated_rules=["r1"],
        )
        assert fb.severity == 0.8

    def test_severity_too_high(self):
        with pytest.raises(ValueError):
            OutcomeFeedback(
                action_id="a3",
                predicted_safe=True,
                actual_safe=False,
                severity=1.5,
            )

    def test_severity_too_low(self):
        with pytest.raises(ValueError):
            OutcomeFeedback(
                action_id="a4",
                predicted_safe=True,
                actual_safe=False,
                severity=-0.1,
            )


class TestRuleStats:
    def test_defaults(self):
        s = RuleStats()
        assert s.total_triggers == 0

    def test_precision(self):
        s = RuleStats(true_positives=8, false_positives=2)
        assert s.precision == pytest.approx(0.8)

    def test_precision_empty(self):
        assert RuleStats().precision == 0.0

    def test_recall(self):
        s = RuleStats(true_positives=7, false_negatives=3)
        assert s.recall == pytest.approx(0.7)

    def test_recall_empty(self):
        assert RuleStats().recall == 0.0

    def test_f1(self):
        s = RuleStats(true_positives=8, false_positives=2, false_negatives=2)
        p, r = 0.8, 0.8
        assert s.f1_score == pytest.approx(2 * p * r / (p + r))

    def test_f1_empty(self):
        assert RuleStats().f1_score == 0.0

    def test_accuracy(self):
        s = RuleStats(
            true_positives=4,
            false_positives=1,
            true_negatives=3,
            false_negatives=2,
        )
        assert s.accuracy == pytest.approx(0.7)

    def test_accuracy_empty(self):
        assert RuleStats().accuracy == 0.0


class TestSafetyLearnerInit:
    def test_default(self):
        learner = SafetyLearner(SafetyRuleEngine())
        assert learner.total_outcomes == 0
        assert learner.rule_stats == {}

    def test_custom(self):
        learner = SafetyLearner(
            SafetyRuleEngine(),
            learning_rate=0.05,
            min_samples=5,
        )
        assert learner._config.learning_rate == 0.05

    def test_from_config(self):
        cfg = SafetyLearnerConfig(
            learning_rate=0.02,
            min_samples=15,
            anomaly_threshold=0.3,
        )
        learner = SafetyLearner.from_config(SafetyRuleEngine(), cfg)
        assert learner._config.anomaly_threshold == 0.3


class TestRecordOutcome:
    def test_true_negative(self):
        learner = SafetyLearner(SafetyRuleEngine())
        learner.record_outcome(
            OutcomeFeedback(
                action_id="a1",
                predicted_safe=True,
                actual_safe=True,
                violated_rules=["r1"],
            )
        )
        assert learner.rule_stats["r1"].true_negatives == 1

    def test_true_positive(self):
        learner = SafetyLearner(SafetyRuleEngine())
        learner.record_outcome(
            OutcomeFeedback(
                action_id="a2",
                predicted_safe=False,
                actual_safe=False,
                violated_rules=["r2"],
            )
        )
        assert learner.rule_stats["r2"].true_positives == 1

    def test_false_negative_increases_weight(self):
        engine, rule = _make_engine("rfn", 100)
        learner = SafetyLearner(engine, learning_rate=0.1)
        learner.record_outcome(
            OutcomeFeedback(
                action_id="a3",
                predicted_safe=True,
                actual_safe=False,
                violated_rules=["rfn"],
                severity=0.5,
            )
        )
        assert rule.priority > 100

    def test_false_positive_decreases_weight(self):
        engine, rule = _make_engine("rfp", 500)
        learner = SafetyLearner(engine, learning_rate=0.1)
        learner.record_outcome(
            OutcomeFeedback(
                action_id="a4",
                predicted_safe=False,
                actual_safe=True,
                violated_rules=["rfp"],
            )
        )
        assert rule.priority < 500

    def test_accumulate(self):
        learner = SafetyLearner(SafetyRuleEngine())
        for i in range(5):
            learner.record_outcome(
                OutcomeFeedback(
                    action_id=f"a{i}",
                    predicted_safe=True,
                    actual_safe=True,
                    violated_rules=["rx"],
                )
            )
        assert learner.total_outcomes == 5
        assert learner.rule_stats["rx"].true_negatives == 5

    def test_reset(self):
        learner = SafetyLearner(SafetyRuleEngine())
        learner.record_outcome(
            OutcomeFeedback(
                action_id="a1",
                predicted_safe=True,
                actual_safe=True,
                violated_rules=["r1"],
            )
        )
        learner.reset_stats()
        assert learner.total_outcomes == 0


class TestPerformance:
    def test_empty(self):
        assert SafetyLearner(SafetyRuleEngine()).get_rule_performance() == {}

    def test_below_min(self):
        learner = SafetyLearner(SafetyRuleEngine(), min_samples=10)
        for i in range(5):
            learner.record_outcome(
                OutcomeFeedback(
                    action_id=f"a{i}",
                    predicted_safe=True,
                    actual_safe=True,
                    violated_rules=["r1"],
                )
            )
        assert learner.get_rule_performance() == {}

    def test_above_min(self):
        learner = SafetyLearner(SafetyRuleEngine(), min_samples=5)
        for i in range(5):
            learner.record_outcome(
                OutcomeFeedback(
                    action_id=f"a{i}",
                    predicted_safe=False,
                    actual_safe=False,
                    violated_rules=["rp"],
                )
            )
        perf = learner.get_rule_performance()
        assert perf["rp"]["true_positives"] == 5


class TestAnomalies:
    def test_no_data(self):
        assert SafetyLearner(SafetyRuleEngine()).detect_anomalies() == []

    def test_high_fn_rate(self):
        learner = SafetyLearner(
            SafetyRuleEngine(),
            min_samples=5,
            learning_rate=0.0,
        )
        learner._config.anomaly_threshold = 0.3
        for i in range(10):
            learner.record_outcome(
                OutcomeFeedback(
                    action_id=f"a{i}",
                    predicted_safe=True,
                    actual_safe=False,
                    severity=0.1,
                )
            )
        anomalies = learner.detect_anomalies(recent_window=10)
        assert any("false negative rate" in a.lower() for a in anomalies)

    def test_low_precision(self):
        learner = SafetyLearner(
            SafetyRuleEngine(),
            min_samples=5,
            learning_rate=0.0,
        )
        learner._config.anomaly_threshold = 0.5
        for i in range(10):
            learner.record_outcome(
                OutcomeFeedback(
                    action_id=f"a{i}",
                    predicted_safe=False,
                    actual_safe=True,
                    violated_rules=["lp"],
                )
            )
        assert any("low precision" in a.lower() for a in learner.detect_anomalies())

    def test_good_performance(self):
        learner = SafetyLearner(SafetyRuleEngine(), min_samples=5)
        for i in range(20):
            learner.record_outcome(
                OutcomeFeedback(
                    action_id=f"a{i}",
                    predicted_safe=True,
                    actual_safe=True,
                    violated_rules=["gr"],
                )
            )
        assert learner.detect_anomalies() == []
