# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Safety Learning Service for AGI-HPC.

Provides online learning for safety rule weights using Bayesian updates
based on action outcome feedback. Tracks rule performance metrics,
detects anomalous safety patterns, and adjusts rule priorities to
improve safety prediction accuracy over time.

Sprint 6 implementation.
"""

from agi.safety.learning.service import (
    OutcomeFeedback,
    RuleStats,
    SafetyLearner,
    SafetyLearnerConfig,
)

__all__ = [
    "OutcomeFeedback",
    "RuleStats",
    "SafetyLearner",
    "SafetyLearnerConfig",
]
