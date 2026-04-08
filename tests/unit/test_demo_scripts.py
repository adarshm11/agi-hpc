# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the demo scripts.

Tests script existence, importability, and the safety demo
which can run locally without external services.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class TestDemoScriptsExist:
    """Tests that all demo scripts exist."""

    def test_safety_demo(self) -> None:
        assert Path("scripts/demo_safety_pipeline.py").exists()

    def test_dream_demo(self) -> None:
        assert Path("scripts/demo_dream_cycle.py").exists()

    def test_full_loop_demo(self) -> None:
        assert Path("scripts/demo_full_loop.py").exists()


class TestSafetyDemo:
    """Tests the safety demo which runs locally."""

    def test_runs_successfully(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/demo_safety_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Safety Pipeline Demo" in result.stdout

    def test_detects_injections(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/demo_safety_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "VETO" in result.stdout

    def test_passes_safe_queries(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/demo_safety_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "PASS" in result.stdout

    def test_reports_audit_log(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/demo_safety_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "Audit log:" in result.stdout


class TestDreamDemoDryRun:
    """Tests the dream demo in dry-run mode."""

    def test_dry_run(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/demo_dream_cycle.py", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "EPISODIC REPLAY" in result.stdout
        assert "TOPIC CLUSTERING" in result.stdout
        assert "CERTAINTY ASSESSMENT" in result.stdout
        assert "CREATIVE DREAMING" in result.stdout
        assert "HOUSEKEEPING" in result.stdout


class TestFullLoopDemoDryRun:
    """Tests the full loop demo in dry-run mode."""

    def test_dry_run(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/demo_full_loop.py", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "SAFETY INPUT GATE" in result.stdout
        assert "PSYCHE DEBATE" in result.stdout
        assert "EPISODE STORED" in result.stdout
        assert "DM TRAINING" in result.stdout
        assert "DREAMING NAP" in result.stdout
