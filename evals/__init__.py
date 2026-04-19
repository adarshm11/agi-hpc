"""Evaluation harness — offline, repeatable, read-only.

Public API:
    from evals.harness import run_harness, load_portfolio

See evals/README.md for design and usage.
"""

from .harness import HARNESS_VERSION, CaseResult, load_portfolio, run_harness

__all__ = ["HARNESS_VERSION", "CaseResult", "load_portfolio", "run_harness"]
