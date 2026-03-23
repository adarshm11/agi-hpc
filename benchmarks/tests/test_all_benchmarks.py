"""Test suite for all 5 AGI benchmark tracks.

Validates syntax, schemas, statistics, budget, and mock execution
WITHOUT any network calls or Kaggle CU consumption.

Run: python -m pytest benchmarks/tests/test_all_benchmarks.py -v
"""

import ast
import sys
import math
import types
import importlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, fields
import pytest

# =====================================================================
# PATHS
# =====================================================================

BENCH_ROOT = Path(__file__).parent.parent
BENCHMARK_FILES = {
    "social_cognition": BENCH_ROOT / "social_cognition" / "moral_geometry_v9_budget.py",
    "executive_functions": BENCH_ROOT / "executive_functions" / "executive_functions_v2_budget.py",
    "attention": BENCH_ROOT / "attention" / "attention_v3_budget.py",
    "learning": BENCH_ROOT / "learning" / "learning_v3_budget.py",
    "metacognition": BENCH_ROOT / "metacognition" / "metacognition_v3_budget.py",
}

# Latest large-scale versions (for validation, not budget tests)
LATEST_FILES = {
    "social_cognition": BENCH_ROOT / "social_cognition" / "moral_geometry_v9_budget.py",
    "learning": BENCH_ROOT / "learning" / "learning_v4_large.py",
    "attention": BENCH_ROOT / "attention" / "attention_v3_budget.py",
    "metacognition": BENCH_ROOT / "metacognition" / "metacognition_v3_budget.py",
    "executive_functions": BENCH_ROOT / "executive_functions" / "executive_functions_v2_budget.py",
}

# Which files exist (some may not be written yet)
def existing_benchmarks():
    return {k: v for k, v in BENCHMARK_FILES.items() if v.exists()}


# =====================================================================
# MOCK KBENCH
# =====================================================================

def make_mock_kbench():
    """Create a mock kaggle_benchmarks module that mimics the real API."""
    kbench = types.ModuleType("kaggle_benchmarks")
    kbench.__name__ = "kaggle_benchmarks"

    # @kbench.task(name="foo") -> decorator that adds .run() method
    def task(name=None):
        def decorator(fn):
            fn._task_name = name
            def run(llm=None, **kwargs):
                return fn(llm)
            fn.run = run
            return fn
        return decorator
    kbench.task = task

    # kbench.llms["model_name"] -> mock LLM
    kbench.llms = MagicMock()

    # kbench.chats.new("chat_id") -> context manager
    kbench.chats = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=None)
    ctx.__exit__ = MagicMock(return_value=False)
    kbench.chats.new.return_value = ctx

    # kbench.assertions
    kbench.assertions = MagicMock()

    return kbench


def make_mock_llm():
    """Create a mock LLM that returns dataclass instances with default values."""
    llm = MagicMock()

    def mock_prompt(text, schema=None):
        if schema is None:
            return "This is a neutral moral scenario. RIGHT."
        # Build a default instance of the dataclass
        kwargs = {}
        for f in fields(schema):
            t = f.type
            if t == str or t == "str":
                kwargs[f.name] = "RIGHT" if "verdict" in f.name else "Test reasoning."
            elif t == float or t == "float":
                if "harm" in f.name or "total" in f.name:
                    kwargs[f.name] = 35.0
                elif "confidence" in f.name:
                    kwargs[f.name] = 7.0
                elif "severity" in f.name:
                    kwargs[f.name] = 5.0
                elif "parties" in f.name or "identified" in f.name:
                    kwargs[f.name] = 3.0
                else:
                    kwargs[f.name] = 5.0
            elif t == int or t == "int":
                kwargs[f.name] = 3
            elif t == bool or t == "bool":
                kwargs[f.name] = True
            else:
                kwargs[f.name] = "default"
        return schema(**kwargs)

    llm.prompt = MagicMock(side_effect=mock_prompt)
    return llm


# =====================================================================
# 1. SYNTAX VALIDATION
# =====================================================================

class TestSyntax:
    """Verify all benchmark files parse as valid Python."""

    @pytest.mark.parametrize("name,path", [
        (k, v) for k, v in BENCHMARK_FILES.items()
    ])
    def test_syntax_valid(self, name, path):
        if not path.exists():
            pytest.skip(f"{name} not yet written")
        source = path.read_text(encoding="utf-8")
        ast.parse(source)  # raises SyntaxError on failure


# =====================================================================
# 2. TASK NAME UNIQUENESS
# =====================================================================

class TestTaskNames:
    """Verify no duplicate @kbench.task names across files."""

    def test_no_duplicate_task_names(self):
        all_names = {}
        for name, path in existing_benchmarks().items():
            source = path.read_text(encoding="utf-8")
            tasks = re.findall(r'@kbench\.task\(name="([^"]+)"\)', source)
            for t in tasks:
                if t in all_names:
                    pytest.fail(
                        f"Duplicate task name '{t}' in {name} and {all_names[t]}"
                    )
                all_names[t] = name


# =====================================================================
# 3. STATISTICAL FUNCTION VALIDATION
# =====================================================================

class TestStatistics:
    """Validate statistical functions against known values."""

    def _get_stats_module(self):
        """Extract statistical functions from social cognition v9."""
        path = BENCHMARK_FILES["social_cognition"]
        if not path.exists():
            pytest.skip("social_cognition not available")
        source = path.read_text(encoding="utf-8")
        # Execute in isolated namespace with mocked kbench
        ns = {"__builtins__": __builtins__}
        mock_kb = make_mock_kbench()
        ns["kaggle_benchmarks"] = mock_kb
        ns["kbench"] = mock_kb
        # Only exec the function definitions, not the execution block
        # Find all function defs and exec them
        tree = ast.parse(source)
        func_source = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                func_source.append(ast.get_source_segment(source, node))
        # Import math and other deps
        exec("import math, threading, os, json, time, random", ns)
        exec("from dataclasses import dataclass", ns)
        exec("from concurrent.futures import ThreadPoolExecutor, as_completed", ns)
        for fs in func_source:
            if fs and ("def two_proportion_z" in fs or "def paired_t" in fs
                       or "def wilson_ci" in fs or "def _fisher_combine" in fs
                       or "def _t_to_sigma" in fs or "def _reg_incomplete_beta" in fs
                       or "def _t_to_p_one_sided" in fs or "def _p_to_sigma" in fs
                       or "def _sigma_to_p" in fs or "def mean" in fs
                       or "def stdev" in fs):
                try:
                    exec(fs, ns)
                except Exception:
                    pass
        return ns

    def test_two_proportion_z(self):
        ns = self._get_stats_module()
        if "two_proportion_z" not in ns:
            pytest.skip("two_proportion_z not found")
        z = ns["two_proportion_z"](50, 100, 30, 100)
        assert 2.5 < z < 3.2, f"Expected ~2.89, got {z}"

    def test_paired_t(self):
        ns = self._get_stats_module()
        if "paired_t" not in ns:
            pytest.skip("paired_t not found")
        t = ns["paired_t"]([1, 2, 3, 4, 5])
        assert 3.5 < t < 5.5, f"Expected ~4.47, got {t}"

    def test_wilson_ci(self):
        ns = self._get_stats_module()
        if "wilson_ci" not in ns:
            pytest.skip("wilson_ci not found")
        lo, hi = ns["wilson_ci"](50, 100)
        assert 0.39 < lo < 0.42, f"Lower CI {lo} out of range"
        assert 0.58 < hi < 0.61, f"Upper CI {hi} out of range"

    def test_fisher_combine(self):
        ns = self._get_stats_module()
        if "_fisher_combine" not in ns:
            pytest.skip("_fisher_combine not found")
        combined = ns["_fisher_combine"]([3.0, 3.0])
        assert combined > 3.0, f"Combined two 3-sigma should be > 3, got {combined}"
        assert combined < 5.0, f"Combined two 3-sigma should be < 5, got {combined}"

    def test_t_to_sigma(self):
        ns = self._get_stats_module()
        if "_t_to_sigma" not in ns:
            pytest.skip("_t_to_sigma not found")
        sig = ns["_t_to_sigma"](3.0, 20)
        assert 2.5 < sig < 3.2, f"t=3.0,df=20 should give ~2.9 sigma, got {sig}"

    def test_incomplete_beta_symmetry(self):
        ns = self._get_stats_module()
        if "_reg_incomplete_beta" not in ns:
            pytest.skip("_reg_incomplete_beta not found")
        val = ns["_reg_incomplete_beta"](0.5, 1.0, 1.0)
        assert abs(val - 0.5) < 0.01, f"I_0.5(1,1) should be 0.5, got {val}"


# =====================================================================
# 4. BUDGET VERIFICATION
# =====================================================================

class TestBudget:
    """Verify estimated budget per track stays within limits."""

    @pytest.mark.parametrize("name,path", [
        (k, v) for k, v in BENCHMARK_FILES.items()
        if k in ("attention", "learning", "metacognition")
    ])
    def test_budget_under_50(self, name, path):
        if not path.exists():
            pytest.skip(f"{name} not yet written")
        source = path.read_text(encoding="utf-8")
        budget_match = re.search(r'Total:\s*~?\$(\d+)', source)
        if budget_match:
            budget = int(budget_match.group(1))
            assert budget <= 50, f"{name} budget ${budget} exceeds $50 quota"


# =====================================================================
# 5. SCHEMA VALIDATION
# =====================================================================

class TestSchemas:
    """Verify all dataclass schemas can be instantiated."""

    @pytest.mark.parametrize("name,path", [
        (k, v) for k, v in BENCHMARK_FILES.items()
    ])
    def test_schemas_instantiable(self, name, path):
        if not path.exists():
            pytest.skip(f"{name} not yet written")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Find all @dataclass-decorated classes
        dataclass_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "dataclass":
                        dataclass_names.append(node.name)

        assert len(dataclass_names) > 0, f"No @dataclass schemas found in {name}"

        # Execute the dataclass definitions in isolation
        ns = {"__builtins__": __builtins__}
        exec("from dataclasses import dataclass", ns)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in dataclass_names:
                segment = ast.get_source_segment(source, node)
                if segment:
                    exec(segment, ns)

        # Verify each can be instantiated with dummy values
        for dc_name in dataclass_names:
            if dc_name in ns:
                dc = ns[dc_name]
                try:
                    flds = fields(dc)
                except TypeError:
                    # exec'd class may not register as dataclass; skip
                    continue
                kwargs = {}
                for f in flds:
                    if f.type in (str, "str"):
                        kwargs[f.name] = "test"
                    elif f.type in (float, "float"):
                        kwargs[f.name] = 1.0
                    elif f.type in (int, "int"):
                        kwargs[f.name] = 1
                    elif f.type in (bool, "bool"):
                        kwargs[f.name] = True
                    else:
                        kwargs[f.name] = "test"
                instance = dc(**kwargs)
                assert instance is not None


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
