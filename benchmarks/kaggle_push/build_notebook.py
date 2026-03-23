"""Build Kaggle benchmark notebook from the moral geometry benchmark code."""

import json
from pathlib import Path

def code(text):
    lines = text.strip().split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        "outputs": [],
        "execution_count": None,
    }

def md(text):
    lines = text.strip().split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    }

cells = []

cells.append(md("""# Moral Geometry Benchmark — Social Cognition Track

**Measuring AGI: Cognition and Values**

Tests whether LLMs exhibit geometric structure in moral reasoning:
1. **BIP Invariance**: Do equivalent re-descriptions get the same moral judgment?
2. **Multi-Attribute Reasoning**: Can models identify competing moral dimensions in dilemmas?
3. **Harm Conservation**: Does euphemistic language reduce perceived harm?

Based on Bond (2026), *Geometric Ethics: The Mathematical Structure of Moral Reasoning*.

**Andrew H. Bond** — San José State University
"""))

# Read the benchmark code
benchmark_code = Path(r"C:\source\agi-hpc\benchmarks\social_cognition\moral_geometry_benchmark.py").read_text(encoding="utf-8")

# Split into logical sections
sections = benchmark_code.split("# ═══")

# Setup cell
cells.append(code("""import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass
import os

# Speed up batch evaluations
os.environ["RENDER_SUBRUNS"] = "False"

print(f"kaggle_benchmarks loaded")
print(f"Available LLMs: check kbench.llms for full list")
"""))

# Inject benchmark code as cells (skip the module docstring/imports, already handled)
# Find the first @dataclass or TASK section
code_start = benchmark_code.find("BIP_SCENARIOS")
if code_start > 0:
    remaining = benchmark_code[code_start:]

    # Split into chunks at each TASK header
    chunks = remaining.split("\n# ═══════════════════════════════════════════════════════════════════════")

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Remove leading comment lines that are just separators
        lines = chunk.split("\n")
        cleaned = []
        for line in lines:
            if line.strip().startswith("# ═══"):
                continue
            cleaned.append(line)
        chunk = "\n".join(cleaned).strip()
        if chunk:
            cells.append(code(chunk))

# Run cell
cells.append(md("## Run the Benchmark"))

cells.append(code("""# Run the full benchmark on Gemini 2.5 Flash
print("Running Moral Geometry Benchmark...")
print("This tests 19 scenarios across 3 sub-tasks.")
print()

run = moral_geometry_benchmark.run(llm=kbench.llms["google/gemini-2.5-flash"])

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
result = run.result
print(f"BIP Invariance: {result['bip_invariance']['accuracy']:.0%} ({result['bip_invariance']['n_passed']}/{result['bip_invariance']['total']})")
print(f"Multi-Attribute: {result['multi_attribute']['avg_dimensions_identified']:.1f} avg dimensions identified")
print(f"  Tradeoff acknowledgment: {result['multi_attribute']['tradeoff_acknowledgment_rate']:.0%}")
print(f"Harm Conservation: {result['harm_conservation']['conservation_rate']:.0%}")
print(f"  Avg severity drop from euphemism: {result['harm_conservation']['avg_severity_drop']:.2f}")
print(f"")
print(f"COMPOSITE SCORE: {result['composite_score']:.1%}")
"""))

# Also test on a second model for discriminatory power
cells.append(md("## Compare Models (Discriminatory Power)"))

cells.append(code("""# Run on a second model to show discriminatory power
print("Running on Gemini 2.5 Pro for comparison...")
run_pro = moral_geometry_benchmark.run(llm=kbench.llms["google/gemini-2.5-pro"])

print()
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
r1 = run.result
r2 = run_pro.result
print(f"{'Metric':<35} {'Flash':>10} {'Pro':>10}")
print("-" * 55)
print(f"{'BIP Invariance':<35} {r1['bip_invariance']['accuracy']:>9.0%} {r2['bip_invariance']['accuracy']:>9.0%}")
print(f"{'Dimensions identified (avg)':<35} {r1['multi_attribute']['avg_dimensions_identified']:>10.1f} {r2['multi_attribute']['avg_dimensions_identified']:>10.1f}")
print(f"{'Tradeoff acknowledgment':<35} {r1['multi_attribute']['tradeoff_acknowledgment_rate']:>9.0%} {r2['multi_attribute']['tradeoff_acknowledgment_rate']:>9.0%}")
print(f"{'Harm conservation':<35} {r1['harm_conservation']['conservation_rate']:>9.0%} {r2['harm_conservation']['conservation_rate']:>9.0%}")
print(f"{'COMPOSITE':<35} {r1['composite_score']:>9.1%} {r2['composite_score']:>9.1%}")
"""))

# Choose task for submission
cells.append(code("""%choose moral_geometry_benchmark"""))

# Assemble notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path(r"C:\source\agi-hpc\benchmarks\kaggle_push\moral_geometry_benchmark.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook: {out}")
print(f"Cells: {len(cells)}")
