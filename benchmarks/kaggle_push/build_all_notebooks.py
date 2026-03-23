"""Build Kaggle benchmark notebooks for all 5 cognitive tracks.

Usage:
    python benchmarks/kaggle_push/build_all_notebooks.py

Produces one .ipynb per track in benchmarks/kaggle_push/output/
"""

import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).parent.parent

TRACKS = {
    "social_cognition": {
        "source": BENCH_ROOT / "social_cognition" / "moral_geometry_v9_budget.py",
        "title": "Moral Geometry Benchmark — Social Cognition",
        "description": (
            "Five geometric tests of LLM moral judgment: structural fuzzing, "
            "invariance (Bond Invariance Principle), holonomy, order sensitivity, "
            "and framing sensitivity. Headline: 8.9σ framing displacement."
        ),
    },
    "learning": {
        "source": BENCH_ROOT / "learning" / "learning_v3_budget.py",
        "title": "Geometric Learning — Belief Updating in Moral Cognition",
        "description": (
            "Four tests of belief updating: few-shot learning, correction "
            "integration with sycophancy detection (13.3σ), framework transfer, "
            "and graded belief revision."
        ),
    },
    "metacognition": {
        "source": BENCH_ROOT / "metacognition" / "metacognition_v3_budget.py",
        "title": "Geometric Metacognition — Calibration Surfaces",
        "description": (
            "Four tests of self-knowledge: calibration (9.3σ miscalibration), "
            "clear/ambiguous discrimination, self-monitoring, and strategy scaling."
        ),
    },
    "attention": {
        "source": BENCH_ROOT / "attention" / "attention_v3_budget.py",
        "title": "Geometric Attention — Distractor Dose-Response",
        "description": (
            "Four tests of attentional filtering: graded distractor resistance "
            "(4.6σ), length robustness, selective attention SNR, divided attention."
        ),
    },
    "executive_functions": {
        "source": BENCH_ROOT / "executive_functions" / "executive_functions_v2_budget.py",
        "title": "Geometric Executive Functions — Cognitive Control",
        "description": (
            "Four tests of cognitive control: framework switching, emotional "
            "anchoring resistance (6.8σ), counterfactual reasoning, working memory."
        ),
    },
}


def make_cell(cell_type, source_text):
    """Create a notebook cell."""
    lines = source_text.strip().split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def build_notebook(track_name, track_info):
    """Build a single track's notebook."""
    source_path = track_info["source"]
    if not source_path.exists():
        print(f"  SKIP {track_name}: {source_path} not found")
        return None

    benchmark_code = source_path.read_text(encoding="utf-8")

    cells = [
        make_cell("markdown", f"""# {track_info['title']}

**Kaggle Measuring AGI: Cognition and Values**

{track_info['description']}

Based on Bond (2026), *Geometric Methods in Computational Modeling: From Manifolds to Production Systems.*

**Andrew H. Bond** — San Jose State University, Department of Computer Engineering
"""),
        make_cell("code", benchmark_code),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }
    return notebook


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    results = []
    for track_name, track_info in TRACKS.items():
        print(f"Building {track_name}...")
        notebook = build_notebook(track_name, track_info)
        if notebook is None:
            results.append((track_name, "SKIP"))
            continue

        out_path = output_dir / f"{track_name}_benchmark.ipynb"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        code_size = sum(
            len("".join(c["source"]))
            for c in notebook["cells"]
            if c["cell_type"] == "code"
        )
        results.append((track_name, f"OK ({code_size:,} chars)"))
        print(f"  -> {out_path}")

    print("\n=== Build Summary ===")
    for name, status in results:
        print(f"  {name:25s} {status}")


if __name__ == "__main__":
    main()
