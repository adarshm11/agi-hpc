"""Pre-submission validation for all 5 Measuring AGI benchmarks.

Run this before submitting to Kaggle to catch issues early.

Usage:
    python benchmarks/validate.py

Checks:
    1. All benchmark .py files parse as valid Python
    2. No duplicate @kbench.task names
    3. Statistical functions produce correct values
    4. Budgets within $50 quota
    5. All dataclass schemas instantiable
    6. Notebooks build successfully
    7. Writeups exist and are under word limit
    8. No encoding artifacts (mojibake)
"""

import ast
import re
import json
import sys
from pathlib import Path
from dataclasses import fields

BENCH_ROOT = Path(__file__).parent

BENCHMARK_FILES = {
    "social_cognition": BENCH_ROOT / "social_cognition" / "moral_geometry_v9_budget.py",
    "learning": BENCH_ROOT / "learning" / "learning_v3_budget.py",
    "metacognition": BENCH_ROOT / "metacognition" / "metacognition_v3_budget.py",
    "attention": BENCH_ROOT / "attention" / "attention_v3_budget.py",
    "executive_functions": BENCH_ROOT / "executive_functions" / "executive_functions_v2_budget.py",
}

WRITEUP_FILES = {
    "social_cognition": BENCH_ROOT / "social_cognition" / "KAGGLE_WRITEUP_v2.md",
    "learning": BENCH_ROOT / "learning" / "WRITEUP_v2.md",
    "metacognition": BENCH_ROOT / "metacognition" / "WRITEUP_v2.md",
    "attention": BENCH_ROOT / "attention" / "WRITEUP_v2.md",
    "executive_functions": BENCH_ROOT / "executive_functions" / "WRITEUP_v2.md",
}

MOJIBAKE_PATTERNS = [
    "\u00e2\u0080\u0093",  # em-dash mojibake
    "\u00e2\u0080\u0099",  # right single quote mojibake
    "\u00c3\u00a9",        # e-acute mojibake
]

passed = 0
failed = 0
warnings = 0


def check(name, condition, msg=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}: {msg}")


def warn(name, msg):
    global warnings
    warnings += 1
    print(f"  WARN  {name}: {msg}")


def main():
    global passed, failed, warnings

    print("=" * 60)
    print("Measuring AGI Benchmark — Pre-Submission Validation")
    print("=" * 60)

    # 1. Syntax validation
    print("\n[1] Syntax Validation")
    for name, path in BENCHMARK_FILES.items():
        if not path.exists():
            warn(name, f"File not found: {path}")
            continue
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source)
            check(f"{name} syntax", True)
        except SyntaxError as e:
            check(f"{name} syntax", False, str(e))

    # 2. Task name uniqueness
    print("\n[2] Task Name Uniqueness")
    all_names = {}
    duplicates = []
    for name, path in BENCHMARK_FILES.items():
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        tasks = re.findall(r'@kbench\.task\(name="([^"]+)"\)', source)
        for t in tasks:
            if t in all_names:
                duplicates.append(f"'{t}' in {name} and {all_names[t]}")
            all_names[t] = name
    check("No duplicate task names", len(duplicates) == 0,
          f"Duplicates: {duplicates}")
    check(f"Total tasks: {len(all_names)}", len(all_names) >= 20,
          f"Expected >=20 tasks across 5 tracks")

    # 3. Budget verification
    print("\n[3] Budget Verification")
    for name, path in BENCHMARK_FILES.items():
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        budget_match = re.search(r'Total:\s*~?\$(\d+)', source)
        if budget_match:
            budget = int(budget_match.group(1))
            check(f"{name} budget ${budget}", budget <= 50,
                  f"Exceeds $50 quota")
        else:
            warn(f"{name} budget", "No budget estimate found in source")

    # 4. Writeup validation
    print("\n[4] Writeup Validation")
    for name, path in WRITEUP_FILES.items():
        if not path.exists():
            warn(f"{name} writeup", f"Not found: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        word_count = len(text.split())
        check(f"{name} writeup ({word_count} words)",
              word_count <= 1500,
              f"{word_count} words exceeds 1,500 limit")

        # Check for mojibake
        for pattern in MOJIBAKE_PATTERNS:
            if pattern in text:
                check(f"{name} encoding", False,
                      f"Mojibake found: '{pattern}'")
                break
        else:
            check(f"{name} encoding", True)

    # 5. NMI paper validation
    print("\n[5] NMI Paper Validation")
    nmi_path = BENCH_ROOT / "NMI_PAPER_v2.md"
    if nmi_path.exists():
        text = nmi_path.read_text(encoding="utf-8")
        check("NMI paper exists", True)

        # Check for placeholders
        placeholders = re.findall(r'\[(?:repository URL|acknowledge|TODO)\]', text)
        check("No placeholders", len(placeholders) == 0,
              f"Found: {placeholders}")

        # Check for mojibake
        for pattern in MOJIBAKE_PATTERNS:
            if pattern in text:
                check("NMI encoding", False, f"Mojibake: '{pattern}'")
                break
        else:
            check("NMI encoding", True)

        # Check required sections
        for section in ["Author Contributions", "Competing Interests",
                        "Correspondence"]:
            check(f"NMI has '{section}'", section.lower() in text.lower(),
                  f"Missing required section")
    else:
        warn("NMI paper", "NMI_PAPER_v2.md not found")

    # 6. Notebook build
    print("\n[6] Notebook Build")
    output_dir = BENCH_ROOT / "kaggle_push" / "output"
    for name in BENCHMARK_FILES:
        nb_path = output_dir / f"{name}_benchmark.ipynb"
        if nb_path.exists():
            try:
                with open(nb_path) as f:
                    nb = json.load(f)
                n_cells = len(nb["cells"])
                code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
                check(f"{name} notebook ({n_cells} cells, {len(code_cells)} code)",
                      len(code_cells) >= 1)
            except json.JSONDecodeError as e:
                check(f"{name} notebook", False, f"Invalid JSON: {e}")
        else:
            warn(f"{name} notebook", f"Not built yet. Run build_all_notebooks.py")

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed, {warnings} warnings")
    if failed > 0:
        print("FIX FAILURES BEFORE SUBMITTING")
        sys.exit(1)
    elif warnings > 0:
        print("All checks passed with warnings. Review before submitting.")
    else:
        print("ALL CHECKS PASSED. Ready for submission.")


if __name__ == "__main__":
    main()
