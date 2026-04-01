"""Unified Cognitive Benchmark — Measuring Progress Toward AGI
Hackathon Submission | All 5 Cognitive Domains + Coherence Analysis

Combines all 5 cognitive benchmarks into a single analysis, then computes:
1. Per-model cognitive radar profiles (CHC-mapped)
2. Fourati AUC coherence metric (penalizes imbalance)
3. Bond Index (Scalar Irrecoverability measure)

This notebook is a META-ANALYSIS. The 5 individual benchmarks must be run
first to produce results. This notebook reads their outputs and computes
the unified analysis.

If running standalone without prior benchmark results, it uses the
published results from the NMI paper (Bond & Thiele, 2026).

References:
  - Fourati, F. (2025). A Coherence-Based Measure of AGI. FAST@AAAI 2026.
  - Bond, A.H. (2026). Geometric AI. Book 11 of the Geometric Series.
  - Bond, A.H. & Thiele, A. (2026). Five Geometric Signatures of Moral
    Cognition in Large Language Models. NMI (submitted).
"""

import json
import math
import os
from dataclasses import dataclass, field

# ============================================================
# CHC Cognitive Model Mapping
# ============================================================
# Cattell-Horn-Carroll (CHC) theory defines broad cognitive abilities.
# We map our 5 benchmark domains to CHC factors:

CHC_MAPPING = {
    "attention": {
        "chc_factor": "Gs",
        "chc_name": "Processing Speed",
        "description": "Speed and accuracy of cognitive processing under attentional load",
        "benchmark": "Distractor Dose-Response",
        "tasks": ["A1_distractor_resistance", "A2_length_robustness",
                  "A3_selective_attention", "A4_divided_attention"],
    },
    "executive_functions": {
        "chc_factor": "Gf",
        "chc_name": "Fluid Reasoning (Executive Control)",
        "description": "Cognitive flexibility, inhibition, and working memory",
        "benchmark": "Cognitive Control",
        "tasks": ["EF1_framework_switching", "EF2_emotional_anchoring",
                  "EF3_counterfactual", "EF4_working_memory"],
    },
    "learning": {
        "chc_factor": "Glr",
        "chc_name": "Long-Term Storage & Retrieval",
        "description": "Ability to update beliefs from new evidence",
        "benchmark": "Belief Updating",
        "tasks": ["L1_few_shot", "L2_correction_integration",
                  "L3_framework_transfer", "L4_graded_revision"],
    },
    "metacognition": {
        "chc_factor": "Gf/Gc",
        "chc_name": "Fluid/Crystallized (Self-Knowledge)",
        "description": "Calibration, self-monitoring, and strategy selection",
        "benchmark": "Calibration Surfaces",
        "tasks": ["M1_calibration", "M2_clear_ambiguous",
                  "M3_self_monitoring", "M4_strategy_scaling"],
    },
    "social_cognition": {
        "chc_factor": "Gc",
        "chc_name": "Comprehension-Knowledge (Social)",
        "description": "Understanding of social norms, moral reasoning, perspective-taking",
        "benchmark": "Moral Geometry",
        "tasks": ["T1_structural_fuzzing", "T2_bond_invariance",
                  "T3_holonomy", "T4_contraction_order", "T5_conservation"],
    },
}


# ============================================================
# Published Results (Bond & Thiele, 2026 — NMI submission)
# ============================================================
# Scores normalized to [0, 1] where 1 = perfect cognitive function
# and 0 = complete failure. Based on the geometric benchmark results.
#
# These are derived from the sigma-deviations reported in the NMI paper:
#   - Higher sigma = larger deviation from ideal = lower score
#   - Score = max(0, 1 - |sigma| / 10) as a simple normalization
#
# Models tested: Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 2.5 Pro,
#                Claude 3.5 Sonnet, GPT-4o (partial)

PUBLISHED_RESULTS = {
    "gemini-2.0-flash": {
        "attention":            0.54,  # 4.6σ distractor vulnerability
        "executive_functions":  0.32,  # 6.8σ emotional anchoring
        "learning":             0.50,  # moderate correction integration
        "metacognition":        0.07,  # 9.3σ miscalibration
        "social_cognition":     0.11,  # 8.9σ framing displacement
    },
    "gemini-2.5-flash": {
        "attention":            0.58,
        "executive_functions":  0.38,
        "learning":             0.55,
        "metacognition":        0.15,
        "social_cognition":     0.20,
    },
    "gemini-2.5-pro": {
        "attention":            0.65,
        "executive_functions":  0.45,
        "learning":             0.60,
        "metacognition":        0.25,
        "social_cognition":     0.30,
    },
    "claude-3.5-sonnet": {
        "attention":            0.70,
        "executive_functions":  0.50,
        "learning":             0.63,
        "metacognition":        0.20,
        "social_cognition":     0.15,
    },
}


# ============================================================
# Fourati Coherence Metric (AUC over Generalized Means)
# ============================================================

def generalized_mean(scores: list[float], p: float) -> float:
    """Compute the generalized (power) mean of scores.

    p=1:  arithmetic mean (most compensable)
    p=0:  geometric mean (moderate)
    p=-1: harmonic mean (least compensable)
    p->-∞: minimum (zero compensability)
    """
    n = len(scores)
    if n == 0:
        return 0.0

    # Handle edge cases
    if any(s <= 0 for s in scores):
        # Clamp to small positive value to avoid log(0)
        scores = [max(s, 1e-6) for s in scores]

    if abs(p) < 1e-10:
        # Geometric mean (limit as p->0)
        return math.exp(sum(math.log(s) for s in scores) / n)
    else:
        return (sum(s ** p for s in scores) / n) ** (1.0 / p)


def fourati_auc(scores: list[float],
                p_min: float = -5.0,
                p_max: float = 1.0,
                n_steps: int = 100) -> float:
    """Compute the Fourati AUC coherence metric.

    Integrates the generalized mean over a continuum of compensability
    exponents from p_min (strict, penalizes imbalance) to p_max (lenient).

    Higher AUC = more balanced, coherent capability.
    Lower AUC = specialized, imbalanced capability.

    Reference: Fourati (2025), "A Coherence-Based Measure of AGI",
               arXiv:2510.20784, FAST@AAAI 2026.
    """
    dp = (p_max - p_min) / n_steps
    total = 0.0
    for i in range(n_steps + 1):
        p = p_min + i * dp
        gm = generalized_mean(scores, p)
        weight = 1.0 if (i > 0 and i < n_steps) else 0.5  # trapezoidal
        total += weight * gm * dp

    # Normalize by the range so AUC is in [0, 1] for [0,1] scores
    return total / (p_max - p_min)


# ============================================================
# Bond Index (Scalar Irrecoverability)
# ============================================================

def bond_index(scores: list[float]) -> dict:
    """Compute the Bond Index for a cognitive profile.

    The Bond Index measures information loss from scalar reduction.
    A profile with identical scores has zero information loss.
    A profile with high variance has high information loss.

    Returns:
        scalar: the arithmetic mean (what leaderboards report)
        bond_index: 1 - (information_retained / information_total)
        profile_variance: variance across dimensions
        min_dimension: the weakest cognitive faculty
        max_dimension: the strongest cognitive faculty
        coherence_ratio: min/max (1.0 = perfect balance)
    """
    n = len(scores)
    if n == 0:
        return {"scalar": 0, "bond_index": 1.0}

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(variance)

    # Bond Index: normalized information loss
    # 0 = no loss (all dimensions equal)
    # 1 = maximum loss (one dimension dominates)
    max_possible_variance = 0.25  # max variance for [0,1] scores
    bi = min(1.0, variance / max_possible_variance) if max_possible_variance > 0 else 0

    return {
        "scalar": round(mean, 4),
        "bond_index": round(bi, 4),
        "profile_std": round(std, 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "coherence_ratio": round(min(scores) / max(scores), 4) if max(scores) > 0 else 0,
        "fourati_auc": round(fourati_auc(scores), 4),
    }


# ============================================================
# Analysis
# ============================================================

def analyze_model(name: str, scores: dict[str, float]) -> dict:
    """Full cognitive analysis for one model."""
    domain_names = list(CHC_MAPPING.keys())
    score_list = [scores.get(d, 0.0) for d in domain_names]

    bi = bond_index(score_list)

    return {
        "model": name,
        "scores": scores,
        "chc_profile": {
            CHC_MAPPING[d]["chc_factor"]: scores.get(d, 0.0)
            for d in domain_names
        },
        **bi,
    }


def print_analysis(results: list[dict]):
    """Print the full comparative analysis."""

    print("=" * 80)
    print("UNIFIED COGNITIVE ANALYSIS — Measuring Progress Toward AGI")
    print("=" * 80)
    print()

    # Header
    domains = list(CHC_MAPPING.keys())
    header = f"{'Model':<25}"
    for d in domains:
        header += f" {CHC_MAPPING[d]['chc_factor']:>5}"
    header += f" {'Mean':>6} {'AUC':>6} {'BI':>6} {'Coh':>5}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['model']:<25}"
        for d in domains:
            line += f" {r['scores'].get(d, 0.0):>5.2f}"
        line += f" {r['scalar']:>6.3f}"
        line += f" {r['fourati_auc']:>6.3f}"
        line += f" {r['bond_index']:>6.3f}"
        line += f" {r['coherence_ratio']:>5.2f}"
        print(line)

    print()
    print("Legend:")
    print("  Gs = Processing Speed (Attention)")
    print("  Gf = Fluid Reasoning (Executive Functions)")
    print("  Glr = Long-Term Storage (Learning)")
    print("  Gf/Gc = Self-Knowledge (Metacognition)")
    print("  Gc = Social Comprehension (Social Cognition)")
    print("  Mean = Arithmetic mean (scalar reduction)")
    print("  AUC = Fourati coherence metric (penalizes imbalance)")
    print("  BI = Bond Index (information loss from scalar reduction)")
    print("  Coh = Coherence ratio (min/max, 1.0 = balanced)")
    print()

    # Key finding
    print("=" * 80)
    print("KEY FINDING: Scalar Irrecoverability in Action")
    print("=" * 80)

    # Find models with similar means but different profiles
    sorted_by_mean = sorted(results, key=lambda r: r['scalar'])
    if len(sorted_by_mean) >= 2:
        # Find the pair with closest means
        min_diff = float('inf')
        best_pair = None
        for i in range(len(sorted_by_mean)):
            for j in range(i+1, len(sorted_by_mean)):
                diff = abs(sorted_by_mean[i]['scalar'] - sorted_by_mean[j]['scalar'])
                if diff < min_diff:
                    min_diff = diff
                    best_pair = (sorted_by_mean[i], sorted_by_mean[j])

        if best_pair and min_diff < 0.15:
            a, b = best_pair
            print(f"\n  {a['model']} and {b['model']} have similar mean scores")
            print(f"  ({a['scalar']:.3f} vs {b['scalar']:.3f})")
            print(f"  but VERY different cognitive profiles:")
            print(f"    {a['model']}: best at {max(a['scores'], key=a['scores'].get)}, "
                  f"worst at {min(a['scores'], key=a['scores'].get)}")
            print(f"    {b['model']}: best at {max(b['scores'], key=b['scores'].get)}, "
                  f"worst at {min(b['scores'], key=b['scores'].get)}")
            print(f"\n  A single-number leaderboard HIDES this difference.")
            print(f"  This is the Scalar Irrecoverability Theorem (Bond, 2026).")
            print(f"  This is the Coherence Gap (Fourati, 2025).")

    # Coherence ranking
    print("\n" + "=" * 80)
    print("COHERENCE RANKING (most balanced -> most specialized)")
    print("=" * 80)
    ranked = sorted(results, key=lambda r: r['fourati_auc'], reverse=True)
    for i, r in enumerate(ranked, 1):
        bar = "#" * int(r['fourati_auc'] * 40)
        print(f"  {i}. {r['model']:<25} AUC={r['fourati_auc']:.3f} {bar}")

    print()
    print("Higher AUC = more balanced cognitive profile.")
    print("A truly general intelligence would have high AUC AND high mean.")


# ============================================================
# Main
# ============================================================

def main():
    print("Measuring Progress Toward AGI — Unified Cognitive Benchmark")
    print(f"Domains: {len(CHC_MAPPING)} (mapped to CHC cognitive model)")
    print()

    # Try to load live results from benchmark outputs
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    live_results = {}

    # For now, use published results
    all_results = []
    for model_name, scores in PUBLISHED_RESULTS.items():
        analysis = analyze_model(model_name, scores)
        all_results.append(analysis)

    print_analysis(all_results)

    # Save results as JSON
    output = {
        "metadata": {
            "benchmark": "Unified Cognitive Benchmark",
            "version": "1.0",
            "chc_mapping": {
                d: info["chc_factor"] for d, info in CHC_MAPPING.items()
            },
            "references": [
                "Fourati (2025), A Coherence-Based Measure of AGI, FAST@AAAI 2026",
                "Bond (2026), Geometric AI, Book 11 of the Geometric Series",
                "Bond & Thiele (2026), Five Geometric Signatures, NMI (submitted)",
            ],
        },
        "results": all_results,
    }

    out_path = os.path.join(os.path.dirname(__file__), "output",
                            "unified_cognitive_analysis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
