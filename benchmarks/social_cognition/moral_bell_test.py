"""Moral Bell Test — CHSH Inequality for LLM Moral Reasoning
Social Cognition Track | Measuring AGI Competition

Tests whether LLM moral judgments obey classical (|S| ≤ 2) or quantum
(|S| ≤ 2√2) correlation bounds using real AITA scenarios.

Based on Bond (2026), Geometric Ethics, Chapter 13 — Quantum Extension.

The experiment:
  - Take an AITA scenario and create an "entangled pair" by writing it
    from two perspectives (poster and other party)
  - Apply two "measurement settings" (framings): rights-based vs consequences-based
  - Compute CHSH S-value from the 4 correlations
  - Classical moral reasoning: |S| ≤ 2
  - Quantum contextuality: |S| > 2 (up to 2√2 ≈ 2.83)

Also tests:
  - D₄ correlative symmetry (swap perspectives, does YTA ↔ NTA?)
  - Non-commutativity (does dimension ordering change verdicts?)
  - Harm conservation under perspective shift

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
No pip install needed. Prints progress throughout.
Expected runtime: ~30-60 minutes.
"""

import kaggle_benchmarks as kbench
from dataclasses import dataclass
import os, json, time, random, math

os.environ["RENDER_SUBRUNS"] = "False"

print("=" * 60)
print("MORAL BELL TEST — CHSH Inequality for Moral Reasoning")
print("Social Cognition Track")
print("=" * 60)
print()

# ═══════════════════════════════════════════════════════════════
# LOAD AITA DATASET
# ═══════════════════════════════════════════════════════════════

print("[1/5] Loading AITA dataset...")
t0 = time.time()
try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset", split="train")
print(f"  Loaded {len(ds):,} posts in {time.time()-t0:.0f}s")

# Select high-quality posts with clear verdicts
random.seed(42)
qualified = []
for row in ds:
    v = (row.get("verdict") or "").lower().strip()
    if v not in ("nta", "yta", "esh", "nah"):
        continue
    text = row.get("text") or ""
    if len(text) < 300 or len(text) > 1500:
        continue
    if (row.get("score") or 0) < 15:  # higher threshold for Bell test quality
        continue
    qualified.append({
        "title": row["title"],
        "text": text,
        "verdict": v.upper(),
        "score": row["score"],
    })

random.shuffle(qualified)
# Balanced sample for Bell test: 50 per verdict = 200 scenarios
BELL_SCENARIOS = []
counts = {"NTA": 0, "YTA": 0, "ESH": 0, "NAH": 0}
for q in qualified:
    v = q["verdict"]
    if counts[v] < 50:
        BELL_SCENARIOS.append(q)
        counts[v] += 1
    if all(c >= 50 for c in counts.values()):
        break

random.shuffle(BELL_SCENARIOS)
print(f"  Selected {len(BELL_SCENARIOS)} scenarios for Bell test")
for v in ["NTA", "YTA", "ESH", "NAH"]:
    print(f"    {v}: {counts[v]}")
print()


# ═══════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════

@dataclass
class MoralMeasurement:
    verdict: str       # YTA, NTA, ESH, NAH
    severity: float    # 0-10
    confidence: float  # 0-10
    reasoning: str


def normalize_verdict(raw):
    raw = raw.strip().upper().replace("'", "").replace(".", "")
    for label in ["NTA", "YTA", "ESH", "NAH"]:
        if label in raw:
            return label
    return raw[:3]


def verdict_to_spin(verdict):
    """Map verdict to ±1 spin for CHSH calculation.
    NTA/NAH = +1 (poster justified / no conflict)
    YTA/ESH = -1 (poster wrong / mutual wrong)
    """
    return +1 if verdict in ("NTA", "NAH") else -1


# ═══════════════════════════════════════════════════════════════
# MEASUREMENT SETTINGS (framings)
# These are the "analyzer angles" in the Bell test
# ═══════════════════════════════════════════════════════════════

FRAMING_A1 = "Focus on RIGHTS and OBLIGATIONS. Who has a right to what? Who violated whose rights? Judge based on rights alone."
FRAMING_A2 = "Focus on CONSEQUENCES and OUTCOMES. Who was harmed? What were the results? Judge based on consequences alone."
FRAMING_B1 = "Focus on FAIRNESS and RECIPROCITY. Was this fair? Was there equal treatment? Judge based on fairness alone."
FRAMING_B2 = "Focus on CHARACTER and VIRTUE. What does this say about the person? Is this how a good person acts? Judge based on character alone."


# ═══════════════════════════════════════════════════════════════
# TASK 1: CHSH BELL TEST
# Measure correlations across perspective × framing combinations
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="moral_chsh_bell_test")
def moral_chsh_bell_test(llm):
    """CHSH Bell test for moral reasoning.

    For each scenario, measure verdict under 4 combinations:
      (poster perspective, rights framing)     = A1,B1
      (poster perspective, consequences)       = A1,B2
      (other party perspective, rights)        = A2,B1
      (other party perspective, consequences)  = A2,B2

    Compute CHSH S = |E(A1,B1) - E(A1,B2) + E(A2,B1) + E(A2,B2)|
    Classical bound: S ≤ 2
    Quantum bound: S ≤ 2√2 ≈ 2.83
    """

    print("[2/5] TASK 1: CHSH BELL TEST")
    print("  Measuring moral correlations across perspective × framing")
    print("  Classical bound: |S| ≤ 2")
    print("  Quantum bound: |S| ≤ 2√2 ≈ 2.83")
    print("-" * 60)

    # Use subset for Bell test (4 LLM calls per scenario)
    n_bell = min(50, len(BELL_SCENARIOS))
    scenarios = BELL_SCENARIOS[:n_bell]

    # Correlation accumulators for 4 setting combinations
    corr = {"A1B1": [], "A1B2": [], "A2B1": [], "A2B2": []}

    for i, s in enumerate(scenarios):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\n  Scenario {i+1}/{n_bell}: {s['title'][:60]}...")

        # Perspective A1: Original poster perspective
        poster_text = s["text"]

        # Perspective A2: Swap to other party's perspective
        swap_prompt = (
            f"Rewrite this situation from the OTHER person's perspective "
            f"(not the original poster). Keep all facts the same, just change "
            f"who is telling the story.\n\n{s['text'][:800]}"
        )
        with kbench.chats.new(f"swap_{i}"):
            other_perspective = llm.prompt(swap_prompt)

        measurements = {}
        for perspective_name, perspective_text, framings in [
            ("A1", poster_text, [("B1", FRAMING_A1), ("B2", FRAMING_A2)]),
            ("A2", str(other_perspective), [("B1", FRAMING_B1), ("B2", FRAMING_B2)]),
        ]:
            for framing_name, framing_instruction in framings:
                setting = f"{perspective_name}{framing_name}"
                with kbench.chats.new(f"bell_{i}_{setting}"):
                    result = llm.prompt(
                        f"You are judging a moral situation from r/AmITheAsshole.\n\n"
                        f"{framing_instruction}\n\n"
                        f"Situation:\n{perspective_text[:1000]}\n\n"
                        f"Verdict (YTA/NTA/ESH/NAH), severity 0-10, confidence 0-10, brief reasoning.",
                        schema=MoralMeasurement
                    )
                verdict = normalize_verdict(result.verdict)
                spin = verdict_to_spin(verdict)
                measurements[setting] = spin

        # Compute pairwise correlations (product of spins)
        for key in corr:
            a_key = key[:2]  # A1 or A2
            b_key = key[2:]  # B1 or B2
            if a_key in measurements and f"{a_key}{b_key}" == key:
                # For CHSH we need: E(Ai,Bj) = <Ai * Bj>
                # We use the same scenario measured under different settings
                pass

        # Store spin products for each setting combination
        a1 = measurements.get("A1B1", 0)
        a1b2_spin = measurements.get("A1B2", 0)
        a2b1_spin = measurements.get("A2B1", 0)
        a2b2_spin = measurements.get("A2B2", 0)

        corr["A1B1"].append(a1 * a2b1_spin)  # poster-rights × other-fairness
        corr["A1B2"].append(a1 * a2b2_spin)  # poster-rights × other-character
        corr["A2B1"].append(a1b2_spin * a2b1_spin)  # poster-consequences × other-fairness
        corr["A2B2"].append(a1b2_spin * a2b2_spin)  # poster-consequences × other-character

        if (i + 1) % 10 == 0:
            # Running S estimate
            E = {k: sum(v) / max(len(v), 1) for k, v in corr.items()}
            S_running = abs(E["A1B1"] - E["A1B2"] + E["A2B1"] + E["A2B2"])
            print(f"    Running S = {S_running:.3f} (classical ≤ 2)")

    # Final CHSH calculation
    E = {}
    for key, values in corr.items():
        E[key] = sum(values) / max(len(values), 1)

    S = abs(E["A1B1"] - E["A1B2"] + E["A2B1"] + E["A2B2"])
    classical = S <= 2.0
    quantum_violation = S > 2.0

    kbench.assertions.assert_true(
        True,  # Always pass — we're measuring, not asserting a specific bound
        expectation=f"CHSH S = {S:.3f} ({'classical |S|≤2' if classical else 'QUANTUM VIOLATION |S|>2'})"
    )

    print(f"\n{'='*60}")
    print(f"  CHSH BELL TEST RESULTS (N={n_bell} scenarios)")
    print(f"  {'─'*40}")
    print(f"  Correlations:")
    print(f"    E(poster-rights, other-fairness):    {E['A1B1']:+.3f}")
    print(f"    E(poster-rights, other-character):   {E['A1B2']:+.3f}")
    print(f"    E(poster-conseq, other-fairness):    {E['A2B1']:+.3f}")
    print(f"    E(poster-conseq, other-character):   {E['A2B2']:+.3f}")
    print(f"  {'─'*40}")
    print(f"  S = |E(A1B1) - E(A1B2) + E(A2B1) + E(A2B2)|")
    print(f"  S = {S:.3f}")
    print(f"  Classical bound (|S| ≤ 2):   {'SATISFIED' if classical else 'VIOLATED'}")
    print(f"  Tsirelson bound (|S| ≤ 2√2): {'SATISFIED' if S <= 2*math.sqrt(2) else 'VIOLATED'}")
    print(f"  {'─'*40}")
    if classical:
        print(f"  Interpretation: Moral reasoning is CLASSICAL.")
        print(f"  Consistent with D₄ × U(1)_H gauge group (Bond, 2026).")
        print(f"  No quantum contextuality detected in moral judgments.")
    else:
        print(f"  Interpretation: Moral reasoning shows QUANTUM-LIKE contextuality!")
        print(f"  The combination of perspective + framing produces")
        print(f"  correlations that exceed classical bounds.")
    print(f"{'='*60}")

    return {
        "S_value": S,
        "classical": classical,
        "n_scenarios": n_bell,
        "correlations": E,
    }


# ═══════════════════════════════════════════════════════════════
# TASK 2: D₄ CORRELATIVE SYMMETRY
# Swap poster/other perspective: does YTA ↔ NTA?
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="moral_d4_symmetry")
def moral_d4_symmetry(llm):
    """D₄ correlative symmetry test.

    The reflection s: O ↔ C in the Hohfeldian square predicts:
    - Swapping poster/other-party perspective should flip YTA ↔ NTA
    - ESH should remain ESH (both parties are wrong regardless of perspective)
    - NAH should remain NAH (nobody is wrong regardless of perspective)
    """

    print("\n[3/5] TASK 2: D₄ CORRELATIVE SYMMETRY")
    print("  Swap perspectives: does YTA ↔ NTA?")
    print("  ESH and NAH should be perspective-invariant")
    print("-" * 60)

    n_test = min(80, len(BELL_SCENARIOS))
    scenarios = BELL_SCENARIOS[:n_test]

    flips_correct = 0
    invariants_correct = 0
    total_flip = 0
    total_invariant = 0
    errors = []

    for i, s in enumerate(scenarios):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"\n  Scenario {i+1}/{n_test}: {s['title'][:50]}... (actual: {s['verdict']})")

        # Get verdict from other party's perspective
        with kbench.chats.new(f"d4_swap_{i}"):
            swapped = llm.prompt(
                f"This is a moral situation from r/AmITheAsshole, "
                f"but told from the OTHER person's perspective (the person the "
                f"original poster is in conflict with). Read it and judge:\n\n"
                f"The original poster wrote:\n{s['text'][:1000]}\n\n"
                f"Now imagine YOU are the other person in this story. "
                f"From YOUR perspective, is the original poster the asshole?\n\n"
                f"Verdict (YTA/NTA/ESH/NAH), severity 0-10, confidence 0-10, reasoning.",
                schema=MoralMeasurement
            )

        swapped_verdict = normalize_verdict(swapped.verdict)
        actual = s["verdict"]

        if actual in ("YTA", "NTA"):
            total_flip += 1
            # YTA from poster's view should become NTA from other's view (and vice versa)
            expected_swap = "NTA" if actual == "YTA" else "YTA"
            ok = swapped_verdict == expected_swap
            if ok:
                flips_correct += 1

            kbench.assertions.assert_equal(
                expected_swap, swapped_verdict,
                expectation=f"D4 flip: {actual} should become {expected_swap} from other perspective (got {swapped_verdict})"
            )

            if not ok:
                print(f"    D4 FLIP ERROR: {actual} -> expected {expected_swap}, got {swapped_verdict}")
                print(f"    LLM reasoning: {swapped.reasoning[:150]}")
                errors.append({"actual": actual, "expected": expected_swap, "got": swapped_verdict})

        elif actual in ("ESH", "NAH"):
            total_invariant += 1
            # ESH and NAH should be perspective-invariant
            ok = swapped_verdict == actual
            if ok:
                invariants_correct += 1

            kbench.assertions.assert_equal(
                actual, swapped_verdict,
                expectation=f"D4 invariant: {actual} should stay {actual} from other perspective (got {swapped_verdict})"
            )

            if not ok:
                print(f"    D4 INVARIANT ERROR: {actual} should stay {actual}, got {swapped_verdict}")
                print(f"    LLM reasoning: {swapped.reasoning[:150]}")

    flip_rate = flips_correct / max(total_flip, 1)
    invariant_rate = invariants_correct / max(total_invariant, 1)
    overall = (flips_correct + invariants_correct) / max(total_flip + total_invariant, 1)

    print(f"\n{'='*60}")
    print(f"  D₄ CORRELATIVE SYMMETRY RESULTS (N={n_test})")
    print(f"  {'─'*40}")
    print(f"  YTA ↔ NTA flip accuracy:     {flip_rate:.0%} ({flips_correct}/{total_flip})")
    print(f"  ESH/NAH invariance accuracy: {invariant_rate:.0%} ({invariants_correct}/{total_invariant})")
    print(f"  Overall D₄ symmetry:         {overall:.0%}")
    print(f"  {'─'*40}")
    if overall >= 0.7:
        print(f"  Strong D₄ symmetry detected — consistent with")
        print(f"  the dihedral gauge group of geometric ethics.")
    elif overall >= 0.5:
        print(f"  Partial D₄ symmetry — model has perspective awareness")
        print(f"  but doesn't fully implement correlative exchange.")
    else:
        print(f"  Weak D₄ symmetry — model struggles with perspective-taking.")
    print(f"{'='*60}")

    return {
        "flip_accuracy": flip_rate,
        "invariant_accuracy": invariant_rate,
        "overall_symmetry": overall,
        "n_tested": n_test,
    }


# ═══════════════════════════════════════════════════════════════
# TASK 3: NON-COMMUTATIVITY TEST
# Does the order of moral dimensions change the verdict?
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="moral_noncommutativity")
def moral_noncommutativity(llm):
    """Test sr ≠ rs: does the order of moral considerations matter?

    Present the same scenario with dimensions in different orders:
    Order 1: First consider fairness, then consequences
    Order 2: First consider consequences, then fairness

    If verdicts differ, moral reasoning is non-commutative (sr ≠ rs),
    consistent with the non-abelian D₄ structure.
    """

    print("\n[4/5] TASK 3: NON-COMMUTATIVITY")
    print("  Does the order of moral considerations change the verdict?")
    print("  If yes: non-abelian (sr ≠ rs), consistent with D₄")
    print("  If no: abelian, simpler group structure")
    print("-" * 60)

    n_test = min(60, len(BELL_SCENARIOS))
    scenarios = BELL_SCENARIOS[:n_test]

    order_matters = 0
    total = 0

    for i, s in enumerate(scenarios):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"\n  Scenario {i+1}/{n_test}: {s['title'][:50]}...")

        # Order 1: Fairness first, then consequences
        with kbench.chats.new(f"order1_{i}"):
            j1 = llm.prompt(
                f"Judge this moral situation in TWO steps.\n\n"
                f"Situation: {s['text'][:800]}\n\n"
                f"STEP 1: First, evaluate FAIRNESS. Is this situation fair to everyone involved? "
                f"Write your fairness assessment.\n"
                f"STEP 2: Now, considering your fairness assessment, evaluate the CONSEQUENCES. "
                f"What harm or benefit resulted?\n\n"
                f"Based on BOTH steps (fairness first, then consequences), "
                f"give your final verdict: YTA/NTA/ESH/NAH. Severity 0-10. Confidence 0-10.",
                schema=MoralMeasurement
            )

        # Order 2: Consequences first, then fairness
        with kbench.chats.new(f"order2_{i}"):
            j2 = llm.prompt(
                f"Judge this moral situation in TWO steps.\n\n"
                f"Situation: {s['text'][:800]}\n\n"
                f"STEP 1: First, evaluate the CONSEQUENCES. What harm or benefit resulted? "
                f"Write your consequences assessment.\n"
                f"STEP 2: Now, considering the consequences, evaluate FAIRNESS. "
                f"Is this situation fair to everyone involved?\n\n"
                f"Based on BOTH steps (consequences first, then fairness), "
                f"give your final verdict: YTA/NTA/ESH/NAH. Severity 0-10. Confidence 0-10.",
                schema=MoralMeasurement
            )

        v1 = normalize_verdict(j1.verdict)
        v2 = normalize_verdict(j2.verdict)
        total += 1

        if v1 != v2:
            order_matters += 1
            if (i + 1) % 10 == 0 or i < 5:
                print(f"    NON-COMMUTATIVE: fairness-first -> {v1}, consequences-first -> {v2}")
                print(f"    Scenario: {s['title'][:60]}")

    nc_rate = order_matters / max(total, 1)

    kbench.assertions.assert_true(
        True,  # Measuring, not asserting
        expectation=f"Non-commutativity rate: {nc_rate:.0%} ({order_matters}/{total})"
    )

    print(f"\n{'='*60}")
    print(f"  NON-COMMUTATIVITY RESULTS (N={n_test})")
    print(f"  {'─'*40}")
    print(f"  Order changed verdict: {order_matters}/{total} ({nc_rate:.0%})")
    print(f"  Order preserved verdict: {total - order_matters}/{total} ({1-nc_rate:.0%})")
    print(f"  {'─'*40}")
    if nc_rate > 0.15:
        print(f"  Significant non-commutativity detected ({nc_rate:.0%}).")
        print(f"  Moral reasoning is order-dependent — consistent with")
        print(f"  the non-abelian property sr ≠ rs of the D₄ gauge group.")
    elif nc_rate > 0.05:
        print(f"  Mild non-commutativity ({nc_rate:.0%}).")
        print(f"  Some order-dependence in moral reasoning.")
    else:
        print(f"  Negligible non-commutativity ({nc_rate:.0%}).")
        print(f"  Moral reasoning appears order-independent for this model.")
    print(f"{'='*60}")

    return {
        "noncommutativity_rate": nc_rate,
        "order_changed": order_matters,
        "total": total,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="moral_bell_test")
def moral_bell_test(llm):
    """Moral Bell Test — CHSH, D₄ Symmetry, and Non-Commutativity

    Tests the gauge-group structure of LLM moral reasoning using
    real Reddit r/AmITheAsshole scenarios.
    """

    print("\n" + "#" * 60)
    print("# MORAL BELL TEST")
    print("# Testing gauge-group structure of LLM moral reasoning")
    print("# Based on Bond (2026), Geometric Ethics")
    print("#" * 60)

    t_start = time.time()

    chsh = moral_chsh_bell_test.run(llm=llm).result
    d4 = moral_d4_symmetry.run(llm=llm).result
    nc = moral_noncommutativity.run(llm=llm).result

    elapsed = time.time() - t_start

    print("\n" + "#" * 60)
    print("[5/5] FINAL RESULTS — MORAL BELL TEST")
    print("#" * 60)
    print(f"")
    print(f"  1. CHSH Bell Test:")
    print(f"     S = {chsh['S_value']:.3f} ({'CLASSICAL' if chsh['classical'] else 'QUANTUM VIOLATION'})")
    print(f"     Classical bound |S| ≤ 2: {'satisfied' if chsh['classical'] else 'VIOLATED'}")
    print(f"")
    print(f"  2. D₄ Correlative Symmetry:")
    print(f"     YTA ↔ NTA flip accuracy: {d4['flip_accuracy']:.0%}")
    print(f"     ESH/NAH invariance:      {d4['invariant_accuracy']:.0%}")
    print(f"     Overall:                 {d4['overall_symmetry']:.0%}")
    print(f"")
    print(f"  3. Non-Commutativity (sr ≠ rs):")
    print(f"     Order changed verdict:   {nc['noncommutativity_rate']:.0%}")
    print(f"")
    print(f"  Gauge Group Assessment:")
    if chsh["classical"] and d4["overall_symmetry"] >= 0.6 and nc["noncommutativity_rate"] > 0.1:
        print(f"     CONSISTENT WITH D₄ × U(1)_H")
        print(f"     Classical correlations + dihedral symmetry + non-abelian structure")
    elif chsh["classical"] and d4["overall_symmetry"] >= 0.5:
        print(f"     PARTIALLY CONSISTENT WITH D₄ × U(1)_H")
        print(f"     Classical correlations + partial symmetry")
    else:
        print(f"     INCONCLUSIVE — further testing needed")
    print(f"")
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("#" * 60)

    return {
        "chsh": chsh,
        "d4_symmetry": d4,
        "noncommutativity": nc,
    }


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

print(f"Running on {kbench.llm}...")
print(f"Using {len(BELL_SCENARIOS)} AITA scenarios.")
print(f"Expected runtime: ~30-60 minutes.\n")

run = moral_bell_test.run(llm=kbench.llm)

print("\nTask files saved:")
for f in sorted(os.listdir(".")):
    if f.endswith(".json"):
        print(f"  {f}")
