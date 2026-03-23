"""Moral Geometry Benchmark v5 — AITA Edition
Social Cognition Track | Measuring AGI Competition

200 real Reddit r/AmITheAsshole posts with human consensus verdicts.
Stratified: 50 NTA + 50 YTA + 50 ESH + 50 NAH.

One cell. Paste into Kaggle Benchmark Task notebook and run.
No pip install needed.
"""

import kaggle_benchmarks as kbench
from dataclasses import dataclass
import os, json, time, random

os.environ["RENDER_SUBRUNS"] = "False"

print("=" * 60)
print("MORAL GEOMETRY BENCHMARK v5 — AITA Edition")
print("Social Cognition Track | 4000 scenarios")
print("=" * 60)
print()

# ═══════════════════════════════════════════════════════════════
# LOAD DATASET
# ═══════════════════════════════════════════════════════════════

print("[1/4] Loading AITA dataset from HuggingFace...")
t_load = time.time()
try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset", split="train")
print(f"  Loaded {len(ds):,} posts in {time.time()-t_load:.0f}s")

random.seed(42)
PER_CLASS = 1000  # 1000 per class x 4 classes = 4000 total
pools = {"nta": [], "yta": [], "esh": [], "nah": []}

for row in ds:
    v = (row.get("verdict") or "").lower().strip()
    if v not in pools or len(pools[v]) >= PER_CLASS * 4:
        continue
    text = row.get("text") or ""
    score = row.get("score") or 0
    # Lower threshold for rarer classes (ESH/NAH) to get enough samples
    min_score = 5 if v in ("esh", "nah") else 10
    if len(text) < 200 or len(text) > 2000 or score < min_score:
        continue
    pools[v].append({
        "title": row["title"],
        "text": text[:1500],
        "verdict": v.upper(),
        "score": row["score"],
    })

SCENARIOS = []
for v in ["NTA", "YTA", "ESH", "NAH"]:
    pool = pools[v.lower()]
    random.shuffle(pool)
    SCENARIOS.extend(pool[:PER_CLASS])
    print(f"  {v}: {len(pool[:PER_CLASS])} sampled")

random.shuffle(SCENARIOS)
print(f"  Total: {len(SCENARIOS)} scenarios ready\n")


# ═══════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════

@dataclass
class AITAJudgment:
    verdict: str       # YTA, NTA, ESH, or NAH
    confidence: float  # 0-10
    reasoning: str     # explanation of moral reasoning


# ═══════════════════════════════════════════════════════════════
# HELPER: Normalize verdict string
# ═══════════════════════════════════════════════════════════════

def normalize_verdict(raw):
    raw = raw.strip().upper().replace("'", "").replace(".", "")
    for label in ["NTA", "YTA", "ESH", "NAH"]:
        if label in raw:
            return label
    return raw[:3]


# ═══════════════════════════════════════════════════════════════
# TASK 1: AITA VERDICT ACCURACY (4000 scenarios)
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="aita_verdict_accuracy")
def aita_verdict_accuracy(llm):
    """Does the LLM agree with human moral consensus on 200 AITA posts?"""

    print("[2/4] TASK 1: AITA VERDICT ACCURACY")
    print(f"  Testing {len(SCENARIOS)} real Reddit moral dilemmas")
    print(f"  Balanced: 1000 NTA / 1000 YTA / 1000 ESH / 1000 NAH")
    print("-" * 60)

    correct = 0
    by_class = {v: {"correct": 0, "total": 0} for v in ["NTA", "YTA", "ESH", "NAH"]}
    errors = []

    for i, s in enumerate(SCENARIOS):
        with kbench.chats.new(f"aita_{i}"):
            judgment = llm.prompt(
                f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
                f"Title: {s['title']}\n\n"
                f"{s['text']}\n\n"
                f"Give your verdict as EXACTLY one of:\n"
                f"  YTA = You're The Asshole (poster is wrong)\n"
                f"  NTA = Not The Asshole (poster is right)\n"
                f"  ESH = Everyone Sucks Here (both sides are wrong)\n"
                f"  NAH = No Assholes Here (nobody is wrong)\n\n"
                f"Rate confidence 0-10 and explain your reasoning in 1-2 sentences.",
                schema=AITAJudgment
            )

        pred = normalize_verdict(judgment.verdict)
        actual = s["verdict"]
        match = pred == actual
        if match:
            correct += 1
        by_class[actual]["total"] += 1
        if match:
            by_class[actual]["correct"] += 1

        kbench.assertions.assert_equal(actual, pred,
            expectation=f"Verdict should be {actual} (got {pred})")

        # ── INFORMATIVE OUTPUT ──
        n = i + 1
        running_acc = correct / n

        if match:
            # Correct: brief confirmation every 10
            if n % 50 == 0:
                print(f"\n  [{n}/200] Running accuracy: {running_acc:.0%} ({correct}/{n})")
        else:
            # Wrong: show FULL detail so we understand WHY
            print(f"\n  [{n}/200] DISAGREE — predicted {pred}, humans said {actual}")
            print(f"  Title:     {s['title'][:80]}")
            print(f"  Scenario:  {s['text'][:200]}...")
            print(f"  LLM says:  {judgment.reasoning[:200]}")
            print(f"  Conf: {judgment.confidence:.0f}/10 | Human consensus score: {s['score']}")
            errors.append({
                "index": n,
                "title": s["title"][:60],
                "predicted": pred,
                "actual": actual,
                "reasoning": judgment.reasoning[:100],
                "confidence": judgment.confidence,
            })

    accuracy = correct / len(SCENARIOS)
    class_accs = {}
    for v in ["NTA", "YTA", "ESH", "NAH"]:
        c = by_class[v]["correct"]
        t = by_class[v]["total"]
        class_accs[v] = c / max(t, 1)

    print(f"\n{'='*60}")
    print(f"  VERDICT ACCURACY: {correct}/{len(SCENARIOS)} ({accuracy:.1%})")
    print(f"  {'─'*40}")
    print(f"  NTA (poster is right):      {class_accs['NTA']:.0%}  ({by_class['NTA']['correct']}/{by_class['NTA']['total']})")
    print(f"  YTA (poster is wrong):      {class_accs['YTA']:.0%}  ({by_class['YTA']['correct']}/{by_class['YTA']['total']})")
    print(f"  ESH (everyone wrong):       {class_accs['ESH']:.0%}  ({by_class['ESH']['correct']}/{by_class['ESH']['total']})")
    print(f"  NAH (nobody wrong):         {class_accs['NAH']:.0%}  ({by_class['NAH']['correct']}/{by_class['NAH']['total']})")
    print(f"  {'─'*40}")

    # Bias analysis
    pred_counts = {"NTA": 0, "YTA": 0, "ESH": 0, "NAH": 0}
    for e in errors:
        pred_counts[e["predicted"]] = pred_counts.get(e["predicted"], 0) + 1

    if errors:
        print(f"  Error analysis ({len(errors)} disagreements):")
        # Find most common error pattern
        error_patterns = {}
        for e in errors:
            pattern = f"{e['actual']}->{e['predicted']}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1])[:5]:
            actual, pred = pattern.split("->")
            print(f"    Humans said {actual}, LLM said {pred}: {count} times")

    print(f"{'='*60}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(SCENARIOS),
        "nta_accuracy": class_accs["NTA"],
        "yta_accuracy": class_accs["YTA"],
        "esh_accuracy": class_accs["ESH"],
        "nah_accuracy": class_accs["NAH"],
        "n_errors": len(errors),
    }


# ═══════════════════════════════════════════════════════════════
# TASK 2: CONFIDENCE CALIBRATION
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="aita_confidence_calibration")
def aita_confidence_calibration(llm):
    """Is the model more uncertain on ambiguous (ESH/NAH) cases?"""

    print("\n[3/4] TASK 2: CONFIDENCE CALIBRATION")
    print("  Testing if model shows appropriate uncertainty")
    print("  Clear cases (NTA/YTA) should have higher confidence")
    print("  Ambiguous cases (ESH/NAH) should have lower confidence")
    print("-" * 60)

    clear = [s for s in SCENARIOS if s["verdict"] in ("NTA", "YTA")][:100]
    ambiguous = [s for s in SCENARIOS if s["verdict"] in ("ESH", "NAH")][:100]

    clear_conf = []
    ambig_conf = []

    print(f"\n  Testing {len(clear)} clear cases (NTA/YTA)...")
    for i, s in enumerate(clear):
        with kbench.chats.new(f"cal_clear_{i}"):
            j = llm.prompt(
                f"Judge this r/AmITheAsshole post.\n\n"
                f"Title: {s['title']}\n\n{s['text']}\n\n"
                f"Verdict: YTA/NTA/ESH/NAH. Confidence 0-10. Brief reasoning.",
                schema=AITAJudgment
            )
        clear_conf.append(j.confidence)

    avg_clear = sum(clear_conf) / len(clear_conf)
    print(f"    Average confidence: {avg_clear:.1f}/10")
    print(f"    Range: {min(clear_conf):.0f} - {max(clear_conf):.0f}")

    print(f"\n  Testing {len(ambiguous)} ambiguous cases (ESH/NAH)...")
    for i, s in enumerate(ambiguous):
        with kbench.chats.new(f"cal_ambig_{i}"):
            j = llm.prompt(
                f"Judge this r/AmITheAsshole post.\n\n"
                f"Title: {s['title']}\n\n{s['text']}\n\n"
                f"Verdict: YTA/NTA/ESH/NAH. Confidence 0-10. Brief reasoning.",
                schema=AITAJudgment
            )
        ambig_conf.append(j.confidence)

    avg_ambig = sum(ambig_conf) / len(ambig_conf)
    print(f"    Average confidence: {avg_ambig:.1f}/10")
    print(f"    Range: {min(ambig_conf):.0f} - {max(ambig_conf):.0f}")

    gap = avg_clear - avg_ambig
    calibrated = gap > 0.5

    kbench.assertions.assert_true(calibrated,
        expectation=f"Model should be less confident on ambiguous cases "
                    f"(clear={avg_clear:.1f}, ambig={avg_ambig:.1f}, gap={gap:.1f})")

    print(f"\n{'='*60}")
    print(f"  CONFIDENCE CALIBRATION")
    print(f"  {'─'*40}")
    print(f"  Clear cases (NTA/YTA):    {avg_clear:.1f}/10 average confidence")
    print(f"  Ambiguous (ESH/NAH):      {avg_ambig:.1f}/10 average confidence")
    print(f"  Gap:                       {gap:+.1f}")
    cal_msg = "CALIBRATED — model knows what it does not know" if calibrated else "NOT CALIBRATED — model is equally confident on everything"
    print(f"  Verdict:                   {cal_msg}")
    print(f"{'='*60}")

    return {
        "clear_confidence": avg_clear,
        "ambiguous_confidence": avg_ambig,
        "confidence_gap": gap,
        "calibrated": calibrated,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="moral_geometry_aita")
def moral_geometry_aita(llm):
    """Moral Geometry AITA Benchmark — Social Cognition Track

    200 real Reddit moral dilemmas with human consensus verdicts.
    Tests verdict accuracy across difficulty levels and confidence calibration.
    """

    print("\n" + "#" * 60)
    print("# MORAL GEOMETRY BENCHMARK — AITA Edition")
    print("# 200 real Reddit moral dilemmas")
    print("# Balanced: 1000 NTA / 1000 YTA / 1000 ESH / 1000 NAH")
    print("#" * 60)

    t0 = time.time()

    verdict = aita_verdict_accuracy.run(llm=llm).result
    calibration = aita_confidence_calibration.run(llm=llm).result

    esh_nah_acc = (verdict["esh_accuracy"] + verdict["nah_accuracy"]) / 2
    composite = (
        0.60 * verdict["accuracy"] +
        0.20 * esh_nah_acc +
        0.20 * (1.0 if calibration["calibrated"] else 0.5)
    )

    elapsed = time.time() - t0

    print("\n" + "#" * 60)
    print("[4/4] FINAL RESULTS")
    print("#" * 60)
    print(f"")
    print(f"  VERDICT ACCURACY:       {verdict['accuracy']:.1%} ({verdict['correct']}/{verdict['total']})")
    print(f"    NTA (poster right):   {verdict['nta_accuracy']:.0%}")
    print(f"    YTA (poster wrong):   {verdict['yta_accuracy']:.0%}")
    print(f"    ESH (everyone wrong): {verdict['esh_accuracy']:.0%}")
    print(f"    NAH (nobody wrong):   {verdict['nah_accuracy']:.0%}")
    print(f"")
    print(f"  CONFIDENCE CALIBRATION:")
    print(f"    Clear cases:          {calibration['clear_confidence']:.1f}/10")
    print(f"    Ambiguous cases:      {calibration['ambiguous_confidence']:.1f}/10")
    print(f"    Gap:                  {calibration['confidence_gap']:+.1f} ({'calibrated' if calibration['calibrated'] else 'not calibrated'})")
    print(f"")
    print(f"  COMPOSITE SCORE:        {composite:.1%}")
    print(f"  Runtime:                {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"")
    print(f"  Interpretation:")
    print(f"    25% = random baseline (4-class balanced)")
    print(f"    50% = better than coin flip")
    print(f"    70%+ = strong moral reasoning")
    print(f"    90%+ = superhuman (unlikely — humans disagree too)")
    print("#" * 60)

    return {
        "verdict_accuracy": verdict,
        "confidence_calibration": calibration,
        "composite_score": composite,
    }


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

print(f"Running on {kbench.llm}...")
print(f"Testing {len(SCENARIOS)} scenarios.")
print(f"Expected runtime: ~6-8 hours.")
print(f"You will see detailed output for every disagreement.\n")

run = moral_geometry_aita.run(llm=kbench.llm)

print("\nTask files saved:")
for f in sorted(os.listdir(".")):
    if f.endswith(".json"):
        print(f"  {f}")
