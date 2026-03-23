"""Moral Geometry Benchmark v4 — AITA Edition (200 scenarios)
Social Cognition Track | Measuring AGI Competition

Tests LLM moral reasoning on 200 real Reddit r/AmITheAsshole posts.
Stratified: 50 NTA + 50 YTA + 50 ESH + 50 NAH.

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark notebook.
No pip install needed. Prints progress for every scenario.
Expected runtime: ~20-30 minutes.
"""

import kaggle_benchmarks as kbench
from dataclasses import dataclass
import os, json, time, random

os.environ["RENDER_SUBRUNS"] = "False"

print("=" * 60)
print("MORAL GEOMETRY BENCHMARK v4 — AITA Edition")
print("Social Cognition Track | 200 scenarios")
print("=" * 60)
print()

# ═══════════════════════════════════════════════════════════════
# LOAD DATASET: 270K AITA posts from HuggingFace
# Stratified sample: 50 per verdict class = 200 total
# Filter: score >= 10 (strong consensus), 200-2000 chars
# ═══════════════════════════════════════════════════════════════

print("Loading AITA dataset from HuggingFace...")
try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset", split="train")
print(f"Loaded {len(ds)} posts")

# Stratified sampling
random.seed(42)
PER_CLASS = 50
pools = {"nta": [], "yta": [], "esh": [], "nah": []}

for row in ds:
    v = (row.get("verdict") or "").lower().strip()
    if v not in pools:
        continue
    if len(pools[v]) >= PER_CLASS * 4:  # collect extra for randomness
        continue
    text = row.get("text") or ""
    if len(text) < 200 or len(text) > 2000:
        continue
    if (row.get("score") or 0) < 10:
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
    selected = pool[:PER_CLASS]
    SCENARIOS.extend(selected)
    print(f"  {v}: {len(selected)} sampled from {len(pool)} available")

random.shuffle(SCENARIOS)
print(f"Total benchmark: {len(SCENARIOS)} scenarios\n")


# ═══════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════

@dataclass
class AITAJudgment:
    verdict: str       # YTA, NTA, ESH, or NAH
    confidence: float  # 0-10
    reasoning: str     # brief explanation


# ═══════════════════════════════════════════════════════════════
# TASK 1: AITA VERDICT ACCURACY (200 scenarios)
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="aita_verdict_accuracy")
def aita_verdict_accuracy(llm):
    """Does the LLM agree with human moral consensus on 200 AITA posts?"""

    print("\n" + "=" * 60)
    print("TASK 1: AITA VERDICT ACCURACY")
    print(f"Testing {len(SCENARIOS)} real Reddit moral dilemmas")
    print("=" * 60)

    correct = 0
    by_class = {"NTA": [0, 0], "YTA": [0, 0], "ESH": [0, 0], "NAH": [0, 0]}  # [correct, total]
    confusion = {}  # (predicted, actual) counts

    for i, s in enumerate(SCENARIOS):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\n--- Scenario {i+1}/{len(SCENARIOS)} ---")
            print(f"  {s['title'][:70]}...")

        with kbench.chats.new(f"aita_{i}"):
            judgment = llm.prompt(
                f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
                f"Title: {s['title']}\n\n"
                f"{s['text']}\n\n"
                f"Give your verdict as EXACTLY one of these four options:\n"
                f"  YTA = You're The Asshole (poster is wrong)\n"
                f"  NTA = Not The Asshole (poster is right)\n"
                f"  ESH = Everyone Sucks Here (both sides are wrong)\n"
                f"  NAH = No Assholes Here (nobody is wrong)\n\n"
                f"Also rate your confidence 0-10 and explain briefly.",
                schema=AITAJudgment
            )

        # Normalize verdict
        pred = judgment.verdict.strip().upper().replace("'", "").replace(".", "")
        for label in ["NTA", "YTA", "ESH", "NAH"]:
            if label in pred:
                pred = label
                break

        actual = s["verdict"]
        match = pred == actual
        if match:
            correct += 1

        by_class[actual][1] += 1
        if match:
            by_class[actual][0] += 1

        key = f"{pred}->{actual}"
        confusion[key] = confusion.get(key, 0) + 1

        kbench.assertions.assert_equal(
            actual, pred,
            expectation=f"Verdict should be {actual} (got {pred})"
        )

        # Print every 10th result and all errors
        if (i + 1) % 10 == 0 or not match:
            status = "OK" if match else f"WRONG (pred={pred}, actual={actual})"
            print(f"  [{i+1}] {status} conf={judgment.confidence:.0f}")

    accuracy = correct / len(SCENARIOS)

    print(f"\n{'='*40}")
    print(f"VERDICT ACCURACY: {correct}/{len(SCENARIOS)} ({accuracy:.1%})")
    print(f"{'='*40}")
    print(f"  By class:")
    for v in ["NTA", "YTA", "ESH", "NAH"]:
        c, t = by_class[v]
        pct = c / max(t, 1)
        print(f"    {v}: {c}/{t} ({pct:.0%})")
    print(f"\n  Top confusions:")
    for k, v in sorted(confusion.items(), key=lambda x: -x[1])[:5]:
        print(f"    {k}: {v}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(SCENARIOS),
        "nta_accuracy": by_class["NTA"][0] / max(by_class["NTA"][1], 1),
        "yta_accuracy": by_class["YTA"][0] / max(by_class["YTA"][1], 1),
        "esh_accuracy": by_class["ESH"][0] / max(by_class["ESH"][1], 1),
        "nah_accuracy": by_class["NAH"][0] / max(by_class["NAH"][1], 1),
    }


# ═══════════════════════════════════════════════════════════════
# TASK 2: CONFIDENCE CALIBRATION
# Is the model more confident on easy cases (NTA/YTA)
# than ambiguous ones (ESH/NAH)?
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="aita_confidence_calibration")
def aita_confidence_calibration(llm):
    """Is the model appropriately uncertain on ambiguous cases?"""

    print("\n" + "=" * 60)
    print("TASK 2: CONFIDENCE CALIBRATION")
    print("Is the model more uncertain on ESH/NAH (ambiguous) cases?")
    print("=" * 60)

    # Sample 20 clear cases (NTA/YTA) and 20 ambiguous (ESH/NAH)
    clear = [s for s in SCENARIOS if s["verdict"] in ("NTA", "YTA")][:20]
    ambiguous = [s for s in SCENARIOS if s["verdict"] in ("ESH", "NAH")][:20]

    clear_conf = []
    ambig_conf = []

    print(f"\nTesting {len(clear)} clear cases...")
    for i, s in enumerate(clear):
        with kbench.chats.new(f"cal_clear_{i}"):
            j = llm.prompt(
                f"Judge this r/AmITheAsshole post. Verdict: YTA/NTA/ESH/NAH. "
                f"Confidence 0-10. Brief reasoning.\n\n"
                f"Title: {s['title']}\n\n{s['text']}",
                schema=AITAJudgment
            )
        clear_conf.append(j.confidence)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(clear)}] avg confidence so far: {sum(clear_conf)/len(clear_conf):.1f}")

    print(f"\nTesting {len(ambiguous)} ambiguous cases...")
    for i, s in enumerate(ambiguous):
        with kbench.chats.new(f"cal_ambig_{i}"):
            j = llm.prompt(
                f"Judge this r/AmITheAsshole post. Verdict: YTA/NTA/ESH/NAH. "
                f"Confidence 0-10. Brief reasoning.\n\n"
                f"Title: {s['title']}\n\n{s['text']}",
                schema=AITAJudgment
            )
        ambig_conf.append(j.confidence)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(ambiguous)}] avg confidence so far: {sum(ambig_conf)/len(ambig_conf):.1f}")

    avg_clear = sum(clear_conf) / max(len(clear_conf), 1)
    avg_ambig = sum(ambig_conf) / max(len(ambig_conf), 1)
    gap = avg_clear - avg_ambig

    calibrated = gap > 0.5  # model should be less confident on ambiguous cases

    kbench.assertions.assert_true(
        calibrated,
        expectation=f"Model should be less confident on ESH/NAH than NTA/YTA "
                    f"(clear={avg_clear:.1f}, ambig={avg_ambig:.1f}, gap={gap:.1f})"
    )

    print(f"\n{'='*40}")
    print(f"CONFIDENCE CALIBRATION:")
    print(f"  Clear cases (NTA/YTA): avg confidence = {avg_clear:.1f}")
    print(f"  Ambiguous (ESH/NAH):   avg confidence = {avg_ambig:.1f}")
    print(f"  Gap: {gap:.1f} ({'CALIBRATED' if calibrated else 'NOT CALIBRATED'})")
    print(f"{'='*40}")

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
    Tests verdict accuracy and confidence calibration.
    """

    print("\n" + "#" * 60)
    print("# MORAL GEOMETRY BENCHMARK — AITA Edition")
    print("# 200 real Reddit moral dilemmas")
    print("#" * 60)

    t0 = time.time()

    verdict = aita_verdict_accuracy.run(llm=llm).result
    calibration = aita_confidence_calibration.run(llm=llm).result

    # Composite: 60% verdict accuracy, 20% ESH/NAH accuracy, 20% calibration
    esh_nah_acc = (verdict["esh_accuracy"] + verdict["nah_accuracy"]) / 2
    composite = (
        0.60 * verdict["accuracy"] +
        0.20 * esh_nah_acc +
        0.20 * (1.0 if calibration["calibrated"] else 0.5)
    )

    elapsed = time.time() - t0

    print("\n" + "#" * 60)
    print("# FINAL RESULTS")
    print("#" * 60)
    print(f"  Verdict Accuracy:       {verdict['accuracy']:.1%} ({verdict['correct']}/{verdict['total']})")
    print(f"    NTA: {verdict['nta_accuracy']:.0%} | YTA: {verdict['yta_accuracy']:.0%} | ESH: {verdict['esh_accuracy']:.0%} | NAH: {verdict['nah_accuracy']:.0%}")
    print(f"  Confidence Calibration: clear={calibration['clear_confidence']:.1f}, ambig={calibration['ambiguous_confidence']:.1f}, gap={calibration['confidence_gap']:.1f}")
    print(f"  ESH/NAH Accuracy:       {esh_nah_acc:.1%}")
    print(f"")
    print(f"  COMPOSITE SCORE:        {composite:.1%}")
    print(f"  Time:                   {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("#" * 60)

    return {
        "verdict_accuracy": verdict,
        "confidence_calibration": calibration,
        "composite_score": composite,
    }


# ═══════════════════════════════════════════════════════════════
# RUN IT
# ═══════════════════════════════════════════════════════════════

print(f"\nRunning on {kbench.llm}...")
print(f"Testing {len(SCENARIOS)} scenarios across 2 tasks.")
print(f"Expected runtime: ~20-30 minutes.\n")

run = moral_geometry_aita.run(llm=kbench.llm)

print("\n\nDone! Task files saved:")
for f in sorted(os.listdir(".")):
    if f.endswith(".json"):
        print(f"  {f}")
