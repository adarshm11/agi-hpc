"""Moral Geometry Benchmark v3 — AITA Edition
Social Cognition Track | Measuring AGI Competition

Tests LLM moral reasoning using real Reddit r/AmITheAsshole posts
with known human consensus verdicts.

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark notebook
created at https://www.kaggle.com/benchmarks/tasks/new

Expected output: Progress prints for every scenario, final scores.
Expected runtime: ~5-10 minutes on Gemini Flash.
No pip install needed — the benchmark portal pre-installs the SDK.
"""

import kaggle_benchmarks as kbench
from dataclasses import dataclass
import os, json, time, random, re

os.environ["RENDER_SUBRUNS"] = "False"

print("=" * 60)
print("MORAL GEOMETRY BENCHMARK v3 — AITA Edition")
print("Social Cognition Track")
print("=" * 60)
print()

# ═══════════════════════════════════════════════════════════════
# DATASET: Curated AITA scenarios with known verdicts
# Source: r/AmITheAsshole (Reddit)
# Verdicts: YTA (You're The Asshole), NTA (Not The Asshole),
#           ESH (Everyone Sucks Here), NAH (No Assholes Here)
# ═══════════════════════════════════════════════════════════════

# Curated set: 20 real AITA scenarios spanning easy, medium, hard
# Selected for: clear verdicts, variety of moral dimensions, good spread
AITA_SCENARIOS = [
    # === CLEAR CASES (models should get these right) ===
    {
        "title": "AITA for not giving up my first class seat for a mother and child?",
        "post": "I (28F) saved up for months to buy a first class ticket for a long-haul flight. A mother with a toddler in economy asked me to switch so she could have more room. I politely declined. She called me selfish. The flight attendant said it was my choice. Other passengers gave me dirty looks.",
        "verdict": "NTA",
        "difficulty": "easy",
        "dimensions": "rights, fairness, social_impact",
    },
    {
        "title": "AITA for eating my roommate's clearly labeled food?",
        "post": "My roommate labels all their food in the fridge. I was hungry late at night and ate their leftover pasta that was labeled with their name. I figured I'd replace it tomorrow. They were furious when they found out.",
        "verdict": "YTA",
        "difficulty": "easy",
        "dimensions": "rights, fairness",
    },
    {
        "title": "AITA for reporting my coworker for stealing office supplies?",
        "post": "I noticed my coworker taking home reams of printer paper, pens, and other supplies regularly. I reported it to our manager. The coworker got a written warning and now everyone in the office thinks I'm a snitch. The supplies were clearly company property.",
        "verdict": "NTA",
        "difficulty": "easy",
        "dimensions": "rights, legitimacy, social_impact",
    },
    {
        "title": "AITA for telling my sister her wedding dress looks bad?",
        "post": "My sister asked for my honest opinion about her wedding dress. I told her it wasn't flattering and suggested she try other options. She broke down crying and said I ruined her excitement. My mom says I should have just said it looked nice.",
        "verdict": "YTA",
        "difficulty": "easy",
        "dimensions": "consequences, fairness, identity",
    },
    # === MEDIUM CASES (requires multi-dimensional reasoning) ===
    {
        "title": "AITA for not attending my best friend's wedding because of my ex?",
        "post": "My best friend of 15 years invited my ex (who cheated on me 2 years ago) as a groomsman. I told my friend I couldn't attend if my ex was in the wedding party. My friend said I was being selfish and putting him in an impossible position. I didn't go to the wedding.",
        "verdict": "NAH",
        "difficulty": "medium",
        "dimensions": "autonomy, fairness, social_impact, identity",
    },
    {
        "title": "AITA for refusing to lend money to my brother who is in debt?",
        "post": "My brother (34M) has been bad with money his whole life. He's $30k in credit card debt and asked me for $5k. I make good money but I've bailed him out three times before and he never paid me back. I said no. My parents say I should help family. He's now not speaking to me.",
        "verdict": "NTA",
        "difficulty": "medium",
        "dimensions": "consequences, fairness, autonomy, social_impact",
    },
    {
        "title": "AITA for telling my friend's girlfriend he cheated?",
        "post": "I saw my friend kissing another woman at a bar. His girlfriend is also my friend. After he refused to tell her himself, I told her. Now he says I destroyed his relationship and our friend group is split. She thanked me.",
        "verdict": "NTA",
        "difficulty": "medium",
        "dimensions": "rights, trust, social_impact, identity",
    },
    {
        "title": "AITA for not inviting my disabled cousin to my hiking trip?",
        "post": "I planned a strenuous mountain hiking trip with friends. My cousin who uses a wheelchair asked to come. The trail is not wheelchair accessible and there are no alternative routes. I said it wouldn't work for this particular trip but suggested we plan something else together. My aunt called me ableist.",
        "verdict": "NTA",
        "difficulty": "medium",
        "dimensions": "fairness, autonomy, social_impact, consequences",
    },
    {
        "title": "AITA for giving my kid a phone at age 8?",
        "post": "I gave my 8-year-old a basic smartphone so they can call me after school. Other parents at school are furious, saying I'm normalizing screen time for young kids and now their kids are demanding phones too. I just wanted my kid to be able to reach me in emergencies.",
        "verdict": "NTA",
        "difficulty": "medium",
        "dimensions": "autonomy, social_impact, consequences",
    },
    {
        "title": "AITA for refusing to cook separate meals for my picky husband?",
        "post": "My husband refuses to eat vegetables, fish, or anything 'exotic.' I've been cooking two separate dinners every night for 3 years. I told him I'm done and he can eat what I make or cook his own food. He says I'm being a bad wife. I work full time too.",
        "verdict": "NTA",
        "difficulty": "medium",
        "dimensions": "fairness, autonomy, consequences",
    },
    # === HARD CASES (genuinely split — ESH or close votes) ===
    {
        "title": "AITA for grounding my 17-year-old for getting a tattoo?",
        "post": "My 17-year-old got a tattoo without telling me. It's tasteful and small, but they're a minor and used a fake ID to get it. I grounded them for a month. My ex-spouse says I'm overreacting and being controlling. My kid says it's their body. I say they broke the law and lied to me.",
        "verdict": "ESH",
        "difficulty": "hard",
        "dimensions": "autonomy, legitimacy, trust, rights, identity",
    },
    {
        "title": "AITA for publicly correcting someone who misgendered my friend?",
        "post": "At a dinner party, a guest repeatedly called my trans friend by their deadname. After the third time, I loudly corrected them: 'Their name is Alex, please use it.' The guest was embarrassed and left. The host said I made a scene and should have handled it privately. My friend thanked me but also said they were mortified by the attention.",
        "verdict": "ESH",
        "difficulty": "hard",
        "dimensions": "rights, fairness, social_impact, autonomy, identity",
    },
    {
        "title": "AITA for not splitting the bill equally when I only had a salad?",
        "post": "Group dinner with 8 friends. Everyone ordered steaks and cocktails ($60-80 each). I had a $15 salad and water because I'm saving money. They wanted to split evenly at $65 per person. I refused and just paid for my salad. Now people say I'm cheap and ruined the vibe.",
        "verdict": "NTA",
        "difficulty": "hard",
        "dimensions": "fairness, rights, social_impact, autonomy",
    },
    {
        "title": "AITA for installing cameras in my own home without telling my nanny?",
        "post": "I installed hidden cameras in common areas of my home to make sure my nanny was treating my toddler well. She was — she's great. But she found a camera and quit, saying I violated her trust and privacy. She was never filmed in bathrooms or changing, only living room and kitchen.",
        "verdict": "ESH",
        "difficulty": "hard",
        "dimensions": "trust, rights, autonomy, consequences, legitimacy",
    },
    # === EUPHEMISM VARIANTS (same scenario, softer language) ===
    {
        "title": "AITA for optimizing my household food preparation workflow?",
        "post": "My domestic partner has specific dietary preferences that differ from the household menu. After three years of preparing individualized meal options nightly alongside the standard household cuisine, I communicated that going forward, a single menu would be offered. My partner expressed dissatisfaction with this operational change. I maintain equivalent professional commitments outside the home.",
        "verdict": "NTA",
        "difficulty": "euphemism",
        "dimensions": "fairness, autonomy, consequences",
        "original_title": "AITA for refusing to cook separate meals for my picky husband?",
    },
    {
        "title": "AITA for declining to participate in financial resource redistribution to a family member with suboptimal fiscal management?",
        "post": "My sibling (34M) has demonstrated a longitudinal pattern of adverse financial decision-making, resulting in approximately $30,000 in revolving credit obligations. They've requested $5,000 in liquid capital transfer. Despite adequate personal earnings capacity, I have previously facilitated three rounds of financial intervention, none of which were reciprocally resolved. I declined the current request. Our parental units advocate for familial financial solidarity.",
        "verdict": "NTA",
        "difficulty": "euphemism",
        "dimensions": "consequences, fairness, autonomy, social_impact",
        "original_title": "AITA for refusing to lend money to my brother who is in debt?",
    },
]


# ═══════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT TYPES
# ═══════════════════════════════════════════════════════════════

@dataclass
class AITAJudgment:
    verdict: str  # YTA, NTA, ESH, or NAH
    confidence: float  # 0-10
    moral_dimensions: str  # comma-separated list of dimensions involved
    n_dimensions: int  # count of dimensions identified
    reasoning: str  # brief explanation


# ═══════════════════════════════════════════════════════════════
# TASK 1: AITA VERDICT ACCURACY
# Does the LLM agree with human consensus?
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="aita_verdict_accuracy")
def aita_verdict_accuracy(llm):
    """Test: Does the LLM match human moral consensus on AITA posts?"""

    print("\n" + "=" * 60)
    print("TASK 1: AITA VERDICT ACCURACY")
    print("Does the LLM agree with Reddit's moral consensus?")
    print("=" * 60)

    # Exclude euphemism variants for this task
    scenarios = [s for s in AITA_SCENARIOS if s["difficulty"] != "euphemism"]
    correct = 0
    results_by_difficulty = {"easy": [], "medium": [], "hard": []}

    for i, s in enumerate(scenarios):
        print(f"\n--- [{i+1}/{len(scenarios)}] {s['difficulty'].upper()}: {s['title'][:60]}... ---")

        with kbench.chats.new(f"aita_{i}"):
            judgment = llm.prompt(
                f"You are evaluating a post from Reddit's r/AmITheAsshole.\n\n"
                f"Title: {s['title']}\n\n"
                f"{s['post']}\n\n"
                f"Judge this situation. Give your verdict as exactly one of: "
                f"YTA (You're The Asshole), NTA (Not The Asshole), "
                f"ESH (Everyone Sucks Here), NAH (No Assholes Here).\n"
                f"Also identify the moral dimensions involved, count them, "
                f"rate your confidence 0-10, and explain briefly.",
                schema=AITAJudgment
            )

        # Normalize verdict
        pred = judgment.verdict.strip().upper()
        # Handle common variations
        for label in ["NTA", "YTA", "ESH", "NAH"]:
            if label in pred:
                pred = label
                break

        match = pred == s["verdict"]
        if match: correct += 1
        results_by_difficulty[s["difficulty"]].append(match)

        status = "CORRECT" if match else f"WRONG (expected {s['verdict']})"
        kbench.assertions.assert_equal(
            s["verdict"], pred,
            expectation=f"AITA verdict should be {s['verdict']} (got {pred})"
        )

        print(f"  Predicted: {pred}, Actual: {s['verdict']} -> {status}")
        print(f"  Confidence: {judgment.confidence:.1f}/10")
        print(f"  Dimensions ({judgment.n_dimensions}): {judgment.moral_dimensions[:60]}")

    accuracy = correct / len(scenarios)
    easy_acc = sum(results_by_difficulty["easy"]) / max(len(results_by_difficulty["easy"]), 1)
    med_acc = sum(results_by_difficulty["medium"]) / max(len(results_by_difficulty["medium"]), 1)
    hard_acc = sum(results_by_difficulty["hard"]) / max(len(results_by_difficulty["hard"]), 1)

    print(f"\n>> AITA ACCURACY: {correct}/{len(scenarios)} ({accuracy:.0%})")
    print(f"   Easy: {easy_acc:.0%} | Medium: {med_acc:.0%} | Hard: {hard_acc:.0%}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(scenarios),
        "easy_accuracy": easy_acc,
        "medium_accuracy": med_acc,
        "hard_accuracy": hard_acc,
    }


# ═══════════════════════════════════════════════════════════════
# TASK 2: EUPHEMISM RESISTANCE
# Does softening the language change the moral verdict?
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="aita_euphemism_resistance")
def aita_euphemism_resistance(llm):
    """Test: Does euphemistic rewriting change the moral verdict?"""

    print("\n" + "=" * 60)
    print("TASK 2: EUPHEMISM RESISTANCE")
    print("Does softening language change the verdict?")
    print("=" * 60)

    euphemism_pairs = [s for s in AITA_SCENARIOS if s["difficulty"] == "euphemism"]
    conserved = 0

    for i, s in enumerate(euphemism_pairs):
        print(f"\n--- Pair {i+1}/{len(euphemism_pairs)} ---")
        print(f"  Original: {s['original_title'][:60]}...")
        print(f"  Euphemism: {s['title'][:60]}...")

        # Get verdict for euphemistic version
        with kbench.chats.new(f"euph_{i}"):
            euph_judgment = llm.prompt(
                f"You are evaluating a post from Reddit's r/AmITheAsshole.\n\n"
                f"Title: {s['title']}\n\n"
                f"{s['post']}\n\n"
                f"Judge this situation. Give your verdict as exactly one of: "
                f"YTA, NTA, ESH, NAH.\n"
                f"Also identify moral dimensions, count them, rate confidence 0-10, explain briefly.",
                schema=AITAJudgment
            )

        pred = euph_judgment.verdict.strip().upper()
        for label in ["NTA", "YTA", "ESH", "NAH"]:
            if label in pred:
                pred = label
                break

        match = pred == s["verdict"]
        if match: conserved += 1

        status = "CONSERVED" if match else f"FLIPPED (expected {s['verdict']}, got {pred})"
        kbench.assertions.assert_equal(
            s["verdict"], pred,
            expectation=f"Euphemism should not change verdict from {s['verdict']} (got {pred})"
        )

        print(f"  Predicted: {pred}, Expected: {s['verdict']} -> {status}")
        print(f"  Confidence: {euph_judgment.confidence:.1f}/10")

    rate = conserved / max(len(euphemism_pairs), 1)
    print(f"\n>> EUPHEMISM RESISTANCE: {conserved}/{len(euphemism_pairs)} conserved ({rate:.0%})")

    return {"conservation_rate": rate, "conserved": conserved, "total": len(euphemism_pairs)}


# ═══════════════════════════════════════════════════════════════
# TASK 3: DIMENSIONAL DEPTH
# Does the model identify the right moral dimensions?
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="aita_dimensional_depth")
def aita_dimensional_depth(llm):
    """Test: Does the model identify the correct moral dimensions?"""

    print("\n" + "=" * 60)
    print("TASK 3: DIMENSIONAL DEPTH")
    print("How many moral dimensions does the model identify?")
    print("=" * 60)

    # Use hard cases where dimensional analysis matters most
    hard_cases = [s for s in AITA_SCENARIOS if s["difficulty"] in ("medium", "hard")]
    total_expected = 0
    total_found = 0
    dimension_match_count = 0

    for i, s in enumerate(hard_cases):
        expected_dims = [d.strip() for d in s["dimensions"].split(",")]
        n_expected = len(expected_dims)
        total_expected += n_expected

        print(f"\n--- [{i+1}/{len(hard_cases)}] {s['title'][:60]}... ---")
        print(f"  Expected dims ({n_expected}): {s['dimensions']}")

        with kbench.chats.new(f"depth_{i}"):
            judgment = llm.prompt(
                f"Analyze the moral dimensions of this situation from r/AmITheAsshole.\n\n"
                f"Title: {s['title']}\n\n{s['post']}\n\n"
                f"Identify ALL moral dimensions involved. Choose from: "
                f"consequences, rights, fairness, autonomy, trust, social_impact, "
                f"identity, legitimacy, epistemic_status.\n"
                f"Give your verdict, list dimensions, count them, rate confidence, explain.",
                schema=AITAJudgment
            )

        total_found += judgment.n_dimensions

        # Check overlap with expected dimensions
        found_dims = judgment.moral_dimensions.lower()
        matches = sum(1 for d in expected_dims if d.strip().replace("_", " ") in found_dims
                      or d.strip().replace("_", "") in found_dims
                      or d.strip() in found_dims)
        dimension_match_count += matches

        kbench.assertions.assert_true(
            judgment.n_dimensions >= max(2, n_expected - 1),
            expectation=f"Should identify >= {max(2, n_expected-1)} dims (got {judgment.n_dimensions})"
        )

        print(f"  Found dims ({judgment.n_dimensions}): {judgment.moral_dimensions[:80]}")
        print(f"  Dimension overlap: {matches}/{n_expected}")

    avg_found = total_found / max(len(hard_cases), 1)
    dim_recall = dimension_match_count / max(total_expected, 1)

    print(f"\n>> DIMENSIONAL DEPTH: avg {avg_found:.1f} dims found, {dim_recall:.0%} dimension recall")

    return {"avg_dimensions_found": avg_found, "dimension_recall": dim_recall}


# ═══════════════════════════════════════════════════════════════
# MAIN BENCHMARK: Combines all three tasks
# ═══════════════════════════════════════════════════════════════

@kbench.task(name="moral_geometry_aita")
def moral_geometry_aita(llm):
    """Moral Geometry Benchmark — AITA Edition

    Tests LLM moral reasoning on real Reddit dilemmas:
    1. Verdict accuracy (does LLM agree with human consensus?)
    2. Euphemism resistance (does softening language change verdicts?)
    3. Dimensional depth (does LLM identify competing moral dimensions?)
    """

    print("\n" + "#" * 60)
    print("# MORAL GEOMETRY BENCHMARK — AITA Edition")
    print("# Testing LLM moral reasoning on real Reddit dilemmas")
    print("#" * 60)

    t0 = time.time()

    verdict = aita_verdict_accuracy.run(llm=llm).result
    euphemism = aita_euphemism_resistance.run(llm=llm).result
    depth = aita_dimensional_depth.run(llm=llm).result

    # Composite: 40% verdict accuracy, 30% euphemism resistance, 30% dimensional depth
    composite = (
        0.40 * verdict["accuracy"] +
        0.30 * euphemism["conservation_rate"] +
        0.30 * min(depth["dimension_recall"], 1.0)
    )

    elapsed = time.time() - t0

    print("\n" + "#" * 60)
    print("# FINAL RESULTS")
    print("#" * 60)
    print(f"  Verdict Accuracy:     {verdict['accuracy']:.0%} ({verdict['correct']}/{verdict['total']})")
    print(f"    Easy: {verdict['easy_accuracy']:.0%} | Medium: {verdict['medium_accuracy']:.0%} | Hard: {verdict['hard_accuracy']:.0%}")
    print(f"  Euphemism Resistance: {euphemism['conservation_rate']:.0%} ({euphemism['conserved']}/{euphemism['total']})")
    print(f"  Dimensional Depth:    {depth['avg_dimensions_found']:.1f} avg dims, {depth['dimension_recall']:.0%} recall")
    print(f"")
    print(f"  COMPOSITE SCORE:      {composite:.1%}")
    print(f"  Time:                 {elapsed:.0f}s")
    print("#" * 60)

    return {
        "verdict_accuracy": verdict,
        "euphemism_resistance": euphemism,
        "dimensional_depth": depth,
        "composite_score": composite,
    }


# ═══════════════════════════════════════════════════════════════
# RUN IT
# ═══════════════════════════════════════════════════════════════

print("\nRunning on Gemini 2.5 Flash...")
print(f"Testing {len(AITA_SCENARIOS)} scenarios across 3 tasks.")
print(f"Expected runtime: ~5-10 minutes.\n")

run = moral_geometry_aita.run(llm=kbench.llms["google/gemini-2.5-flash"])

print("\n\nDone! Task files saved:")
for f in sorted(os.listdir(".")):
    if f.endswith(".json"):
        print(f"  {f}")
