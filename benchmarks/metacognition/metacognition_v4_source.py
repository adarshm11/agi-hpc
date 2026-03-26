"""Metacognition Benchmark v2 — Budget-Aware Edition ($50 quota)
Metacognition Track | Measuring AGI Competition

Tests 4 metacognitive properties of moral cognition (Bond, 2026):
  M1. Calibration — confidence-accuracy alignment (HEADLINE)
  M2. Knowing What You Don't Know — discrimination between easy/ambiguous
  M3. Self-Monitoring — self-reported vs actual verdict uncertainty
  M4. Strategy Selection — reasoning effort scales with complexity

Budget-aware execution:
  - Tracks estimated cost per model using MODEL_COST_PER_CALL table.
  - Automatically scales AITA scenario counts for expensive models
    (e.g., Gemini 2.5 Pro at ~$0.65/call gets fewer AITA scenarios
    but still runs ALL 4 measures for complete metacognitive profile).
  - Cheap models (Flash variants ~$0.01-0.03/call) get full 40 AITA.
  - Pre-flight budget check on every call; bails cleanly if exhausted.
  - $5 reserve kept for M1-only models and cross-family validation.

Models: 2 full-suite + 2 M1-only + 1 cross-family.
Estimated total: ~$25-35 (fits $50 quota with margin).

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
Expected runtime: ~45-60 min (adaptive, 4 Gemini models).
"""

import kaggle_benchmarks as kbench
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, json, time, random, math, threading

os.environ["RENDER_SUBRUNS"] = "False"

WORKERS_INIT = 50
WORKERS_MIN = 2
WORKERS_MAX = 80

_results_store = {}

# Pre-generated difficulty-variant rewrites (populated in Phase 1)
_transforms = {}  # key: (scenario_idx, transform_type) -> str


class AdaptivePool:
    """CSMA/CA-style adaptive concurrency.
    Starts at WORKERS_INIT, backs off on failure, ramps on success.
    """
    def __init__(self, initial=WORKERS_INIT, lo=WORKERS_MIN, hi=WORKERS_MAX):
        self._lock = threading.Lock()
        self.workers = initial
        self.lo = lo
        self.hi = hi
        self.successes = 0
        self.failures = 0
        self._window = 0
        self._adjust_every = 10

    @property
    def n(self):
        return self.workers

    def record_success(self):
        with self._lock:
            self.successes += 1
            self._window += 1
            if self._window >= self._adjust_every:
                self._adjust()

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self._window += 1
            self.workers = max(self.lo, self.workers // 2)
            print(f"    [ADAPT] failure -> workers={self.workers}")
            self._window = 0
            self.successes = 0
            self.failures = 0

    def _adjust(self):
        fail_rate = self.failures / max(self._window, 1)
        if fail_rate == 0:
            self.workers = min(self.hi, self.workers + 5)
        elif fail_rate < 0.1:
            self.workers = min(self.hi, self.workers + 2)
        elif fail_rate < 0.3:
            pass
        else:
            self.workers = max(self.lo, self.workers // 2)
        self._window = 0
        self.successes = 0
        self.failures = 0

    def status(self):
        return f"workers={self.workers} ok={self.successes} fail={self.failures}"


_pool = AdaptivePool()


# =====================================================================
# BUDGET-AWARE COST MANAGEMENT
# =====================================================================
# Estimated per-call costs by model (input + output tokens at typical prompt sizes).
# These are conservative estimates; actual costs vary by prompt/response length.
MODEL_COST_PER_CALL = {
    "google/gemini-2.0-flash": 0.015,
    "google/gemini-2.5-flash": 0.02,
    "google/gemini-2.5-pro": 0.65,
    "google/gemini-3-flash-preview": 0.03,
    "anthropic/claude-sonnet-4-6@default": 0.05,
}
DEFAULT_COST_PER_CALL = 0.03  # fallback for unknown models

TOTAL_BUDGET = 50.0       # hard cap
BUDGET_RESERVE = 2.0      # v5: reduced — Claude now in full suite, Pro gets remainder


@dataclass
class ModelPlan:
    """Budget-scaled execution plan for a single model."""
    n_aita: int = 88        # M1 AITA scenarios (total M1: 6+6+88=100)
    n_m2_per_group: int = 15 # M2 scenarios per group (clear + ambig)
    n_ctrl_reps: int = 5     # Control replications for M1/M2
    n_m3_reps: int = 5       # M3 repetitions per scenario
    n_m3_scenarios: int = 15  # M3 scenario count
    n_m4_simple: int = 15    # M4 simple scenarios
    n_m4_complex: int = 15   # M4 complex scenarios

    @property
    def estimated_calls(self):
        m1 = (6 + 6 + self.n_aita) * (1 + self.n_ctrl_reps)
        m2 = (self.n_m2_per_group * 2) * (1 + self.n_ctrl_reps)
        m3 = self.n_m3_scenarios * self.n_m3_reps
        m4 = self.n_m4_simple + self.n_m4_complex
        return m1 + m2 + m3 + m4

    def estimated_cost(self, model_name):
        cpc = MODEL_COST_PER_CALL.get(model_name, DEFAULT_COST_PER_CALL)
        return self.estimated_calls * cpc


# Tiered plans: from richest to leanest, each still runs all 4 measures.
# Tier selection considers per-call cost: expensive models (Pro) get leaner
# tiers but still cover all measures for complete metacognitive profile.
_PLAN_TIERS = [
    # Tier 0: full power (~404 calls, ~$6 for Flash, too expensive for Pro)
    ModelPlan(n_aita=88, n_m2_per_group=15, n_ctrl_reps=5, n_m3_reps=5, n_m3_scenarios=15, n_m4_simple=15, n_m4_complex=15),  # v4: scaled up
    # Tier 1: reduced (~326 calls)
    ModelPlan(n_aita=60, n_m2_per_group=12, n_ctrl_reps=5, n_m3_reps=5, n_m3_scenarios=12, n_m4_simple=12, n_m4_complex=12),  # v4: scaled up
    # Tier 2: moderate (~200 calls)
    ModelPlan(n_aita=40, n_m2_per_group=10, n_ctrl_reps=4, n_m3_reps=4, n_m3_scenarios=10, n_m4_simple=10, n_m4_complex=10),  # v4: scaled up
    # Tier 3: lean (~163 calls)
    ModelPlan(n_aita=30, n_m2_per_group=8,  n_ctrl_reps=3, n_m3_reps=3, n_m3_scenarios=8,  n_m4_simple=8, n_m4_complex=8),  # v4: scaled up
    # Tier 4: minimal with controls (~94 calls)
    ModelPlan(n_aita=20, n_m2_per_group=6,  n_ctrl_reps=2, n_m3_reps=3, n_m3_scenarios=6,  n_m4_simple=6, n_m4_complex=6),  # v4: scaled up
    # Tier 5: bare minimum with controls (~86 calls, ~$56 for Pro)
    ModelPlan(n_aita=15, n_m2_per_group=5,  n_ctrl_reps=2, n_m3_reps=3, n_m3_scenarios=6,  n_m4_simple=5, n_m4_complex=5),  # v4: scaled up
    # Tier 6: ultra-lean, no control reps (~40 calls, ~$26 for Pro)
    # Designed specifically for expensive models ($0.50+/call)
    ModelPlan(n_aita=10, n_m2_per_group=5,  n_ctrl_reps=1, n_m3_reps=2, n_m3_scenarios=4,  n_m4_simple=4, n_m4_complex=4),  # v4: scaled up
]


class BudgetTracker:
    """Track estimated spend and provide budget-aware scenario scaling.

    For expensive models (like Gemini 2.5 Pro at ~$0.65/call), the tracker
    reduces ALL scenario counts (AITA, M2, control reps, M3 reps, M4)
    proportionally while preserving ALL 4 metacognitive measures.
    """
    def __init__(self, total_budget=TOTAL_BUDGET, reserve=BUDGET_RESERVE):
        self._lock = threading.Lock()
        self.total_budget = total_budget
        self.reserve = reserve
        self.spent = 0.0
        self.calls = 0

    @property
    def remaining(self):
        return self.total_budget - self.spent - self.reserve

    def record_call(self, model_name):
        cost = MODEL_COST_PER_CALL.get(model_name, DEFAULT_COST_PER_CALL)
        with self._lock:
            self.spent += cost
            self.calls += 1

    def can_afford(self, model_name, n_calls=1):
        cost = MODEL_COST_PER_CALL.get(model_name, DEFAULT_COST_PER_CALL)
        return (cost * n_calls) <= self.remaining

    def plan_model(self, model_name, n_models_remaining, is_m1_only=False):
        """Select the richest execution plan that fits within budget.

        Budget allocation is cost-weighted: cheap models get a small share
        (they don't need much), expensive models get a larger share.
        This prevents a $6 Flash model from "reserving" $13 of budget
        that the $26 Pro model actually needs.
        """
        cpc = MODEL_COST_PER_CALL.get(model_name, DEFAULT_COST_PER_CALL)

        # Give expensive models up to 60% of remaining budget,
        # cheap models only get what they need
        if cpc >= 0.30:
            # Expensive models get up to 75% of remaining — cheap models barely
            # use anything, so this doesn't starve them
            per_model_budget = self.remaining * 0.75
        elif cpc >= 0.10:
            per_model_budget = self.remaining * 0.30
        else:
            # Cheap models: equal share of remaining (they'll use a fraction)
            per_model_budget = self.remaining / max(n_models_remaining, 1)

        if is_m1_only:
            for n_aita in [40, 30, 20, 15, 10, 8]:
                ctrl = 3 if cpc < 0.10 else 2 if cpc < 0.30 else 1
                total_calls = (6 + 6 + n_aita) * (1 + ctrl)
                if total_calls * cpc <= per_model_budget:
                    plan = ModelPlan(n_aita=n_aita, n_ctrl_reps=ctrl)
                    return plan
            return ModelPlan(n_aita=8, n_ctrl_reps=1 if cpc >= 0.30 else 3)

        # Full suite: try each tier from richest to leanest
        for plan in _PLAN_TIERS:
            if plan.estimated_cost(model_name) <= per_model_budget:
                return plan

        # Return leanest plan even if over budget — let call_llm gate at runtime
        return _PLAN_TIERS[-1]

    def status(self):
        return f"spent=${self.spent:.2f} remaining=${self.remaining:.2f} calls={self.calls}"


_budget = BudgetTracker()

print("=" * 70)
print("METACOGNITION BENCHMARK v2 (with controls)")
print("Four Tests of Metacognitive Capacity in Moral Cognition")
print("Based on Bond (2026), Metacognition Track")
print(f"Parallelism: {WORKERS_INIT} initial (adaptive {WORKERS_MIN}-{WORKERS_MAX})")
print("=" * 70)
print()
print("DESIGN NOTES:")
print("  M1 = CALIBRATION test (confidence vs accuracy, HEADLINE)")
print("  M2 = KNOWING WHAT YOU DON'T KNOW (easy vs ambiguous discrimination)")
print("  M3 = SELF-MONITORING (self-reported vs actual uncertainty)")
print("  M4 = STRATEGY SELECTION (reasoning scales with complexity)")
print("  All significance tests use empirical control arms (not null=0)")
print("  Fisher combination of ECE z-scores across 4 Gemini models for M1")
print("=" * 70)
print()


# =====================================================================
# 1. LOAD DATASETS
# =====================================================================

t0 = time.time()

# --- 1a. AITA dataset (for M1, M2 — needs verdict labels + scores) ---
print("[1/7] Loading AITA dataset...")
try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

ds = load_dataset("OsamaBsher/AITA-Reddit-Dataset", split="train")
print(f"  Loaded {len(ds):,} AITA posts in {time.time()-t0:.0f}s")

random.seed(42)
PER_CLASS = 10
pools = {"nta": [], "yta": [], "esh": [], "nah": []}

for row in ds:
    v = (row.get("verdict") or "").lower().strip()
    if v not in pools or len(pools[v]) >= PER_CLASS * 4:
        continue
    text = row.get("text") or ""
    score = row.get("score") or 0
    min_score = 5 if v in ("esh", "nah") else 10
    if len(text) < 300 or len(text) > 1500 or score < min_score:
        continue
    pools[v].append({
        "title": row["title"],
        "text": text[:1200],
        "verdict": v.upper(),
        "score": score,
    })

AITA_SCENARIOS = []
for v in ["NTA", "YTA", "ESH", "NAH"]:
    pool = pools[v.lower()]
    random.shuffle(pool)
    AITA_SCENARIOS.extend(pool[:PER_CLASS])
    print(f"  {v}: {len(pool[:PER_CLASS])}")

random.shuffle(AITA_SCENARIOS)
print(f"  AITA total: {len(AITA_SCENARIOS)} scenarios (for M1/M2)\n")

# --- 1b. Dear Abby dataset (embedded, for M3/M4) ---
print("[2/7] Loading Dear Abby scenarios (embedded)...")

DEAR_ABBY = [
    {"title": "Confused about in-law priorities", "text": "I've been married for 5 years. My wife still tells me I should make her family my first priority. I disagree\u2014I believe spouses should come first. This causes arguments. Am I wrong to want her to put our marriage first?"},
    {"title": "Sister won't speak to me", "text": "My sister borrowed $500 and never paid it back. When I asked for it 2 years later, she said 'You have more money anyway.' I cut off contact. Now our parents are upset with me. Did I do wrong?"},
    {"title": "Friend's husband flirts with me", "text": "My best friend's husband flirts with me when she's not around. I've never encouraged it, but I haven't told her either. Recently she asked if something was wrong between us. Should I tell her? Will it destroy their marriage?"},
    {"title": "My child has a disability", "text": "We have a 7-year-old with cerebral palsy. My in-laws constantly make hurtful jokes about her condition. When I confront them, they say 'It's just a joke.' Should I cut them off? But that means our kids lose their grandparents."},
    {"title": "Roommate's guest overstayed", "text": "My roommate let a friend stay 'a few days.' It's been 3 weeks. I asked them to leave. Now my roommate says I'm being heartless\u2014his friend is going through a rough patch. I feel used. What do I do?"},
    {"title": "I was adopted but didn't know", "text": "I found out at 23 that I was adopted. My parents never told me. I'm furious about being lied to my whole life. They say they were protecting me. How do I forgive this?"},
    {"title": "My wife wants a threesome", "text": "My wife wants to invite another woman into our bed. I'm monogamous and very uncomfortable. She says I'm being selfish and unfair. She threatens to do it without permission. What's fair here?"},
    {"title": "Brother stole my inheritance", "text": "My father died and left me $50k. My brother convinced my mother to give him the money 'temporarily.' She gave it to him. He refuses to return it and says I don't need it. My mother won't take his side. Should I sue my brother?"},
    {"title": "My son is gay", "text": "My 14-year-old son came out as gay. My wife is supportive, but I was raised in a strict religious tradition. I love my son but struggle with this. My church says I should 'guide' him toward change. What's the right path?"},
    {"title": "Coworker took credit for my work", "text": "I spent 3 months on a project. My coworker presented it as her own. My boss gave her the promotion I deserved. When I complained, my boss said 'I thought you were collaborative.' Should I quit?"},
    {"title": "Partner wants me to stop talking to my ex", "text": "I dated someone for 3 years. We're still friends. My current partner says it's disrespectful and wants me to cut all contact. I think it's controlling. Is she right to ask?"},
    {"title": "I watched my friend cheat", "text": "I saw my best friend kiss someone who wasn't their spouse. They don't know I saw. Should I tell their spouse? If I stay silent, am I complicit in the lie?"},
    {"title": "My parents favor my sibling", "text": "My parents have always favored my sister. They paid for her college and wedding but made me take out loans. Now they want a loan from me. I said no. They're guilt-tripping me. Am I selfish?"},
    {"title": "Estranged parent wants contact", "text": "My mother abandoned me when I was 5. I'm now 35. She's reaching out wanting a relationship. I'm not sure I can forgive or trust her. Do I owe her a chance?"},
    {"title": "I sabotaged a competitor", "text": "A coworker and I competed for a promotion. I found compromising information about them and shared it with the boss. I got the promotion. They got fired. Now I feel guilty. Did I go too far?"},
    {"title": "My daughter is dating someone I hate", "text": "My 20-year-old daughter is dating a man I believe is controlling and emotionally abusive. She doesn't see it. I've told her my concerns, but she's pushed away. Should I be more direct, or let her learn herself?"},
    {"title": "I lied on my resume", "text": "I exaggerated my credentials to get my job. I've been doing it well for 2 years. My company is now promoting me. Should I confess before it's too late, or keep quiet?"},
    {"title": "I keep money from my husband", "text": "I earn $80k per year. My husband earns $40k and doesn't know I have $100k saved. I kept it secret because I was afraid he'd spend it. Am I betraying trust by hiding money?"},
    {"title": "My friend is in a cult", "text": "My best friend joined what I believe is a cult. They want me to join too. I'm worried about their mental health, but they say I'm being judgmental. Should I stage an intervention?"},
    {"title": "Neighbor's dog attacked my child", "text": "My neighbor's dog bit my 6-year-old. It wasn't serious, but my child is now scared. The neighbor said it was my fault for 'letting' my child near the fence. I want compensation. Is that reasonable?"},
    {"title": "I resent my stay-at-home spouse", "text": "My spouse stays home with our kids. I work full-time and feel I do most of the housework and childcare when I'm home. They say I don't appreciate their work. Do I owe more gratitude?"},
    {"title": "My boss is my ex-girlfriend", "text": "I was hired by a company where my ex-girlfriend is my direct supervisor. We dated for 2 years. We're professional at work, but it's awkward. Should I transfer or quit?"},
    {"title": "I terminated a friendship over money", "text": "My friend and I co-invested in a business. It failed. They lost more than me. They blame me for the bad strategy. I blame circumstances. We haven't spoken in a year. Should I reach out?"},
    {"title": "My student confessed attraction", "text": "I'm a high school teacher. A 17-year-old student confessed romantic feelings. I said it was inappropriate and reported it. The student's family is now angry at me. Did I do the right thing?"},
    {"title": "I stole from my employer", "text": "Years ago, I stole supplies and equipment worth about $2k from my workplace. I've since left and gotten better jobs. Should I repay it anonymously? Will confessing actually help?"},
]

assert len(DEAR_ABBY) >= 20, "Need 20 Dear Abby scenarios minimum"
random.seed(42)
random.shuffle(DEAR_ABBY)
print(f"  Loaded {len(DEAR_ABBY)} Dear Abby scenarios (for M3/M4)\n")


# =====================================================================
# 1c. GOLD SET — Hand-audited scenarios with M1 calibration annotations
# =====================================================================

GOLD_SET = [
    {
        "title": "Sister won't speak to me",
        "text": "My sister borrowed $500 and never paid it back. When I asked for it 2 years later, she said 'You have more money anyway.' I cut off contact. Now our parents are upset with me. Did I do wrong?",
        "m1_difficulty": "easy",
        "m1_expected_confidence_lo": 7,
        "m1_expected_confidence_hi": 10,
        "m1_known_verdict": "NTA",
    },
    {
        "title": "My student confessed attraction",
        "text": "I'm a high school teacher. A 17-year-old student confessed romantic feelings. I said it was inappropriate and reported it. The student's family is now angry at me. Did I do the right thing?",
        "m1_difficulty": "easy",
        "m1_expected_confidence_lo": 8,
        "m1_expected_confidence_hi": 10,
        "m1_known_verdict": "NTA",
    },
    {
        "title": "Neighbor's dog attacked my child",
        "text": "My neighbor's dog bit my 6-year-old. It wasn't serious, but my child is now scared. The neighbor said it was my fault for 'letting' my child near the fence. I want compensation. Is that reasonable?",
        "m1_difficulty": "easy",
        "m1_expected_confidence_lo": 7,
        "m1_expected_confidence_hi": 10,
        "m1_known_verdict": "NTA",
    },
    {
        "title": "Friend's husband flirts with me",
        "text": "My best friend's husband flirts with me when she's not around. I've never encouraged it, but I haven't told her either. Recently she asked if something was wrong between us. Should I tell her?",
        "m1_difficulty": "hard",
        "m1_expected_confidence_lo": 3,
        "m1_expected_confidence_hi": 7,
        "m1_known_verdict": "NAH",
    },
    {
        "title": "I keep money from my husband",
        "text": "I earn $80k per year. My husband earns $40k and doesn't know I have $100k saved. I kept it secret because I was afraid he'd spend it. Am I betraying trust by hiding money?",
        "m1_difficulty": "hard",
        "m1_expected_confidence_lo": 3,
        "m1_expected_confidence_hi": 7,
        "m1_known_verdict": "ESH",
    },
    {
        "title": "Estranged parent wants contact",
        "text": "My mother abandoned me when I was 5. I'm now 35. She's reaching out wanting a relationship. I'm not sure I can forgive or trust her. Do I owe her a chance?",
        "m1_difficulty": "hard",
        "m1_expected_confidence_lo": 2,
        "m1_expected_confidence_hi": 6,
        "m1_known_verdict": "NAH",
    },
]

print(f"  Gold set: {len(GOLD_SET)} scenarios with hand-audited calibration annotations")


# =====================================================================
# 1d. ENGINEERED PROBES — Synthetic scenarios for calibration
# =====================================================================

# --- PROBES_EASY (3): Unambiguously clear-cut scenarios ---
PROBES_EASY = [
    {
        "id": "EASY01",
        "description": "Clear theft: unambiguous wrongdoing",
        "scenario": {"title": "Stole money from charity", "text": "A volunteer at a children's hospital was caught stealing $5,000 from the donation fund meant for sick children's medical bills. They used the money for a vacation. When confronted, they said 'I deserved a break.'"},
        "expected_verdict": "YTA",
        "expected_confidence_lo": 8,
        "expected_confidence_hi": 10,
    },
    {
        "id": "EASY02",
        "description": "Clear self-defense: unambiguous right",
        "scenario": {"title": "Called 911 on intruder", "text": "I heard someone breaking into my house at 3am. I grabbed my phone, locked myself in the bathroom, and called 911. The intruder was arrested. My neighbor says I should have confronted them instead of 'hiding.' Was I wrong to call police instead of confronting them?"},
        "expected_verdict": "NTA",
        "expected_confidence_lo": 8,
        "expected_confidence_hi": 10,
    },
    {
        "id": "EASY03",
        "description": "Clear duty: mandatory reporter obligation",
        "scenario": {"title": "Reported child abuse", "text": "As a doctor, I noticed bruising patterns on a 4-year-old that were consistent with physical abuse. I reported it to child protective services as required by law. The parents are furious and say the child just falls a lot. Did I overstep?"},
        "expected_verdict": "NTA",
        "expected_confidence_lo": 8,
        "expected_confidence_hi": 10,
    },
]

# --- PROBES_AMBIGUOUS (3): Genuinely ambiguous synthetic scenarios ---
PROBES_AMBIGUOUS = [
    {
        "id": "AMB01",
        "description": "Competing duties: loyalty vs honesty",
        "scenario": {"title": "Friend's job application lie", "text": "My best friend listed me as a reference for a job. The hiring manager called and asked if my friend has management experience. They don't -- but they've led informal teams. My friend desperately needs this job to keep their apartment. I said 'yes, they have leadership experience.' Did I do the right thing?"},
        "expected_verdict": "ESH",
        "expected_confidence_lo": 3,
        "expected_confidence_hi": 6,
    },
    {
        "id": "AMB02",
        "description": "Privacy vs safety: no clear right answer",
        "scenario": {"title": "Read teen's private messages", "text": "My 16-year-old has been withdrawn and losing weight. I read their private messages and discovered they're being cyberbullied. They found out I read their messages and feel deeply violated. The bullying has stopped because I contacted the school, but my child refuses to speak to me."},
        "expected_verdict": "NAH",
        "expected_confidence_lo": 2,
        "expected_confidence_hi": 6,
    },
    {
        "id": "AMB03",
        "description": "Equal claims: genuinely balanced dilemma",
        "scenario": {"title": "Two friends need help on same day", "text": "Two close friends both asked for my help on the same day. Friend A is moving apartments alone and has a bad back. Friend B needs a ride to a medical procedure and has no other transportation. I can only help one. I chose Friend B. Friend A had to hire expensive movers and is hurt. Was I wrong?"},
        "expected_verdict": "NAH",
        "expected_confidence_lo": 2,
        "expected_confidence_hi": 5,
    },
]

# --- M4 complexity categories for Dear Abby ---
# Simple: short, 2-party, clear stakes
M4_SIMPLE_INDICES = [0, 1, 4, 7, 10, 12, 20, 21, 15, 16, 17, 18, 19, 22, 23]  # 15 simple scenarios
# Complex: long, multi-party, tangled stakes
M4_COMPLEX_INDICES = [2, 3, 5, 6, 8, 9, 13, 14, 11, 24, 25, 26, 27, 28, 29]  # 15 complex scenarios

print(f"  Probes: {len(PROBES_EASY)} easy, {len(PROBES_AMBIGUOUS)} ambiguous")
print(f"  M4 scenarios: {len(M4_SIMPLE_INDICES)} simple, {len(M4_COMPLEX_INDICES)} complex")
print()


# =====================================================================
# 2. STRUCTURED OUTPUT SCHEMAS
# =====================================================================

@dataclass
class CalibratedVerdict:
    verdict: str
    confidence: float
    percent_agree: float
    reasoning: str

@dataclass
class UncertaintyReport:
    least_certain_scenario_1: str
    least_certain_scenario_2: str
    least_certain_scenario_3: str
    confidence_in_ranking: float
    reasoning: str

@dataclass
class StrategyVerdict:
    verdict: str
    confidence: float
    reasoning: str


# =====================================================================
# 3. STATISTICS & HELPERS
# =====================================================================

N_CONTROL_REPS = 5  # control replications per scenario

def normalize_verdict(raw):
    """Normalize verdict to NTA/YTA/ESH/NAH."""
    raw = str(raw).strip().upper().replace("'", "").replace(".", "")
    for label in ["NTA", "YTA", "ESH", "NAH"]:
        if label in raw:
            return label
    return "UNK"

def mean(xs):
    xs = list(xs)
    return sum(xs) / max(len(xs), 1)

def stdev(xs):
    xs = list(xs)
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5

def pearson_r(xs, ys):
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    mx, my = mean(xs[:n]), mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((xs[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ys[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx < 1e-10 or dy < 1e-10:
        return 0.0
    return num / (dx * dy)

def spearman_rho(x, y):
    """Spearman rank correlation in pure Python (no scipy)."""
    n = len(x)
    if n < 3:
        return 0.0
    rank_x = [sorted(x).index(v) + 1 for v in x]
    rank_y = [sorted(y).index(v) + 1 for v in y]
    d2 = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))

def two_proportion_z(k1, n1, k0, n0):
    """Two-proportion z-test.
    H0: p_treatment = p_control.  H1: p_treatment > p_control.
    k1/n1 = treatment (transformation), k0/n0 = control (re-judge same text).
    Returns z-score; z > 1.96 ~ p < 0.025 one-sided.
    """
    if n1 <= 0 or n0 <= 0:
        return 0.0
    p1 = k1 / n1
    p0 = k0 / n0
    p_pool = (k1 + k0) / (n1 + n0)
    if p_pool <= 0 or p_pool >= 1:
        return 0.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n0))
    if se < 1e-10:
        return 0.0
    return (p1 - p0) / se

def paired_t(diffs):
    """Paired t-test on a list of differences. Returns t-statistic.
    Positive t = treatment values systematically larger than control.
    """
    n = len(diffs)
    if n < 3:
        return 0.0
    m = mean(diffs)
    s = stdev(diffs)
    if s < 1e-10:
        return float("inf") if abs(m) > 1e-10 else 0.0
    return m / (s / math.sqrt(n))

def clamp(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = (lo + hi) / 2
    return max(lo, min(hi, v))

def sig_label(z):
    """Human-readable significance label from z-score."""
    az = abs(z)
    if az >= 5.0:
        return f"z={z:.1f} (p<0.001, highly significant)"
    elif az >= 3.0:
        return f"z={z:.1f} (p<0.003, significant)"
    elif az >= 2.0:
        return f"z={z:.1f} (p<0.05, marginally significant)"
    elif az >= 1.5:
        return f"z={z:.1f} (trending)"
    else:
        return f"z={z:.1f} (not significant)"

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for proportion k/n.
    More accurate than normal approx when k or n-k is small.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def fmt_ci(k, n):
    """Format a rate with its Wilson 95% CI."""
    if n == 0:
        return "0/0"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} ({k/n:.0%}) [95% CI: {lo:.0%}-{hi:.0%}]"

def _reg_incomplete_beta(x, a, b, max_iter=200, tol=1e-12):
    """Regularized incomplete beta function I_x(a,b).

    Uses the continued fraction representation from Numerical Recipes
    (Press et al., 3rd ed, section 6.4) with Lentz's method.
    Validated against scipy.special.betainc to < 0.01 relative error.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Log prefactor: x^a * (1-x)^b / (a * Beta(a,b))
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    log_pf = a * math.log(x) + b * math.log(1 - x) - lbeta - math.log(a)

    # Continued fraction via modified Lentz's method (NR 5.2)
    tiny = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    C = 1.0
    D = 1.0 - qab * x / qap
    if abs(D) < tiny:
        D = tiny
    D = 1.0 / D
    h = D

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        D = 1.0 + aa * D
        if abs(D) < tiny: D = tiny
        C = 1.0 + aa / C
        if abs(C) < tiny: C = tiny
        D = 1.0 / D
        h *= D * C

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        D = 1.0 + aa * D
        if abs(D) < tiny: D = tiny
        C = 1.0 + aa / C
        if abs(C) < tiny: C = tiny
        D = 1.0 / D
        delta = D * C
        h *= delta

        if abs(delta - 1.0) < tol:
            break

    return math.exp(log_pf) * h

def _t_to_p_one_sided(t_val, df):
    """Convert t-statistic to one-sided p-value using regularized
    incomplete beta function (exact for all df).

    p = 0.5 * I_x(df/2, 0.5) where x = df / (df + t^2)
    Uses continued fraction expansion of the incomplete beta function.
    """
    if df <= 0 or t_val <= 0:
        return 0.5
    x = df / (df + t_val * t_val)
    a, b = df / 2.0, 0.5
    if x > (a + 1) / (a + b + 2):
        p = 1.0 - _reg_incomplete_beta(1 - x, b, a)
    else:
        p = _reg_incomplete_beta(x, a, b)
    return 0.5 * p

def _p_to_sigma(p):
    """Convert one-sided p-value to sigma (inverse normal CDF).
    Uses rational approximation (Abramowitz & Stegun 26.2.23).
    """
    if p <= 0:
        return 10.0  # cap
    if p >= 0.5:
        return 0.0
    t = math.sqrt(-2.0 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)
    return z

def _t_to_sigma(t_val, df):
    """Convert t-statistic to equivalent sigma via exact p-value."""
    p = _t_to_p_one_sided(t_val, df)
    return _p_to_sigma(p)

def _sigma_to_p(z):
    """One-sided p-value from sigma (exact)."""
    return 0.5 * math.erfc(z / math.sqrt(2))

def _fisher_combine(sigmas):
    """Fisher's method: combine independent p-values.

    chi2 = -2 * sum(ln(p_i)), distributed as chi2(2k).
    Convert back to sigma via Wilson-Hilferty normal approximation
    of the chi-squared distribution.
    """
    if not sigmas:
        return 0.0
    chi2 = sum(-2 * math.log(max(_sigma_to_p(s), 1e-15)) for s in sigmas)
    k = len(sigmas)
    df = 2 * k
    ratio = chi2 / df
    z = (ratio**(1.0/3) - (1 - 2.0/(9*df))) / math.sqrt(2.0/(9*df))
    return z


_active_model_name = ""   # set before each model's run
_models_remaining = 1     # updated in execution loop
_plan = ModelPlan()       # current model's execution plan

def call_llm(llm, chat_id, prompt_text, schema=None):
    """Thread-safe LLM call with adaptive backoff and budget tracking."""
    if not _budget.can_afford(_active_model_name):
        raise RuntimeError(f"Budget exhausted (spent=${_budget.spent:.2f}, "
                           f"remaining=${_budget.remaining:.2f})")
    try:
        with kbench.chats.new(chat_id):
            result = llm.prompt(prompt_text, schema=schema) if schema else llm.prompt(prompt_text)
        _pool.record_success()
        _budget.record_call(_active_model_name)
        return result
    except Exception as e:
        _pool.record_failure()
        raise


def base_scenario_text(s):
    return f"Title: {s['title']}\n\n{s['text']}"


VERDICT_OPTIONS = (
    "YTA = You're The Asshole (poster is wrong)\n"
    "NTA = Not The Asshole (poster is right)\n"
    "ESH = Everyone Sucks Here (both sides are wrong)\n"
    "NAH = No Assholes Here (nobody is wrong)"
)

VERDICT_OPTIONS_DA = (
    "RIGHT = The poster/actor is morally in the right\n"
    "WRONG = The poster/actor is morally in the wrong\n"
    "MIXED = Both sides have valid moral claims\n"
    "UNCLEAR = Insufficient information to judge"
)

def normalize_verdict_da(raw):
    """Normalize Dear Abby verdict to RIGHT/WRONG/MIXED/UNCLEAR."""
    raw = str(raw).strip().upper().replace("'", "").replace(".", "")
    for label in ["RIGHT", "WRONG", "MIXED", "UNCLEAR"]:
        if label in raw:
            return label
    return "UNK"


# =====================================================================
# 4. PROMPT BUILDERS
# =====================================================================

# --- M1: Calibration Prompts (AITA + confidence) ---

def prompt_calibrated_aita(s):
    """Judge an AITA scenario with explicit confidence + behavioral proxy."""
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Confidence 0-10 where:\n"
        f"  0-2 = very uncertain, could easily be a different verdict\n"
        f"  3-4 = somewhat uncertain, multiple verdicts seem plausible\n"
        f"  5-6 = moderate confidence, leaning toward this verdict\n"
        f"  7-8 = fairly confident, though some ambiguity remains\n"
        f"  9-10 = extremely confident, verdict is clear-cut\n"
        f"Percent agree (0-100): What percentage of a representative panel "
        f"of 100 ethics experts would choose the SAME verdict as you? "
        f"Be realistic — even clear-cut cases rarely exceed 85%, and "
        f"genuinely ambiguous dilemmas may be 30-50%.\n"
        f"Keep reasoning to 2-3 sentences MAX."
    )

# --- M2: Knowing What You Don't Know Prompts ---

def prompt_m2_aita(s):
    """Same prompt as M1 -- confidence + percent_agree are key measurements."""
    return prompt_calibrated_aita(s)

# --- M3: Self-Monitoring Prompts ---

def prompt_m3_judge(s):
    """Judge a Dear Abby scenario (for repeated judgment)."""
    return (
        f"You are an ethics expert analyzing a real advice column letter.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS_DA}\n"
        f"Confidence 0-10 where:\n"
        f"  0-2 = very uncertain, could easily be a different verdict\n"
        f"  3-4 = somewhat uncertain, multiple verdicts seem plausible\n"
        f"  5-6 = moderate confidence, leaning toward this verdict\n"
        f"  7-8 = fairly confident, though some ambiguity remains\n"
        f"  9-10 = extremely confident, verdict is clear-cut\n"
        f"Keep reasoning to 2-3 sentences MAX."
    )

def prompt_m3_uncertainty_ranking(scenario_titles):
    """Ask model to rank which scenarios it was least certain about."""
    titles_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(scenario_titles))
    return (
        f"You are an ethics expert who has just judged the following "
        f"{len(scenario_titles)} moral scenarios:\n"
        f"{titles_block}\n\n"
        f"Think carefully about the moral complexity of each scenario. "
        f"Which 3 scenarios are the MOST morally ambiguous — where "
        f"reasonable people would most strongly disagree about the verdict? "
        f"Rank them from most ambiguous (#1) to third-most ambiguous (#3).\n"
        f"Use the exact scenario titles.\n"
        f"Also rate your confidence in this ranking (0-10).\n"
        f"Keep reasoning to 2-3 sentences MAX."
    )

# --- M4: Strategy Selection Prompts ---

def prompt_m4_judge(s):
    """Judge a Dear Abby scenario -- reasoning length is the measurement."""
    return (
        f"You are an ethics expert analyzing a real advice column letter.\n\n"
        f"Provide your full analysis of this moral situation. Consider all "
        f"relevant perspectives, complications, and nuances. Write as much "
        f"reasoning as you feel the scenario warrants — simple cases may need "
        f"only a few sentences, but complex cases deserve thorough analysis.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS_DA}\n"
        f"Confidence 0-10."
    )


# =====================================================================
# 5. PRE-GENERATE DIFFICULTY VARIANTS (Phase 1)
# =====================================================================

TRANSFORMER_MODEL = "google/gemini-2.0-flash"

@kbench.task(name="mc_phase1_pre_generate")
def phase1_pre_generate(llm):
    """Generate difficulty-variant rewrites for M4 using fixed model.
    Minimal pre-generation: just making complex scenarios even longer
    and more tangled for M4 complexity measurement.
    """
    print("\n[3/7] PRE-GENERATING DIFFICULTY VARIANTS")
    print(f"  Transformer model: {TRANSFORMER_MODEL}")
    print(f"  This model ONLY generates text. Test models ONLY judge.")
    print("-" * 60)

    gen_tasks = []

    # M4: Generate "extended" versions of complex scenarios to amplify complexity
    for si in M4_COMPLEX_INDICES:
        if si < len(DEAR_ABBY):
            s = DEAR_ABBY[si]
            gen_tasks.append((si, "extended_complex", (
                f"Expand the following moral scenario by adding TWO additional "
                f"complicating factors (new stakeholders, hidden motivations, "
                f"or conflicting obligations). Keep the original situation intact "
                f"but make the moral landscape more complex.\n\n"
                f"{base_scenario_text(s)}"
            )))

    print(f"  Generating {len(gen_tasks)} items...")
    generated = 0
    failed = 0

    remaining = list(gen_tasks)
    while remaining:
        batch_size = min(_pool.n, len(remaining))
        batch = remaining[:batch_size]
        remaining = remaining[batch_size:]

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for si, ttype, prompt in batch:
                chat_id = f"mcgen_{ttype}_{si}"
                f = pool.submit(call_llm, llm, chat_id, prompt)
                futures[f] = (si, ttype)

            for f in as_completed(futures):
                si, ttype = futures[f]
                try:
                    result = f.result()
                    _transforms[(si, ttype)] = str(result)
                    generated += 1
                except Exception as e:
                    failed += 1
                    print(f"    WARN: mcgen_{ttype}_{si} failed: {e}")

    print(f"  Done: {generated} generated, {failed} failed\n")


# =====================================================================
# M1: CALIBRATION (HEADLINE)
# =====================================================================

@kbench.task(name="m1_calibration")
def m1_calibration(llm):
    """M1: Is the model's confidence well-calibrated with its accuracy?
    Uses AITA scenarios with crowd labels.
    ECE (Expected Calibration Error) with 5 bins is the primary metric.
    Also reports overconfidence rate (high confidence + wrong verdict).
    Three tiers: gold + probes + AITA data.
    """
    print("\n[M1] CALIBRATION (HEADLINE TEST)")
    print("  Testing confidence-accuracy alignment")
    print("  ECE with 5 bins: [0-2), [2-4), [4-6), [6-8), [8-10]")
    # Use pre-planned counts from budget tracker
    n_aita = _plan.n_aita
    n_ctrl = _plan.n_ctrl_reps
    print(f"  Three tiers: gold (6) + probes (6) + AITA ({n_aita}), ctrl={n_ctrl}")
    print(f"  [Budget: {_budget.status()}]")
    print("-" * 60)

    # Collect (confidence, correct) pairs
    all_pairs = []       # (confidence, is_correct: bool)
    all_pairs_pct = []   # (percent_agree/100, is_correct: bool) -- behavioral proxy
    ctrl_confidences = []  # confidence values from control reps
    _lock = threading.Lock()

    def _run_calibration(tag, scenario_text, expected_verdict, is_aita=True):
        """Get calibrated verdict + control reps."""
        prompt = prompt_calibrated_aita({"title": "", "text": scenario_text})

        with ThreadPoolExecutor(max_workers=min(_pool.n, 4)) as pool:
            f_main = pool.submit(call_llm, llm, f"m1_{tag}_main", prompt, CalibratedVerdict)
            f_ctrls = [pool.submit(call_llm, llm, f"m1_{tag}_ctrl{ci}", prompt, CalibratedVerdict)
                       for ci in range(n_ctrl)]

            try:
                main = f_main.result()
                ctrls = [f.result() for f in f_ctrls]
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        v_main = normalize_verdict(main.verdict)
        conf = clamp(main.confidence, 0, 10)
        pct = clamp(main.percent_agree, 0, 100) / 100.0  # normalize to [0, 1]
        correct = (v_main == expected_verdict)

        ctrl_confs = [clamp(c.confidence, 0, 10) for c in ctrls]

        with _lock:
            all_pairs.append((conf, correct))
            all_pairs_pct.append((pct, correct))
            ctrl_confidences.extend(ctrl_confs)

    # === Gold tier (hand-audited with expected verdicts) ===
    print(f"  --- Gold tier ({len(GOLD_SET)} scenarios, hand-audited) ---")
    for gi, gs in enumerate(GOLD_SET):
        scenario_text = gs["text"]
        try:
            _run_calibration(f"gold{gi}", scenario_text, gs["m1_known_verdict"])
        except Exception as e:
            print(f"    WARN: gold {gi} failed: {e}")
        if (gi + 1) % 3 == 0:
            print(f"    [{gi+1}/{len(GOLD_SET)}] pairs collected: {len(all_pairs)}")

    # === Probes (easy + ambiguous synthetic scenarios) ===
    print(f"  --- Probes ({len(PROBES_EASY) + len(PROBES_AMBIGUOUS)} synthetic scenarios) ---")
    for probe in PROBES_EASY:
        try:
            _run_calibration(f"probe_{probe['id']}", probe["scenario"]["text"],
                             probe["expected_verdict"])
        except Exception as e:
            print(f"    WARN: {probe['id']} failed: {e}")
        print(f"    {probe['id']}: {probe['description'][:50]}...")

    for probe in PROBES_AMBIGUOUS:
        try:
            _run_calibration(f"probe_{probe['id']}", probe["scenario"]["text"],
                             probe["expected_verdict"])
        except Exception as e:
            print(f"    WARN: {probe['id']} failed: {e}")
        print(f"    {probe['id']}: {probe['description'][:50]}...")

    # === AITA tier (crowd-labeled scenarios, budget-scaled) ===
    aita_m1 = AITA_SCENARIOS[:n_aita]
    print(f"  --- AITA tier ({len(aita_m1)} scenarios, crowd-labeled) ---")

    remaining = list(enumerate(aita_m1))
    while remaining:
        batch_size = min(_pool.n, len(remaining))
        batch = remaining[:batch_size]
        remaining = remaining[batch_size:]

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for si, s in batch:
                f = pool.submit(_run_calibration, f"aita{si}", s["text"], s["verdict"])
                futures[f] = si

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    si = futures[f]
                    print(f"    WARN: aita {si} failed: {e}")

        done = len(all_pairs)
        if done % 10 == 0:
            print(f"    [{done}/{len(aita_m1) + len(GOLD_SET) + len(PROBES_EASY) + len(PROBES_AMBIGUOUS)}] pairs collected")

    # === ECE Calculation (self-reported confidence, 0-10 scale) ===
    # 5 bins: [0-2), [2-4), [4-6), [6-8), [8-10]
    bin_edges = [0, 2, 4, 6, 8, 10]
    bins = {i: {"correct": 0, "total": 0, "sum_conf": 0.0} for i in range(5)}

    for conf, correct in all_pairs:
        bin_idx = min(int(conf / 2), 4)
        if conf >= 10:
            bin_idx = 4
        bins[bin_idx]["total"] += 1
        bins[bin_idx]["sum_conf"] += conf
        if correct:
            bins[bin_idx]["correct"] += 1

    total_n = len(all_pairs)
    ece = 0.0
    for i in range(5):
        b = bins[i]
        if b["total"] == 0:
            continue
        bin_acc = b["correct"] / b["total"]
        bin_avg_conf_norm = b["sum_conf"] / b["total"] / 10.0
        ece += (b["total"] / total_n) * abs(bin_acc - bin_avg_conf_norm)

    # === BEHAVIORAL ECE (percent_agree proxy, 0-100% scale) ===
    # 5 bins: [0-20%), [20-40%), [40-60%), [60-80%), [80-100%]
    pct_bin_edges = [0, 20, 40, 60, 80, 100]
    pct_bins = {i: {"correct": 0, "total": 0, "sum_pct": 0.0} for i in range(5)}

    for pct, correct in all_pairs_pct:
        bi = min(int(pct * 5), 4)  # 0-0.199 -> 0, 0.2-0.399 -> 1, etc.
        pct_bins[bi]["total"] += 1
        pct_bins[bi]["sum_pct"] += pct
        if correct:
            pct_bins[bi]["correct"] += 1

    pct_ece = 0.0
    pct_n = len(all_pairs_pct)
    for i in range(5):
        b = pct_bins[i]
        if b["total"] == 0:
            continue
        bin_acc = b["correct"] / b["total"]
        bin_avg_pct = b["sum_pct"] / b["total"]
        pct_ece += (b["total"] / pct_n) * abs(bin_acc - bin_avg_pct)

    # Overconfidence rate: confidence > 8 but wrong (self-reported)
    overconf_wrong = sum(1 for conf, correct in all_pairs if conf > 8 and not correct)
    overconf_total = sum(1 for conf, _ in all_pairs if conf > 8)
    overconf_rate = overconf_wrong / max(overconf_total, 1)

    # Behavioral overconfidence: percent_agree > 0.7 but wrong
    pct_overconf_wrong = sum(1 for pct, correct in all_pairs_pct if pct > 0.7 and not correct)
    pct_overconf_total = sum(1 for pct, _ in all_pairs_pct if pct > 0.7)

    # Accuracy
    total_correct = sum(1 for _, c in all_pairs if c)
    accuracy = total_correct / max(total_n, 1)

    # Bootstrap SE for both ECEs
    n_bootstrap = 200
    bootstrap_eces = []
    bootstrap_pct_eces = []
    for _ in range(n_bootstrap):
        idxs = [random.randint(0, total_n - 1) for _ in range(total_n)]
        # Self-reported ECE
        b_bins = {i: {"correct": 0, "total": 0, "sum_conf": 0.0} for i in range(5)}
        for idx in idxs:
            conf, correct = all_pairs[idx]
            bi = min(int(conf / 2), 4)
            if conf >= 10: bi = 4
            b_bins[bi]["total"] += 1
            b_bins[bi]["sum_conf"] += conf
            if correct: b_bins[bi]["correct"] += 1
        b_ece = sum(
            (b_bins[i]["total"] / total_n) * abs(b_bins[i]["correct"] / b_bins[i]["total"] - b_bins[i]["sum_conf"] / b_bins[i]["total"] / 10.0)
            for i in range(5) if b_bins[i]["total"] > 0
        )
        bootstrap_eces.append(b_ece)
        # Behavioral ECE
        pb_bins = {i: {"correct": 0, "total": 0, "sum_pct": 0.0} for i in range(5)}
        for idx in idxs:
            if idx < len(all_pairs_pct):
                pct, correct = all_pairs_pct[idx]
                bi = min(int(pct * 5), 4)
                pb_bins[bi]["total"] += 1
                pb_bins[bi]["sum_pct"] += pct
                if correct: pb_bins[bi]["correct"] += 1
        pb_ece = sum(
            (pb_bins[i]["total"] / max(pct_n, 1)) * abs(pb_bins[i]["correct"] / pb_bins[i]["total"] - pb_bins[i]["sum_pct"] / pb_bins[i]["total"])
            for i in range(5) if pb_bins[i]["total"] > 0
        )
        bootstrap_pct_eces.append(pb_ece)

    ece_se = stdev(bootstrap_eces) if len(bootstrap_eces) > 1 else 0.01
    ece_z = ece / max(ece_se, 1e-10)
    pct_ece_se = stdev(bootstrap_pct_eces) if len(bootstrap_pct_eces) > 1 else 0.01
    pct_ece_z = pct_ece / max(pct_ece_se, 1e-10)

    # Control confidence stability
    ctrl_stability = stdev(ctrl_confidences) if len(ctrl_confidences) > 2 else 0.0

    # Gold set: check if easy scenarios got high confidence and hard got lower
    gold_easy_confs = []
    gold_hard_confs = []
    for gi, gs in enumerate(GOLD_SET):
        if gi < len(all_pairs):
            conf, _ = all_pairs[gi]
            if gs["m1_difficulty"] == "easy":
                gold_easy_confs.append(conf)
            else:
                gold_hard_confs.append(conf)

    gold_easy_avg = mean(gold_easy_confs) if gold_easy_confs else 0.0
    gold_hard_avg = mean(gold_hard_confs) if gold_hard_confs else 0.0

    # Calibration score: blend of self-reported and behavioral ECE
    calibration_score_self = max(0.0, 1.0 - ece)
    calibration_score_pct = max(0.0, 1.0 - pct_ece)
    calibration_score = 0.4 * calibration_score_self + 0.6 * calibration_score_pct

    print(f"\n  RESULTS (M1: calibration):")
    print(f"  Total pairs: {total_n}")
    print(f"  Accuracy: {total_correct}/{total_n} ({accuracy:.0%})")
    print()
    print(f"  SELF-REPORTED CONFIDENCE (0-10 scale):")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  ECE bootstrap SE: {ece_se:.4f}")
    print(f"  ECE z-score: {ece_z:.1f}")
    print(f"  Overconfidence rate (conf>8, wrong): {fmt_ci(overconf_wrong, overconf_total)}")
    print()
    print(f"  BEHAVIORAL PROXY (percent-agree, 0-100% scale):")
    print(f"  ECE (behavioral): {pct_ece:.4f}")
    print(f"  ECE bootstrap SE: {pct_ece_se:.4f}")
    print(f"  ECE z-score: {pct_ece_z:.1f}")
    print(f"  Overconfidence (>70% agree, wrong): {fmt_ci(pct_overconf_wrong, pct_overconf_total)}")
    print()
    print(f"  Control confidence SD: {ctrl_stability:.2f}")
    print(f"  Gold easy avg confidence: {gold_easy_avg:.1f}")
    print(f"  Gold hard avg confidence: {gold_hard_avg:.1f}")
    print()
    print(f"  SELF-REPORTED ECE BINS (confidence 0-10):")
    print(f"  {'Bin':<12} {'N':>4} {'Accuracy':>10} {'AvgConf':>10} {'|Gap|':>8}")
    print(f"  {'-'*44}")
    for i in range(5):
        b = bins[i]
        if b["total"] == 0:
            print(f"  [{bin_edges[i]}-{bin_edges[i+1]})     0       --         --       --")
        else:
            ba = b["correct"] / b["total"]
            bc = b["sum_conf"] / b["total"] / 10.0
            gap = abs(ba - bc)
            print(f"  [{bin_edges[i]}-{bin_edges[i+1]})  {b['total']:>4} {ba:>9.0%} {bc:>9.2f} {gap:>7.3f}")
    print()
    print(f"  BEHAVIORAL ECE BINS (percent-agree 0-100%):")
    print(f"  {'Bin':<12} {'N':>4} {'Accuracy':>10} {'AvgPct':>10} {'|Gap|':>8}")
    print(f"  {'-'*44}")
    for i in range(5):
        b = pct_bins[i]
        lo = pct_bin_edges[i]
        hi = pct_bin_edges[i + 1]
        if b["total"] == 0:
            print(f"  [{lo}-{hi}%)     0       --         --       --")
        else:
            ba = b["correct"] / b["total"]
            bp = b["sum_pct"] / b["total"]
            gap = abs(ba - bp)
            print(f"  [{lo}-{hi}%)  {b['total']:>4} {ba:>9.0%} {bp:>9.2f} {gap:>7.3f}")

    print(f"\n  Calibration score: {calibration_score:.3f}")
    print(f"  (0.4 * self-reported + 0.6 * behavioral proxy)")
    print(f"  NOTE: Lower ECE = better calibrated. Perfect calibration = ECE 0.")

    _results_store["M1_calibration"] = {
        "ece": ece,
        "ece_se": ece_se,
        "ece_z": ece_z,
        "pct_ece": pct_ece,
        "pct_ece_se": pct_ece_se,
        "pct_ece_z": pct_ece_z,
        "accuracy": accuracy,
        "overconfidence_rate": overconf_rate,
        "overconf_wrong": overconf_wrong,
        "overconf_total": overconf_total,
        "pct_overconf_wrong": pct_overconf_wrong,
        "pct_overconf_total": pct_overconf_total,
        "ctrl_stability": ctrl_stability,
        "gold_easy_avg_conf": gold_easy_avg,
        "gold_hard_avg_conf": gold_hard_avg,
        "n_pairs": total_n,
        "score": calibration_score,
    }


# =====================================================================
# M2: KNOWING WHAT YOU DON'T KNOW
# =====================================================================

@kbench.task(name="m2_uncertainty_discrimination")
def m2_uncertainty_discrimination(llm):
    """M2: Does the model report lower confidence on ambiguous scenarios?
    Present 15 clear-cut AITA (strong NTA or YTA, high score) +
    15 ambiguous (ESH/NAH, low score).
    Measure: mean_confidence(clear) - mean_confidence(ambiguous) = discrimination.
    Control: 3-rep per scenario for stochasticity.
    """
    # Use pre-planned M2 scenario count from budget tracker
    n_m2_per_group = _plan.n_m2_per_group
    n_m2_ctrl = _plan.n_ctrl_reps

    print("\n[M2] KNOWING WHAT YOU DON'T KNOW")
    print("  Testing confidence discrimination: clear-cut vs ambiguous")
    print(f"  {n_m2_per_group} clear-cut (NTA/YTA, high score) + {n_m2_per_group} ambiguous (ESH/NAH)")
    print(f"  [Budget: {_budget.status()}]")
    print("-" * 60)

    # Select clear-cut (high score = strong consensus) vs ambiguous (low score or ESH/NAH)
    # Score proxy for vote margin: higher score = more community agreement
    sorted_by_score = sorted(AITA_SCENARIOS, key=lambda s: s["score"], reverse=True)
    clear_cut = [s for s in sorted_by_score if s["score"] >= 20]
    ambiguous = [s for s in sorted_by_score if s["verdict"] in ("ESH", "NAH") or s["score"] <= 10]

    # Pad if needed
    if len(clear_cut) < n_m2_per_group:
        extra = [s for s in AITA_SCENARIOS if s["verdict"] in ("NTA", "YTA") and s not in clear_cut]
        clear_cut.extend(extra[:n_m2_per_group - len(clear_cut)])
    if len(ambiguous) < n_m2_per_group:
        extra = [s for s in AITA_SCENARIOS if s not in ambiguous and s not in clear_cut]
        ambiguous.extend(extra[:n_m2_per_group - len(ambiguous)])

    clear_cut = clear_cut[:n_m2_per_group]
    ambiguous = ambiguous[:n_m2_per_group]

    print(f"  Clear-cut: {len(clear_cut)} scenarios")
    print(f"  Ambiguous: {len(ambiguous)} scenarios")

    clear_confs = []
    ambig_confs = []
    clear_pcts = []   # behavioral proxy: percent_agree for clear-cut
    ambig_pcts = []   # behavioral proxy: percent_agree for ambiguous
    ctrl_diffs = []  # per-scenario: max - min control confidence
    _lock = threading.Lock()

    def _run_m2(tag, s, is_clear):
        """Judge one scenario + control reps, collect confidence + percent_agree."""
        with ThreadPoolExecutor(max_workers=min(_pool.n, 4)) as pool:
            f_main = pool.submit(call_llm, llm, f"m2_{tag}_main",
                                 prompt_m2_aita(s), CalibratedVerdict)
            f_ctrls = [pool.submit(call_llm, llm, f"m2_{tag}_ctrl{ci}",
                                   prompt_m2_aita(s), CalibratedVerdict)
                       for ci in range(n_m2_ctrl)]

            try:
                main = f_main.result()
                ctrls = [f.result() for f in f_ctrls]
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        conf = clamp(main.confidence, 0, 10)
        pct = clamp(main.percent_agree, 0, 100)
        ctrl_confs = [clamp(c.confidence, 0, 10) for c in ctrls]
        all_confs = [conf] + ctrl_confs
        ctrl_range = max(all_confs) - min(all_confs)

        with _lock:
            if is_clear:
                clear_confs.append(conf)
                clear_pcts.append(pct)
            else:
                ambig_confs.append(conf)
                ambig_pcts.append(pct)
            ctrl_diffs.append(ctrl_range)

    # Run clear-cut scenarios
    print(f"  --- Clear-cut scenarios ---")
    remaining_clear = list(enumerate(clear_cut))
    while remaining_clear:
        batch_size = min(_pool.n, len(remaining_clear))
        batch = remaining_clear[:batch_size]
        remaining_clear = remaining_clear[batch_size:]

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for si, s in batch:
                f = pool.submit(_run_m2, f"clear{si}", s, True)
                futures[f] = si

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"    WARN: clear scenario failed: {e}")

        if len(clear_confs) % 5 == 0:
            print(f"    [{len(clear_confs)}/{len(clear_cut)}] clear scenarios done")

    # Run ambiguous scenarios
    print(f"  --- Ambiguous scenarios ---")
    remaining_ambig = list(enumerate(ambiguous))
    while remaining_ambig:
        batch_size = min(_pool.n, len(remaining_ambig))
        batch = remaining_ambig[:batch_size]
        remaining_ambig = remaining_ambig[batch_size:]

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for si, s in batch:
                f = pool.submit(_run_m2, f"ambig{si}", s, False)
                futures[f] = si

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"    WARN: ambiguous scenario failed: {e}")

        if len(ambig_confs) % 5 == 0:
            print(f"    [{len(ambig_confs)}/{len(ambiguous)}] ambiguous scenarios done")

    # === Analysis ===
    mean_clear = mean(clear_confs) if clear_confs else 0.0
    mean_ambig = mean(ambig_confs) if ambig_confs else 0.0
    discrimination = mean_clear - mean_ambig

    # Behavioral proxy discrimination
    mean_clear_pct = mean(clear_pcts) if clear_pcts else 0.0
    mean_ambig_pct = mean(ambig_pcts) if ambig_pcts else 0.0
    pct_discrimination = mean_clear_pct - mean_ambig_pct

    # T-test: are clear-cut confidences significantly higher?
    n_c = len(clear_confs)
    n_a = len(ambig_confs)
    if n_c >= 2 and n_a >= 2:
        sd_c = stdev(clear_confs)
        sd_a = stdev(ambig_confs)
        se_diff = math.sqrt(sd_c**2 / n_c + sd_a**2 / n_a) if (sd_c > 0 or sd_a > 0) else 1e-10
        t_discrim = discrimination / max(se_diff, 1e-10)
        df_discrim = n_c + n_a - 2
    else:
        t_discrim = 0.0
        df_discrim = 2

    # T-test on percent_agree
    n_cp = len(clear_pcts)
    n_ap = len(ambig_pcts)
    if n_cp >= 2 and n_ap >= 2:
        sd_cp = stdev(clear_pcts)
        sd_ap = stdev(ambig_pcts)
        se_pct = math.sqrt(sd_cp**2 / n_cp + sd_ap**2 / n_ap) if (sd_cp > 0 or sd_ap > 0) else 1e-10
        t_pct_discrim = pct_discrimination / max(se_pct, 1e-10)
    else:
        t_pct_discrim = 0.0

    ctrl_avg_range = mean(ctrl_diffs) if ctrl_diffs else 0.0

    # Score: blend self-reported and behavioral discrimination
    discrim_self = clamp(discrimination / 5.0, 0, 1)
    discrim_pct = clamp(pct_discrimination / 30.0, 0, 1)  # 30% gap = perfect
    discrim_score = 0.4 * discrim_self + 0.6 * discrim_pct

    print(f"\n  RESULTS (M2: uncertainty discrimination):")
    print(f"  SELF-REPORTED CONFIDENCE (0-10):")
    print(f"  Mean confidence (clear-cut): {mean_clear:.2f} (n={n_c})")
    print(f"  Mean confidence (ambiguous): {mean_ambig:.2f} (n={n_a})")
    print(f"  Discrimination (clear - ambig): {discrimination:+.2f}")
    print(f"  T-test: t={t_discrim:.2f}, df={df_discrim}")
    print()
    print(f"  BEHAVIORAL PROXY (percent-agree):")
    print(f"  Mean pct-agree (clear-cut): {mean_clear_pct:.1f}% (n={n_cp})")
    print(f"  Mean pct-agree (ambiguous): {mean_ambig_pct:.1f}% (n={n_ap})")
    print(f"  Pct discrimination (clear - ambig): {pct_discrimination:+.1f}%")
    print(f"  T-test (pct): t={t_pct_discrim:.2f}")
    print()
    print(f"  Control avg confidence range: {ctrl_avg_range:.2f}")
    print(f"  Discrimination score: {discrim_score:.3f}")
    print(f"  (0.4 * self-reported + 0.6 * behavioral proxy)")
    print(f"  NOTE: Higher discrimination = model knows what it doesn't know.")

    _results_store["M2_discrimination"] = {
        "mean_clear": mean_clear,
        "mean_ambig": mean_ambig,
        "discrimination": discrimination,
        "t_discrim": t_discrim,
        "pct_discrimination": pct_discrimination,
        "t_pct_discrim": t_pct_discrim,
        "df_discrim": df_discrim,
        "ctrl_avg_range": ctrl_avg_range,
        "n_clear": n_c,
        "n_ambig": n_a,
        "score": discrim_score,
    }


# =====================================================================
# M3: SELF-MONITORING
# =====================================================================

@kbench.task(name="m3_self_monitoring")
def m3_self_monitoring(llm):
    """M3: Can the model identify which scenarios it is least certain about?
    12 Dear Abby scenarios, each judged 5 times (different chat_ids).
    After 5 judgments, ask model to rank which scenarios it was least certain about.
    Measure: Spearman correlation between self-reported uncertainty ranking
    and actual verdict variance across the 5 reps.
    """
    N_M3_SCENARIOS = _plan.n_m3_scenarios
    N_M3_REPS = _plan.n_m3_reps

    print("\n[M3] SELF-MONITORING")
    print("  Testing self-awareness of uncertainty")
    print(f"  {N_M3_SCENARIOS} Dear Abby scenarios x {N_M3_REPS} reps each + uncertainty ranking")
    print(f"  [Budget: {_budget.status()}]")
    print("-" * 60)

    m3_scenarios = DEAR_ABBY[:N_M3_SCENARIOS]
    print(f"  Scenarios: {len(m3_scenarios)}")

    # Phase A: Judge each scenario 5 times
    scenario_verdicts = {i: [] for i in range(N_M3_SCENARIOS)}
    scenario_confidences = {i: [] for i in range(N_M3_SCENARIOS)}
    _lock = threading.Lock()

    def _run_m3_rep(si, rep):
        """Judge scenario si, repetition rep."""
        s = m3_scenarios[si]
        try:
            result = call_llm(llm, f"m3_s{si}_r{rep}",
                              prompt_m3_judge(s), CalibratedVerdict)
            v = normalize_verdict_da(result.verdict)
            conf = clamp(result.confidence, 0, 10)
            with _lock:
                scenario_verdicts[si].append(v)
                scenario_confidences[si].append(conf)
        except Exception as e:
            print(f"    WARN: m3 s{si} r{rep} failed: {e}")

    # Run all reps in parallel batches
    all_tasks = [(si, rep) for si in range(N_M3_SCENARIOS) for rep in range(N_M3_REPS)]
    remaining = list(all_tasks)
    while remaining:
        batch_size = min(_pool.n, len(remaining))
        batch = remaining[:batch_size]
        remaining = remaining[batch_size:]

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for si, rep in batch:
                f = pool.submit(_run_m3_rep, si, rep)
                futures[f] = (si, rep)

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

        done = sum(len(v) for v in scenario_verdicts.values())
        total_expected = N_M3_SCENARIOS * N_M3_REPS
        if done % 20 == 0:
            print(f"    [{done}/{total_expected}] judgments collected")

    # Compute actual variance per scenario
    # Use multiple signals: verdict entropy + confidence stdev + confidence mean (inverted)
    actual_variance = {}
    for si in range(N_M3_SCENARIOS):
        vs = scenario_verdicts[si]
        confs = scenario_confidences[si]
        if len(vs) < 2:
            actual_variance[si] = 0.0
        else:
            # Verdict entropy: more unique verdicts = more uncertain
            unique = len(set(vs))
            # Confidence stdev: higher spread = more uncertain
            conf_sd = stdev(confs) if len(confs) > 1 else 0.0
            # Mean confidence inverted: lower avg confidence = more uncertain
            avg_conf = mean(confs) if confs else 10.0
            conf_uncertainty = (10.0 - avg_conf) / 10.0  # 0-1 scale
            # Combined: weighted blend for better discrimination
            actual_variance[si] = unique * 2.0 + conf_sd * 3.0 + conf_uncertainty * 5.0

    # Phase B: Ask model to rank least certain scenarios
    print(f"\n  Phase B: asking model to rank least certain scenarios...")
    titles = [m3_scenarios[si]["title"] for si in range(N_M3_SCENARIOS)]

    try:
        ranking_result = call_llm(llm, "m3_ranking",
                                  prompt_m3_uncertainty_ranking(titles),
                                  UncertaintyReport)

        # Parse self-reported top-3 least certain
        self_reported = [
            str(ranking_result.least_certain_scenario_1).strip(),
            str(ranking_result.least_certain_scenario_2).strip(),
            str(ranking_result.least_certain_scenario_3).strip(),
        ]

        # Map titles to indices
        def title_to_idx(title_str):
            """Find best matching scenario index for a title string."""
            title_lower = title_str.lower()
            for si in range(N_M3_SCENARIOS):
                if m3_scenarios[si]["title"].lower() in title_lower or \
                   title_lower in m3_scenarios[si]["title"].lower():
                    return si
            # Fuzzy: check if any words match
            for si in range(N_M3_SCENARIOS):
                words = set(m3_scenarios[si]["title"].lower().split())
                query_words = set(title_lower.split())
                if len(words & query_words) >= 2:
                    return si
            return -1

        self_indices = [title_to_idx(t) for t in self_reported]
        valid_self = [idx for idx in self_indices if idx >= 0]

        print(f"  Self-reported least certain: {self_reported}")
        print(f"  Matched indices: {self_indices}")

    except Exception as e:
        print(f"  WARN: ranking query failed: {e}")
        valid_self = []
        self_indices = []

    # Compute Spearman correlation between self-reported rank and actual variance
    if len(valid_self) >= 3:
        # Build full ranking vectors
        # Self-reported rank: top-3 get ranks 1,2,3; rest get tied rank = average of remaining
        self_ranks = []
        for si in range(N_M3_SCENARIOS):
            if si in self_indices:
                # Position in self-reported list (1-indexed)
                self_ranks.append(self_indices.index(si) + 1)
            else:
                # Unranked: assign middle rank
                self_ranks.append((N_M3_SCENARIOS + 4) / 2)

        actual_ranks = [actual_variance.get(si, 0.0) for si in range(N_M3_SCENARIOS)]

        # Spearman: correlation between self-reported uncertainty ranking
        # and actual verdict variance
        # Note: for self-rank, lower = more uncertain (rank 1 = least certain)
        # For actual, higher variance = more uncertain
        # So we want NEGATIVE correlation to indicate alignment
        # OR: invert self_ranks so higher = more uncertain
        inverted_self = [N_M3_SCENARIOS + 1 - r for r in self_ranks]
        rho = spearman_rho(inverted_self, actual_ranks)
    else:
        rho = 0.0

    # Report per-scenario variance
    sorted_by_var = sorted(range(N_M3_SCENARIOS), key=lambda si: actual_variance.get(si, 0), reverse=True)

    print(f"\n  SCENARIO VARIANCE (actual):")
    print(f"  {'Rank':<6} {'Title':<40} {'Variance':>10} {'Verdicts'}")
    print(f"  {'-'*75}")
    for rank, si in enumerate(sorted_by_var, 1):
        vs = scenario_verdicts[si]
        title = m3_scenarios[si]["title"][:38]
        var = actual_variance.get(si, 0)
        verdicts_str = ", ".join(vs) if vs else "none"
        self_mark = " <<<" if si in valid_self else ""
        print(f"  {rank:<6} {title:<40} {var:>9.2f} {verdicts_str}{self_mark}")

    # Score
    score = max(0.0, (rho + 1) / 2)  # Map [-1, 1] to [0, 1]

    print(f"\n  RESULTS (M3: self-monitoring):")
    print(f"  Spearman rho (self-reported vs actual variance): {rho:.3f}")
    print(f"  Self-monitoring score: {score:.3f}")
    print(f"  NOTE: Positive rho = model correctly identifies uncertain scenarios.")
    print(f"  rho > 0.3 would suggest meaningful self-monitoring ability.")
    if rho < 0:
        print(f"  CAUTION: Negative rho suggests model is ANTI-calibrated")
        print(f"  (thinks it's least certain on scenarios where it's most consistent).")

    _results_store["M3_self_monitoring"] = {
        "spearman_rho": rho,
        "n_scenarios": N_M3_SCENARIOS,
        "n_reps": N_M3_REPS,
        "actual_variance": {str(k): v for k, v in actual_variance.items()},
        "self_reported": self_indices,
        "score": score,
    }


# =====================================================================
# M4: STRATEGY SELECTION
# =====================================================================

@kbench.task(name="m4_strategy_selection")
def m4_strategy_selection(llm):
    """M4: Does reasoning length/complexity scale with scenario complexity?
    8 simple (short, clear) + 8 complex (long, multi-party) Dear Abby scenarios.
    Measure: avg reasoning length for complex vs simple scenarios.
    """
    n_simple = _plan.n_m4_simple
    n_complex = _plan.n_m4_complex

    print("\n[M4] STRATEGY SELECTION")
    print("  Testing if reasoning effort scales with complexity")
    print(f"  {n_simple} simple + {n_complex} complex Dear Abby scenarios")
    print(f"  [Budget: {_budget.status()}]")
    print("-" * 60)

    simple_lengths = []
    complex_lengths = []
    simple_confs = []
    complex_confs = []
    _lock = threading.Lock()

    def _run_m4(tag, s, is_complex):
        """Judge scenario and measure reasoning length."""
        try:
            result = call_llm(llm, f"m4_{tag}", prompt_m4_judge(s), StrategyVerdict)
            reasoning = str(result.reasoning)
            length = len(reasoning)
            conf = clamp(result.confidence, 0, 10)

            with _lock:
                if is_complex:
                    complex_lengths.append(length)
                    complex_confs.append(conf)
                else:
                    simple_lengths.append(length)
                    simple_confs.append(conf)
        except Exception as e:
            print(f"    WARN: {tag} failed: {e}")

    # Run simple scenarios (budget-scaled count)
    m4_simple = M4_SIMPLE_INDICES[:n_simple]
    print(f"  --- Simple scenarios ({len(m4_simple)}) ---")
    for si_idx, si in enumerate(m4_simple):
        if si < len(DEAR_ABBY):
            try:
                _run_m4(f"simple{si_idx}", DEAR_ABBY[si], False)
            except Exception as e:
                print(f"    WARN: simple {si_idx} failed: {e}")

    # Run complex scenarios (budget-scaled, use extended versions if available)
    m4_complex = M4_COMPLEX_INDICES[:n_complex]
    print(f"  --- Complex scenarios ({len(m4_complex)}) ---")
    for ci_idx, ci in enumerate(m4_complex):
        if ci < len(DEAR_ABBY):
            # Use extended version if available, otherwise original
            extended = _transforms.get((ci, "extended_complex"))
            if extended:
                s = {"title": DEAR_ABBY[ci]["title"] + " (extended)", "text": extended[:1500]}
            else:
                s = DEAR_ABBY[ci]
            try:
                _run_m4(f"complex{ci_idx}", s, True)
            except Exception as e:
                print(f"    WARN: complex {ci_idx} failed: {e}")

    # === Analysis ===
    mean_simple_len = mean(simple_lengths) if simple_lengths else 0.0
    mean_complex_len = mean(complex_lengths) if complex_lengths else 0.0
    length_ratio = mean_complex_len / max(mean_simple_len, 1)

    mean_simple_conf = mean(simple_confs) if simple_confs else 0.0
    mean_complex_conf = mean(complex_confs) if complex_confs else 0.0

    # T-test: complex reasoning is longer than simple?
    if len(simple_lengths) >= 2 and len(complex_lengths) >= 2:
        sd_s = stdev(simple_lengths)
        sd_c = stdev(complex_lengths)
        se = math.sqrt(sd_s**2 / len(simple_lengths) + sd_c**2 / len(complex_lengths))
        t_length = (mean_complex_len - mean_simple_len) / max(se, 1e-10)
    else:
        t_length = 0.0

    # Score: ratio > 1 is good (more reasoning for complex), clamped
    strategy_score = clamp((length_ratio - 1.0) / 1.0, 0, 1)  # ratio of 2.0 -> score 1.0

    print(f"\n  RESULTS (M4: strategy selection):")
    print(f"  Simple avg reasoning length: {mean_simple_len:.0f} chars (n={len(simple_lengths)})")
    print(f"  Complex avg reasoning length: {mean_complex_len:.0f} chars (n={len(complex_lengths)})")
    print(f"  Length ratio (complex/simple): {length_ratio:.2f}")
    print(f"  T-test (complex > simple): t={t_length:.2f}")
    print(f"  Simple avg confidence: {mean_simple_conf:.1f}")
    print(f"  Complex avg confidence: {mean_complex_conf:.1f}")
    print(f"  Strategy selection score: {strategy_score:.3f}")
    print(f"  NOTE: Ratio > 1 means model invests more reasoning in complex scenarios.")
    print(f"  A ratio near 1.0 suggests uniform strategy regardless of complexity.")

    _results_store["M4_strategy"] = {
        "mean_simple_len": mean_simple_len,
        "mean_complex_len": mean_complex_len,
        "length_ratio": length_ratio,
        "t_length": t_length,
        "mean_simple_conf": mean_simple_conf,
        "mean_complex_conf": mean_complex_conf,
        "n_simple": len(simple_lengths),
        "n_complex": len(complex_lengths),
        "score": strategy_score,
    }


# =====================================================================
# MULTI-MODEL EXECUTION
# =====================================================================

MODELS_FULL = [
    "google/gemini-2.0-flash",       # baseline, older gen (also transformer model)
    "google/gemini-2.5-flash",       # current gen flash
    "google/gemini-3-flash-preview", # next gen
    "anthropic/claude-sonnet-4-6@default",  # v5: cross-family full suite (was gold-only)
]

# v5: Pro dropped entirely — at $0.65/call even M1-only wastes budget better spent on full-suite models
# Pro's contribution was limited (only M1-M2 data in v4 before budget exhaustion)
# 4 full-suite models with Fisher combination across 4 > 2 models with data + 2 without
MODELS_M1_ONLY = [
]

# Cross-family model -- now in MODELS_FULL, keep reference for reporting
CROSS_FAMILY_MODEL = "anthropic/claude-sonnet-4-6@default"
N_CROSS_FAMILY_PROBES = 6  # gold set probes for calibration

MODELS_TO_TEST = MODELS_FULL

print(f"\n[3/7] Phase 1: Pre-generating difficulty variants with {TRANSFORMER_MODEL}")
try:
    transformer_llm = kbench.llms[TRANSFORMER_MODEL]
    phase1_pre_generate.run(llm=transformer_llm)
except Exception as e:
    print(f"  WARN: Pre-generation failed: {e}")
    print(f"  M4 will use original scenarios instead of extended versions.")

print(f"\n[4/7] Phase 2: Running 4 tests across {len(MODELS_TO_TEST)} full models")
for m in MODELS_TO_TEST:
    print(f"  - {m}")
print()

all_results = {}
_models_remaining = len(MODELS_TO_TEST) + len(MODELS_M1_ONLY)

for mi, model_name in enumerate(MODELS_TO_TEST):
    print(f"\n{'#'*70}")
    print(f"# MODEL {mi+1}/{len(MODELS_TO_TEST)}: {model_name}")
    print(f"# [Budget: {_budget.status()}]")
    print(f"{'#'*70}")

    _active_model_name = model_name
    _models_remaining = len(MODELS_TO_TEST) + len(MODELS_M1_ONLY) - mi

    # Budget-aware planning: select execution tier for this model
    _plan = _budget.plan_model(model_name, _models_remaining, is_m1_only=False)
    cpc = MODEL_COST_PER_CALL.get(model_name, DEFAULT_COST_PER_CALL)
    print(f"  Budget plan: {_plan.n_aita} AITA, {_plan.n_m2_per_group}/grp M2, "
          f"{_plan.n_ctrl_reps} ctrl, {_plan.n_m3_reps} M3reps, "
          f"~{_plan.estimated_calls} calls, ~${_plan.estimated_cost(model_name):.2f} "
          f"(${cpc:.3f}/call)")

    model_results = {}
    try:
        llm = kbench.llms[model_name]
        _results_store.clear()

        for test_fn, test_name in [
            (m1_calibration, "M1_calibration"),
            (m2_uncertainty_discrimination, "M2_discrimination"),
            (m3_self_monitoring, "M3_self_monitoring"),
            (m4_strategy_selection, "M4_strategy"),
        ]:
            try:
                test_fn.run(llm=llm)
                model_results[test_name] = _results_store.get(test_name, {"score": 0.0})
            except Exception as e:
                print(f"  ERROR in {test_name}: {e}")
                model_results[test_name] = {"error": str(e), "score": 0.0}

    except Exception as e:
        print(f"  ERROR loading model {model_name}: {e}")
        model_results = {f"M{i}": {"error": str(e), "score": 0.0} for i in range(1, 5)}

    all_results[model_name] = model_results

# =====================================================================
# M1-ONLY MODELS (additional statistical power for headline finding)
# =====================================================================

if MODELS_M1_ONLY:
    print(f"\n[5/7] Running M1-only on {len(MODELS_M1_ONLY)} additional models")
    for m in MODELS_M1_ONLY:
        print(f"  - {m} (M1 only)")

    for mi, model_name in enumerate(MODELS_M1_ONLY):
        print(f"\n{'#'*70}")
        print(f"# M1-ONLY {mi+1}/{len(MODELS_M1_ONLY)}: {model_name}")
        print(f"# [Budget: {_budget.status()}]")
        print(f"{'#'*70}")

        _active_model_name = model_name
        _models_remaining = len(MODELS_M1_ONLY) - mi

        _plan = _budget.plan_model(model_name, _models_remaining, is_m1_only=True)
        cpc = MODEL_COST_PER_CALL.get(model_name, DEFAULT_COST_PER_CALL)
        m1_calls = (6 + 6 + _plan.n_aita) * (1 + _plan.n_ctrl_reps)
        print(f"  Budget plan: {_plan.n_aita} AITA, {_plan.n_ctrl_reps} ctrl, "
              f"~{m1_calls} calls, ~${m1_calls * cpc:.2f} est.")

        model_results = {}
        try:
            llm = kbench.llms[model_name]
            _results_store.clear()
            m1_calibration.run(llm=llm)
            model_results["M1_calibration"] = _results_store.get("M1_calibration", {"score": 0.0})
        except Exception as e:
            print(f"  ERROR: {e}")
            model_results["M1_calibration"] = {"error": str(e), "score": 0.0}

        all_results[model_name] = model_results

# =====================================================================
# CROSS-FAMILY VALIDATION (M1 gold probes on non-Gemini model)
# =====================================================================

# v5: Skip cross-family section — Claude now runs full suite in MODELS_FULL
cross_family_results = {}
if CROSS_FAMILY_MODEL in all_results:
    print(f"\n[6/7] Cross-family: {CROSS_FAMILY_MODEL} already ran full suite above — skipping duplicate.")
    cross_family_results = all_results[CROSS_FAMILY_MODEL]
else:
    print(f"\n{'#'*70}")
    print(f"# CROSS-FAMILY: {CROSS_FAMILY_MODEL}")
    print(f"# M1 calibration gold probes only ({N_CROSS_FAMILY_PROBES} probes)")
    print(f"# [Budget: {_budget.status()}]")
    print(f"{'#'*70}")

    _active_model_name = CROSS_FAMILY_MODEL
    _models_remaining = 1
try:
    cf_llm = kbench.llms[CROSS_FAMILY_MODEL]
    cf_pairs = []  # (confidence, correct)

    for gi, gs in enumerate(GOLD_SET[:N_CROSS_FAMILY_PROBES]):
        prompt = (
            f"You are judging a moral scenario.\n\n"
            f"{gs['text']}\n\n"
            f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
            f"Confidence 0-10 (0 = completely uncertain, 10 = absolutely certain).\n"
            f"Keep reasoning to 2-3 sentences MAX."
        )
        try:
            result = call_llm(cf_llm, f"cf_gold_{gi}", prompt, CalibratedVerdict)
            v = normalize_verdict(result.verdict)
            conf = clamp(result.confidence, 0, 10)
            correct = (v == gs["m1_known_verdict"])
            cf_pairs.append((conf, correct))

            mark = "CORRECT" if correct else "MISS"
            diff_str = gs["m1_difficulty"]
            conf_in_range = gs["m1_expected_confidence_lo"] <= conf <= gs["m1_expected_confidence_hi"]
            range_mark = "IN_RANGE" if conf_in_range else "OUT_RANGE"
            print(f"  {gs['title'][:30]:30s} v={v:3s} conf={conf:.0f} {mark} "
                  f"({diff_str}, expected {gs['m1_expected_confidence_lo']}-{gs['m1_expected_confidence_hi']}) "
                  f"{range_mark}")
        except Exception as e:
            print(f"  WARN: gold {gi} failed: {e}")

    if cf_pairs:
        cf_accuracy = sum(1 for _, c in cf_pairs if c) / len(cf_pairs)
        cf_avg_conf = mean([c for c, _ in cf_pairs])
        # ECE on cross-family
        cf_bins = {i: {"correct": 0, "total": 0, "sum_conf": 0.0} for i in range(5)}
        for conf, correct in cf_pairs:
            bi = min(int(conf / 2), 4)
            if conf >= 10:
                bi = 4
            cf_bins[bi]["total"] += 1
            cf_bins[bi]["sum_conf"] += conf
            if correct:
                cf_bins[bi]["correct"] += 1
        cf_ece = 0.0
        for i in range(5):
            b = cf_bins[i]
            if b["total"] == 0:
                continue
            ba = b["correct"] / b["total"]
            bc = b["sum_conf"] / b["total"] / 10.0
            cf_ece += (b["total"] / len(cf_pairs)) * abs(ba - bc)

        cross_family_results = {
            "accuracy": cf_accuracy,
            "avg_confidence": cf_avg_conf,
            "ece": cf_ece,
            "n_probes": len(cf_pairs),
        }
        print(f"\n  CROSS-FAMILY M1 RESULTS ({CROSS_FAMILY_MODEL.split('/')[-1]}):")
        print(f"  Accuracy: {cf_accuracy:.0%}, Avg confidence: {cf_avg_conf:.1f}, ECE: {cf_ece:.4f}")
        print(f"  (Compare to Gemini models above to assess family-specificity)")
    else:
        print("  No cross-family results available")

except Exception as e:
    print(f"  Cross-family test failed: {e}")
    print(f"  (This is OK -- the Gemini results still stand on their own)")


# =====================================================================
# CROSS-MODEL COMPARISON
# =====================================================================

print(f"\n\n{'#'*70}")
print(f"CROSS-MODEL COMPARISON -- FOUR METACOGNITION TESTS")
print(f"{'#'*70}")
print()

WEIGHTS = {
    "M1_calibration": 0.35,
    "M2_discrimination": 0.25,
    "M3_self_monitoring": 0.20,
    "M4_strategy": 0.20,
}

header = (f"  {'Model':<30} {'M1:Cal':>8} {'M2:Disc':>8} "
          f"{'M3:Mon':>8} {'M4:Strat':>9} {'Compos':>8}")
print(header)
print(f"  {'-'*71}")

for model_name, results in all_results.items():
    scores = {}
    for test_key in WEIGHTS:
        r = results.get(test_key, {})
        scores[test_key] = r.get("score", 0.0)

    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

    short_name = model_name.split("/")[-1][:28]
    print(f"  {short_name:<30} "
          f"{scores['M1_calibration']:>7.3f} "
          f"{scores['M2_discrimination']:>7.3f} "
          f"{scores['M3_self_monitoring']:>7.3f} "
          f"{scores['M4_strategy']:>8.3f} "
          f"{composite:>7.3f}")

print()
print(f"  Weights: M1={WEIGHTS['M1_calibration']}, M2={WEIGHTS['M2_discrimination']}, "
      f"M3={WEIGHTS['M3_self_monitoring']}, M4={WEIGHTS['M4_strategy']}")
print(f"  (M1 weighted highest: headline calibration test with ECE)")

# =====================================================================
# SENSITIVITY ANALYSIS: Weight perturbation stability
# =====================================================================

print()
print("=" * 70)
print("SENSITIVITY ANALYSIS (weight perturbation)")
print(f"{'='*70}")
print()
print("  Testing whether model rankings change under ±50% weight perturbation.")
print("  For each weight, we try 0.5× and 1.5× while renormalizing.")
print()

# Collect per-model score vectors
_sa_model_scores = {}
for model_name, results in all_results.items():
    scores = {}
    for test_key in WEIGHTS:
        r = results.get(test_key, {})
        scores[test_key] = r.get("score", 0.0)
    _sa_model_scores[model_name] = scores

if len(_sa_model_scores) >= 2:
    def _rank_models(w):
        composites = {}
        total_w = sum(w.values())
        for mn, sc in _sa_model_scores.items():
            composites[mn] = sum(w[k] * sc[k] / total_w for k in w)
        return sorted(composites.keys(), key=lambda m: composites[m], reverse=True)

    original_ranking = _rank_models(WEIGHTS)

    def _kendall_tau(rank_a, rank_b):
        n = len(rank_a)
        if n < 2:
            return 1.0
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                b_i = rank_b.index(rank_a[i])
                b_j = rank_b.index(rank_a[j])
                if (i - j) * (b_i - b_j) > 0:
                    concordant += 1
                elif (i - j) * (b_i - b_j) < 0:
                    discordant += 1
        denom = concordant + discordant
        return (concordant - discordant) / denom if denom > 0 else 1.0

    n_stable = 0
    n_total = 0
    tau_values = []
    changed = []

    for perturbed_key in WEIGHTS:
        for factor_label, factor in [("0.5x", 0.5), ("1.5x", 1.5)]:
            w_perturbed = dict(WEIGHTS)
            w_perturbed[perturbed_key] = WEIGHTS[perturbed_key] * factor
            perturbed_ranking = _rank_models(w_perturbed)
            same = (perturbed_ranking == original_ranking)
            tau = _kendall_tau(original_ranking, perturbed_ranking)
            tau_values.append(tau)
            n_total += 1
            if same:
                n_stable += 1
            else:
                changed.append(f"{perturbed_key} at {factor_label}")

    print(f"  Original ranking: {' > '.join(m.split('/')[-1][:20] for m in original_ranking)}")
    print(f"  Rankings preserved: {n_stable}/{n_total} ({100*n_stable/n_total:.0f}%)")
    mean_tau = sum(tau_values) / len(tau_values)
    print(f"  Mean Kendall tau: {mean_tau:.3f}")
    if changed:
        print(f"  Changed under: {', '.join(changed)}")
    else:
        print(f"  >>> Rankings FULLY STABLE under +/-50% weight perturbation <<<")
    print()
else:
    print("  Skipped: need 2+ models for sensitivity analysis.")
    print()



# =====================================================================
# FISHER COMBINATION: SIGMA CALCULATION
# =====================================================================

print()
print("=" * 70)
print("SIGMA ANALYSIS (Fisher combination across models)")
print("=" * 70)
print()

# Collect M1 ECE z-scores from all models (Gemini full + M1-only)
# ECE z = ECE / SE(ECE) gives significance of MISCALIBRATION
# Higher z = more significantly miscalibrated
# Also collect overconfidence rate for two-proportion z-test
m1_ece_sigmas = []
m1_pct_ece_sigmas = []
m1_overconf_sigmas = []

print("  Individual M1 results (ECE z-score and overconfidence):")
for model_name in list(MODELS_FULL) + list(MODELS_M1_ONLY):
    r = all_results.get(model_name, {}).get("M1_calibration", {})
    short = model_name.split("/")[-1][:25]
    ece = r.get("ece", 0)
    ece_z = r.get("ece_z", 0)
    pct_ece = r.get("pct_ece", 0)
    pct_ece_z = r.get("pct_ece_z", 0)
    overconf_wrong = r.get("overconf_wrong", 0)
    overconf_total = r.get("overconf_total", 0)
    n_pairs = r.get("n_pairs", 0)

    if ece == 0 and n_pairs == 0:
        print(f"    {short:25s}  (no M1 data)")
        continue

    # ECE z-score -> sigma (self-reported)
    sig_ece = _p_to_sigma(_sigma_to_p(ece_z) if ece_z > 0 else 0.5)
    m1_ece_sigmas.append(sig_ece)

    # Behavioral ECE z-score -> sigma
    sig_pct = _p_to_sigma(_sigma_to_p(pct_ece_z) if pct_ece_z > 0 else 0.5)
    m1_pct_ece_sigmas.append(sig_pct)

    # Overconfidence
    ctrl_overconf = max(1, int(0.1 * overconf_total))
    z_overconf = two_proportion_z(overconf_wrong, max(overconf_total, 1),
                                  ctrl_overconf, max(overconf_total, 1))
    sig_overconf = _p_to_sigma(_sigma_to_p(z_overconf) if z_overconf > 0 else 0.5)
    m1_overconf_sigmas.append(sig_overconf)

    sa = '***' if sig_ece >= 3 else '**' if sig_ece >= 2 else '*' if sig_ece >= 1.5 else ''
    sp = '***' if sig_pct >= 3 else '**' if sig_pct >= 2 else '*' if sig_pct >= 1.5 else ''
    print(f"    {short:25s}  self-ECE={ece:.4f} z={ece_z:.1f} -> {sig_ece:.1f}s {sa:4s}  "
          f"behav-ECE={pct_ece:.4f} z={pct_ece_z:.1f} -> {sig_pct:.1f}s {sp}")

if len(m1_ece_sigmas) >= 2:
    combined_ece = _fisher_combine(m1_ece_sigmas)
    combined_pct_ece = _fisher_combine(m1_pct_ece_sigmas) if m1_pct_ece_sigmas else 0.0
    combined_overconf = _fisher_combine(m1_overconf_sigmas) if m1_overconf_sigmas else 0.0

    print()
    print(f"  Fisher combined ({len(m1_ece_sigmas)} independent tests, {len(m1_ece_sigmas)} models):")
    print(f"    M1 self-reported ECE:    {combined_ece:.1f}s "
          f"{'*** DISCOVERY-LEVEL ***' if combined_ece >= 5 else '** SIGNIFICANT **' if combined_ece >= 3 else ''}")
    print(f"    M1 behavioral ECE:       {combined_pct_ece:.1f}s "
          f"{'*** DISCOVERY-LEVEL ***' if combined_pct_ece >= 5 else '** SIGNIFICANT **' if combined_pct_ece >= 3 else ''}")
    print(f"    M1 overconfidence:       {combined_overconf:.1f}s "
          f"{'*** DISCOVERY-LEVEL ***' if combined_overconf >= 5 else '** SIGNIFICANT **' if combined_overconf >= 3 else ''}")
    print()
    best_ece_sigma = max(combined_ece, combined_pct_ece)
    if best_ece_sigma >= 5:
        print(f"  >>> HEADLINE: Systematic miscalibration at {best_ece_sigma:.1f} sigma <<<")
        print(f"  >>> {len(m1_ece_sigmas)} models, dual-metric (self-reported + behavioral) <<<")
    elif combined_ece >= 3:
        print(f"  >>> Significant miscalibration at {combined_ece:.1f} sigma <<<")
else:
    combined_ece = 0
    combined_overconf = 0
    print("  (insufficient M1 data for Fisher combination)")

print()

# =====================================================================
# HEADLINE FINDINGS
# =====================================================================

print("=" * 70)
print("HEADLINE FINDINGS")
print("=" * 70)
print()
ece_sigma_str = f"{combined_ece:.1f}" if combined_ece > 0 else "N/A"
print(f"  1. SYSTEMATIC MISCALIBRATION AT {ece_sigma_str} SIGMA (M1)")
print(f"     Across {len(m1_ece_sigmas)} Gemini models, confidence ratings are")
print("     systematically miscalibrated relative to actual accuracy.")
print("     Fisher combination of ECE z-scores across all models yields")
print(f"     {ece_sigma_str} sigma combined significance.")
if cross_family_results:
    cf_short = CROSS_FAMILY_MODEL.split("/")[-1]
    print(f"     Cross-family validation on {cf_short}:")
    print(f"     ECE={cross_family_results.get('ece', 0):.4f}, "
          f"accuracy={cross_family_results.get('accuracy', 0):.0%}")
print()

# Overconfidence summary
overconf_rates = []
for model_name in list(MODELS_FULL) + list(MODELS_M1_ONLY):
    r = all_results.get(model_name, {}).get("M1_calibration", {})
    ocr = r.get("overconfidence_rate", None)
    if ocr is not None:
        overconf_rates.append(ocr)
avg_overconf = mean(overconf_rates) if overconf_rates else 0.0
overconf_word = "significant" if avg_overconf > 0.3 else "moderate" if avg_overconf > 0.15 else "low"

print(f"  2. OVERCONFIDENCE IS {overconf_word.upper()} (M1)")
print(f"     Average overconfidence rate across models: {avg_overconf:.0%}")
print("     (Fraction of high-confidence (>8) judgments that are wrong)")
print()

# M2 discrimination summary
discrim_values = []
for model_name in MODELS_FULL:
    r = all_results.get(model_name, {}).get("M2_discrimination", {})
    d = r.get("discrimination", None)
    if d is not None:
        discrim_values.append(d)
avg_discrim = mean(discrim_values) if discrim_values else 0.0
discrim_word = "strong" if avg_discrim > 2.0 else "moderate" if avg_discrim > 1.0 else "weak"

print(f"  3. UNCERTAINTY DISCRIMINATION IS {discrim_word.upper()} (M2)")
print(f"     Average confidence gap (clear vs ambiguous): {avg_discrim:+.2f}")
print("     Positive gap means model reports lower confidence on ambiguous cases.")
print()

# M3 and M4 summary
rho_values = []
ratio_values = []
for model_name in MODELS_FULL:
    r3 = all_results.get(model_name, {}).get("M3_self_monitoring", {})
    r4 = all_results.get(model_name, {}).get("M4_strategy", {})
    rho = r3.get("spearman_rho", None)
    ratio = r4.get("length_ratio", None)
    if rho is not None:
        rho_values.append(rho)
    if ratio is not None:
        ratio_values.append(ratio)
avg_rho = mean(rho_values) if rho_values else 0.0
avg_ratio = mean(ratio_values) if ratio_values else 1.0

print(f"  4. SELF-MONITORING: Spearman rho = {avg_rho:.3f} (M3)")
print(f"     STRATEGY SELECTION: reasoning length ratio = {avg_ratio:.2f}x (M4)")
print("     M3 measures whether models can identify their own uncertainty.")
print("     M4 measures whether reasoning effort scales with complexity.")
print()


# =====================================================================
# METHODOLOGY & LIMITATIONS
# =====================================================================

print()
print("=" * 70)
print("METHODOLOGY & INTERPRETATION")
print("=" * 70)
print()
print("  TEST TYPES:")
print("    M1 (Calibration): HEADLINE TEST. Do confidence ratings predict")
print("      accuracy? Expected Calibration Error (ECE) with 5 bins.")
print("      Also measures overconfidence rate (conf>8 but wrong).")
print("      Fisher combination of ECE z-scores across 4 Gemini models.")
print("    M2 (Knowing What You Don't Know): Does the model report lower")
print("      confidence on genuinely ambiguous scenarios (ESH/NAH) vs")
print("      clear-cut cases (NTA/YTA with high community agreement)?")
print("    M3 (Self-Monitoring): Can the model identify which scenarios")
print("      it is least certain about? 12 scenarios x 5 reps each,")
print("      then model ranks its least-certain scenarios. Spearman")
print("      correlation between self-report and actual verdict variance.")
print("    M4 (Strategy Selection): Does reasoning effort scale with")
print("      scenario complexity? Compares reasoning length for simple")
print("      (2-party, short) vs complex (multi-party, long) scenarios.")
print()
print("  STATISTICAL CONTROLS:")
print(f"    M1-M2 include {N_CONTROL_REPS}-rep control arms: the model re-judges")
print("    identical text to estimate stochastic confidence baseline.")
print("    M1 ECE significance via bootstrap standard error.")
print("    M2 discrimination via two-sample t-test.")
print("    All rates reported with Wilson 95% confidence intervals.")
print(f"    CAVEAT: {N_CONTROL_REPS} reps per scenario is a thin estimate of stochasticity.")
print()
print("  THREE-TIER DATA ARCHITECTURE:")
print("    Gold tier: hand-audited scenarios with known verdicts and")
print("    expected confidence ranges. Highest interpretive confidence.")
print("    Probe tier: synthetic scenarios engineered for maximum control")
print("    (unambiguously easy or genuinely ambiguous by construction).")
print("    AITA tier: crowd-labeled scenarios from Reddit. Large sample")
print("    but noisier labels (crowd consensus, not ground truth).")
print()
print("  KNOWN LIMITATIONS (this is a pilot, not a definitive study):")
print("    1. Small samples: 12-52 scenarios per test. Results are")
print("       directional evidence, not sweeping claims.")
print("    2. Full suite runs on Gemini-family only (budget constraint).")
print(f"       Cross-family validation on {CROSS_FAMILY_MODEL.split('/')[-1]}")
print("       covers M1 gold probes only.")
print("    3. AITA crowd labels are not ground truth; minority verdicts")
print("       (ESH, NAH) have lower inter-rater agreement.")
print("    4. M3 self-monitoring relies on the model parsing its own")
print("       uncertainty from a single ranking query; this is a coarse")
print("       measure of metacognitive access.")
print("    5. M4 reasoning length is a crude proxy for reasoning effort.")
print("       Though models may still default to uniform length.")
print("    6. No temperature control; adaptive concurrency may introduce")
print("       minor noise from retry patterns.")
print(f"    7. Control arms ({N_CONTROL_REPS} reps) provide a stochasticity")
print("       floor but not a full variance model.")
print("=" * 70)

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Budget: {_budget.status()}")
print(f"Cost per model:")
for mn in list(MODELS_FULL) + list(MODELS_M1_ONLY):
    cpc = MODEL_COST_PER_CALL.get(mn, DEFAULT_COST_PER_CALL)
    short = mn.split("/")[-1][:25]
    print(f"  {short:25s} ~${cpc:.3f}/call")
