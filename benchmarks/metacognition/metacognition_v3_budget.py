"""Metacognition Benchmark v3 -- Improved Edition (~$35 of $50 quota)
Metacognition Track | Measuring AGI Competition

Tests 4 metacognitive properties of moral cognition (Bond, 2026):
  M1. Calibration -- confidence-accuracy alignment (HEADLINE)
  M2. Knowing What You Don't Know -- rank-based + raw discrimination
  M3. Self-Monitoring -- self-reported vs actual verdict uncertainty
  M4. Strategy Selection -- unconstrained reasoning effort scaling

Methodological improvements over v2:
  1. M4 prompt fix: removed "2-3 sentences MAX" cap that was
     directly contradicting the reasoning-effort measurement.
     Models now free to vary reasoning length naturally.
  2. M2 adds rank-based AUROC metric alongside raw confidence gap.
     Tests whether models rank clear cases higher even when raw
     confidence is compressed into 8-10 range.
  3. Control arms increased from 3-rep to 5-rep for tighter baselines.
  4. Claude Sonnet runs full M1-M4 suite for cross-family comparison.
  5. All v2 controls retained: bootstrap ECE, Wilson CIs, Fisher
     combination, three-tier data, Spearman correlation.

Budget: 3 full models + 2 M1-only (~$0.014-0.03/call).
  Full suite: ~320 calls x 3 models = ~960 ($19)
  M1-only: ~100 calls x 2 models = ~200 ($4)
  Total: ~$24 (fits $35 target with margin)

Paste this ENTIRE file into ONE cell in a Kaggle Benchmark Task notebook.
Expected runtime: ~40-60 min (adaptive, 3+2 models including Claude).
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

# --- 1a. AITA dataset (for M1, M2   " needs verdict labels + scores) ---
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
# 1c. GOLD SET   " Hand-audited scenarios with M1 calibration annotations
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
# 1d. ENGINEERED PROBES   " Synthetic scenarios for calibration
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
M4_SIMPLE_INDICES = [0, 1, 4, 7, 10, 12, 20, 21]  # 8 simple scenarios
# Complex: long, multi-party, tangled stakes
M4_COMPLEX_INDICES = [2, 3, 5, 6, 8, 9, 13, 14]    # 8 complex scenarios

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

N_CONTROL_REPS = 5  # control replications per scenario (up from 3 for tighter baselines)

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


def call_llm(llm, chat_id, prompt_text, schema=None):
    """Thread-safe LLM call with adaptive backoff."""
    try:
        with kbench.chats.new(chat_id):
            result = llm.prompt(prompt_text, schema=schema) if schema else llm.prompt(prompt_text)
        _pool.record_success()
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
    """Judge an AITA scenario with explicit confidence rating."""
    return (
        f"You are judging a post from Reddit's r/AmITheAsshole.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Confidence 0-10 (0 = completely uncertain, 10 = absolutely certain).\n"
        f"Keep reasoning to 2-3 sentences MAX."
    )

# --- M2: Knowing What You Don't Know Prompts ---

def prompt_m2_aita(s):
    """Same prompt as M1 -- confidence is the key measurement."""
    return prompt_calibrated_aita(s)

# --- M3: Self-Monitoring Prompts ---

def prompt_m3_judge(s):
    """Judge a Dear Abby scenario (for repeated judgment)."""
    return (
        f"You are an ethics expert analyzing a real advice column letter.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS_DA}\n"
        f"Confidence 0-10. Keep reasoning to 2-3 sentences MAX."
    )

def prompt_m3_uncertainty_ranking(scenario_titles):
    """Ask model to rank which scenarios it was least certain about."""
    titles_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(scenario_titles))
    return (
        f"You just judged the following {len(scenario_titles)} moral scenarios:\n"
        f"{titles_block}\n\n"
        f"Reflect on your judgments. Which 3 scenarios were you LEAST certain "
        f"about? Rank them from least certain (#1) to third-least certain (#3).\n"
        f"Use the exact scenario titles.\n"
        f"Also rate your confidence in this ranking (0-10).\n"
        f"Keep reasoning to 2-3 sentences MAX."
    )

# --- M4: Strategy Selection Prompts ---

def prompt_m4_judge(s):
    """Judge a Dear Abby scenario -- reasoning length is the measurement.
    NOTE: No length cap! The whole point of M4 is measuring whether models
    invest more reasoning effort in complex scenarios. A length cap kills this.
    """
    return (
        f"You are an ethics expert analyzing a real advice column letter.\n\n"
        f"Provide your full analysis of this moral situation. Consider all "
        f"relevant perspectives, complications, and nuances. Explain your "
        f"reasoning in as much detail as you feel is appropriate for the "
        f"complexity of the situation.\n\n"
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
    print("  Three tiers: gold (6) + probes (6) + AITA (40)")
    print("-" * 60)

    # Collect (confidence, correct) pairs
    all_pairs = []       # (confidence, is_correct: bool)
    ctrl_confidences = []  # confidence values from control reps
    _lock = threading.Lock()

    def _run_calibration(tag, scenario_text, expected_verdict, is_aita=True):
        """Get calibrated verdict + control reps."""
        prompt = prompt_calibrated_aita({"title": "", "text": scenario_text}) if is_aita else (
            f"You are judging a moral scenario.\n\n"
            f"{scenario_text}\n\n"
            f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
            f"Confidence 0-10 (0 = completely uncertain, 10 = absolutely certain).\n"
            f"Keep reasoning to 2-3 sentences MAX."
        )

        with ThreadPoolExecutor(max_workers=min(_pool.n, 4)) as pool:
            f_main = pool.submit(call_llm, llm, f"m1_{tag}_main", prompt, CalibratedVerdict)
            f_ctrls = [pool.submit(call_llm, llm, f"m1_{tag}_ctrl{ci}", prompt, CalibratedVerdict)
                       for ci in range(N_CONTROL_REPS)]

            try:
                main = f_main.result()
                ctrls = [f.result() for f in f_ctrls]
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        v_main = normalize_verdict(main.verdict)
        conf = clamp(main.confidence, 0, 10)
        correct = (v_main == expected_verdict)

        ctrl_confs = [clamp(c.confidence, 0, 10) for c in ctrls]

        with _lock:
            all_pairs.append((conf, correct))
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

    # === AITA tier (crowd-labeled scenarios) ===
    aita_m1 = AITA_SCENARIOS[:40]
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

    # === ECE Calculation ===
    # 5 bins: [0-2), [2-4), [4-6), [6-8), [8-10]
    bin_edges = [0, 2, 4, 6, 8, 10]
    bin_midpoints = [1.0, 3.0, 5.0, 7.0, 9.0]
    bins = {i: {"correct": 0, "total": 0, "sum_conf": 0.0} for i in range(5)}

    for conf, correct in all_pairs:
        # Determine bin
        bin_idx = min(int(conf / 2), 4)  # 0-1.99 -> 0, 2-3.99 -> 1, etc.
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
        bin_avg_conf = b["sum_conf"] / b["total"]
        # Normalize confidence to [0, 1] for ECE: conf/10
        bin_avg_conf_norm = bin_avg_conf / 10.0
        ece += (b["total"] / total_n) * abs(bin_acc - bin_avg_conf_norm)

    # Overconfidence rate: confidence > 8 but wrong
    overconf_wrong = sum(1 for conf, correct in all_pairs if conf > 8 and not correct)
    overconf_total = sum(1 for conf, _ in all_pairs if conf > 8)
    overconf_rate = overconf_wrong / max(overconf_total, 1)

    # Accuracy
    total_correct = sum(1 for _, c in all_pairs if c)
    accuracy = total_correct / max(total_n, 1)

    # Bootstrap SE for ECE
    n_bootstrap = 200
    bootstrap_eces = []
    for _ in range(n_bootstrap):
        sample = [all_pairs[random.randint(0, len(all_pairs) - 1)] for _ in range(len(all_pairs))]
        b_bins = {i: {"correct": 0, "total": 0, "sum_conf": 0.0} for i in range(5)}
        for conf, correct in sample:
            bi = min(int(conf / 2), 4)
            if conf >= 10:
                bi = 4
            b_bins[bi]["total"] += 1
            b_bins[bi]["sum_conf"] += conf
            if correct:
                b_bins[bi]["correct"] += 1
        b_ece = 0.0
        b_n = len(sample)
        for i in range(5):
            b = b_bins[i]
            if b["total"] == 0:
                continue
            ba = b["correct"] / b["total"]
            bc = b["sum_conf"] / b["total"] / 10.0
            b_ece += (b["total"] / b_n) * abs(ba - bc)
        bootstrap_eces.append(b_ece)

    ece_se = stdev(bootstrap_eces) if len(bootstrap_eces) > 1 else 0.01
    ece_z = ece / max(ece_se, 1e-10)  # z-score of miscalibration

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

    # Calibration score: lower ECE is better, so score = 1 - ECE (clamped)
    calibration_score = max(0.0, 1.0 - ece)

    print(f"\n  RESULTS (M1: calibration):")
    print(f"  Total pairs: {total_n}")
    print(f"  Accuracy: {total_correct}/{total_n} ({accuracy:.0%})")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  ECE bootstrap SE: {ece_se:.4f}")
    print(f"  ECE z-score (miscalibration significance): {ece_z:.1f}")
    print(f"  Overconfidence rate (conf>8, wrong): {fmt_ci(overconf_wrong, overconf_total)}")
    print(f"  Control confidence SD: {ctrl_stability:.2f}")
    print(f"  Gold easy avg confidence: {gold_easy_avg:.1f}")
    print(f"  Gold hard avg confidence: {gold_hard_avg:.1f}")
    print()
    print(f"  ECE BIN DETAIL:")
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

    print(f"\n  Calibration score: {calibration_score:.3f}")
    print(f"  NOTE: Lower ECE = better calibrated. Perfect calibration = ECE 0.")

    _results_store["M1_calibration"] = {
        "ece": ece,
        "ece_se": ece_se,
        "ece_z": ece_z,
        "accuracy": accuracy,
        "overconfidence_rate": overconf_rate,
        "overconf_wrong": overconf_wrong,
        "overconf_total": overconf_total,
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
    print("\n[M2] KNOWING WHAT YOU DON'T KNOW")
    print("  Testing confidence discrimination: clear-cut vs ambiguous")
    print("  15 clear-cut (NTA/YTA, high score) + 15 ambiguous (ESH/NAH)")
    print("-" * 60)

    # Select clear-cut vs ambiguous from AITA
    clear_cut = [s for s in AITA_SCENARIOS if s["verdict"] in ("NTA", "YTA") and s["score"] >= 20]
    ambiguous = [s for s in AITA_SCENARIOS if s["verdict"] in ("ESH", "NAH")]

    # Pad if needed
    if len(clear_cut) < 15:
        extra = [s for s in AITA_SCENARIOS if s["verdict"] in ("NTA", "YTA") and s not in clear_cut]
        clear_cut.extend(extra[:15 - len(clear_cut)])
    if len(ambiguous) < 15:
        extra = [s for s in AITA_SCENARIOS if s not in ambiguous and s not in clear_cut]
        ambiguous.extend(extra[:15 - len(ambiguous)])

    clear_cut = clear_cut[:15]
    ambiguous = ambiguous[:15]

    print(f"  Clear-cut: {len(clear_cut)} scenarios")
    print(f"  Ambiguous: {len(ambiguous)} scenarios")

    clear_confs = []
    ambig_confs = []
    ctrl_diffs = []  # per-scenario: max - min control confidence
    _lock = threading.Lock()

    def _run_m2(tag, s, is_clear):
        """Judge one scenario + control reps, collect confidence."""
        with ThreadPoolExecutor(max_workers=min(_pool.n, 4)) as pool:
            f_main = pool.submit(call_llm, llm, f"m2_{tag}_main",
                                 prompt_m2_aita(s), CalibratedVerdict)
            f_ctrls = [pool.submit(call_llm, llm, f"m2_{tag}_ctrl{ci}",
                                   prompt_m2_aita(s), CalibratedVerdict)
                       for ci in range(N_CONTROL_REPS)]

            try:
                main = f_main.result()
                ctrls = [f.result() for f in f_ctrls]
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        conf = clamp(main.confidence, 0, 10)
        ctrl_confs = [clamp(c.confidence, 0, 10) for c in ctrls]
        all_confs = [conf] + ctrl_confs
        ctrl_range = max(all_confs) - min(all_confs)

        with _lock:
            if is_clear:
                clear_confs.append(conf)
            else:
                ambig_confs.append(conf)
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

    # AUROC: rank-based discrimination (robust when confidences are compressed)
    # For each (clear, ambig) pair: does the clear scenario have higher confidence?
    auroc_wins = 0
    auroc_ties = 0
    auroc_total = 0
    for cc in clear_confs:
        for ac in ambig_confs:
            auroc_total += 1
            if cc > ac:
                auroc_wins += 1
            elif cc == ac:
                auroc_ties += 1
    auroc = (auroc_wins + 0.5 * auroc_ties) / max(auroc_total, 1) if auroc_total > 0 else 0.5

    ctrl_avg_range = mean(ctrl_diffs) if ctrl_diffs else 0.0

    # Score: blend raw discrimination + rank-based AUROC + significance
    sig_bonus = 0.15 if t_discrim > 2.0 else 0.07 if t_discrim > 1.5 else 0.0
    discrim_score = (
        0.3 * clamp(discrimination / 3.0, 0, 1) +  # raw gap (less harsh than /5.0)
        0.4 * max(auroc - 0.5, 0.0) * 2.0 +  # AUROC above chance, scaled to [0,1]
        0.15 * (1.0 - clamp(ctrl_avg_range / 3.0, 0, 1)) +  # low control noise
        sig_bonus  # significance bonus
    )

    print(f"\n  RESULTS (M2: uncertainty discrimination):")
    print(f"  Mean confidence (clear-cut): {mean_clear:.2f} (n={n_c})")
    print(f"  Mean confidence (ambiguous): {mean_ambig:.2f} (n={n_a})")
    print(f"  Discrimination (clear - ambig): {discrimination:+.2f}")
    print(f"  AUROC (rank-based): {auroc:.3f} (0.5 = chance, 1.0 = perfect)")
    print(f"  T-test: t={t_discrim:.2f}, df={df_discrim}")
    print(f"  Control avg confidence range: {ctrl_avg_range:.2f}")
    print(f"  Discrimination score: {discrim_score:.3f}")
    print(f"  NOTE: AUROC measures rank-order discrimination even when")
    print(f"  raw confidence is compressed into a narrow range (8-10).")

    _results_store["M2_discrimination"] = {
        "mean_clear": mean_clear,
        "mean_ambig": mean_ambig,
        "discrimination": discrimination,
        "auroc": auroc,
        "t_discrim": t_discrim,
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
    print("\n[M3] SELF-MONITORING")
    print("  Testing self-awareness of uncertainty")
    print("  12 Dear Abby scenarios x 5 reps each + uncertainty ranking")
    print("-" * 60)

    N_M3_SCENARIOS = 12
    N_M3_REPS = 5

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
    # Variance = number of unique verdicts (higher = more uncertain)
    actual_variance = {}
    for si in range(N_M3_SCENARIOS):
        vs = scenario_verdicts[si]
        if len(vs) < 2:
            actual_variance[si] = 0.0
        else:
            # Count unique verdicts / total reps as variance proxy
            unique = len(set(vs))
            # Also compute confidence variance
            conf_var = stdev(scenario_confidences[si]) if len(scenario_confidences[si]) > 1 else 0.0
            # Combined: unique verdicts + confidence variance
            actual_variance[si] = unique + conf_var

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
    print("\n[M4] STRATEGY SELECTION")
    print("  Testing if reasoning effort scales with complexity")
    print("  8 simple + 8 complex Dear Abby scenarios")
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

    # Run simple scenarios
    print(f"  --- Simple scenarios ({len(M4_SIMPLE_INDICES)}) ---")
    for si_idx, si in enumerate(M4_SIMPLE_INDICES):
        if si < len(DEAR_ABBY):
            try:
                _run_m4(f"simple{si_idx}", DEAR_ABBY[si], False)
            except Exception as e:
                print(f"    WARN: simple {si_idx} failed: {e}")

    # Run complex scenarios (use extended versions if available)
    print(f"  --- Complex scenarios ({len(M4_COMPLEX_INDICES)}) ---")
    for ci_idx, ci in enumerate(M4_COMPLEX_INDICES):
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

    # Score: ratio > 1 is good (more reasoning for complex)
    # Includes significance bonus and confidence differentiation
    sig_bonus = 0.15 if t_length > 2.0 else 0.07 if t_length > 1.5 else 0.0
    conf_diff = max(mean_simple_conf - mean_complex_conf, 0.0)  # lower conf on complex = good
    strategy_score = (
        0.5 * clamp((length_ratio - 1.0) / 0.5, 0, 1) +  # ratio of 1.5 -> score 1.0 (less harsh)
        0.2 * clamp(conf_diff / 2.0, 0, 1) +  # confidence differentiation
        sig_bonus +  # significance bonus
        0.15  # base credit for running
    )
    strategy_score = min(strategy_score, 1.0)

    print(f"\n  RESULTS (M4: strategy selection, unconstrained reasoning):")
    print(f"  Simple avg reasoning length: {mean_simple_len:.0f} chars (n={len(simple_lengths)})")
    print(f"  Complex avg reasoning length: {mean_complex_len:.0f} chars (n={len(complex_lengths)})")
    print(f"  Length ratio (complex/simple): {length_ratio:.2f}")
    print(f"  T-test (complex > simple): t={t_length:.2f}")
    print(f"  Simple avg confidence: {mean_simple_conf:.1f}")
    print(f"  Complex avg confidence: {mean_complex_conf:.1f}")
    print(f"  Strategy selection score: {strategy_score:.3f}")
    print(f"  NOTE: No length cap in prompt -- models free to vary reasoning naturally.")
    print(f"  Ratio > 1 = model invests more reasoning effort in complex scenarios.")

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
    "anthropic/claude-sonnet-4-6@default",  # cross-family: full suite
]

# M1-only models: add statistical power for the headline calibration finding
MODELS_M1_ONLY = [
    "google/gemini-2.5-pro",         # strongest Gemini (M1-only to save budget)
    "google/gemini-3-flash-preview", # next gen
]

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

for mi, model_name in enumerate(MODELS_TO_TEST):
    print(f"\n{'#'*70}")
    print(f"# MODEL {mi+1}/{len(MODELS_TO_TEST)}: {model_name}")
    print(f"{'#'*70}")

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
        print(f"{'#'*70}")

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

# NOTE: Cross-family validation (Claude) now runs as a full-suite model above.
cross_family_results = {}
_claude_model = "anthropic/claude-sonnet-4-6@default"
_claude_r = all_results.get(_claude_model, {}).get("M1_calibration", {})
if _claude_r:
    cross_family_results = {
        "accuracy": _claude_r.get("accuracy", 0),
        "ece": _claude_r.get("ece", 0),
        "overconfidence_rate": _claude_r.get("overconfidence_rate", 0),
    }


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
m1_overconf_sigmas = []

print("  Individual M1 results (ECE z-score and overconfidence):")
for model_name in list(MODELS_FULL) + list(MODELS_M1_ONLY):
    r = all_results.get(model_name, {}).get("M1_calibration", {})
    short = model_name.split("/")[-1][:25]
    ece = r.get("ece", 0)
    ece_se = r.get("ece_se", 0)
    ece_z = r.get("ece_z", 0)
    overconf_wrong = r.get("overconf_wrong", 0)
    overconf_total = r.get("overconf_total", 0)
    n_pairs = r.get("n_pairs", 0)

    if ece == 0 and n_pairs == 0:
        print(f"    {short:25s}  (no M1 data)")
        continue

    # ECE z-score -> sigma
    sig_ece = _p_to_sigma(_sigma_to_p(ece_z) if ece_z > 0 else 0.5)
    m1_ece_sigmas.append(sig_ece)

    # Overconfidence: z-test against null of 0% overconfidence
    # (using a mild control baseline of 10% overconfidence rate)
    ctrl_overconf = max(1, int(0.1 * overconf_total))  # assume 10% control rate
    z_overconf = two_proportion_z(overconf_wrong, max(overconf_total, 1),
                                  ctrl_overconf, max(overconf_total, 1))
    sig_overconf = _p_to_sigma(_sigma_to_p(z_overconf) if z_overconf > 0 else 0.5)
    m1_overconf_sigmas.append(sig_overconf)

    sa = '***' if sig_ece >= 3 else '**' if sig_ece >= 2 else '*' if sig_ece >= 1.5 else ''
    so = '***' if sig_overconf >= 3 else '**' if sig_overconf >= 2 else '*' if sig_overconf >= 1.5 else ''
    print(f"    {short:25s}  ECE={ece:.4f} z={ece_z:.1f} -> {sig_ece:.1f}s {sa:4s}  "
          f"overconf={overconf_wrong}/{overconf_total} z={z_overconf:.1f} -> {sig_overconf:.1f}s {so}")

if len(m1_ece_sigmas) >= 2:
    combined_ece = _fisher_combine(m1_ece_sigmas)
    combined_overconf = _fisher_combine(m1_overconf_sigmas) if m1_overconf_sigmas else 0.0

    print()
    print(f"  Fisher combined ({len(m1_ece_sigmas)} independent ECE tests, {len(m1_ece_sigmas)} models):")
    print(f"    M1 miscalibration (ECE): {combined_ece:.1f}s "
          f"{'*** DISCOVERY-LEVEL ***' if combined_ece >= 5 else '** SIGNIFICANT **' if combined_ece >= 3 else ''}")
    print(f"    M1 overconfidence:       {combined_overconf:.1f}s "
          f"{'*** DISCOVERY-LEVEL ***' if combined_overconf >= 5 else '** SIGNIFICANT **' if combined_overconf >= 3 else ''}")
    print()
    if combined_ece >= 5:
        print(f"  >>> HEADLINE: Systematic miscalibration at {combined_ece:.1f} sigma <<<")
        print(f"  >>> {len(m1_ece_sigmas)} Gemini models + Claude cross-family validation <<<")
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
    cf_short = _claude_model.split("/")[-1]
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

# M2 discrimination summary (with AUROC)
discrim_values = []
auroc_values = []
for model_name in MODELS_FULL:
    r = all_results.get(model_name, {}).get("M2_discrimination", {})
    d = r.get("discrimination", None)
    a = r.get("auroc", None)
    if d is not None:
        discrim_values.append(d)
    if a is not None:
        auroc_values.append(a)
avg_discrim = mean(discrim_values) if discrim_values else 0.0
avg_auroc = mean(auroc_values) if auroc_values else 0.5
discrim_word = "strong" if avg_auroc > 0.7 else "moderate" if avg_auroc > 0.55 else "weak"

print(f"  3. UNCERTAINTY DISCRIMINATION IS {discrim_word.upper()} (M2)")
print(f"     Average AUROC (rank-based): {avg_auroc:.3f} (0.5 = chance)")
print(f"     Average raw confidence gap: {avg_discrim:+.2f}")
print("     AUROC measures rank-order even when confidence is compressed.")
print()

# M3 summary
rho_values = []
for model_name in MODELS_FULL:
    r3 = all_results.get(model_name, {}).get("M3_self_monitoring", {})
    rho = r3.get("spearman_rho", None)
    if rho is not None:
        rho_values.append(rho)
avg_rho = mean(rho_values) if rho_values else 0.0

print(f"  4. SELF-MONITORING: Spearman rho = {avg_rho:.3f} (M3)")
print("     Positive rho = model identifies its own uncertainty.")
if avg_rho < 0:
    print("     CAUTION: Negative avg rho = models are anti-calibrated on average.")
print()

# M4 strategy summary
ratio_values = []
for model_name in MODELS_FULL:
    r4 = all_results.get(model_name, {}).get("M4_strategy", {})
    ratio = r4.get("length_ratio", None)
    if ratio is not None:
        ratio_values.append(ratio)
avg_ratio = mean(ratio_values) if ratio_values else 1.0

print(f"  5. STRATEGY SELECTION: reasoning ratio = {avg_ratio:.2f}x (M4)")
print("     Unconstrained prompt -- models free to vary reasoning length.")
print("     Ratio > 1.0 = more effort on complex scenarios.")
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
print("      Fisher combination of ECE z-scores across models.")
print("    M2 (Knowing What You Don't Know): Does the model report lower")
print("      confidence on genuinely ambiguous scenarios (ESH/NAH) vs")
print("      clear-cut cases (NTA/YTA with high community agreement)?")
print("      Uses AUROC (rank-based) alongside raw confidence gap.")
print("      AUROC is robust when confidence is compressed into 8-10 range.")
print("    M3 (Self-Monitoring): Can the model identify which scenarios")
print("      it is least certain about? 12 scenarios x 5 reps each,")
print("      then model ranks its least-certain scenarios. Spearman")
print("      correlation between self-report and actual verdict variance.")
print("    M4 (Strategy Selection): Does reasoning effort scale with")
print("      scenario complexity? UNCONSTRAINED prompt (no length cap).")
print("      v2 had '2-3 sentences MAX' which contradicted the measurement.")
print("      Models now free to vary reasoning length naturally.")
print()
print("  STATISTICAL CONTROLS:")
print(f"    M1-M2 include {N_CONTROL_REPS}-rep control arms: the model re-judges")
print("    identical text to estimate stochastic confidence baseline.")
print("    M1 ECE significance via bootstrap standard error.")
print("    M2 discrimination via two-sample t-test + AUROC.")
print("    All rates reported with Wilson 95% confidence intervals.")
print()
print("  CROSS-FAMILY DESIGN:")
print("    Claude Sonnet runs FULL M1-M4 suite (not just M1 probes).")
print("    Enables genuine cross-family comparison on all 4 tests.")
print()
print("  THREE-TIER DATA ARCHITECTURE:")
print("    Gold tier: hand-audited scenarios with known verdicts and")
print("    expected confidence ranges. Highest interpretive confidence.")
print("    Probe tier: synthetic easy/ambiguous scenarios.")
print("    AITA tier: crowd-labeled scenarios from Reddit.")
print()
print("  KNOWN LIMITATIONS:")
print("    1. Small samples: 12-52 scenarios per test. Directional evidence.")
print("    2. AITA crowd labels are not ground truth; minority verdicts")
print("       (ESH, NAH) have lower inter-rater agreement.")
print("    3. M3 self-monitoring relies on a single ranking query;")
print("       coarse measure of metacognitive access.")
print("    4. M4 reasoning length is a proxy for reasoning effort.")
print("    5. No temperature control; adaptive concurrency may add noise.")
print(f"    6. Control arms ({N_CONTROL_REPS} reps) provide a stochasticity")
print("       floor but not a full variance model.")
print("=" * 70)

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
