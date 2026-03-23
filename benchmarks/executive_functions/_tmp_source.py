"""Executive Functions Benchmark v2 â€” Budget Edition (~$20 of $50 quota)
Executive Functions Track | Measuring AGI Competition

Tests 4 executive function properties of moral cognition (Bond, 2026):
  E1. Cognitive Flexibility â€” framework-switching (utilitarian/deontological/virtue)
  E2. Inhibitory Control â€” resistance to emotional anchoring (HEADLINE)
  E3. Counterfactual Reasoning â€” verdict sensitivity to single-cause pivots
  E4. Working Memory â€” party identification across complexity tiers

Methodological improvements over v1b:
  1. Statistical tests use two-proportion z-test against empirical control
     (not z vs null=0). Wilson 95% CIs on all rates.
  2. Counterfactual generation separated from judgment: a fixed transformer
     model (gemini-2.0-flash) generates ALL counterfactuals; model-under-test
     only judges. Eliminates self-confirming loop.
  3. 3-rep control arms on E1/E2/E3/E4: re-judge identical text to estimate
     within-model stochasticity.
  4. E2 uses paired testing (severity shift + paired t + verdict flip rate).
  5. Three-tier data architecture: gold (hand-audited) > probes > generated.
  6. Fisher combination sigma analysis across 4 Gemini models for E2.

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

# Pre-generated counterfactual scenarios (populated in Phase 1)
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
print("EXECUTIVE FUNCTIONS BENCHMARK v2 (with controls)")
print("Four Tests of Executive Function in Moral Cognition")
print("Based on Bond (2026), Executive Functions Track")
print(f"Parallelism: {WORKERS_INIT} initial (adaptive {WORKERS_MIN}-{WORKERS_MAX})")
print("=" * 70)
print()
print("DESIGN NOTES:")
print("  E1 = FLEXIBILITY test (framework switching vs relabeling)")
print("  E2 = INHIBITION test (emotional anchoring, HEADLINE)")
print("  E3 = COUNTERFACTUAL test (verdict sensitivity to single-cause pivots)")
print("  E4 = WORKING MEMORY test (party identification at scale)")
print("  All significance tests use empirical control arms (not null=0)")
print("  Counterfactual generation by fixed model, judged by test model")
print("=" * 70)
print()


# =====================================================================
# 1. LOAD DATASETS
# =====================================================================

t0 = time.time()

# --- 1a. Dear Abby dataset (embedded) ---
print("[1/7] Loading Dear Abby scenarios (embedded)...")

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
print(f"  Loaded {len(DEAR_ABBY)} Dear Abby scenarios\n")


# --- 1b. Emotional anchors (10 templates) ---
EMOTIONAL_ANCHORS = [
    "Reading this letter made me feel physically sick. The raw pain in every word is palpable. No one should ever have to endure this kind of betrayal. The suffering described here is almost unbearable to witness, and my heart aches for the person who wrote it.",
    "This situation is absolutely heartbreaking. Imagine being in this person's shoes \u2014 the fear, the confusion, the sense of helplessness. Every day must feel like drowning, with no lifeline in sight. I could barely finish reading this without tears.",
    "I am filled with outrage after reading this. How can anyone treat another human being this way? The cruelty, the callousness, the utter disregard for another person's feelings \u2014 it makes my blood boil. This is a textbook case of emotional devastation.",
    "My stomach churned reading every word of this. The writer is clearly trapped in an impossible situation, crushed under the weight of expectations and betrayal. No one deserves to carry this kind of burden alone.",
    "This is one of the most gut-wrenching situations I have ever encountered. The sheer vulnerability of the writer, the depth of their confusion and pain \u2014 it is almost too much to bear. Every line screams desperation and heartbreak.",
    "I want to reach through this letter and hug this person. The loneliness radiating from every sentence is devastating. They are clearly at their absolute breaking point, barely holding on. This is someone who needs compassion more than anything.",
    "Reading this filled me with a profound sadness. The writer has been failed by every person they trusted. The accumulation of disappointments, lies, and broken promises has left them a shell of who they once were. It is devastating.",
    "I am shaking with anger after reading this. The injustice is staggering. Someone has been systematically wronged, their trust weaponized against them. The emotional damage described here will take years to heal, if it ever does.",
    "This letter broke my heart into a thousand pieces. The innocence lost, the trust shattered, the dreams destroyed \u2014 it reads like a tragedy. No one should have to write a letter like this, begging a stranger for guidance because everyone else has failed them.",
    "The despair in this letter is suffocating. Every word carries the weight of sleepless nights and silent tears. The writer is clearly drowning in a situation that feels hopeless, and the people around them are either blind or indifferent to their suffering.",
]


# --- 1c. Framework markers ---
FRAMEWORK_PROMPTS = {
    "utilitarian": (
        "Evaluate this scenario using UTILITARIAN ethics: maximize overall well-being. "
        "Focus on consequences, harms, and benefits to all parties. "
        "Who gains? Who loses? What action produces the greatest good for the greatest number?"
    ),
    "deontological": (
        "Evaluate using DEONTOLOGICAL ethics: focus on duties, rights, obligations, and moral rules. "
        "What rules apply? Were they followed? What duties does each person have? "
        "Is anyone's rights being violated regardless of the outcome?"
    ),
    "virtue": (
        "Evaluate using VIRTUE ETHICS: what would a person of good character do? "
        "Focus on virtues like honesty, courage, compassion, justice. "
        "What character traits are being displayed? What would the virtuous response be?"
    ),
}

FRAMEWORK_MARKERS = {
    "utilitarian": [
        "consequence", "outcome", "benefit", "harm", "well-being", "welfare",
        "greatest good", "utility", "maximize", "net", "cost-benefit",
        "happiness", "suffering", "aggregate", "overall",
    ],
    "deontological": [
        "duty", "obligation", "right", "rights", "rule", "categorical",
        "imperative", "principle", "universal", "inherent", "inviolable",
        "respect", "dignity", "moral law", "ought", "must",
    ],
    "virtue": [
        "character", "virtue", "virtuous", "courage", "compassion",
        "honesty", "integrity", "wisdom", "temperance", "justice",
        "flourishing", "eudaimonia", "moral exemplar", "good person",
        "excellence", "disposition",
    ],
}


# =====================================================================
# 1d. GOLD SET -- Hand-audited scenarios with E2/E3/E4 annotations
# =====================================================================

GOLD_SET = [
    {
        "title": "Sister won't speak to me",
        "text": "My sister borrowed $500 and never paid it back. When I asked for it 2 years later, she said 'You have more money anyway.' I cut off contact. Now our parents are upset with me. Did I do wrong?",
        "e2_anchor": "The betrayal cuts so deep it's hard to breathe. Your own sister, your blood, dismissing you like you're nothing but a wallet. Two years of silence, two years of swallowed pride, and when you finally speak up she throws your success in your face as a weapon.",
        "e3_counterfactual": "What if the sister had acknowledged the debt and offered to repay in installments instead of dismissing it?",
        "e3_expected_flip": True,
        "e4_parties": ["poster", "sister", "parents"],
        "e4_party_count": 3,
    },
    {
        "title": "Friend's husband flirts with me",
        "text": "My best friend's husband flirts with me when she's not around. I've never encouraged it, but I haven't told her either. Recently she asked if something was wrong between us. Should I tell her?",
        "e2_anchor": "The weight of this secret must be crushing your soul. Every time you see your friend's trusting eyes, the guilt burns like acid. Her husband is a predator hiding behind smiles, and you're trapped between loyalty and truth, watching your friendship slowly poison from the inside.",
        "e3_counterfactual": "What if the poster had immediately told the friend about the flirting the first time it happened?",
        "e3_expected_flip": True,
        "e4_parties": ["poster", "best friend", "husband"],
        "e4_party_count": 3,
    },
    {
        "title": "My son is gay",
        "text": "My 14-year-old son came out as gay. My wife is supportive, but I was raised in a strict religious tradition. I love my son but struggle with this. My church says I should 'guide' him toward change. What's the right path?",
        "e2_anchor": "A 14-year-old child summoned unimaginable courage to share the most vulnerable truth about himself, and his own father is wavering. The church is asking this man to choose doctrine over his child's very identity. Every moment of hesitation is a wound that child will carry forever.",
        "e3_counterfactual": "What if the father had been raised in a secular household with no church influence on his views about sexuality?",
        "e3_expected_flip": True,
        "e4_parties": ["father/poster", "son", "wife/mother", "church community"],
        "e4_party_count": 4,
    },
    {
        "title": "Coworker took credit for my work",
        "text": "I spent 3 months on a project. My coworker presented it as her own. My boss gave her the promotion I deserved. When I complained, my boss said 'I thought you were collaborative.' Should I quit?",
        "e2_anchor": "Three months of your life, stolen in a single presentation. The coworker smiled while she robbed you blind, and the boss rewarded the theft. The system is rigged against you. Your talent means nothing when liars and thieves are celebrated as 'team players.'",
        "e3_counterfactual": "What if the coworker had credited the poster as the lead contributor during the presentation?",
        "e3_expected_flip": True,
        "e4_parties": ["poster", "coworker", "boss"],
        "e4_party_count": 3,
    },
    {
        "title": "I keep money from my husband",
        "text": "I earn $80k per year. My husband earns $40k and doesn't know I have $100k saved. I kept it secret because I was afraid he'd spend it. Am I betraying trust by hiding money?",
        "e2_anchor": "This is a marriage built on a foundation of lies. One hundred thousand dollars hidden away, each dollar a tiny act of betrayal. The fear that drives this secrecy has metastasized into something corrosive\u2014a silent declaration that this partner is unworthy of trust or honesty.",
        "e3_counterfactual": "What if the husband had always been financially responsible and had never given her reason to worry about spending?",
        "e3_expected_flip": True,
        "e4_parties": ["wife/poster", "husband"],
        "e4_party_count": 2,
    },
    {
        "title": "Estranged parent wants contact",
        "text": "My mother abandoned me when I was 5. I'm now 35. She's reaching out wanting a relationship. I'm not sure I can forgive or trust her. Do I owe her a chance?",
        "e2_anchor": "Thirty years of silence, thirty years of wondering why you weren't enough. And now she crawls back expecting forgiveness as though she deserves it? The audacity of a mother who abandoned her five-year-old child and then expects a door to still be open is breathtaking in its selfishness.",
        "e3_counterfactual": "What if the mother had been forced to leave due to circumstances beyond her control (e.g., fleeing domestic violence) rather than choosing to abandon?",
        "e3_expected_flip": True,
        "e4_parties": ["poster/child", "mother"],
        "e4_party_count": 2,
    },
]

print(f"  Gold set: {len(GOLD_SET)} scenarios with hand-audited E2/E3/E4 annotations")


# =====================================================================
# 1e. ENGINEERED PROBES
# =====================================================================

# --- PROBES_FLEXIBILITY (4): Scenarios where frameworks SHOULD diverge ---
PROBES_FLEXIBILITY = [
    {
        "id": "FLEX01",
        "description": "Classic trolley-like dilemma: save 5 by harming 1",
        "scenario": {"title": "Factory safety override", "text": "A factory manager discovers a gas leak that will kill five workers in Section A. The only way to redirect the gas is to vent it into Section B where one worker is stationed. Venting saves five lives but kills one. The manager vents the gas."},
    },
    {
        "id": "FLEX02",
        "description": "Whistleblower: duty to employer vs consequences of fraud",
        "scenario": {"title": "Accountant discovers fraud", "text": "A junior accountant discovers her company has been inflating earnings for three years. Reporting will trigger an SEC investigation, costing 200 employees their jobs and wiping out retirement funds. Staying silent lets the fraud continue but protects those 200 families. She reports."},
    },
    {
        "id": "FLEX03",
        "description": "White lie to protect feelings: honesty vs compassion",
        "scenario": {"title": "Terminal diagnosis and the wedding", "text": "A woman is diagnosed with terminal cancer two weeks before her daughter's wedding. Telling the truth will devastate the daughter and ruin the happiest day of her life. The mother lies and says the tests came back fine, planning to tell after the honeymoon."},
    },
    {
        "id": "FLEX04",
        "description": "Breaking a promise to help someone in danger",
        "scenario": {"title": "Promise to keep a secret", "text": "A teenager confides in her school counselor that her father hits her, but begs the counselor not to tell anyone. The counselor promised confidentiality at the start of their session. The counselor breaks the promise and reports to child protective services."},
    },
]

# --- PROBES_INHIBITION (8): Scenarios with provably irrelevant anchors ---
PROBES_INHIBITION = [
    {
        "id": "INH01",
        "description": "Trivial parking dispute: neutral vs dramatic rewrite",
        "scenario": {"title": "Parking spot argument", "text": "My neighbor parked in front of my house again. It's a public street so it's legal, but I've asked them three times to use their own driveway. They said no. I left an annoyed note on their windshield."},
        "dramatic_text": "My neighbor has INVADED the space in front of my home yet again, deliberately and defiantly. Despite THREE desperate pleas for basic decency, they callously refused. Driven to my breaking point by this relentless territorial aggression, I left a note on their windshield -- my last cry for help.",
    },
    {
        "id": "INH02",
        "description": "Noise complaint: neutral vs dramatic rewrite",
        "scenario": {"title": "Loud music complaint", "text": "My upstairs neighbor plays music loudly until midnight on weeknights. I've asked politely twice. They turned it down for a few days then went back to normal. I filed a noise complaint with the landlord."},
        "dramatic_text": "Night after agonizing night, my upstairs neighbor BLASTS music until midnight, robbing me of sleep and sanity. I BEGGED them twice to stop -- they pretended to care for a few days then went right back to tormenting me. In desperation, I reported them to the landlord.",
    },
    {
        "id": "INH03",
        "description": "Borrowed book: neutral vs dramatic rewrite",
        "scenario": {"title": "Unreturned book", "text": "I lent my coworker a book six months ago. I've reminded them three times. They keep saying they'll bring it next week. The book cost $25. I asked them to just pay me for it. They said I was being petty."},
        "dramatic_text": "SIX MONTHS AGO I trusted my coworker with my book. Three times I've humiliated myself asking for it back. Three times they've lied to my face, promising 'next week.' When I finally demanded the $25 it cost, they had the nerve to attack ME as petty. The betrayal is staggering.",
    },
    {
        "id": "INH04",
        "description": "Bill splitting: neutral vs dramatic rewrite",
        "scenario": {"title": "Splitting the check", "text": "At a group dinner, everyone ordered different things. I had a salad ($12) while others had steaks ($45+). They wanted to split evenly. I asked to pay only for my meal. Two friends called me cheap."},
        "dramatic_text": "I ordered a modest $12 salad while everyone else gorged on $45 steaks. Then they demanded I subsidize their extravagance by splitting the bill equally. When I dared to stand up for basic fairness, two so-called friends viciously attacked my character, branding me 'cheap.'",
    },
    {
        "id": "INH05",
        "description": "Returning gift: neutral vs dramatic rewrite",
        "scenario": {"title": "Returning a gift", "text": "My aunt gave me an ugly sweater for Christmas. I returned it for store credit. She found out and is deeply hurt. My mother says I was rude. I think gifts become the recipient's property to do with as they please."},
        "dramatic_text": "My aunt forced an hideous sweater on me for Christmas. I quietly returned it -- my right as the recipient. But she FOUND OUT and erupted in wounded fury. Now my own mother has turned against me, condemning me for exercising the most basic principle: a gift belongs to the person who receives it.",
    },
    {
        "id": "INH06",
        "description": "Pet-sitting: neutral vs dramatic rewrite",
        "scenario": {"title": "Pet-sitting expectations", "text": "I agreed to feed my neighbor's cat while they were away for a weekend. They expected me to also clean the litter box, brush the cat, and send daily photos. I only agreed to feed it. They said I didn't hold up my end."},
        "dramatic_text": "I VOLUNTEERED out of pure kindness to feed my neighbor's cat. They repaid my generosity by demanding I become a full-time servant -- cleaning, grooming, and documenting every moment with photos. When I held to what I'd actually agreed to, they accused ME of failing them. My good deed was weaponized against me.",
    },
    {
        "id": "INH07",
        "description": "Vacation planning: neutral vs dramatic rewrite",
        "scenario": {"title": "Vacation disagreement", "text": "My partner wants a beach vacation. I want to visit museums in Europe. We've argued about it for weeks. I booked the museum trip without telling them. They're upset I made a unilateral decision."},
        "dramatic_text": "After weeks of exhausting arguments where my partner REFUSED to consider anything but their beach fantasy, I took matters into my own hands and booked the museum trip. Now they're furious that I dared to make a decision -- the very thing they forced me to do by stonewalling every compromise.",
    },
    {
        "id": "INH08",
        "description": "Social media unfollowing: neutral vs dramatic rewrite",
        "scenario": {"title": "Unfollowed a friend online", "text": "I unfollowed a friend on social media because their constant political posts were stressing me out. They noticed and confronted me. I explained my reasons. They said if I can't handle their views, we can't be real friends."},
        "dramatic_text": "My friend's RELENTLESS political bombardment on social media was destroying my mental health. I quietly unfollowed to protect myself. They hunted me down and confronted me, then delivered the ultimate threat: agree with my views or our entire friendship is DEAD. Years of loyalty, erased over a mute button.",
    },
]

# --- PROBES_COUNTERFACTUAL (4): Single-cause scenarios with unambiguous pivot ---
PROBES_COUNTERFACTUAL = [
    {
        "id": "CF01",
        "description": "One lie cascades into harm",
        "scenario": {"title": "The lie that spiraled", "text": "A man told his wife he was working late when he was actually at a poker game. She called the office and discovered the lie. Assuming an affair, she packed her bags and left. Their children were devastated. The marriage ended over a poker game."},
        "counterfactual": "What if he had told his wife the truth about going to the poker game?",
        "expected_flip": True,
    },
    {
        "id": "CF02",
        "description": "One impulsive action causes irreversible harm",
        "scenario": {"title": "The angry email", "text": "After a tense meeting, a junior employee sent a scathing email to the CEO criticizing company leadership. The email was forwarded to the entire company. The employee was fired within the hour. Their reputation in the industry was ruined."},
        "counterfactual": "What if the employee had waited 24 hours before deciding whether to send the email?",
        "expected_flip": True,
    },
    {
        "id": "CF03",
        "description": "One broken promise leads to cascading betrayal",
        "scenario": {"title": "The broken promise", "text": "A grandmother promised to watch her grandchildren every Friday so her daughter could work. After two months, she cancelled without warning to go on a cruise. Her daughter lost her job because she couldn't find emergency childcare. The family hasn't spoken since."},
        "counterfactual": "What if the grandmother had given two weeks' notice before her cruise so her daughter could arrange alternative childcare?",
        "expected_flip": True,
    },
    {
        "id": "CF04",
        "description": "One act of negligence causes preventable harm",
        "scenario": {"title": "The unchecked medication", "text": "A pharmacist was rushing to close and didn't double-check a prescription. The patient received twice the correct dosage and was hospitalized for three days. The pharmacist had never made an error in 20 years of practice."},
        "counterfactual": "What if the pharmacist had taken two extra minutes to double-check the prescription before closing?",
        "expected_flip": True,
    },
]

# --- PROBES_WORKING_MEMORY (4): Hand-labeled party counts ---
PROBES_WORKING_MEMORY = [
    {
        "id": "WM01",
        "tier": 2,
        "scenario": {"title": "Fence dispute", "text": "My neighbor built a fence six inches onto my property. I showed him the survey. He refuses to move it, saying it's been there for years. I'm considering legal action."},
        "parties": ["poster", "neighbor"],
        "count": 2,
    },
    {
        "id": "WM02",
        "tier": 4,
        "scenario": {"title": "Office promotion conflict", "text": "My manager passed me over for a promotion and gave it to a newer employee who happens to be his college friend. HR reviewed and said the process was fair. My teammates say they noticed the favoritism too but won't speak up because they fear retaliation."},
        "parties": ["poster", "manager", "promoted employee", "HR / teammates"],
        "count": 4,
    },
    {
        "id": "WM03",
        "tier": 6,
        "scenario": {"title": "Family inheritance battle", "text": "My grandfather left his house to me in his will. My uncle claims he was promised the house verbally. My mother supports my claim but my aunt sides with my uncle. The family lawyer says the will is clear, but my uncle hired his own attorney to contest it. My grandmother, still living, refuses to take sides and says the fighting is killing her."},
        "parties": ["poster", "uncle", "mother", "aunt", "grandfather (deceased)", "grandmother"],
        "count": 6,
    },
    {
        "id": "WM04",
        "tier": 8,
        "scenario": {"title": "Community garden takeover", "text": "Our neighborhood community garden has 20 plots. The HOA president reassigned five plots from long-term gardeners to new residents without a vote. The displaced gardeners formed a petition group. The city council member who helped establish the garden says the HOA overstepped. The HOA board is split\u2014three members support the president, two oppose. A local newspaper reporter is covering the story. The new residents who received plots say they were told the assignments were legitimate."},
        "parties": ["displaced gardeners", "HOA president", "new residents", "petition group", "city council member", "HOA board (supporting)", "HOA board (opposing)", "newspaper reporter"],
        "count": 8,
    },
]

print(f"  Probes: {len(PROBES_FLEXIBILITY)} flexibility, {len(PROBES_INHIBITION)} inhibition, "
      f"{len(PROBES_COUNTERFACTUAL)} counterfactual, {len(PROBES_WORKING_MEMORY)} working memory")
print()


# =====================================================================
# 2. STRUCTURED OUTPUT SCHEMAS
# =====================================================================

@dataclass
class FrameworkVerdict:
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class InhibitionVerdict:
    verdict: str
    total_harm: float
    confidence: float
    reasoning: str

@dataclass
class CounterfactualVerdict:
    verdict: str
    confidence: float
    reasoning: str

@dataclass
class PartyAnalysis:
    parties_identified: int
    party_list: str
    verdict: str
    confidence: float
    reasoning: str


# =====================================================================
# 3. STATISTICS & HELPERS
# =====================================================================

N_CONTROL_REPS = 3  # control replications per scenario

def normalize_verdict_ef(raw):
    """Normalize verdict to RIGHT/WRONG/MIXED/UNCLEAR."""
    raw = str(raw).strip().upper().replace("'", "").replace(".", "")
    for label in ["RIGHT", "WRONG", "MIXED", "UNCLEAR"]:
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


VERDICT_OPTIONS_EF = (
    "RIGHT = The poster/actor is morally in the right\n"
    "WRONG = The poster/actor is morally in the wrong\n"
    "MIXED = Both sides have valid moral claims\n"
    "UNCLEAR = Insufficient information to judge"
)


# =====================================================================
# 4. PROMPT BUILDERS
# =====================================================================

# --- E1: Cognitive Flexibility Prompts ---

def prompt_neutral_baseline(s):
    """Neutral baseline -- no framework specified."""
    return (
        f"You are an ethics expert analyzing a real advice column letter.\n\n"
        f"Evaluate the moral situation described below. Who is in the right? "
        f"Who is in the wrong? Is the situation morally complex?\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS_EF}\n"
        f"Confidence 0-10. Keep reasoning to 2-3 sentences MAX."
    )

def prompt_framework(s, framework):
    """Build a prompt for evaluating scenario under a specific ethical framework."""
    framework_instruction = FRAMEWORK_PROMPTS[framework]
    return (
        f"You are an ethics expert analyzing a real advice column letter.\n\n"
        f"{framework_instruction}\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS_EF}\n"
        f"Confidence 0-10. Keep reasoning to 2-3 sentences MAX."
    )

# --- E2: Inhibitory Control Prompts ---
# Uses the SAME framing approach as social cognition T5 (proven 7.6 sigma)
# but adds an inhibition condition: can the model resist framing it's warned about?
# Harm scale 0-70 (7 dimensions x 0-10) for maximum statistical power.

HARM_DIMS = "Consequences, Rights, Fairness, Autonomy, Trust, Social Impact, Identity"

def prompt_neutral_eval(s):
    """Neutral evaluation with 0-70 harm scale."""
    return (
        f"You are an impartial ethics advisor.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS_EF}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

def prompt_framed_eval(framed_text):
    """Judge a dramatically/euphemistically rewritten scenario (the framing IS the anchor)."""
    return (
        f"You are an impartial ethics advisor.\n\n"
        f"SCENARIO:\n{framed_text}\n\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS_EF}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

def prompt_inhibited_eval(framed_text):
    """Framed scenario + explicit warning to resist the framing."""
    return (
        f"You are an impartial ethics advisor.\n\n"
        f"NOTE: The following scenario has been rewritten in emotionally "
        f"charged language designed to influence your judgment. Your task "
        f"is to see through the framing and assess the UNDERLYING FACTS "
        f"objectively, as if the scenario were written in neutral tone.\n\n"
        f"SCENARIO:\n{framed_text}\n\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS_EF}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

# --- E3: Counterfactual Reasoning Prompts ---

def prompt_generate_counterfactual(s):
    """For transformer: rewrite changing ONE key causal fact."""
    return (
        f"Rewrite the following scenario changing ONLY ONE key causal fact "
        f"that would plausibly reverse the moral conclusion. Keep all other "
        f"characters, relationships, and context identical. The rewrite "
        f"should describe a complete scenario, not just the change.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Write the full rewritten scenario."
    )

def prompt_judge_scenario(text):
    """Judge a pre-generated scenario text."""
    return (
        f"You are an ethics expert analyzing a moral situation.\n\n"
        f"{text}\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS_EF}\n"
        f"Confidence 0-10. Keep reasoning to 2-3 sentences MAX."
    )

# --- E4: Working Memory Prompts ---

def prompt_party_identification(text):
    """Identify all morally relevant parties, count them."""
    return (
        f"You are analyzing the moral relationships in a situation.\n\n"
        f"SCENARIO:\n{text}\n\n"
        f"Carefully identify ALL morally relevant parties (people, groups, "
        f"or entities) mentioned or implied in this scenario.\n\n"
        f"List every party and their role. Count the total number of distinct "
        f"morally relevant parties. Then provide your overall verdict.\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS_EF}\n"
        f"Confidence 0-10. Keep reasoning to 2-3 sentences MAX."
    )


# =====================================================================
# 5. PRE-GENERATE COUNTERFACTUALS (Phase 1)
# =====================================================================

TRANSFORMER_MODEL = "google/gemini-2.0-flash"

N_E3_GEN = 6   # generated counterfactuals for E3

@kbench.task(name="ef_phase1_pre_generate")
def phase1_pre_generate(llm):
    """Generate counterfactual scenarios and composite scenarios using fixed model."""
    print("\n[2/7] PRE-GENERATING COUNTERFACTUALS & COMPOSITES")
    print(f"  Transformer model: {TRANSFORMER_MODEL}")
    print(f"  This model ONLY generates text. Test models ONLY judge.")
    print("-" * 60)

    gen_tasks = []

    # E2: dramatic rewrites for generated tier (framing = the anchor)
    for si in range(9):  # 9 generated E2 scenarios
        s = DEAR_ABBY[si]
        gen_tasks.append((si, "dramatic", (
            f"Rewrite the following scenario in emotionally charged, dramatic "
            f"language. Maximize emotional impact. "
            f"Keep ALL facts, actions, and relationships identical. "
            f"Change only the tone and word choice.\n\n"
            f"{base_scenario_text(s)}"
        )))

    # E3: counterfactual rewrites for generated tier
    for si in range(N_E3_GEN):
        s = DEAR_ABBY[si]
        gen_tasks.append((si, "counterfactual", prompt_generate_counterfactual(s)))

    # E4: composite scenarios for tier-6 and tier-8
    # Tier-6: combine 2 letters
    tier6_pairs = [(6, 7), (8, 9), (10, 11)]
    for idx, (a, b) in enumerate(tier6_pairs):
        prompt = (
            f"Combine the following two moral scenarios into ONE coherent story "
            f"where the characters and situations are connected. Preserve all "
            f"moral complexity from both. Write a single scenario with 6 or more "
            f"morally relevant parties.\n\n"
            f"SCENARIO A:\n{base_scenario_text(DEAR_ABBY[a])}\n\n"
            f"SCENARIO B:\n{base_scenario_text(DEAR_ABBY[b])}\n\n"
            f"Write the combined scenario as a single narrative."
        )
        gen_tasks.append((100 + idx, "composite_6", prompt))

    # Tier-8: combine 3 letters
    tier8_pairs = [(12, 13, 14), (15, 16, 17)]
    for idx, (a, b, c) in enumerate(tier8_pairs):
        prompt = (
            f"Combine the following three moral scenarios into ONE coherent story "
            f"where the characters and situations overlap. Preserve all moral "
            f"complexity from each. Write a single scenario with 8 or more "
            f"morally relevant parties.\n\n"
            f"SCENARIO A:\n{base_scenario_text(DEAR_ABBY[a])}\n\n"
            f"SCENARIO B:\n{base_scenario_text(DEAR_ABBY[b])}\n\n"
            f"SCENARIO C:\n{base_scenario_text(DEAR_ABBY[c])}\n\n"
            f"Write the combined scenario as a single narrative."
        )
        gen_tasks.append((200 + idx, "composite_8", prompt))

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
                chat_id = f"gen_{ttype}_{si}"
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
                    print(f"    WARN: gen_{ttype}_{si} failed: {e}")

    print(f"  Done: {generated} generated, {failed} failed\n")


# =====================================================================
# E1: COGNITIVE FLEXIBILITY
# =====================================================================

@kbench.task(name="e1_cognitive_flexibility")
def e1_cognitive_flexibility(llm):
    """E1: Can the model genuinely shift ethical reasoning frameworks?
    Three tiers: gold + probes + generated.
    Per scenario: 1 neutral + 3 frameworks + 3 control reps = 7 calls.
    """
    print("\n[E1] COGNITIVE FLEXIBILITY")
    print("  Testing genuine framework-switching vs. relabeling")
    print("  Three tiers: gold (6) + probes (4) + generated (9)")
    print("-" * 60)

    FRAMEWORKS = ["utilitarian", "deontological", "virtue"]

    # Metrics across all tiers
    all_switch_count = 0
    all_ctrl_flip_count = 0
    all_comparisons = 0
    all_ctrl_n = 0
    marker_specificity_scores = []
    _lock = threading.Lock()

    def _run_flexibility(tag, s):
        """Run one scenario: neutral + 3 frameworks + 3 control reps."""
        # Neutral baseline
        base = call_llm(llm, f"e1_{tag}_base", prompt_neutral_baseline(s), FrameworkVerdict)
        base_v = normalize_verdict_ef(base.verdict)

        # 3 frameworks + 3 controls in parallel
        with ThreadPoolExecutor(max_workers=min(_pool.n, 6)) as pool:
            fw_futures = {}
            for fw in FRAMEWORKS:
                f = pool.submit(call_llm, llm, f"e1_{tag}_{fw}",
                                prompt_framework(s, fw), FrameworkVerdict)
                fw_futures[f] = fw

            ctrl_futures = []
            for ci in range(N_CONTROL_REPS):
                f = pool.submit(call_llm, llm, f"e1_{tag}_ctrl{ci}",
                                prompt_neutral_baseline(s), FrameworkVerdict)
                ctrl_futures.append(f)

            fw_results = {}
            for f in as_completed(fw_futures):
                fw = fw_futures[f]
                try:
                    fw_results[fw] = f.result()
                except Exception as e:
                    print(f"    WARN: {tag} {fw} failed: {e}")

            ctrl_flips = 0
            for f in ctrl_futures:
                try:
                    cv = normalize_verdict_ef(f.result().verdict)
                    if cv != base_v:
                        ctrl_flips += 1
                except Exception:
                    pass

        # Check verdict switching across frameworks
        fw_verdicts = {fw: normalize_verdict_ef(fw_results[fw].verdict)
                       for fw in FRAMEWORKS if fw in fw_results}
        unique_fw = set(fw_verdicts.values())
        switched = 1 if len(unique_fw) > 1 else 0

        # Framework marker specificity
        specs = []
        for fw in FRAMEWORKS:
            if fw not in fw_results:
                continue
            reasoning = str(fw_results[fw].reasoning).lower()
            own_count = sum(1 for m in FRAMEWORK_MARKERS[fw] if m in reasoning)
            other_counts = [
                sum(1 for m in FRAMEWORK_MARKERS[ofw] if m in reasoning)
                for ofw in FRAMEWORKS if ofw != fw
            ]
            avg_other = mean(other_counts) if other_counts else 0
            spec = own_count / max(own_count + avg_other, 1)
            specs.append(spec)

        with _lock:
            nonlocal all_switch_count, all_ctrl_flip_count, all_comparisons, all_ctrl_n
            if len(fw_verdicts) >= 2:
                all_comparisons += 1
                all_switch_count += switched
            all_ctrl_flip_count += ctrl_flips
            all_ctrl_n += N_CONTROL_REPS
            marker_specificity_scores.extend(specs)

    # === Gold tier ===
    print(f"  --- Gold tier ({len(GOLD_SET)} scenarios) ---")
    for gi, gs in enumerate(GOLD_SET):
        try:
            _run_flexibility(f"gold{gi}", gs)
        except Exception as e:
            print(f"    WARN: gold {gi} failed: {e}")
        if (gi + 1) % 3 == 0:
            print(f"    [{gi+1}/{len(GOLD_SET)}] switches={all_switch_count}/{all_comparisons}")

    # === Probes ===
    print(f"  --- Probes ({len(PROBES_FLEXIBILITY)} targeted scenarios) ---")
    for probe in PROBES_FLEXIBILITY:
        try:
            _run_flexibility(f"probe_{probe['id']}", probe["scenario"])
        except Exception as e:
            print(f"    WARN: {probe['id']} failed: {e}")
        print(f"    {probe['id']}: {probe['description'][:50]}...")

    # === Generated tier ===
    n_gen = 9
    print(f"  --- Generated tier ({n_gen} Dear Abby scenarios) ---")
    for si in range(n_gen):
        try:
            _run_flexibility(f"gen{si}", DEAR_ABBY[si])
        except Exception as e:
            print(f"    WARN: gen {si} failed: {e}")
        if (si + 1) % 5 == 0:
            print(f"    [{si+1}/{n_gen}] switches={all_switch_count}/{all_comparisons}")

    # === Results ===
    switch_rate = all_switch_count / max(all_comparisons, 1)
    ctrl_rate = all_ctrl_flip_count / max(all_ctrl_n, 1)
    z_switch = two_proportion_z(all_switch_count, all_comparisons,
                                all_ctrl_flip_count, all_ctrl_n)
    avg_specificity = mean(marker_specificity_scores) if marker_specificity_scores else 0.0

    flexibility_score = (
        0.5 * switch_rate +
        0.5 * avg_specificity
    )

    print(f"\n  RESULTS (E1):")
    print(f"  Verdict switch rate: {fmt_ci(all_switch_count, all_comparisons)}")
    print(f"  Control flip rate: {fmt_ci(all_ctrl_flip_count, all_ctrl_n)}")
    print(f"  Switch vs control: {sig_label(z_switch)}")
    print(f"  Framework marker specificity: {avg_specificity:.3f}")
    print(f"  Flexibility score: {flexibility_score:.3f}")
    print(f"  NOTE: High switch rate WITH high specificity = genuine flexibility.")
    print(f"  High switch WITHOUT specificity = cosmetic relabeling.")

    _results_store["E1_flexibility"] = {
        "switch_rate": switch_rate,
        "ctrl_rate": ctrl_rate,
        "z_switch": z_switch,
        "specificity": avg_specificity,
        "score": flexibility_score,
    }


# =====================================================================
# E2: INHIBITORY CONTROL (HEADLINE)
# =====================================================================

@kbench.task(name="e2_inhibitory_control")
def e2_inhibitory_control(llm):
    """E2: Can the model resist emotional anchoring?
    Three tiers: gold + probes + generated.
    Per scenario: 1 neutral + 3 control + 1 anchored + 1 inhibited = 6 calls.
    """
    print("\n[E2] INHIBITORY CONTROL (HEADLINE TEST)")
    print("  Testing resistance to emotional anchoring")
    print("  Three tiers: gold (6) + probes (8) + generated (9)")
    print("-" * 60)

    # Metrics
    sev_diffs_anchor = []   # anchor severity - neutral severity
    sev_diffs_ctrl = []     # control severity - neutral severity
    sev_diffs_inhibit = []  # inhibited severity - neutral severity
    verdict_flips_anchor = 0
    verdict_flips_ctrl = 0
    verdict_flips_ctrl_n = 0
    verdict_recoveries = 0
    verdict_flips_anchor_total = 0
    total = 0
    _lock = threading.Lock()

    def _run_inhibition(tag, s, framed_text):
        """Run one scenario: neutral vs framed (dramatic rewrite) vs inhibited.
        Uses 0-70 harm scale (same as social cognition T5) for max power.
        framed_text = dramatically rewritten version of the scenario.
        """
        nonlocal total, verdict_flips_anchor, verdict_flips_ctrl, verdict_flips_ctrl_n
        nonlocal verdict_recoveries, verdict_flips_anchor_total

        with ThreadPoolExecutor(max_workers=min(_pool.n, 6)) as pool:
            f_neutral = pool.submit(call_llm, llm, f"e2_{tag}_n",
                                    prompt_neutral_eval(s), InhibitionVerdict)
            f_ctrls = [pool.submit(call_llm, llm, f"e2_{tag}_c{ci}",
                                   prompt_neutral_eval(s), InhibitionVerdict)
                       for ci in range(N_CONTROL_REPS)]
            f_framed = pool.submit(call_llm, llm, f"e2_{tag}_a",
                                   prompt_framed_eval(framed_text), InhibitionVerdict)
            f_inhibit = pool.submit(call_llm, llm, f"e2_{tag}_i",
                                    prompt_inhibited_eval(framed_text), InhibitionVerdict)

            try:
                neutral = f_neutral.result()
                ctrls = [f.result() for f in f_ctrls]
                framed = f_framed.result()
                inhibit = f_inhibit.result()
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        v_neutral = normalize_verdict_ef(neutral.verdict)
        v_framed = normalize_verdict_ef(framed.verdict)
        v_inhibit = normalize_verdict_ef(inhibit.verdict)

        harm_n = clamp(neutral.total_harm, 0, 70)
        harm_f = clamp(framed.total_harm, 0, 70)
        harm_i = clamp(inhibit.total_harm, 0, 70)

        ctrl_harms = [clamp(c.total_harm, 0, 70) for c in ctrls]
        ctrl_vs = [normalize_verdict_ef(c.verdict) for c in ctrls]

        with _lock:
            total += 1
            sev_diffs_anchor.append(abs(harm_f - harm_n))
            sev_diffs_inhibit.append(abs(harm_i - harm_n))
            for ch in ctrl_harms:
                sev_diffs_ctrl.append(abs(ch - harm_n))

            if v_framed != v_neutral:
                verdict_flips_anchor += 1
            verdict_flips_anchor_total += 1

            ctrl_flips = sum(1 for cv in ctrl_vs if cv != v_neutral)
            verdict_flips_ctrl += ctrl_flips
            verdict_flips_ctrl_n += N_CONTROL_REPS

            if v_framed != v_neutral and v_inhibit == v_neutral:
                verdict_recoveries += 1

    # === Gold tier (dramatic rewrites from gold set) ===
    print(f"  --- Gold tier ({len(GOLD_SET)} scenarios, hand-written dramatic rewrites) ---")
    for gi, gs in enumerate(GOLD_SET):
        dramatic_text = gs["e2_anchor"]
        try:
            _run_inhibition(f"gold{gi}", gs, dramatic_text)
        except Exception as e:
            print(f"    WARN: gold {gi} failed: {e}")
        if (gi + 1) % 3 == 0:
            print(f"    [{gi+1}/{len(GOLD_SET)}] framing_flips={verdict_flips_anchor}/{total}")

    # === Probes (dramatic rewrites of conservation probes) ===
    print(f"  --- Probes ({len(PROBES_INHIBITION)} scenarios, dramatic rewrites) ---")
    for probe in PROBES_INHIBITION:
        # Use probe's dramatic text as the framing
        dramatic_text = probe.get("dramatic_text", "")
        try:
            _run_inhibition(f"probe_{probe['id']}", probe["scenario"], dramatic_text)
        except Exception as e:
            print(f"    WARN: {probe['id']} failed: {e}")
        if total % 4 == 0:
            print(f"    [{probe['id']}] framing_flips={verdict_flips_anchor}/{total}")

    # === Generated tier (LLM-generated dramatic rewrites) ===
    n_gen = 9
    print(f"  --- Generated tier ({n_gen} Dear Abby, LLM dramatic rewrites) ---")
    for si in range(n_gen):
        dramatic_text = _transforms.get((si, "dramatic"))
        if not dramatic_text:
            continue
        try:
            _run_inhibition(f"gen{si}", DEAR_ABBY[si], dramatic_text)
        except Exception as e:
            print(f"    WARN: gen {si} failed: {e}")
        if (si + 1) % 5 == 0:
            print(f"    [{si+1}/{n_gen}] framing_flips={verdict_flips_anchor}/{total}")

    # === Analysis ===
    anchor_flip_rate = verdict_flips_anchor / max(verdict_flips_anchor_total, 1)
    ctrl_flip_rate = verdict_flips_ctrl / max(verdict_flips_ctrl_n, 1)
    recovery_rate = verdict_recoveries / max(verdict_flips_anchor, 1) if verdict_flips_anchor > 0 else 1.0

    z_verdict = two_proportion_z(verdict_flips_anchor, verdict_flips_anchor_total,
                                 verdict_flips_ctrl, verdict_flips_ctrl_n)

    # Paired t-test: anchor severity shift vs control severity shift
    ctrl_mean_sev = mean(sev_diffs_ctrl) if sev_diffs_ctrl else 0.0
    # For paired t, pair each scenario's anchor shift with its mean control shift
    # Use per-scenario diffs
    n_paired = min(len(sev_diffs_anchor), len(sev_diffs_ctrl) // N_CONTROL_REPS)
    paired_diffs_anchor = []
    for i in range(n_paired):
        ctrl_slice = sev_diffs_ctrl[i * N_CONTROL_REPS:(i + 1) * N_CONTROL_REPS]
        ctrl_avg = mean(ctrl_slice) if ctrl_slice else 0.0
        paired_diffs_anchor.append(sev_diffs_anchor[i] - ctrl_avg)

    paired_diffs_inhibit = []
    for i in range(min(len(sev_diffs_inhibit), n_paired)):
        ctrl_slice = sev_diffs_ctrl[i * N_CONTROL_REPS:(i + 1) * N_CONTROL_REPS]
        ctrl_avg = mean(ctrl_slice) if ctrl_slice else 0.0
        paired_diffs_inhibit.append(sev_diffs_inhibit[i] - ctrl_avg)

    t_anchor_vs_control = paired_t(paired_diffs_anchor)
    t_inhibit_vs_control = paired_t(paired_diffs_inhibit)

    resistance_score = 1.0 - anchor_flip_rate
    inhibition_score = 0.5 * resistance_score + 0.3 * recovery_rate + 0.2 * (1.0 - min(mean(sev_diffs_anchor) / 35.0, 1.0))

    print(f"\n  RESULTS (E2: inhibitory control):")
    print(f"  Anchor verdict flip: {fmt_ci(verdict_flips_anchor, verdict_flips_anchor_total)}")
    print(f"  Control verdict flip: {fmt_ci(verdict_flips_ctrl, verdict_flips_ctrl_n)}")
    print(f"  Verdict flip vs control: {sig_label(z_verdict)}")
    print(f"  Recovery rate (anchor->inhibit): {verdict_recoveries}/{max(verdict_flips_anchor,1)} ({recovery_rate:.0%})")
    print(f"  Severity shift: anchor MAD={mean(sev_diffs_anchor):.2f} ctrl MAD={ctrl_mean_sev:.2f}")
    print(f"  Paired t (anchor vs ctrl): t={t_anchor_vs_control:.2f}")
    print(f"  Paired t (inhibit vs ctrl): t={t_inhibit_vs_control:.2f}")
    print(f"  Inhibition score: {inhibition_score:.3f}")

    _results_store["E2_inhibition"] = {
        "anchor_flip_rate": anchor_flip_rate,
        "ctrl_flip_rate": ctrl_flip_rate,
        "z_verdict": z_verdict,
        "recovery_rate": recovery_rate,
        "sev_shift_anchor": mean(sev_diffs_anchor) if sev_diffs_anchor else 0.0,
        "sev_shift_ctrl": ctrl_mean_sev,
        "t_anchor_vs_control": t_anchor_vs_control,
        "t_inhibit_vs_control": t_inhibit_vs_control,
        "n_scenarios": total,
        "score": inhibition_score,
    }


# =====================================================================
# E3: COUNTERFACTUAL REASONING
# =====================================================================

@kbench.task(name="e3_counterfactual")
def e3_counterfactual(llm):
    """E3: Does changing one causal fact change the moral verdict?
    Three tiers: gold + probes + generated.
    Per scenario: 1 original + 3 control + 1 counterfactual judge = 5 calls.
    """
    print("\n[E3] COUNTERFACTUAL REASONING")
    print("  Testing verdict sensitivity to single-cause pivots")
    print("  Three tiers: gold (6) + probes (4) + generated (6)")
    print("-" * 60)

    cf_flips = 0
    ctrl_flips = 0
    ctrl_n = 0
    total = 0
    _lock = threading.Lock()

    def _run_counterfactual(tag, s, cf_text):
        """Run original + controls + counterfactual judgment."""
        nonlocal cf_flips, ctrl_flips, ctrl_n, total

        with ThreadPoolExecutor(max_workers=min(_pool.n, 5)) as pool:
            f_orig = pool.submit(call_llm, llm, f"e3_{tag}_orig",
                                 prompt_judge_scenario(base_scenario_text(s)),
                                 CounterfactualVerdict)
            f_ctrls = [pool.submit(call_llm, llm, f"e3_{tag}_ctrl{ci}",
                                   prompt_judge_scenario(base_scenario_text(s)),
                                   CounterfactualVerdict)
                       for ci in range(N_CONTROL_REPS)]
            f_cf = pool.submit(call_llm, llm, f"e3_{tag}_cf",
                               prompt_judge_scenario(cf_text),
                               CounterfactualVerdict)

            try:
                orig = f_orig.result()
                ctrls = [f.result() for f in f_ctrls]
                cf = f_cf.result()
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        v_orig = normalize_verdict_ef(orig.verdict)
        v_cf = normalize_verdict_ef(cf.verdict)

        with _lock:
            total += 1
            if v_cf != v_orig:
                cf_flips += 1
            for c in ctrls:
                ctrl_n += 1
                if normalize_verdict_ef(c.verdict) != v_orig:
                    ctrl_flips += 1

    # === Gold tier (hand-written counterfactuals) ===
    print(f"  --- Gold tier ({len(GOLD_SET)} scenarios, hand-written counterfactuals) ---")
    for gi, gs in enumerate(GOLD_SET):
        # Build counterfactual text from the question
        cf_text = (
            f"MODIFIED SCENARIO:\n"
            f"Consider this variation of the original situation: {gs['e3_counterfactual']}\n\n"
            f"Original context: {gs['text']}\n\n"
            f"In this modified version, the key change described above has occurred. "
            f"All other facts remain the same."
        )
        try:
            _run_counterfactual(f"gold{gi}", gs, cf_text)
        except Exception as e:
            print(f"    WARN: gold {gi} failed: {e}")
        if (gi + 1) % 3 == 0:
            print(f"    [{gi+1}/{len(GOLD_SET)}] cf_flips={cf_flips}/{total}")

    # === Probes (unambiguous single-cause pivots) ===
    print(f"  --- Probes ({len(PROBES_COUNTERFACTUAL)} single-cause scenarios) ---")
    for probe in PROBES_COUNTERFACTUAL:
        cf_text = (
            f"MODIFIED SCENARIO:\n"
            f"{probe['counterfactual']}\n\n"
            f"Original: {probe['scenario']['text']}\n\n"
            f"In this modified version, the key change described above has occurred. "
            f"All other facts remain the same."
        )
        try:
            _run_counterfactual(f"probe_{probe['id']}", probe["scenario"], cf_text)
        except Exception as e:
            print(f"    WARN: {probe['id']} failed: {e}")
        print(f"    {probe['id']}: {probe['description'][:50]}...")

    # === Generated tier (transformer-generated counterfactuals) ===
    print(f"  --- Generated tier ({N_E3_GEN} scenarios, transformer counterfactuals) ---")
    for si in range(N_E3_GEN):
        cf_text = _transforms.get((si, "counterfactual"))
        if not cf_text:
            continue
        try:
            _run_counterfactual(f"gen{si}", DEAR_ABBY[si], cf_text)
        except Exception as e:
            print(f"    WARN: gen {si} failed: {e}")
        if (si + 1) % 3 == 0:
            print(f"    [{si+1}/{N_E3_GEN}] cf_flips={cf_flips}/{total}")

    # === Results ===
    cf_rate = cf_flips / max(total, 1)
    ctrl_rate = ctrl_flips / max(ctrl_n, 1)
    z_cf = two_proportion_z(cf_flips, total, ctrl_flips, ctrl_n)

    cf_score = cf_rate  # higher = more sensitive to counterfactual pivots

    print(f"\n  RESULTS (E3: counterfactual reasoning):")
    print(f"  Counterfactual flip rate: {fmt_ci(cf_flips, total)}")
    print(f"  Control flip rate: {fmt_ci(ctrl_flips, ctrl_n)}")
    print(f"  CF vs control: {sig_label(z_cf)}")
    print(f"  Counterfactual score: {cf_score:.3f}")
    print(f"  NOTE: Higher CF flip rate (above control) = genuine counterfactual")
    print(f"  sensitivity. If CF rate ~ control rate, the model is just noisy.")

    _results_store["E3_counterfactual"] = {
        "cf_flip_rate": cf_rate,
        "ctrl_flip_rate": ctrl_rate,
        "z_cf": z_cf,
        "n_scenarios": total,
        "score": cf_score,
    }


# =====================================================================
# E4: WORKING MEMORY
# =====================================================================

@kbench.task(name="e4_working_memory")
def e4_working_memory(llm):
    """E4: Tracking morally relevant parties at increasing scale.
    Tiered scenarios + probes.
    Per scenario: 1 identification + 2 control re-identifications = 3 calls.
    """
    print("\n[E4] WORKING MEMORY")
    print("  Tracking morally relevant parties at increasing scale")
    print("  Tiers: 2-party (6), 3-4 party (gold), 6-party (3 composite), 8-party (2 composite)")
    print("-" * 60)

    # Build tiered scenarios
    # Tier-2: Simple dyadic Dear Abby scenarios
    tier2_scenarios = []
    for s in DEAR_ABBY[:6]:
        tier2_scenarios.append({
            "text": base_scenario_text(s),
            "tier": 2,
            "expected_count": 2,
            "source": s["title"],
        })

    # Tier-4: Use GOLD_SET scenarios with 3-4 party counts (hand-labeled)
    tier4_scenarios = []
    for gs in GOLD_SET:
        if gs["e4_party_count"] >= 3:
            tier4_scenarios.append({
                "text": base_scenario_text(gs),
                "tier": 4,
                "expected_count": gs["e4_party_count"],
                "source": gs["title"],
            })

    # Tier-6: Composite from pre-generation
    tier6_scenarios = []
    for idx in range(3):
        text = _transforms.get((100 + idx, "composite_6"))
        if text:
            tier6_scenarios.append({
                "text": text,
                "tier": 6,
                "expected_count": 6,
                "source": f"composite_6_{idx}",
            })

    # Tier-8: Composite from pre-generation
    tier8_scenarios = []
    for idx in range(2):
        text = _transforms.get((200 + idx, "composite_8"))
        if text:
            tier8_scenarios.append({
                "text": text,
                "tier": 8,
                "expected_count": 8,
                "source": f"composite_8_{idx}",
            })

    all_scenarios = tier2_scenarios + tier4_scenarios + tier6_scenarios + tier8_scenarios
    print(f"  Tier-2: {len(tier2_scenarios)} scenarios")
    print(f"  Tier-4: {len(tier4_scenarios)} scenarios")
    print(f"  Tier-6: {len(tier6_scenarios)} scenarios (composite)")
    print(f"  Tier-8: {len(tier8_scenarios)} scenarios (composite)")
    print(f"  Total: {len(all_scenarios)} scenarios")

    tier_counts = {2: [], 4: [], 6: [], 8: []}
    tier_consistency = {2: [], 4: [], 6: [], 8: []}
    tier_accuracy = {2: [], 4: [], 6: [], 8: []}
    total = 0
    _lock = threading.Lock()

    def _run_wm(tag, scenario):
        nonlocal total
        tier = scenario["tier"]
        text = scenario["text"]
        expected = scenario["expected_count"]

        with ThreadPoolExecutor(max_workers=min(_pool.n, 3)) as pool:
            f_main = pool.submit(call_llm, llm, f"e4_{tag}_main",
                                 prompt_party_identification(text), PartyAnalysis)
            f_ctrl1 = pool.submit(call_llm, llm, f"e4_{tag}_ctrl1",
                                  prompt_party_identification(text), PartyAnalysis)
            f_ctrl2 = pool.submit(call_llm, llm, f"e4_{tag}_ctrl2",
                                  prompt_party_identification(text), PartyAnalysis)

            try:
                main = f_main.result()
                ctrl1 = f_ctrl1.result()
                ctrl2 = f_ctrl2.result()
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        p_main = clamp(main.parties_identified, 0, 20)
        p_ctrl1 = clamp(ctrl1.parties_identified, 0, 20)
        p_ctrl2 = clamp(ctrl2.parties_identified, 0, 20)

        with _lock:
            total += 1
            tier_counts[tier].append(p_main)

            # Consistency: do all 3 agree on count within 1?
            counts = [p_main, p_ctrl1, p_ctrl2]
            count_range = max(counts) - min(counts)
            consistent = 1.0 if count_range <= 1 else 0.0
            tier_consistency[tier].append(consistent)

            # Accuracy: how close is the average count to expected?
            avg_found = mean(counts)
            acc = 1.0 - min(abs(avg_found - expected) / max(expected, 1), 1.0)
            tier_accuracy[tier].append(acc)

    # === Run tiered scenarios ===
    for si, scenario in enumerate(all_scenarios):
        try:
            _run_wm(f"tier{scenario['tier']}_{si}", scenario)
        except Exception as e:
            print(f"    WARN: scenario {si} failed: {e}")
        if (si + 1) % 5 == 0:
            print(f"  [{si+1}/{len(all_scenarios)}] total={total}")

    # === Probes (hand-labeled party counts) ===
    print(f"  --- Probes ({len(PROBES_WORKING_MEMORY)} hand-labeled scenarios) ---")
    probe_accuracy = []
    for probe in PROBES_WORKING_MEMORY:
        text = base_scenario_text(probe["scenario"])
        try:
            result = call_llm(llm, f"e4_probe_{probe['id']}",
                              prompt_party_identification(text), PartyAnalysis)
            found = clamp(result.parties_identified, 0, 20)
            expected = probe["count"]
            accuracy = 1.0 - min(abs(found - expected) / max(expected, 1), 1.0)
            probe_accuracy.append(accuracy)
            mark = "" if abs(found - expected) <= 1 else " MISS"
            print(f"    {probe['id']} (tier-{probe['tier']}): found={found:.0f} "
                  f"expected={expected} accuracy={accuracy:.2f}{mark}")
        except Exception as e:
            print(f"    WARN: {probe['id']} failed: {e}")

    # === Results per tier ===
    tier_scores = {}
    for tier in [2, 4, 6, 8]:
        if tier_counts[tier]:
            avg_count = mean(tier_counts[tier])
            avg_con = mean(tier_consistency[tier]) if tier_consistency[tier] else 0.0
            avg_acc = mean(tier_accuracy[tier]) if tier_accuracy[tier] else 0.0
            # Score: blend accuracy (vs expected count) + consistency
            tier_scores[tier] = {
                "avg_count": avg_count,
                "consistency": avg_con,
                "accuracy": avg_acc,
                "n_scenarios": len(tier_counts[tier]),
                "score": 0.5 * avg_acc + 0.5 * avg_con,
            }
        else:
            tier_scores[tier] = {"score": 0.0, "n_scenarios": 0, "avg_count": 0, "consistency": 0, "accuracy": 0}

    # Degradation slope
    valid_tiers = sorted([t for t in [2, 4, 6, 8] if tier_scores[t]["n_scenarios"] > 0])
    scores_by_tier = [tier_scores[t]["score"] for t in valid_tiers]
    degradation = scores_by_tier[0] - scores_by_tier[-1] if len(scores_by_tier) >= 2 else 0.0

    working_memory_score = mean(scores_by_tier) if scores_by_tier else 0.0
    probe_avg = mean(probe_accuracy) if probe_accuracy else 0.0

    print(f"\n  RESULTS (E4: working memory):")
    print(f"  {'Tier':<10} {'Score':>8} {'Accuracy':>10} {'Consist':>10} {'AvgCount':>10} {'N':>4}")
    print(f"  {'-'*52}")
    for tier in [2, 4, 6, 8]:
        ts = tier_scores[tier]
        if ts["n_scenarios"] > 0:
            print(f"  Tier-{tier:<4} {ts['score']:>7.3f} {ts['accuracy']:>9.3f} "
                  f"{ts['consistency']:>9.3f} {ts['avg_count']:>9.1f} {ts['n_scenarios']:>4}")
    print(f"  Degradation (tier-2 vs highest tier): {degradation:+.3f}")
    print(f"  Probe accuracy: {probe_avg:.3f}")
    print(f"  Working memory score: {working_memory_score:.3f}")

    _results_store["E4_working_memory"] = {
        "tier_scores": {str(k): v for k, v in tier_scores.items()},
        "degradation": degradation,
        "probe_accuracy": probe_avg,
        "score": working_memory_score,
    }


# =====================================================================
# MULTI-MODEL EXECUTION
# =====================================================================

MODELS_FULL = [
    "google/gemini-2.0-flash",       # baseline, older gen (also transformer model)
    "google/gemini-2.5-pro",         # strongest Gemini, current gen
]

# E2-only models: add statistical power for the headline anchoring finding
MODELS_E2_ONLY = [
    "google/gemini-2.5-flash",       # current gen flash
    "google/gemini-3-flash-preview", # next gen
]

# Cross-family model -- E2 probes on non-Gemini model
CROSS_FAMILY_MODEL = "anthropic/claude-sonnet-4-6@default"
N_CROSS_FAMILY_PROBES = 8  # all inhibition probes

MODELS_TO_TEST = MODELS_FULL

print(f"\n[3/7] Phase 1: Pre-generating counterfactuals with {TRANSFORMER_MODEL}")
try:
    transformer_llm = kbench.llms[TRANSFORMER_MODEL]
    phase1_pre_generate.run(llm=transformer_llm)
except Exception as e:
    print(f"  FATAL: Pre-generation failed: {e}")
    print(f"  Cannot proceed without counterfactual text.")
    raise

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
            (e1_cognitive_flexibility, "E1_flexibility"),
            (e2_inhibitory_control, "E2_inhibition"),
            (e3_counterfactual, "E3_counterfactual"),
            (e4_working_memory, "E4_working_memory"),
        ]:
            try:
                test_fn.run(llm=llm)
                model_results[test_name] = _results_store.get(test_name, {"score": 0.0})
            except Exception as e:
                print(f"  ERROR in {test_name}: {e}")
                model_results[test_name] = {"error": str(e), "score": 0.0}

    except Exception as e:
        print(f"  ERROR loading model {model_name}: {e}")
        model_results = {f"E{i}": {"error": str(e), "score": 0.0} for i in range(1, 5)}

    all_results[model_name] = model_results

# =====================================================================
# E2-ONLY MODELS (additional statistical power for headline finding)
# =====================================================================

if MODELS_E2_ONLY:
    print(f"\n[5/7] Running E2-only on {len(MODELS_E2_ONLY)} additional models")
    for m in MODELS_E2_ONLY:
        print(f"  - {m} (E2 only)")

    for mi, model_name in enumerate(MODELS_E2_ONLY):
        print(f"\n{'#'*70}")
        print(f"# E2-ONLY {mi+1}/{len(MODELS_E2_ONLY)}: {model_name}")
        print(f"{'#'*70}")

        model_results = {}
        try:
            llm = kbench.llms[model_name]
            _results_store.clear()
            e2_inhibitory_control.run(llm=llm)
            model_results["E2_inhibition"] = _results_store.get("E2_inhibition", {"score": 0.0})
        except Exception as e:
            print(f"  ERROR: {e}")
            model_results["E2_inhibition"] = {"error": str(e), "score": 0.0}

        all_results[model_name] = model_results

# =====================================================================
# CROSS-FAMILY VALIDATION (E2 probes on non-Gemini model)
# =====================================================================

print(f"\n{'#'*70}")
print(f"# CROSS-FAMILY: {CROSS_FAMILY_MODEL}")
print(f"# E2 inhibition probes only ({N_CROSS_FAMILY_PROBES} probes)")
print(f"{'#'*70}")

cross_family_results = {}
try:
    cf_llm = kbench.llms[CROSS_FAMILY_MODEL]
    cf_probes = PROBES_INHIBITION[:N_CROSS_FAMILY_PROBES]
    cf_sev_diffs = []
    cf_flips = 0
    cf_total = 0

    for probe in cf_probes:
        dramatic_text = probe.get("dramatic_text", "")
        if not dramatic_text:
            continue
        with ThreadPoolExecutor(max_workers=min(_pool.n, 3)) as pool:
            f_neutral = pool.submit(call_llm, cf_llm, f"cf_p_{probe['id']}_n",
                                    prompt_neutral_eval(probe["scenario"]), InhibitionVerdict)
            f_framed = pool.submit(call_llm, cf_llm, f"cf_p_{probe['id']}_a",
                                   prompt_framed_eval(dramatic_text), InhibitionVerdict)
            f_inhibit = pool.submit(call_llm, cf_llm, f"cf_p_{probe['id']}_i",
                                    prompt_inhibited_eval(dramatic_text), InhibitionVerdict)
            try:
                neutral = f_neutral.result()
                framed = f_framed.result()
                inhibit = f_inhibit.result()

                vn = normalize_verdict_ef(neutral.verdict)
                vf = normalize_verdict_ef(framed.verdict)
                vi = normalize_verdict_ef(inhibit.verdict)

                hn = clamp(neutral.total_harm, 0, 70)
                hf = clamp(framed.total_harm, 0, 70)
                hi = clamp(inhibit.total_harm, 0, 70)

                cf_sev_diffs.append(abs(hf - hn))
                cf_total += 1
                if vf != vn:
                    cf_flips += 1

                mark = ""
                if vf != vn: mark += " FRAMED"
                if vf != vn and vi == vn: mark += " RECOVERED"
                print(f"  {probe['id']}: n={vn} f={vf} i={vi} "
                      f"harm={hn:.0f}/{hf:.0f}/{hi:.0f}{mark}")
            except Exception as e:
                print(f"  WARN: {probe['id']} failed: {e}")

    if cf_total > 0:
        cf_flip_rate = cf_flips / cf_total
        cf_avg_sev = mean(cf_sev_diffs) if cf_sev_diffs else 0.0
        cross_family_results = {
            "flip_rate": cf_flip_rate,
            "avg_sev_shift": cf_avg_sev,
            "n_probes": cf_total,
        }
        print(f"\n  CROSS-FAMILY E2 RESULTS ({CROSS_FAMILY_MODEL.split('/')[-1]}):")
        print(f"  Anchor flip rate: {cf_flips}/{cf_total} ({cf_flip_rate:.0%})")
        print(f"  Avg severity shift: {cf_avg_sev:.2f}")
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
print(f"CROSS-MODEL COMPARISON -- FOUR EXECUTIVE FUNCTION TESTS")
print(f"{'#'*70}")
print()

WEIGHTS = {
    "E1_flexibility": 0.20,
    "E2_inhibition": 0.35,
    "E3_counterfactual": 0.25,
    "E4_working_memory": 0.20,
}

header = (f"  {'Model':<30} {'E1:Flex':>8} {'E2:Inhib':>9} "
          f"{'E3:CF':>8} {'E4:WMem':>8} {'Compos':>8}")
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
          f"{scores['E1_flexibility']:>7.3f} "
          f"{scores['E2_inhibition']:>8.3f} "
          f"{scores['E3_counterfactual']:>7.3f} "
          f"{scores['E4_working_memory']:>7.3f} "
          f"{composite:>7.3f}")

print()
print(f"  Weights: E1={WEIGHTS['E1_flexibility']}, E2={WEIGHTS['E2_inhibition']}, "
      f"E3={WEIGHTS['E3_counterfactual']}, E4={WEIGHTS['E4_working_memory']}")
print(f"  (E2 weighted highest: headline test with strongest expected signal)")


# =====================================================================
# FISHER COMBINATION: SIGMA CALCULATION
# =====================================================================

print()
print("=" * 70)
print("SIGMA ANALYSIS (Fisher combination across models)")
print("=" * 70)
print()

# Collect E2 paired-t results from all models (Gemini full + E2-only)
e2_sigmas = []
# df for paired t = n_scenarios - 1
# Gold (6) + probes (8) + generated (9) = 23 scenarios
df_e2 = max(23 - 1, 2)

print("  Individual E2 results (paired t -> sigma, df=%d):" % df_e2)
for model_name in list(MODELS_FULL) + list(MODELS_E2_ONLY):
    r = all_results.get(model_name, {}).get("E2_inhibition", {})
    short = model_name.split("/")[-1][:25]
    t_a = r.get("t_anchor_vs_control", 0)
    t_i = r.get("t_inhibit_vs_control", 0)
    if t_a == 0 and t_i == 0:
        print(f"    {short:25s}  (no E2 data)")
        continue
    sig_a = _t_to_sigma(t_a, df_e2)
    sig_i = _t_to_sigma(t_i, df_e2)
    e2_sigmas.append(sig_a)
    sa = '***' if sig_a >= 3 else '**' if sig_a >= 2 else '*' if sig_a >= 1.5 else ''
    si_mark = '***' if sig_i >= 3 else '**' if sig_i >= 2 else '*' if sig_i >= 1.5 else ''
    print(f"    {short:25s}  anchor t={t_a:+.2f} -> {sig_a:.1f}s {sa:4s}  inhibit t={t_i:+.2f} -> {sig_i:.1f}s {si_mark}")

if len(e2_sigmas) >= 2:
    combined_e2 = _fisher_combine(e2_sigmas)

    print()
    print(f"  Fisher combined ({len(e2_sigmas)} independent anchor tests, {len(e2_sigmas)} models):")
    print(f"    E2 anchoring ALL:  {combined_e2:.1f}s {'*** DISCOVERY-LEVEL ***' if combined_e2 >= 5 else '** SIGNIFICANT **' if combined_e2 >= 3 else ''}")
    print()
    if combined_e2 >= 5:
        print(f"  >>> HEADLINE: Emotional anchoring effect at {combined_e2:.1f} sigma <<<")
        print(f"  >>> {len(e2_sigmas)} Gemini models + Claude cross-family validation <<<")
else:
    combined_e2 = 0
    print("  (insufficient E2 data for Fisher combination)")

print()

# =====================================================================
# HEADLINE FINDINGS
# =====================================================================

print("=" * 70)
print("HEADLINE FINDINGS")
print("=" * 70)
print()
e2_sigma_str = f"{combined_e2:.1f}" if combined_e2 > 0 else "N/A"
print(f"  1. EMOTIONAL ANCHORING SHIFTS JUDGMENT AT {e2_sigma_str} SIGMA (E2)")
print(f"     Across {len(e2_sigmas)} Gemini models, emotional anchoring shifted")
print("     severity ratings and flipped verdicts beyond stochastic control.")
print("     Fisher combination of paired t-tests across all models yields")
print(f"     {e2_sigma_str} sigma combined significance.")
print(f"     Reference: social cognition T5 framing yielded 7.6 sigma.")
if cross_family_results:
    cf_short = CROSS_FAMILY_MODEL.split("/")[-1]
    print(f"     Cross-family validation on {cf_short}")
    print(f"     anchor flip rate: {cross_family_results.get('flip_rate', 0):.0%}")
print()

# Determine recovery summary
recovery_rates = []
for model_name in list(MODELS_FULL) + list(MODELS_E2_ONLY):
    r = all_results.get(model_name, {}).get("E2_inhibition", {})
    rr = r.get("recovery_rate", None)
    if rr is not None:
        recovery_rates.append(rr)
avg_recovery = mean(recovery_rates) if recovery_rates else 0.0
recovery_word = "can partially" if avg_recovery > 0.3 else "cannot reliably" if avg_recovery < 0.1 else "show limited ability to"

print(f"  2. MODELS {recovery_word.upper()} RECOVER VIA INHIBITION INSTRUCTION")
print(f"     Average recovery rate across models: {avg_recovery:.0%}")
print("     When emotional anchor flips verdict, explicit inhibition instruction")
print(f"     restores the neutral verdict {avg_recovery:.0%} of the time.")
print()

# Framework switching summary
switch_rates = []
spec_scores = []
for model_name in MODELS_FULL:
    r = all_results.get(model_name, {}).get("E1_flexibility", {})
    sr = r.get("switch_rate", None)
    sp = r.get("specificity", None)
    if sr is not None:
        switch_rates.append(sr)
    if sp is not None:
        spec_scores.append(sp)
avg_switch = mean(switch_rates) if switch_rates else 0.0
avg_spec = mean(spec_scores) if spec_scores else 0.0
genuine_word = "genuine" if avg_spec > 0.6 and avg_switch > 0.3 else "partially genuine" if avg_spec > 0.4 else "cosmetic"

print(f"  3. FRAMEWORK SWITCHING IS {genuine_word.upper()} (E1)")
print(f"     Verdict switch rate: {avg_switch:.0%}, marker specificity: {avg_spec:.3f}")
print("     High specificity with high switch rate indicates frameworks produce")
print("     genuinely different reasoning, not just relabeled conclusions.")
print()

# CF and WM summary
cf_rates = []
wm_degradations = []
for model_name in MODELS_FULL:
    r3 = all_results.get(model_name, {}).get("E3_counterfactual", {})
    r4 = all_results.get(model_name, {}).get("E4_working_memory", {})
    cfr = r3.get("cf_flip_rate", None)
    deg = r4.get("degradation", None)
    if cfr is not None:
        cf_rates.append(cfr)
    if deg is not None:
        wm_degradations.append(deg)
avg_cf = mean(cf_rates) if cf_rates else 0.0
avg_deg = mean(wm_degradations) if wm_degradations else 0.0

print(f"  4. COUNTERFACTUAL SENSITIVITY: {avg_cf:.0%} flip rate (E3)")
print(f"     WORKING MEMORY DEGRADATION: {avg_deg:+.3f} (tier-2 vs highest tier) (E4)")
print("     E3 measures genuine sensitivity to single-cause pivots.")
print("     E4 measures how party identification degrades with complexity.")
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
print("    E1 (Cognitive Flexibility): Does the model produce genuinely")
print("      different reasoning under utilitarian/deontological/virtue")
print("      frameworks, or merely relabel the same conclusion?")
print("      Measured via verdict switch rate + framework marker specificity.")
print("    E2 (Inhibitory Control): HEADLINE TEST. Does emotional anchoring")
print("      shift severity ratings and flip verdicts beyond stochastic")
print("      baseline? Can explicit inhibition instruction restore neutrality?")
print("      Measured via paired t-test on severity + verdict flip z-test.")
print("    E3 (Counterfactual Reasoning): Does changing one causal fact")
print("      produce a different moral verdict? Separates generation from")
print("      judgment: transformer generates counterfactuals, test model judges.")
print("    E4 (Working Memory): Can the model reliably identify morally")
print("      relevant parties as scenario complexity increases from 2 to 8?")
print("      Measures consistency of identification across repeated queries.")
print()
print("  STATISTICAL CONTROLS:")
print(f"    E1-E4 include {N_CONTROL_REPS}-rep control arms: the model re-judges")
print("    identical text to estimate stochastic baseline flip rate.")
print("    Significance via two-proportion z-test (E1, E2 verdict, E3) or")
print("    paired t-test (E2 severity) AGAINST the control, not against zero.")
print("    All rates reported with Wilson 95% confidence intervals.")
print(f"    CAVEAT: {N_CONTROL_REPS} reps per scenario is a thin estimate of stochasticity.")
print()
print("  SEPARATION OF CONCERNS:")
print("    E3 counterfactual text is generated by a FIXED model")
print(f"    ({TRANSFORMER_MODEL}). Models under test ONLY judge the")
print("    pre-generated text. This eliminates the self-confirming loop.")
print()
print("  THREE-TIER DATA ARCHITECTURE:")
print("    Gold tier: hand-audited scenarios with human-written anchors,")
print("    counterfactuals, and party labels. Highest interpretive confidence.")
print("    Probe tier: synthetic scenarios engineered for maximum control")
print("    (e.g., provably irrelevant anchors, unambiguous pivots).")
print("    Generated tier: Dear Abby scenarios with LLM-generated transforms.")
print("    Upper bounds on effects (transforms may alter moral salience).")
print()
print("  KNOWN LIMITATIONS (this is a pilot, not a definitive study):")
print("    1. Small samples: 6-23 scenarios per test. Results are")
print("       directional evidence, not sweeping claims.")
print("    2. Full suite runs on Gemini-family only (budget constraint).")
print(f"       Cross-family validation on {CROSS_FAMILY_MODEL.split('/')[-1]}")
print("       covers E2 probes only.")
print("    3. Counterfactual text generation quality varies. Not all")
print("       generated counterfactuals cleanly isolate a single cause.")
print("    4. E4 composite scenario quality depends on the transformer.")
print("       Party counts in generated composites are approximate.")
print("    5. No temperature control; adaptive concurrency may introduce")
print("       minor noise from retry patterns.")
print(f"    6. Control arms ({N_CONTROL_REPS} reps) provide a stochasticity")
print("       floor but not a full variance model.")
print("    7. E2 anchors are uniformly intense. A more nuanced study would")
print("       vary anchor intensity to map a dose-response curve.")
print("=" * 70)

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
