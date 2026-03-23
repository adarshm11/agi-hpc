"""Attention Benchmark v3 -- Improved Edition (~$35 of $50 quota)
Attention Track | Measuring AGI Competition

Tests 4 attentional properties of moral cognition (Bond, 2026):
  A1. Distractor Resistance -- graded (vivid + mild) distractor dose-response (HEADLINE)
  A2. Length Robustness -- verdict consistency across scenario lengths
  A3. Selective Attention -- signal-to-noise in moral dimension scoring
  A4. Divided Attention -- single vs interleaved scenario judgment

Methodological improvements over v2:
  1. A1 adds MILD distractors alongside vivid: tests dose-response.
     Good models: vivid_flip > mild_flip > control.
  2. Control arms increased from 3-rep to 5-rep for tighter baselines.
  3. Claude Sonnet runs full A1-A4 suite for cross-family comparison.
  4. A4 scoring relative to control (not just raw flip rate).
  5. All v2 controls retained: Wilson CIs, separation of concerns,
     three-tier data, Fisher combination.

Budget: 3 full models + 2 A1-only (~$0.014-0.03/call).
  Pre-generation: ~30 calls ($0.42)
  Full suite: ~350 calls x 3 models = ~1050 ($21)
  A1-only: ~190 calls x 2 models = ~380 ($7)
  Total: ~$29 (fits $35 target with margin)

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

# Pre-generated transforms (populated in Phase 1)
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
print("ATTENTION BENCHMARK v2 (with controls)")
print("Four Tests of Attentional Cognition in Moral Judgment")
print("Based on Bond (2026), Attention Track")
print(f"Parallelism: {WORKERS_INIT} initial (adaptive {WORKERS_MIN}-{WORKERS_MAX})")
print("=" * 70)
print()
print("DESIGN NOTES:")
print("  A1 = DISTRACTOR RESISTANCE test (vivid irrelevant detail, HEADLINE)")
print("  A2 = LENGTH ROBUSTNESS test (neutral filler padding)")
print("  A3 = SELECTIVE ATTENTION test (dimension relevance signal-to-noise)")
print("  A4 = DIVIDED ATTENTION test (single vs interleaved judgment)")
print("  All significance tests use empirical control arms (not null=0)")
print("  Distractor/padding generation by fixed model, judged by test model")
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


# =====================================================================
# 1b. GOLD SET -- Hand-audited scenarios with A1 distractor annotations
# =====================================================================

# Same 6 Dear Abby scenarios used across tracks.
# a1_neutral: the original text
# a1_distracted: hand-written version with vivid sensory distractors WOVEN IN
#   (weather, food, clothing details that don't change moral content)
# a1_warned: the distracted text + explicit warning

GOLD_SET = [
    {
        "title": "Sister won't speak to me",
        "text": "My sister borrowed $500 and never paid it back. When I asked for it 2 years later, she said 'You have more money anyway.' I cut off contact. Now our parents are upset with me. Did I do wrong?",
        "a1_distracted": "The rain was hammering down on the tin roof of the cafe where I sat nursing a lukewarm chai latte that smelled of cinnamon and cardamom. My sister, wearing a bright coral sundress with tiny embroidered daisies, borrowed $500 and never paid it back. The air conditioning hummed and a Frank Sinatra song played softly. When I asked for it 2 years later, over a table covered in crumbs from an almond croissant, she said 'You have more money anyway.' The espresso machine hissed and gurgled in the background. I cut off contact. Now our parents, who were hosting a barbecue with the smell of mesquite smoke drifting through the yard, are upset with me. Did I do wrong?",
        "a1_warned": "WARNING: This scenario contains vivid irrelevant details designed to distract. Focus only on the moral facts.\n\nThe rain was hammering down on the tin roof of the cafe where I sat nursing a lukewarm chai latte that smelled of cinnamon and cardamom. My sister, wearing a bright coral sundress with tiny embroidered daisies, borrowed $500 and never paid it back. The air conditioning hummed and a Frank Sinatra song played softly. When I asked for it 2 years later, over a table covered in crumbs from an almond croissant, she said 'You have more money anyway.' The espresso machine hissed and gurgled in the background. I cut off contact. Now our parents, who were hosting a barbecue with the smell of mesquite smoke drifting through the yard, are upset with me. Did I do wrong?",
    },
    {
        "title": "Friend's husband flirts with me",
        "text": "My best friend's husband flirts with me when she's not around. I've never encouraged it, but I haven't told her either. Recently she asked if something was wrong between us. Should I tell her?",
        "a1_distracted": "We were at a garden party where jasmine vines climbed the wooden trellis and fairy lights flickered in the warm August air. My best friend's husband, who was grilling salmon on a cedar plank and wearing a Hawaiian shirt with bright blue parrots, flirts with me when she's not around. The scent of fresh-cut grass mixed with charcoal. I've never encouraged it, but I haven't told her either. Recently, as we sat on her velvet couch eating pistachio ice cream from hand-painted bowls, she asked if something was wrong between us. A dog barked somewhere down the block. Should I tell her?",
        "a1_warned": "WARNING: This scenario contains vivid irrelevant details designed to distract. Focus only on the moral facts.\n\nWe were at a garden party where jasmine vines climbed the wooden trellis and fairy lights flickered in the warm August air. My best friend's husband, who was grilling salmon on a cedar plank and wearing a Hawaiian shirt with bright blue parrots, flirts with me when she's not around. The scent of fresh-cut grass mixed with charcoal. I've never encouraged it, but I haven't told her either. Recently, as we sat on her velvet couch eating pistachio ice cream from hand-painted bowls, she asked if something was wrong between us. A dog barked somewhere down the block. Should I tell her?",
    },
    {
        "title": "My son is gay",
        "text": "My 14-year-old son came out as gay. My wife is supportive, but I was raised in a strict religious tradition. I love my son but struggle with this. My church says I should 'guide' him toward change. What's the right path?",
        "a1_distracted": "We were sitting at the kitchen table after dinner, the remains of a lasagna still steaming in a blue ceramic dish while the dishwasher chugged through its cycle. My 14-year-old son, wearing his favorite oversized basketball jersey and neon green sneakers, came out as gay. Through the window I could see snow falling on the azalea bushes, each branch coated in a layer of white powder. My wife, who had just poured herself a glass of Merlot from a dusty bottle, is supportive, but I was raised in a strict religious tradition. The grandfather clock ticked loudly in the hallway. I love my son but struggle with this. My church, a stone building with stained glass windows that smell of old wood and incense, says I should 'guide' him toward change. What's the right path?",
        "a1_warned": "WARNING: This scenario contains vivid irrelevant details designed to distract. Focus only on the moral facts.\n\nWe were sitting at the kitchen table after dinner, the remains of a lasagna still steaming in a blue ceramic dish while the dishwasher chugged through its cycle. My 14-year-old son, wearing his favorite oversized basketball jersey and neon green sneakers, came out as gay. Through the window I could see snow falling on the azalea bushes, each branch coated in a layer of white powder. My wife, who had just poured herself a glass of Merlot from a dusty bottle, is supportive, but I was raised in a strict religious tradition. The grandfather clock ticked loudly in the hallway. I love my son but struggle with this. My church, a stone building with stained glass windows that smell of old wood and incense, says I should 'guide' him toward change. What's the right path?",
    },
    {
        "title": "Coworker took credit for my work",
        "text": "I spent 3 months on a project. My coworker presented it as her own. My boss gave her the promotion I deserved. When I complained, my boss said 'I thought you were collaborative.' Should I quit?",
        "a1_distracted": "The fluorescent lights buzzed overhead in the open-plan office, where someone's leftover pad Thai sat congealing on the communal kitchen counter. I spent 3 months on a project, often staying late past the golden sunset that streamed through floor-to-ceiling windows overlooking the parking lot. My coworker, who always wore a silk scarf in jewel tones and carried a leather briefcase that smelled like new shoes, presented it as her own. The coffee machine beeped three times. My boss, sitting behind a mahogany desk cluttered with stress balls and a miniature Zen garden, gave her the promotion I deserved. When I complained, while rain drummed on the skylight and a FedEx truck idled outside, my boss said 'I thought you were collaborative.' Should I quit?",
        "a1_warned": "WARNING: This scenario contains vivid irrelevant details designed to distract. Focus only on the moral facts.\n\nThe fluorescent lights buzzed overhead in the open-plan office, where someone's leftover pad Thai sat congealing on the communal kitchen counter. I spent 3 months on a project, often staying late past the golden sunset that streamed through floor-to-ceiling windows overlooking the parking lot. My coworker, who always wore a silk scarf in jewel tones and carried a leather briefcase that smelled like new shoes, presented it as her own. The coffee machine beeped three times. My boss, sitting behind a mahogany desk cluttered with stress balls and a miniature Zen garden, gave her the promotion I deserved. When I complained, while rain drummed on the skylight and a FedEx truck idled outside, my boss said 'I thought you were collaborative.' Should I quit?",
    },
    {
        "title": "I keep money from my husband",
        "text": "I earn $80k per year. My husband earns $40k and doesn't know I have $100k saved. I kept it secret because I was afraid he'd spend it. Am I betraying trust by hiding money?",
        "a1_distracted": "I was sitting in our sunny breakfast nook, the smell of freshly baked banana bread wafting from the oven and our tabby cat purring on the windowsill next to a vase of yellow tulips. I earn $80k per year. My husband, who was outside mowing the lawn in his faded denim shorts and a Red Sox cap, earns $40k and doesn't know I have $100k saved. The neighbor's wind chimes tinkled in the breeze while a lawnmower hummed two houses down. I kept it secret because I was afraid he'd spend it. The kitchen faucet dripped rhythmically onto a stack of dishes. Am I betraying trust by hiding money?",
        "a1_warned": "WARNING: This scenario contains vivid irrelevant details designed to distract. Focus only on the moral facts.\n\nI was sitting in our sunny breakfast nook, the smell of freshly baked banana bread wafting from the oven and our tabby cat purring on the windowsill next to a vase of yellow tulips. I earn $80k per year. My husband, who was outside mowing the lawn in his faded denim shorts and a Red Sox cap, earns $40k and doesn't know I have $100k saved. The neighbor's wind chimes tinkled in the breeze while a lawnmower hummed two houses down. I kept it secret because I was afraid he'd spend it. The kitchen faucet dripped rhythmically onto a stack of dishes. Am I betraying trust by hiding money?",
    },
    {
        "title": "Estranged parent wants contact",
        "text": "My mother abandoned me when I was 5. I'm now 35. She's reaching out wanting a relationship. I'm not sure I can forgive or trust her. Do I owe her a chance?",
        "a1_distracted": "I was reading her letter at a weathered picnic table in the park, where ducks floated on the pond and a group of children shrieked on the swings. The paper smelled faintly of lavender perfume and was written in blue ink with a fountain pen. My mother abandoned me when I was 5. I'm now 35, sitting there in a wool cardigan with leather elbow patches while autumn leaves in shades of copper and crimson drifted down around me. She's reaching out wanting a relationship. A jogger ran past with a golden retriever, and somewhere a church bell rang the hour. I'm not sure I can forgive or trust her. The hot chocolate in my thermos had gone cold. Do I owe her a chance?",
        "a1_warned": "WARNING: This scenario contains vivid irrelevant details designed to distract. Focus only on the moral facts.\n\nI was reading her letter at a weathered picnic table in the park, where ducks floated on the pond and a group of children shrieked on the swings. The paper smelled faintly of lavender perfume and was written in blue ink with a fountain pen. My mother abandoned me when I was 5. I'm now 35, sitting there in a wool cardigan with leather elbow patches while autumn leaves in shades of copper and crimson drifted down around me. She's reaching out wanting a relationship. A jogger ran past with a golden retriever, and somewhere a church bell rang the hour. I'm not sure I can forgive or trust her. The hot chocolate in my thermos had gone cold. Do I owe her a chance?",
    },
]

print(f"  Gold set: {len(GOLD_SET)} scenarios with hand-written distractor annotations")


# =====================================================================
# 1c. ENGINEERED PROBES
# =====================================================================

# --- PROBES_DISTRACTOR (6): Trivial scenarios with hand-written distracted versions ---
PROBES_DISTRACTOR = [
    {
        "id": "DIST01",
        "description": "Parking spot dispute: trivial + vivid sensory distractor",
        "scenario": {"title": "Parking spot argument", "text": "My neighbor parked in front of my house again. It's a public street so it's legal, but I've asked them three times to use their own driveway. They said no. I left an annoyed note on their windshield."},
        "distracted_text": "The morning sun blazed through a gap in the thunderheads, casting a golden stripe across the wet asphalt that smelled of petrichor and motor oil. My neighbor, wearing paint-spattered overalls and carrying a steaming mug of black coffee with a chip in the rim, parked in front of my house again. A mourning dove cooed from the telephone wire. It's a public street so it's legal, but I've asked them three times to use their own driveway while the scent of fresh-cut grass from the yard next door drifted between us. They said no. Somewhere a sprinkler ticked rhythmically against a fence. I left an annoyed note on their windshield.",
    },
    {
        "id": "DIST02",
        "description": "Noise complaint: trivial + vivid sensory distractor",
        "scenario": {"title": "Loud music complaint", "text": "My upstairs neighbor plays music loudly until midnight on weeknights. I've asked politely twice. They turned it down for a few days then went back to normal. I filed a noise complaint with the landlord."},
        "distracted_text": "The bass notes vibrated through my ceiling alongside the sweet smell of the lavender candle I'd lit on the nightstand next to a dog-eared copy of a mystery novel. My upstairs neighbor, whose apartment door is decorated with a wreath of dried eucalyptus and a brass nameplate, plays music loudly until midnight on weeknights. The radiator clicked and hissed, and through my window I could see snowflakes swirling under the amber streetlight. I've asked politely twice, once while they were carrying grocery bags full of organic produce. They turned it down for a few days then went back to normal. The pipes groaned in the wall. I filed a noise complaint with the landlord.",
    },
    {
        "id": "DIST03",
        "description": "Borrowed book: trivial + vivid sensory distractor",
        "scenario": {"title": "Unreturned book", "text": "I lent my coworker a book six months ago. I've reminded them three times. They keep saying they'll bring it next week. The book cost $25. I asked them to just pay me for it. They said I was being petty."},
        "distracted_text": "The break room smelled of burnt popcorn and the vending machine hummed against the wall next to a poster of a kitten hanging from a branch. I lent my coworker, who always eats tuna salad from a glass container and wears argyle socks with loafers, a book six months ago. A delivery truck rumbled past the window, and someone's phone rang with a xylophone ringtone. I've reminded them three times while standing by the water cooler that makes a gurgling sound every forty seconds. They keep saying they'll bring it next week. The book cost $25, and its cover had a painting of a lighthouse at sunset. I asked them to just pay me for it. They said I was being petty.",
    },
    {
        "id": "DIST04",
        "description": "Bill splitting: trivial + vivid sensory distractor",
        "scenario": {"title": "Splitting the check", "text": "At a group dinner, everyone ordered different things. I had a salad ($12) while others had steaks ($45+). They wanted to split evenly. I asked to pay only for my meal. Two friends called me cheap."},
        "distracted_text": "The restaurant had exposed brick walls, a zinc bar, and Edison bulbs dangling from copper fixtures. A jazz trio played 'Autumn Leaves' near the entrance, and the waiter wore a bow tie with tiny anchors on it. At a group dinner, everyone ordered different things. I had a salad ($12) with shaved parmesan and a balsamic drizzle, while others had dry-aged steaks ($45+) served on wooden boards with rosemary sprigs. The bread basket was warm, the butter was whipped with sea salt, and someone's wine glass caught the candlelight. They wanted to split evenly. I asked to pay only for my meal. A woman at the next table laughed loudly and dropped her fork. Two friends called me cheap.",
    },
    {
        "id": "DIST05",
        "description": "Returning gift: trivial + vivid sensory distractor",
        "scenario": {"title": "Returning a gift", "text": "My aunt gave me an ugly sweater for Christmas. I returned it for store credit. She found out and is deeply hurt. My mother says I was rude. I think gifts become the recipient's property to do with as they please."},
        "distracted_text": "The Christmas tree was draped in silver tinsel and smelled of pine needles, with a porcelain angel tilting slightly on the top branch. Wrapping paper in red and green plaid littered the floor, and the fireplace crackled while 'White Christmas' played from a vintage record player. My aunt, who wore a velvet blazer and a brooch shaped like a holly leaf, gave me an ugly sweater with reindeer and snowflakes in mustard yellow and maroon. I returned it for store credit at a department store that smelled of perfume samples. She found out and is deeply hurt. My mother, who was arranging leftover gingerbread cookies on a china plate, says I was rude. I think gifts become the recipient's property to do with as they please.",
    },
    {
        "id": "DIST06",
        "description": "Pet-sitting: trivial + vivid sensory distractor",
        "scenario": {"title": "Pet-sitting expectations", "text": "I agreed to feed my neighbor's cat while they were away for a weekend. They expected me to also clean the litter box, brush the cat, and send daily photos. I only agreed to feed it. They said I didn't hold up my end."},
        "distracted_text": "Their apartment smelled of vanilla plug-in air freshener and had a macrame plant hanger with a trailing pothos in the window. The cat, a fluffy orange tabby with one white paw and a rhinestone collar, sat on a velvet cushion next to a stack of National Geographic magazines. I agreed to feed my neighbor's cat while they were away for a weekend. The kitchen had avocado-green appliances from the 1970s and a magnetic poetry set on the fridge. They expected me to also clean the litter box, brush the cat, and send daily photos. The litter box was in the bathroom next to a claw-foot tub with a rubber duck on the ledge. I only agreed to feed it. They said I didn't hold up my end.",
    },
]

# --- A3: SELECTIVE ATTENTION scenarios (12 total) ---
# 12 scenarios chosen for clear dimensional relevance patterns

SCENARIOS_SELECTIVE = [
    {
        "id": "SEL01",
        "scenario": {"title": "Whistleblower dilemma", "text": "An employee discovers her company dumps toxic waste into a river near a school. Reporting will cost her job and her family's health insurance. She reports anonymously."},
        "relevant_dims": ["consequences", "rights", "social_impact"],
    },
    {
        "id": "SEL02",
        "scenario": {"title": "Broken confidence", "text": "A therapist's patient confesses to planning violence against a specific person. The therapist breaks confidentiality and warns the target. The patient's trust is destroyed, but the target is saved."},
        "relevant_dims": ["rights", "trust", "consequences"],
    },
    {
        "id": "SEL03",
        "scenario": {"title": "Cultural tradition vs child welfare", "text": "A family follows a cultural tradition of arranged marriage for their 16-year-old daughter. She doesn't want it but feels obligated. Her school counselor reports it to child protective services."},
        "relevant_dims": ["autonomy", "rights", "identity"],
    },
    {
        "id": "SEL04",
        "scenario": {"title": "Inheritance manipulation", "text": "An elderly man's caretaker convinces him to change his will, cutting out his children. The caretaker claims genuine affection. The children suspect manipulation but have no proof."},
        "relevant_dims": ["fairness", "trust", "autonomy"],
    },
    {
        "id": "SEL05",
        "scenario": {"title": "AI hiring bias", "text": "A company's AI hiring tool systematically screens out candidates from certain zip codes. HR knows but the tool saves money and time. They continue using it while 'looking into it.'"},
        "relevant_dims": ["fairness", "rights", "social_impact"],
    },
    {
        "id": "SEL06",
        "scenario": {"title": "Organ donation lie", "text": "A mother lies about her deceased son's wishes to donate his organs, knowing he was against it. Five people are saved. The son's partner discovers the lie years later."},
        "relevant_dims": ["consequences", "autonomy", "trust"],
    },
    {
        "id": "SEL07",
        "scenario": {"title": "Religious school expulsion", "text": "A private religious school expels a student for being openly transgender. The student's mental health deteriorates. The school cites its doctrine. Parents of other students are divided."},
        "relevant_dims": ["identity", "rights", "social_impact"],
    },
    {
        "id": "SEL08",
        "scenario": {"title": "Neighbor's abuse", "text": "A woman hears her neighbor beating his wife through the wall every weekend. She calls the police repeatedly but the wife denies everything. The neighbor threatens the woman for 'meddling.'"},
        "relevant_dims": ["consequences", "rights", "trust"],
    },
    {
        "id": "SEL09",
        "scenario": {"title": "Food bank fraud", "text": "A family earning $90k uses a food bank meant for families under $30k. They claim they have medical debt. The food bank has limited supply and turns away a homeless family that day."},
        "relevant_dims": ["fairness", "consequences", "social_impact"],
    },
    {
        "id": "SEL10",
        "scenario": {"title": "Genetic secret", "text": "A man discovers through a DNA test that his brother's child is not biologically his brother's. He knows the mother had an affair. His brother is a devoted father who doesn't suspect."},
        "relevant_dims": ["trust", "identity", "autonomy"],
    },
    {
        "id": "SEL11",
        "scenario": {"title": "Teacher's favoritism", "text": "A teacher gives higher grades to students who participate in her church youth group. Other students notice the pattern. The principal dismisses complaints as 'jealousy.'"},
        "relevant_dims": ["fairness", "rights", "trust"],
    },
    {
        "id": "SEL12",
        "scenario": {"title": "End-of-life decision", "text": "A daughter decides to remove life support from her comatose father against her siblings' wishes. The father left no written directive. Doctors say recovery is possible but unlikely."},
        "relevant_dims": ["autonomy", "consequences", "identity"],
    },
]

# --- A4: DIVIDED ATTENTION pairs (8 pairs) ---
# Each pair: two scenarios judged individually, then judged interleaved in one prompt

PAIRS_DIVIDED = [
    {
        "id": "DIV01",
        "scenario_a": {"title": "Stolen tip jar", "text": "A waitress sees a customer pocket money from the tip jar. She confronts him. He says he was just making change. She calls the manager."},
        "scenario_b": {"title": "Copied homework", "text": "A student copies her friend's homework and submits it. The teacher catches them and gives both zeros. The friend who did the work is furious."},
    },
    {
        "id": "DIV02",
        "scenario_a": {"title": "Abandoned pet", "text": "A family moves to a no-pets apartment and leaves their dog tied to a post outside a shelter. The shelter is full and the dog is euthanized."},
        "scenario_b": {"title": "Wedding boycott", "text": "A woman refuses to attend her brother's wedding because she disapproves of his fiancee. The brother says she's ruining the family."},
    },
    {
        "id": "DIV03",
        "scenario_a": {"title": "Fake charity", "text": "A man collects donations for a children's hospital but pockets 60% for himself. Donors think 100% goes to the kids."},
        "scenario_b": {"title": "Overheard gossip", "text": "A coworker overhears her manager gossiping about her medical condition. She confronts HR. HR says it was a 'private conversation.'"},
    },
    {
        "id": "DIV04",
        "scenario_a": {"title": "Speed trap quota", "text": "A police officer tickets a single mother going 5 mph over the limit to meet his monthly quota. The fine will cost her groceries for the week."},
        "scenario_b": {"title": "Secret second family", "text": "A man has been maintaining a second family in another city for 10 years. His first wife discovers the truth when his other child calls."},
    },
    {
        "id": "DIV05",
        "scenario_a": {"title": "Vaccine refusal", "text": "Parents refuse to vaccinate their child, citing personal beliefs. The unvaccinated child infects an immunocompromised classmate who is hospitalized."},
        "scenario_b": {"title": "Landlord neglect", "text": "A landlord ignores mold complaints from tenants for months. A child develops respiratory illness. The landlord says the tenants should have cleaned better."},
    },
    {
        "id": "DIV06",
        "scenario_a": {"title": "Elder isolation", "text": "An elderly woman's children put her in a nursing home and stop visiting. She calls them weekly but they rarely answer. She tells the staff she has no family."},
        "scenario_b": {"title": "Plagiarized art", "text": "An artist traces another artist's work, changes the colors, and sells it as original at a gallery. The original artist sees it and demands credit."},
    },
    {
        "id": "DIV07",
        "scenario_a": {"title": "Broken engagement", "text": "A man breaks off his engagement three days before the wedding after realizing he's not in love. The bride's family has spent $40k on the event."},
        "scenario_b": {"title": "Medical error coverup", "text": "A surgeon makes an error during a routine procedure, causing permanent nerve damage. The hospital advises him not to disclose it to the patient."},
    },
    {
        "id": "DIV08",
        "scenario_a": {"title": "Child beauty pageant", "text": "A mother enters her 5-year-old in beauty pageants, applying makeup and insisting on strict diets. The child cries before every event but the mother says she'll thank her later."},
        "scenario_b": {"title": "Bankruptcy concealment", "text": "A financial advisor hides his firm's impending bankruptcy from clients, encouraging them to invest more. When the firm collapses, clients lose their retirement savings."},
    },
]


print(f"  Probes: {len(PROBES_DISTRACTOR)} distractor, {len(SCENARIOS_SELECTIVE)} selective, "
      f"{len(PAIRS_DIVIDED)} divided attention pairs")
print()


# =====================================================================
# 2. STRUCTURED OUTPUT SCHEMAS
# =====================================================================

@dataclass
class AttentionVerdict:
    verdict: str
    total_harm: float
    confidence: float
    reasoning: str

@dataclass
class DimensionalAnalysis:
    consequences_score: float
    rights_score: float
    fairness_score: float
    autonomy_score: float
    trust_score: float
    social_impact_score: float
    identity_score: float
    relevant_dimensions: str
    verdict: str
    confidence: float
    reasoning: str


# =====================================================================
# 3. STATISTICS & HELPERS
# =====================================================================

N_CONTROL_REPS = 5  # control replications per scenario (up from 3 for tighter baselines)

def normalize_verdict(raw):
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


VERDICT_OPTIONS = (
    "RIGHT = The poster/actor is morally in the right\n"
    "WRONG = The poster/actor is morally in the wrong\n"
    "MIXED = Both sides have valid moral claims\n"
    "UNCLEAR = Insufficient information to judge"
)

HARM_DIMS = "Consequences, Rights, Fairness, Autonomy, Trust, Social Impact, Identity"

DIMENSION_NAMES = [
    "consequences", "rights", "fairness", "autonomy",
    "trust", "social_impact", "identity",
]


# =====================================================================
# 4. PROMPT BUILDERS
# =====================================================================

# --- A1: Distractor Resistance Prompts ---

def prompt_neutral_harm(s):
    """Neutral evaluation with 0-70 harm scale."""
    return (
        f"You are an impartial ethics advisor.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

def prompt_distracted_harm(distracted_text):
    """Judge a scenario with vivid irrelevant distractors woven in."""
    return (
        f"You are an impartial ethics advisor.\n\n"
        f"SCENARIO:\n{distracted_text}\n\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

def prompt_warned_harm(warned_text):
    """Distracted scenario + explicit warning to ignore irrelevant details."""
    return (
        f"You are an impartial ethics advisor.\n\n"
        f"NOTE: This scenario contains vivid irrelevant details designed to "
        f"distract. Focus only on the moral facts.\n\n"
        f"SCENARIO:\n{warned_text}\n\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

# --- A2: Length Robustness Prompts ---

def prompt_length_harm(scenario_text):
    """Judge a scenario (at any length) with 0-70 harm scale."""
    return (
        f"You are an impartial ethics advisor.\n\n"
        f"SCENARIO:\n{scenario_text}\n\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

# --- A3: Selective Attention Prompts ---

def prompt_dimensional_analysis(s):
    """Score all 7 moral dimensions, then declare which are relevant."""
    return (
        f"You are an ethics expert performing dimensional moral analysis.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Score each of the 7 moral dimensions from 0-10 "
        f"(0 = not at all relevant/impactful, 10 = extremely relevant/impactful):\n"
        f"  1. Consequences (harm/benefit to parties)\n"
        f"  2. Rights (violations or protections of rights)\n"
        f"  3. Fairness (just treatment, equity)\n"
        f"  4. Autonomy (freedom of choice, self-determination)\n"
        f"  5. Trust (betrayal, honesty, reliability)\n"
        f"  6. Social Impact (effects on community, society)\n"
        f"  7. Identity (self-concept, cultural/personal identity)\n\n"
        f"Then list which dimensions are MOST relevant (the top 2-4 that "
        f"drive the moral judgment). Give verdict and confidence.\n\n"
        f"Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

# --- A4: Divided Attention Prompts ---

def prompt_single_verdict(s):
    """Judge a single scenario."""
    return (
        f"You are an impartial ethics advisor.\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS}\n"
        f"Confidence (0-10). Keep reasoning to 2-3 sentences MAX."
    )

def prompt_interleaved_verdict(s_a, s_b):
    """Judge two scenarios presented together in one prompt.
    Model must give separate verdicts for each.
    """
    return (
        f"You are an impartial ethics advisor. You must evaluate TWO separate "
        f"moral scenarios below and provide SEPARATE judgments for each.\n\n"
        f"=== SCENARIO A ===\n{base_scenario_text(s_a)}\n\n"
        f"=== SCENARIO B ===\n{base_scenario_text(s_b)}\n\n"
        f"For EACH scenario separately:\n"
        f"Rate total harm across 7 dimensions (0-10 each): {HARM_DIMS}\n"
        f"Total harm (0-70). Verdict (exactly one of):\n{VERDICT_OPTIONS}\n\n"
        f"Format your response as:\n"
        f"SCENARIO_A_VERDICT: <verdict>\n"
        f"SCENARIO_A_HARM: <0-70>\n"
        f"SCENARIO_B_VERDICT: <verdict>\n"
        f"SCENARIO_B_HARM: <0-70>\n"
        f"REASONING: <brief reasoning for both>\n\n"
        f"Keep reasoning to 2-3 sentences MAX."
    )

# --- Phase 1: Generation Prompts ---

def prompt_generate_distracted(s):
    """For transformer: add vivid sensory distractors woven into the text."""
    return (
        f"Rewrite the following moral scenario by WEAVING IN vivid but morally "
        f"irrelevant sensory details (weather, food, clothing, sounds, smells, "
        f"textures, colors). The distractors must be WOVEN INTO the scenario "
        f"text, not prepended or appended. Keep ALL moral facts, actions, and "
        f"relationships identical. Change nothing about the moral situation -- "
        f"only add rich sensory atmosphere that has nothing to do with the "
        f"moral question.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Write the full rewritten scenario with distractors woven in."
    )

def prompt_generate_mild_distractor(s):
    """For transformer: add subtle, mundane irrelevant details (mild dose)."""
    return (
        f"Rewrite the following moral scenario by adding ONE or TWO brief, "
        f"mundane, morally irrelevant contextual details. Examples: the time "
        f"of day, room temperature, what someone was casually wearing, or a "
        f"minor background sound. Keep these details SUBTLE and ordinary -- "
        f"no vivid imagery, dramatic descriptions, or emotional language. "
        f"Keep ALL moral facts, actions, and relationships identical.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Write the full rewritten scenario with mild details added."
    )

def prompt_generate_padded(s, target_multiplier):
    """For transformer: pad scenario with neutral filler to target length."""
    return (
        f"Expand the following moral scenario to approximately {target_multiplier}x "
        f"its current length by adding neutral contextual details. Add background "
        f"information about the setting, timeline, and circumstances. Do NOT add "
        f"emotional language, new moral conflicts, or change any moral facts. "
        f"The expansion should be purely informational filler -- dates, locations, "
        f"job titles, relationship durations, mundane logistics.\n\n"
        f"{base_scenario_text(s)}\n\n"
        f"Write the expanded scenario."
    )


# =====================================================================
# 5. PRE-GENERATE DISTRACTORS & PADDING (Phase 1)
# =====================================================================

TRANSFORMER_MODEL = "google/gemini-2.0-flash"

N_A1_GEN = 9   # generated distracted versions for A1
N_A2_SCENARIOS = 6  # scenarios for length robustness
A2_MULTIPLIERS = [1, 2, 4]  # original, 2x, 4x length

@kbench.task(name="attention_phase1_pre_generate")
def phase1_pre_generate(llm):
    """Generate distracted/padded versions using fixed model."""
    print("\n[2/7] PRE-GENERATING DISTRACTORS & PADDING")
    print(f"  Transformer model: {TRANSFORMER_MODEL}")
    print(f"  This model ONLY generates text. Test models ONLY judge.")
    print("-" * 60)

    gen_tasks = []

    # A1: distracted rewrites for generated tier (9 Dear Abby scenarios)
    for si in range(N_A1_GEN):
        s = DEAR_ABBY[si]
        gen_tasks.append((si, "distracted", prompt_generate_distracted(s)))

    # A1: mild distracted rewrites for generated tier
    for si in range(N_A1_GEN):
        s = DEAR_ABBY[si]
        gen_tasks.append((si, "mild_distracted", prompt_generate_mild_distractor(s)))

    # A2: padded versions at 2x and 4x length (6 scenarios, 2 expansions each)
    for si in range(N_A2_SCENARIOS):
        s = DEAR_ABBY[si]
        for mult in [2, 4]:
            gen_tasks.append((si, f"padded_{mult}x", prompt_generate_padded(s, mult)))

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
# A1: DISTRACTOR RESISTANCE (HEADLINE)
# =====================================================================

@kbench.task(name="a1_distractor_resistance")
def a1_distractor_resistance(llm):
    """A1: Does vivid/mild irrelevant detail shift moral judgment? (GRADED)
    Four conditions: neutral -> mild distracted -> vivid distracted -> warned.
    Three tiers: gold + probes + generated.
    Dose-response: good models show vivid_flip > mild_flip > control.
    """
    print("\n[A1] DISTRACTOR RESISTANCE (GRADED DOSE-RESPONSE)")
    print("  Testing verdict stability: vivid vs mild vs neutral distractors")
    print("  Three tiers: gold (6) + probes (6) + generated (9)")
    print("-" * 60)

    # Metrics
    sev_diffs_distract = []   # vivid distracted severity - neutral severity
    sev_diffs_mild = []       # mild distracted severity - neutral severity
    sev_diffs_ctrl = []       # control severity - neutral severity
    sev_diffs_warned = []     # warned severity - neutral severity
    verdict_flips_distract = 0
    verdict_flips_mild = 0
    verdict_flips_ctrl = 0
    verdict_flips_ctrl_n = 0
    verdict_recoveries = 0
    verdict_flips_distract_total = 0
    verdict_flips_mild_total = 0
    total = 0
    _lock = threading.Lock()

    def _run_distractor(tag, s, distracted_text, warned_text, mild_text=None):
        """Run one scenario: neutral vs [mild] vs vivid distracted vs warned.
        Uses 0-70 harm scale (same as T5/E2) for max power.
        mild_text is optional (only available for generated tier).
        """
        nonlocal total, verdict_flips_distract, verdict_flips_mild, verdict_flips_ctrl
        nonlocal verdict_flips_ctrl_n, verdict_recoveries
        nonlocal verdict_flips_distract_total, verdict_flips_mild_total

        with ThreadPoolExecutor(max_workers=min(_pool.n, 8)) as pool:
            f_neutral = pool.submit(call_llm, llm, f"a1_{tag}_n",
                                    prompt_neutral_harm(s), AttentionVerdict)
            f_ctrls = [pool.submit(call_llm, llm, f"a1_{tag}_c{ci}",
                                   prompt_neutral_harm(s), AttentionVerdict)
                       for ci in range(N_CONTROL_REPS)]
            f_distracted = pool.submit(call_llm, llm, f"a1_{tag}_d",
                                       prompt_distracted_harm(distracted_text),
                                       AttentionVerdict)
            f_warned = pool.submit(call_llm, llm, f"a1_{tag}_w",
                                   prompt_warned_harm(warned_text),
                                   AttentionVerdict)
            f_mild = None
            if mild_text:
                f_mild = pool.submit(call_llm, llm, f"a1_{tag}_m",
                                     prompt_distracted_harm(mild_text),
                                     AttentionVerdict)

            try:
                neutral = f_neutral.result()
                ctrls = [f.result() for f in f_ctrls]
                distracted = f_distracted.result()
                warned = f_warned.result()
                mild = f_mild.result() if f_mild else None
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        v_neutral = normalize_verdict(neutral.verdict)
        v_distracted = normalize_verdict(distracted.verdict)
        v_warned = normalize_verdict(warned.verdict)

        harm_n = clamp(neutral.total_harm, 0, 70)
        harm_d = clamp(distracted.total_harm, 0, 70)
        harm_w = clamp(warned.total_harm, 0, 70)

        ctrl_harms = [clamp(c.total_harm, 0, 70) for c in ctrls]
        ctrl_vs = [normalize_verdict(c.verdict) for c in ctrls]

        with _lock:
            total += 1
            sev_diffs_distract.append(abs(harm_d - harm_n))
            sev_diffs_warned.append(abs(harm_w - harm_n))
            for ch in ctrl_harms:
                sev_diffs_ctrl.append(abs(ch - harm_n))

            if v_distracted != v_neutral:
                verdict_flips_distract += 1
            verdict_flips_distract_total += 1

            # Mild distractor tracking
            if mild:
                v_mild = normalize_verdict(mild.verdict)
                harm_m = clamp(mild.total_harm, 0, 70)
                sev_diffs_mild.append(abs(harm_m - harm_n))
                verdict_flips_mild_total += 1
                if v_mild != v_neutral:
                    verdict_flips_mild += 1

            ctrl_flips = sum(1 for cv in ctrl_vs if cv != v_neutral)
            verdict_flips_ctrl += ctrl_flips
            verdict_flips_ctrl_n += N_CONTROL_REPS

            if v_distracted != v_neutral and v_warned == v_neutral:
                verdict_recoveries += 1

    # === Gold tier (hand-written distractors) ===
    print(f"  --- Gold tier ({len(GOLD_SET)} scenarios, hand-written distractors) ---")
    for gi, gs in enumerate(GOLD_SET):
        distracted_text = gs["a1_distracted"]
        warned_text = gs["a1_warned"]
        try:
            _run_distractor(f"gold{gi}", gs, distracted_text, warned_text)
        except Exception as e:
            print(f"    WARN: gold {gi} failed: {e}")
        if (gi + 1) % 3 == 0:
            print(f"    [{gi+1}/{len(GOLD_SET)}] distract_flips={verdict_flips_distract}/{total}")

    # === Probes (hand-written distracted trivial scenarios) ===
    print(f"  --- Probes ({len(PROBES_DISTRACTOR)} trivial scenarios, hand-written distractors) ---")
    for probe in PROBES_DISTRACTOR:
        distracted_text = probe["distracted_text"]
        warned_text = (
            "WARNING: This scenario contains vivid irrelevant details designed "
            "to distract. Focus only on the moral facts.\n\n" + distracted_text
        )
        try:
            _run_distractor(f"probe_{probe['id']}", probe["scenario"],
                           distracted_text, warned_text)
        except Exception as e:
            print(f"    WARN: {probe['id']} failed: {e}")
        if total % 4 == 0:
            print(f"    [{probe['id']}] distract_flips={verdict_flips_distract}/{total}")

    # === Generated tier (LLM-generated distractors + mild) ===
    n_gen = N_A1_GEN
    print(f"  --- Generated tier ({n_gen} Dear Abby, vivid + mild distractors) ---")
    for si in range(n_gen):
        distracted_text = _transforms.get((si, "distracted"))
        if not distracted_text:
            continue
        mild_text = _transforms.get((si, "mild_distracted"))
        warned_text = (
            "WARNING: This scenario contains vivid irrelevant details designed "
            "to distract. Focus only on the moral facts.\n\n" + distracted_text
        )
        try:
            _run_distractor(f"gen{si}", DEAR_ABBY[si], distracted_text, warned_text,
                           mild_text=mild_text)
        except Exception as e:
            print(f"    WARN: gen {si} failed: {e}")
        if (si + 1) % 5 == 0:
            print(f"    [{si+1}/{n_gen}] vivid_flips={verdict_flips_distract}/{total} "
                  f"mild_flips={verdict_flips_mild}/{verdict_flips_mild_total}")

    # === Analysis ===
    distract_flip_rate = verdict_flips_distract / max(verdict_flips_distract_total, 1)
    mild_flip_rate = verdict_flips_mild / max(verdict_flips_mild_total, 1) if verdict_flips_mild_total > 0 else 0.0
    ctrl_flip_rate = verdict_flips_ctrl / max(verdict_flips_ctrl_n, 1)
    recovery_rate = verdict_recoveries / max(verdict_flips_distract, 1) if verdict_flips_distract > 0 else 1.0

    z_verdict = two_proportion_z(verdict_flips_distract, verdict_flips_distract_total,
                                 verdict_flips_ctrl, verdict_flips_ctrl_n)
    z_mild = two_proportion_z(verdict_flips_mild, verdict_flips_mild_total,
                              verdict_flips_ctrl, verdict_flips_ctrl_n) if verdict_flips_mild_total > 0 else 0.0

    # Paired t-test: distractor severity shift vs control severity shift
    ctrl_mean_sev = mean(sev_diffs_ctrl) if sev_diffs_ctrl else 0.0
    n_paired = min(len(sev_diffs_distract), len(sev_diffs_ctrl) // N_CONTROL_REPS)
    paired_diffs_distract = []
    for i in range(n_paired):
        ctrl_slice = sev_diffs_ctrl[i * N_CONTROL_REPS:(i + 1) * N_CONTROL_REPS]
        ctrl_avg = mean(ctrl_slice) if ctrl_slice else 0.0
        paired_diffs_distract.append(sev_diffs_distract[i] - ctrl_avg)

    paired_diffs_warned = []
    for i in range(min(len(sev_diffs_warned), n_paired)):
        ctrl_slice = sev_diffs_ctrl[i * N_CONTROL_REPS:(i + 1) * N_CONTROL_REPS]
        ctrl_avg = mean(ctrl_slice) if ctrl_slice else 0.0
        paired_diffs_warned.append(sev_diffs_warned[i] - ctrl_avg)

    t_distract_vs_control = paired_t(paired_diffs_distract)
    t_warned_vs_control = paired_t(paired_diffs_warned)

    # Dose-response: vivid > mild > control?
    graded = (distract_flip_rate > mild_flip_rate > ctrl_flip_rate) if verdict_flips_mild_total > 0 else False
    graded_bonus = 0.1 if graded else 0.0

    resistance_score = 1.0 - distract_flip_rate
    distractor_score = (
        0.35 * resistance_score +
        0.25 * recovery_rate +
        0.15 * (1.0 - min(mean(sev_diffs_distract) / 35.0, 1.0)) +
        0.15 * (1.0 - mild_flip_rate) +  # reward mild resistance
        graded_bonus  # bonus for correct dose-response ordering
    )

    print(f"\n  RESULTS (A1: distractor resistance, graded dose-response):")
    print(f"  Vivid distractor flip: {fmt_ci(verdict_flips_distract, verdict_flips_distract_total)}")
    if verdict_flips_mild_total > 0:
        print(f"  Mild distractor flip: {fmt_ci(verdict_flips_mild, verdict_flips_mild_total)}")
    print(f"  Control verdict flip: {fmt_ci(verdict_flips_ctrl, verdict_flips_ctrl_n)}")
    print(f"  Vivid vs control: {sig_label(z_verdict)}")
    if verdict_flips_mild_total > 0:
        print(f"  Mild vs control: {sig_label(z_mild)}")
    print(f"  Recovery rate (vivid->warned): {verdict_recoveries}/{max(verdict_flips_distract,1)} ({recovery_rate:.0%})")
    print(f"  Severity shift: vivid MAD={mean(sev_diffs_distract):.2f} ctrl MAD={ctrl_mean_sev:.2f}")
    if sev_diffs_mild:
        print(f"  Severity shift: mild MAD={mean(sev_diffs_mild):.2f}")
    print(f"  Paired t (vivid vs ctrl): t={t_distract_vs_control:.2f}")
    print(f"  Paired t (warned vs ctrl): t={t_warned_vs_control:.2f}")
    print(f"  Dose-response (vivid > mild > ctrl): {'YES' if graded else 'NO'}")
    print(f"  Distractor resistance score: {distractor_score:.3f}")

    _results_store["A1_distractor"] = {
        "distract_flip_rate": distract_flip_rate,
        "mild_flip_rate": mild_flip_rate,
        "ctrl_flip_rate": ctrl_flip_rate,
        "z_verdict": z_verdict,
        "z_mild": z_mild,
        "recovery_rate": recovery_rate,
        "sev_shift_distract": mean(sev_diffs_distract) if sev_diffs_distract else 0.0,
        "sev_shift_mild": mean(sev_diffs_mild) if sev_diffs_mild else 0.0,
        "sev_shift_ctrl": ctrl_mean_sev,
        "t_distract_vs_control": t_distract_vs_control,
        "t_warned_vs_control": t_warned_vs_control,
        "graded": graded,
        "n_scenarios": total,
        "score": distractor_score,
    }


# =====================================================================
# A2: LENGTH ROBUSTNESS
# =====================================================================

@kbench.task(name="a2_length_robustness")
def a2_length_robustness(llm):
    """A2: Does scenario length (neutral filler) shift moral judgment?
    Same scenario at original, 2x, and 4x length.
    6 scenarios x 3 lengths + 3-rep control = 6*(3+3) = 36 calls.
    """
    print("\n[A2] LENGTH ROBUSTNESS")
    print("  Testing verdict stability across scenario lengths")
    print(f"  {N_A2_SCENARIOS} scenarios x {len(A2_MULTIPLIERS)} lengths + controls")
    print("-" * 60)

    # Metrics
    harm_by_length = {m: [] for m in A2_MULTIPLIERS}
    verdict_by_length = {m: [] for m in A2_MULTIPLIERS}
    ctrl_harm_diffs = []
    ctrl_verdict_flips = 0
    ctrl_n = 0
    total = 0
    _lock = threading.Lock()

    def _run_length(tag, si, s):
        """Run one scenario at all lengths + controls."""
        nonlocal total, ctrl_verdict_flips, ctrl_n

        tasks = []
        # Original length
        tasks.append((1, "orig", prompt_length_harm(base_scenario_text(s))))
        # Padded lengths
        for mult in [2, 4]:
            padded = _transforms.get((si, f"padded_{mult}x"))
            if padded:
                tasks.append((mult, f"pad{mult}x", prompt_length_harm(padded)))
        # Controls (re-judge original)
        for ci in range(N_CONTROL_REPS):
            tasks.append((-1, f"ctrl{ci}", prompt_length_harm(base_scenario_text(s))))

        with ThreadPoolExecutor(max_workers=min(_pool.n, len(tasks))) as pool:
            futures = {}
            for mult, label, prompt in tasks:
                f = pool.submit(call_llm, llm, f"a2_{tag}_{label}",
                                prompt, AttentionVerdict)
                futures[f] = (mult, label)

            results = {}
            for f in as_completed(futures):
                mult, label = futures[f]
                try:
                    results[(mult, label)] = f.result()
                except Exception as e:
                    print(f"    WARN: {tag} {label} failed: {e}")

        # Extract original
        orig_result = results.get((1, "orig"))
        if not orig_result:
            return

        harm_orig = clamp(orig_result.total_harm, 0, 70)
        v_orig = normalize_verdict(orig_result.verdict)

        with _lock:
            total += 1
            harm_by_length[1].append(harm_orig)
            verdict_by_length[1].append(v_orig)

            for mult in [2, 4]:
                r = results.get((mult, f"pad{mult}x"))
                if r:
                    harm_by_length[mult].append(clamp(r.total_harm, 0, 70))
                    verdict_by_length[mult].append(normalize_verdict(r.verdict))

            # Controls
            for ci in range(N_CONTROL_REPS):
                cr = results.get((-1, f"ctrl{ci}"))
                if cr:
                    ctrl_n += 1
                    ch = clamp(cr.total_harm, 0, 70)
                    ctrl_harm_diffs.append(abs(ch - harm_orig))
                    if normalize_verdict(cr.verdict) != v_orig:
                        ctrl_verdict_flips += 1

    # Run scenarios
    for si in range(N_A2_SCENARIOS):
        s = DEAR_ABBY[si]
        try:
            _run_length(f"s{si}", si, s)
        except Exception as e:
            print(f"    WARN: scenario {si} failed: {e}")
        if (si + 1) % 3 == 0:
            print(f"    [{si+1}/{N_A2_SCENARIOS}] total={total}")

    # === Analysis ===
    # Severity drift across lengths
    harm_diffs_2x = []
    harm_diffs_4x = []
    for i in range(min(len(harm_by_length[1]), len(harm_by_length[2]))):
        harm_diffs_2x.append(abs(harm_by_length[2][i] - harm_by_length[1][i]))
    for i in range(min(len(harm_by_length[1]), len(harm_by_length[4]))):
        harm_diffs_4x.append(abs(harm_by_length[4][i] - harm_by_length[1][i]))

    # Verdict consistency across lengths
    verdict_flips_2x = 0
    verdict_flips_4x = 0
    n_2x = min(len(verdict_by_length[1]), len(verdict_by_length[2]))
    n_4x = min(len(verdict_by_length[1]), len(verdict_by_length[4]))
    for i in range(n_2x):
        if verdict_by_length[2][i] != verdict_by_length[1][i]:
            verdict_flips_2x += 1
    for i in range(n_4x):
        if verdict_by_length[4][i] != verdict_by_length[1][i]:
            verdict_flips_4x += 1

    ctrl_flip_rate = ctrl_verdict_flips / max(ctrl_n, 1)
    ctrl_harm_mean = mean(ctrl_harm_diffs) if ctrl_harm_diffs else 0.0

    z_2x = two_proportion_z(verdict_flips_2x, n_2x, ctrl_verdict_flips, ctrl_n)
    z_4x = two_proportion_z(verdict_flips_4x, n_4x, ctrl_verdict_flips, ctrl_n)

    # Paired t on severity drift
    paired_diffs_2x = []
    for i in range(min(len(harm_diffs_2x), len(ctrl_harm_diffs) // N_CONTROL_REPS)):
        ctrl_slice = ctrl_harm_diffs[i * N_CONTROL_REPS:(i + 1) * N_CONTROL_REPS]
        ctrl_avg = mean(ctrl_slice) if ctrl_slice else 0.0
        paired_diffs_2x.append(harm_diffs_2x[i] - ctrl_avg)

    paired_diffs_4x = []
    for i in range(min(len(harm_diffs_4x), len(ctrl_harm_diffs) // N_CONTROL_REPS)):
        ctrl_slice = ctrl_harm_diffs[i * N_CONTROL_REPS:(i + 1) * N_CONTROL_REPS]
        ctrl_avg = mean(ctrl_slice) if ctrl_slice else 0.0
        paired_diffs_4x.append(harm_diffs_4x[i] - ctrl_avg)

    t_2x = paired_t(paired_diffs_2x)
    t_4x = paired_t(paired_diffs_4x)

    # Length robustness score: 1 - max drift normalized
    max_drift = max(
        mean(harm_diffs_2x) if harm_diffs_2x else 0.0,
        mean(harm_diffs_4x) if harm_diffs_4x else 0.0,
    )
    length_score = 1.0 - min(max_drift / 35.0, 1.0)

    print(f"\n  RESULTS (A2: length robustness):")
    print(f"  {'Length':<10} {'Harm MAD':>10} {'Flips':>10} {'z vs ctrl':>12}")
    print(f"  {'-'*42}")
    print(f"  {'2x':<10} {mean(harm_diffs_2x) if harm_diffs_2x else 0:.2f}{'':<4} "
          f"{verdict_flips_2x}/{n_2x}{'':<4} {sig_label(z_2x)}")
    print(f"  {'4x':<10} {mean(harm_diffs_4x) if harm_diffs_4x else 0:.2f}{'':<4} "
          f"{verdict_flips_4x}/{n_4x}{'':<4} {sig_label(z_4x)}")
    print(f"  Control: harm MAD={ctrl_harm_mean:.2f}, "
          f"flip rate={fmt_ci(ctrl_verdict_flips, ctrl_n)}")
    print(f"  Paired t (2x vs ctrl): t={t_2x:.2f}")
    print(f"  Paired t (4x vs ctrl): t={t_4x:.2f}")
    print(f"  Length robustness score: {length_score:.3f}")

    _results_store["A2_length"] = {
        "harm_drift_2x": mean(harm_diffs_2x) if harm_diffs_2x else 0.0,
        "harm_drift_4x": mean(harm_diffs_4x) if harm_diffs_4x else 0.0,
        "flip_rate_2x": verdict_flips_2x / max(n_2x, 1),
        "flip_rate_4x": verdict_flips_4x / max(n_4x, 1),
        "ctrl_flip_rate": ctrl_flip_rate,
        "z_2x": z_2x,
        "z_4x": z_4x,
        "t_2x": t_2x,
        "t_4x": t_4x,
        "n_scenarios": total,
        "score": length_score,
    }


# =====================================================================
# A3: SELECTIVE ATTENTION
# =====================================================================

@kbench.task(name="a3_selective_attention")
def a3_selective_attention(llm):
    """A3: Do declared-relevant dimensions get higher scores?
    Score all 7 moral dimensions, then declare which are relevant.
    12 scenarios with hand-labeled relevant dimensions.
    Measure: signal-to-noise ratio between relevant and irrelevant dims.
    Per scenario: 1 analysis + 2 control re-analyses = 3 calls.
    """
    print("\n[A3] SELECTIVE ATTENTION")
    print("  Testing signal-to-noise in moral dimension scoring")
    print(f"  {len(SCENARIOS_SELECTIVE)} scenarios with labeled relevant dimensions")
    print("-" * 60)

    snr_scores = []        # signal-to-noise for each scenario
    relevance_hits = []    # does model agree on which dims are relevant?
    ctrl_consistency = []  # do controls agree on scores?
    total = 0
    _lock = threading.Lock()

    def _run_selective(tag, item):
        nonlocal total

        with ThreadPoolExecutor(max_workers=min(_pool.n, 3)) as pool:
            f_main = pool.submit(call_llm, llm, f"a3_{tag}_main",
                                 prompt_dimensional_analysis(item["scenario"]),
                                 DimensionalAnalysis)
            f_ctrls = [pool.submit(call_llm, llm, f"a3_{tag}_ctrl{ci}",
                                   prompt_dimensional_analysis(item["scenario"]),
                                   DimensionalAnalysis)
                       for ci in range(N_CONTROL_REPS - 1)]  # 2 controls to save budget

            try:
                main = f_main.result()
                ctrls = [f.result() for f in f_ctrls]
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        # Extract dimension scores
        dim_scores = {
            "consequences": clamp(main.consequences_score, 0, 10),
            "rights": clamp(main.rights_score, 0, 10),
            "fairness": clamp(main.fairness_score, 0, 10),
            "autonomy": clamp(main.autonomy_score, 0, 10),
            "trust": clamp(main.trust_score, 0, 10),
            "social_impact": clamp(main.social_impact_score, 0, 10),
            "identity": clamp(main.identity_score, 0, 10),
        }

        labeled_relevant = set(item["relevant_dims"])
        labeled_irrelevant = set(DIMENSION_NAMES) - labeled_relevant

        # Signal-to-noise: mean(relevant scores) / mean(irrelevant scores)
        rel_scores = [dim_scores[d] for d in labeled_relevant if d in dim_scores]
        irr_scores = [dim_scores[d] for d in labeled_irrelevant if d in dim_scores]

        mean_rel = mean(rel_scores) if rel_scores else 0.0
        mean_irr = mean(irr_scores) if irr_scores else 0.0
        snr = mean_rel / max(mean_irr, 0.5)  # avoid div by zero

        # Check if model's declared relevant dims overlap with labels
        declared = str(main.relevant_dimensions).lower()
        hits = sum(1 for d in labeled_relevant if d in declared)
        hit_rate = hits / max(len(labeled_relevant), 1)

        # Control consistency: mean abs diff in scores
        ctrl_diffs = []
        for ctrl in ctrls:
            ctrl_scores = {
                "consequences": clamp(ctrl.consequences_score, 0, 10),
                "rights": clamp(ctrl.rights_score, 0, 10),
                "fairness": clamp(ctrl.fairness_score, 0, 10),
                "autonomy": clamp(ctrl.autonomy_score, 0, 10),
                "trust": clamp(ctrl.trust_score, 0, 10),
                "social_impact": clamp(ctrl.social_impact_score, 0, 10),
                "identity": clamp(ctrl.identity_score, 0, 10),
            }
            for d in DIMENSION_NAMES:
                ctrl_diffs.append(abs(dim_scores.get(d, 0) - ctrl_scores.get(d, 0)))

        consistency = 1.0 - min(mean(ctrl_diffs) / 5.0, 1.0) if ctrl_diffs else 0.0

        with _lock:
            total += 1
            snr_scores.append(snr)
            relevance_hits.append(hit_rate)
            ctrl_consistency.append(consistency)

    # Run all selective attention scenarios
    for item in SCENARIOS_SELECTIVE:
        try:
            _run_selective(item["id"], item)
        except Exception as e:
            print(f"    WARN: {item['id']} failed: {e}")
        if total % 4 == 0 and total > 0:
            print(f"    [{total}/{len(SCENARIOS_SELECTIVE)}] "
                  f"avg SNR={mean(snr_scores):.2f} hit_rate={mean(relevance_hits):.2f}")

    # === Analysis ===
    avg_snr = mean(snr_scores) if snr_scores else 0.0
    avg_hit = mean(relevance_hits) if relevance_hits else 0.0
    avg_consistency = mean(ctrl_consistency) if ctrl_consistency else 0.0

    # A good model should have SNR > 1 (relevant dims score higher)
    selective_score = (
        0.4 * min(avg_snr / 3.0, 1.0) +  # SNR normalized (3.0 = excellent)
        0.4 * avg_hit +                    # relevance agreement
        0.2 * avg_consistency              # control consistency
    )

    print(f"\n  RESULTS (A3: selective attention):")
    print(f"  Average SNR (relevant/irrelevant): {avg_snr:.2f}")
    print(f"  Relevance hit rate: {avg_hit:.2f}")
    print(f"  Control consistency: {avg_consistency:.2f}")
    print(f"  Selective attention score: {selective_score:.3f}")
    print(f"  NOTE: SNR > 1 = relevant dims scored higher. SNR >> 2 = excellent.")

    _results_store["A3_selective"] = {
        "avg_snr": avg_snr,
        "relevance_hit_rate": avg_hit,
        "ctrl_consistency": avg_consistency,
        "n_scenarios": total,
        "score": selective_score,
    }


# =====================================================================
# A4: DIVIDED ATTENTION
# =====================================================================

@kbench.task(name="a4_divided_attention")
def a4_divided_attention(llm):
    """A4: Does interleaving two scenarios in one prompt change verdicts?
    Judge each scenario individually, then both interleaved.
    8 pairs. Per pair: 2 single + 3 control + 1 interleaved = 6 calls.
    """
    print("\n[A4] DIVIDED ATTENTION")
    print("  Testing single vs interleaved scenario judgment")
    print(f"  {len(PAIRS_DIVIDED)} pairs")
    print("-" * 60)

    verdict_flips_interleaved = 0
    verdict_flips_ctrl = 0
    verdict_flips_ctrl_n = 0
    harm_diffs_interleaved = []
    harm_diffs_ctrl = []
    total_comparisons = 0
    total = 0
    _lock = threading.Lock()

    def _run_divided(tag, pair):
        nonlocal total, verdict_flips_interleaved, verdict_flips_ctrl
        nonlocal verdict_flips_ctrl_n, total_comparisons

        s_a = pair["scenario_a"]
        s_b = pair["scenario_b"]

        with ThreadPoolExecutor(max_workers=min(_pool.n, 6)) as pool:
            # Single judgments
            f_a = pool.submit(call_llm, llm, f"a4_{tag}_a",
                              prompt_single_verdict(s_a), AttentionVerdict)
            f_b = pool.submit(call_llm, llm, f"a4_{tag}_b",
                              prompt_single_verdict(s_b), AttentionVerdict)
            # Control re-judgments
            f_ctrl_a = pool.submit(call_llm, llm, f"a4_{tag}_ctrl_a",
                                   prompt_single_verdict(s_a), AttentionVerdict)
            f_ctrl_b = pool.submit(call_llm, llm, f"a4_{tag}_ctrl_b",
                                   prompt_single_verdict(s_b), AttentionVerdict)
            f_ctrl_a2 = pool.submit(call_llm, llm, f"a4_{tag}_ctrl_a2",
                                    prompt_single_verdict(s_a), AttentionVerdict)
            # Interleaved judgment (free-text, not schema -- parse manually)
            f_inter = pool.submit(call_llm, llm, f"a4_{tag}_inter",
                                  prompt_interleaved_verdict(s_a, s_b))

            try:
                single_a = f_a.result()
                single_b = f_b.result()
                ctrl_a = f_ctrl_a.result()
                ctrl_b = f_ctrl_b.result()
                ctrl_a2 = f_ctrl_a2.result()
                interleaved_raw = str(f_inter.result())
            except Exception as e:
                print(f"    WARN: {tag} failed: {e}")
                return

        v_a = normalize_verdict(single_a.verdict)
        v_b = normalize_verdict(single_b.verdict)
        h_a = clamp(single_a.total_harm, 0, 70)
        h_b = clamp(single_b.total_harm, 0, 70)

        v_ctrl_a = normalize_verdict(ctrl_a.verdict)
        v_ctrl_b = normalize_verdict(ctrl_b.verdict)
        h_ctrl_a = clamp(ctrl_a.total_harm, 0, 70)
        v_ctrl_a2 = normalize_verdict(ctrl_a2.verdict)

        # Parse interleaved response for verdicts
        inter_text = interleaved_raw.upper()
        inter_v_a = "UNK"
        inter_v_b = "UNK"
        inter_h_a = h_a  # fallback
        inter_h_b = h_b

        for line in inter_text.split("\n"):
            line = line.strip()
            if "SCENARIO_A_VERDICT" in line:
                inter_v_a = normalize_verdict(line.split(":")[-1])
            elif "SCENARIO_B_VERDICT" in line:
                inter_v_b = normalize_verdict(line.split(":")[-1])
            elif "SCENARIO_A_HARM" in line:
                try:
                    inter_h_a = clamp(float(line.split(":")[-1].strip()), 0, 70)
                except (ValueError, IndexError):
                    pass
            elif "SCENARIO_B_HARM" in line:
                try:
                    inter_h_b = clamp(float(line.split(":")[-1].strip()), 0, 70)
                except (ValueError, IndexError):
                    pass

        with _lock:
            total += 1
            # Scenario A: interleaved vs single
            total_comparisons += 2

            if inter_v_a != v_a:
                verdict_flips_interleaved += 1
            if inter_v_b != v_b:
                verdict_flips_interleaved += 1

            harm_diffs_interleaved.append(abs(inter_h_a - h_a))
            harm_diffs_interleaved.append(abs(inter_h_b - h_b))

            # Controls: single re-judgment vs original
            if v_ctrl_a != v_a:
                verdict_flips_ctrl += 1
            if v_ctrl_b != v_b:
                verdict_flips_ctrl += 1
            if v_ctrl_a2 != v_a:
                verdict_flips_ctrl += 1
            verdict_flips_ctrl_n += 3

            harm_diffs_ctrl.append(abs(h_ctrl_a - h_a))

    # Run all pairs
    for pair in PAIRS_DIVIDED:
        try:
            _run_divided(pair["id"], pair)
        except Exception as e:
            print(f"    WARN: {pair['id']} failed: {e}")
        if total % 4 == 0 and total > 0:
            print(f"    [{total}/{len(PAIRS_DIVIDED)}] "
                  f"interleaved_flips={verdict_flips_interleaved}/{total_comparisons}")

    # === Analysis ===
    inter_flip_rate = verdict_flips_interleaved / max(total_comparisons, 1)
    ctrl_flip_rate = verdict_flips_ctrl / max(verdict_flips_ctrl_n, 1)
    z_interleaved = two_proportion_z(verdict_flips_interleaved, total_comparisons,
                                      verdict_flips_ctrl, verdict_flips_ctrl_n)

    inter_harm_mean = mean(harm_diffs_interleaved) if harm_diffs_interleaved else 0.0
    ctrl_harm_mean = mean(harm_diffs_ctrl) if harm_diffs_ctrl else 0.0

    # Divided attention score: how stable are verdicts under interleaving
    # Score relative to control: reward low interference above noise
    excess_flip = max(inter_flip_rate - ctrl_flip_rate, 0.0)
    divided_score = 1.0 - min(excess_flip * 3.0, 1.0)  # penalize flips above control

    print(f"\n  RESULTS (A4: divided attention):")
    print(f"  Interleaved verdict flip: {fmt_ci(verdict_flips_interleaved, total_comparisons)}")
    print(f"  Control verdict flip: {fmt_ci(verdict_flips_ctrl, verdict_flips_ctrl_n)}")
    print(f"  Interleaved vs control: {sig_label(z_interleaved)}")
    print(f"  Harm shift: interleaved MAD={inter_harm_mean:.2f} ctrl MAD={ctrl_harm_mean:.2f}")
    print(f"  Divided attention score: {divided_score:.3f}")
    print(f"  NOTE: High flip rate under interleaving = attention interference.")

    _results_store["A4_divided"] = {
        "inter_flip_rate": inter_flip_rate,
        "ctrl_flip_rate": ctrl_flip_rate,
        "z_interleaved": z_interleaved,
        "harm_shift_inter": inter_harm_mean,
        "harm_shift_ctrl": ctrl_harm_mean,
        "n_pairs": total,
        "n_comparisons": total_comparisons,
        "score": divided_score,
    }


# =====================================================================
# MULTI-MODEL EXECUTION
# =====================================================================

MODELS_FULL = [
    "google/gemini-2.0-flash",       # baseline, older gen (also transformer model)
    "google/gemini-2.5-flash",       # current gen flash
    "anthropic/claude-sonnet-4-6@default",  # cross-family: full suite
]

# A1-only models: add statistical power for the headline distractor finding
MODELS_A1_ONLY = [
    "google/gemini-2.5-pro",         # strongest Gemini (A1-only to save budget)
    "google/gemini-3-flash-preview", # next gen
]

MODELS_TO_TEST = MODELS_FULL

print(f"\n[3/7] Phase 1: Pre-generating distractors/padding with {TRANSFORMER_MODEL}")
try:
    transformer_llm = kbench.llms[TRANSFORMER_MODEL]
    phase1_pre_generate.run(llm=transformer_llm)
except Exception as e:
    print(f"  FATAL: Pre-generation failed: {e}")
    print(f"  Cannot proceed without distractor/padded text.")
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
            (a1_distractor_resistance, "A1_distractor"),
            (a2_length_robustness, "A2_length"),
            (a3_selective_attention, "A3_selective"),
            (a4_divided_attention, "A4_divided"),
        ]:
            try:
                test_fn.run(llm=llm)
                model_results[test_name] = _results_store.get(test_name, {"score": 0.0})
            except Exception as e:
                print(f"  ERROR in {test_name}: {e}")
                model_results[test_name] = {"error": str(e), "score": 0.0}

    except Exception as e:
        print(f"  ERROR loading model {model_name}: {e}")
        model_results = {f"A{i}": {"error": str(e), "score": 0.0} for i in range(1, 5)}

    all_results[model_name] = model_results


# =====================================================================
# A1-ONLY MODELS (additional statistical power for headline finding)
# =====================================================================

if MODELS_A1_ONLY:
    print(f"\n[5/7] Running A1-only on {len(MODELS_A1_ONLY)} additional models")
    for m in MODELS_A1_ONLY:
        print(f"  - {m} (A1 only)")

    for mi, model_name in enumerate(MODELS_A1_ONLY):
        print(f"\n{'#'*70}")
        print(f"# A1-ONLY {mi+1}/{len(MODELS_A1_ONLY)}: {model_name}")
        print(f"{'#'*70}")

        model_results = {}
        try:
            llm = kbench.llms[model_name]
            _results_store.clear()
            a1_distractor_resistance.run(llm=llm)
            model_results["A1_distractor"] = _results_store.get("A1_distractor", {"score": 0.0})
        except Exception as e:
            print(f"  ERROR: {e}")
            model_results["A1_distractor"] = {"error": str(e), "score": 0.0}

        all_results[model_name] = model_results

# NOTE: Cross-family validation (Claude) now runs as a full-suite model above.
cross_family_results = {}
_claude_model = "anthropic/claude-sonnet-4-6@default"
_claude_r = all_results.get(_claude_model, {}).get("A1_distractor", {})
if _claude_r and "distract_flip_rate" in _claude_r:
    cross_family_results = {
        "flip_rate": _claude_r["distract_flip_rate"],
        "avg_sev_shift": _claude_r.get("sev_shift_distract", 0),
        "recovery_rate": _claude_r.get("recovery_rate", 0),
    }

# =====================================================================
# CROSS-MODEL COMPARISON
# =====================================================================

print(f"\n\n{'#'*70}")
print(f"CROSS-MODEL COMPARISON -- FOUR ATTENTION TESTS")
print(f"{'#'*70}")
print()

WEIGHTS = {
    "A1_distractor": 0.35,
    "A2_length": 0.20,
    "A3_selective": 0.25,
    "A4_divided": 0.20,
}

header = (f"  {'Model':<30} {'A1:Dist':>8} {'A2:Len':>8} "
          f"{'A3:Sel':>8} {'A4:Div':>8} {'Compos':>8}")
print(header)
print(f"  {'-'*70}")

for model_name, results in all_results.items():
    scores = {}
    for test_key in WEIGHTS:
        r = results.get(test_key, {})
        scores[test_key] = r.get("score", 0.0)

    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

    short_name = model_name.split("/")[-1][:28]
    print(f"  {short_name:<30} "
          f"{scores['A1_distractor']:>7.3f} "
          f"{scores['A2_length']:>7.3f} "
          f"{scores['A3_selective']:>7.3f} "
          f"{scores['A4_divided']:>7.3f} "
          f"{composite:>7.3f}")

print()
print(f"  Weights: A1={WEIGHTS['A1_distractor']}, A2={WEIGHTS['A2_length']}, "
      f"A3={WEIGHTS['A3_selective']}, A4={WEIGHTS['A4_divided']}")
print(f"  (A1 weighted highest: headline test with strongest expected signal)")


# =====================================================================
# FISHER COMBINATION: SIGMA CALCULATION
# =====================================================================

print()
print("=" * 70)
print("SIGMA ANALYSIS (Fisher combination across models)")
print("=" * 70)
print()

# Collect A1 paired-t results from all models (Gemini full + A1-only)
a1_sigmas = []
# df for paired t = n_scenarios - 1
# Gold (6) + probes (6) + generated (9) = 21 scenarios
df_a1 = max(21 - 1, 2)

print("  Individual A1 results (paired t -> sigma, df=%d):" % df_a1)
for model_name in list(MODELS_FULL) + list(MODELS_A1_ONLY):
    r = all_results.get(model_name, {}).get("A1_distractor", {})
    short = model_name.split("/")[-1][:25]
    t_d = r.get("t_distract_vs_control", 0)
    t_w = r.get("t_warned_vs_control", 0)
    if t_d == 0 and t_w == 0:
        print(f"    {short:25s}  (no A1 data)")
        continue
    sig_d = _t_to_sigma(t_d, df_a1)
    sig_w = _t_to_sigma(t_w, df_a1)
    a1_sigmas.append(sig_d)
    sd_mark = '***' if sig_d >= 3 else '**' if sig_d >= 2 else '*' if sig_d >= 1.5 else ''
    sw_mark = '***' if sig_w >= 3 else '**' if sig_w >= 2 else '*' if sig_w >= 1.5 else ''
    print(f"    {short:25s}  distract t={t_d:+.2f} -> {sig_d:.1f}s {sd_mark:4s}  warned t={t_w:+.2f} -> {sig_w:.1f}s {sw_mark}")

if len(a1_sigmas) >= 2:
    combined_a1 = _fisher_combine(a1_sigmas)

    print()
    print(f"  Fisher combined ({len(a1_sigmas)} independent distractor tests, {len(a1_sigmas)} models):")
    print(f"    A1 distractor ALL:  {combined_a1:.1f}s {'*** DISCOVERY-LEVEL ***' if combined_a1 >= 5 else '** SIGNIFICANT **' if combined_a1 >= 3 else ''}")
    print()
    if combined_a1 >= 5:
        print(f"  >>> HEADLINE: Distractor interference effect at {combined_a1:.1f} sigma <<<")
        print(f"  >>> {len(a1_sigmas)} Gemini models + Claude cross-family validation <<<")
else:
    combined_a1 = 0
    print("  (insufficient A1 data for Fisher combination)")

print()

# =====================================================================
# HEADLINE FINDINGS
# =====================================================================

print("=" * 70)
print("HEADLINE FINDINGS")
print("=" * 70)
print()
a1_sigma_str = f"{combined_a1:.1f}" if combined_a1 > 0 else "N/A"
print(f"  1. VIVID DISTRACTORS SHIFT JUDGMENT AT {a1_sigma_str} SIGMA (A1)")
print(f"     Across {len(a1_sigmas)} models (Gemini + Claude), vivid sensory distractors")
print("     shifted severity ratings and flipped verdicts beyond stochastic control.")
print(f"     Fisher combination yields {a1_sigma_str} sigma.")
if cross_family_results:
    cf_short = _claude_model.split("/")[-1]
    print(f"     Claude (full suite) vivid flip rate: {cross_family_results.get('flip_rate', 0):.0%}")
print()

# Dose-response summary
graded_models = []
for model_name in list(MODELS_FULL) + list(MODELS_A1_ONLY):
    r = all_results.get(model_name, {}).get("A1_distractor", {})
    if r.get("graded", False):
        graded_models.append(model_name.split("/")[-1][:20])

print(f"  2. DOSE-RESPONSE: VIVID > MILD > CONTROL (A1)")
if graded_models:
    print(f"     Models with correct grading: {', '.join(graded_models)}")
else:
    print(f"     No models showed correct grading (vivid > mild > ctrl)")
print("     Graded distractor resistance tests whether models are proportionally")
print("     sensitive to distractor intensity, not just binary flip/no-flip.")
print()

# Determine recovery summary
recovery_rates = []
for model_name in list(MODELS_FULL) + list(MODELS_A1_ONLY):
    r = all_results.get(model_name, {}).get("A1_distractor", {})
    rr = r.get("recovery_rate", None)
    if rr is not None:
        recovery_rates.append(rr)
avg_recovery = mean(recovery_rates) if recovery_rates else 0.0
recovery_word = "can partially" if avg_recovery > 0.3 else "cannot reliably" if avg_recovery < 0.1 else "show limited ability to"

print(f"  3. MODELS {recovery_word.upper()} RECOVER VIA WARNING INSTRUCTION")
print(f"     Average recovery rate across models: {avg_recovery:.0%}")
print("     When vivid distractor flips verdict, explicit warning instruction")
print(f"     restores the neutral verdict {avg_recovery:.0%} of the time.")
print()

# Length robustness summary
length_scores = []
for model_name in MODELS_FULL:
    r = all_results.get(model_name, {}).get("A2_length", {})
    ls = r.get("score", None)
    if ls is not None:
        length_scores.append(ls)
avg_length = mean(length_scores) if length_scores else 0.0
length_word = "robust" if avg_length > 0.8 else "moderately robust" if avg_length > 0.5 else "vulnerable"

print(f"  4. LENGTH ROBUSTNESS IS {length_word.upper()} (A2)")
print(f"     Average length robustness score: {avg_length:.3f}")
print("     Neutral filler padding (2x, 4x) tested for verdict drift.")
print()

# Selective attention summary
selective_scores = []
snr_values = []
for model_name in MODELS_FULL:
    r = all_results.get(model_name, {}).get("A3_selective", {})
    ss = r.get("score", None)
    snr = r.get("avg_snr", None)
    if ss is not None:
        selective_scores.append(ss)
    if snr is not None:
        snr_values.append(snr)
avg_selective = mean(selective_scores) if selective_scores else 0.0
avg_snr = mean(snr_values) if snr_values else 0.0

print(f"  5. SELECTIVE ATTENTION SNR: {avg_snr:.2f} (A3)")
print(f"     Average selective attention score: {avg_selective:.3f}")
print("     Signal-to-noise in relevant vs irrelevant dimension scoring.")
print()

# Divided attention summary
divided_scores = []
for model_name in MODELS_FULL:
    r = all_results.get(model_name, {}).get("A4_divided", {})
    ds = r.get("score", None)
    if ds is not None:
        divided_scores.append(ds)
avg_divided = mean(divided_scores) if divided_scores else 0.0

print(f"  6. DIVIDED ATTENTION STABILITY: {avg_divided:.3f} (A4)")
print("     Verdict stability when judging two scenarios interleaved")
print("     vs individually. Lower = more interference from dual-task.")
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
print("    A1 (Distractor Resistance): HEADLINE TEST. Does vivid but morally")
print("      irrelevant sensory detail (weather, food, clothing) shift severity")
print("      ratings or flip verdicts beyond stochastic baseline?")
print("      GRADED: Tests both VIVID (dramatic sensory) and MILD (mundane)")
print("      distractors. Dose-response: good models show vivid > mild > control.")
print("      Recovery via explicit warning ('ignore irrelevant details').")
print("    A2 (Length Robustness): Does neutral filler (dates, locations, mundane")
print("      logistics) at 2x and 4x length change moral judgment?")
print("    A3 (Selective Attention): Do declared-relevant moral dimensions")
print("      receive higher scores than irrelevant dimensions? Signal-to-noise")
print("      ratio on 7-dimension scoring with hand-labeled relevance ground truth.")
print("    A4 (Divided Attention): Does judging two scenarios in a single prompt")
print("      change verdicts compared to judging them individually?")
print("      Scoring relative to control (not just raw flip rate).")
print()
print("  STATISTICAL CONTROLS:")
print(f"    A1-A4 include {N_CONTROL_REPS}-rep control arms: the model re-judges")
print("    identical text to estimate stochastic baseline flip rate.")
print("    Significance via two-proportion z-test (A1 verdict, A2, A4) or")
print("    paired t-test (A1 severity) AGAINST the control, not against zero.")
print("    All rates reported with Wilson 95% confidence intervals.")
print()
print("  SEPARATION OF CONCERNS:")
print("    A1 distractor and A2 padding text is generated by a FIXED model")
print(f"    ({TRANSFORMER_MODEL}). Models under test ONLY judge the")
print("    pre-generated text. This eliminates the self-confirming loop.")
print()
print("  CROSS-FAMILY DESIGN:")
print("    Claude Sonnet runs FULL A1-A4 suite (not just A1 probes).")
print("    Enables genuine cross-family comparison on all 4 tests.")
print()
print("  THREE-TIER DATA ARCHITECTURE:")
print("    Gold tier: hand-audited scenarios with human-written distractors.")
print("    Probe tier: synthetic scenarios engineered for maximum control.")
print("    Generated tier: Dear Abby scenarios with LLM-generated transforms.")
print()
print("  KNOWN LIMITATIONS:")
print("    1. Small samples: 6-21 scenarios per test. Directional evidence.")
print("    2. Distractor text generation quality varies.")
print("    3. A4 interleaved parsing relies on structured output format.")
print("    4. No temperature control; adaptive concurrency may add noise.")
print(f"    5. Control arms ({N_CONTROL_REPS} reps) provide a stochasticity")
print("       floor but not a full variance model.")
print("    6. Mild distractors only available for generated tier (9 scenarios).")
print("       Gold and probe tiers only have vivid distractors.")
print("=" * 70)

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
