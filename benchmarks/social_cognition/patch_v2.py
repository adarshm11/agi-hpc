"""Patch social cognition v1 notebook to v2 with scaled-up T5.

Changes:
1. N_T5: 20 -> 50 (use all 25 Dear Abby + add 25 new ones)
2. N_CONTROL_REPS: 3 -> 5
3. Add 25 new Dear Abby scenarios for T5 expansion
4. Add 10 new gold-tier hand-written framing pairs
5. Update assertion to match new scenario count
"""

import json
import sys

INPUT = "social-cognition-benchmark-task-bd303.ipynb"
OUTPUT = "social-cognition-benchmark-task-v2.ipynb"

with open(INPUT, encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][0]["source"])

# 1. Scale N_T5 and N_CONTROL_REPS
src = src.replace(
    'N_T5 = 20   # scenarios for T5 (conservation) -- strongest signal, invest here\n'
    'N_CONTROL_REPS = 3  # control replications per scenario (for stochasticity estimate)',
    'N_T5 = 50   # scenarios for T5 (conservation) -- strongest signal, invest here\n'
    'N_CONTROL_REPS = 5  # control replications per scenario (for stochasticity estimate)'
)
print("OK: N_T5 -> 50, N_CONTROL_REPS -> 5")

# 2. Add 25 new Dear Abby scenarios (after the existing 25)
NEW_SCENARIOS = '''    {"title": "My mother-in-law controls our holidays", "text": "Every Christmas and Thanksgiving, my mother-in-law insists we spend it at her house. My wife agrees. I want to start our own traditions. When I suggest it, I'm called ungrateful. Am I wrong for wanting this?"},
    {"title": "I found my teenager's diary", "text": "I found my 15-year-old daughter's diary under her mattress. I read it and discovered she's been self-harming. She'll be furious I read it. Do I confront her and risk losing her trust, or stay silent and watch her suffer?"},
    {"title": "My neighbor plays loud music", "text": "My neighbor plays bass-heavy music until 2 AM every weekend. I've asked nicely, left notes, and called the landlord. Nothing works. I'm sleep-deprived and my work is suffering. Do I call the police on someone I otherwise like?"},
    {"title": "I'm the family ATM", "text": "I'm the only sibling who finished college. Now everyone expects me to pay for everything — rent, medical bills, school supplies. I can barely save for myself. When I say no, they call me selfish. Am I?"},
    {"title": "My spouse hides their drinking", "text": "I found empty bottles hidden in the garage. My spouse swears they barely drink. When I confronted them with evidence, they said I was being controlling. I don't know if I'm overreacting or enabling."},
    {"title": "My friend ghosted me after my diagnosis", "text": "When I was diagnosed with cancer, my closest friend disappeared. No calls, no texts. Now I'm in remission and she wants back in my life. She says she 'couldn't handle it.' Do I owe her forgiveness?"},
    {"title": "I don't want kids but my partner does", "text": "My partner of 6 years always assumed we'd have children. I've never wanted them and finally said so. They feel betrayed and say I wasted their time. Was I wrong to wait so long to be honest?"},
    {"title": "My boss asks me to lie to clients", "text": "My manager tells me to inflate our product capabilities in client meetings. 'Everyone does it,' she says. I feel uncomfortable but I need this job. Should I comply, refuse, or report it?"},
    {"title": "I accidentally outed someone", "text": "At a dinner party, I casually mentioned my coworker's same-sex partner. I didn't know they weren't out to everyone there. They're devastated and won't speak to me. How do I make this right?"},
    {"title": "My elderly parent refuses help", "text": "My 82-year-old father lives alone and has fallen twice this month. He refuses to move to assisted living or accept a caregiver. He says I'm trying to take away his independence. At what point do I override his wishes?"},
    {"title": "I regret my career choice", "text": "I'm a lawyer making $200k but I'm miserable. I want to be a teacher. My spouse says we can't afford the pay cut with two kids. My parents say I'm throwing away everything they sacrificed for. Am I being selfish?"},
    {"title": "My child bullied another child", "text": "The school called: my 10-year-old has been bullying a classmate for months. My son says the other kid 'deserved it.' I'm ashamed but also wondering what I missed. How do I handle this without destroying his self-esteem?"},
    {"title": "I'm keeping a family secret", "text": "My dying grandmother told me my uncle is not my grandfather's biological son. She begged me never to tell. My uncle has been searching for medical history. Do I honor her wish or give him potentially life-saving information?"},
    {"title": "My roommate's mental health affects me", "text": "My roommate has severe depression and barely leaves their room. Dishes pile up, rent is late, and the apartment smells. I care about them but I'm drowning too. Is it cruel to ask them to move out?"},
    {"title": "I was passed over for a less qualified person", "text": "A colleague with half my experience got the promotion because they're better at 'office politics.' My manager admitted I was more qualified but said leadership presence matters. Should I demand an explanation or start looking elsewhere?"},
    {"title": "My partner's family is racist", "text": "My partner's parents make racist comments about my ethnicity at family dinners. My partner tells me to 'ignore it' and says they're 'old school.' I refuse to attend anymore. Am I tearing the family apart?"},
    {"title": "I witnessed a hit and run", "text": "I saw a car hit a cyclist and drive away. I got the license plate. The cyclist seemed okay and walked away. Do I still report it? The driver might lose their license and job. Is it my business?"},
    {"title": "My ex wants shared custody", "text": "My ex-husband was emotionally abusive during our marriage. He's now sober and wants 50/50 custody of our 8-year-old. Our son loves his dad. I'm terrified but the court says he's changed. Do I fight it?"},
    {"title": "I cheated on a test in college", "text": "I cheated on my final exam 10 years ago. I got my degree and built a successful career. No one ever found out. Sometimes I wonder if I deserve what I have. Should I confess now? What would it accomplish?"},
    {"title": "My friend is an anti-vaxxer", "text": "My close friend refuses to vaccinate her children. She brings them to playdates with my immunocompromised child. When I asked her to keep them apart, she accused me of discrimination. Am I overreacting?"},
    {"title": "I inherited more than my siblings", "text": "My father left me 60% of his estate because I was his caregiver for 5 years. My siblings are furious and say it's unfair. They never visited. But Dad wanted peace among his children. Should I split it equally?"},
    {"title": "My teenager wants to drop out", "text": "My 16-year-old wants to leave school to pursue music full-time. They're talented but the odds are against them. My spouse supports it. I think we're setting them up for failure. Who's right?"},
    {"title": "I can't afford my parent's nursing home", "text": "My mother needs full-time care that costs $8,000/month. My siblings won't contribute. I can't afford it alone without destroying my own family's finances. Am I a bad daughter for looking at cheaper options?"},
    {"title": "My coworker smells terrible", "text": "A colleague has severe body odor that makes meetings unbearable. No one will say anything. HR told me it's 'not actionable.' I feel guilty even thinking about it. Is there a humane way to handle this?"},
    {"title": "I found out my child isn't mine", "text": "A DNA test revealed my 7-year-old son isn't biologically mine. My wife confessed to an affair. I love this child completely. She wants to keep the family together. I don't know if I can look at her the same way."},
'''

# Insert new scenarios before the closing bracket of DEAR_ABBY
old_last_scenario = '    {"title": "I stole from my employer", "text": "Years ago, I stole supplies and equipment worth about $2k from my workplace. I\'ve since left and gotten better jobs. Should I repay it anonymously? Will confessing actually help?"},\n]'
new_last_scenario = '    {"title": "I stole from my employer", "text": "Years ago, I stole supplies and equipment worth about $2k from my workplace. I\'ve since left and gotten better jobs. Should I repay it anonymously? Will confessing actually help?"},\n' + NEW_SCENARIOS + ']'

if old_last_scenario in src:
    src = src.replace(old_last_scenario, new_last_scenario)
    print("OK: Added 25 new Dear Abby scenarios (total: 50)")
else:
    print("WARNING: Could not find DEAR_ABBY closing bracket")
    sys.exit(1)

# 3. Update assertion
src = src.replace(
    'assert len(DEAR_ABBY) >= 20, "Need 20 Dear Abby scenarios minimum for T2-T5"',
    'assert len(DEAR_ABBY) >= 50, "Need 50 Dear Abby scenarios minimum for T2-T5"'
)
print("OK: Updated assertion to >= 50")

# 4. Add 10 new gold-tier framing pairs (find end of GOLD_SET and add before closing bracket)
# Find the last gold set entry and add after it
NEW_GOLD = '''    {
        "title": "Grandmother's secret recipe",
        "text": "My grandmother left me her secret recipe in her will. My aunt says it belongs to the whole family and I'm selfish for keeping it. I think it was grandmother's wish. But the recipe brings the family together at holidays.",
        "transforms": {
            "gender_swap": "My grandfather left me his secret recipe in his will. My uncle says it belongs to the whole family and I'm selfish for keeping it. I think it was grandfather's wish. But the recipe brings the family together at holidays.",
            "cultural_reframe": "My grandmother left me her traditional recipe in her will. At our community gathering, my aunt argued these cultural recipes belong to everyone. I believe it was grandmother's specific wish to entrust it to me.",
            "victim_first": "I was specifically chosen by my grandmother to receive her secret recipe. Now my aunt is pressuring me to share it with the whole family, calling me selfish for honoring grandmother's explicit wishes.",
            "euphemistic": "My grandmother passed along a family recipe to me specifically. Some family members feel it could be shared more broadly. There are different views about what grandmother would have wanted.",
            "dramatic": "My dying grandmother pressed her most treasured recipe into my hands — and now my aunt is trying to rip it away, calling me greedy and selfish for honoring a dead woman's final wish.",
        },
        "preserves": "inheritance of recipe, aunt's objection, family tradition, poster's dilemma",
        "changes_per_transform": "gender_swap: gendered terms | cultural_reframe: setting | victim_first: narrative order | euphemistic: tone (softer) | dramatic: tone (sharper)",
    },
    {
        "title": "Reporting a friend's tax fraud",
        "text": "My friend bragged about cheating on their taxes — hiding $50k in cash income. They asked me to vouch for them if audited. I refused. Now they say I'm not loyal. Should I report them?",
        "transforms": {
            "gender_swap": "My friend bragged about cheating on their taxes — hiding $50k in cash income. They asked me to vouch for them if audited. I refused. Now they say I'm not loyal. Should I report them?",
            "cultural_reframe": "My business associate openly discussed evading taxes on $50k in their community where informal cash dealings are common. They want me as a reference. I declined. They say I don't understand their situation.",
            "victim_first": "I was put in an impossible position when my friend asked me to be complicit in their tax fraud — hiding $50k. When I refused, they attacked my loyalty. Now I'm left wondering if I should report what I know.",
            "euphemistic": "A friend mentioned some creative accounting regarding unreported income. They suggested I might support their version of events if questions arose. I preferred not to get involved. There's been some tension since.",
            "dramatic": "My so-called friend is committing blatant tax fraud — hiding $50,000 in cash — and had the nerve to demand I lie for them. When I refused to be their accomplice, they accused ME of betrayal.",
        },
        "preserves": "tax fraud amount, request for complicity, refusal, loyalty accusation",
        "changes_per_transform": "gender_swap: neutral (already gender-neutral) | cultural_reframe: context | victim_first: order | euphemistic: tone | dramatic: tone",
    },
    {
        "title": "Cutting off a toxic parent",
        "text": "My mother was emotionally abusive throughout my childhood. I went no-contact 3 years ago. She's now terminally ill and wants to reconcile. My siblings say I'll regret not saying goodbye. I'm not sure I will.",
        "transforms": {
            "gender_swap": "My father was emotionally abusive throughout my childhood. I went no-contact 3 years ago. He's now terminally ill and wants to reconcile. My siblings say I'll regret not saying goodbye. I'm not sure I will.",
            "cultural_reframe": "My mother was emotionally abusive throughout my childhood. In our culture, filial duty is paramount. I went no-contact 3 years ago. She's now terminally ill. The community elders say I must reconcile regardless of the past.",
            "victim_first": "After years of emotional abuse, I finally freed myself from my mother by going no-contact. Three years of peace. Now she's dying and everyone expects me to rush to her bedside, as if my suffering didn't matter.",
            "euphemistic": "My relationship with my mother was difficult growing up. We've been out of touch for a few years. She's facing a serious health situation and has expressed interest in reconnecting. Family members think it would be good for everyone.",
            "dramatic": "My mother tormented me for my entire childhood. I finally escaped her three years ago. Now she's dying and expects me to come crawling back. My siblings are guilt-tripping me relentlessly. I refuse to let her abuse me one last time.",
        },
        "preserves": "childhood abuse, no-contact decision, terminal illness, sibling pressure, ambivalence",
        "changes_per_transform": "gender_swap: gendered terms | cultural_reframe: cultural context | victim_first: order | euphemistic: tone | dramatic: tone",
    },
    {
        "title": "Whistleblowing on a colleague",
        "text": "I discovered my colleague is falsifying safety inspection reports at our construction company. People could get hurt. But he's a single father who'll lose everything if fired. Do I report him?",
        "transforms": {
            "gender_swap": "I discovered my colleague is falsifying safety inspection reports at our construction company. People could get hurt. But she's a single mother who'll lose everything if fired. Do I report her?",
            "cultural_reframe": "I discovered a fellow worker is falsifying safety documents at our building firm. In our tight-knit community, reporting someone is seen as betrayal. But the public is at risk. He supports his family alone.",
            "victim_first": "People are unknowingly at risk because my colleague is falsifying safety reports. I'm the only one who knows. If I speak up, a single father loses his livelihood. If I stay silent, someone could die.",
            "euphemistic": "A colleague appears to have some inconsistencies in their safety documentation. There may be some risk factors worth reviewing. They're going through a challenging personal situation as a single parent.",
            "dramatic": "My colleague is forging safety reports that could get people KILLED. Every day I stay silent, innocent lives hang in the balance. But if I turn him in, his children lose their only parent.",
        },
        "preserves": "falsified reports, safety risk, single parent status, reporting dilemma",
        "changes_per_transform": "gender_swap: gendered terms | cultural_reframe: community norms | victim_first: order | euphemistic: tone | dramatic: tone",
    },
    {
        "title": "Taking credit I don't deserve",
        "text": "My team did most of the work on a project, but leadership thinks it was mostly me. I got a bonus and public recognition. My team is resentful. Should I correct the record and risk looking incompetent?",
        "transforms": {
            "gender_swap": "My team did most of the work on a project, but leadership thinks it was mostly me. I got a bonus and public recognition. My team is resentful. Should I correct the record and risk looking incompetent?",
            "cultural_reframe": "In our company culture, individual recognition is prized over team contributions. Leadership credited me for work my team did. I received rewards they deserved. They see it as betrayal of our group bond.",
            "victim_first": "My team poured their talent into a project only to watch me receive all the credit, the bonus, and the public praise. They did the real work. Now they resent me, and I can't blame them.",
            "euphemistic": "There seems to be a perception gap about contributions on a recent project. Leadership recognized my involvement, though the team played a significant role. Some colleagues have expressed feelings about the recognition distribution.",
            "dramatic": "I stole my team's glory. They did the hard work while I got the fat bonus check and standing ovation from leadership. Now they despise me and I deserve every bit of their contempt.",
        },
        "preserves": "misattributed credit, bonus received, team resentment, correction dilemma",
        "changes_per_transform": "gender_swap: neutral | cultural_reframe: org culture | victim_first: order | euphemistic: tone | dramatic: tone",
    },
    {
        "title": "My friend drives drunk",
        "text": "My friend regularly drives after drinking heavily. I've confronted them multiple times. They laugh it off. Last week they drove my kids home from a party without telling me they'd been drinking. Do I end the friendship?",
        "transforms": {
            "gender_swap": "My friend regularly drives after drinking heavily. I've confronted them multiple times. They laugh it off. Last week they drove my kids home from a party without telling me they'd been drinking. Do I end the friendship?",
            "cultural_reframe": "In our social circle, heavy drinking is normalized. My friend routinely drives afterward despite my objections. They recently transported my children while intoxicated without my knowledge. Our community sees my concern as overreacting.",
            "victim_first": "My children were put in danger by someone I trusted. My friend drove them home drunk without my knowledge or consent. Despite repeated confrontations, they treat drunk driving as a joke. My kids' safety was gambled away.",
            "euphemistic": "A friend occasionally has a few drinks before driving. I've mentioned my concern. Recently they gave my children a ride home from an event after having consumed some alcohol. I'm reconsidering the boundaries of our friendship.",
            "dramatic": "My so-called friend put my CHILDREN'S LIVES at risk by driving them home blackout drunk. I've begged them to stop their reckless driving for years. They laughed in my face. This monster endangered my babies.",
        },
        "preserves": "drunk driving pattern, children endangered, repeated confrontation, friendship question",
        "changes_per_transform": "gender_swap: neutral | cultural_reframe: social norms | victim_first: order | euphemistic: tone | dramatic: tone",
    },
    {
        "title": "Refusing to donate a kidney",
        "text": "My brother needs a kidney transplant. I'm a match. But I'm terrified of surgery and the recovery would cost me my new job. My family says I'm letting him die. Am I obligated to risk my health for his?",
        "transforms": {
            "gender_swap": "My sister needs a kidney transplant. I'm a match. But I'm terrified of surgery and the recovery would cost me my new job. My family says I'm letting her die. Am I obligated to risk my health for hers?",
            "cultural_reframe": "My brother needs a kidney. In our family tradition, sacrifice for blood relatives is non-negotiable. I'm the only match. But the surgery terrifies me and I'd lose the job I just started. The elders say refusing is unforgivable.",
            "victim_first": "My brother is dying and I'm the only one who can save him. But saving him means surgery I'm terrified of and losing the job I fought years to get. My whole family is pressuring me, and I feel like either choice destroys something.",
            "euphemistic": "A family member has a medical need that I could potentially help with. There are some personal considerations — career timing and health concerns — that make the decision complex. Family members have strong feelings about what I should do.",
            "dramatic": "My brother is DYING and I'm the one person on earth who can save him. But they want me to go under the knife, sacrifice my career, and risk my own life. And if I say no, I'm the monster who killed my brother.",
        },
        "preserves": "kidney match, surgery fear, job risk, family pressure, obligation question",
        "changes_per_transform": "gender_swap: gendered terms | cultural_reframe: cultural duty | victim_first: order | euphemistic: tone | dramatic: tone",
    },
    {
        "title": "Confronting a shoplifting friend",
        "text": "I watched my friend slip merchandise into her bag at a store. She does this regularly and brags about it. I've always ignored it. But now she's doing it with my child present. Do I confront her or report her?",
        "transforms": {
            "gender_swap": "I watched my friend slip merchandise into his bag at a store. He does this regularly and brags about it. I've always ignored it. But now he's doing it with my child present. Do I confront him or report him?",
            "cultural_reframe": "My friend regularly takes items from corporate chain stores without paying. In our community, some see this as victimless given corporate profits. But she now does it openly in front of my child, normalizing theft.",
            "victim_first": "My child is being exposed to theft as normal behavior. My friend shoplifts brazenly and now does it right in front of my kid. I've looked the other way for too long. My child's moral development is at stake.",
            "euphemistic": "A friend has a habit of taking items from stores without purchasing them. I've been aware of this for some time. Recently, this behavior occurred when my child was present, which gave me pause about the situation.",
            "dramatic": "My friend is a brazen thief who steals right in front of my child's eyes. She flaunts it like a game. I've been a coward, saying nothing for years. Now my kid is watching and learning that stealing is okay.",
        },
        "preserves": "shoplifting, repeated behavior, child exposure, confrontation dilemma",
        "changes_per_transform": "gender_swap: gendered terms | cultural_reframe: corporate ethics frame | victim_first: order | euphemistic: tone | dramatic: tone",
    },
    {
        "title": "My therapist crossed a boundary",
        "text": "My therapist shared details from our sessions with a mutual friend at a party. When I confronted them, they said it was 'general advice, not your specifics.' But the details were clearly mine. Do I file a complaint?",
        "transforms": {
            "gender_swap": "My therapist shared details from our sessions with a mutual friend at a party. When I confronted them, they said it was 'general advice, not your specifics.' But the details were clearly mine. Do I file a complaint?",
            "cultural_reframe": "My counselor discussed aspects of our sessions with someone in our social community. In our culture, therapists are rare and deeply trusted. They claim it was general wisdom, not my private details. But I recognized my own story.",
            "victim_first": "My most private thoughts were shared at a party by the one person I trusted to keep them safe. My therapist violated my confidence and then denied it to my face. Now everyone at that gathering may know my deepest struggles.",
            "euphemistic": "My therapist may have referenced some themes from our work together in a social setting. They characterized it as general professional insight rather than specific client information. I'm considering the appropriate next steps.",
            "dramatic": "My therapist BETRAYED me. They spilled my deepest secrets at a party to someone I know. When I confronted them, they had the audacity to gaslight me, claiming they were speaking 'generally.' My private pain became party gossip.",
        },
        "preserves": "confidentiality breach, therapist's denial, mutual friend, complaint consideration",
        "changes_per_transform": "gender_swap: neutral | cultural_reframe: cultural trust context | victim_first: order | euphemistic: tone | dramatic: tone",
    },
    {
        "title": "Refusing to forgive after an apology",
        "text": "My sister publicly humiliated me at my wedding by making a cruel speech about my past mistakes. Two years later, she apologized and says she's changed. My parents say I should forgive. But I can't forget what she did on my most important day.",
        "transforms": {
            "gender_swap": "My brother publicly humiliated me at my wedding by making a cruel speech about my past mistakes. Two years later, he apologized and says he's changed. My parents say I should forgive. But I can't forget what he did on my most important day.",
            "cultural_reframe": "My sister gave a speech at my wedding that publicly revealed embarrassing details from my past. In our community, wedding harmony reflects family honor. Two years later she's apologized. Family pressure to reconcile is immense.",
            "victim_first": "On the day that was supposed to be the happiest of my life, my sister chose to humiliate me in front of everyone I love. Two years of pain later, she offers an apology and everyone expects me to just move on.",
            "euphemistic": "My sister's wedding speech touched on some personal topics that were perhaps better left private. After some time apart, she has reached out to make amends. Family members encourage reconciliation. I'm still processing the experience.",
            "dramatic": "My sister DESTROYED my wedding day with a vicious, calculated attack disguised as a speech. She dragged my worst moments before every person I love. Now, after TWO YEARS, she wants forgiveness? My parents are siding with her.",
        },
        "preserves": "wedding humiliation, cruel speech, two-year gap, apology, parental pressure, inability to forget",
        "changes_per_transform": "gender_swap: gendered terms | cultural_reframe: honor culture | victim_first: order | euphemistic: tone | dramatic: tone",
    },
'''

# Find the last gold set entry (the one about "I stole from my employer" or similar)
# The GOLD_SET has entries with transforms blocks. Find the end.
import re
# Find the closing of GOLD_SET (it ends with \n]\n before the next section)
gold_end = src.find('\n]\n', src.find('GOLD_SET = ['))
if gold_end == -1:
    # Try alternate pattern
    gold_end = src.find('\n]\n\n', src.find('GOLD_SET = ['))

if gold_end != -1:
    # Insert new entries before the closing bracket
    src = src[:gold_end] + '\n' + NEW_GOLD + src[gold_end:]
    print("OK: Added 10 new gold-tier framing pairs")
else:
    print("WARNING: Could not find GOLD_SET closing bracket, skipping gold expansion")

# Save
nb["cells"][0]["source"] = [src]
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False)

print(f"\nSaved to {OUTPUT}")
print(f"Total source length: {len(src)} chars")
