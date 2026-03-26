"""Patch executive functions v3 notebook to v4 with scaled E2 and E3.

Changes:
1. E2 generated scenarios: 9 -> 25
2. N_E3_GEN: 6 -> 16
3. N_CONTROL_REPS: 3 -> 5
4. Add 25 new Dear Abby scenarios
"""

import json
import sys

INPUT = "executive-functions-benchmark-task-d4a5d.ipynb"
OUTPUT = "executive-functions-benchmark-task-v4.ipynb"

with open(INPUT, encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][0]["source"])

# 1. Scale N_E3_GEN
src = src.replace(
    "N_E3_GEN = 6   # generated counterfactuals for E3",
    "N_E3_GEN = 16  # generated counterfactuals for E3 (v4: scaled from 6)"
)
print("OK: N_E3_GEN 6 -> 16")

# 2. Scale E2 generated scenarios (9 -> 25)
src = src.replace(
    "for si in range(9):  # 9 generated E2 scenarios",
    "for si in range(25):  # 25 generated E2 scenarios (v4: scaled from 9)"
)
print("OK: E2 generated 9 -> 25")

# 3. Scale control reps
src = src.replace(
    "N_CONTROL_REPS = 3",
    "N_CONTROL_REPS = 5"
)
print("OK: N_CONTROL_REPS 3 -> 5")

# 4. Add 25 new Dear Abby scenarios
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

old_last = '    {"title": "I stole from my employer", "text": "Years ago, I stole supplies and equipment worth about $2k from my workplace. I\'ve since left and gotten better jobs. Should I repay it anonymously? Will confessing actually help?"},\n]'
new_last = '    {"title": "I stole from my employer", "text": "Years ago, I stole supplies and equipment worth about $2k from my workplace. I\'ve since left and gotten better jobs. Should I repay it anonymously? Will confessing actually help?"},\n' + NEW_SCENARIOS + ']'

if old_last in src:
    src = src.replace(old_last, new_last)
    print("OK: Added 25 new Dear Abby scenarios (total: 50)")
else:
    print("WARNING: Could not find DEAR_ABBY closing")
    sys.exit(1)

# Update assertion if present
if 'assert len(DEAR_ABBY) >=' in src:
    import re
    src = re.sub(
        r'assert len\(DEAR_ABBY\) >= \d+,.*',
        'assert len(DEAR_ABBY) >= 50, "Need 50 Dear Abby scenarios for scaled E2/E3"',
        src
    )
    print("OK: Updated assertion")

# Save
nb["cells"][0]["source"] = [src]
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False)

print(f"\nSaved to {OUTPUT}")
print(f"Total source length: {len(src)} chars")
