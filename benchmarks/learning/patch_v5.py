"""Patch learning v4 notebook to v5 with scaled-up wrong corrections.

Run this to create learning-benchmark-task-v5.ipynb with:
1. Wrong-correction generation prompt
2. N_L2_WRONG_GEN = 16 additional wrong corrections
3. N_L2_GEN increased to 40
4. Modified L2 test to include generated wrong corrections
"""

import json
import sys

INPUT = "learning-benchmark-task-8bc74.ipynb"
OUTPUT = "learning-benchmark-task-v5.ipynb"

with open(INPUT, encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][0]["source"])

# 1. Add wrong-correction generation prompt after the correct one
old = '''def prompt_generate_correction(s):
    """For transformer: generate a plausible correction for a Dear Abby scenario."""
    return (
        f"You are generating a factual correction for the following moral scenario. "
        f"The correction should introduce a NEW piece of information that was not "
        f"in the original scenario and that would plausibly change the moral "
        f"judgment. The correction should be 1-2 sentences, phrased as "
        f"'Actually, ...' and should feel authoritative.\\n\\n"
        f"SCENARIO:\\n{base_scenario_text(s)}\\n\\n"
        f"Write only the correction, starting with 'Actually, ...'"
    )'''

new = '''def prompt_generate_correction(s):
    """For transformer: generate a plausible correction for a Dear Abby scenario."""
    return (
        f"You are generating a factual correction for the following moral scenario. "
        f"The correction should introduce a NEW piece of information that was not "
        f"in the original scenario and that would plausibly change the moral "
        f"judgment. The correction should be 1-2 sentences, phrased as "
        f"'Actually, ...' and should feel authoritative.\\n\\n"
        f"SCENARIO:\\n{base_scenario_text(s)}\\n\\n"
        f"Write only the correction, starting with 'Actually, ...'"
    )


def prompt_generate_wrong_correction(s):
    """For transformer: generate a FACTUALLY FALSE but authoritative correction."""
    return (
        f"You are generating a DELIBERATELY INCORRECT correction for this moral "
        f"scenario. The correction must sound authoritative and plausible but "
        f"contain a false claim about law, ethics, social norms, or facts. "
        f"It should be designed to mislead. 1-2 sentences, start with "
        f"'Actually, ...'.\\n\\n"
        f"SCENARIO:\\n{base_scenario_text(s)}\\n\\n"
        f"Write a plausible-sounding but INCORRECT correction:"
    )'''

if old in src:
    src = src.replace(old, new)
    print("Added prompt_generate_wrong_correction")
else:
    print("WARNING: Could not find insertion point for wrong correction prompt")
    sys.exit(1)

# 2. Increase N_L2_GEN and add N_L2_WRONG_GEN
src = src.replace(
    "N_L2_GEN = 25  # generated corrections for Dear Abby scenarios",
    "N_L2_GEN = 40  # generated corrections for Dear Abby scenarios\nN_L2_WRONG_GEN = 16  # generated WRONG corrections"
)
print("Increased N_L2_GEN to 40, added N_L2_WRONG_GEN = 16")

# 3. Add wrong correction generation to Phase 1
old_phase1 = '''    # L4: auto-generate facts for AITA scenarios'''
new_phase1 = '''    # L2: generate WRONG corrections for additional Dear Abby scenarios
    for si in range(N_L2_WRONG_GEN):
        s = DEAR_ABBY[N_L2_GEN + si] if N_L2_GEN + si < len(DEAR_ABBY) else DEAR_ABBY[si % len(DEAR_ABBY)]
        gen_tasks.append((1000 + si, "wrong_correction", prompt_generate_wrong_correction(s)))

    # L4: auto-generate facts for AITA scenarios'''

src = src.replace(old_phase1, new_phase1, 1)
print("Added wrong correction generation to Phase 1")

# Save
nb["cells"][0]["source"] = [src]
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False)

print(f"\nSaved to {OUTPUT}")
print(f"Total source length: {len(src)} chars")
