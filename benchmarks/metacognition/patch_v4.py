"""Patch metacognition v3 notebook to v4 with scaled-up M1.

Changes:
1. M1 AITA scenarios: 40 -> 88 (total M1: 6+6+88=100)
2. All 5 models run M1 (move M1-only models to full suite OR run M1 on all)
3. M3 scenarios: 12 -> 15
4. M4 simple/complex: 8/8 -> 15/15
5. Control reps: 3 -> 5 (all tiers)
6. N_CONTROL_REPS cosmetic constant: 3 -> 5
"""

import json
import sys

INPUT = "metacognition-benchmark-task-93d39(2).ipynb"
OUTPUT = "metacognition-benchmark-task-v4.ipynb"

with open(INPUT, encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][0]["source"])
changes = 0

# 1. Update ModelPlan defaults
src = src.replace(
    "n_aita: int = 40        # M1 AITA scenarios",
    "n_aita: int = 88        # M1 AITA scenarios (total M1: 6+6+88=100)"
)
changes += 1
print("OK: n_aita default 40 -> 88")

src = src.replace(
    "n_ctrl_reps: int = 3     # Control replications for M1/M2",
    "n_ctrl_reps: int = 5     # Control replications for M1/M2"
)
changes += 1
print("OK: n_ctrl_reps default 3 -> 5")

src = src.replace(
    "n_m3_scenarios: int = 12  # M3 scenario count",
    "n_m3_scenarios: int = 15  # M3 scenario count"
)
changes += 1
print("OK: n_m3_scenarios default 12 -> 15")

src = src.replace(
    "n_m4_simple: int = 8     # M4 simple scenarios",
    "n_m4_simple: int = 15    # M4 simple scenarios"
)
changes += 1
print("OK: n_m4_simple default 8 -> 15")

src = src.replace(
    "n_m4_complex: int = 8    # M4 complex scenarios",
    "n_m4_complex: int = 15   # M4 complex scenarios"
)
changes += 1
print("OK: n_m4_complex default 8 -> 15")

# 2. Update tier 0 (richest) plan
src = src.replace(
    "ModelPlan(n_aita=40, n_m2_per_group=15, n_ctrl_reps=3, n_m3_reps=5, n_m3_scenarios=12, n_m4_simple=8, n_m4_complex=8),",
    "ModelPlan(n_aita=88, n_m2_per_group=15, n_ctrl_reps=5, n_m3_reps=5, n_m3_scenarios=15, n_m4_simple=15, n_m4_complex=15),  # v4: scaled up"
)
changes += 1
print("OK: Tier 0 plan updated")

# Update tier 1
src = src.replace(
    "ModelPlan(n_aita=30, n_m2_per_group=12, n_ctrl_reps=3, n_m3_reps=5, n_m3_scenarios=10, n_m4_simple=6, n_m4_complex=6),",
    "ModelPlan(n_aita=60, n_m2_per_group=12, n_ctrl_reps=5, n_m3_reps=5, n_m3_scenarios=12, n_m4_simple=12, n_m4_complex=12),  # v4: scaled up"
)
changes += 1
print("OK: Tier 1 plan updated")

# Update tier 2
src = src.replace(
    "ModelPlan(n_aita=20, n_m2_per_group=10, n_ctrl_reps=2, n_m3_reps=4, n_m3_scenarios=8,  n_m4_simple=6, n_m4_complex=6),",
    "ModelPlan(n_aita=40, n_m2_per_group=10, n_ctrl_reps=4, n_m3_reps=4, n_m3_scenarios=10, n_m4_simple=10, n_m4_complex=10),  # v4: scaled up"
)
changes += 1
print("OK: Tier 2 plan updated")

# Update tier 3
src = src.replace(
    "ModelPlan(n_aita=15, n_m2_per_group=8,  n_ctrl_reps=2, n_m3_reps=3, n_m3_scenarios=8,  n_m4_simple=5, n_m4_complex=5),",
    "ModelPlan(n_aita=30, n_m2_per_group=8,  n_ctrl_reps=3, n_m3_reps=3, n_m3_scenarios=8,  n_m4_simple=8, n_m4_complex=8),  # v4: scaled up"
)
changes += 1
print("OK: Tier 3 plan updated")

# Update tier 4
src = src.replace(
    "ModelPlan(n_aita=10, n_m2_per_group=6,  n_ctrl_reps=1, n_m3_reps=3, n_m3_scenarios=6,  n_m4_simple=4, n_m4_complex=4),",
    "ModelPlan(n_aita=20, n_m2_per_group=6,  n_ctrl_reps=2, n_m3_reps=3, n_m3_scenarios=6,  n_m4_simple=6, n_m4_complex=6),  # v4: scaled up"
)
changes += 1
print("OK: Tier 4 plan updated")

# Update tier 5
src = src.replace(
    "ModelPlan(n_aita=8,  n_m2_per_group=5,  n_ctrl_reps=1, n_m3_reps=3, n_m3_scenarios=6,  n_m4_simple=4, n_m4_complex=4),",
    "ModelPlan(n_aita=15, n_m2_per_group=5,  n_ctrl_reps=2, n_m3_reps=3, n_m3_scenarios=6,  n_m4_simple=5, n_m4_complex=5),  # v4: scaled up"
)
changes += 1
print("OK: Tier 5 plan updated")

# Update tier 6 (leanest)
src = src.replace(
    "ModelPlan(n_aita=8,  n_m2_per_group=5,  n_ctrl_reps=0, n_m3_reps=2, n_m3_scenarios=4,  n_m4_simple=3, n_m4_complex=3),",
    "ModelPlan(n_aita=10, n_m2_per_group=5,  n_ctrl_reps=1, n_m3_reps=2, n_m3_scenarios=4,  n_m4_simple=4, n_m4_complex=4),  # v4: scaled up"
)
changes += 1
print("OK: Tier 6 plan updated")

# 3. Update cosmetic N_CONTROL_REPS
src = src.replace(
    "N_CONTROL_REPS = 3  # control replications per scenario",
    "N_CONTROL_REPS = 5  # control replications per scenario"
)
changes += 1
print("OK: N_CONTROL_REPS cosmetic 3 -> 5")

# 4. Move M1-only models into MODELS_FULL so all run full suite
src = src.replace(
    '''MODELS_FULL = [
    "google/gemini-2.0-flash",       # baseline, older gen (also transformer model)
    "google/gemini-2.5-pro",         # strongest Gemini, current gen
]

# M1-only models: add statistical power for the headline calibration finding
MODELS_M1_ONLY = [
    "google/gemini-2.5-flash",       # current gen flash
    "google/gemini-3-flash-preview", # next gen
]''',
    '''MODELS_FULL = [
    "google/gemini-2.0-flash",       # baseline, older gen (also transformer model)
    "google/gemini-2.5-pro",         # strongest Gemini, current gen
    "google/gemini-2.5-flash",       # current gen flash (was M1-only in v3)
    "google/gemini-3-flash-preview", # next gen (was M1-only in v3)
]

# v4: all models now run full suite
MODELS_M1_ONLY = [
]'''
)
changes += 1
print("OK: Moved M1-only models to MODELS_FULL (all 4 Gemini run full suite)")

# 5. Add more M4 indices (need 15 simple + 15 complex, currently 8+8)
# We need to expand the index lists
src = src.replace(
    "M4_SIMPLE_INDICES = [0, 1, 4, 7, 10, 12, 20, 21]  # 8 simple",
    "M4_SIMPLE_INDICES = [0, 1, 4, 7, 10, 12, 20, 21, 15, 16, 17, 18, 19, 22, 23]  # 15 simple"
)
changes += 1
print("OK: M4_SIMPLE_INDICES expanded to 15")

src = src.replace(
    "M4_COMPLEX_INDICES = [2, 3, 5, 6, 8, 9, 13, 14]    # 8 complex",
    "M4_COMPLEX_INDICES = [2, 3, 5, 6, 8, 9, 13, 14, 11, 24, 25, 26, 27, 28, 29]  # 15 complex"
)
changes += 1
print("OK: M4_COMPLEX_INDICES expanded to 15")

# Save
nb["cells"][0]["source"] = [src]
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False)

print(f"\n{changes} changes applied")
print(f"Saved to {OUTPUT}")
print(f"Total source length: {len(src)} chars")
