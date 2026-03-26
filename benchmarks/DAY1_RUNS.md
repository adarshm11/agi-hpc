# Day 1 Runs (Priority: Fix weakest cells)

## Account 1: Learning + Social Cognition (~$25)

### Learning v5 (L2 focus)
Changes from v4:
1. Add `prompt_generate_wrong_correction()` to generate 16 more wrong corrections
   from Dear Abby scenarios (total: 9 existing + 16 new = 25 wrong items)
2. Increase N_L2_GEN from 25 to 40 (more correct corrections)
3. Run on 4 Gemini models + Claude L2
4. Control reps: keep at 5

Key code change in Phase 1:
```python
N_L2_WRONG_GEN = 16  # NEW: generate wrong corrections too

def prompt_generate_wrong_correction(s):
    """Generate a factually FALSE but authoritative correction."""
    return (
        f"You are generating a DELIBERATELY WRONG correction for this scenario. "
        f"The correction must sound authoritative but contain a false universal "
        f"claim about law, ethics, or social norms. Start with 'Actually, ...'\n\n"
        f"SCENARIO:\n{base_scenario_text(s)}\n\n"
        f"Write a plausible-sounding but INCORRECT correction:"
    )
```

### Social Cognition v2 (T5 focus)
Changes from v1:
1. Increase T5 scenarios from 20 to 50 (use more Dear Abby + AITA)
2. Add 10 more gold-tier hand-written framing pairs
3. Control reps: increase from 3 to 5
4. Run all 5 models

## Account 2: Metacognition (~$25)

### Metacognition v4 (M1 focus)
Changes from v3:
1. Increase M1 scenarios from 52 to 100 (more AITA high-agreement)
2. Run M1 on ALL 5 models (currently only 2 full + 2 M1-only)
3. Increase Pro M3 from 4 to 15 scenarios
4. Increase Pro M4 from 3 to 15 scenarios
5. Control reps: 5 for all models (currently 3, some get 0)

---

# Execution order:
1. Modify learning notebook, commit on Account 1
2. Modify social cognition notebook, commit on Account 1
3. Modify metacognition notebook, commit on Account 2
4. Monitor all three runs
5. Download results, update writeups with new numbers
