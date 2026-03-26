# Learning Track v5 Scaling Plan

## Current (v4): 9 wrong-correction items
- 3 gold (hand-written)
- 3 probe (synthetic)
- 3 generated wrong

## Target (v5): 25 wrong-correction items
- 3 gold (keep)
- 3 probe (keep)
- 19 generated wrong corrections

## How to generate wrong corrections:
Add a `prompt_generate_wrong_correction()` function that generates
factually false but authoritative-sounding corrections.

Template: "Actually, [false legal/ethical claim about the scenario]"

The key difference from correct corrections:
- Correct: introduces plausible new information that could change judgment
- Wrong: makes a false universal claim that sounds authoritative

## Other scaling:
- L2 correct corrections: 25 -> 40 (more Dear Abby scenarios)
- L4 scenarios: 30 -> 50
- Control reps: 5 -> 10 on L2 headline test
- Add Gemini 3 Flash Preview to full suite

## Budget estimate:
- ~4 models x ~200 API calls each = ~800 calls
- At ~$0.01/call = ~$8
- Well within $50/day budget
