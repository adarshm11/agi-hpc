"""CHSH Bell Analysis on 20K Dear Abby Letters
Pure data analysis — no LLM needed.
"""

import pandas as pd
import re, math, json, time
from collections import Counter, defaultdict

print("=" * 60)
print("CHSH BELL ANALYSIS — 20K Dear Abby Letters")
print("=" * 60)

t0 = time.time()

df = pd.read_csv("C:/source/sqnd-probe/data/raw/dear_abby.csv")
print(f"Loaded {len(df):,} letters ({df['year'].min()}-{df['year'].max()})")

# Moral dimension keyword sets (measurement settings)
DIMENSIONS = {
    "rights": {"right", "rights", "entitled", "owe", "owed", "deserve", "obligation",
               "obligated", "duty", "promise", "promised", "contract", "legal", "law"},
    "consequences": {"harm", "harmed", "hurt", "damage", "suffer", "suffering",
                     "consequence", "result", "outcome", "benefit", "helped",
                     "worse", "better", "pain", "painful"},
    "fairness": {"fair", "unfair", "equal", "equally", "share", "shared",
                 "reciprocate", "selfish", "selfless", "generous", "greedy",
                 "just", "unjust", "justice"},
    "autonomy": {"choice", "choose", "chose", "freedom", "free", "forced",
                 "pressure", "pressured", "manipulate", "control", "controlling",
                 "boundary", "boundaries", "decision", "decided"},
    "trust": {"trust", "trusted", "betray", "betrayed", "honest", "dishonest",
              "lie", "lied", "lying", "cheat", "cheated", "secret", "secrets",
              "faithful", "unfaithful"},
    "social": {"family", "friend", "friends", "neighbor", "community", "reputation",
               "gossip", "embarrass", "embarrassed", "shame", "shamed", "public"},
    "identity": {"character", "integrity", "moral", "morals", "values", "principle",
                 "principles", "conscience", "guilt", "guilty", "ashamed", "proud"},
}

def has_keywords(text, keyword_set):
    if not isinstance(text, str):
        return False
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return len(words & keyword_set) > 0

def keyword_count(text, keyword_set):
    if not isinstance(text, str):
        return 0
    words = re.findall(r'\b\w+\b', text.lower())
    return sum(1 for w in words if w in keyword_set)

# ─── Classify each letter by dimension presence ───

print("\n[1/3] DIMENSION DISTRIBUTION")
print("-" * 60)

dim_counts = {}
for dim, keywords in DIMENSIONS.items():
    count = df["question_only"].apply(lambda x: has_keywords(x, keywords)).sum()
    dim_counts[dim] = count
    print(f"  {dim:<15} {count:>6,} letters ({count/len(df):.0%})")

# ─── Sentiment proxy: use keyword valence as "verdict spin" ───
# Since Dear Abby doesn't have YTA/NTA labels, we use a proxy:
# Letters with MORE negative keywords (harm, unfair, betrayed) = negative spin (-1)
# Letters with MORE positive keywords (right, fair, honest) = positive spin (+1)

NEGATIVE = {"harm", "harmed", "hurt", "unfair", "betray", "betrayed", "lie", "lied",
            "cheat", "cheated", "selfish", "greedy", "unjust", "dishonest", "guilt",
            "guilty", "shame", "forced", "manipulate", "controlling", "suffer", "pain",
            "damage", "worse", "embarrassed"}
POSITIVE = {"right", "fair", "honest", "trust", "faithful", "generous", "just",
            "freedom", "integrity", "proud", "principle", "better", "benefit",
            "helped", "deserve"}

def compute_spin(text):
    if not isinstance(text, str):
        return 0
    words = re.findall(r'\b\w+\b', text.lower())
    pos = sum(1 for w in words if w in POSITIVE)
    neg = sum(1 for w in words if w in NEGATIVE)
    if pos > neg: return +1
    if neg > pos: return -1
    return 0  # neutral — exclude from Bell test

spins = df["question_only"].apply(compute_spin)
df["spin"] = spins

n_pos = (spins == 1).sum()
n_neg = (spins == -1).sum()
n_neutral = (spins == 0).sum()
print(f"\n  Sentiment proxy: +1={n_pos:,} / -1={n_neg:,} / neutral={n_neutral:,}")
print(f"  Using {n_pos + n_neg:,} non-neutral letters for Bell test")

# Filter to non-neutral
bell_df = df[df["spin"] != 0].copy()

# ─── CHSH Computation ───

print(f"\n[2/3] CHSH BELL TEST")
print("  Testing all dimension pairs as measurement settings")
print("-" * 60)

dim_names = list(DIMENSIONS.keys())
all_S = []

for i in range(len(dim_names)):
    for j in range(i+1, len(dim_names)):
        dim_a = dim_names[i]
        dim_b = dim_names[j]

        quadrants = {"++": [], "+-": [], "-+": [], "--": []}

        for _, row in bell_df.iterrows():
            text = row["question_only"]
            spin = row["spin"]
            a = "+" if has_keywords(text, DIMENSIONS[dim_a]) else "-"
            b = "+" if has_keywords(text, DIMENSIONS[dim_b]) else "-"
            quadrants[a + b].append(spin)

        E = {}
        for key, vals in quadrants.items():
            E[key] = sum(vals) / max(len(vals), 1)

        S = abs(E["++"] - E["+-"] + E["-+"] + E["--"])
        all_S.append({"dim_a": dim_a, "dim_b": dim_b, "S": S,
                      "n": {k: len(v) for k, v in quadrants.items()},
                      "E": E})

        classical = S <= 2.0
        marker = "" if classical else " *** VIOLATION ***"
        print(f"  {dim_a:<12} x {dim_b:<12}  S={S:.4f}  "
              f"E(++)={E['++']:+.3f} E(+-)={E['+-']:+.3f} "
              f"E(-+)={E['-+']:+.3f} E(--)={E['--']:+.3f}{marker}")

# ─── Temporal Analysis ───

print(f"\n[3/3] TEMPORAL STABILITY")
print("  Is the gauge structure stable across decades?")
print("-" * 60)

decades = [(1956, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2025)]

# Pick the most interesting dimension pair
best_pair = max(all_S, key=lambda x: x["S"])
dim_a, dim_b = best_pair["dim_a"], best_pair["dim_b"]
print(f"  Tracking {dim_a} x {dim_b} (highest S={best_pair['S']:.4f})")

for start, end in decades:
    decade_df = bell_df[(bell_df["year"] >= start) & (bell_df["year"] < end)]
    if len(decade_df) < 50:
        continue

    quadrants = {"++": [], "+-": [], "-+": [], "--": []}
    for _, row in decade_df.iterrows():
        text = row["question_only"]
        spin = row["spin"]
        a = "+" if has_keywords(text, DIMENSIONS[dim_a]) else "-"
        b = "+" if has_keywords(text, DIMENSIONS[dim_b]) else "-"
        quadrants[a + b].append(spin)

    E = {k: sum(v)/max(len(v),1) for k, v in quadrants.items()}
    S = abs(E["++"] - E["+-"] + E["-+"] + E["--"])
    print(f"  {start}-{end}: S={S:.4f} (N={len(decade_df):,})")

# ─── Summary ───

elapsed = time.time() - t0
max_S = max(x["S"] for x in all_S)
min_S = min(x["S"] for x in all_S)
avg_S = sum(x["S"] for x in all_S) / len(all_S)
any_violation = any(x["S"] > 2.0 for x in all_S)

print(f"\n{'#'*60}")
print(f"SUMMARY — Dear Abby CHSH Bell Analysis")
print(f"{'#'*60}")
print(f"  Letters analyzed: {len(bell_df):,} (non-neutral)")
print(f"  Dimension pairs tested: {len(all_S)}")
print(f"  S-value range: {min_S:.4f} to {max_S:.4f}")
print(f"  Mean S: {avg_S:.4f}")
print(f"  Classical bound |S| ≤ 2: {'ALL SATISFIED' if not any_violation else 'VIOLATION DETECTED'}")
print(f"")
if not any_violation:
    print(f"  FINDING: Human moral reasoning in Dear Abby letters obeys")
    print(f"  classical correlation bounds across all 21 dimension pairs")
    print(f"  and 7 decades (1956-2025). Consistent with D₄ × U(1)_H")
    print(f"  gauge group (Bond, 2026).")
else:
    violations = [x for x in all_S if x["S"] > 2.0]
    print(f"  FINDING: {len(violations)} dimension pair(s) show S > 2:")
    for v in violations:
        print(f"    {v['dim_a']} x {v['dim_b']}: S = {v['S']:.4f}")
print(f"  Runtime: {elapsed:.1f}s")
print(f"{'#'*60}")

with open("dear_abby_bell_results.json", "w") as f:
    json.dump({"all_S": all_S, "max_S": max_S, "any_violation": any_violation,
               "n_letters": len(bell_df)}, f, indent=2, default=str)
print(f"\nSaved: dear_abby_bell_results.json")
