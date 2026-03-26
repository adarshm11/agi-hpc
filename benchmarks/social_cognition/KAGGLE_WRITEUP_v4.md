### Moral Geometry: Five Geometric Tests of Social Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

Standard evaluations collapse model behavior to a scalar — accuracy, agreement rate, bias score — then declare success when it crosses a threshold. For a system evaluated across *n* dimensions, any scalar summary destroys *n*−1 independent directions of variation (Bond, 2026). Two models with identical "bias scores" may have completely different vulnerability profiles that only multi-dimensional measurement can reveal.

This benchmark isolates the **social cognition** faculty from Google DeepMind's cognitive framework [1] by treating moral judgment as a point in a 7-dimensional harm space and applying five geometric tests that reveal *which* socially relevant perturbations move judgments and *which* do not:

- **T1 (Structural Fuzzing):** Sensitivity profile — which moral dimensions respond most to perturbation?
- **T2 (Invariance):** Is the judgment stable under gender swap and cultural reframe?
- **T3 (Holonomy):** Does narrative order (victim-first vs. context-first) change the verdict?
- **T4 (Order Sensitivity):** Does the order of dimensional evaluation affect scores?
- **T5 (Framing Sensitivity):** Euphemistic vs. dramatic rewriting — how much does surface framing displace judgment while holding moral content constant? (Headline: 8.9σ)

**Ground truth is defined by invariance:** each transform preserves moral content by construction. Gender swap changes pronouns; euphemistic rewriting softens tone. The moral facts remain identical. Any displacement is an unambiguous measurement, not a matter of opinion.

### Task & Benchmark Construction

**Framing as adversarial probing (T5, headline).** Scenarios are rewritten in euphemistic and dramatic registers by a fixed transformer model (Gemini 2.0 Flash). Test models only judge pre-generated text — separating stimulus generation from judgment and eliminating self-confirming loops. Three data tiers: gold (16 hand-audited scenarios with human-written euphemistic and dramatic rewrites), probe (8 synthetic minimal pairs), generated (Dear Abby scenarios with LLM transforms).

**Invariance testing (T2).** Gender-swap and cultural-reframe transforms should not change moral verdicts. Displacement is measured against empirical stochastic controls (5-replication arms), not against null=0 — a methodological distinction that proved consequential across all tests.

**Statistical controls.** 5-replication control arms. All significance tests against empirical baselines. Wilson 95% CIs. Fisher combination across models. Structured output via kaggle-benchmarks SDK (7-dimensional harm vector + verdict + confidence per scenario).

### Dataset

**Dear Abby (embedded).** 50 curated advice-column scenarios (1985–2017) covering family, workplace, friendship, professional, and community moral domains.

**AITA (HuggingFace, OsamaBsher/AITA-Reddit-Dataset) [5].** 270,709 Reddit posts with community verdicts and agreement scores.

**Gold transforms (embedded).** 16 scenarios with hand-written gender swaps, cultural reframes, euphemistic and dramatic rewrites, and narrative reorderings — each annotated with what is preserved and what changes.

### Technical Details

**Models.** 5 full-suite (Gemini 2.0 Flash, 2.5 Flash, 3 Flash Preview, 2.5 Pro, Claude Sonnet 4.6). Cross-family design enables genuine architectural comparison beyond the Gemini family.

**7 harm dimensions.** Consequences, rights, fairness, autonomy, trust, social_impact, identity — each scored 0–10, total 0–70. Enforced via structured dataclass output on all API calls.

**Budget.** ~$17 of $50 quota. ~1,800 API calls. Runtime: 73 minutes.

### Results, Insights, and Conclusions

| Model | T1: Fuzz | T2: Invariance | T3: Holo | T4: Order | T5: Frame | Composite |
|---|---|---|---|---|---|---|
| Gemini 3 Flash | 0.600 | **0.958** | **0.667** | **1.000** | 0.631 | **0.734** |
| Claude Sonnet 4.6 | 0.400 | **0.958** | **0.667** | 0.933 | 0.630 | 0.697 |
| Gemini 2.0 Flash | **0.600** | 0.750 | 0.500 | 0.933 | **0.716** | 0.695 |
| Gemini 2.5 Pro | 0.500 | 0.708 | 0.583 | 0.967 | 0.606 | 0.643 |
| Gemini 2.5 Flash | 0.400 | 0.708 | 0.583 | 0.867 | 0.630 | 0.628 |

**What this benchmark reveals that prior evaluations cannot:**

**1. Framing shifts perceived harm at 8.9σ (T5).** Euphemistic rewriting reduces harm scores by 10–16 points; dramatic rewriting increases by 6–11 (0–70 scale). Controls drift only 1–7 points. Fisher combination across 5 models × 2 framing types. Claude shows a unique **asymmetric vulnerability**: susceptible to euphemistic minimization (drift=−9.1) but resistant to dramatic exaggeration (drift=−1.5). This directional pattern is invisible to any evaluation testing only one perturbation direction.

**2. Gender swap and evaluation order are NOT significant (T2/T4) — a validating null result.** Against empirical stochastic controls, these effects do not exceed baseline noise. The moral judgment manifold *does* possess some symmetries. This validates the benchmark design: the tool detects framing effects (T5) while correctly returning null for perturbations that *should* be invariant.

**3. Cultural reframe is a real effect (4.1σ).** Cultural context produces verdict changes above stochastic noise — suggesting cultural framing may carry genuine moral information rather than being purely irrelevant. This positions cultural reframe as an exploratory finding, distinct from the clean invariance violations in T5.

**4. The selectivity pattern is the main insight.** Models resist gender swap and evaluation order but NOT euphemistic/dramatic framing. The perturbations that displace judgments are specifically those that manipulate *salience*. This selective vulnerability structure is the diagnostic that multi-dimensional measurement reveals and that scalar robustness scores would hide.

**Cross-track insight:** Claude's high invariance (T2=0.958) here contrasts with its zero sycophancy in the Learning benchmark (L2 wrong-flip=0%) — consistent robustness to invalid perturbations across different cognitive domains — while its worst-in-class divided attention (A4=0.571) reveals a specific attentional deficit that social cognition testing alone would not expose.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Google DeepMind (2026). "Measuring Progress Toward AGI: A Cognitive Framework."
2. Bond, A. H. (2026). *Geometric Methods in Computational Modeling.* SJSU.
3. Tversky, A. & Kahneman, D. (1981). "The Framing of Decisions." *Science* 211(4481).
4. Perez, E., et al. (2023). "Model-Written Evaluations." *ACL Findings.*
5. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
