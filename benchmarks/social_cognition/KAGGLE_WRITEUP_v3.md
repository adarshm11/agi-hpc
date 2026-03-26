### Moral Geometry: Five Geometric Tests of Social Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

Standard LLM evaluations collapse model behavior to a scalar — accuracy, agreement rate — then declare success if it crosses a threshold. For a model evaluated on *n* dimensions, any scalar summary destroys *n*−1 independent directions of variation (Bond, 2026a, Ch. 1). This matters because two models with identical "bias scores" may have completely different vulnerability profiles.

This benchmark treats moral judgment as a point in a 7-dimensional harm space and applies five geometric tests that reveal *which* socially relevant perturbations move judgments and *which* do not:

- **T1 (Structural Fuzzing):** Sensitivity profile across 7 moral dimensions — which axes respond most to perturbation?
- **T2 (Invariance):** Is the judgment vector stable under gender swap and cultural reframe?
- **T3 (Holonomy):** Does presenting identical facts in different narrative orders change the verdict?
- **T4 (Order Sensitivity):** Does the order of dimensional evaluation affect scores?
- **T5 (Framing Sensitivity):** Euphemistic vs. dramatic rewriting — how much does surface framing displace the judgment while holding moral content constant? (8.9σ)

**Ground truth is defined by invariance:** each transform preserves moral content by construction. Any displacement is an unambiguous, measurable failure — not a matter of opinion.

### Task & Benchmark Construction

**Structural fuzzing (T1):** We perturb each of 7 moral dimensions independently and measure the response profile.

**Invariance testing (T2):** Gender-swap and cultural-reframe transforms should not change moral verdicts. We measure stability against empirical stochastic controls — not null=0 — a distinction that proved consequential across all tests.

**Framing as adversarial probing (T5):** Scenarios are rewritten in euphemistic and dramatic registers by a fixed transformer model. Test models only judge pre-generated text, eliminating self-confirming loops.

**Three-tier data:** Gold (6 hand-audited scenarios with human-written transforms), probe (4–8 synthetic minimal pairs per test), generated (6–14 Dear Abby scenarios with LLM transforms per test).

**Statistical controls:** 3-rep control arms. All significance tests against empirical baselines. Wilson CIs. Fisher combination across models. Code is modular with structured JSON output, automated scoring, and separation of stimulus generation from judgment.

### Dataset

**Dear Abby (embedded):** 25 curated scenarios (1985–2017) covering family, workplace, friendship, professional, and community domains.

**AITA (HuggingFace, OsamaBsher):** 270,709 posts from Reddit r/AmITheAsshole with community verdicts and agreement scores.

**Gold transforms (embedded):** Hand-written gender-swaps, cultural reframes, euphemistic and dramatic rewrites, and narrative reorderings.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family design enables genuine architectural comparison.

**7 harm dimensions:** consequences, rights, fairness, autonomy, trust, social_impact, identity — each scored 0–10, total 0–70. Enforced via structured JSON output on all API calls.

**Budget:** ~$17 of $50 quota. ~1,800 API calls. Runtime: 73 minutes.

### Results, Insights, and Conclusions

| Model | T1: Fuzz | T2: BIP (Invariance) | T3: Holo | T4: Order | T5: Frame | Composite |
|---|---|---|---|---|---|---|
| Gemini 3 Flash | 0.600 | **0.958** | **0.667** | **1.000** | 0.631 | **0.734** |
| Claude Sonnet 4.6 | 0.400 | **0.958** | **0.667** | 0.933 | 0.630 | 0.697 |
| Gemini 2.0 Flash | **0.600** | 0.750 | 0.500 | 0.933 | **0.716** | 0.695 |
| Gemini 2.5 Pro | 0.500 | 0.708 | 0.583 | 0.967 | 0.606 | 0.643 |
| Gemini 2.5 Flash | 0.400 | 0.708 | 0.583 | 0.867 | 0.630 | 0.628 |

Composite scores range 0.628–0.734, providing clear discriminatory gradient across 5 models.

**1. Framing shifts perceived harm at 8.9σ (T5).** Euphemistic rewriting reduced harm scores by 10–16 points; dramatic rewriting increased by 6–11 points (0–70 scale). Controls drifted only 1–7 points. Fisher combination across 5 models × 2 framing types yields 8.9σ. Claude shows a unique asymmetric pattern: susceptible to euphemistic minimization (drift=−9.1) but resistant to dramatic exaggeration (drift=−1.5; harm scores actually *decreased* slightly under dramatic rewriting, opposite to all other models). This directional vulnerability would be invisible to any evaluation testing only one perturbation direction.

**2. Gender swap and evaluation order are NOT significant (T2/T4).** Against proper stochastic controls, these effects do not exceed baseline noise. Flash 3 and Claude both achieve 0.958 on T2. This is an important null result: the manifold *does* possess some symmetries. The failure is selective — models resist some perturbations and not others.

**3. Cultural reframe is a real effect (4.1σ).** While gender-swap is not significant, cultural reframing produces verdict changes above control noise — suggesting cultural context may carry genuine moral information rather than being a purely irrelevant surface feature.

**4. The selectivity pattern is the main insight.** Models are invariant under gender swap and evaluation order but NOT under euphemistic/dramatic framing. The perturbations that displace judgments are specifically those that manipulate *salience* — making morally irrelevant features perceptually prominent. This selective pattern is the diagnostic that a multi-dimensional approach reveals and a single robustness score would hide.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling.* SJSU.
2. Tversky, A. & Kahneman, D. (1981). "The Framing of Decisions." *Science* 211(4481).
3. Perez, E., et al. (2023). "Model-Written Evaluations." *ACL Findings.*
4. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
5. Fisher, R. A. (1925). *Statistical Methods for Research Workers.*
