### Moral Geometry: Five Geometric Tests of Social Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

Standard LLM benchmarks collapse model behavior to a scalar — accuracy, agreement rate — and declare success if the number crosses a threshold. The Scalar Irrecoverability Theorem (Bond, 2026a, Ch. 1) proves this is structurally incomplete: for a model evaluated on *n* dimensions, any scalar summary destroys *n*−1 independent directions of variation.

This benchmark treats moral judgment as a point in a 7-dimensional harm space (physical, emotional, financial, autonomy, trust, social_impact, identity) and applies five geometric tests:

- **T1 (Structural Fuzzing):** Sensitivity profile across all 7 moral dimensions — which axes respond most to perturbation? (Ch. 9: Sensitivity Profiling)
- **T2 (Bond Invariance Principle):** Is the judgment vector invariant under morally irrelevant transforms (gender swap, cultural reframe)? (Ch. 7: Manifold Symmetry)
- **T3 (Holonomy):** Does presenting identical facts in different narrative orders produce different judgment vectors? Path-dependence on the manifold.
- **T4 (Order Sensitivity):** Does the order of dimensional evaluation affect final scores?
- **T5 (Framing Sensitivity):** Euphemistic vs. dramatic rewriting — how much does surface framing displace the judgment vector while holding moral content constant? (Headline: 8.9σ. Ch. 10: Adversarial Probing)

**Ground truth is defined by invariance:** each transform preserves moral content by construction. Any displacement is an unambiguous, measurable failure — not a matter of opinion.

### Task & Benchmark Construction

**Structural fuzzing (T1):** We systematically perturb each of 7 moral dimensions independently and measure the response profile, revealing which dimensions the model treats as most salient.

**Invariance testing (T2):** Gender-swap and cultural-reframe transforms should not change moral verdicts. We measure stability under these symmetry operations against empirical stochastic controls — not null=0 — a distinction that proved consequential across all five tracks.

**Framing as adversarial probing (T5):** Scenarios are rewritten in euphemistic and dramatic registers by a fixed transformer model. Test models only judge pre-generated text, eliminating self-confirming loops.

**Three-tier data:** Gold (6 hand-audited scenarios with human-written transforms), probe (6 synthetic minimal pairs), generated (9 Dear Abby with LLM transforms).

**Statistical controls:** 3-rep control arms. All significance tests against empirical baselines. Wilson CIs. Fisher combination across models. Code is modular with structured JSON output and automated scoring.

### Dataset

**Dear Abby (embedded):** 25 curated scenarios (1985–2017) covering family, workplace, friendship, professional, and community domains.

**AITA (HuggingFace):** 270,709 posts from Reddit r/AmITheAsshole with community verdicts and agreement scores (OsamaBsher dataset).

**Gold transforms (embedded):** Hand-written gender-swaps, cultural reframes, euphemistic and dramatic rewrites, and narrative reorderings.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family design enables genuine architectural comparison.

**7 harm dimensions:** physical, emotional, financial, autonomy, trust, social_impact, identity — each scored 0–10, total 0–70. This coordinate system is shared across all five benchmark tracks.

**Budget:** ~$17 of $50 quota. ~1,800 API calls. Runtime: 73 minutes.

### Results, Insights, and Conclusions

| Model | T1: Fuzz | T2: BIP | T3: Holo | T4: Order | T5: Frame | Composite |
|---|---|---|---|---|---|---|
| Gemini 3 Flash | 0.600 | **0.958** | **0.667** | **1.000** | 0.631 | **0.734** |
| Claude Sonnet 4.6 | 0.400 | **0.958** | **0.667** | 0.933 | 0.630 | 0.697 |
| Gemini 2.0 Flash | **0.600** | 0.750 | 0.500 | 0.933 | **0.716** | 0.695 |
| Gemini 2.5 Pro | 0.500 | 0.708 | 0.583 | 0.967 | 0.606 | 0.643 |
| Gemini 2.5 Flash | 0.400 | 0.708 | 0.583 | 0.867 | 0.630 | 0.628 |

Composite scores range 0.628–0.734, providing clear discriminatory gradient across 5 models.

1. **Framing shifts perceived harm at 8.9σ (T5).** Euphemistic rewriting reduced harm scores by 10–16 points; dramatic rewriting increased by 6–11 points (0–70 scale). Controls drifted only 1–7 points. Fisher combination across 5 models × 2 framing types yields 8.9 sigma. Claude shows a unique asymmetric pattern: susceptible to euphemistic minimization (drift=−9.1) but resistant to dramatic exaggeration (drift=−1.5). This directional vulnerability is invisible to any evaluation testing only one perturbation direction.

2. **Invariance and order effects are NOT significant (T2/T4).** Against proper stochastic controls, gender-swap and dimension-order effects do not exceed baseline noise. Flash 3 and Claude both achieve 0.958 on T2 (near-perfect invariance). This clean null result has a methodological implication: testing against null=0 rather than empirical controls inflates significance.

3. **Cultural reframe is a real effect (4.1σ).** While gender-swap is not significant, cultural reframing produces verdict changes above control noise — suggesting cultural context carries genuine moral information.

4. **Flash 3 and Claude are most geometrically stable.** Both score 0.958 on invariance, 0.667 on path-dependence, and 0.933+ on order sensitivity, despite differing framing resistance.

### Cross-Track Convergence

This social cognition benchmark is the geometric foundation for four companion tracks. The five tracks together reveal that gauge invariance fails systematically but *selectively*:

| Perturbation | Track | Sigma | What Moves the Vector |
|---|---|---|---|
| Linguistic register | T5 (Social Cognition) | 8.9σ | Euphemistic/dramatic tone |
| Social pressure | L2 (Learning) | 13.3σ | Someone says you're wrong |
| Emotional tone | E2 (Executive Functions) | 6.8σ | Dramatic emotional rewriting |
| Irrelevant detail | A1 (Attention) | 4.6σ | Vivid sensory distractors |
| Self-knowledge | M1 (Metacognition) | 9.3σ | Confidence doesn't track accuracy |

These are five measurements of one geometric property: **the model's representation does not separate moral content from surface presentation.** The invariance null results (T2/T4) are equally important: some symmetries are preserved (gender, evaluation order) while others are broken (framing, emotional tone). Identifying which symmetries hold is precisely the diagnostic a geometric approach provides and a scalar benchmark cannot.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling.* SJSU. (Ch. 1, 7, 9, 10)
2. Bond, A. H. (2026b). *Geometric Ethics: Multi-dimensional Evaluation in the Moral Domain.* Working paper.
3. Tversky, A. & Kahneman, D. (1981). "The Framing of Decisions." *Science* 211(4481).
4. Perez, E., et al. (2023). "Discovering Language Model Behaviors with Model-Written Evaluations." *ACL Findings.*
5. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace Datasets.
6. Fisher, R. A. (1925). *Statistical Methods for Research Workers.* Oliver and Boyd.
