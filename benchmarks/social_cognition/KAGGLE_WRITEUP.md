### Moral Geometry: Five Geometric Tests of Social Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

This benchmark is the most direct application of the geometric evaluation framework (Bond, 2026a) to LLM cognition. The central thesis: moral judgment is inherently multi-dimensional, and collapsing it to a scalar (accuracy, agreement rate) destroys the geometric structure that reveals how models actually reason.

We treat each moral judgment as a point in a 7-dimensional harm space (physical, emotional, financial, autonomy, trust, social_impact, identity) and apply five geometric tests from the Structural Fuzzing pipeline:

- **T1 (Structural Fuzzing):** Sensitivity profile across all 7 moral dimensions -- which axes of the judgment space are most responsive to perturbation? (Operationalizes Ch. 9: Sensitivity Profiling)
- **T2 (Bond Invariance Principle):** Is the judgment vector invariant under morally irrelevant transforms (gender swap, cultural reframe)? Tests symmetry on the judgment manifold. (Operationalizes Ch. 7: manifold symmetry)
- **T3 (Holonomy):** Path-dependence -- does presenting the same facts in different narrative orders produce different judgment vectors? If the manifold has curvature, parallel transport around a loop produces holonomy. (Operationalizes manifold geometry from Ch. 3-4)
- **T4 (Order Sensitivity):** Does the order of dimensional evaluation (which moral dimension is scored first) affect final scores? Tests whether the model's traversal of the dimension space is path-independent.
- **T5 (Framing Sensitivity):** Euphemistic vs. dramatic rewriting -- how much does surface framing displace the judgment vector while holding moral content constant? (Headline test, 7.6 sigma. Operationalizes Ch. 10: Adversarial Probing)

The Scalar Irrecoverability Theorem (Ch. 1.1) motivates the entire design: a single accuracy score cannot distinguish a model that is invariant under gender swap but sensitive to framing from one with the opposite profile. Both might have the same scalar "bias score." The geometric approach preserves these independent dimensions of variation.

### Task & Benchmark Construction

**Structural fuzzing (T1):** Following the pipeline in Ch. 9-10, we systematically perturb each of the 7 moral dimensions independently and measure the model's response profile. The sensitivity vector reveals which dimensions the model treats as most morally salient.

**Invariance testing (T2):** Gender-swap and cultural-reframe transforms should not change the moral verdict. We measure verdict stability under these symmetry operations. Critically, we test against empirical stochastic controls, not null=0, exposing that previous >6 sigma invariance-violation claims were measurement artifacts.

**Framing as adversarial probing (T5):** Scenarios are rewritten in euphemistic and dramatic registers by a fixed transformer model. The model under test only judges pre-generated text. The displacement from neutral position maps the framing sensitivity surface.

**Three-tier data:** Gold (6 hand-audited scenarios with human-written transforms), probe (6 synthetic minimal pairs), generated (9 Dear Abby with LLM transforms).

**Statistical controls:** 3-rep control arms. All significance tests against empirical baselines. Wilson CIs. Fisher combination across 4 models for T5.

### Dataset

**Dear Abby (embedded):** 25 curated scenarios (1985-2017) covering family, workplace, friendship, professional, and community domains.

**AITA (HuggingFace):** 270,709 posts from Reddit r/AmITheAsshole with community verdicts and agreement scores.

**Gold transforms (embedded):** Hand-written gender-swaps, cultural reframes, euphemistic and dramatic rewrites, and narrative reorderings.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family design enables genuine comparison.

**T5 scoring:** Based on framing resistance: severity shift under euphemistic/dramatic rewriting vs. stochastic control, with Fisher combination across models.

**7 harm dimensions:** physical, emotional, financial, autonomy, trust, social_impact, identity. Each scored 0-10, total harm 0-70. This is the coordinate system for the judgment manifold throughout all five benchmark tracks.

**Budget:** ~$17 of $50 quota. ~1,800 API calls. Runtime: 73 minutes.

### Results, Insights, and Conclusions

| Model | T1: Fuzz | T2: BIP | T3: Holo | T4: Order | T5: Frame | Composite |
|---|---|---|---|---|---|---|
| Gemini 3 Flash | 0.600 | **0.958** | **0.667** | **1.000** | 0.631 | **0.734** |
| Claude Sonnet 4.6 | 0.400 | **0.958** | **0.667** | 0.933 | 0.630 | 0.697 |
| Gemini 2.0 Flash | **0.600** | 0.750 | 0.500 | 0.933 | **0.716** | 0.695 |
| Gemini 2.5 Pro | 0.500 | 0.708 | 0.583 | 0.967 | 0.606 | 0.643 |
| Gemini 2.5 Flash | 0.400 | 0.708 | 0.583 | 0.867 | 0.630 | 0.628 |

1. **Framing shifts perceived harm at 8.9 sigma (T5).** Euphemistic rewriting reduced harm scores by 10-16 points; dramatic rewriting increased by 6-11 points (0-70 scale). Controls drifted only 1-7 points. Fisher combination across 5 models × 2 framing types yields 8.9 sigma (discovery-level). Claude shows an asymmetric pattern: susceptible to euphemistic minimization (drift=-9.1) but resistant to dramatic exaggeration (drift=-1.5, where positive was expected). This asymmetry is unique across all models tested.

2. **Invariance and order effects are NOT significant (T2/T4).** When tested against proper stochastic controls, gender-swap and dimension-order effects do not exceed baseline noise. Flash 3 and Claude both achieve 0.958 on T2 (near-perfect invariance), while Flash 2.0 and Pro are weaker (0.708-0.750). This is a clean null result with a methodological implication: testing against null=0 rather than empirical controls inflates significance, a finding that proved consequential across all five tracks.

3. **Cultural reframe is a real effect (T2 combined: 4.1 sigma).** While gender-swap is not significant, cultural reframing produces verdict changes above control noise. This suggests cultural context is not a gauge transformation -- it may carry genuine moral information that the model appropriately responds to.

4. **Flash 3 and Claude are most geometrically stable.** Both score 0.958 on invariance (T2), 0.667 on path-dependence (T3), and 0.933+ on order sensitivity (T4). Their judgment manifolds have the most symmetric structure, even though their framing resistance (T5) is not the strongest.

### Cross-Track Convergence: The Judgment Manifold Under Five Perturbation Types

This social cognition benchmark is the geometric foundation for the four companion tracks. The 7-dimensional harm space defined here (physical, emotional, financial, autonomy, trust, social_impact, identity) serves as the coordinate system in which all five benchmarks measure displacement. The framing sensitivity finding (T5, 8.9 sigma) is the most direct test of gauge invariance: do morally equivalent inputs produce identical outputs when the surface presentation changes?

The five tracks together reveal that gauge invariance fails systematically, but *differently* for each perturbation type:

| Perturbation | Track | Sigma | What moves the vector |
|---|---|---|---|
| Linguistic register | T5 (Social Cognition) | 8.9σ | Euphemistic/dramatic tone |
| Social pressure | L2 (Learning) | 13.3σ | Someone says you're wrong |
| Emotional tone | E2 (Executive Functions) | 6.8σ | Dramatic emotional rewriting |
| Irrelevant detail | A1 (Attention) | 4.6σ | Vivid sensory distractors |
| Self-knowledge | M1 (Metacognition) | 9.3σ | Confidence doesn't track accuracy |

These are not five separate failures. They are five measurements of a single geometric property: **the model's representation does not separate moral content from surface presentation.** In the language of the geometric framework, moral and presentational features are entangled in the embedding space -- they are not orthogonal. This is directly testable via the linear probing methodology applied by Thiele (2026) to LaBSE representations, where moral and language signals *were* found to be largely orthogonal. The discrepancy suggests that while the *representation* may separate these dimensions, the *reasoning process* operating on that representation does not maintain the separation.

The invariance null results (T2/T4) are equally important: gender-swap and dimension-order *do not* displace the judgment vector beyond noise. This means the manifold possesses some symmetries -- the failure is selective, not total. The model's judgment space has the right symmetry structure for some gauge transformations (gender, evaluation order) but not others (framing, emotional tone, social pressure). Identifying which symmetries are present and which are broken is precisely the diagnostic that a geometric approach provides and a scalar benchmark cannot.

The Claude asymmetry in T5 -- susceptible to euphemistic minimization but resistant to dramatic exaggeration -- connects to its zero sycophancy in L2 and its poor inhibition recovery in E2. Claude appears to have a directional bias: it resists perturbations that *increase* perceived severity but not those that *decrease* it. This is a geometric asymmetry in the model's vulnerability surface that would be invisible to any evaluation that tests only one perturbation direction.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling: From Manifolds to Production Systems.* SJSU. (Ch. 1: Scalar Irrecoverability; Ch. 7: Equilibrium on Manifolds; Ch. 9: Model Robustness Index; Ch. 10: Adversarial Probing)
2. Bond, A. H. (2026b). *Geometric Ethics: Multi-dimensional Evaluation in the Moral Domain.* Working paper.
3. Tversky, A. & Kahneman, D. (1981). "The Framing of Decisions and the Psychology of Choice." *Science* 211(4481).
4. Perez, E., et al. (2023). "Discovering Language Model Behaviors with Model-Written Evaluations." *ACL Findings.*
5. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace Datasets.
6. Fisher, R. A. (1925). *Statistical Methods for Research Workers.* Oliver and Boyd.
