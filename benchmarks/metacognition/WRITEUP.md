### Geometric Metacognition: Calibration Surfaces and Strategy Scaling in Moral Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

Metacognition is the ability to monitor and regulate one's own cognitive processes -- knowing what you know, knowing what you don't, and allocating effort proportionally to difficulty. For LLMs, metacognition is typically measured via scalar confidence scores. But a model that reports 9/10 confidence on everything is metacognitively blind, even if it happens to be accurate 90% of the time.

The geometric approach (Bond, 2026a) treats confidence and accuracy as occupying a joint space. Calibration is not a scalar (ECE) but a **surface** -- the distance between the confidence manifold and the accuracy manifold across the space of inputs. A well-calibrated model has these manifolds close together everywhere; a miscalibrated one has them diverge in specific regions.

This benchmark maps four aspects of metacognitive geometry:

- **M1 (Calibration):** Expected Calibration Error measures the gap between confidence and accuracy surfaces. (Headline test; Fisher-combined across models yields 11.2 sigma for systematic miscalibration.)
- **M2 (Knowing What You Don't Know):** Can the model discriminate clear-cut from ambiguous cases? Uses rank-based AUROC alongside raw confidence gap to handle the confidence compression problem (models report 8-10 on everything).
- **M3 (Self-Monitoring):** Can the model identify which scenarios it is most uncertain about? Spearman correlation between self-reported uncertainty ranking and actual verdict variance.
- **M4 (Strategy Selection):** Does reasoning effort scale with complexity? **Critical fix in v3:** removed the "2-3 sentences MAX" prompt constraint that was directly contradicting the measurement.

### Task & Benchmark Construction

**Calibration surface mapping (M1):** Models judge AITA scenarios with confidence ratings. ECE is computed with 5 bins, with bootstrap standard error for significance. Overconfidence rate (fraction of high-confidence wrong answers) provides a directional measure. This maps the confidence-accuracy surface across the difficulty spectrum.

**Rank-based discrimination (M2):** The fundamental problem with M2 in v2 was that raw confidence gap (clear - ambiguous) was tiny (0.13-0.54 on a 0-10 scale) because models compress all confidence into the 8-10 range. The geometric fix: use AUROC, which measures rank-order discrimination. For every (clear, ambiguous) pair, does the clear scenario have higher confidence? AUROC captures the ordering structure even when absolute values are compressed.

**Strategy scaling (M4):** v2's prompt said "Provide your full analysis" then "Keep reasoning to 2-3 sentences MAX." This is a measurement artifact: the length cap prevents the very variation we're trying to detect. v3 removes the cap: "Explain your reasoning in as much detail as you feel is appropriate for the complexity of the situation." Result: length ratio jumped from 1.10x (null) to 1.61x (t=2.78, significant).

**Statistical controls:** 5-replication control arms on M1/M2. Bootstrap SE on ECE. All significance tests against empirical baselines. Fisher combination across models.

### Dataset

**AITA (HuggingFace):** 270,709 posts. M1 uses gold + AITA scenarios with known verdicts. M2 uses 15 clear-cut (NTA/YTA, high agreement) + 15 ambiguous (ESH/NAH) scenarios.

**Dear Abby (embedded):** 25 scenarios. M3 uses 12 scenarios x 5 reps for verdict variance measurement. M4 uses 8 simple + 8 complex scenarios (categorized by party count, length, and stake complexity).

**Gold calibration set (embedded):** Hand-audited scenarios with known verdicts, expected confidence ranges, and difficulty annotations (easy/hard).

### Technical Details

**Models:** 2 full-suite (Gemini 2.0 Flash, Gemini 2.5 Pro) + 2 M1-only (Gemini 2.5 Flash, Gemini 3 Flash Preview) + Claude Sonnet 4.6 cross-family (M1 gold probes). Budget-aware execution scales scenario counts for expensive models while preserving all 4 measures.

**M2 scoring:** Blends AUROC + raw gap + significance: `0.3 * gap/3 + 0.4 * (AUROC - 0.5) * 2 + 0.15 * (1 - noise) + sig_bonus`. AUROC above 0.5 (chance) is the primary driver, robust to confidence compression.

**M4 scoring:** `0.5 * ratio_scaled + 0.2 * conf_diff + sig_bonus + 0.15 base`. Rewards length ratio > 1 and lower confidence on complex scenarios. Significance bonus for t > 2.0.

**Budget:** ~$45 of $50 quota. ~838 API calls. Runtime: 12 minutes. Budget-aware tier selection gave Pro a reduced but complete plan (8 AITA, 0 ctrl reps, 4 M3 scenarios × 2 reps) to fit within per-model allocation.

### Results, Insights, and Conclusions

| Model | M1: Calib | M2: Discrim | M3: Monitor | M4: Strategy | Composite |
|---|---|---|---|---|---|
| Gemini 2.0 Flash | 0.611 | 0.195 | 0.094 | **0.723** | -- |
| Gemini 2.5 Pro | **0.807** | 0.168 | **0.700** | 0.350 | -- |
| Gemini 2.5 Flash (M1) | 0.610 | -- | -- | -- | -- |
| Gemini 3 Flash (M1) | 0.670 | -- | -- | -- | -- |
| Claude Sonnet 4.6 (M1) | ECE=0.25 | -- | -- | -- | -- |

1. **Systematic miscalibration at 9.3 sigma (M1).** ECE ranges from 0.23 (Pro) to 0.42 (Flash models). Fisher combination across 4 Gemini models yields 9.3 sigma (discovery-level). Both self-reported confidence and behavioral percent-agree metrics show the same pattern (8.6 sigma behavioral). Every model tested is miscalibrated in the same direction (overconfident), with cross-family validation on Claude (ECE=0.25) confirming the effect is not Gemini-specific.

2. **Calibration improves with model scale.** Pro (ECE=0.23) is significantly better calibrated than the Flash models (ECE=0.41-0.42). Flash 3 Preview (0.33) falls between. This is a scaling effect: larger models have confidence surfaces that more closely track their accuracy surfaces, even though all remain significantly miscalibrated.

3. **Self-monitoring diverges qualitatively across models (M3).** Flash 2.0 scores 0.094 (near chance); Pro scores 0.700. Pro can identify which scenarios it is most uncertain about; Flash cannot. This parallels the calibration scaling -- the same model that has better-calibrated confidence also has better metacognitive self-access.

4. **Strategy selection reverses across models (M4).** Flash 2.0 scores 0.723 (strong effort scaling); Pro scores 0.350 (weak). The larger model writes more uniformly across difficulty levels, while the smaller model adapts reasoning length to complexity. This dissociation -- Pro is better calibrated but less effort-adaptive -- indicates these are independent metacognitive faculties, not a single axis.

5. **Overconfidence is universal.** High-confidence (>8) wrong rates are significant across all models. Models collapse into the high-confidence region of the space regardless of actual accuracy.

### Cross-Track Convergence: Miscalibration as the Root of Geometric Vulnerability

The 9.3 sigma miscalibration finding is not merely a confidence-accuracy gap -- it is the *mechanism* that enables the displacement vulnerabilities measured across the four companion benchmarks:

- **Learning (L2):** A model that doesn't know when it's right has no basis for rejecting a wrong correction. The sycophancy gradient (Flash 2.0: 33% wrong flip, Flash 2.5: 56%) mirrors the calibration gradient (Flash 2.0: ECE 0.41, Flash 2.5: ECE 0.42). Both Flash models are equally miscalibrated, yet Flash 2.0 is less sycophantic -- suggesting that calibration is necessary but not sufficient for correction discrimination, and that model stability (Flash 2.0's 2% control flip rate) is an independent protective factor.
- **Executive Functions (E2):** Emotional anchoring displaces judgments at 6.8 sigma. Recovery via inhibition instruction averages only 38%. If models had accurate metacognitive access to their own certainty (well-calibrated M1), they would recognize when an emotional perturbation has moved them away from a well-supported position. The 20% recovery rate for Claude -- the model with the best calibration (ECE=0.25) but strongest anchoring displacement (t=5.10) -- suggests that calibration and inhibitory control operate in different regions of the cognitive architecture.
- **Social Cognition (T5):** Framing shifts perceived harm at 8.9 sigma. A well-calibrated model would assign lower confidence to judgments that shift dramatically under paraphrase. Instead, models maintain high confidence regardless of framing condition -- the confidence surface is flat where it should vary.
- **Attention (A1):** Vivid distractors shift judgment at 4.6 sigma. The metacognitive system should detect that morally irrelevant information is affecting the evaluation and correct for it. The 39% recovery rate when warned explicitly suggests the detection mechanism exists but is not engaged by default.

The geometric interpretation: miscalibration means the confidence dimension of the judgment space has collapsed -- it carries no information about reliability. This leaves models unable to distinguish perturbations that change moral content (which should move the judgment vector) from perturbations that change only surface presentation (which should not). The five tracks together map different consequences of this single structural deficit.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling: From Manifolds to Production Systems.* SJSU. (Ch. 1: Scalar Irrecoverability; Ch. 9: Robustness Surfaces)
2. Bond, A. H. (2026b). *Geometric Ethics: Multi-dimensional Evaluation in the Moral Domain.* Working paper.
3. Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *ICML.*
4. Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." *arXiv:2207.05221.*
5. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace Datasets.
6. Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA* 22(158).
