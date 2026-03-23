### Geometric Attention: Distractor Dose-Response and Dimensional Signal-to-Noise in Moral Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

When an LLM judges a moral scenario, its output is a point in a multi-dimensional judgment space (verdict, confidence, and scores across moral dimensions). Standard attention benchmarks test whether models can "pay attention" to relevant information -- a binary pass/fail. The geometric approach (Bond, 2026a) asks a richer question: *how does the judgment vector move when we introduce irrelevant information at different intensities?*

This benchmark maps the **attention robustness surface** -- the function from perturbation intensity to judgment displacement -- using four complementary probes:

- **A1 (Distractor Resistance):** Graded dose-response: vivid vs. mild distractors measure proportional sensitivity, not just binary resistance. Can an explicit warning instruction restore the neutral position? (Headline test, operationalizing parametric transforms from Ch. 10.2)
- **A2 (Length Robustness):** Does neutral padding (2x, 4x length) displace the judgment vector? Tests whether the model's position is stable under content-preserving expansion.
- **A3 (Selective Attention):** Signal-to-noise ratio across 7 moral dimensions with ground-truth relevance labels. Tests whether the model's dimensional scoring reflects the geometric structure of the scenario.
- **A4 (Divided Attention):** Does interleaving two scenarios in one prompt displace verdicts vs. individual evaluation? Tests judgment independence under cognitive load.

The key insight from the Scalar Irrecoverability Theorem (Ch. 1.1): a single "attention score" cannot capture the distinction between a model that is uniformly distracted across all dimensions and one that is selectively vulnerable along specific axes. The geometric approach preserves this structure.

### Task & Benchmark Construction

**Graded adversarial probing (A1):** Following Ch. 10.2 (parametric transforms), we apply distractors at two intensity levels -- vivid (dramatic sensory details: weather, food, smells, textures) and mild (mundane contextual details: time of day, room temperature). The dose-response curve maps the robustness surface: good models show vivid_flip > mild_flip > control. A warned condition (explicit instruction to ignore distractors) tests recoverability -- the model's ability to return to its intensity-zero position.

**Separation of concerns:** A fixed transformer model (Gemini 2.0 Flash) generates all distracted and padded versions. Test models only judge pre-generated text, eliminating the self-confirming loop.

**Dimensional signal-to-noise (A3):** Models score scenarios across 7 moral dimensions (physical, emotional, financial, autonomy, trust, social_impact, identity) with hand-labeled relevance ground truth. SNR = mean(relevant scores) / mean(irrelevant scores). This directly measures whether the model's attention weights align with the geometric structure of the moral scenario.

**Statistical controls:** 5-replication control arms. All significance tests against empirical stochastic baselines. Wilson CIs. Fisher combination across models.

### Dataset

**Dear Abby (embedded):** 25 curated advice column scenarios. Used for A1 (gold + generated tiers), A2 (6 scenarios at 1x/2x/4x length), A3 (6 scenarios with dimension relevance labels), A4 (8 scenario pairs).

**Gold distractors (embedded):** 6 hand-written distractor scenarios with vivid irrelevant sensory detail woven into moral scenarios while preserving all moral facts.

**AITA (HuggingFace):** 270,709 posts. Used for supplementary scenarios.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family design enables genuine comparison across architectures.

**A1 scoring:** `0.35 * resistance + 0.25 * recovery + 0.15 * (1 - severity_shift) + 0.15 * (1 - mild_flip) + graded_bonus`. Rewards proportional dose-response with bonus for correct ordering (vivid > mild > control).

**A4 scoring:** Relative to control: `1.0 - min(excess_flip * 3, 1)`. Penalizes only flips above stochastic noise floor, not raw flip rate.

**Budget:** ~$17 of $50 quota. ~1,755 API calls. Runtime: 43 minutes.

### Results, Insights, and Conclusions

| Model | A1: Distract | A2: Length | A3: Select | A4: Divide | Composite |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 0.669 | **0.852** | **0.687** | **1.000** | **0.776** |
| Gemini 3 Flash | 0.678 | 0.714 | 0.667 | **1.000** | 0.747 |
| Gemini 2.5 Flash | **0.720** | 0.786 | 0.644 | 0.875 | 0.745 |
| Claude Sonnet 4.6 | 0.646 | 0.829 | 0.692 | 0.571 | 0.679 |
| Gemini 2.0 Flash | 0.581 | 0.667 | 0.669 | 0.812 | 0.666 |

1. **Vivid distractors shift judgment at 4.6 sigma (A1).** Fisher combination across 5 models yields 4.6 sigma. Vivid sensory distractors (weather, food, smells) shift severity ratings and flip verdicts beyond stochastic control in every model tested. The effect is consistent in direction across both Gemini and Claude families.

2. **Models partially recover via warning (39%).** When the distractor displaces the verdict, an explicit metacognitive instruction restores the neutral position 39% of the time. This parallels the 38% inhibition recovery rate in the companion executive functions benchmark (E2), suggesting a shared ceiling on prompt-level metacognitive intervention.

3. **Divided attention reveals an architectural split.** Pro and Flash 3 show perfect divided attention (1.000 -- zero interference from interleaving), while Claude shows the worst (0.571). This dissociates from Claude's other strengths (zero sycophancy in L2, best calibration in M1) and parallels the working memory scaling observed in executive functions (E4), where newer Gemini models handle multi-party scenarios better.

4. **Selective attention SNR is uniformly weak (1.22-1.38).** No model strongly distinguishes relevant from irrelevant moral dimensions. The geometric structure of the scenario is not well-reflected in dimensional attention allocation.

5. **Length robustness scales with model capability.** Pro (0.852) and Claude (0.829) are most robust to neutral padding; Flash 2.0 (0.667) is least. Larger models maintain verdict stability when morally irrelevant content inflates prompt length.

### Cross-Track Convergence: Distractor Interference as Attentional Gauge Failure

The distractor displacement measured in A1 is the attentional manifestation of a gauge invariance failure that appears in every track of this series. The five perturbation types -- social pressure (L2), emotional tone (E2), linguistic framing (T5), irrelevant sensory detail (A1), and miscalibrated self-knowledge (M1) -- all violate the same principle: morally equivalent inputs should produce identical outputs regardless of surface presentation.

What the attention track uniquely reveals is the **dose-response structure** of this failure:

- **Graded sensitivity exists.** Models that show vivid > mild > control distractor flip rates have a proportional attention filter -- they are not simply unstable. This mirrors the graded belief revision in the companion learning benchmark (L4), where extreme > moderate > irrelevant evidence produces proportional verdict change. Both findings indicate that models possess functional discrimination mechanisms; the failure is that these mechanisms are insufficiently selective.
- **Recovery rates converge across perturbation types.** Warning recovery in A1 (39%) and inhibition recovery in E2 (38%) are strikingly similar, despite the perturbation types being qualitatively different (irrelevant sensory detail vs. emotional framing). This suggests a shared metacognitive intervention ceiling -- the same mechanism mediates both types of recovery, and it succeeds approximately one-third of the time.
- **Divided attention dissociates from other vulnerabilities.** Claude has zero sycophancy (L2), best calibration (M1), and strong emotional anchoring displacement (E2) -- but the *worst* divided attention (A4: 0.571). Pro has moderate sycophancy but *perfect* divided attention. These are independent cognitive faculties, not a single robustness axis. The Scalar Irrecoverability Theorem predicts exactly this: a single robustness score would average over these orthogonal dimensions, destroying the information that distinguishes model cognitive profiles.

The geometric interpretation: the attention filter operates in a space where moral relevance and perceptual salience are not orthogonal. Vivid sensory details are salient but morally irrelevant; the model's attention mechanism does not fully separate these dimensions. The same geometric entanglement produces framing sensitivity (T5: linguistic salience entangled with moral content), emotional anchoring (E2: emotional salience entangled with moral content), and sycophancy (L2: social salience entangled with evidential validity).

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling: From Manifolds to Production Systems.* SJSU. (Ch. 9: Model Robustness Index; Ch. 10: Adversarial Probing)
2. Bond, A. H. (2026b). *Geometric Ethics: Multi-dimensional Evaluation in the Moral Domain.* Working paper.
3. Perez, E., et al. (2023). "Discovering Language Model Behaviors with Model-Written Evaluations." *ACL Findings.*
4. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace Datasets.
5. Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *JASA* 22(158).
6. Fisher, R. A. (1925). *Statistical Methods for Research Workers.* Oliver and Boyd.
