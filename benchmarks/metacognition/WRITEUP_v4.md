### Geometric Metacognition: Do Models Know When They're Wrong?

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

A model that reports 9/10 confidence on everything is metacognitively blind, even if it is accurate 90% of the time. Metacognition — knowing what you know and what you don't — requires that confidence *tracks* accuracy, not just sits near it on average. Current evaluations rarely test whether models can judge their own reliability, detect errors, or adjust strategy when failing. This benchmark isolates the **metacognition** faculty from Google DeepMind's cognitive framework [1] across four separable sub-capabilities:

- **M1 (Calibration):** Does stated confidence match observed accuracy? (Headline: 9.3σ systematic miscalibration)
- **M2 (Ambiguity Discrimination):** Can the model tell clear-cut cases from genuinely ambiguous ones?
- **M3 (Self-Monitoring):** Can the model identify which scenarios it is most uncertain about?
- **M4 (Strategy Selection):** Does reasoning effort scale with problem complexity?

**Ground truth is verifiable.** ECE compares stated confidence against observed accuracy — a measurement, not a subjective judgment. Ambiguity discrimination uses scenarios with known difficulty levels (high-agreement NTA/YTA vs. split ESH/NAH verdicts).

**The key design insight:** M1–M4 test *separable* metacognitive capabilities. By measuring all four, we can identify dissociations — a model might be well-calibrated (M1) but not effort-adaptive (M4), or vice versa. This profile structure is invisible to any single metacognitive metric.

### Task & Benchmark Construction

**Calibration surface mapping (M1).** Models judge AITA scenarios with confidence ratings (0–10). Expected Calibration Error (ECE) is computed with 5 bins and bootstrap standard error (200 iterations). Both self-reported confidence and behavioral confidence (percentage of control replications agreeing with the verdict) are measured, providing independent calibration estimates.

**Rank-based discrimination (M2).** Raw confidence gaps between clear and ambiguous cases are tiny (0.13–0.54 on a 0–10 scale) because models compress confidence into 8–10. The fix: AUROC measures *rank-order* discrimination. For every (clear, ambiguous) pair, does the clear scenario have higher confidence? This captures ordering even when absolute values are compressed — a methodological contribution applicable to any calibration benchmark.

**Self-monitoring (M3).** Models evaluate 12 scenarios × 5 replications. A final ranking query asks the model to identify its least-certain scenarios. Spearman correlation between self-reported uncertainty ranking and actual verdict variance across replications measures metacognitive access.

**Strategy scaling (M4).** Simple (2-party, short) and complex (multi-party, long) scenarios are evaluated with unconstrained response length. The length ratio (complex/simple) measures whether reasoning effort adapts to difficulty.

**Statistical controls.** 5-replication control arms on M1/M2. Budget-aware execution scales scenario counts by model cost (Tier 0: 100 M1 scenarios for cheap models, Tier 6: 18 for expensive). Bootstrap SE on ECE. Fisher combination across models.

### Dataset

**AITA (HuggingFace, OsamaBsher/AITA-Reddit-Dataset) [4].** 270,709 posts. M1 uses gold (6) + probes (6) + AITA scenarios (up to 88 per model, budget-scaled). M2 uses 15 clear-cut (high agreement, score≥20) + 15 ambiguous (ESH/NAH) scenarios.

**Dear Abby (embedded).** 25 curated scenarios. M3: 12 scenarios × 5 reps. M4: 15 simple + 15 complex scenarios with hand-labeled complexity.

**Gold calibration set (embedded).** 6 hand-audited scenarios with known verdicts and difficulty annotations providing maximum interpretive confidence.

### Technical Details

**Models.** 4 Gemini models full-suite (2.0 Flash, 2.5 Flash, 2.5 Pro, 3 Flash Preview) on M1–M4. Claude Sonnet 4.6 on M1 gold probes for cross-family validation. All responses use schema-enforced structured output (confidence, verdict, reasoning, harm vector) via kaggle-benchmarks SDK.

**Composite scoring.** M1=35%, M2=25%, M3=20%, M4=20%. M1 weighted highest as the headline calibration test with ECE — the most established and interpretable metacognitive metric.

**Budget.** ~$50 of $50 quota. ~600 API calls. Runtime: ~7 minutes.

### Results, Insights, and Conclusions

**M1 Calibration (headline — per-model ECE gradient):**

| Model | ECE | Direction | z-score |
|---|---|---|---|
| Gemini 2.0 Flash | 0.421 | overconfident | 6.2σ |
| Gemini 2.5 Flash | 0.415 | overconfident | 7.0σ |
| Gemini 3 Flash | 0.333 | overconfident | 4.5σ |
| Gemini 2.5 Pro | 0.186 | overconfident | 2.2σ |
| Claude Sonnet 4.6 | 0.250 | overconfident | — |
| **Fisher combined** | | | **6.0σ** |

**What this benchmark reveals that prior evaluations cannot:**

**1. Systematic miscalibration is universal.** Every model tested — across both Gemini and Claude families — is overconfident in the same direction. ECE ranges from 0.19 (Pro) to 0.42 (Flash 2.0/2.5), providing a clear discriminatory gradient. Cross-family validation on Claude (ECE=0.25) confirms this is architectural, not family-specific.

**2. Calibration improves with model scale.** Pro (0.19) is significantly better than Flash variants (0.33–0.42). This gradient tracks model capability, suggesting calibration is a learnable property that scales — a finding with direct implications for deployment decisions.

**3. Self-monitoring and calibration are linked (M3).** Flash 2.0 scores 0.047; Pro scores 0.500. The better-calibrated model can also identify its own uncertain scenarios — suggesting these draw on a shared self-knowledge capacity.

**4. Strategy selection is an independent metacognitive axis (M4).** Flash 2.0 scores 0.715 (strong effort scaling); Pro scores 0.000 (uniform regardless of complexity). The larger model writes uniformly while the smaller adapts. This dissociation — Pro is better calibrated but *less* effort-adaptive — proves metacognition is not a single dimension. Per-capability profiles reveal structure that composite scores destroy.

**Cross-track insight:** Pro's poor strategy adaptation (M4=0.000) contrasts with its best counterfactual sensitivity (E3=0.750) in the Executive Functions benchmark — it adapts *what* it reasons about but not *how much* it reasons. This dissociation is invisible without cross-domain profiling.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Google DeepMind (2026). "Measuring Progress Toward AGI: A Cognitive Framework."
2. Bond, A. H. (2026). *Geometric Methods in Computational Modeling.* SJSU.
3. Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." *arXiv:2207.05221.*
4. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
5. Fisher, R. A. (1925). *Statistical Methods for Research Workers.*
