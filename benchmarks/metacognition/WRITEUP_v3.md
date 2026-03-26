### Geometric Metacognition: Do Models Know When They're Wrong?

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

A model that reports 9/10 confidence on everything is metacognitively blind, even if it happens to be accurate 90% of the time. Metacognition — knowing what you know and what you don't — requires that confidence *tracks* accuracy, not just sits near it on average.

This benchmark measures four aspects of metacognitive capability:

- **M1 (Calibration):** Does stated confidence match observed accuracy? (9.3σ systematic miscalibration)
- **M2 (Ambiguity Discrimination):** Can the model tell clear-cut cases from genuinely ambiguous ones? Uses rank-based AUROC to handle the confidence compression problem (models report 8–10 on everything).
- **M3 (Self-Monitoring):** Can the model identify which scenarios it is most uncertain about? Measured by correlation between self-reported uncertainty and actual verdict variance.
- **M4 (Strategy Selection):** Does reasoning effort scale with problem complexity?

**Ground truth is verifiable:** ECE compares stated confidence against observed accuracy. Ambiguity discrimination uses scenarios with known difficulty levels (high-agreement NTA/YTA vs. ESH/NAH splits). These are measurements, not subjective judgments.

### Task & Benchmark Construction

**Calibration surface mapping (M1):** Models judge AITA scenarios with confidence ratings. ECE computed with 5 bins and bootstrap standard error. Overconfidence rate (fraction of high-confidence wrong answers) provides the directional measure.

**Rank-based discrimination (M2):** Raw confidence gap between clear and ambiguous cases is tiny (0.13–0.54 on 0–10 scale) because models compress confidence into 8–10. The fix: AUROC measures *rank-order* discrimination. For every (clear, ambiguous) pair, does the clear scenario have higher confidence? This captures ordering even when absolute values are compressed.

**Strategy scaling (M4):** Models evaluate simple and complex scenarios with unconstrained response length. Complexity scaling is measured by the length ratio (complex/simple responses).

**Statistical controls:** 3-replication control arms on M1/M2, budget-scaled (some models receive fewer). Bootstrap SE on ECE. Fisher combination across models. Structured JSON output with automated scoring.

### Dataset

**AITA (HuggingFace, OsamaBsher):** 270,709 posts. M1 uses gold + AITA scenarios with known verdicts. M2 uses 15 clear-cut (high agreement) + 15 ambiguous (ESH/NAH) scenarios.

**Dear Abby (embedded):** 25 scenarios. M3 uses 12 scenarios × 5 reps. M4 uses 8 simple + 8 complex scenarios.

**Gold calibration set (embedded):** Hand-audited scenarios with known verdicts and difficulty annotations.

### Technical Details

**Models:** 2 full-suite (Gemini 2.0 Flash, Gemini 2.5 Pro) on M1–M4. 2 additional models (Gemini 2.5 Flash, Gemini 3 Flash Preview) on M1 only. Claude Sonnet 4.6 on M1 gold probes for cross-family validation. Budget-aware execution scales scenario counts for expensive models while preserving all 4 measures.

**M2 scoring:** `0.4 × discrim_self + 0.6 × discrim_pct`. The first term measures the model's confidence gap between clear-cut and ambiguous scenarios; the second measures behavioral discrimination via the `percent_agree` proxy.

**Budget:** ~$45 of $50 quota. ~838 API calls. Runtime: 12 minutes.

### Results, Insights, and Conclusions

**M1 Calibration (headline, all 5 models):**

| Model | ECE | Direction | z-score |
|---|---|---|---|
| Gemini 2.0 Flash | 0.414 | overconfident | 5.8σ |
| Gemini 2.5 Flash | 0.415 | overconfident | 7.0σ |
| Gemini 3 Flash | 0.333 | overconfident | 4.5σ |
| Gemini 2.5 Pro | 0.230 | overconfident | 2.5σ |
| Claude Sonnet 4.6 | 0.250 | overconfident | — |
| **Fisher combined** | | | **9.3σ** |

**Full-suite results (2 models):**

| Model | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| Gemini 2.0 Flash | 0.611 | 0.195 | 0.094 | **0.723** |
| Gemini 2.5 Pro | **0.807** | 0.168 | **0.700** | 0.350 |

**1. Systematic miscalibration at 9.3σ (M1).** Every model tested is overconfident in the same direction. ECE ranges from 0.23 (Pro) to 0.42 (Flash). Cross-family validation on Claude (ECE=0.25) confirms this is not Gemini-specific.

**2. Calibration improves with model scale.** Pro (ECE=0.23) is significantly better than Flash 2.0 and Flash 2.5 (ECE=0.41–0.42). Flash 3 Preview (0.33) falls between. Larger models have confidence that more closely tracks accuracy, though all remain significantly miscalibrated.

**3. Self-monitoring and calibration track together (M3).** Flash 2.0 scores 0.094 (near chance); Pro scores 0.700. The better-calibrated model can also identify which scenarios it is most uncertain about — suggesting these draw on a shared self-knowledge capacity.

**4. Strategy selection is an independent dimension (M4).** Flash 2.0 scores 0.723 (strong effort scaling); Pro scores 0.350. The larger model writes uniformly regardless of difficulty, while the smaller model adapts reasoning length to complexity. This dissociation — Pro is better calibrated but *less* effort-adaptive — shows these are separate metacognitive capabilities, not a single axis.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling.* SJSU.
2. Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." *arXiv:2207.05221.*
3. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
4. Fisher, R. A. (1925). *Statistical Methods for Research Workers.*
