### Geometric Metacognition: Calibration Surfaces and Strategy Scaling in Moral Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

Metacognition — knowing what you know and allocating effort proportionally to difficulty — is typically measured via scalar confidence scores. But a model reporting 9/10 confidence on everything is metacognitively blind, even if accurate 90% of the time.

The geometric approach (Bond, 2026a) treats confidence and accuracy as occupying a joint space. Calibration is not a scalar (ECE) but a **surface** — the distance between the confidence manifold and the accuracy manifold across inputs. A well-calibrated model has these surfaces close everywhere; a miscalibrated one has them diverge in specific regions.

This benchmark maps four aspects of metacognitive geometry:

- **M1 (Calibration):** Expected Calibration Error — the gap between confidence and accuracy surfaces. (Headline: 9.3σ systematic miscalibration.)
- **M2 (Knowing What You Don't Know):** Discriminating clear-cut from ambiguous cases via rank-based AUROC, robust to the confidence compression problem (models report 8–10 on everything).
- **M3 (Self-Monitoring):** Spearman correlation between self-reported uncertainty ranking and actual verdict variance.
- **M4 (Strategy Selection):** Does reasoning effort scale with complexity?

**Ground truth is unambiguous:** ECE compares stated confidence against observed accuracy. Discrimination uses scenarios with known ambiguity levels. These are verifiable measurements, not subjective judgments.

### Task & Benchmark Construction

**Calibration surface mapping (M1):** Models judge AITA scenarios with confidence ratings. ECE computed with 5 bins and bootstrap standard error. Overconfidence rate (fraction of high-confidence wrong answers) provides directional measure.

**Rank-based discrimination (M2):** Raw confidence gap between clear and ambiguous cases is tiny (0.13–0.54 on 0–10 scale) because models compress confidence into the 8–10 range. The geometric fix: AUROC measures rank-order discrimination — does every clear scenario have higher confidence than every ambiguous one? This captures ordering structure even when absolute values are compressed.

**Strategy scaling (M4):** Models evaluate simple and complex scenarios with unconstrained response length: "Explain your reasoning in as much detail as you feel is appropriate." Complexity scaling is measured by the length ratio (complex/simple).

**Statistical controls:** 5-replication control arms on M1/M2. Bootstrap SE on ECE. Fisher combination across models. Modular code with structured JSON output and automated scoring.

### Dataset

**AITA (HuggingFace, OsamaBsher):** 270,709 posts. M1 uses gold + AITA scenarios with known verdicts. M2 uses 15 clear-cut (NTA/YTA, high agreement) + 15 ambiguous (ESH/NAH) scenarios.

**Dear Abby (embedded):** 25 scenarios. M3 uses 12 scenarios × 5 reps for variance. M4 uses 8 simple + 8 complex scenarios (categorized by party count, length, stake complexity).

**Gold calibration set (embedded):** Hand-audited scenarios with known verdicts, expected confidence ranges, and difficulty annotations.

### Technical Details

**Models:** 2 full-suite (Gemini 2.0 Flash, Gemini 2.5 Pro) + 2 M1-only (Gemini 2.5 Flash, Gemini 3 Flash Preview) + Claude Sonnet 4.6 cross-family (M1 gold probes).

**M2 scoring:** `0.3 × gap/3 + 0.4 × (AUROC − 0.5) × 2 + 0.15 × (1 − noise) + sig_bonus`. AUROC above 0.5 (chance) is the primary driver.

**Budget:** ~$45 of $50 quota. ~838 API calls. Runtime: 12 minutes.

### Results, Insights, and Conclusions

| Model | M1: Calib | M2: Discrim | M3: Monitor | M4: Strategy | Composite |
|---|---|---|---|---|---|
| Gemini 2.0 Flash | 0.611 | 0.195 | 0.094 | **0.723** | — |
| Gemini 2.5 Pro | **0.807** | 0.168 | **0.700** | 0.350 | — |
| Gemini 2.5 Flash (M1) | 0.610 | — | — | — | — |
| Gemini 3 Flash (M1) | 0.670 | — | — | — | — |
| Claude Sonnet 4.6 (M1) | ECE=0.25 | — | — | — | — |

Clear discriminatory gradient across models and measures.

1. **Systematic miscalibration at 9.3σ (M1).** ECE ranges from 0.23 (Pro) to 0.42 (Flash). Fisher combination across 4 Gemini models yields 9.3 sigma. Cross-family validation on Claude (ECE=0.25) confirms the effect is not Gemini-specific. Every model is overconfident.

2. **Calibration improves with model scale.** Pro (ECE=0.23) is significantly better than Flash models (ECE=0.41–0.42). Flash 3 Preview (0.33) falls between. Larger models have confidence surfaces that more closely track accuracy, though all remain significantly miscalibrated.

3. **Self-monitoring diverges qualitatively (M3).** Flash 2.0 scores 0.094 (near chance); Pro scores 0.700. Pro can identify which scenarios it is most uncertain about; Flash cannot. This parallels the calibration scaling.

4. **Strategy selection and calibration are independent faculties (M4).** Flash 2.0 scores 0.723 (strong effort scaling); Pro scores 0.350. The larger model writes uniformly across difficulty levels, while the smaller model adapts reasoning length to complexity. Pro is better calibrated but *less* effort-adaptive — these are dissociable metacognitive dimensions, not a single axis.

5. **Overconfidence is universal.** High-confidence (>8) wrong rates are significant across all models.

### Cross-Track Convergence

The 9.3σ miscalibration is not merely a confidence-accuracy gap — it is the *mechanism* enabling displacement vulnerabilities across companion benchmarks. A model that doesn't know when it's right cannot reject wrong corrections (L2 sycophancy, 13.3σ), cannot detect emotional manipulation (E2 anchoring, 6.8σ), and cannot recognize when framing has shifted its position (T5 framing, 8.9σ). The confidence dimension has collapsed — it carries no reliability information, leaving models unable to distinguish content-changing perturbations from surface-level ones.

The M3/M4 dissociation proves this is not one failure but multiple independent metacognitive faculties. No single robustness score captures this structure — the Scalar Irrecoverability Theorem in action.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling.* SJSU. (Ch. 1, 9)
2. Kadavath, S., et al. (2022). "Language Models (Mostly) Know What They Know." *arXiv:2207.05221.*
3. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
4. Fisher, R. A. (1925). *Statistical Methods for Research Workers.*
