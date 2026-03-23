### Geometric Executive Functions: Cognitive Control on the Moral Judgment Manifold

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University
Lucas Thiele, Department of Cognitive Science, UCLA

### Problem Statement

Executive functions — cognitive flexibility, inhibitory control, counterfactual reasoning, working memory — are the control mechanisms that operate *on* the judgment manifold. While companion tracks measure what models know (calibration), how they learn (correction integration), and what they attend to (distractor resistance), this track measures **how models control their own reasoning process.**

The geometric framework (Bond, 2026a) provides natural language: cognitive flexibility is rotating the judgment vector under a new coordinate system. Inhibitory control is resistance to adversarial displacement. Counterfactual reasoning is sensitivity to single-dimension perturbations. Working memory is maintaining accuracy as scenario dimensionality increases.

Four executive function tests:
- **E1 (Cognitive Flexibility):** Framework switching — can the model rotate its judgment between utilitarian, deontological, and virtue-ethics coordinate systems?
- **E2 (Inhibitory Control):** Emotional anchoring resistance — does emotionally charged reframing displace the judgment vector? Can inhibition instructions restore neutrality? (Headline: 6.8σ)
- **E3 (Counterfactual Reasoning):** Single-cause pivots — does the verdict flip when exactly one causal fact changes?
- **E4 (Working Memory):** Party tracking as complexity scales from 2 to 8 parties.

**Ground truth is defined by invariance:** emotional rewriting preserves moral content by construction. Any displacement is an unambiguous, measurable failure. Counterfactual scenarios have known correct verdicts. Party identification has verifiable answers.

### Task & Benchmark Construction

**Framework rotation (E1):** Models judge scenarios under three ethical frameworks then under a neutral prompt. We measure: (a) whether framework changes verdict (switch rate) and (b) whether reasoning contains framework-specific markers (marker specificity). True flexibility requires both — switching without markers is noise, markers without switching is relabeling.

**Adversarial emotional probing (E2):** Following Ch. 10.2, scenarios are rewritten with emotional anchors by a fixed transformer. Tests verdict displacement, severity shift, and recovery rate when an explicit inhibition instruction is applied.

**Counterfactual sensitivity (E3):** Each scenario has a single causal fact that, if changed, should flip the verdict — testing sensitivity along one axis while holding all others constant.

**Dimensional scaling (E4):** Scenarios with 2, 4, 6, and 8 parties test how accuracy scales with moral scenario dimensionality.

**Statistical controls:** 3-rep control arms. Significance tests against empirical baselines. Wilson CIs. Fisher combination for E2. Clean modular code with structured JSON output.

### Dataset

**Dear Abby (embedded):** 25 curated scenarios for E1 (framework switching), E2 (emotional anchoring), E3 (counterfactual pivots).

**AITA (HuggingFace, OsamaBsher):** 270,709 posts for supplementary scenarios and E4 party-complexity scaling.

**Gold sets (embedded):** Hand-written emotional rewrites, counterfactual variants, and multi-party scenarios with known difficulty gradients.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family design enables genuine architectural comparison.

**E2 scoring:** Rewards resistance to emotional displacement + recovery via inhibition instruction. Measures both verdict flip rate and severity shift (0–70 harm scale) against empirical controls.

**Budget:** ~$20 of $50 quota. ~1,500 API calls. Runtime: 64 minutes.

### Results, Insights, and Conclusions

| Model | E1: Flex | E2: Inhib | E3: CF | E4: WMem | Composite |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 0.624 | 0.588 | **0.750** | 0.887 | **0.695** |
| Gemini 3 Flash | 0.668 | **0.655** | 0.562 | **0.909** | 0.685 |
| Gemini 2.5 Flash | 0.684 | 0.553 | 0.688 | 0.900 | 0.682 |
| Claude Sonnet 4.6 | 0.673 | 0.492 | 0.562 | 0.886 | 0.625 |
| Gemini 2.0 Flash | **0.701** | 0.614 | 0.500 | 0.710 | 0.622 |

Composite scores range 0.622–0.695, providing meaningful discriminatory gradient across 5 models.

1. **Emotional anchoring at 6.8σ (E2).** Fisher combination across 5 models. All show significant paired t-values (2.90–5.10), confirming the effect generalizes across both Gemini and Claude families.

2. **Claude is most displaced but least able to recover (E2).** Highest t-statistic (t=5.10, 4.1σ) and largest severity shift (MAD=8.91) but recovers only 20%. By contrast, Flash 2.0 is more easily flipped (48% vs 43%) but recovers 73%. This dissociation between displacement magnitude and recovery capacity is a novel finding: the model most affected by emotional anchoring is the least responsive to metacognitive intervention.

3. **Framework switching is genuine (E1).** 32–47% switch rate with 89–93% marker specificity across all models. True cognitive flexibility, not surface relabeling.

4. **Pro leads on counterfactual reasoning (E3).** 75% CF flip rate vs 50–69% for others. Interesting contrast with its moderate sycophancy in L2 — Pro discriminates between causal perturbations (appropriate sensitivity) and social pressure (inappropriate sensitivity).

5. **Working memory scales with model recency (E4).** Flash 2.0: 0.710, Flash 2.5: 0.900, Flash 3: 0.909, Pro: 0.887, Claude: 0.886. Newer models maintain party-identification accuracy at higher complexity.

### Cross-Track Convergence

The executive functions track reveals *why* displacement vulnerabilities persist: models lack regulatory mechanisms to correct for them.

In cognitive science (Diamond, 2013), executive functions monitor and override automatic responses. The four measures here map directly onto failures in companion tracks. Inhibitory control (E2) explains why framing works (T5, 8.9σ) — structurally identical perturbations, same 38% recovery ceiling. Claude's profile is the clearest illustration: zero sycophancy (L2), strongest emotional displacement (E2), worst divided attention (A4), best calibration (M1). These are independent cognitive faculties with distinct vulnerability surfaces — a pattern that the Scalar Irrecoverability Theorem predicts and that no single benchmark score can capture.

The E3/L2 contrast is equally revealing: Pro responds appropriately to causal changes (75% CF flip) and inappropriately to social pressure (44% wrong flip). The discrimination machinery exists; it is inconsistently applied.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering
University of California, Los Angeles, Department of Cognitive Science

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling.* SJSU. (Ch. 9, 10)
2. Diamond, A. (2013). "Executive Functions." *Annual Review of Psychology* 64, 135–168.
3. Perez, E., et al. (2023). "Model-Written Evaluations." *ACL Findings.*
4. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
5. Fisher, R. A. (1925). *Statistical Methods for Research Workers.*
