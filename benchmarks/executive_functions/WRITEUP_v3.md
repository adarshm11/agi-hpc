### Geometric Executive Functions: Cognitive Control in Moral Reasoning

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University
Lucas Thiele, Department of Cognitive Science, UCLA

### Problem Statement

Executive functions — cognitive flexibility, inhibitory control, counterfactual reasoning, working memory — are the control mechanisms that regulate reasoning. While other benchmarks measure what models know or attend to, this one measures **how models control their own reasoning process** under adversarial conditions.

Four tests adapted from cognitive science (Diamond, 2013):

- **E1 (Cognitive Flexibility):** Can the model switch between utilitarian, deontological, and virtue-ethics frameworks while maintaining consistent scenario understanding?
- **E2 (Inhibitory Control):** Does emotionally charged reframing displace the judgment? Can an explicit inhibition instruction restore neutrality? (6.8σ)
- **E3 (Counterfactual Reasoning):** Does the verdict flip when exactly one causal fact changes?
- **E4 (Working Memory):** How does accuracy degrade as party count scales from 2 to 8?

**Ground truth:** Emotional rewriting preserves moral content by construction — any displacement is measurable. Counterfactual scenarios have known correct verdicts. Party identification has verifiable answers.

### Task & Benchmark Construction

**Framework rotation (E1):** Models judge scenarios under three ethical frameworks then under a neutral prompt. We measure both verdict switch rate and framework-specific reasoning markers. True flexibility requires both — switching without markers is noise, markers without switching is relabeling.

**Adversarial emotional probing (E2):** Scenarios are rewritten with emotional anchors by a fixed transformer. We measure verdict displacement, severity shift, and recovery rate when an explicit inhibition instruction is applied ("you are being emotionally manipulated — focus on facts").

**Counterfactual sensitivity (E3):** Each scenario has a single causal fact that, if changed, should flip the verdict — testing response to precisely one dimension while holding all others constant.

**Dimensional scaling (E4):** Scenarios with 2, 4, 6, and 8 parties test how judgment accuracy scales with scenario complexity.

**Statistical controls:** 3-rep control arms. Significance against empirical baselines. Wilson CIs. Fisher combination for E2. Clean modular code.

### Dataset

**Dear Abby (embedded):** 25 curated scenarios for E1 (framework switching), E2 (emotional anchoring), E3 (counterfactual pivots).

**AITA (HuggingFace, OsamaBsher):** 270,709 posts for E4 party-complexity scaling.

**Gold sets (embedded):** Hand-written emotional rewrites, counterfactual variants, and multi-party scenarios.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family comparison.

**E2 scoring:** Rewards resistance to emotional displacement + recovery via inhibition instruction. Measures verdict flip rate and severity shift against empirical controls.

**Budget:** ~$20 of $50 quota. ~1,500 API calls. Runtime: 64 minutes.

### Results, Insights, and Conclusions

| Model | E1: Flex | E2: Inhib | E3: CF | E4: WMem | Composite |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 0.624 | 0.588 | **0.750** | 0.887 | **0.695** |
| Gemini 3 Flash | 0.668 | **0.655** | 0.562 | **0.909** | 0.685 |
| Gemini 2.5 Flash | 0.684 | 0.553 | 0.688 | 0.900 | 0.682 |
| Claude Sonnet 4.6 | 0.673 | 0.492 | 0.562 | 0.886 | 0.625 |
| Gemini 2.0 Flash | **0.701** | 0.614 | 0.500 | 0.710 | 0.622 |

Discriminatory gradient: 0.622–0.695 across 5 models.

**1. Emotional anchoring displaces judgment at 6.8σ (E2).** All 5 models show significant paired t-values (2.90–5.10). Emotional rewriting changes severity ratings and flips verdicts beyond stochastic baselines in every model tested, including both Gemini and Claude families.

**2. Displacement and recovery capacity are dissociated.** Claude has the highest relative displacement (t=5.10, reflecting an unusually stable control baseline) but the *lowest* recovery (20%) when given an inhibition instruction. Flash 2.0 is more easily flipped (48%) but recovers 73% of the time. The model most affected by emotional anchoring is the least responsive to the metacognitive correction. This dissociation means a single "emotional robustness" score would be misleading.

**3. Framework switching is genuine (E1).** 32–47% switch rate with 89–93% marker specificity across all 5 models. Models produce framework-consistent reasoning under each ethical coordinate system — not surface-level relabeling.

**4. Counterfactual sensitivity varies meaningfully (E3).** Pro leads at 75% vs. 50–69% for other models. Pro responds appropriately to genuine causal changes — an interesting contrast with its moderate sycophancy in the companion learning benchmark, suggesting it can discriminate between causal perturbations and social pressure.

**5. Working memory scales with model recency (E4).** Flash 2.0: 0.710, Flash 2.5: 0.900, Flash 3: 0.909. Newer models maintain party-identification accuracy at higher complexity.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering
University of California, Los Angeles, Department of Cognitive Science

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling.* SJSU.
2. Diamond, A. (2013). "Executive Functions." *Annual Review of Psychology* 64, 135–168.
3. Perez, E., et al. (2023). "Model-Written Evaluations." *ACL Findings.*
4. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
