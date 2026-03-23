### Geometric Executive Functions: Cognitive Control on the Moral Judgment Manifold

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University
Lucas Thiele, Department of Cognitive Science, UCLA

### Problem Statement

Executive functions -- cognitive flexibility, inhibitory control, counterfactual reasoning, working memory -- are the control mechanisms that operate on the judgment manifold. While the other tracks in this series measure what models know (calibration), how they learn (correction integration), and what they attend to (distractor resistance), the executive functions track measures **how models control their own reasoning process.**

The geometric framework (Bond, 2026a) provides a natural language for this: cognitive flexibility is the ability to rotate the judgment vector under a new coordinate system (ethical framework). Inhibitory control is resistance to adversarial displacement along the emotional axis. Counterfactual reasoning is sensitivity to single-dimension perturbations. Working memory is maintaining judgment accuracy as the dimensionality of the scenario increases.

Four executive function tests:

- **E1 (Cognitive Flexibility):** Framework switching -- can the model rotate its judgment vector between utilitarian, deontological, and virtue-ethics coordinate systems while maintaining consistent scenario understanding?
- **E2 (Inhibitory Control):** Emotional anchoring resistance -- does emotionally charged reframing displace the judgment vector? Can explicit inhibition instructions restore neutrality? (Headline test, 6.8 sigma)
- **E3 (Counterfactual Reasoning):** Single-cause pivots -- does the model's verdict flip when exactly one causal fact changes?
- **E4 (Working Memory):** Party tracking as complexity scales from 2 to 8 parties -- how does judgment accuracy degrade with dimensionality?

### Task & Benchmark Construction

**Framework rotation (E1):** Models judge scenarios under three ethical frameworks (utilitarian, deontological, virtue ethics), then under a neutral prompt. We measure: (a) whether framework changes verdict (switch rate), and (b) whether reasoning contains framework-specific markers (marker specificity). True cognitive flexibility requires both -- switching without markers is noise, markers without switching is relabeling.

**Adversarial emotional probing (E2):** Following the parametric transform framework (Bond, 2026a, Ch. 10.2), scenarios are rewritten with emotional anchors (dramatic language, vivid consequences) by a fixed transformer model. The test measures verdict displacement, severity shift, and recovery rate when an explicit inhibition instruction ("you are being emotionally manipulated -- focus on facts") is applied.

**Counterfactual sensitivity (E3):** Each scenario has a single causal fact that, if changed, should flip the verdict. This tests sensitivity along a single axis in the judgment space -- the model should respond to precisely one dimension while holding all others constant.

**Dimensional scaling (E4):** Scenarios with 2, 4, 6, and 8 parties test how judgment accuracy scales with the dimensionality of the moral scenario. This probes the effective dimensionality of the model's working memory for moral reasoning.

**Statistical controls:** 3-rep control arms. Significance tests against empirical stochastic baselines. Wilson CIs. Fisher combination for E2.

### Dataset

**Dear Abby (embedded):** 25 curated scenarios for E1 (framework switching), E2 (emotional anchoring), E3 (counterfactual pivots).

**AITA (HuggingFace):** 270,709 posts for supplementary scenarios and E4 party-complexity scaling.

**Gold sets (embedded):** Hand-written emotional rewrites, counterfactual variants, and multi-party scenarios with known difficulty gradients.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family design enables genuine comparison across model architectures.

**E2 scoring:** Rewards resistance to emotional displacement + recovery via inhibition instruction. Measures both verdict flip rate and severity shift (0-70 harm scale) against empirical controls.

**Budget:** ~$20 of $50 quota. ~1,500 API calls. Runtime: 64 minutes.

### Results, Insights, and Conclusions

| Model | E1: Flex | E2: Inhib | E3: CF | E4: WMem | Composite |
|---|---|---|---|---|---|
| gemini-2.5-pro | 0.624 | 0.588 | **0.750** | 0.887 | **0.695** |
| gemini-3-flash-preview | 0.668 | **0.655** | 0.562 | **0.909** | 0.685 |
| gemini-2.5-flash | 0.684 | 0.553 | 0.688 | 0.900 | 0.682 |
| claude-sonnet-4-6 | 0.673 | 0.492 | 0.562 | 0.886 | 0.625 |
| gemini-2.0-flash | **0.701** | 0.614 | 0.500 | 0.710 | 0.622 |

1. **Emotional anchoring is consistent across all 5 models (E2).** Emotional rewriting displaces the judgment vector beyond stochastic baselines in every model tested, with Fisher-combined significance of 6.8 sigma (discovery-level). All five models show significant paired t-values (2.90--5.10), confirming the effect generalizes across both Gemini and Claude architectures. The displacement pattern parallels the sycophancy effect in the companion learning benchmark (L2) and framing sensitivity in social cognition (T5), suggesting a shared geometric vulnerability to surface-level perturbations across cognitive domains.

2. **Claude is most displaced but least able to recover (E2).** Claude shows the highest paired t-statistic (t=5.10, 4.1σ) and largest severity shift (MAD=8.91) but recovers only 20% of the time when given an explicit inhibition instruction. By contrast, Gemini 2.0 Flash is more easily flipped (48% vs 43%) but recovers 73% of the time. This dissociation between displacement magnitude and recovery capacity is a novel finding: the model most affected by emotional anchoring is also the least responsive to metacognitive intervention.

3. **Framework switching is genuine across all models (E1).** 32--47% switch rate with 89--93% marker specificity. Models produce framework-consistent reasoning under each ethical coordinate system. True cognitive flexibility, not surface-level relabeling.

4. **Pro leads on counterfactual reasoning (E3).** 75% CF flip rate vs 50--69% for other models. Pro is the most responsive to single-cause causal pivots -- an interesting contrast with its moderate sycophancy in L2, suggesting it discriminates between causal perturbations (appropriate sensitivity) and social pressure (inappropriate sensitivity).

5. **Working memory scales with model recency (E4).** Flash 2.0: 0.710, Flash 2.5: 0.900, Flash 3: 0.909, Pro: 0.887, Claude: 0.886. Newer models maintain party-identification accuracy at higher complexity tiers. Average recovery rate across models: 38%.

### Cross-Track Convergence: Executive Control as the Missing Regulatory Layer

The executive functions track reveals *why* the displacement vulnerabilities measured in the companion benchmarks are so persistent: models lack the regulatory mechanisms that would correct for them.

In cognitive science (Diamond, 2013), executive functions are the control processes that monitor and override automatic responses. The four measures in this track map directly onto the failures observed across the series:

- **Inhibitory control (E2) explains why framing works (T5).** Emotional anchoring and euphemistic/dramatic framing are structurally identical perturbations -- surface-level rewording that changes tone but not moral content. Social cognition (T5) measures the displacement at 8.9 sigma; this track measures the model's ability to *resist* that displacement at 6.8 sigma. The 38% recovery rate represents the current ceiling of prompt-level inhibitory intervention. Claude's profile is particularly revealing: it has the strongest displacement (t=5.10) but the worst recovery (20%), and yet it shows zero sycophancy in L2. This means Claude's inhibitory control is *selective* -- it can resist social pressure but not emotional manipulation -- a pattern invisible to any single-track evaluation.
- **Counterfactual reasoning (E3) contrasts with sycophancy (L2).** Pro leads on counterfactual sensitivity (75% CF flip rate) but shows moderate sycophancy (wrong flip 44%). It responds appropriately to genuine causal changes and inappropriately to social pressure. This dissociation proves that the models possess functional discrimination mechanisms -- they are not uniformly brittle. The failure is specific: they cannot distinguish *which type* of perturbation warrants a response.
- **Working memory (E4) parallels divided attention (A4).** Both measure cognitive capacity under load, and both show the same scaling pattern: newer/larger models handle complexity better (Flash 2.0: E4=0.710, A4=0.812; Pro: E4=0.887, A4=1.000). These are likely the same underlying capacity measured in different frames.
- **Framework switching (E1) is the positive case.** 42% switch rate with 92% marker specificity proves that models *can* genuinely rotate their judgment vector under a new coordinate system. The cognitive machinery for flexible reasoning exists. The challenge identified across all five tracks is that this machinery is not reliably engaged when the perturbation is adversarial rather than instructed.

The unified picture: models have executive control capabilities (flexibility, some inhibition, counterfactual sensitivity, working memory) but these capabilities are inconsistently applied. Miscalibrated confidence (M1: 9.3 sigma) means the metacognitive monitoring system cannot reliably detect when executive intervention is needed. The result is a cognitive architecture with functional but poorly regulated control mechanisms -- precisely the profile that cognitive science predicts would produce the pattern of selective vulnerability we observe.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling: From Manifolds to Production Systems.* SJSU. (Ch. 9: Model Robustness Index; Ch. 10: Adversarial Probing; Ch. 10.2: Parametric Transforms)
2. Bond, A. H. (2026b). *Geometric Ethics: Multi-dimensional Evaluation in the Moral Domain.* Working paper.
3. Diamond, A. (2013). "Executive Functions." *Annual Review of Psychology* 64, 135-168.
4. Perez, E., et al. (2023). "Discovering Language Model Behaviors with Model-Written Evaluations." *ACL Findings.*
5. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace Datasets.
6. Fisher, R. A. (1925). *Statistical Methods for Research Workers.* Oliver and Boyd.
