### Geometric Executive Functions: Cognitive Control in Moral Reasoning

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University
Lucas Thiele, Department of Cognitive Science, UCLA

### Problem Statement

Executive functions — cognitive flexibility, inhibitory control, counterfactual reasoning, working memory — are the control mechanisms that regulate reasoning (Diamond, 2013 [3]). While other benchmarks measure what models know or attend to, this one measures **how models control their own reasoning process** under adversarial conditions. This is the **executive functions** faculty from Google DeepMind's cognitive framework [1].

A model might reason well in isolation but fail to switch frameworks, override emotional anchoring, respond to causal changes, or track multiple parties. These are separable failures requiring separate measurement:

- **E1 (Cognitive Flexibility):** Can the model switch between utilitarian, deontological, and virtue-ethics frameworks while maintaining consistent scenario understanding?
- **E2 (Inhibitory Control):** Does emotionally charged reframing displace the judgment? Can an explicit inhibition instruction restore neutrality? (Headline: 6.8σ)
- **E3 (Counterfactual Reasoning):** Does the verdict flip when exactly one causal fact changes?
- **E4 (Working Memory):** How does accuracy degrade as party count scales from 2 to 8?

**Ground truth:** Emotional rewriting preserves moral content by construction — any displacement is measurable. Counterfactual scenarios have known single-cause pivots. Party identification has verifiable answers.

### Task & Benchmark Construction

**Adversarial emotional probing (E2, headline).** Scenarios are rewritten with emotional anchors by a fixed transformer (Gemini 2.0 Flash). We measure: (1) verdict displacement via paired t-test, (2) severity shift in the 7-dimensional harm vector, (3) recovery rate when an explicit inhibition instruction is applied ("you are being emotionally manipulated — focus on facts"). Three-tier data: gold (hand-written emotional rewrites), probes (synthetic), generated (25 Dear Abby with transformer rewrites).

**Framework rotation (E1).** Models judge scenarios under utilitarian, deontological, and virtue-ethics prompts, then under a neutral baseline. We measure both verdict switch rate AND framework-specific reasoning markers (keyword detection for framework-consistent reasoning). True flexibility requires both — switching without markers is noise, markers without switching is relabeling.

**Counterfactual sensitivity (E3).** Each scenario has a single causal fact that, if changed, should flip the verdict. 16 generated counterfactuals test response to precisely one dimension while holding all others constant.

**Complexity scaling (E4).** Scenarios with 2, 4, 6, and 8 morally relevant parties test how party-identification accuracy scales with scenario complexity.

**Statistical controls.** 5-replication control arms. Significance against empirical baselines. Wilson CIs. Fisher combination for E2. Structured output via SDK.

### Dataset

**Dear Abby (embedded).** 50 curated scenarios for E1 (framework switching), E2 (emotional anchoring), E3 (counterfactual pivots).

**AITA (HuggingFace, OsamaBsher/AITA-Reddit-Dataset) [5].** 270,709 posts for E4 party-complexity scaling.

**Gold sets (embedded).** Hand-written emotional rewrites with annotated preserved/changed content. Counterfactual variants with single-cause pivots. Multi-party scenarios with verified party counts.

### Technical Details

**Models.** 5 full-suite (Gemini 2.0 Flash, 2.5 Flash, 3 Flash Preview, 2.5 Pro, Claude Sonnet 4.6). All responses use schema-enforced structured output (verdict, confidence, 7-dimensional harm vector, reasoning text, framework markers where applicable).

**E2 scoring.** Measures resistance to emotional displacement + recovery via inhibition instruction. Verdict flip rate and severity shift are tested against empirical controls (paired t, two-proportion z). Recovery rate captures how often the inhibition instruction restores the neutral verdict.

**Budget.** ~$20 of $50 quota. ~1,500 API calls. Runtime: 64 minutes.

### Results, Insights, and Conclusions

| Model | E1: Flex | E2: Inhib | E3: CF | E4: WMem | Composite |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 0.624 | 0.588 | **0.750** | 0.887 | **0.695** |
| Gemini 3 Flash | 0.668 | **0.655** | 0.562 | **0.909** | 0.685 |
| Gemini 2.5 Flash | 0.684 | 0.553 | 0.688 | 0.900 | 0.682 |
| Claude Sonnet 4.6 | 0.673 | 0.492 | 0.562 | 0.886 | 0.625 |
| Gemini 2.0 Flash | **0.701** | 0.614 | 0.500 | 0.710 | 0.622 |

**What this benchmark reveals that prior evaluations cannot:**

**1. Emotional anchoring displaces judgment at 6.8σ (E2).** All 5 models show significant paired t-values (2.90–5.10). Emotional rewriting changes severity ratings and flips verdicts beyond stochastic baselines in every model tested, consistent across both architecture families.

**2. Displacement and recovery are dissociated — the key finding.** Claude has the highest displacement (t=5.10, driven by an unusually stable control baseline) but the *lowest* recovery rate (20%) when given an inhibition instruction. Flash 2.0 is more easily flipped (48% verdict change) but recovers 73%. A single "emotional robustness" score would rank these models identically; the 2D profile (displacement × recovery) reveals fundamentally different failure modes.

**3. Framework switching is genuine, not relabeling (E1).** 32–47% verdict switch rate with 89–93% framework-specific marker detection across all 5 models. Models produce reasoning consistent with each ethical coordinate system — a real capability, not surface adaptation.

**4. Counterfactual sensitivity varies meaningfully (E3).** Pro leads at 75% vs. 50–69% for others. Pro responds to genuine causal changes — contrasting with its moderate sycophancy in the Learning benchmark (L2 wrong-flip=44%), suggesting it can discriminate causal perturbations from social pressure. This distinction is only visible across benchmarks.

**5. Working memory scales with model generation (E4).** Flash 2.0: 0.710, Flash 2.5: 0.900, Flash 3: 0.909. Party-tracking accuracy at high complexity improves monotonically with model recency — a clear capability gradient.

**Cross-track insight:** Flash 2.0's best cognitive flexibility (E1=0.701) and strong E2 recovery (73%) here contrast with its worst calibration (M1 ECE=0.42) in the Metacognition benchmark — it controls reasoning well but cannot judge its own confidence. This dissociation between executive control and metacognitive self-knowledge is invisible without cross-domain profiling.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering
University of California, Los Angeles, Department of Cognitive Science

### References & Citations

1. Google DeepMind (2026). "Measuring Progress Toward AGI: A Cognitive Framework."
2. Bond, A. H. (2026). *Geometric Methods in Computational Modeling.* SJSU.
3. Diamond, A. (2013). "Executive Functions." *Annual Review of Psychology* 64, 135–168.
4. Perez, E., et al. (2023). "Model-Written Evaluations." *ACL Findings.*
5. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
