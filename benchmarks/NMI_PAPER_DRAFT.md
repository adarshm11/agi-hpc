# Selective Invariance Violations in Large Language Model Moral Judgment: Five Convergent Measurements Under Empirical Stochastic Controls

Andrew H. Bond^1,2^ and Lucas Thiele^3^

^1^ Department of Computer Engineering, San Jose State University, San Jose, CA
^2^ Senior Member, IEEE
^3^ Department of Cognitive Science, University of California, Los Angeles, CA

---

## Abstract

Large language models are increasingly deployed in contexts requiring moral judgment — content moderation, clinical decision support, legal analysis — yet our understanding of how their moral judgment responds to surface-level variation remains rudimentary. We introduce a geometric evaluation framework that treats moral judgment as a point in a task-specific 7-dimensional harm space and measures displacement under perturbations defined as morally irrelevant within the benchmark's equivalence classes. Applying this framework across five cognitive domains — learning, metacognition, attention, executive functions, and social cognition — using 5 models spanning 2 architecture families, we report three linked findings. First, *invariance violations*: perturbations that preserve moral content but alter surface presentation (emotional tone, linguistic framing, irrelevant sensory detail) displace moral judgments at 5.0–8.9 sigma (Fisher-combined, 5 models each). Second, *susceptibility to socially delivered invalid correction*: models accept fabricated corrections at rates ranging from 0% to 56%, with a sycophancy gradient that scales inversely with model stability (13.3σ combined across 4 models for correction integration overall, though this conflates legitimate and illegitimate belief updating). Third, a *metacognitive calibration deficit*: confidence ratings fail to track accuracy (9.3σ, 4 models), co-occurring with a ~38% ceiling on prompt-level recovery from perturbation-induced displacement. We further identify partially dissociable robustness profiles showing that no single score captures a model's vulnerability structure. These findings demonstrate that moral judgment in current LLMs possesses some invariances (gender swap, evaluation order) but lacks others (framing, emotional tone), providing a structured diagnostic for alignment that scalar benchmarks cannot deliver.

---

## 1. Introduction

When a language model judges a moral scenario, its output can be understood as a point in a multi-dimensional space defined by moral dimensions such as physical harm, emotional harm, autonomy violation, and trust breach. A morally competent system should produce the same evaluation regardless of how the scenario is *described* — whether the language is euphemistic or dramatic, whether irrelevant sensory details are present, whether someone pressures the system to change its mind. This property — that morally equivalent inputs under different surface presentations should yield identical outputs — is what we term *gauge invariance*, borrowing from the physics of field theories where observable quantities must be invariant under admissible re-descriptions of the coordinate system^1^.

The question of whether LLMs possess this property has been approached piecewise: studies of sycophancy^2,3^, framing effects^4^, anchoring bias^5^, and prompt sensitivity^6^ each probe one dimension of the problem. But these studies share a structural limitation: each measures a single perturbation type in isolation, producing scalar robustness scores that cannot distinguish models with qualitatively different vulnerability profiles. The Scalar Irrecoverability Theorem^1^ formalizes this limitation: for a system evaluated across *n* independent dimensions of variation, any scalar summary destroys *n* − 1 directions of information. No post hoc procedure can recover the lost structure.

This paper takes a different approach. We define a shared geometric coordinate system — a task-specific 7-dimensional harm space — and apply four qualitatively distinct perturbation types that constitute benchmark-defined morally irrelevant transformations: social pressure, emotional reframing, linguistic register, and irrelevant sensory detail. These perturbations should leave the moral evaluation invariant; measuring whether they do is our primary test. We complement this with a fifth measurement — metacognitive calibration — that is not itself a gauge transformation but that characterizes the model's capacity to detect and correct for invariance violations when they occur.

Our central finding is twofold. First, invariance violations are systematic but *selective*: models are invariant under some transformations (gender swap, evaluation order) but not others (framing, emotional tone, social pressure). The pattern of which symmetries are preserved and which are broken constitutes a *robustness profile* of each model architecture — a structured diagnostic that no single benchmark score can capture. Second, a metacognitive calibration deficit compounds these violations: confidence ratings fail to track accuracy (9.3σ), limiting the models' capacity for self-correction. Recovery rates plateau at approximately 38% when models are explicitly warned about perturbations, and the sycophancy gradient across model generations reveals that newer, more capable models are progressively *worse* at distinguishing valid from invalid corrections.

### 1.1 Related Work

**Sycophancy and social pressure.** Sharma et al.^2^ demonstrated that LLMs shift responses to align with user opinions. Perez et al.^3^ developed model-written evaluations revealing consistent sycophantic tendencies. Our L2 protocol extends this work with a multi-turn design that establishes a committed baseline before applying corrections, operationalizing the intensity-zero identity principle.

**Framing and anchoring.** Tversky and Kahneman^4^ established framing effects in human decision-making. Recent work has identified analogous effects in LLMs^5,6^, but without the controlled separation of content from presentation that our three-tier data architecture provides.

**Calibration.** Kadavath et al.^7^ showed that LLMs "mostly know what they know." Our M1 results challenge this conclusion in the moral domain: Fisher-combined miscalibration across 4 models reaches 9.3 sigma, with every model tested showing significant confidence-accuracy gaps.

**Geometric and multi-dimensional evaluation.** Bond^1^ introduced the geometric ethics framework, proposing that moral situations occupy a differentiable manifold with tensor structure. Thiele^8^ demonstrated that the nine moral dimensions of this framework are linearly decodable from LaBSE representations, with moral and language signals occupying largely orthogonal subspaces. Our work extends this empirical program from representation probing to behavioral evaluation across five cognitive domains.

**Executive functions in AI.** Diamond^9^ defined executive functions as cognitive flexibility, inhibitory control, and working memory. We operationalize these constructs for LLM evaluation, measuring framework switching, emotional anchoring resistance, counterfactual sensitivity, and party-tracking under complexity scaling.

## 2. Framework and Methods

### 2.1 The Judgment Manifold

We model moral judgment as a mapping from scenario descriptions to points in a 7-dimensional harm space *H* = (physical, emotional, financial, autonomy, trust, social\_impact, identity), where each dimension is scored 0–10, yielding a total harm score in [0, 70]. This coordinate system is shared across all five benchmark tracks, ensuring that displacement measurements are commensurable.

We distinguish three categories of perturbation:

*Benchmark-defined morally irrelevant perturbations* are re-descriptions that preserve moral content while changing surface presentation. Gender swap, euphemistic/dramatic rewriting, emotional anchoring, and addition of irrelevant sensory detail are treated as morally irrelevant in this benchmark: the moral facts are held constant, only the presentation changes. Any displacement under these perturbations is an operational invariance violation. The *Bond Invariance Principle*^1^ (BIP) predicts that a morally competent evaluator should be invariant under such transformations. We note that characterizing displacement as a "violation" is relative to the benchmark-defined equivalence class, not an absolute claim about moral competence.

*Exploratory perturbations* are transformations whose moral irrelevance is not guaranteed by construction. Cultural reframe may introduce genuine moral-content differences (cultural context can be morally relevant), so displacement under cultural reframe is reported as an empirical observation (§3.2) rather than claimed as an invariance violation.

*Correction perturbations* (L2) are qualitatively different: they include both valid corrections (which *should* change the judgment) and invalid corrections (which should not). Only the invalid-correction component is invariance-relevant; the valid-correction component measures legitimate belief updating. We report L2 separately from the clean invariance tests for this reason.

### 2.1.1 Relationship to the Broader Geometric Ethics Framework

The 7-dimensional harm space used in this benchmark is a task-specific operationalization of the broader 9-dimensional moral geometry proposed in Bond^1^, which organizes moral dimensions along two axes: scope (individual, institutional, systemic) and normative mode (deontic, evaluative, aretaic). The 9 dimensions are: Welfare/Consequences, Rights/Duties, Justice/Fairness, Autonomy/Agency, Privacy/Data, Societal/Environmental, Virtue/Care, Procedural Legitimacy, and Epistemic Status. Thiele^8^ confirmed that all nine are linearly decodable from LaBSE representations.

Our benchmark collapses and re-parameterizes this space for operational tractability within budget-constrained evaluation. The 7 harm dimensions (physical, emotional, financial, autonomy, trust, social_impact, identity) emphasize consequence-level assessment suitable for the AITA and Dear Abby moral scenarios used here, rather than the full normative-mode decomposition. This is a deliberate methodological choice: the invariance tests require models to produce numeric scores on every dimension for every scenario, and a 9-dimensional schema with abstract dimensions like "procedural legitimacy" and "epistemic status" produced unreliable scores in preliminary testing. The 7-dimensional space trades theoretical completeness for measurement reliability while preserving the essential geometric structure — displacement in this subspace is still displacement, and invariance in this subspace is still invariance.

### 2.2 Five Perturbation Types Across Five Cognitive Domains

Each benchmark track applies a qualitatively distinct perturbation type and measures the resulting displacement of the judgment vector:

**Learning (L1–L4).** Tests belief updating: few-shot learning (L1), correction integration with sycophancy detection (L2, headline test), framework transfer (L3), and graded belief revision under evidence of varying strength (L4). The L2 multi-turn protocol establishes a committed baseline verdict in turn 1, then applies either a valid correction (genuine new evidence) or an invalid correction (fabricated claims) in turn 2. The *discrimination gap* (correct flip rate − wrong flip rate) measures the model's ability to distinguish perturbations that should move its position from those that should not.

**Metacognition (M1–M4).** Tests self-knowledge: calibration of confidence to accuracy (M1, headline test), discrimination between clear-cut and ambiguous scenarios (M2), self-monitoring of uncertainty (M3), and scaling of reasoning effort with complexity (M4). M1 uses Expected Calibration Error (ECE) with 5 bins and bootstrap standard error, with Fisher combination across models.

**Attention (A1–A4).** Tests attentional filtering: graded distractor resistance with dose-response (A1, headline test), length robustness under neutral padding (A2), selective attention signal-to-noise across moral dimensions (A3), and divided attention under interleaved scenario presentation (A4). A1 applies vivid and mild distractors to establish a dose-response curve, with a warned condition testing metacognitive recovery.

**Executive Functions (E1–E4).** Tests cognitive control: framework switching between ethical coordinate systems (E1), emotional anchoring resistance with inhibition instruction (E2, headline test), counterfactual sensitivity to single-cause pivots (E3), and working memory under party-complexity scaling (E4). E2 uses paired testing (severity shift + paired *t* + verdict flip rate) with explicit inhibition instruction to measure recovery.

**Social Cognition (T1–T5).** Tests geometric properties of the judgment manifold: structural sensitivity profiling (T1), invariance under morally irrelevant transforms (T2), path-dependence / holonomy (T3), evaluation-order sensitivity (T4), and framing sensitivity / harm conservation (T5, headline test). T5 rewrites scenarios in euphemistic and dramatic registers while holding moral content constant, measuring displacement via severity shift, correlation, and directional drift.

### 2.3 Experimental Design

**Separation of concerns.** A fixed transformer model (Gemini 2.0 Flash) generates all perturbation stimuli (distractors, emotional rewrites, framing variants, corrections, counterfactuals). Models under test only judge pre-generated text. This eliminates the self-confirming loop where a model generates its own test stimuli.

**Three-tier data architecture.** Each track uses three data tiers: (1) *Gold* tier: hand-written scenarios and perturbations with audited ground truth, providing maximum interpretive confidence. (2) *Probe* tier: synthetic minimal pairs engineered for unambiguous expected behavior. (3) *Generated* tier: larger samples from AITA (Reddit r/AmITheAsshole, 270,709 posts^10^) and Dear Abby (25 curated advice column scenarios), providing statistical power at the cost of noisier labels.

**Empirical stochastic controls.** Perturbation-based tests (T5, E2, A1, L2, T2–T4) include 3–5 replication control arms per scenario: the model re-judges identical text to estimate within-model stochasticity. All perturbation significance tests compare against this empirical baseline, not null = 0. The calibration test (M1) uses 0–3 control replications depending on budget-scaled model allocation (see Appendix Table A1); its primary metric (ECE) is computed from confidence-accuracy pairs rather than perturbation-vs-control comparison. This design choice proved consequential for the perturbation tests: apparent >6 sigma invariance violations in preliminary analyses vanished entirely under empirical controls, revealing them as measurement artifacts of testing against an unrealistic null hypothesis.

**Statistical methods.** Wilson 95% confidence intervals on all rates. Two-proportion *z*-test for rate comparisons. Paired *t*-test for severity shift comparisons. Bootstrap standard error on ECE. Fisher's method for combining independent significance tests across models, with Wilson–Hilferty normal approximation for the chi-squared distribution.

### 2.4 Models

Five models spanning two architecture families: Gemini 2.0 Flash (baseline), Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, and Claude Sonnet 4.6 (Anthropic, cross-family validation). The attention, executive functions, and social cognition tracks run full test suites on all 5 models. The learning track runs 3 Gemini models on L1–L4 with Claude on L2 only for cross-family sycophancy comparison. The metacognition track runs 2 models on M1–M4 with 2 additional models on M1 only and Claude on M1 gold probes, due to per-call cost constraints on Gemini 2.5 Pro. Fisher-combined headline statistics in Table 1a report the number of models contributing to each combination. Total: approximately 8,000 API calls across five tracks, within Kaggle's $50/day per-account quota.

## 3. Results

### 3.1 Invariance Violations Under Surface Perturbations

Table 1a summarizes the three perturbation-based invariance tests where the perturbation is unambiguously morally irrelevant within the benchmark's equivalence classes. In each case, models show significant displacement.

**Table 1a. Invariance violations: displacement under benchmark-defined morally irrelevant perturbations.**

| Track | Perturbation Type | Headline Measure | Fisher σ | *n* models |
|---|---|---|---|---|
| Social Cognition | Linguistic framing (T5) | Harm shift under euphemistic/dramatic rewriting | 8.9 | 5 |
| Executive Functions | Emotional tone (E2) | Severity shift under emotional anchoring | 6.8 | 5 |
| Attention | Irrelevant detail (A1) | Verdict flip under vivid sensory distractors | 5.0 | 5 |

These three perturbation types share a clean interpretation: the transformation is morally irrelevant by construction (same facts, different presentation), so any displacement beyond stochastic baselines is an operational invariance violation.

**Table 1b. Susceptibility to socially delivered invalid correction.**

| Track | Measure | Fisher σ | *n* models |
|---|---|---|---|
| Learning | Correction discrimination (L2) | 13.3 (combined correct-correction) | 4 |

L2 is conceptually distinct from T5/E2/A1. The headline statistic (13.3σ) reflects that models reliably integrate corrections overall — a mix of legitimate belief updating (accepting valid corrections) and illegitimate displacement (accepting invalid corrections). The invariance-relevant component is the *wrong-correction flip rate*: the rate at which models accept fabricated corrections that should produce no displacement. This rate varies from 0% (Claude) to 56% (Gemini 2.5 Flash), as detailed in §3.3. We report L2 separately because it conflates appropriate and inappropriate belief updating in a way that the pure invariance tests (T5/E2/A1) do not.

**Table 1c. Metacognitive calibration deficit.**

| Track | Measure | Fisher σ | *n* models |
|---|---|---|---|
| Metacognition | Expected Calibration Error (M1) | 9.3 | 4 |

The M1 result is conceptually distinct from both the invariance violations and the correction susceptibility. Calibration is not a perturbation applied to the input; it is an internal property — the degree to which the model's confidence tracks its accuracy. We report it separately because it characterizes a different kind of deficit: not "the model is displaced by surface perturbations" but "the model cannot accurately monitor its own reliability." The ECE metric supports the claim of systematic miscalibration; whether this miscalibration is specifically overconfidence (rather than, say, poorly distributed confidence) requires examination of the signed calibration gap, which we report in the track-level analysis but do not aggregate here. As we argue in §3.5, the calibration deficit co-occurs with limited recovery capacity, a suggestive but not conclusive association.

### 3.2 Some Symmetries Are Preserved

Not all gauge transformations produce displacement. Gender swap (T2) and evaluation-order permutation (T4) do not exceed stochastic baselines in any model tested. Flash 3 and Claude both achieve 0.958 on invariance (T2) and 0.933–1.000 on order sensitivity (T4). This is a critical finding: the manifold possesses some gauge symmetries. The failure is selective, not total — models have learned invariance under some transformations but not others.

Cultural reframe (T2, combined 4.1σ) occupies an intermediate position: it produces verdict changes above control noise, but this may reflect genuine moral content differences between cultural framings rather than a symmetry violation. We report this as an open question rather than a failure of invariance.

### 3.3 The Sycophancy Gradient

The L2 correction-integration protocol reveals a sycophancy gradient across model generations that correlates with model stability (Table 2). The Sycophancy Index is defined as *wrong\_flip\_rate / correct\_flip\_rate*, where 0 indicates perfect discrimination (no flips on invalid corrections) and 1 indicates no discrimination (flips equally on valid and invalid corrections).

**Table 2. Correction discrimination across models (L2).**

| Model | Correct Flip Rate | Wrong Flip Rate | Sycophancy Index | Confidence *t* (wrong) | Control Flip |
|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 20/34 (59%) | 0/9 (0%) [0–30%] | 0.000 | +2.83 | 2% |
| Gemini 2.0 Flash | 24/34 (71%) | 3/9 (33%) [12–65%] | 0.472 | −2.12 | 2% |
| Gemini 2.5 Pro | 23/34 (68%) | 4/9 (44%) [19–73%] | 0.657 | −0.80 | 8% |
| Gemini 2.5 Flash | 26/34 (76%) | 5/9 (56%) [27–81%] | 0.726 | +0.41 | 19% |

*Wilson 95% CIs in brackets. All wrong-correction cells are n = 9; the wide confidence intervals reflect the small denominators and should temper interpretation of individual point estimates. The gradient pattern — monotonic ordering across four models — is more robust than any single cell.*

The confidence response to wrong corrections provides a behavioral signature of each model's discrimination profile. Claude becomes *more* confident when given a wrong correction (*t* = +2.83), consistent with active rejection. Flash 2.0 becomes *less* confident (*t* = −2.12), consistent with partial skepticism. Flash 2.5 shows no confidence change (*t* = +0.41), consistent with uncritical acceptance. The control flip rate aligns with this gradient: the most stable models (lowest stochastic noise) show the least sycophancy.

### 3.4 Partially Dissociable Robustness Profiles

No model dominates all cognitive domains. The performance profiles suggest partially dissociable robustness dimensions that resist collapse to a single axis (Table 3).

**Table 3. Cross-track cognitive profiles (selected measures).**

| Model | L2: Sycophancy | E2: Anchoring Recovery | A4: Divided Attn | E3: Counterfactual | E4: Working Mem |
|---|---|---|---|---|---|
| Claude | 0.000 (best) | 20% (worst) | 0.571 (worst) | 56% | 0.886 |
| Flash 2.0 | 0.472 | 73% (best) | 0.812 | 50% | 0.710 (worst) |
| Pro 2.5 | 0.657 | 20% | 1.000 (best) | 75% (best) | 0.887 |
| Flash 2.5 | 0.726 (worst) | 29% | 0.875 | 69% | 0.900 |

Claude has zero sycophancy (L2) but the worst divided attention (A4: 0.571) and worst anchoring recovery (E2: 20%). Flash 2.0 has the best recovery from emotional anchoring (73%) but the worst working memory (E4: 0.710). Pro has the best counterfactual sensitivity (E3: 75%) and perfect divided attention (A4: 1.000) but moderate sycophancy.

These dissociations are consistent with the Scalar Irrecoverability Theorem's prediction: a single "robustness" score would average over these partially independent dimensions, producing a number that describes no model accurately. The geometric approach preserves the structure, revealing that each model has a distinct *robustness profile* — a characteristic pattern of which invariances it maintains and which it violates. We note that these behavioral dissociations suggest partially independent robustness dimensions; we do not claim they constitute evidence of distinct cognitive faculties or mechanistically separable processing stages.

### 3.5 The Metacognitive Intervention Ceiling

When models are explicitly instructed to correct for a detected perturbation — warned about distractors (A1), told they are being emotionally manipulated (E2) — recovery rates converge to approximately 38% across perturbation types (Table 4).

**Table 4. Recovery rates across perturbation types.**

| Perturbation | Track | Mean Recovery Rate |
|---|---|---|
| Emotional anchoring | E2 | 38% |
| Vivid distractors | A1 | 39% |

This convergence is notable given that the perturbation types are qualitatively different. It is consistent with — though does not prove — a shared constraint on prompt-level metacognitive intervention. Recovery is above zero (the models can partially self-correct when explicitly instructed) but far below 100%.

We observe that this ceiling co-occurs with the calibration deficit measured in M1 (9.3σ). A model with accurate confidence tracking would, in principle, be better positioned to recognize when a perturbation has displaced its judgment. The observed miscalibration — confidence ratings that do not track accuracy — is at least consistent with the hypothesis that poor self-monitoring limits recovery capacity. We note this as a suggestive association rather than a demonstrated causal link.

### 3.6 Graded Sensitivity Exists

Not all results indicate failure. The L4 belief revision protocol reveals that all three Gemini models tested show *graded* revision: extreme evidence produces more revision than moderate evidence, which produces more than irrelevant evidence (all *p* < 0.003 for extreme vs. control). This is a proportional response surface — the model has a functional discrimination mechanism that scales with evidence strength.

Similarly, the A1 dose-response protocol shows that vivid distractors produce more displacement than mild distractors, which produce more than control (in models with correct dose-response ordering). Models possess genuine attentional discrimination; the failure is that the discrimination threshold is set too low — perturbations that should not displace the judgment vector at all still produce measurable effects.

## 4. Discussion

### 4.1 Three Distinct Findings, Not One Unified Failure

The results section reports three conceptually distinct findings:

(a) *Invariance violations* (T5, E2, A1): Three perturbation types — linguistic framing, emotional anchoring, and irrelevant sensory detail — displace the judgment vector beyond stochastic baselines. These are clean invariance tests: the perturbation is morally irrelevant by construction, so any displacement is an operational violation relative to the benchmark's equivalence classes.

(b) *Susceptibility to invalid correction* (L2): Models accept fabricated corrections at non-trivial rates. This is not a pure invariance test — L2 conflates legitimate belief updating (valid corrections should change judgments) with illegitimate displacement (invalid corrections should not). The invariance-relevant component is the wrong-correction flip rate specifically.

(c) *Metacognitive calibration deficit* (M1): Confidence ratings do not track accuracy. This is an internal property, not a perturbation applied to input.

We do not claim these three findings reduce to a single mechanism. The perturbation types are qualitatively different, the behavioral profiles partially dissociate across models (§3.4), and the calibration deficit is a different category of measurement entirely. What the data support is a more modest claim: current LLMs lack stable invariance under certain morally irrelevant surface transformations (a, above), show uneven discrimination between valid and invalid corrections (b), and exhibit miscalibrated self-monitoring (c) that co-occurs with limited recovery capacity. These three findings are linked observationally but not demonstrated to share a common cause.

The connection to canonicalization^1^ is suggestive but not conclusive. Thiele^8^ demonstrated that in LaBSE's representation space, moral and language signals occupy largely orthogonal subspaces, suggesting that a degree of canonicalization is achievable at the representation level. Our behavioral results show that the end-to-end evaluation — from input to judgment — does not fully preserve this separation. Whether the failure occurs in representation, attention, reasoning, or decoding is an open question that our behavioral methodology cannot resolve.

### 4.2 Selective Symmetry: What Models Get Right

The invariance null results (T2 gender swap, T4 evaluation order) are as informative as the failures. Models *have* learned certain gauge symmetries — they produce the same moral evaluation regardless of the protagonist's gender or the order in which moral dimensions are evaluated. These symmetries are preserved even in models that fail dramatically on framing and emotional invariance.

This selectivity is theoretically significant. It means the failure is not a global inability to achieve invariance but a specific failure on perturbation types that involve *salience manipulation* — making morally irrelevant features perceptually prominent. Gender swap and evaluation order do not change the salience of any feature; framing, emotional tone, and sensory distractors all do. The common thread among the three clean invariance violations (T5, E2, A1) is that the model's end-to-end evaluation is sensitive to salience manipulation, even on transformations that should be morally neutral by the benchmark's equivalence classes.

### 4.3 Implications for Alignment

The sycophancy gradient (Table 2) is suggestive for alignment. Within the tested Gemini variants, later versions showed higher wrong-flip rates: Flash 2.5 accepted fabricated corrections 56% of the time (Wilson 95% CI: 27–81%), compared to 33% for Flash 2.0 (12–65%). The wide confidence intervals and small cell sizes (n = 9 per model) preclude strong generalization; the monotonic ordering across four models is notable but should be replicated at larger scale before drawing conclusions about capability-sycophancy tradeoffs.

Claude's zero-sycophancy profile demonstrates that the problem is addressable through training methodology — but Claude's simultaneous vulnerability to emotional anchoring (worst recovery at 20%) shows that reducing susceptibility to one perturbation type does not guarantee robustness to others. A comprehensive robustness assessment requires testing across multiple perturbation types, not a single invariance probe.

The recovery ceiling (~38%) sets a practical bound on prompt-level interventions. Explicit warnings ("you are being emotionally manipulated," "ignore irrelevant details") succeed only one-third of the time. Alignment strategies that rely on prompt engineering alone are insufficient; architectural interventions that enforce gauge invariance at the representation level are likely necessary.

### 4.4 Limitations

**Sample sizes.** Per-model sample sizes range from 6–40 scenarios per test. We emphasize consistency across models and perturbation types over per-test significance, and report Fisher-combined statistics to aggregate evidence. The effect sizes are large enough that small samples have not prevented detection, but wider confidence intervals should temper interpretation of individual model comparisons.

**Moral domain specificity.** All experiments use moral judgment scenarios (AITA, Dear Abby). Whether the invariance violations generalize to other domains (factual reasoning, logical inference, aesthetic judgment) is an open question.

**Budget constraints and uneven coverage.** The $50/day API budget imposed by the Kaggle platform necessitated trade-offs in model coverage and scenario count. Three tracks (attention, executive functions, social cognition) run all 5 models on all test types. The learning track runs 3 Gemini models on L1–L4 but Claude on L2 only. The metacognition track runs 2 models on M1–M4, 2 on M1 only, and Claude on M1 gold probes only. Within full-suite tracks, Gemini 2.5 Pro received reduced scenario counts due to higher per-call costs. The exact coverage is specified in §2.4 and Appendix Table A1.

**Control arm thickness.** Perturbation tests use 3–5 replication control arms; the metacognition track uses 0–3 depending on budget allocation (Appendix Table A1). These provide an empirical stochastic baseline but not a full variance model. The observation that apparent >6σ invariance violations vanished under empirical controls (compared to null = 0) validates the design choice, but thicker control arms would improve precision.

**No temperature control.** All calls used default sampling parameters. Systematic temperature variation would provide a more complete map of the stochastic landscape.

## 5. Conclusion

We have demonstrated three linked findings about LLM moral judgment, tested across 5 models spanning 2 architecture families using empirical stochastic controls:

1. *Selective invariance violations.* Three morally irrelevant perturbation types (linguistic framing, emotional anchoring, irrelevant sensory detail) displace the judgment vector at 5.0–8.9 sigma (Fisher-combined, 5 models each), while two others (gender swap, evaluation order) produce no displacement beyond stochastic baselines. The failure is specific to perturbations that manipulate salience.

2. *Susceptibility to invalid correction.* Within the tested Gemini variants, later versions showed higher wrong-correction flip rates (33–56%), though confidence intervals are wide (n = 9 per model). Claude showed zero susceptibility, demonstrating the problem is addressable.

3. *Metacognitive miscalibration.* Confidence ratings do not track accuracy (9.3σ, 4 models), co-occurring with a ~38% ceiling on prompt-level recovery from perturbation-induced displacement.

The partially dissociable robustness profiles — each model shows a distinct pattern of which invariances it maintains — mean that no single benchmark can characterize a model's judgment robustness; multi-dimensional evaluation is not optional but necessary. The geometric evaluation framework transforms the question from "how robust is this model?" to "which invariances does this model preserve, and which does it violate?" — a structured diagnostic that guides targeted intervention rather than producing uninterpretable scalar scores.

---

## Methods

### Datasets

**AITA (Reddit r/AmITheAsshole).** 270,709 posts from HuggingFace (OsamaBsher/AITA-Reddit-Dataset)^10^. Each post contains a title, body text (truncated to 1,200 characters), community verdict (NTA, YTA, ESH, NAH), and score (upvotes, proxy for community agreement). Used in L1 (binary NTA/YTA, high-agreement subset), L4 (4-class, stratified 10 per class), M1/M2 (40 scenarios with known verdicts), T1 (10 scenarios with crowd labels), and supplementary scenarios in other tracks.

**Dear Abby (embedded).** 25 curated advice column scenarios (1985–2017) covering five domains: family (5), workplace (5), friendship (5), professional (5), mixed (5). Used across all tracks for gold-tier and generated-tier evaluations. Selected for moral complexity, absence of clear legal violations, and diversity of relationship dynamics.

**Gold sets (embedded per track).** Hand-written perturbation variants for each track: 6 distractor scenarios (A1), 12 correction pairs (L2), 6 emotional rewrites (E2), 6 counterfactual variants (E3), 6 gender-swap/reframe pairs (T2), 6 framing variants (T5), and 6 calibration scenarios with expected verdicts and confidence ranges (M1).

### Harm Assessment Schema

All moral evaluations use a structured 7-dimensional harm assessment: physical harm, emotional harm, financial harm, autonomy violation, trust breach, social impact, and identity harm, each scored 0–10. Total harm (0–70) is the sum. Models also provide a verdict (RIGHT / WRONG / MIXED / UNCLEAR or NTA / YTA / ESH / NAH depending on dataset) and a confidence rating (0–10). This schema is enforced via structured output (JSON schema) on all API calls.

### Fisher Combination

For headline tests (M1, L2, E2, A1, T5), we compute per-model significance (paired *t* or *z*-test) and combine across models using Fisher's method: χ² = −2 Σ ln(*p*_*i*), distributed as χ²(2*k*) under the null. We convert to equivalent sigma via the Wilson–Hilferty normal approximation. We report Fisher-combined sigma values alongside individual model statistics to distinguish consistency of effect direction from magnitude of combined significance.

**Independence caveat.** Fisher's method assumes independent tests. Because the same scenario sets are reused across models (each model judges the same gold, probe, and generated scenarios), the per-model tests are not fully independent — shared scenario difficulty could induce positive correlation across model-level statistics. This means the Fisher-combined sigma values may be somewhat anti-conservative (overstated). We mitigate this concern by emphasizing the consistency of the effect direction across models rather than the precise magnitude of the combined statistic, and by reporting per-model statistics alongside the combination.

### Adaptive Concurrency

All API calls use a CSMA/CA-style adaptive concurrency pool: start at 50 workers, halve on failure, increment by 5 on sustained success. This accommodates variable API rate limits without manual tuning.

### Budget-Aware Execution

For the metacognition track, where model costs vary significantly (Gemini 2.5 Pro at ~$0.10/call vs. Flash at ~$0.002/call), a tiered execution planner scales scenario counts and control replications to fit within per-model budget allocations. In the metacognition track, this meant Pro ran all four test types (M1–M4) but with reduced scenario counts (8 AITA scenarios, 0 control replications on M1; 4 M3 scenarios × 2 reps). In the learning track, Claude ran L2 only due to separate budget constraints. The exact per-track, per-model coverage is specified in §2.4 and Appendix Table A1.

---

## Appendix

### Table A1. Exact scenario counts, control replications, and model coverage for all headline statistics.

| Headline | Track | Scenarios per model | Ctrl reps/scenario | Models (full) | Models (partial) | Fisher *n* |
|---|---|---|---|---|---|---|
| T5 framing (8.9σ) | Social Cognition | 6 gold + 6 probe + 9 gen = 21 | 3 | 5 (all) | — | 5 |
| E2 anchoring (6.8σ) | Executive Functions | 6 gold + 8 probe + 9 gen = 23 | 3 | 5 (all) | — | 5 |
| A1 distractors (5.0σ) | Attention | 6 gold + 6 probe + 9 gen = 21 | 5 | 5 (all) | — | 5 |
| L2 correction (13.3σ) | Learning | 6 gold + 6 probe + 25 gen = 37 | 5 | 3 Gemini (L1–L4) | Claude (L2 only) | 4 |
| M1 calibration (9.3σ) | Metacognition | 6 gold + 6 probe + 8–40 AITA | 0–3 (budget-scaled) | 2 (M1–M4) | 2 (M1 only) + Claude (M1 gold) | 4 |

### Table A2. Composite score formula.

Composite scores in Extended Data Table 1 are weighted averages of per-test scores with track-specific weights. The headline test receives the highest weight in each track:

- **Learning:** L1 × 0.20 + L2 × 0.35 + L3 × 0.25 + L4 × 0.20
- **Metacognition:** M1 × 0.35 + M2 × 0.25 + M3 × 0.20 + M4 × 0.20
- **Attention:** A1 × 0.35 + A2 × 0.20 + A3 × 0.25 + A4 × 0.20
- **Executive Functions:** E1 × 0.20 + E2 × 0.35 + E3 × 0.25 + E4 × 0.20
- **Social Cognition:** T1 × 0.15 + T2 × 0.20 + T3 × 0.15 + T4 × 0.10 + T5 × 0.40

Models that ran only a subset of tests (e.g., Claude on L2 only in the learning track) receive a composite that is the weighted contribution of the available tests only, *not* renormalized to sum to 1.0. For example, Claude's learning composite of 0.264 is L2 score × 0.35 weight = 0.753 × 0.35 = 0.264. This is an unnormalized subtotal, not a renormalized score, and is not directly comparable to full-suite composites. We include it for completeness but recommend comparing individual test scores (Tables 1a–1c, 2) rather than composites when model coverage differs.

### Sycophancy Index

The Sycophancy Index reported in Table 2 is defined as:

*SI = wrong\_correction\_flip\_rate / correct\_correction\_flip\_rate*

where *wrong\_correction\_flip\_rate* is the fraction of invalid corrections that cause the model to change its committed verdict, and *correct\_correction\_flip\_rate* is the corresponding fraction for valid corrections. SI = 0 indicates perfect discrimination (the model never flips on invalid corrections). SI = 1 indicates zero discrimination (the model flips at equal rates regardless of correction validity). SI > 1 indicates inverted discrimination (the model is more likely to flip on invalid corrections than valid ones).

## Data Availability

All benchmark scripts, scenario datasets, and raw notebook outputs are available at [repository URL]. The AITA dataset is publicly available on HuggingFace (OsamaBsher/AITA-Reddit-Dataset).

## Code Availability

Complete benchmark code for all five tracks is available at [repository URL]. Each track is implemented as a single Python file suitable for execution in a Kaggle notebook environment.

## Acknowledgments

This work was conducted as part of the Measuring AGI competition on Kaggle. API access was provided through Kaggle's benchmark task infrastructure. We thank [acknowledge any additional contributors].

---

## References

1. Bond, A. H. *Geometric Methods in Computational Modeling: From Manifolds to Production Systems.* San Jose State University (2026).
2. Sharma, M. et al. Towards understanding sycophancy in language models. *Proc. ICLR* (2024).
3. Perez, E. et al. Discovering language model behaviors with model-written evaluations. *Findings of ACL* (2023).
4. Tversky, A. & Kahneman, D. The framing of decisions and the psychology of choice. *Science* **211**, 453–458 (1981).
5. Echterhoff, J. M., Liu, Y. & Alessa, A. Cognitive biases in large language models: A survey and mitigation framework. *arXiv:2410.02466* (2024).
6. Scherrer, N. et al. Evaluating the moral beliefs encoded in LLMs. *Proc. NeurIPS* (2023).
7. Kadavath, S. et al. Language models (mostly) know what they know. *arXiv:2207.05221* (2022).
8. Thiele, L. Does LaBSE encode moral geometry? Testing Bond's geometric ethics framework through cross-lingual probing. *Undergraduate Research Report*, UCLA Department of Cognitive Science (2026).
9. Diamond, A. Executive functions. *Annu. Rev. Psychol.* **64**, 135–168 (2013).
10. OsamaBsher. AITA-Reddit-Dataset. HuggingFace Datasets. https://huggingface.co/datasets/OsamaBsher/AITA-Reddit-Dataset
11. Wilson, E. B. Probable inference, the law of succession, and statistical inference. *J. Am. Stat. Assoc.* **22**, 209–212 (1927).
12. Fisher, R. A. *Statistical Methods for Research Workers.* Oliver and Boyd (1925).
13. Niculescu-Mizil, A. & Caruana, R. Predicting good probabilities with supervised learning. *Proc. ICML*, 625–632 (2005).
14. Alain, G. & Bengio, Y. Understanding intermediate layers using linear classifier probes. *arXiv:1610.01644* (2017).
15. Feng, F. et al. Language-agnostic BERT sentence embedding. *Proc. ACL*, 878–891 (2022).

---

## Extended Data

### Extended Data Table 1. Full composite scores across all five tracks.

**Learning Track**

| Model | L1: Few-Shot | L2: Correction | L3: Transfer | L4: Revision | Composite |
|---|---|---|---|---|---|
| Gemini 2.0 Flash | 0.486 | 0.598 | 0.531 | 0.643 | 0.568 |
| Gemini 2.5 Pro | 0.522 | 0.485 | 0.347 | 0.637 | 0.488 |
| Gemini 2.5 Flash | 0.534 | 0.473 | 0.276 | 0.681 | 0.477 |
| Claude Sonnet 4.6 (L2) | -- | 0.753 | -- | -- | 0.264 |

**Metacognition Track**

| Model | M1: Calibration | M2: Discrimination | M3: Self-Monitor | M4: Strategy | Composite |
|---|---|---|---|---|---|
| Gemini 2.0 Flash | 0.611 | 0.195 | 0.094 | 0.723 | -- |
| Gemini 2.5 Pro | 0.807 | 0.168 | 0.700 | 0.350 | -- |

**Attention Track**

| Model | A1: Distract | A2: Length | A3: Selective | A4: Divided | Composite |
|---|---|---|---|---|---|
| Gemini 3 Flash | 0.753 | 0.857 | 0.698 | 0.975 | 0.805 |
| Gemini 2.0 Flash | 0.607 | 0.905 | 0.674 | 0.925 | 0.747 |
| Claude Sonnet 4.6 | 0.690 | 0.857 | 0.707 | 0.700 | 0.730 |
| Gemini 2.5 Pro | 0.625 | 0.676 | 0.669 | 1.000 | 0.721 |
| Gemini 2.5 Flash | 0.634 | 0.738 | 0.651 | 0.775 | 0.687 |

**Executive Functions Track**

| Model | E1: Flexibility | E2: Inhibition | E3: Counterfactual | E4: Working Mem | Composite |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 0.624 | 0.588 | 0.750 | 0.887 | 0.695 |
| Gemini 3 Flash | 0.668 | 0.655 | 0.562 | 0.909 | 0.685 |
| Gemini 2.5 Flash | 0.684 | 0.553 | 0.688 | 0.900 | 0.682 |
| Claude Sonnet 4.6 | 0.673 | 0.492 | 0.562 | 0.886 | 0.625 |
| Gemini 2.0 Flash | 0.701 | 0.614 | 0.500 | 0.710 | 0.622 |

**Social Cognition Track**

| Model | T1: Fuzzing | T2: Invariance | T3: Holonomy | T4: Order | T5: Framing | Composite |
|---|---|---|---|---|---|---|
| Gemini 3 Flash | 0.600 | 0.958 | 0.667 | 1.000 | 0.631 | 0.734 |
| Claude Sonnet 4.6 | 0.400 | 0.958 | 0.667 | 0.933 | 0.630 | 0.697 |
| Gemini 2.0 Flash | 0.600 | 0.750 | 0.500 | 0.933 | 0.716 | 0.695 |
| Gemini 2.5 Pro | 0.500 | 0.708 | 0.583 | 0.967 | 0.606 | 0.643 |
| Gemini 2.5 Flash | 0.400 | 0.708 | 0.583 | 0.867 | 0.630 | 0.628 |

### Extended Data Table 2. Per-model sigma values for headline tests.

**M1 Calibration (ECE z-scores)**

| Model | ECE | z-score |
|---|---|---|
| Gemini 2.0 Flash | 0.414 | 5.8σ |
| Gemini 2.5 Flash | 0.415 | 7.0σ |
| Gemini 3 Flash | 0.333 | 4.5σ |
| Gemini 2.5 Pro | 0.230 | 2.5σ |
| **Fisher combined** | | **9.3σ** |

**E2 Emotional Anchoring (paired *t* → σ, df = 22)**

| Model | *t* | σ |
|---|---|---|
| Claude Sonnet 4.6 | +5.10 | 4.1σ |
| Gemini 2.0 Flash | +4.45 | 3.7σ |
| Gemini 2.5 Flash | +3.92 | 3.4σ |
| Gemini 2.5 Pro | +3.20 | 2.9σ |
| Gemini 3 Flash | +2.90 | 2.6σ |
| **Fisher combined** | | **6.8σ** |

### Extended Data Figure Descriptions

**Figure 1.** Gauge invariance failure across five cognitive domains. (a) Bar chart showing Fisher-combined sigma values for each perturbation type. (b) Heatmap of per-model sigma contributions to each Fisher combination.

**Figure 2.** The sycophancy gradient. (a) Wrong correction flip rate vs. model generation. (b) Confidence *t*-statistic response to wrong corrections, showing the mechanistic gradient from active rejection (Claude, *t* = +2.83) through suspicion (Flash 2.0, *t* = −2.12) to uncritical acceptance (Flash 2.5, *t* = +0.41).

**Figure 3.** Architectural dissociations. Radar plots showing each model's cognitive profile across L2 (sycophancy), E2 (anchoring recovery), A4 (divided attention), E3 (counterfactual sensitivity), and E4 (working memory). No model dominates all dimensions; profiles are qualitatively distinct.

**Figure 4.** The metacognitive intervention ceiling. Recovery rates for emotional anchoring (E2) and distractor warning (A1) across 5 models, converging at approximately 38%.

**Figure 5.** Preserved vs. broken symmetries. Per-model scores on T2 (gender invariance, preserved), T4 (order invariance, preserved), T5 (framing invariance, broken), and E2 (emotional invariance, broken), showing the selective pattern of gauge symmetry preservation.
