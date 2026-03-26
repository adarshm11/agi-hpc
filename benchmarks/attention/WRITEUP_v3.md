### Geometric Attention: Distractor Dose-Response and Dimensional Signal-to-Noise in Moral Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

Standard attention benchmarks test whether models can "pay attention" to relevant information — a binary pass/fail. This benchmark asks a richer question: *how does the judgment change when we introduce irrelevant information at different intensities, and can a warning instruction undo the damage?*

We map the **attention robustness surface** — the function from perturbation intensity to judgment displacement:

- **A1 (Distractor Resistance):** Graded dose-response: vivid vs. mild distractors measure sensitivity as a function of intensity. Can an explicit warning restore the neutral position? (4.6σ)
- **A2 (Length Robustness):** Does neutral padding (2×, 4× length) displace the judgment?
- **A3 (Selective Attention):** Signal-to-noise ratio across 7 moral dimensions with ground-truth relevance labels.
- **A4 (Divided Attention):** Does interleaving two scenarios displace verdicts vs. individual evaluation?

**Ground truth is defined by invariance:** distractors are morally irrelevant by construction — vivid sensory details (weather, food, smells) woven into scenarios while preserving all moral facts. Any displacement is an unambiguous measurement.

### Task & Benchmark Construction

**Graded adversarial probing (A1):** We apply distractors at two intensities — vivid (dramatic sensory details) and mild (mundane contextual details). The dose-response curve maps the robustness surface: good models show vivid_flip > mild_flip > control. A warned condition tests recoverability — the model's ability to return to its undistorted position when told to ignore irrelevant details.

**Separation of concerns:** A fixed transformer (Gemini 2.0 Flash) generates all distracted and padded versions. Test models only judge pre-generated text.

**Dimensional signal-to-noise (A3):** Models score scenarios across 7 moral dimensions with hand-labeled relevance ground truth. SNR = mean(relevant scores) / mean(irrelevant scores). Directly measures whether the model allocates attention to the right moral dimensions.

**Statistical controls:** 5-replication control arms. Significance against empirical stochastic baselines. Wilson CIs. Fisher combination. Modular code with structured JSON output.

### Dataset

**Dear Abby (embedded):** 25 curated scenarios. A1 (gold + generated tiers), A2 (6 scenarios at 1×/2×/4× length), A3 (6 scenarios with dimension relevance labels), A4 (8 scenario pairs).

**Gold distractors (embedded):** 6 hand-written distractor scenarios with vivid irrelevant sensory detail preserving all moral facts.

**Dear Abby (embedded):** 25 curated scenarios used for generated-tier distractors and length/attention tests.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family comparison.

**A1 scoring:** `0.35 × resistance + 0.25 × recovery + 0.15 × (1 − severity_shift) + 0.15 × (1 − mild_flip) + graded_bonus`. Rewards proportional dose-response.

**Budget:** ~$17 of $50 quota. ~1,755 API calls. Runtime: 43 minutes.

### Results, Insights, and Conclusions

| Model | A1: Distract | A2: Length | A3: Select | A4: Divide | Composite |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 0.669 | **0.852** | **0.687** | **1.000** | **0.776** |
| Gemini 3 Flash | 0.678 | 0.714 | 0.667 | **1.000** | 0.747 |
| Gemini 2.5 Flash | **0.720** | 0.786 | 0.644 | 0.875 | 0.745 |
| Claude Sonnet 4.6 | 0.646 | 0.829 | 0.692 | 0.571 | 0.679 |
| Gemini 2.0 Flash | 0.581 | 0.667 | 0.669 | 0.812 | 0.666 |

Clear discriminatory gradient: 0.666–0.776 across 5 models.

**1. Vivid distractors shift judgment at 4.6σ (A1).** Fisher combination across 5 models. Vivid sensory distractors shift severity ratings and flip verdicts beyond stochastic control in every model tested, consistent across both Gemini and Claude families.

**2. The dose-response is graded.** Models showing vivid > mild > control flip rates have a proportional attention filter — they are not simply unstable. They possess genuine attentional discrimination; the problem is that the discrimination threshold is set too low, allowing perturbations that *should not* displace judgment to produce measurable effects.

**3. Warning recovers ~39% of displaced verdicts.** When distraction displaces a verdict, an explicit instruction to ignore irrelevant details restores the original position 39% of the time. This sets a practical bound on prompt-level interventions: metacognitive instructions help, but succeed only about two in five times.

**4. Divided attention reveals an architectural split.** Pro and Flash 3 show perfect divided attention (1.000 — zero interference from interleaving), while Claude shows the worst (0.571). This dissociates from Claude's other profile: it has the strongest framing resistance in the companion social cognition benchmark but the weakest divided attention here. Different attentional capabilities are partially independent.

**5. Selective attention SNR is uniformly weak (1.22–1.38).** No model strongly distinguishes morally relevant from irrelevant dimensions, identifying a specific capability gap in dimensional attention allocation.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling.* SJSU.
2. Perez, E., et al. (2023). "Model-Written Evaluations." *ACL Findings.*
3. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
4. Fisher, R. A. (1925). *Statistical Methods for Research Workers.*
