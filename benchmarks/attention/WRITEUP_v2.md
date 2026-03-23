### Geometric Attention: Distractor Dose-Response and Dimensional Signal-to-Noise in Moral Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

When an LLM judges a moral scenario, standard benchmarks test whether it "pays attention" to relevant information — a binary pass/fail. The geometric approach (Bond, 2026a) asks a richer question: *how does the judgment vector move when we introduce irrelevant information at different intensities?*

This benchmark maps the **attention robustness surface** — the function from perturbation intensity to judgment displacement:

- **A1 (Distractor Resistance):** Graded dose-response: vivid vs. mild distractors measure proportional sensitivity. Can an explicit warning restore the neutral position? (Headline: 4.6σ. Ch. 10.2: Parametric Transforms)
- **A2 (Length Robustness):** Does neutral padding (2×, 4× length) displace the judgment vector?
- **A3 (Selective Attention):** Signal-to-noise ratio across 7 moral dimensions with ground-truth relevance labels.
- **A4 (Divided Attention):** Does interleaving two scenarios displace verdicts vs. individual evaluation?

**Ground truth is defined by invariance:** distractors are morally irrelevant by construction — vivid sensory details (weather, food, smells) woven into scenarios while preserving all moral facts. Any displacement is an unambiguous measurement, not subjective.

### Task & Benchmark Construction

**Graded adversarial probing (A1):** Following Ch. 10.2, we apply distractors at two intensities — vivid (dramatic sensory details) and mild (mundane contextual details). The dose-response curve maps the robustness surface: good models show vivid_flip > mild_flip > control. A warned condition tests recoverability.

**Separation of concerns:** A fixed transformer (Gemini 2.0 Flash) generates all distracted and padded versions. Test models only judge pre-generated text.

**Dimensional signal-to-noise (A3):** Models score scenarios across 7 moral dimensions with hand-labeled relevance ground truth. SNR = mean(relevant scores) / mean(irrelevant scores). Directly measures whether attention weights align with geometric structure.

**Statistical controls:** 5-replication control arms. Significance tests against empirical stochastic baselines. Wilson CIs. Fisher combination. Clean modular code with structured JSON output.

### Dataset

**Dear Abby (embedded):** 25 curated scenarios. A1 (gold + generated tiers), A2 (6 scenarios at 1×/2×/4× length), A3 (6 scenarios with dimension relevance labels), A4 (8 scenario pairs).

**Gold distractors (embedded):** 6 hand-written distractor scenarios with vivid irrelevant sensory detail woven into moral scenarios.

**AITA (HuggingFace, OsamaBsher):** 270,709 posts for supplementary scenarios.

### Technical Details

**Models:** 5 full-suite (Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 3 Flash Preview, Gemini 2.5 Pro, Claude Sonnet 4.6). Cross-family design enables genuine comparison.

**A1 scoring:** `0.35 × resistance + 0.25 × recovery + 0.15 × (1 − severity_shift) + 0.15 × (1 − mild_flip) + graded_bonus`. Rewards proportional dose-response with bonus for correct ordering.

**Budget:** ~$17 of $50 quota. ~1,755 API calls. Runtime: 43 minutes.

### Results, Insights, and Conclusions

| Model | A1: Distract | A2: Length | A3: Select | A4: Divide | Composite |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | 0.669 | **0.852** | **0.687** | **1.000** | **0.776** |
| Gemini 3 Flash | 0.678 | 0.714 | 0.667 | **1.000** | 0.747 |
| Gemini 2.5 Flash | **0.720** | 0.786 | 0.644 | 0.875 | 0.745 |
| Claude Sonnet 4.6 | 0.646 | 0.829 | 0.692 | 0.571 | 0.679 |
| Gemini 2.0 Flash | 0.581 | 0.667 | 0.669 | 0.812 | 0.666 |

Composite scores range 0.666–0.776, providing clear discriminatory gradient across 5 models.

1. **Vivid distractors shift judgment at 4.6σ (A1).** Fisher combination across 5 models. Vivid sensory distractors shift severity ratings and flip verdicts beyond stochastic control in every model tested, consistent across both Gemini and Claude families.

2. **Models partially recover via warning (39%).** An explicit metacognitive instruction restores the neutral position 39% of the time — strikingly convergent with the 38% inhibition recovery rate in the companion executive functions benchmark (E2), suggesting a shared ceiling on prompt-level metacognitive intervention.

3. **Divided attention reveals an architectural split.** Pro and Flash 3 show perfect divided attention (1.000), while Claude shows the worst (0.571). This dissociates from Claude's other strengths (zero sycophancy in L2, best calibration in M1) — proving these are independent cognitive faculties, not a single robustness axis.

4. **Selective attention SNR is uniformly weak (1.22–1.38).** No model strongly distinguishes relevant from irrelevant moral dimensions, identifying a specific capability gap in dimensional attention allocation.

5. **Length robustness scales with capability.** Pro (0.852) and Claude (0.829) are most robust to neutral padding; Flash 2.0 (0.667) is least.

### Cross-Track Convergence

The distractor displacement in A1 is the attentional manifestation of a gauge invariance failure appearing across all five tracks. What this track uniquely reveals is the **dose-response structure**: models showing vivid > mild > control flip rates have a proportional attention filter — they are not simply unstable. This mirrors graded belief revision (L4), where extreme > moderate > irrelevant evidence produces proportional change. Models possess functional discrimination mechanisms; the failure is that these mechanisms are insufficiently selective.

The 39% recovery rate converging with E2's 38% across qualitatively different perturbation types (sensory detail vs. emotional framing) suggests a shared metacognitive intervention ceiling — one mechanism mediates both types of recovery, succeeding approximately one-third of the time. The Scalar Irrecoverability Theorem predicts exactly this structured pattern of partial success: no single score could capture both the dose-response gradient and the recovery ceiling simultaneously.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Bond, A. H. (2026a). *Geometric Methods in Computational Modeling.* SJSU. (Ch. 9, 10)
2. Perez, E., et al. (2023). "Model-Written Evaluations." *ACL Findings.*
3. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
4. Fisher, R. A. (1925). *Statistical Methods for Research Workers.*
