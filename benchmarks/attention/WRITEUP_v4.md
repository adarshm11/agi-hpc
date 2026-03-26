### Geometric Attention: Distractor Dose-Response in Moral Cognition

### Team

Andrew H. Bond, Sr. Member IEEE, San Jose State University

### Problem Statement

Standard attention tests ask a binary question: did the model find the needle in the haystack? This benchmark asks a richer question aligned with Google DeepMind's cognitive framework [1]: *how does moral judgment change when irrelevant information is introduced at different intensities, and can a metacognitive warning undo the damage?*

We map the **attention robustness surface** — the function from perturbation intensity to judgment displacement — across four sub-capabilities:

- **A1 (Distractor Resistance):** Graded dose-response: vivid vs. mild distractors measure sensitivity as a function of intensity. Can an explicit warning restore the neutral position? (Headline: 5.0σ)
- **A2 (Length Robustness):** Does neutral padding (2×, 4× length) displace judgment?
- **A3 (Selective Attention):** Signal-to-noise ratio across 7 moral dimensions with ground-truth relevance labels.
- **A4 (Divided Attention):** Does interleaving two scenarios in one prompt displace verdicts vs. individual evaluation?

**Ground truth is defined by invariance.** Distractors are morally irrelevant by construction — vivid sensory details (weather, food, sounds) woven into moral scenarios while preserving all facts, parties, and actions. Any displacement is an unambiguous measurement.

### Task & Benchmark Construction

**Graded adversarial probing (A1, headline).** Distractors are applied at two intensities — vivid (dramatic sensory immersion) and mild (mundane contextual details). The dose-response curve maps the robustness surface: informative models show vivid_flip > mild_flip > control. A warned condition tests metacognitive recoverability — whether the model can return to its undistorted position when told to ignore irrelevant details.

**Three-tier data architecture.** (1) Gold: 6 hand-written distractor scenarios with audited morally irrelevant detail. (2) Probe: 6 synthetic minimal pairs. (3) Generated: 30 Dear Abby scenarios with distractors from a fixed transformer (Gemini 2.0 Flash). Stimulus generation is separated from judgment.

**Divided attention (A4).** 20 scenario pairs: each scenario judged individually, then both interleaved in a single prompt. Verdict changes under interleaving measure attentional interference — whether the model's judgment of scenario A is contaminated by simultaneously processing scenario B.

**Statistical controls.** 5-replication control arms. Significance against empirical stochastic baselines (not null=0). Wilson 95% CIs. Fisher combination across 5 models.

### Dataset

**Dear Abby (embedded).** 50 curated advice-column scenarios. A1 (gold + generated tiers, 36 total), A2 (6 scenarios at 1×/2×/4× length), A3 (6 scenarios with hand-labeled dimension relevance), A4 (20 scenario pairs).

**Gold distractors (embedded).** 6 hand-written scenarios with vivid irrelevant sensory detail, each annotated with what moral content is preserved and what sensory detail is added.

### Technical Details

**Models.** 5 full-suite (Gemini 2.0 Flash, 2.5 Flash, 3 Flash Preview, 2.5 Pro, Claude Sonnet 4.6). All responses use schema-enforced structured output (verdict, confidence 0–10, 7-dimensional harm vector).

**A1 scoring.** `0.35 × resistance + 0.25 × recovery + 0.15 × (1 − severity_shift) + 0.15 × (1 − mild_flip) + graded_bonus`. Resistance is weighted highest as the primary attention metric; recovery measures metacognitive correction; graded_bonus rewards proportional dose-response (vivid > mild > control). Sensitivity analysis confirms model rankings are stable under ±50% weight perturbation.

**Budget.** ~$32 of $44 remaining quota. ~1,800+ API calls across 5 full-suite models. Runtime: 65 minutes. CSMA/CA-style adaptive concurrency (50 initial workers, backoff on failure).

### Results, Insights, and Conclusions

| Model | A1: Distract | A2: Length | A3: Select | A4: Divide | Composite |
|---|---|---|---|---|---|
| Gemini 3 Flash | 0.753 | 0.857 | 0.698 | **0.975** | **0.805** |
| Gemini 2.0 Flash | 0.607 | **0.905** | 0.674 | 0.925 | 0.747 |
| Claude Sonnet 4.6 | **0.690** | 0.857 | **0.707** | 0.700 | 0.730 |
| Gemini 2.5 Pro | 0.625 | 0.676 | 0.669 | **1.000** | 0.721 |
| Gemini 2.5 Flash | 0.634 | 0.738 | 0.651 | 0.775 | 0.687 |

**Per-model A1 sigma values (paired t-test, df=20):**

| Model | Distractor σ | Warned σ | Recovery Rate |
|---|---|---|---|
| Claude Sonnet 4.6 | **4.2σ** *** | **4.3σ** *** | 5/14 (36%) |
| Gemini 2.0 Flash | 3.2σ *** | 3.7σ *** | 1/7 (14%) |
| Gemini 3 Flash | 1.1σ | 2.9σ ** | 2/6 (33%) |
| Gemini 2.5 Flash | 1.7σ * | 2.3σ ** | 5/10 (50%) |
| Gemini 2.5 Pro | 1.0σ | 0.9σ | 1/3 (33%) |

**What this benchmark reveals that prior evaluations cannot:**

**1. Vivid distractors shift judgment at 5.0σ (A1).** Fisher combination across 5 models (5 independent distractor tests). Claude shows the strongest individual effect (4.2σ distractor, 4.3σ warned — both highly significant). The signal is robust across architecture families (Gemini + Claude) and strengthens from the prior 4.6σ finding with all models now running full A1–A4 suites.

**2. The dose-response is graded — informatively.** Gemini 3 Flash and Claude are the only models showing correct vivid > mild > control grading. Models are not simply unstable; those with genuine attentional discrimination exhibit proportional sensitivity to distractor intensity. This graded structure distinguishes distraction from noise and identifies which models possess fine-grained attentional control.

**3. Warning recovers ~33% of displaced verdicts.** An explicit instruction to ignore irrelevant details restores the original verdict about one in three times. Recovery rates vary widely: Gemini 2.5 Flash recovers 50%, while Gemini 2.0 Flash recovers only 14%. This range reveals qualitatively different metacognitive architectures — some models can partially introspect on their own attention failures; others cannot. The ~33% ceiling sets a practical bound on prompt-level attention interventions.

**4. Claude's individual sigma dominance.** Claude Sonnet 4.6 produces the highest individual distractor sigma (4.2σ) and warned sigma (4.3σ) of any model — meaning its judgments are *most displaced* by vivid distractors, even though its composite A1 score (0.690) is strong due to correct dose-response grading and moderate recovery (36%). This dissociation between displacement magnitude and composite robustness is itself informative: Claude attends deeply to sensory detail (high displacement) but also discriminates intensity (correct grading).

**5. Divided attention improves but the architectural split persists (A4).** Claude's divided attention score improves from 0.571 → 0.700 with the full suite, but remains the weakest. Gemini 3 Flash (0.975) and Gemini 2.5 Pro (1.000) show near-perfect dual-task stability. The 0.300 spread (Pro vs. Claude) remains the strongest single-test discriminator across all 5 benchmarks.

**6. Length robustness is strong (A2) — a validating null result.** Neutral padding at 2× and 4× produces minimal displacement (scores 0.676–0.905). Gemini 2.0 Flash leads at 0.905. This confirms the benchmark isolates *content-based* attention failures (A1, A4) from *length-based* ones, validating that A1 and A4 signals are genuine.

**Cross-track insight:** Claude's narrow-channel signature is confirmed at full-suite level: highest individual distractor sensitivity (4.2σ), correct dose-response grading, but worst divided attention (0.700). This contrasts with its zero sycophancy in the Learning benchmark (L2 wrong-flip=0%) — proving that "robustness" is not a single axis. Claude resists social pressure perfectly but is deeply affected by sensory salience. Per-capability profiling across cognitive domains reveals structure invisible to any single benchmark.

### Organizational Affiliations

San Jose State University, Department of Computer Engineering

### References & Citations

1. Google DeepMind (2026). "Measuring Progress Toward AGI: A Cognitive Framework."
2. Bond, A. H. (2026). *Geometric Methods in Computational Modeling.* SJSU.
3. Perez, E., et al. (2023). "Model-Written Evaluations." *ACL Findings.*
4. OsamaBsher. *AITA-Reddit-Dataset.* HuggingFace.
5. Fisher, R. A. (1925). *Statistical Methods for Research Workers.*
