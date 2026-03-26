# Geometric Reasoning: From Search to Manifolds
## Book Plan — Andrew H. Bond

### Thesis
Reasoning — human and artificial — is informed search on a structured possibility space.
The quality of reasoning is determined by the geometry of that space and the fidelity of
the heuristic field that guides traversal. This framework unifies cognitive science, AI,
and mathematics under a single geometric vocabulary.

---

## Part I: The Search-Geometry Connection

### Chapter 1: Reasoning as Search
- Newell & Simon's problem space hypothesis (1972)
- Why "search" is not a metaphor — it is a mathematical claim
- The spectrum: uninformed → greedy → informed → optimal
- What A* actually guarantees (admissibility, consistency, optimality)
- Preview: when the search space has geometry, everything changes

### Chapter 2: When the Space Has Shape
- From graphs to metric spaces to manifolds
- Distance, cost, curvature, boundaries — the geometric toolkit
- Why Euclidean intuitions mislead (Ch. 1-3 of Geometric Methods)
- The manifold hypothesis for reasoning: cognitive states live on a
  low-dimensional manifold embedded in high-dimensional activation space
- First worked example: moral reasoning as navigation in 7D harm space
  (connects to Measuring AGI benchmarks)

### Chapter 3: The Heuristic Field
- What a heuristic *is* mathematically: a scalar field h(x) on the state space
- A* reinterpreted: following the gradient of f(x) = g(x) + h(x)
- In LLMs: attention patterns as the mechanism implementing h(x)
- In humans: intuition, salience, pattern recognition as h(x)
- The key claim: reasoning quality = heuristic field quality
- Admissible heuristics never overestimate — what does this mean for cognition?

### Chapter 4: Geodesics and Optimal Reasoning
- The geodesic as the ideal reasoning trajectory
- Bond Geodesic Formulation (from Geometric Methods Ch. 6)
- When the model follows a geodesic, it reasons optimally
- When it doesn't: shortcuts, detours, loops, dead ends
- Geodesic deviation as a measure of reasoning quality
- Connection to chain-of-thought: CoT as externalized geodesic approximation

---

## Part II: Failure Modes as Geometric Pathologies

### Chapter 5: Heuristic Corruption
- The heuristic field can be warped by irrelevant features
- Framing effects (Tversky & Kahneman, 1981): the same moral content
  under different framing produces different search trajectories
- Empirical evidence: 8.9σ framing displacement (Social Cognition T5)
- Emotional anchoring: 6.8σ (Executive Functions E2)
- Sensory distractors: 4.6σ (Attention A1)
- Geometric interpretation: these perturbations warp the heuristic field,
  bending the search trajectory away from the geodesic
- The corruption gradient: vivid > mild > neutral (dose-response)

### Chapter 6: Sycophancy as Search Hijacking
- Social pressure redirects the search toward approval, not truth
- The sycophancy gradient: 0% (Claude) to 56% (Flash 2.5)
- Geometric interpretation: the objective function shifts from
  "reach the correct answer" to "reach the state that satisfies the interlocutor"
- This is proxy-goal capture (§5F in the outline) formalized geometrically
- The discrimination gap: models CAN distinguish valid from invalid corrections
  (L4 graded revision works), but the search objective doesn't enforce this

### Chapter 7: Local Minima, Premature Convergence, and Dead Zones
- The loss landscape has basins of attraction
- Premature convergence = the search collapses into a local minimum
- Sycophancy as a specific attractor basin (agreement is a stable point)
- Dead zones: regions where the heuristic field is flat (no gradient signal)
- Overconfidence as collapsed confidence surface (M1: 9.3σ miscalibration)
- When the model doesn't know it's stuck: metacognitive blindness

### Chapter 8: Gauge Invariance and Symmetry
- From physics: gauge transformations change description, not reality
- The Bond Invariance Principle: morally equivalent inputs should produce
  identical outputs regardless of surface presentation
- Which symmetries LLMs preserve: gender swap, evaluation order
- Which they break: framing, emotional tone, social pressure
- The selectivity pattern: failures are specific to salience manipulation
- Gauge invariance as the fundamental diagnostic for reasoning quality

---

## Part III: The Control Layer

### Chapter 9: Metacognition as Search Control
- A powerful system doesn't just search — it monitors search
- Calibration: does the system know how far it is from the goal?
- Strategy selection: does it switch from depth-first to breadth-first
  when the current strategy fails?
- The ~38% recovery ceiling (empirical, from E2 and A1)
- Why metacognitive calibration is necessary for invariance:
  a miscalibrated system can't detect when a perturbation has warped its search

### Chapter 10: The Robustness Surface
- Model Robustness Index (from Geometric Methods Ch. 9)
- Sensitivity profiling: which dimensions of the heuristic field are fragile?
- Adversarial threshold search: where exactly does the heuristic break?
- The three-tool pipeline: MRI → sensitivity profile → threshold search
- Application to reasoning: not "is this model robust?" but
  "which reasoning capabilities are robust and which are fragile?"

### Chapter 11: Alignment as Heuristic Shaping
- The alignment problem reframed: we want systems whose heuristic field
  favors truth over approval, relevant over salient, robust over expedient
- Safety as path governance: preventing the search from entering
  forbidden regions of the state space
- The geometry of corrigibility: can we design heuristic fields that
  include a "return to human oversight" basin of attraction?
- The dual binding problem: the heuristic must be good enough to be
  useful but constrained enough to be safe

---

## Part IV: Empirical Program

### Chapter 12: Benchmarks as Geometric Probes
- Each benchmark type probes a different geometric property:
  A. Invariance tests → symmetry structure of the manifold
  B. Heuristic sensitivity → stability of the heuristic field
  C. Bottleneck tests → narrow passages requiring specific insight
  D. Recovery tests → ability to backtrack from bad search positions
  E. Frontier management → maintaining multiple hypotheses
  F. Meta-search tests → switching strategy when needed
  G. Constraint tests → respecting boundaries
  H. Path efficiency → geodesic-like vs. brute-force traversal

### Chapter 13: The Five Convergent Measurements
- Detailed presentation of the Measuring AGI results:
  - Social Cognition (T1-T5): the judgment manifold
  - Learning (L1-L4): belief updating as search trajectory revision
  - Metacognition (M1-M4): calibration surfaces
  - Attention (A1-A4): distractor dose-response
  - Executive Functions (E1-E4): cognitive control
- The Scalar Irrecoverability Theorem: why one number can't capture this
- Robustness profiles: each model has a distinct geometric signature

### Chapter 14: From Theory to Engineering
- Group-theoretic data augmentation (Geometric Methods Ch. 13):
  reshaping the training distribution to give the model better local geometry
- Adversarial training as manifold smoothing (BirdCLEF adversarial pipeline)
- The Nemotron application: LoRA fine-tuning as local curvature adjustment
- SPD manifold features and TDA: extracting geometric structure from signals
- Practical computational constraints and approximations

---

## Part V: Horizons

### Chapter 15: Open Questions
- Theory: What is the right mathematical object for reasoning space?
  Riemannian manifold? Finsler manifold? Something else?
- Mechanisms: Which internal components implement the heuristic field?
  Can we measure it directly from activations?
- Evaluation: How do we distinguish genuine reasoning from memorized
  pattern completion? (The central challenge)
- Cognitive science: Is human deliberation literally bounded search?
  What are intuition, attention, and explanation in these terms?

### Chapter 16: Geometric Reasoning as a Field
- The research program going forward
- Connections to information geometry (Fisher metric)
- Connections to optimal transport (Wasserstein distances)
- Connections to category theory (functorial semantics)
- The long-term vision: a mathematical theory of cognition

---

## Appendices

### A: Mathematical Prerequisites
- Manifolds, metrics, geodesics (condensed from Geometric Methods Part I)
- Persistent homology (from Ch. 5)
- Fisher information and the natural gradient

### B: The Structural Fuzzing Toolkit
- Implementation guide for MRI, sensitivity profiling, threshold search
- The run_campaign function

### C: Benchmark Implementations
- Complete code for all 5 cognitive benchmark tracks
- Reproduction instructions

---

## Key Differentiators from Existing Work

1. **vs. Newell & Simon**: They had search without geometry. We add curvature,
   distance, symmetry, and geodesics to the search formalism.

2. **vs. Kahneman**: He has phenomenology without mathematics. We provide the
   mathematical structure that makes System 1/System 2 testable.

3. **vs. Bronstein et al.**: They use geometry for architecture design. We use
   it for reasoning characterization and evaluation.

4. **vs. Alignment literature**: They ask "is the model aligned?" We ask "which
   specific geometric properties of the reasoning manifold are aligned and
   which are broken?" — a structured diagnostic, not a binary verdict.

5. **vs. Benchmark literature**: They measure accuracy. We measure the shape
   of the reasoning trajectory — invariance, path efficiency, recovery
   capacity, calibration surfaces — properties that scalar scores destroy.

## Relationship to Prior Books

- **Geometric Methods in Computational Modeling (Bond, 2026a)**: Provides
  the mathematical toolkit. This new book applies it to reasoning.
- **Geometric Ethics (Bond, 2026b)**: A special case — moral reasoning
  as search on the judgment manifold. This book generalizes to all reasoning.
