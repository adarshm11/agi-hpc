<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/atlas_mark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/brand/atlas_mark_light.svg">
    <img src="docs/brand/atlas_mark.svg" width="280" alt="Atlas AI — sphere + Eris apple">
  </picture>
</p>

# Atlas AI

### Neuroscience-inspired cognitive architecture with distributed compute

![CI](https://github.com/ahb-sjsu/agi-hpc/actions/workflows/ci.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Brand assets:** [erisml.org/brand/](https://erisml.org/brand/) | local: [`docs/brand/`](docs/brand/)

Atlas AI is a **production cognitive architecture** that mirrors the vertebrate brain's cortical/subcortical split: high-level reasoning runs on frontier cloud LLMs (the "cortex"), while fast pattern learning and procedural memory run on local GPUs (the "subcortical brain"). The two tiers are coordinated by NATS JetStream as a global workspace (Baars, 1988), with persistent memory in PostgreSQL + pgvector.

The system implements a Freudian psychoanalytic model where three agents (Id, Superego, Ego) negotiate decisions through structured debate, and includes an autonomous learning loop that improves its own problem-solving strategies over time.

---

## Architecture: Cortical / Subcortical Split

```
                        ╔══════════════════════════════════════════╗
                        ║        CORTEX (NRP Nautilus Cloud)        ║
                        ║     Frontier LLMs via Managed API         ║
                        ╠══════════════════════════════════════════╣
                        ║                                          ║
                        ║  Id (Kirk)        Qwen 3.5 397B          ║
                        ║  Superego (Spock) Gemma 4                ║
                        ║  Ego (McCoy)      Kimi K2.5 1T           ║
                        ║  Divine Council   7 advocate agents       ║
                        ║  ARC Scientist    self-improving solver   ║
                        ║                                          ║
                        ║  + GPU burst pods (A100, L4, L40)        ║
                        ╚════════════════════╤═════════════════════╝
                                             │ NATS leaf :7422
                        ╔════════════════════╧═════════════════════╗
                        ║    GLOBAL WORKSPACE (NATS JetStream)      ║
                        ║    :4222 — subjects: agi.*                ║
                        ╚════════════════════╤═════════════════════╝
                                             │
          ┌──────────────────────────────────┼──────────────────────────────┐
          │                                  │                              │
╔═════════╧══════════╗  ╔════════════════════╧═══╗  ╔══════════════════════╗
║ SUBCORTICAL BRAIN  ║  ║    MEMORY (PostgreSQL)  ║  ║   BRAINSTEM          ║
║ 2x GV100 32GB     ║  ║                         ║  ║                      ║
║                    ║  ║  L1  Dream-wiki (1.5x)  ║  ║  Thermal Guardian    ║
║  Conv training     ║  ║  L2  Hand-written wiki  ║  ║  Watchdog            ║
║  Pattern learning  ║  ║  L3  pgvector 3.3M vecs ║  ║  Telemetry           ║
║  Procedural memory ║  ║  L4  Full-text search   ║  ║  Caddy proxy         ║
║  A* search         ║  ║  L5  Live world data    ║  ║  OAuth2              ║
║  Gradient descent  ║  ║                         ║  ║  Backup              ║
╚════════════════════╝  ╚═════════════════════════╝  ╚══════════════════════╝
```

### Neuroscience mapping

| Brain Region | Function | Atlas Implementation |
|---|---|---|
| **Prefrontal cortex** | Planning, reasoning, ethics | NRP frontier LLMs (Qwen 397B, Kimi 1T, Gemma 4) |
| **Limbic system** | Motivation, reward, learning signals | Security Radar (pass/fail), episodic memory of what worked |
| **Basal ganglia** | Habit formation, procedural memory | Conv training on local GPUs, strategy weight updates |
| **Cerebellum** | Motor learning, pattern refinement | A* search, gradient descent, ONNX compilation |
| **Brainstem** | Autonomic functions | Thermal guardian, watchdog, NATS fabric, backup |
| **Global workspace** | Consciousness, integration | NATS JetStream event fabric (Baars 1988) |

---

## The Divine Council

Three psychoanalytic agents negotiate decisions through structured debate:

| Agent | Role | Model | Where |
|-------|------|-------|-------|
| **Id** (Kirk) | System 1 -- fast, intuitive | Qwen 3.5 397B | NRP managed API |
| **Superego** (Spock) | System 2 -- deliberative, ethical | Gemma 4 | NRP managed API |
| **Ego** (McCoy) | Arbitrator -- structured debate | Kimi K2.5 1T | NRP managed API |

The Ego hosts 7 concurrent advocate roles (Judge, Advocate, Synthesizer, Ethicist, Historian, Futurist, Pragmatist) that deliberate on decisions requiring multi-perspective analysis.

**Fallback:** When NRP is unavailable, the Council falls back to local llama.cpp on the GV100s (Qwen 3 32B, Gemma 4 31B, Gemma 4 26B-A4B MoE).

## Autonomous Learning

The **ARC Scientist** (`src/agi/autonomous/arc_scientist.py`) implements a closed-loop scientific reasoning cycle:

1. **Observe** -- Pick an unsolved task, study its patterns
2. **Hypothesize** -- Form a theory about the transformation (via LLM)
3. **Experiment** -- Generate candidate Python transforms
4. **Evaluate** -- Security Radar: verify on ALL examples (any failure = kill)
5. **Learn** -- Store what worked and what failed in episodic memory
6. **Reflect** -- Ask LLM to diagnose WHY a transform failed
7. **Adapt** -- Thompson sampling shifts strategy weights based on evidence
8. **Repeat**

The system genuinely learns: it tracks which prompt framings, task patterns, and reasoning strategies succeed or fail, and shifts its approach over time. Failed attempts include failure reflections that inform future attempts on the same task.

---

## Infrastructure

| Component | Description |
|-----------|-------------|
| **NATS JetStream** | Global workspace at `:4222`. Leaf node at `:7422` bridges to NRP. |
| **PostgreSQL + pgvector** | 3.3M PCA-384 vectors. 5-tier retrieval (L1-L5). |
| **RAG Server** | Flask at `:8081` with dual-hemisphere proxy + search. |
| **Caddy** | Reverse proxy + OAuth2. Serves `atlas-sjsu.duckdns.org`. |
| **Telemetry** | At `:8085`. NATS stats, NRP pod metrics, live GPU/VRAM monitoring. |
| **Dashboard** | Real-time at `/schematic.html`: NATS topology, NRP pods, memory tiers. |
| **Thermal Guardian** | CPU temp monitoring (82C/100C thresholds). |
| **Watchdog** | Health checks and automatic service restarts. |

## NRP Nautilus Integration

NATS leaf node bridges Atlas to [NRP Nautilus](https://nrp.ai) (namespace `ssu-atlas-ai`):

- **Managed LLM API** -- Kimi K2.5, Qwen3.5 397B, Gemma 4 at zero cost
- **GPU burst pods** -- Conv training on A100 80GB, L4 24GB, L40 48GB via `nvidia.com/gpu.product` affinity
- **CPU pods** -- Coverage-guided prompt fuzzer (AFL-style)
- **Live monitoring** -- GPU model, VRAM used/total, utilization % via `kubectl exec nvidia-smi`
- **NRP LLM provider** -- `src/agi/core/llm/nrp_provider.py` routes Divine Council to managed API

## Safety

Three-layer safety architecture:

| Layer | Latency | Function |
|-------|---------|----------|
| **Reflex** | <100us | Emergency stops, thermal limits |
| **Tactical** | 10-100ms | ErisML ethical evaluation, Bond Index |
| **Strategic** | >100ms | Policy enforcement, human oversight |

ErisML provides mathematically grounded ethical reasoning with Hohfeldian analysis and hash-chained decision proofs.

---

## Project Layout

```
agi-hpc/
├── src/agi/
│   ├── autonomous/          # Self-improving agents (ARC Scientist)
│   ├── core/                # gRPC, event fabric (NATS/ZMQ/UCX), DHT, LLM providers
│   ├── reasoning/           # Divine Council, debate, NATS service
│   ├── lh/                  # Left hemisphere: planning, metacognition
│   ├── rh/                  # Right hemisphere: perception, world model, control
│   ├── memory/              # Episodic, semantic, procedural, knowledge
│   ├── safety/              # 3-layer safety, ErisML, privilege gates
│   ├── metacognition/       # Ego monitor, consistency, anomaly detection
│   ├── dreaming/            # Memory consolidation via wiki synthesis
│   ├── training/            # Dungeon Master, gym environment, curriculum
│   ├── attention/           # Distractor detection and filtering
│   ├── thermal/             # Thermal management and job queue
│   ├── integration/         # Cross-subsystem orchestration
│   ├── env/                 # Gymnasium-compatible (MuJoCo/Unity)
│   └── meta/                # LLM-based metacognitive reflection
│
├── configs/                 # Service YAML (lh, rh, memory, safety, nrp_llm)
├── deploy/systemd/          # 19 service units under atlas.target
├── proto/                   # Protocol Buffer definitions
├── infra/
│   ├── hpc/                 # Apptainer, Slurm, Docker
│   └── local/               # Docker Compose, dashboards
├── scripts/                 # Watchdog, telemetry, utilities
├── tests/                   # Unit and integration tests
├── docs/                    # Architecture, operations, sprint plans
└── .github/workflows/       # CI/CD pipeline
```

## Hardware

| Resource | Specification |
|----------|---------------|
| **Atlas** | HP Z840, 2x Xeon E5-2690v3 (48 threads), 251GB RAM |
| **Local GPUs** | 2x Quadro GV100 32GB (Volta) -- subcortical compute |
| **Storage** | 15TB at `/archive` |
| **Network** | Tailscale VPN at `100.68.134.21` |
| **NRP Cortex** | A100 80GB, L4 24GB, L40 48GB across 100+ university nodes |
| **NRP LLMs** | Qwen 3.5 397B, Kimi K2.5 1T, Gemma 4 (managed, zero cost) |

---

## Quickstart

```bash
git clone https://github.com/ahb-sjsu/agi-hpc.git
cd agi-hpc
pip install -e ".[dev]"

# Local development (no network)
export AGI_FABRIC_MODE=local
python src/agi/lh/service.py

# Production (NATS + NRP)
export AGI_FABRIC_MODE=nats
export NRP_LLM_TOKEN=<your-token>
python src/agi/autonomous/arc_scientist.py --task-dir /path/to/tasks
```

For production deployment, see [`docs/ATLAS_OPERATIONS.md`](docs/ATLAS_OPERATIONS.md).

---

## Dev Teams

- Cognitive Architecture (LH/RH)
- Memory Systems
- EventBus and Messaging
- HPC Deployment
- Maritime Digital Twin
- Unity Simulation
- Evaluation and Metrics

---

## License

MIT (c) 2025 Andrew Bond
