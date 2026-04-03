# Atlas AI — AGI-HPC Cognitive Architecture

A local-first, safety-gated cognitive architecture running on consumer hardware.
Built by Andrew H. Bond. Live at [atlas-sjsu.duckdns.org](https://atlas-sjsu.duckdns.org).

## Architecture

```
User → Caddy (HTTPS) → oauth2-proxy (Google Auth) → RAG Server (8081)
  → Hybrid Search (BM25 + Dense + HyDE) + Repo-aware filtering
  → Dual-Hemisphere Debate:
      GPU 0: Spock (Gemma 4 31B) — analytical, precise
      GPU 1: Kirk (Qwen 3 32B) — creative, intuitive
  → 4-round debate → Kirk synthesizes → Safety Gateway checks output
  → Confidence metric from hemisphere disagreement
  → Response to user
```

## Subsystems (10/10 implemented)

| # | Subsystem | Implementation | Port | Status |
|---|-----------|---------------|------|--------|
| 0 | Event Fabric | NATS JetStream | 4222 | Active |
| 1 | Left Hemisphere | Gemma 4 31B via llama.cpp | 8080 | Active |
| 2 | Memory | PostgreSQL + pgvector + SQLite | 50300 | Active |
| 3 | Safety Gateway | ErisML DEME 3-layer pipeline | 50055 | Active |
| 4 | Right Hemisphere | Qwen 3 32B via llama.cpp | 8082 | Active |
| 5 | Metacognition | Monitor + Reflector + Adjuster | — | Active |
| 5 | Environment | System + Repo sensors | — | Active |
| 6 | DHT Registry | Service discovery + config store | — | Active |
| 7 | Integration | Query routing + debate orchestration | 8081 | Active |
| 7 | LLM Layer | LLMClient + InferenceConfig + Templates | — | Active |

## Hardware (HP Z840 Workstation)

- CPU: 2x Xeon E5-2690 v3 (48 threads)
- RAM: 256 GB DDR4
- GPU 0: Quadro GV100 32GB (Spock)
- GPU 1: Quadro GV100 32GB (Kirk)
- Storage: 1.8TB NVMe + 916GB RAID1 + 15TB RAID5
- Location: Bel Marin Keys, Novato, CA

## Knowledge Base

| Dataset | Size | Records |
|---------|------|---------|
| GitHub repos (27) | pgvector | 44K+ chunks |
| Ethics corpora (7 traditions) | pgvector | 102K+ chunks |
| Publications catalog | PostgreSQL FTS | 824K entries |
| Wikipedia | 24 GB | Full English dump |
| Project Gutenberg | Syncing | ~70K books |
| arXiv CS | 1.7 GB | Metadata |
| PostGIS | — | 258 countries + cities |

## Memory Architecture (L1-L5)

| Tier | Medium | Latency | Contents |
|------|--------|---------|----------|
| L1 | VRAM (KV cache) | <1ms | Current conversation |
| L2 | RAM | ~1ms | Hot embeddings, recent sessions |
| L3 | SSD (PostgreSQL) | ~5ms | Episodic, semantic, procedural |
| L4 | HDD (RAID5) | ~50ms | Full repos, old episodes |
| L5 | Network | ~100ms+ | GitHub, web, BitTorrent |

## Retrieval Pipeline

1. **HyDE**: LLM generates hypothetical answer, embed that
2. **BM25**: Full-text search via PostgreSQL tsvector
3. **Dense**: pgvector cosine similarity on BGE-M3 embeddings
4. **RRF**: Reciprocal Rank Fusion merges results
5. **Repo boost**: Named repos get priority
6. **Hemisphere-aware**: Spock gets precise results, Kirk gets diverse

## Safety Pipeline (ErisML DEME)

Three-layer firewall on every interaction:
- **Reflex** (<1ms): PII detection, prompt injection, content policy
- **Tactical** (~100ms): MoralVector assessment via Ethics Modules
- **Strategic**: SHA-256 hash-chained decision proofs, audit trail

Ethics grounded in 3,300 years of cross-civilizational moral texts:
Greco-Roman, Jewish, Buddhist, Islamic, Chinese, UN Human Rights, American Advice

## Dual-Hemisphere Debate

Every query triggers a 4-round debate:
1. Spock + Kirk answer in parallel (both GPUs active)
2. Spock challenges Kirk + Kirk challenges Spock (parallel)
3. Kirk synthesizes as captain
4. Confidence measured from hemisphere disagreement

## Training

- **AtlasGym**: 5 environments (ethics, reasoning, coding, debate, memory)
- **Curriculum**: Auto-promotes at >80%, demotes at <40%
- **Overnight**: Cron switches to training mode midnight-8am PST
- **Unsloth**: Gemma 4 E4B ethics fine-tune ready

## Web UI

- Chat: atlas-sjsu.duckdns.org
- Dashboard: /schematic.html (GPU gauges, job monitor, training metrics)
- Events: /events.html (NATS activity, subsystem events)
- Mobile responsive, Google OAuth

## Quick Start

```bash
# Start all services
bash scripts/start_atlas.sh

# Health check
bash scripts/start_atlas.sh --health

# Stop all
bash scripts/start_atlas.sh --stop

# Run training
python -m agi.training.runner --env ethics --level 2 --episodes 10
```

## Documentation

| Document | Contents |
|----------|----------|
| [atlas-agi-hpc-implementation-plan.md](atlas-agi-hpc-implementation-plan.md) | Full implementation plan (Phases 0-7) |
| [phase7-metacognitive-loop.md](phase7-metacognitive-loop.md) | Metacognitive self-improvement roadmap |
| [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) | Original AGI-HPC architecture |
| [MASTER_IMPLEMENTATION_PLAN.md](MASTER_IMPLEMENTATION_PLAN.md) | Sprint-based development plan |

## Competition

Entered in **Gemma 4 Good Hackathon** (Kaggle, deadline May 18 2026):
- Main Track ($50K), Safety & Trust ($10K), llama.cpp ($10K), Unsloth ($10K)
- Live demo, video, writeup, code repo

## License

AGI-HPC Responsible AI License v1.0
