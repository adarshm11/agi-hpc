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

## TurboQuant KV Cache Compression

Adapted from Theory Radar's TurboBeam implementation, which uses the
Zandieh et al. (ICLR 2026) PolarQuant + QJL algorithm for sub-linear
memory inference.

**Algorithm**: Random rotation (QR) maps each head-dim vector onto the
unit hypersphere, then an optimal Lloyd-Max scalar quantizer compresses
each coordinate to b bits (2, 3, or 4).  Only uint8 indices and a
per-vector L2 norm are stored.

**Memory savings (Gemma 4 27B, fp16 baseline)**:

Current (uint8 storage, ~2x):

| Context | Bits | Original | Compressed | Ratio | Saved |
|---------|------|----------|------------|-------|-------|
| 8,192   | 3    | 4.500 GB | 2.285 GB  | 1.97x | 2.215 GB |
| 16,384  | 3    | 9.000 GB | 4.570 GB  | 1.97x | 4.430 GB |
| 32,768  | 3    | 18.00 GB | 9.141 GB  | 1.97x | 8.859 GB |

With bit-packing (future optimisation, ~5x):

| Context | Bits | Original | Compressed | Ratio | Saved |
|---------|------|----------|------------|-------|-------|
| 8,192   | 3    | 4.500 GB | 0.879 GB  | 5.12x | 3.621 GB |
| 16,384  | 3    | 9.000 GB | 1.758 GB  | 5.12x | 7.242 GB |
| 32,768  | 3    | 18.00 GB | 3.516 GB  | 5.12x | 14.48 GB |

Source: `src/agi/meta/llm/turboquant_kv.py`
Benchmark: `scripts/benchmark_turboquant_kv.py`
Tests: `tests/unit/test_turboquant_kv.py`

### llama.cpp Integration Options

**Option A: Python KV cache wrapper (recommended for prototyping)**
Wrap `llama-server` with a Python process that intercepts KV cache
tensors between layers.  On each forward pass, compress old KV entries
(beyond a sliding window) using `TurboQuantKV.compress()`, freeing VRAM.
Decompress on-demand when attention reaches those positions.  This
requires exposing KV cache tensors via the llama.cpp Python bindings
(`llama-cpp-python`), which supports `kv_cache_view()`.

**Option B: Custom CUDA kernel linked into llama.cpp**
Write a CUDA kernel that performs the rotation + quantization in-place
within the llama.cpp KV cache management code (`llama-kv-cache.cpp`).
This avoids Python overhead and integrates directly with the inference
loop.  Requires modifying `ggml-cuda` to add the quantization as a new
operation type.  Highest performance but most engineering effort.

**Option C: External cache manager (eviction-based)**
Run `TurboQuantKV` as a sidecar process that manages a compressed L2
cache on host RAM.  When VRAM KV cache is full, evict oldest entries
to the compressed store.  On cache miss, decompress and reload into
VRAM.  This is a form of memory tiering (L1: VRAM, L2: compressed RAM)
that fits the existing AGI-HPC memory architecture (see Memory Tiers).

For the Gemma 4 Good Hackathon, Option A is the fastest path to a
working demo.  Option C aligns best with the AGI-HPC L1-L5 memory
hierarchy.

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
