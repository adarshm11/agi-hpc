# Changelog — Atlas AI / AGI-HPC

## April 2-3, 2026 — Atlas AI Launch

### Phase 0: Event Fabric
- NATS server v2.10.24 with JetStream (port 4222, monitoring 8222)
- `AGI_EVENTS` stream with `agi.>` wildcard, 1GB max, 7-day retention
- `Event` dataclass with serialization (src/agi/common/event.py)
- `NatsEventFabric` async wrapper (src/agi/core/events/nats_fabric.py)
- Subject hierarchy config for all 10 subsystems
- 3 integration tests

### Phase 1: Left Hemisphere + LLM Integration
- `LLMClient` async wrapper for OpenAI-compatible APIs
- `InferenceConfig` with LH preset (temp=0.3) and RH preset (temp=0.8)
- Jinja2 `PromptTemplateRegistry` with 4 built-in templates
- `RAGSearcher` class with pgvector semantic search
- `LHNatsService`: subscribes to agi.lh.request.*, publishes responses + COT traces

### Phase 2: Memory Subsystem
- Episodic memory (PostgreSQL: sessions + episodes with embeddings)
- Procedural memory (SQLite: learned behaviors with success tracking, 5 seed procedures)
- Semantic memory (wrapper around pgvector RAG)
- Memory Service Broker connected to NATS (5 subjects)
- 4 integration tests

### Phase 3: Safety Gateway
- `SafetyAdapter`: converts chat interactions to EthicalFacts
- `SafetyGateway` with DEME 3-layer pipeline:
  - Reflex (<20us): PII, prompt injection, dangerous content
  - Tactical: ErisML MoralVector assessment
  - Strategic: SHA-256 hash-chained decision proofs
- Input gate + Output gate via NATS
- 5 integration tests

### Phase 4: Right Hemisphere + Integration
- `RHNatsService`: Qwen 3 32B on GPU 1, creative prompts (temp=0.8)
- Integration Orchestrator: query classification, dual-hemisphere merge
- 4-round debate mode: parallel opening → mutual challenge → captain's call
- Session tracking with episodic memory

### Phase 5: Metacognition + Environment
- Monitor: latency p50/p95/p99, hemisphere ratio, veto rate, throughput
- Reflector: periodic self-assessment via LLM every 10 interactions
- Adjuster: auto-tunes max_tokens, safety thresholds, routing balance
- System sensor: GPU/CPU/RAM/disk polling via NATS
- Repo sensor: watches /archive for git changes

### Phase 6: DHT Service Registry
- `ServiceRegistry` with PostgreSQL-backed service_registry table
- HTTP health probing, stale detection (60s timeout)
- `ConfigStore` for versioned config distribution
- NATS service: agi.dht.{register,deregister,lookup,heartbeat}
- 8 services registered, 7 config entries seeded
- 5 integration tests

### Phase 7: Metacognitive Loop (in progress)
- Hemisphere disagreement metric (7.1) — building
- Adaptive temperature routing (7.2a) — building
- Full roadmap documented in phase7-metacognitive-loop.md

### Infrastructure
- Dual-hemisphere: Gemma 4 31B (Spock/GPU 0) + Qwen 3 32B (Kirk/GPU 1)
- HTTPS via Caddy + Let's Encrypt (atlas-sjsu.duckdns.org)
- Google OAuth via oauth2-proxy
- Custom chat UI with debate collapsible, thinking spinner, mobile responsive
- Operations dashboard with GPU gauges, sparklines, job monitoring
- Event log page with NATS activity
- Consistent nav breadcrumbs across all pages
- Visitor logging (PostgreSQL)
- start_atlas.sh with --health and --stop modes
- Cron: train midnight-8am, chat 8am-midnight
- download_monitor.py for parallel knowledge fetching

### RAG / Search
- Hybrid search: BM25 + dense vector + HyDE + Reciprocal Rank Fusion
- Repo-aware filtering (detects repo names in queries)
- 27 GitHub repos indexed (44K+ chunks)
- tsvector + GIN index for full-text search

### Knowledge Base
- 102K ethics chunks from 7 traditions, 37 languages, 3,300 years
- 824K publications catalog with full-text search
- Wikipedia English dump (24GB)
- Project Gutenberg (syncing, ~70K books)
- arXiv CS metadata (1.7GB)
- Common Crawl WAT sample (2.9GB)
- PostGIS: 258 countries, cities, coastlines
- Kaggle datasets: Dear Abby, Reddit AITA, Philosophy, Jeopardy, arXiv

### Training
- AtlasGym: 5 environments (ethics, reasoning, coding, debate, memory)
- Curriculum manager with auto-promotion/demotion
- Unsloth Gemma 4 E4B ethics fine-tune script ready
- Training results tracked in PostgreSQL + episodic memory

### Authorship Fix
- Corrected A.H. Bowers → A.H. Bond across 28 files in geometric book series

### Newhome Cleanup
- Deleted 444GB stale duplicates (90% → 39% usage)
- Fixed fstab to use UUIDs + nofail

### StoryWealth (Fallout 4)
- 71/928 Nexus mods downloaded via Violentmonkey userscript + feeder
- MO2 configured on Atlas via Steam/Proton
