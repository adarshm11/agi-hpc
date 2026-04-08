# Atlas AI Setup Guide

## Quick Start (Development)

```bash
# Clone
git clone https://github.com/ahb-sjsu/atlas-ai.git
cd atlas-ai

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
pip install batch-probe  # REQUIRED for thermal protection

# Run tests (337 passing)
pytest tests/unit/

# Run safety demo (works without external services)
python scripts/demo_safety_pipeline.py

# Run DM training with retrospective + gap detection
python -m agi.training.dungeon_master --episodes 10 --retrospective

# Run debate benchmark (uses batch-probe for thermal safety)
python scripts/benchmark_debate.py --questions 5 --dry-run
```

## Full Deployment (Atlas Workstation)

### Prerequisites

- Ubuntu 24.04 LTS
- NVIDIA drivers 570+ with CUDA 12.8
- PostgreSQL 16 with pgvector extension
- NATS Server with JetStream
- Python 3.12 with venv

### Hardware Requirements

| Component | Minimum | Atlas (Reference) |
|-----------|---------|-------------------|
| GPU 0 | 24GB VRAM | Quadro GV100 32GB |
| GPU 1 | 24GB VRAM | Quadro GV100 32GB |
| CPU | 24 threads | 2x Xeon E5-2690 v3 (48 threads) |
| RAM | 64GB | 224GB DDR4 |
| Storage | 500GB SSD | 1.8TB NVMe + 15TB RAID5 |

### Models

Download and place in `/home/claude/models/`:

| Model | Size | Role | Source |
|-------|------|------|--------|
| Gemma 4 31B Q5_K_M | ~22GB | Superego (GPU 0) | HuggingFace |
| Qwen 3 32B Q5_K_M | ~22GB | Id (GPU 1) | HuggingFace |
| Gemma 4 E4B Q5_K_M | ~3GB | Ego/DM (CPU) | HuggingFace |

### Database Setup

```bash
sudo -u postgres createdb atlas
sudo -u postgres psql atlas -c "CREATE EXTENSION vector"
```

### Service Installation

```bash
# Install all systemd services + timers (recommended)
sudo bash deploy/systemd/install-services.sh

# Services use Freudian naming:
#   atlas-superego   Gemma 4 31B (GPU 0)
#   atlas-id         Qwen 3 32B (GPU 1)
#   atlas-ego        Gemma 4 E4B (CPU)
#   atlas-rag-server, atlas-nats, atlas-caddy, etc.

# Or start manually via tmux (legacy)
bash scripts/start_atlas.sh
```

### Verify

```bash
# Health check (systemd)
sudo systemctl status atlas-*

# Or tmux-based health check
bash scripts/start_atlas.sh --health

# Follow logs
sudo journalctl -u atlas-superego -f

# Run full test suite
pytest tests/

# Open dashboard
# https://atlas-sjsu.duckdns.org/schematic.html
```

## Architecture

See [ATLAS_AI_README.md](ATLAS_AI_README.md) for the full architecture diagram.

## Daily Cycle (Autonomous)

```
04:00 UTC  Backup -> PostgreSQL + wiki to RAID5
10:00 UTC  DM Training -> 20 scenarios, Id/Superego debate
12:00 UTC  Dreaming Nap -> consolidate episodes into wiki
02:00 UTC  Full dream cycle -> overnight consolidation
```

## Cognitive Science References

| Subsystem | Theory | Theorist | Year |
|-----------|--------|----------|------|
| Id/Ego/Superego | Structural Model | Freud | 1923 |
| System 1/2 | Dual Process Theory | Kahneman | 2011 |
| NATS Fabric | Global Workspace | Baars | 1988 |
| Safety Reflex | Somatic Markers | Damasio | 1994 |
| Dreaming | Default Mode Network | Raichle | 2001 |
| Certainty | Bayesian Brain | Knill & Pouget | 2004 |
| Disagreement | Metacognitive Monitoring | Flavell | 1979 |
| Episode->Wiki | Hippocampal Replay | Wilson & McNaughton | 1994 |
| Procedural | Hebbian Learning | Hebb | 1949 |
| Privilege Levels | Moral Development | Kohlberg | 1958 |
