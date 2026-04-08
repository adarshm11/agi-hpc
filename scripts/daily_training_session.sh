#!/bin/bash
# daily_training_session.sh — Ego-driven DM training + dreaming nap
#
# The AGI's daily routine:
#   Phase 1: DM Training — Ego generates scenarios, Id/Superego debate
#   Phase 2: Nap — Dreaming consolidation of training episodes into wiki
#
# Cognitive science grounding:
#   - Hebbian learning: repeated practice strengthens connections
#   - Hippocampal replay: consolidation during sleep improves retention
#   - Default Mode Network: idle processing integrates new knowledge
#
# Usage:
#   bash scripts/daily_training_session.sh                # default 20 episodes
#   bash scripts/daily_training_session.sh --episodes 10  # custom count
#   bash scripts/daily_training_session.sh --skip-nap     # training only
#
# Cron (via systemd timer):
#   10:00 AM UTC — training session
#   After training — dreaming nap automatically triggered

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="/home/claude/env"
PYTHON="$VENV/bin/python3"
LOG_DIR="/tmp/atlas-training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Defaults
EPISODES=20
DIFFICULTY=2
SKIP_NAP=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes) EPISODES="$2"; shift 2 ;;
        --difficulty) DIFFICULTY="$2"; shift 2 ;;
        --skip-nap) SKIP_NAP=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/session_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ─── Pre-flight checks ────────────────────────────────────────────────
log "=== Atlas Daily Training Session ==="
log "Episodes: $EPISODES | Difficulty: $DIFFICULTY | Nap: $([ "$SKIP_NAP" = true ] && echo 'skip' || echo 'yes')"

# Check Ego (CPU model) is online
EGO_OK=$(curl -s http://localhost:8084/health 2>/dev/null | grep -c ok || true)
if [ "$EGO_OK" != "1" ]; then
    log "ERROR: Ego (Gemma 4 E4B, port 8084) is not online. Aborting."
    exit 1
fi

# Check at least one hemisphere is online
SPOCK_OK=$(curl -s http://localhost:8080/health 2>/dev/null | grep -c ok || true)
KIRK_OK=$(curl -s http://localhost:8082/health 2>/dev/null | grep -c ok || true)
if [ "$SPOCK_OK" != "1" ] && [ "$KIRK_OK" != "1" ]; then
    log "ERROR: No hemispheres online. Need at least Superego or Id. Aborting."
    exit 1
fi

log "Pre-flight: Ego=OK Superego=$([ "$SPOCK_OK" = "1" ] && echo 'OK' || echo 'OFFLINE') Id=$([ "$KIRK_OK" = "1" ] && echo 'OK' || echo 'OFFLINE')"

# ─── Phase 1: DM Training ────────────────────────────────────────────
log ""
log "Phase 1: DM Training ($EPISODES episodes, difficulty=$DIFFICULTY)"
log "─────────────────────────────────────────────────────────────────"

TRAIN_START=$(date +%s)

cd "$PROJECT_DIR"
$PYTHON -m agi.training.dungeon_master \
    --episodes "$EPISODES" \
    --difficulty "$DIFFICULTY" \
    2>&1 | tee -a "$LOG_FILE"

TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
log "Training complete: ${TRAIN_DURATION}s"

# ─── Phase 2: Dreaming Nap ───────────────────────────────────────────
if [ "$SKIP_NAP" = true ]; then
    log ""
    log "Phase 2: Nap skipped (--skip-nap)"
else
    log ""
    log "Phase 2: Dreaming Nap (consolidating training episodes)"
    log "─────────────────────────────────────────────────────────"

    NAP_START=$(date +%s)

    # Trigger dreaming consolidation
    # The consolidator fetches all unconsolidated episodes (including
    # the training episodes just created) and synthesizes wiki articles.
    $PYTHON -c "
import asyncio
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
from agi.dreaming.consolidator import ConsolidatorConfig, MemoryConsolidator
consolidator = MemoryConsolidator(ConsolidatorConfig())
result = asyncio.run(consolidator.run_cycle())
print(f'Nap complete: {result.articles_created} articles created, '
      f'{result.articles_updated} updated, '
      f'{result.dream_insights} insights, '
      f'{result.episodes_processed} episodes processed')
" 2>&1 | tee -a "$LOG_FILE"

    NAP_END=$(date +%s)
    NAP_DURATION=$((NAP_END - NAP_START))
    log "Nap complete: ${NAP_DURATION}s"
fi

# ─── Summary ─────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TRAIN_START))
log ""
log "=== Session Complete ==="
log "Training: ${TRAIN_DURATION}s | Nap: ${NAP_DURATION:-0}s | Total: ${TOTAL_DURATION}s"
log "Log: $LOG_FILE"
