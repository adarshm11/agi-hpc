#!/bin/bash
# deploy_to_atlas.sh — Deploy latest code to Atlas workstation
#
# Can be run manually or by CI/CD pipeline.
# Pulls latest code, copies dashboard HTML, restarts services.
#
# Prerequisites:
#   - SSH access to Atlas (via Tailscale or LAN)
#   - Git repo configured on Atlas at /home/claude/agi-hpc
#
# Usage:
#   bash scripts/deploy_to_atlas.sh                    # uses Tailscale
#   bash scripts/deploy_to_atlas.sh --host 192.168.0.7 # use LAN IP
#   ATLAS_HOST=100.68.134.21 bash scripts/deploy_to_atlas.sh

set -euo pipefail

ATLAS_HOST="${ATLAS_HOST:-100.68.134.21}"  # Tailscale IP
ATLAS_USER="${ATLAS_USER:-claude}"
DEPLOY_PATH="/home/claude/agi-hpc"
STATIC_PATH="/home/claude/atlas-chat"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) ATLAS_HOST="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

log "=== Deploying to Atlas ($ATLAS_HOST) ==="

# Step 1: Pull latest code
log "Pulling latest code..."
ssh "$ATLAS_USER@$ATLAS_HOST" "cd $DEPLOY_PATH && git fetch origin main && git reset --hard origin/main"

# Step 2: Ensure dashboard HTML symlinks point at the canonical git tree
# (not a cp — cp -f through a symlink writes back into the repo and causes
# silent working-tree drift; see post-mortem in commit message.)
log "Refreshing dashboard HTML symlinks..."
ssh "$ATLAS_USER@$ATLAS_HOST" << EOF
for f in schematic.html index.html events.html erebus.html; do
  src="$DEPLOY_PATH/infra/local/atlas-chat/\$f"
  dst="$STATIC_PATH/\$f"
  if [ -f "\$src" ]; then
    ln -sfn "\$src" "\$dst"
  fi
done
EOF

# Step 3: Install/update Python package
log "Installing Python package..."
ssh "$ATLAS_USER@$ATLAS_HOST" "cd $DEPLOY_PATH && /home/claude/env/bin/pip install -e '.[nats]' --quiet"

# Step 4: Restart services that pick up code changes
log "Restarting RAG server..."
ssh "$ATLAS_USER@$ATLAS_HOST" "sudo systemctl restart atlas-rag-server 2>/dev/null || \
    (tmux kill-session -t rag 2>/dev/null; sleep 2; \
     tmux new-session -d -s rag 'CUDA_VISIBLE_DEVICES= /home/claude/env/bin/python3 $DEPLOY_PATH/atlas-rag-server.py 2>&1 | tee /tmp/atlas/rag_server.log')"

log "Restarting telemetry server..."
ssh "$ATLAS_USER@$ATLAS_HOST" "sudo systemctl restart atlas-telemetry 2>/dev/null || \
    (tmux kill-session -t telemetry 2>/dev/null; sleep 1; \
     tmux new-session -d -s telemetry '/home/claude/env/bin/python3 $DEPLOY_PATH/scripts/telemetry_server.py 2>&1 | tee /tmp/atlas/telemetry.log')"

# Step 5: Verify
log "Waiting for services..."
sleep 10

log "Verifying..."
ssh "$ATLAS_USER@$ATLAS_HOST" << 'VERIFY'
for svc in "RAG:8081:/api/search-status" "Telemetry:8085:/health" "Superego:8080:/health" "Id:8082:/health"; do
    IFS=: read -r name port path <<< "$svc"
    status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}${path}" --connect-timeout 3 || echo "000")
    if [ "$status" = "200" ]; then
        echo "  $name ($port): OK"
    else
        echo "  $name ($port): $status (may still be loading)"
    fi
done
VERIFY

log "=== Deploy complete ==="
