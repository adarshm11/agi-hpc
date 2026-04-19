#!/usr/bin/env bash
# Preflight for atlas-scientist.service (Erebus ARC solver).
#
# Runs as ExecStartPre. Nonzero exit aborts the launch, so systemd's
# StartLimitBurst kicks in and stops a crash loop from burning GPU
# on an unhealthy host.
#
# Checks:
#   1. Sentinel file absent (operator hasn't disabled Erebus)
#   2. GPU not hung (nvidia-smi -L returns within 5s)
#   3. NATS reachable on localhost:4222
#   4. /archive has >= 5 GiB free (atomic writes need space)
#   5. No other arc_scientist instance already running

set -euo pipefail

LOG=/archive/neurogolf/preflight.log
log() { echo "$(date -Iseconds) preflight_erebus: $*" >> "$LOG"; }

SENTINEL=/archive/neurogolf/.erebus_disabled
if [[ -e "$SENTINEL" ]]; then
    log "ABORT sentinel file present ($SENTINEL)"
    exit 1
fi

if ! timeout 5 nvidia-smi -L >/dev/null 2>&1; then
    log "ABORT nvidia-smi hung or failed"
    exit 1
fi

if ! timeout 3 bash -c '</dev/tcp/127.0.0.1/4222' 2>/dev/null; then
    log "ABORT NATS not reachable on localhost:4222"
    exit 1
fi

avail_gb=$(df -BG /archive | awk 'NR==2 { gsub(/G/, "", $4); print $4 }')
if (( avail_gb < 5 )); then
    log "ABORT /archive has ${avail_gb} GiB free, need >= 5"
    exit 1
fi

# Single-instance guard. systemd sets MAINPID after ExecStart; during
# preflight we are the only instance systemd knows about, so any
# other arc_scientist process is a leftover (nohup, tmux, etc.)
if pgrep -f 'agi\.autonomous\.arc_scientist|agi/autonomous/arc_scientist\.py' >/dev/null; then
    log "ABORT another arc_scientist process is already running"
    exit 1
fi

log "OK preflight passed (gpu, nats, disk=${avail_gb}G, single-instance)"
exit 0
