#!/usr/bin/env bash
# sbatch wrapper — refuses direct use, points at polite-submit.
#
# Installed in the user notebook image at /usr/local/bin/sbatch with a
# higher PATH priority than the real Slurm sbatch. Students calling
# `sbatch job.sh` from their notebook get a polite nudge toward
# `polite-submit job.sh` instead.
#
# Rationale: Atlas is a shared 2-GPU cluster. Naive sbatch floods the
# queue and disrupts research workloads. polite-submit probes cluster
# state and backs off under load.
#
# Copyright (c) 2026 Andrew H. Bond. AGI-HPC Responsible AI License v1.0.

set -u

REAL_SBATCH="${REAL_SBATCH:-/usr/bin/sbatch}"

# Allow polite-submit itself to bypass the wrapper.
# polite-submit calls sbatch with the env variable POLITE_SUBMIT_BYPASS=1 set.
if [[ "${POLITE_SUBMIT_BYPASS:-}" == "1" ]]; then
    exec "$REAL_SBATCH" "$@"
fi

# Anything else: friendly refusal.
cat >&2 <<'EOF'
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Direct sbatch use is not allowed on Atlas.                         │
│                                                                     │
│  Please submit via polite-submit:                                   │
│                                                                     │
│      polite-submit job.sh                                           │
│      polite-submit --array sweep.sh --range 0-99 --chunk 10         │
│                                                                     │
│  Why?                                                               │
│    Atlas is shared. polite-submit backs off when the queue is       │
│    busy, which keeps the cluster healthy for everyone.              │
│                                                                     │
│  Need help?                                                         │
│    polite-submit --help                                             │
│    cat ~/polite_submit.yaml   # see your defaults                   │
│                                                                     │
│  Edge cases (advisor-only):                                         │
│    POLITE_SUBMIT_BYPASS=1 sbatch ...                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
EOF
exit 1
