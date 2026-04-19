# Deployment Runbook — Atlas AI / agi-hpc

**Resolves part of:** #35
**Scope:** How to deploy a new version, how to roll back safely, how to perform common maintenance.

Companion: [`ONCALL_PLAYBOOK.md`](ONCALL_PLAYBOOK.md) for "something is broken at 3am."

---

## 0. What "deploy" means here

Atlas AI has three distinct deploy targets:

| Target | Ships | Cadence |
|---|---|---|
| **Atlas workstation** (HP Z840, Tailscale `100.68.134.21`) | Python services under systemd, dashboard HTML, Primer daemon | Auto, on push to `main` via `ci.yaml` |
| **NRP Nautilus** (K8s namespace `ssu-atlas-ai`) | Burst Jobs (vision pool), persistent Deployments (erebus-workers), PVCs | Manual via `kubectl apply` or via controller pods |
| **Package distribution** (PyPI) | nats-bursting, erisml, etc. (sibling repos) | Manual release with tag |

This runbook focuses on the first two. PyPI releases live in each sibling repo's own runbook.

---

## 1. Standard deploy to Atlas (happy path)

### 1.1 Push to `main`

CI auto-deploys on every push to `main`. No manual action needed for routine changes.

```bash
git commit -m "..."
git push
```

GitHub Actions runs `ci.yaml`:

1. **Test & Lint** — ruff + black + pytest (ignoring proto / heavy-deps tests). Fails → deploy skipped.
2. **Deploy to Atlas** — SSHes to Atlas via the `ATLAS_PASSWORD` secret, pulls `main`, runs `pip install -e ".[nats]"`, restarts `atlas-rag-server` + `atlas-telemetry`. Uses `ln -sfn` to refresh dashboard HTML symlinks (**do not** re-introduce `cp -f` — see §5.1 post-mortem).
3. **Smoke test** — hits `http://localhost:8081/api/search-status` and `/api/telemetry`; logs OK/FAIL.

### 1.2 Verify the deploy landed

After CI goes green (~3 minutes), verify:

```bash
# The automated drift detector — runs every 30 min, and on every push
gh run list --repo ahb-sjsu/agi-hpc --workflow "Deploy Smoke" --limit 1

# Or manually compare live SHA vs local HEAD
curl -sk https://atlas-sjsu.duckdns.org/api/version | jq -r .sha
git rev-parse --short=7 HEAD
```

If live SHA ≠ local HEAD after ~5 min, the deploy didn't land. Go to [`ONCALL_PLAYBOOK.md`](ONCALL_PLAYBOOK.md) §"Deploy didn't land."

### 1.3 Dashboard smoke

Hard-refresh `https://atlas-sjsu.duckdns.org/schematic.html` and confirm:

- Footer version stamp matches the commit you just pushed
- NATS Topology panel is populated
- NRP Burst Jobs + Worker Pools card has rows
- Erebus — NeuroGolf 2026 card shows current solve score
- No browser-console red errors

---

## 2. Manual deploy (when CI is down or you're out-of-band)

### 2.1 From a dev machine with SSH access to Atlas

```bash
# Via paramiko from any OS — what I use during sessions
python - <<'EOF'
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('100.68.134.21', username='claude', password='<see atlas-operations.md>', timeout=10)
cmds = [
    'cd /home/claude/agi-hpc',
    'git fetch origin main',
    'git reset --hard origin/main',
    '/home/claude/env/bin/pip install -e ".[nats]" --quiet',
    'sudo systemctl restart atlas-telemetry',
    'sudo systemctl restart atlas-rag-server',
    'sleep 5',
    'sudo systemctl is-active atlas-telemetry atlas-rag-server',
]
stdin, stdout, stderr = c.exec_command(' && '.join(cmds))
print(stdout.read().decode())
c.close()
EOF
```

### 2.2 From Atlas directly (if you have shell)

```bash
cd /home/claude/agi-hpc
git fetch origin main && git reset --hard origin/main
/home/claude/env/bin/pip install -e '.[nats]' --quiet
sudo systemctl restart atlas-telemetry atlas-rag-server
sudo systemctl status atlas-telemetry atlas-rag-server
```

---

## 3. Services that restart automatically on deploy

`ci.yaml` restarts only **two** services:

- `atlas-telemetry.service` — picks up dashboard + `/api/*` endpoint changes
- `atlas-rag-server.service` — picks up RAG search changes

Services that **do not** auto-restart on deploy (and therefore need manual attention when their code changes):

| Service | When to restart manually |
|---|---|
| `atlas-primer.service` | When `src/agi/primer/*` or `deploy/systemd/atlas-primer.service` changes |
| `arc_scientist.py` (nohup process) | When `src/agi/autonomous/arc_scientist.py` changes |
| `atlas-id.service` | When llama.cpp model path or GPU 1 config changes |
| `atlas-superego.service` | When GPU 0 config changes |
| `atlas-ego.service` | When Divine Council args change |
| `atlas-dreaming.service` | When dreaming consolidation code changes |
| `atlas-nats.service` | When JetStream config changes |
| `atlas-nats-leaf.service` | When leaf-node credentials change |

All follow the same pattern:

```bash
sudo systemctl restart atlas-<name>.service
sudo systemctl status atlas-<name>.service --no-pager
```

---

## 4. Rollback

### 4.1 Quick rollback — revert a bad commit

```bash
git revert <sha>
git push
```

CI re-deploys the pre-bad state. This is the **preferred rollback**: the bad change + its revert both stay in history.

### 4.2 Emergency rollback — hard reset `main`

Only use when the revert path is blocked (e.g. a merge commit with conflicts).

```bash
git checkout main
git reset --hard <last-known-good-sha>
git push --force-with-lease
```

Prefer `--force-with-lease` over `--force` — refuses to push if someone else's work would be obliterated.

### 4.3 Revert dashboard-only changes

Dashboard HTML is served from symlinks into the git tree. Reverting the commit + redeploy (per §4.1) is sufficient — **do not** manually edit `/home/claude/atlas-chat/*.html` on Atlas, those are symlinks and local edits will be reverted by the next deploy.

### 4.4 Revert a Primer auto-commit (e.g. a wiki note that's miscategorizing)

Primer commits look like `primer: verified sensei note for task NNN (family)`. If you spot one that's wrong:

```bash
# Revert the specific commit
git revert <sha>
git push

# OR rewrite the note in place (the Primer won't republish unless task
# is re-stuck and cooldown expires)
edit wiki/sensei_task_NNN.md
git add wiki/sensei_task_NNN.md
git commit -m "wiki(sensei): correct task NNN — <what was wrong>"
git push
```

Both paths work. The Primer won't fight you because cooldown prevents re-write for 6 hours, and by the time cooldown expires the corrected version is in the wiki context it reads.

---

## 5. Common maintenance

### 5.1 Dashboard deploy drift (historical post-mortem — do not regress)

**Never use `cp -f <root>/atlas-chat-*.html $STATIC_PATH/*.html`.** The destination is a symlink into the git tree. `cp -f` follows the symlink and writes stale content back into the working tree. This caused 16 commits of dashboard features to be silently reverted before anyone noticed.

Use `ln -sfn "$repo/infra/local/atlas-chat/$f" "$static/$f"` always. Both `ci.yaml` and `scripts/deploy_to_atlas.sh` now do this correctly.

Guardrail: the **Dashboard Render** CI workflow (`dashboard-render.yaml`) runs Playwright against the live URL every 30 min; if the rendered page loses widgets, the workflow fails and surfaces in the Actions tab.

### 5.2 NRP pod housekeeping

#### Check current pods

```bash
kubectl -n ssu-atlas-ai get pods -o wide
kubectl -n ssu-atlas-ai get deployments,jobs
```

#### Clean up finished Jobs

```bash
# Delete all Completed vision-pool pods older than 1h
kubectl -n ssu-atlas-ai delete pod -l app=erebus-vision --field-selector=status.phase=Succeeded

# Delete all Jobs that finished
kubectl -n ssu-atlas-ai delete jobs -l app=erebus-vision --field-selector=status.successful=1
```

#### Restart the persistent worker pool

```bash
kubectl -n ssu-atlas-ai rollout restart deployment/erebus-workers
kubectl -n ssu-atlas-ai rollout status deployment/erebus-workers
```

#### Re-provision the ego-models PVC (if corrupted)

```bash
# Only when you're sure — 300 GiB of model weights lost
kubectl -n ssu-atlas-ai delete pvc erebus-ego-models
kubectl apply -f deploy/k8s/erebus-ego-pvc.yaml
# Wait for Bound, then any ego pod will re-download the model on start
```

### 5.3 Arc Scientist restart

The ARC Scientist runs as a nohup process (not systemd), typically launched from a bash one-liner. To restart cleanly:

```bash
ssh claude@100.68.134.21 <<'EOF'
pkill -f "arc_scientist.py --task-dir"
sleep 2
cd /archive/neurogolf
source /home/claude/env/bin/activate
export PYTHONPATH=/home/claude/agi-hpc/src
export NRP_LLM_TOKEN=$(cat ~/.llmtoken)
nohup python3 -u /home/claude/agi-hpc/src/agi/autonomous/arc_scientist.py \
  --task-dir /archive/neurogolf \
  --memory /archive/neurogolf/arc_scientist_memory.json \
  --attempts 400 --cycles 5 \
  >> /archive/neurogolf/scientist.log 2>&1 </dev/null &
disown
EOF
```

The important bit: **always pass `--memory /archive/neurogolf/arc_scientist_memory.json` explicitly.** Without it, the Scientist writes to a fresh relative-path memory file and loses the solve history.

### 5.4 Primer restart

```bash
sudo systemctl restart atlas-primer.service
sudo journalctl -u atlas-primer.service -f
```

Loads wiki + memory at startup, so any new sensei articles take effect immediately. The health tracker state does *not* persist across restarts — a fresh start will probe all experts once before settling.

### 5.5 Clearing the Primer's task cooldown (force re-try of a specific task)

```bash
# Edit the cooldown file on Atlas
ssh claude@100.68.134.21 <<'EOF'
python3 -c "
import json
p = '/archive/neurogolf/primer_cooldown.json'
d = json.load(open(p))
d.pop('NNN', None)  # <-- task number
open(p, 'w').write(json.dumps(d))
"
EOF
sudo systemctl restart atlas-primer.service
```

The Primer will attempt task `NNN` on its next tick.

### 5.6 Thermal management (Atlas only)

High CPU temp (≥82 °C) triggers `atlas-thermal.service` to throttle. If sustained:

```bash
sensors | grep Package
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv

# Kill any runaway tmux sessions
tmux ls
tmux kill-session -t <runaway>
```

See [`ATLAS_OPERATIONS.md`](ATLAS_OPERATIONS.md) §"Thermal Safety" for the full protocol.

---

## 6. NRP-specific deploys

### 6.1 Deploy a new PVC

```bash
kubectl apply -f deploy/k8s/erebus-ego-pvc.yaml
kubectl -n ssu-atlas-ai get pvc erebus-ego-models  # expect STATUS=Bound
```

If stays `Pending` > 1 min, storage provisioner is stuck — switch storage class from `cephfs` (newer) to `rook-cephfs` (older, more reliable) and re-apply. See `reference_nrp_l40_reservation` in operator memory.

### 6.2 Deploy a new burst-job type

Render manifest via `scripts/render_erebus_*_job.py` (currently vision; future: training, eval). Apply:

```bash
python scripts/render_erebus_vision_job.py > /tmp/job.yaml
kubectl apply -f /tmp/job.yaml
```

### 6.3 Update or replace the erebus-workers Deployment

```bash
# From the repo root
kubectl apply -f deploy/k8s/erebus-workers-deployment.yaml
kubectl rollout status -n ssu-atlas-ai deployment/erebus-workers
```

If the Deployment isn't in this repo (it lives in `nats-bursting`), go to that repo's deploy process.

---

## 7. Secret management

No secret manager in use. Credentials live in `~/.*` dotfiles on Atlas:

| File | Contains | Used by |
|---|---|---|
| `~/.llmtoken` | NRP ELLM bearer token | `atlas-rag-server`, `atlas-primer`, chat handler |
| `~/.cache/huggingface/token` | HF token (hf_...) | QLoRA dreaming, model downloads |
| `~/.huggingface/token` | HF token (duplicate) | older tools |
| `~/.kube/config` | NRP kubectl context | dashboard `/api/nrp-burst`, vision-burst dispatch |
| `~/.primer.env` | Primer's env vars | `atlas-primer.service` via `EnvironmentFile=-` |
| `/etc/systemd/system/atlas-telemetry.service` | inline env (no secrets) | telemetry |

GitHub Actions secrets (set via `gh secret set`):

- `ATLAS_HOST` — `100.68.134.21`
- `ATLAS_USER` — `claude`
- `ATLAS_PASSWORD` — for sshpass

Rotate: update the dotfile on Atlas + regenerate the token with the upstream provider. No automation yet for this.

---

## 8. Post-deploy checklist

After any non-trivial deploy:

- [ ] `gh run list --repo ahb-sjsu/agi-hpc --limit 3` shows green on the last CI run
- [ ] `https://atlas-sjsu.duckdns.org/api/version` returns SHA = `git rev-parse --short HEAD`
- [ ] Hard-refresh dashboard, confirm all panels render
- [ ] `sudo systemctl is-active atlas-telemetry atlas-rag-server atlas-primer atlas-nats` all return `active`
- [ ] Erebus NeuroGolf 2026 card shows a solve count ≥ pre-deploy
- [ ] No new `ERROR` lines in `journalctl --since "10 minutes ago"` across atlas services

If any of these fail, go to [`ONCALL_PLAYBOOK.md`](ONCALL_PLAYBOOK.md).
