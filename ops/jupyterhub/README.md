# Atlas JupyterHub — Student Server with Slurm + polite-submit

Complete runbook for a network-segmented JupyterHub instance on Atlas, with:

- **GitHub OAuth** authentication against a static allowlist
- **Per-student Linux accounts** federated from GH usernames (UID-map)
- **DockerSpawner** — one isolated container per student
- **Single-node Slurm** — students submit real batch jobs from their notebooks
- **polite-submit enforcement** — direct `sbatch` is disabled; all submissions go through the politeness wrapper
- **Per-student Slurm accountability** — every job is tagged with the submitter's GH username via Linux UID federation

---

## Table of contents

1. [Architecture](#architecture)
2. [What this setup guarantees](#what-this-setup-guarantees)
3. [Prerequisites on Atlas](#prerequisites-on-atlas)
4. [One-time GitHub OAuth app setup](#one-time-github-oauth-app-setup)
5. [One-time Slurm setup on Atlas](#one-time-slurm-setup-on-atlas)
6. [Student-account federation: the UID map](#student-account-federation-the-uid-map)
7. [Deployment steps](#deployment-steps)
8. [Firewall hardening](#firewall-hardening)
9. [Day-to-day operations](#day-to-day-operations)
10. [Managing students: add / remove / GPU](#managing-students)
11. [How students interact with Slurm](#how-students-interact-with-slurm)
12. [Troubleshooting](#troubleshooting)
13. [Files in this directory](#files-in-this-directory)
14. [Security notes](#security-notes)

---

## Architecture

```
Internet (Comcast)
        │
        │  port forward 443 ──────►  atlas:8443
        ▼
┌───────────────────────┐
│      Caddy            │   TLS termination, auto-cert via Let's Encrypt
│  (container, :8443)   │   No auth here — JupyterHub handles it
└──────────┬────────────┘
           │  reverse proxy to jhub_hub:8000
           ▼
┌───────────────────────┐
│     JupyterHub        │   GitHub OAuth + allowlist + DockerSpawner
│  (container)          │   Binds to localhost only
└──────────┬────────────┘
           │ spawns
           ▼
┌──────────────────────────────────────────────────────┐
│  jhub_users network  (isolated from production)      │
│                                                      │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│   │ alice   │  │ bob     │  │ carol   │              │
│   │ UID 10001  │ UID 10002  │ UID 10003              │
│   │ 2 CPU   │  │ 2 CPU   │  │ 2 CPU   │              │
│   │ 8 GB    │  │ 8 GB    │  │ 8 GB    │              │
│   └────┬────┘  └────┬────┘  └────┬────┘              │
│        │            │            │                   │
│        │   mounts:  /etc/slurm, /var/run/munge,      │
│        │            /home/<user>, polite-submit      │
│        │            wrapper for sbatch               │
└────────┼────────────┼────────────┼───────────────────┘
         │            │            │
         ▼            ▼            ▼
    ┌──────────────────────────────────────────────────┐
    │  Atlas host — Slurm (single-node cluster)        │
    │                                                  │
    │    munge + slurmctld + slurmd                    │
    │    Linux users: alice (UID 10001),               │
    │                 bob (10002), carol (10003) ...   │
    │    Partitions: interactive, cpu, gpu, research   │
    │    GRES: 2x GV100 GPUs                           │
    └──────────────────────────────────────────────────┘

Admin (ahbond) accesses via Tailscale SSH.
Production services (agi-hpc, atlas-ai, postgres) are on separate
Docker networks — student containers cannot reach them.
```

---

## What this setup guarantees

- **Network isolation**: student containers live on `jhub_users` only; they cannot reach the host services, agi-hpc production containers, or postgres.
- **Auth at the edge**: no allowed-user, no container. GitHub OAuth happens at the Hub; the allowlist is rechecked every login.
- **Per-student Linux identity**: each student's container runs as their own UID on Atlas, so Slurm jobs are charged to them individually and `squeue -u alice` shows only alice's jobs.
- **Resource caps**: 2 CPU, 8 GB RAM per notebook container. GPU is opt-in per user (`gpu_users.txt`).
- **Polite-submit enforcement**: direct `sbatch` from a student container prints a nudge and exits 1. Only `polite-submit` (or explicit `POLITE_SUBMIT_BYPASS=1`) can reach the real Slurm submission.
- **Research preemption**: Andrew's `research` partition has Priority=1000 and preempts student jobs — you can always reclaim the GPUs.
- **TLS everywhere**: Caddy obtains + renews Let's Encrypt certs automatically.
- **Idle culling**: sessions idle >1 h are shut down.
- **Auditable allowlist**: `allowed_users.txt` is git-tracked; every change is in history.

---

## Prerequisites on Atlas

1. **Ubuntu 24.04** (kernel 6.8 confirmed via `uname -a`)
2. **Docker 24+** with the `compose` plugin
3. **Public DNS name** pointing at your Comcast public IP (e.g. `jupyter.yourdomain.com`). Required for Let's Encrypt.
4. **Comcast port-forward** 443 → atlas:8443
5. **Tailscale** already running for admin SSH
6. **NVIDIA Container Toolkit** if you plan to enable GPU for any users (`nvidia-ctk runtime configure --runtime=docker`)

---

## One-time GitHub OAuth app setup

1. Go to <https://github.com/settings/developers> → **OAuth Apps** → **New OAuth App**
2. Fill in:
   - Application name: `Atlas JupyterHub`
   - Homepage URL: `https://jupyter.yourdomain.com/`
   - Authorization callback URL: `https://jupyter.yourdomain.com/hub/oauth_callback`
3. Save, then on the app page click **Generate a new client secret**
4. Record the Client ID and Client Secret — paste into `.env` at deployment time.

---

## One-time Slurm setup on Atlas

Run this once, then forget about it unless Slurm itself needs work:

```bash
cd ~/source/agi-hpc/ops/jupyterhub
sudo ./scripts/install-slurm-atlas.sh
```

The script is **idempotent and readable**. Skim it before running. It does:

1. Installs `munge`, `slurm-wlm`, `slurm-client`, `slurmctld`, `slurmd`
2. Creates the munge key and configures permissions
3. Creates the `slurm` system user and Slurm state directories
4. Writes `/etc/slurm/slurm.conf`, `/etc/slurm/cgroup.conf`, `/etc/slurm/gres.conf`
5. Starts + enables `munge`, `slurmctld`, `slurmd`
6. Creates the `jhub` and `gpu_users` Linux groups
7. Runs `sinfo` as a self-test

After it finishes, verify:

```bash
sinfo
# expect: 4 partitions (interactive, cpu, gpu, research), state idle/up

srun hostname
# expect: atlas
```

### Partitions (defined in slurm.conf)

| Partition | Max time | Resources | Default? | Who can use |
|---|---|---|---|---|
| `interactive` | 30 min | 2 CPU / 8 GB | No | Anyone |
| `cpu` | 4 h | 8 CPU / 32 GB | No | Anyone |
| `gpu` | 4 h | 8 CPU / 64 GB / 1 GPU | No | Members of `gpu_users` group |
| `research` | Unlimited | Unrestricted | Yes (for you) | `research` account — preempts student jobs |

### If you need to reconfigure later

Edit `/etc/slurm/slurm.conf`, then:

```bash
sudo scontrol reconfigure
# or if the change is structural:
sudo systemctl restart slurmctld slurmd
```

Keep `slurm/slurm.conf.template` in this repo synced if you change the live config, so the git copy matches what's running.

---

## Student-account federation: the UID map

### How it works

1. You add a GitHub username to `allowed_users.txt` and commit.
2. On Atlas, you run `sudo ./scripts/sync-student-accounts.sh`.
3. The script:
   - For each GH username: ensures a Linux user exists at a stable UID from the `10000–19999` pool.
   - Creates the user's home directory at `/home/<gh_username>/`.
   - Adds them to the `jhub` group (and `gpu_users` group if they're in `gpu_users.txt`).
   - Writes the mapping to `/var/lib/atlas-jhub/uid-map.json`.
   - For users removed from the allowlist: locks their account (`passwd -l`) and sets shell to `/sbin/nologin`. Home directories are preserved.
4. When a student logs into JupyterHub:
   - `pre_spawn_hook` in `jupyterhub_config.py` reads `uid-map.json`.
   - Container spawns with `NB_UID=<their UID>`.
   - Jupyter's base image resets `jovyan` to that UID at startup and drops privileges.
   - Munge on the host authenticates the UID when the container submits a Slurm job.
   - Slurm logs the job under their Linux username = their GH username.

### Add a student

```bash
# On your dev machine (the repo)
$EDITOR ops/jupyterhub/allowed_users.txt
# Add their GitHub username on its own line, lowercase
git add ops/jupyterhub/allowed_users.txt
git commit -m "Allowlist: add @alice-student"
git push

# On Atlas
cd ~/source/agi-hpc && git pull
sudo ops/jupyterhub/scripts/sync-student-accounts.sh
docker compose -f ops/jupyterhub/docker-compose.yml restart jupyterhub
```

Student is good to go. On their next login, the hub spawns them a container under their new UID.

### Remove a student

```bash
# Remove (or comment out with #) their line in allowed_users.txt
$EDITOR ops/jupyterhub/allowed_users.txt
git commit -m "Allowlist: offboard @alice-student"
git push

# On Atlas
cd ~/source/agi-hpc && git pull
sudo ops/jupyterhub/scripts/sync-student-accounts.sh
docker compose -f ops/jupyterhub/docker-compose.yml restart jupyterhub
```

Their account is locked but their home directory is kept. To re-enable: put them back in the allowlist and re-sync.

### Grant GPU access

```bash
$EDITOR ops/jupyterhub/gpu_users.txt   # add their GH username
sudo ops/jupyterhub/scripts/sync-student-accounts.sh
docker compose -f ops/jupyterhub/docker-compose.yml restart jupyterhub
```

Their next container spawn gets the NVIDIA runtime and `NVIDIA_VISIBLE_DEVICES=all`.

---

## Deployment steps

On Atlas, via Tailscale SSH:

```bash
# 1. Clone / pull
cd ~/source/agi-hpc && git pull origin main
cd ops/jupyterhub

# 2. Configure env
cp env.example .env
$EDITOR .env
# Fill in: GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, OAUTH_CALLBACK_URL, DOMAIN

# 3. Review allowlists
$EDITOR allowed_users.txt
$EDITOR gpu_users.txt

# 4. Ensure Slurm is installed (one-time, see above)
sudo ./scripts/install-slurm-atlas.sh

# 5. Sync student accounts (creates Linux users + UID map)
sudo ./scripts/sync-student-accounts.sh

# 6. Build and bring up the stack
docker compose build
docker compose up -d

# 7. Verify
docker compose ps
docker compose logs -f jupyterhub   # Ctrl-C when settled

# 8. External smoke test
curl -I https://jupyter.yourdomain.com
# Expect: HTTP/2 302, Location: /hub/login

# 9. Try logging in with a whitelisted GitHub account
# Open https://jupyter.yourdomain.com in a browser
```

---

## Firewall hardening

After verifying the stack is working end-to-end:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow in on tailscale0 to any port 22    # admin SSH only via Tailscale
sudo ufw allow in to any port 8443 proto tcp      # public JupyterHub (via Comcast NAT)
sudo ufw enable
sudo ufw status numbered
```

**Do not** expose 22 on the public interface. Admin is Tailscale-only.

---

## Day-to-day operations

### Watching logs

```bash
# Hub + Caddy + builder
docker compose logs -f

# Hub only
docker compose logs -f jupyterhub

# A specific user's container
docker logs -f jupyter-<github-username>

# Slurm daemons (on host, not in a container)
sudo journalctl -u slurmctld -f
sudo journalctl -u slurmd -f
sudo journalctl -u munge -f
```

### Restarting after config changes

```bash
# allowlist or gpu_users change (after running sync-student-accounts.sh)
docker compose restart jupyterhub

# jupyterhub_config.py change
docker compose restart jupyterhub

# Dockerfile.user change — rebuild the user image
docker compose build --no-cache user-image-builder

# Caddyfile change
docker compose restart caddy
```

### Rotating OAuth secrets

1. GitHub → your OAuth app → **Revoke user tokens** → **Generate a new client secret**
2. Edit `.env` on Atlas
3. `docker compose up -d jupyterhub`

### Upgrading JupyterHub

Edit the pin in `Dockerfile.hub`, then:

```bash
docker compose build --no-cache jupyterhub
docker compose up -d jupyterhub
```

### Stopping everything

```bash
docker compose down                  # keep volumes
docker compose down -v               # delete hub state (DB, cookie) — nuclear
```

Slurm continues running regardless (it's a host service, not Docker).

---

## Managing students

See [Student-account federation](#student-account-federation-the-uid-map) above for the actual commands.

**Summary cheat-sheet:**

| Action | Files to edit | Command to run on Atlas |
|---|---|---|
| Add student | `allowed_users.txt` | `sudo sync-student-accounts.sh && docker compose restart jupyterhub` |
| Remove student | `allowed_users.txt` | `sudo sync-student-accounts.sh && docker compose restart jupyterhub` |
| Grant GPU access | `gpu_users.txt` | `sudo sync-student-accounts.sh && docker compose restart jupyterhub` |
| Revoke GPU access | `gpu_users.txt` | `sudo sync-student-accounts.sh && docker compose restart jupyterhub` |
| Ban student temporarily | Comment out their line in `allowed_users.txt` | `docker compose restart jupyterhub` |

---

## How students interact with Slurm

From any notebook cell, a student runs:

```python
%%bash
# Check cluster state
sinfo
squeue -u $USER
```

To submit a job, they write `job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --partition=interactive
#SBATCH --time=00:02:00
#SBATCH --output=hello-%j.out

hostname; date
```

Then submit through polite-submit:

```bash
polite-submit job.sh
```

Direct `sbatch job.sh` prints:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Direct sbatch use is not allowed on Atlas.                         │
│  Please submit via polite-submit: polite-submit job.sh              │
│  Why? Atlas is shared. polite-submit backs off when the queue is    │
│  busy, which keeps the cluster healthy for everyone.                │
└─────────────────────────────────────────────────────────────────────┘
```

and exits 1.

### Polite-submit config

Default config is baked into the user image at `~/polite_submit.yaml`:

- `max_concurrent_jobs: 2`, `max_pending_jobs: 2`
- Peak hours 9 AM–5 PM Pacific, weekends exempt
- Backoff: starts 30 s, doubles to max 30 min, max 20 attempts
- Utilization threshold: 75 %

Students can override per-project by creating a local `polite_submit.yaml` in their working directory.

### Emergency bypass (admin only)

If you need to bypass the wrapper (rare — only for your own research jobs):

```bash
POLITE_SUBMIT_BYPASS=1 sbatch your-job.sh
```

The wrapper honors this env var. Don't give it to students.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 502 Bad Gateway from Caddy | Hub isn't running | `docker compose ps`; check hub logs |
| GitHub login says "user not allowed" | GH username not in `allowed_users.txt` or allowlist not loaded | Add them, restart hub |
| Spawn fails with "No Linux UID allocated" | `uid-map.json` missing entry | Run `sudo sync-student-accounts.sh` on Atlas |
| Cert fails to issue | Port 443 not reachable from internet; router forward wrong | Verify with `curl -I http://jupyter.yourdomain.com/` from outside |
| Student container starts but can't access Slurm | munge socket not mounted; user UID wrong | Verify `/var/run/munge/munge.socket.2` exists in container; check `id` inside matches the uid-map |
| `sbatch` in a student container doesn't show the polite-submit banner | Wrapper not on PATH or got overwritten | Rebuild user image; verify `/usr/local/bin/sbatch` is the wrapper |
| `polite-submit` says "cluster host null, but sinfo failed" | Slurm client can't talk to slurmctld from container | Check `munge -n \| unmunge` inside the container; if it fails, munge key/socket mount is wrong |
| `srun` hangs forever | slurmd not running or node is DRAIN | `sudo systemctl status slurmd`, `sinfo -R` |
| User shows up but home dir empty | `/home/<user>` not created by sync script | Re-run sync; check permissions on `/home/<user>` (should be 700, owned by them) |
| Research partition jobs evicting my interactive session | As designed — `research` preempts. Run your interactive in `interactive` or `cpu`. | — |
| Quota violations during peak hours | `polite_submit.yaml` peak hours config | Adjust `peak_hours.max_concurrent` globally, or add a local override in the project dir |

---

## Files in this directory

| File | What it does |
|---|---|
| `README.md` | This file — the runbook |
| `docker-compose.yml` | Orchestrates Caddy + Hub + user-image builder |
| `jupyterhub_config.py` | Hub config: OAuth, allowlist, UID federation, DockerSpawner |
| `Dockerfile.hub` | Hub image (JupyterHub + OAuthenticator + DockerSpawner) |
| `Dockerfile.user` | User notebook image (scipy + PyTorch + erisml-lib + slurm-client + polite-submit + sbatch wrapper) |
| `Caddyfile` | Reverse proxy + TLS |
| `allowed_users.txt` | Student allowlist (git-tracked) |
| `gpu_users.txt` | GPU opt-in list (git-tracked) |
| `env.example` | Template `.env` — never commit the real one |
| `welcome.ipynb` | Opens automatically for first-time users; includes Slurm quickstart |
| `polite_submit.yaml` | Default polite-submit config baked into the user image |
| `bin/sbatch-wrapper.sh` | Replaces `sbatch` in the user image; nudges toward polite-submit |
| `scripts/install-slurm-atlas.sh` | One-time Atlas Slurm installer |
| `scripts/sync-student-accounts.sh` | Sync Linux users + UID map from the allowlist |
| `slurm/slurm.conf.template` | Reference copy of the live Slurm config |
| `.gitignore` | Keeps `.env` and state dirs out of git |

---

## Security notes

- **Secrets discipline.** `.env` is gitignored. The OAuth client secret must never be committed. Rotate via the GitHub OAuth app page and redeploy.
- **Hub binding.** JupyterHub binds to `127.0.0.1:8000` only. Caddy is the only ingress.
- **Container hardening.** Student containers run with `security_opt: no-new-privileges`, `cap_drop: ALL`, re-adding only `CHOWN DAC_OVERRIDE FOWNER SETUID SETGID` for jupyter's startup.
- **Munge key.** `/etc/munge/munge.key` is mounted read-only into containers, owned by the host's `munge:munge`. Munge authenticates the submitter UID for Slurm; if you ever regenerate the key, restart slurmctld and all student containers.
- **Home directories.** `/home/<gh_username>/` on the host is bind-mounted into `/home/jovyan/work` in the container. Each student sees only their own home. Permission 700 enforces this.
- **Network segmentation.** `jhub_users` docker network is dedicated to student containers. Production agi-hpc containers live on different networks and are unreachable from student space.
- **No shell on Atlas for students.** Their Linux account exists only so Slurm can authenticate them. Login shell is `/bin/bash` (so Slurm jobs can `#!/bin/bash`) but they never reach a terminal outside their notebook container — the SSH daemon is firewalled from the public internet (Tailscale-only).
- **Audit trail.**
  - `allowed_users.txt` is git-tracked — every allowlist change is a commit.
  - Slurm `accounting_storage/filetxt` writes per-job records to `/var/log/slurm/accounting`, keyed by username = GH username.
  - JupyterHub logs every login (GH username → container name → spawn time).
  - Polite-submit logs to `~/polite_submit.log` per user.

---

## Quick reference — where to look when something breaks

```
Hub logs:           docker compose logs -f jupyterhub
Caddy logs:         docker compose logs -f caddy
Student container:  docker logs -f jupyter-<username>
Slurm controller:   sudo journalctl -u slurmctld -f
Slurm compute:      sudo journalctl -u slurmd -f
Munge:              sudo journalctl -u munge -f
Slurm accounting:   sudo tail -f /var/log/slurm/accounting
User-side log:      ~<gh_username>/polite_submit.log (on host)
UID map:            cat /var/lib/atlas-jhub/uid-map.json
```
