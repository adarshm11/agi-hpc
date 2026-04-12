# Atlas JupyterHub — Hub configuration
# Copyright (c) 2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""JupyterHub configuration for the Atlas student server.

Guiding principles (see ops/jupyterhub/README.md for context):

- GitHub OAuth is the only way in.
- The allowlist (``allowed_users.txt``) is the only way to get through.
- Each student's container runs as the Linux UID matching their GH
  identity on Atlas (federated via ``uid-map.json``), so Slurm jobs are
  accounted per-student.
- Student containers run on an isolated Docker network.
- Each user is capped at 2 CPU and 8 GB memory by default.
- No GPU by default. ``gpu_users.txt`` opts specific users in.
- Idle sessions (>1 h) are culled.
- Slurm/munge sockets and config are bind-mounted so jobs can be
  submitted from the notebook via the polite-submit wrapper.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dockerspawner import DockerSpawner
from oauthenticator.github import GitHubOAuthenticator

c = get_config()  # noqa: F821  (injected by JupyterHub)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

HUB_DATA_DIR = Path("/srv/jupyterhub")
ALLOWED_USERS_FILE = HUB_DATA_DIR / "allowed_users.txt"
GPU_USERS_FILE = HUB_DATA_DIR / "gpu_users.txt"
UID_MAP_FILE = HUB_DATA_DIR / "uid-map.json"


def _read_user_list(path: Path) -> set[str]:
    """Read a username-per-line file into a lowercase set.

    - Strips whitespace
    - Ignores blank lines
    - Ignores lines starting with '#' (comments)
    - Returns empty set if the file is missing
    """
    if not path.exists():
        return set()
    names: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.lower())
    return names


def _read_uid_map(path: Path) -> dict[str, int]:
    """Read the GH-username → UID map produced by sync-student-accounts.sh."""
    if not path.exists():
        return {}
    with path.open() as f:
        raw = json.load(f)
    return {k.lower(): int(v) for k, v in raw.items()}


ALLOWED_USERS: set[str] = _read_user_list(ALLOWED_USERS_FILE)
GPU_USERS: set[str] = _read_user_list(GPU_USERS_FILE)
UID_MAP: dict[str, int] = _read_uid_map(UID_MAP_FILE)

# Default fallback UID for admin users not in the map (ahb-sjsu runs
# spawn-for-self via the admin UI sometimes). Set to 1000 to match a
# typical host admin account; override via env if needed.
ADMIN_FALLBACK_UID = int(os.environ.get("ADMIN_FALLBACK_UID", "1000"))

# --------------------------------------------------------------------------
# Authentication: GitHub OAuth
# --------------------------------------------------------------------------

c.JupyterHub.authenticator_class = GitHubOAuthenticator

c.GitHubOAuthenticator.client_id = os.environ["GITHUB_CLIENT_ID"]
c.GitHubOAuthenticator.client_secret = os.environ["GITHUB_CLIENT_SECRET"]
c.GitHubOAuthenticator.oauth_callback_url = os.environ["OAUTH_CALLBACK_URL"]

c.Authenticator.auto_login = False
c.Authenticator.allowed_users = ALLOWED_USERS
c.Authenticator.admin_users = {"ahb-sjsu"}

if not ALLOWED_USERS:
    raise RuntimeError(
        "allowed_users.txt is empty or missing. Mount it at "
        f"{ALLOWED_USERS_FILE} or fix the compose config."
    )

# --------------------------------------------------------------------------
# Spawner: DockerSpawner
# --------------------------------------------------------------------------

c.JupyterHub.spawner_class = DockerSpawner

c.DockerSpawner.image = os.environ.get(
    "USER_CONTAINER_IMAGE", "atlas-jupyterhub-user:local"
)
c.DockerSpawner.name_template = "jupyter-{username}"
c.DockerSpawner.network_name = os.environ.get("USER_NETWORK", "jhub_users")
c.DockerSpawner.use_internal_ip = True

c.JupyterHub.hub_connect_ip = "jhub_hub"
c.JupyterHub.hub_ip = "0.0.0.0"

# Per-user persistent home directory — bind-mounted from /home on the
# host so that Slurm jobs (which run on the host as the student's UID)
# see the same files as the notebook.
c.DockerSpawner.volumes = {
    # /home/<username> on the host → /home/jovyan/work in the container
    "/home/{username}": "/home/jovyan/work",
    # Slurm + munge bind-mounts (read-only where possible)
    "/etc/slurm": {"bind": "/etc/slurm", "mode": "ro"},
    "/var/run/munge": {"bind": "/var/run/munge", "mode": "rw"},
    "/etc/munge/munge.key": {"bind": "/etc/munge/munge.key", "mode": "ro"},
}

# Resource limits
USER_CPU_LIMIT = float(os.environ.get("USER_CPU_LIMIT", "2.0"))
USER_MEM_LIMIT = os.environ.get("USER_MEM_LIMIT", "8G")

c.DockerSpawner.cpu_limit = USER_CPU_LIMIT
c.DockerSpawner.mem_limit = USER_MEM_LIMIT

c.DockerSpawner.remove = True
c.DockerSpawner.start_timeout = 120
c.DockerSpawner.http_timeout = 120

# --------------------------------------------------------------------------
# Pre-spawn hook: set per-user UID + GPU opt-in
# --------------------------------------------------------------------------
#
# This is where federation happens. At spawn time we look up the
# authenticated user's GH username in uid-map.json and set NB_UID to
# their Linux UID on Atlas. The jupyter base image reads NB_UID at
# startup and runs the notebook as that UID.


def pre_spawn_hook(spawner: DockerSpawner) -> None:
    """Set per-user UID (federation) and optional GPU access."""
    username = spawner.user.name.lower()

    # -------- UID federation --------
    uid = UID_MAP.get(username)
    if uid is None:
        if username in c.Authenticator.admin_users:
            uid = ADMIN_FALLBACK_UID
        else:
            raise RuntimeError(
                f"No Linux UID allocated for {username}. Run "
                f"sudo ./scripts/sync-student-accounts.sh on Atlas "
                f"after adding them to allowed_users.txt."
            )

    env = dict(spawner.environment or {})
    env["NB_UID"] = str(uid)
    env["NB_GID"] = str(uid)
    env["CHOWN_HOME"] = "yes"
    env["GRANT_SUDO"] = "no"
    # Slurm client needs this to find the config
    env["SLURM_CONF"] = "/etc/slurm/slurm.conf"

    # Container must start as root to drop to NB_UID via start.sh;
    # after start.sh resets UID, the notebook server runs as the student.
    spawner.extra_create_kwargs = {"user": "root"}

    # -------- Extra host config for security + GPU --------
    host_cfg = {
        "security_opt": ["no-new-privileges:true"],
        "cap_drop": ["ALL"],
        "cap_add": ["CHOWN", "DAC_OVERRIDE", "FOWNER", "SETUID", "SETGID"],
    }

    if username in GPU_USERS:
        host_cfg["runtime"] = "nvidia"
        host_cfg["device_requests"] = [
            {
                "Driver": "nvidia",
                "Count": 1,
                "Capabilities": [["gpu"]],
            }
        ]
        env["NVIDIA_VISIBLE_DEVICES"] = "all"
        env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"

    spawner.extra_host_config = host_cfg
    spawner.environment = env


c.Spawner.pre_spawn_hook = pre_spawn_hook

# --------------------------------------------------------------------------
# Idle culling
# --------------------------------------------------------------------------

IDLE_CULL_SECONDS = int(os.environ.get("IDLE_CULL_SECONDS", "3600"))

c.JupyterHub.services = [
    {
        "name": "idle-culler",
        "admin": True,
        "command": [
            "python3",
            "-m",
            "jupyterhub_idle_culler",
            f"--timeout={IDLE_CULL_SECONDS}",
            "--cull-every=300",
            "--cull-users",
            "--remove-named-servers",
        ],
    },
]

# --------------------------------------------------------------------------
# Hub-level settings
# --------------------------------------------------------------------------

c.JupyterHub.bind_url = "http://:8000"
c.JupyterHub.db_url = f"sqlite:///{HUB_DATA_DIR}/jupyterhub.sqlite"
c.JupyterHub.cookie_secret_file = str(HUB_DATA_DIR / "jupyterhub_cookie_secret")
c.JupyterHub.log_level = "INFO"
c.JupyterHub.concurrent_spawn_limit = 5
c.JupyterHub.active_server_limit = 25
