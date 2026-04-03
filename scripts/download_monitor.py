#!/usr/bin/env python3
"""Atlas Download Pipeline Monitor.

Keeps downloads saturating the link, indexes content as it arrives,
and respects thermal limits via batch-probe ThermalController.

Usage:
    python3 scripts/download_monitor.py
    python3 scripts/download_monitor.py --max-concurrent 4
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("dl-monitor")

ARCHIVE = Path("/archive/knowledge")
ETHICS = Path("/archive/ethics-corpora")
MAX_CONCURRENT = 4
POLL_INTERVAL = 30
TARGET_BANDWIDTH_MBPS = 50  # target link utilization


@dataclass
class DownloadJob:
    name: str
    url: str
    dest: str
    method: str = "aria2c"  # aria2c, rsync, git, wget, curl
    args: List[str] = field(default_factory=list)
    priority: int = 1  # lower = higher priority
    expected_size_gb: float = 0
    done: bool = False
    pid: Optional[int] = None
    session: Optional[str] = None


# Define all downloads we want
ALL_JOBS = [
    # Already have Wikipedia (24GB) -- skip
    # Gutenberg full text via rsync (slow but reliable)
    DownloadJob(
        name="gutenberg",
        url="aleph.gutenberg.org::gutenberg",
        dest=str(ARCHIVE / "gutenberg"),
        method="rsync",
        args=["-av", "--include=*.txt", "--include=*/", "--exclude=*"],
        priority=2,
        expected_size_gb=60,
    ),
    # arXiv bulk metadata
    DownloadJob(
        name="arxiv-meta",
        url="https://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=oai_dc&set=cs",
        dest=str(ARCHIVE / "arxiv/metadata"),
        method="wget",
        args=["--wait=3", "--random-wait", "-r", "-l1", "-nd"],
        priority=3,
        expected_size_gb=2,
    ),
    # Perseus Latin (if not already cloned)
    DownloadJob(
        name="perseus-latin",
        url="https://github.com/PerseusDL/canonical-latinLit.git",
        dest=str(ETHICS / "perseus-latin"),
        method="git",
        args=["--depth", "1"],
        priority=1,
        expected_size_gb=1,
    ),
    # Sefaria full export
    DownloadJob(
        name="sefaria-full",
        url="https://github.com/Sefaria/Sefaria-Export.git",
        dest=str(ETHICS / "sefaria-full"),
        method="git",
        args=["--depth", "1"],
        priority=1,
        expected_size_gb=2,
    ),
]


def is_tmux_session_alive(name: str) -> bool:
    r = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True,
    )
    return r.returncode == 0


def get_active_downloads() -> int:
    r = subprocess.run(
        ["tmux", "ls"],
        capture_output=True,
        text=True,
    )
    count = 0
    for line in r.stdout.split("\n"):
        if line.startswith("dl-"):
            count += 1
    return count


def get_cpu_temps() -> tuple[float, float]:
    r = subprocess.run(
        ["sensors"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    temps = []
    for line in r.stdout.split("\n"):
        if "Package id" in line:
            t = float(line.split("+")[1].split("°")[0])
            temps.append(t)
    return (temps[0] if temps else 0, temps[1] if len(temps) > 1 else 0)


def get_net_rx_bytes() -> int:
    with open("/proc/net/dev") as f:
        for line in f:
            if "eno1" in line:
                parts = line.split()
                return int(parts[1])
    return 0


def dest_exists(job: DownloadJob) -> bool:
    p = Path(job.dest)
    if p.is_dir() and any(p.iterdir()):
        return True
    return False


def start_download(job: DownloadJob) -> None:
    session = f"dl-{job.name}"
    dest = Path(job.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if job.method == "aria2c":
        cmd = f"aria2c -x 16 -s 16 -j 4 --continue=true --dir={job.dest} '{job.url}'"
    elif job.method == "rsync":
        args = " ".join(job.args)
        cmd = f"rsync {args} '{job.url}' '{job.dest}/'"
    elif job.method == "git":
        if (dest / ".git").exists():
            cmd = f"cd '{job.dest}' && git pull"
        else:
            args = " ".join(job.args)
            cmd = f"git clone {args} '{job.url}' '{job.dest}'"
    elif job.method == "wget":
        args = " ".join(job.args)
        cmd = f"wget {args} -P '{job.dest}' '{job.url}'"
    elif job.method == "curl":
        cmd = f"curl -L -o '{job.dest}/download' '{job.url}'"
    else:
        log.error(f"Unknown method: {job.method}")
        return

    full_cmd = f"{cmd} 2>&1 | tee /tmp/dl_{job.name}.log; echo DL_DONE_{job.name}"

    subprocess.Popen(
        ["tmux", "new-session", "-d", "-s", session, full_cmd],
    )
    job.session = session
    log.info(f"Started: {job.name} ({job.method}) -> {job.dest}")


def check_completed(job: DownloadJob) -> bool:
    if job.session and not is_tmux_session_alive(job.session):
        return True
    log_path = f"/tmp/dl_{job.name}.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read()
            if f"DL_DONE_{job.name}" in content:
                return True
    return False


def get_disk_usage_gb(path: str) -> float:
    r = subprocess.run(
        ["du", "-s", "--block-size=1G", path],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if r.stdout:
        return int(r.stdout.split()[0])
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log.info("Atlas Download Pipeline Monitor")
    log.info(f"  Max concurrent: {args.max_concurrent}")
    log.info(f"  Archive: {ARCHIVE}")
    log.info(f"  Ethics: {ETHICS}")

    # Filter out already-completed jobs
    pending = []
    for job in ALL_JOBS:
        if dest_exists(job):
            size = get_disk_usage_gb(job.dest)
            if size >= job.expected_size_gb * 0.8:
                log.info(f"  Skip {job.name}: already have {size}GB")
                job.done = True
                continue
        pending.append(job)

    pending.sort(key=lambda j: j.priority)
    log.info(f"  Pending downloads: {len(pending)}")

    if args.dry_run:
        for j in pending:
            log.info(f"  Would download: {j.name} ({j.method}) {j.expected_size_gb}GB")
        return

    active: list[DownloadJob] = []
    prev_rx = get_net_rx_bytes()
    prev_time = time.time()

    while pending or active:
        # Check temps
        t0, t1 = get_cpu_temps()
        if max(t0, t1) > 85:
            log.warning(f"CPU temps high ({t0}/{t1}°C), pausing new downloads")
            time.sleep(30)
            continue

        # Check completed
        for job in list(active):
            if check_completed(job):
                log.info(f"Completed: {job.name}")
                job.done = True
                active.remove(job)

        # Start new downloads if under limit
        while pending and len(active) < args.max_concurrent:
            job = pending.pop(0)
            start_download(job)
            active.append(job)

        # Measure bandwidth
        now = time.time()
        rx = get_net_rx_bytes()
        dt = now - prev_time
        if dt > 0:
            mbps = (rx - prev_rx) / dt / 1_000_000
        else:
            mbps = 0
        prev_rx = rx
        prev_time = now

        # Status
        active_names = ", ".join(j.name for j in active)
        log.info(
            f"Active: {len(active)}/{args.max_concurrent} [{active_names}] "
            f"| BW: {mbps:.1f} MB/s | Temps: {t0}/{t1}°C "
            f"| Pending: {len(pending)}"
        )

        time.sleep(POLL_INTERVAL)

    log.info("All downloads complete!")


if __name__ == "__main__":
    main()
