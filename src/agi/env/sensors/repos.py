# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Repository Sensor for AGI-HPC Phase 5 Environment Module.

Watches a directory of git repositories (default /archive/ahb-sjsu/)
for changes and publishes events to agi.env.sensor.repos when new
commits or file changes are detected. Could trigger RAG re-indexing.

Detection method:
    - Periodically runs ``git log -1 --format=%H`` in each repo
    - Compares HEAD hash to last-known value
    - On change: publishes the repo name, new HEAD, and changed files
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from agi.common.event import Event  # noqa: E402
from agi.core.events.nats_fabric import NatsEventFabric  # noqa: E402

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


@dataclass
class RepoSensorConfig:
    """Configuration for the repository sensor.

    Attributes:
        watch_dir: Directory containing git repositories to watch.
        poll_interval: Seconds between checks.
        max_changed_files: Max number of changed files to report per repo.
    """

    watch_dir: str = "/archive/ahb-sjsu"
    poll_interval: float = 60.0
    max_changed_files: int = 50


# -----------------------------------------------------------------
# Repository Sensor
# -----------------------------------------------------------------


class RepoSensor:
    """Watches git repositories for changes and publishes events.

    Maintains a map of repo -> last-known HEAD hash. On each poll,
    checks for new commits and publishes the diff summary.

    Usage::

        sensor = RepoSensor()
        await sensor.start(fabric)
        # ... publishes when repos change ...
        await sensor.stop()
    """

    def __init__(
        self,
        config: Optional[RepoSensorConfig] = None,
    ) -> None:
        self._config = config or RepoSensorConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._known_heads: Dict[str, str] = {}
        self._running = False

    async def start(self, fabric: NatsEventFabric) -> None:
        """Start the polling loop."""
        self._fabric = fabric
        self._running = True

        # Initial scan to populate known heads
        loop = asyncio.get_event_loop()
        self._known_heads = await loop.run_in_executor(None, self._scan_repos)

        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            "[repo-sensor] started, watching %s (%d repos), polling every %.0fs",
            self._config.watch_dir,
            len(self._known_heads),
            self._config.poll_interval,
        )

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("[repo-sensor] stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            await asyncio.sleep(self._config.poll_interval)
            if not self._running:
                break

            try:
                loop = asyncio.get_event_loop()
                current_heads = await loop.run_in_executor(None, self._scan_repos)

                changes = []
                for repo_name, new_head in current_heads.items():
                    old_head = self._known_heads.get(repo_name)
                    if old_head is None:
                        # New repo discovered
                        changes.append(
                            {
                                "repo": repo_name,
                                "action": "discovered",
                                "head": new_head[:12],
                            }
                        )
                    elif old_head != new_head:
                        # Repo has new commits
                        changed_files = self._get_changed_files(
                            repo_name, old_head, new_head
                        )
                        changes.append(
                            {
                                "repo": repo_name,
                                "action": "updated",
                                "old_head": old_head[:12],
                                "new_head": new_head[:12],
                                "changed_files": changed_files,
                                "changed_count": len(changed_files),
                            }
                        )

                self._known_heads = current_heads

                if changes:
                    event = Event.create(
                        source="env",
                        event_type="env.sensor.repos",
                        payload={
                            "watch_dir": self._config.watch_dir,
                            "repos_total": len(current_heads),
                            "changes": changes,
                            "changes_count": len(changes),
                        },
                    )
                    await self._fabric.publish("agi.env.sensor.repos", event)
                    logger.info(
                        "[repo-sensor] %d repo(s) changed",
                        len(changes),
                    )

            except Exception:
                logger.exception("[repo-sensor] error during poll")

    def _scan_repos(self) -> Dict[str, str]:
        """Scan the watch directory for git repos and their HEAD hashes."""
        heads: Dict[str, str] = {}
        watch_dir = self._config.watch_dir

        if not os.path.isdir(watch_dir):
            logger.warning("[repo-sensor] watch_dir %s does not exist", watch_dir)
            return heads

        for entry in sorted(os.listdir(watch_dir)):
            repo_path = os.path.join(watch_dir, entry)
            git_dir = os.path.join(repo_path, ".git")
            if os.path.isdir(git_dir):
                head = self._get_head(repo_path)
                if head:
                    heads[entry] = head

        return heads

    @staticmethod
    def _get_head(repo_path: str) -> Optional[str]:
        """Get the HEAD commit hash for a repository."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_changed_files(
        self,
        repo_name: str,
        old_head: str,
        new_head: str,
    ) -> List[str]:
        """Get list of files changed between two commits."""
        repo_path = os.path.join(self._config.watch_dir, repo_name)
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", old_head, new_head],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                files = result.stdout.strip().split("\n")
                files = [f for f in files if f]
                return files[: self._config.max_changed_files]
        except Exception:
            pass
        return []
