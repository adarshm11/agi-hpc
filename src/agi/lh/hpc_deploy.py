# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
LH HPC Deployment Module for AGI-HPC.

Provides HPC cluster deployment support:
- SLURM job script generation and submission
- Apptainer/Singularity container management
- Resource allocation and scheduling

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HPCDeployConfig:
    """Configuration for HPC cluster deployment."""

    scheduler: str = "slurm"
    partition: str = os.getenv("AGI_HPC_PARTITION", "gpu")
    nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_task: int = 4
    memory_gb: int = 32
    time_limit: str = "01:00:00"
    container_runtime: str = "apptainer"
    container_image: str = ""
    work_dir: str = os.getenv("AGI_HPC_WORK_DIR", "/scratch")
    log_dir: str = os.getenv("AGI_HPC_LOG_DIR", "/scratch/logs")
    modules: List[str] = field(default_factory=lambda: ["cuda/12.0", "python/3.12"])


# ---------------------------------------------------------------------------
# SLURM Launcher
# ---------------------------------------------------------------------------


class SlurmLauncher:
    """Generate, submit, and manage SLURM batch jobs."""

    def __init__(self, config: Optional[HPCDeployConfig] = None) -> None:
        self._config = config or HPCDeployConfig()
        logger.info(
            "[lh][hpc] SlurmLauncher initialized partition=%s nodes=%d gpus=%d",
            self._config.partition,
            self._config.nodes,
            self._config.gpus_per_node,
        )

    def generate_script(self, job_name: str, command: str, **kwargs: Any) -> str:
        """Generate a SLURM batch script."""
        cfg = self._config
        partition = kwargs.get("partition", cfg.partition)
        nodes = kwargs.get("nodes", cfg.nodes)
        gpus_per_node = kwargs.get("gpus_per_node", cfg.gpus_per_node)
        cpus_per_task = kwargs.get("cpus_per_task", cfg.cpus_per_task)
        memory_gb = kwargs.get("memory_gb", cfg.memory_gb)
        time_limit = kwargs.get("time_limit", cfg.time_limit)
        log_dir = kwargs.get("log_dir", cfg.log_dir)
        modules = kwargs.get("modules", cfg.modules)
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --gpus-per-node={gpus_per_node}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={memory_gb}G",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output={log_dir}/{job_name}_%j.out",
            f"#SBATCH --error={log_dir}/{job_name}_%j.err",
            "",
            "# --- Environment setup ---",
            "set -euo pipefail",
        ]
        for mod in modules:
            lines.append(f"module load {mod}")
        lines.append("")
        lines.append(f"cd {cfg.work_dir}")
        lines.append("")
        container_image = kwargs.get("container_image", cfg.container_image)
        if container_image:
            runtime = kwargs.get("container_runtime", cfg.container_runtime)
            gpu_flag = "--nv" if gpus_per_node > 0 else ""
            lines.append("# --- Container execution ---")
            lines.append(f"{runtime} exec {gpu_flag} {container_image} {command}")
        else:
            lines.append("# --- Direct execution ---")
            lines.append(command)
        lines.append("")
        script = "\n".join(lines)
        logger.debug("[lh][hpc] generated script for job=%s", job_name)
        return script

    def submit(self, script: str) -> str:
        """Submit a SLURM batch script and return the job ID."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            delete=False,
            dir=self._config.work_dir if os.path.isdir(self._config.work_dir) else None,
        ) as tmp:
            tmp.write(script)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["sbatch", tmp_path], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"sbatch failed (rc={result.returncode}): {result.stderr.strip()}"
                )
            output = result.stdout.strip()
            job_id = output.split()[-1]
            logger.info("[lh][hpc] submitted job=%s", job_id)
            return job_id
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def cancel(self, job_id: str) -> bool:
        """Cancel a running or pending SLURM job."""
        result = subprocess.run(
            ["scancel", job_id], capture_output=True, text=True, check=False
        )
        success = result.returncode == 0
        if success:
            logger.info("[lh][hpc] cancelled job=%s", job_id)
        else:
            logger.warning(
                "[lh][hpc] cancel failed job=%s: %s", job_id, result.stderr.strip()
            )
        return success

    def status(self, job_id: str) -> Dict[str, Any]:
        """Query the status of a SLURM job via sacct."""
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode,Elapsed,NodeList",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        info: Dict[str, Any] = {
            "job_id": job_id,
            "state": "UNKNOWN",
            "exit_code": "",
            "elapsed": "",
            "node_list": "",
        }
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            parts = first_line.split("|")
            if len(parts) >= 5:
                info["job_id"] = parts[0]
                info["state"] = parts[1]
                info["exit_code"] = parts[2]
                info["elapsed"] = parts[3]
                info["node_list"] = parts[4]
        logger.debug("[lh][hpc] status job=%s state=%s", job_id, info["state"])
        return info

    def wait_for_completion(
        self, job_id: str, poll_interval: float = 10.0
    ) -> Dict[str, Any]:
        """Block until a SLURM job completes, polling at regular intervals."""
        _terminal_states = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "PREEMPTED",
            "OUT_OF_MEMORY",
        }
        logger.info(
            "[lh][hpc] waiting for job=%s poll_interval=%.1fs", job_id, poll_interval
        )
        while True:
            info = self.status(job_id)
            state = info.get("state", "UNKNOWN")
            if state in _terminal_states:
                logger.info(
                    "[lh][hpc] job=%s finished state=%s elapsed=%s",
                    job_id,
                    state,
                    info.get("elapsed", "?"),
                )
                return info
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Apptainer / Singularity Runner
# ---------------------------------------------------------------------------


class ApptainerRunner:
    """Manage Apptainer (formerly Singularity) containers on HPC clusters."""

    def __init__(self, config: Optional[HPCDeployConfig] = None) -> None:
        self._config = config or HPCDeployConfig()
        self._runtime = self._config.container_runtime
        logger.info("[lh][hpc] ApptainerRunner initialized runtime=%s", self._runtime)

    def build(self, definition_file: str, output_path: str) -> bool:
        """Build a container image from a definition file."""
        logger.info(
            "[lh][hpc] building container def=%s output=%s",
            definition_file,
            output_path,
        )
        result = subprocess.run(
            [self._runtime, "build", output_path, definition_file],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error("[lh][hpc] build failed: %s", result.stderr.strip())
            return False
        logger.info("[lh][hpc] build succeeded: %s", output_path)
        return True

    def run(
        self,
        image: str,
        command: str,
        binds: Optional[List[str]] = None,
        gpu: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command inside a container."""
        cmd: List[str] = [self._runtime, "run"]
        if gpu:
            cmd.append("--nv")
        for bind in binds or []:
            cmd.extend(["--bind", bind])
        cmd.append(image)
        cmd.extend(["bash", "-c", command])
        logger.info("[lh][hpc] running container image=%s cmd=%s", image, command)
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.warning(
                "[lh][hpc] container run returned rc=%d: %s",
                result.returncode,
                result.stderr.strip(),
            )
        return result

    def exec(self, container: str, command: str) -> subprocess.CompletedProcess:
        """Execute a command in a container image."""
        cmd: List[str] = [self._runtime, "exec", container, "bash", "-c", command]
        logger.debug("[lh][hpc] exec container=%s cmd=%s", container, command)
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def pull(self, uri: str, output_path: str) -> bool:
        """Pull a container image from a registry."""
        logger.info("[lh][hpc] pulling image uri=%s -> %s", uri, output_path)
        result = subprocess.run(
            [self._runtime, "pull", output_path, uri],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error("[lh][hpc] pull failed: %s", result.stderr.strip())
            return False
        logger.info("[lh][hpc] pull succeeded: %s", output_path)
        return True
