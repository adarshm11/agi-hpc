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
Environment NATS Service for AGI-HPC Phase 5.

Runs the system sensor and repository sensor together, publishing
hardware telemetry and git change events to the NATS event fabric.

Components:
    - SystemSensor: GPU temp, CPU temp, RAM, disk every 10s
    - RepoSensor:   git repo changes in /archive/ahb-sjsu/
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig  # noqa: E402
from agi.env.sensors.system import SystemSensor, SystemSensorConfig  # noqa: E402
from agi.env.sensors.repos import RepoSensor, RepoSensorConfig  # noqa: E402

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


@dataclass
class EnvironmentServiceConfig:
    """Configuration for the Environment NATS service.

    Attributes:
        nats_servers: NATS server URLs.
        system_poll_interval: System sensor poll interval (seconds).
        gpu_enabled: Whether to poll GPU sensors.
        cpu_temp_enabled: Whether to poll CPU temperature.
        disk_paths: Disk paths to monitor.
        repo_watch_dir: Directory of git repos to watch.
        repo_poll_interval: Repo sensor poll interval (seconds).
        enable_system_sensor: Whether to run the system sensor.
        enable_repo_sensor: Whether to run the repo sensor.
    """

    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    system_poll_interval: float = 10.0
    gpu_enabled: bool = True
    cpu_temp_enabled: bool = True
    disk_paths: List[str] = field(default_factory=lambda: ["/", "/archive"])
    repo_watch_dir: str = "/archive/ahb-sjsu"
    repo_poll_interval: float = 60.0
    enable_system_sensor: bool = True
    enable_repo_sensor: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> EnvironmentServiceConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        env = data.get("environment", data)
        nats_cfg = env.get("nats", {})
        sys_cfg = env.get("system", {})
        repo_cfg = env.get("repos", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            system_poll_interval=sys_cfg.get("poll_interval", 10.0),
            gpu_enabled=sys_cfg.get("gpu_enabled", True),
            cpu_temp_enabled=sys_cfg.get("cpu_temp_enabled", True),
            disk_paths=sys_cfg.get("disk_paths", ["/", "/archive"]),
            repo_watch_dir=repo_cfg.get("watch_dir", "/archive/ahb-sjsu"),
            repo_poll_interval=repo_cfg.get("poll_interval", 60.0),
            enable_system_sensor=env.get("enable_system_sensor", True),
            enable_repo_sensor=env.get("enable_repo_sensor", True),
        )


# -----------------------------------------------------------------
# Environment NATS Service
# -----------------------------------------------------------------


class EnvironmentService:
    """NATS-connected environment monitoring service.

    Orchestrates the system sensor and repo sensor on a single
    NATS connection.

    Usage::

        service = EnvironmentService()
        await service.start()
        # ... runs until stopped ...
        await service.stop()
    """

    def __init__(
        self,
        config: Optional[EnvironmentServiceConfig] = None,
    ) -> None:
        self._config = config or EnvironmentServiceConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._system_sensor: Optional[SystemSensor] = None
        self._repo_sensor: Optional[RepoSensor] = None
        self._running = False

    async def start(self) -> None:
        """Connect to NATS and start all sensors."""
        logger.info("[env-service] starting Phase 5 Environment Service")

        # Initialise NATS fabric
        fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
        self._fabric = NatsEventFabric(config=fabric_config)
        await self._fabric.connect()

        # Start system sensor
        if self._config.enable_system_sensor:
            sys_config = SystemSensorConfig(
                poll_interval=self._config.system_poll_interval,
                gpu_enabled=self._config.gpu_enabled,
                cpu_temp_enabled=self._config.cpu_temp_enabled,
                disk_paths=self._config.disk_paths,
            )
            self._system_sensor = SystemSensor(config=sys_config)
            await self._system_sensor.start(self._fabric)

        # Start repo sensor
        if self._config.enable_repo_sensor:
            repo_config = RepoSensorConfig(
                watch_dir=self._config.repo_watch_dir,
                poll_interval=self._config.repo_poll_interval,
            )
            self._repo_sensor = RepoSensor(config=repo_config)
            await self._repo_sensor.start(self._fabric)

        self._running = True
        logger.info(
            "[env-service] ready -- system=%s repos=%s",
            "on" if self._config.enable_system_sensor else "off",
            "on" if self._config.enable_repo_sensor else "off",
        )

    async def stop(self) -> None:
        """Stop all sensors and disconnect."""
        self._running = False
        if self._repo_sensor:
            await self._repo_sensor.stop()
        if self._system_sensor:
            await self._system_sensor.stop()
        if self._fabric:
            await self._fabric.disconnect()
        logger.info("[env-service] stopped")


# -----------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------


async def run_service(config_path: Optional[str] = None) -> None:
    """Run the Environment Service until interrupted."""
    if config_path:
        config = EnvironmentServiceConfig.from_yaml(config_path)
    else:
        config = EnvironmentServiceConfig()

    service = EnvironmentService(config=config)
    await service.start()

    try:
        while service._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await service.stop()


def main() -> None:
    """CLI entry point for the Environment Service."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AGI-HPC Environment Service (Phase 5)"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to env_config.yaml",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        asyncio.run(run_service(args.config))
    except KeyboardInterrupt:
        logger.info("[env-service] interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
