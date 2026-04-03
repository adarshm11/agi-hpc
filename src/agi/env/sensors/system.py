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
System Sensor for AGI-HPC Phase 5 Environment Module.

Polls GPU temperature, CPU temperature, RAM usage, and disk usage
every N seconds (default 10) and publishes to agi.env.sensor.system.

Uses subprocess calls to:
    - nvidia-smi  (GPU temp, GPU memory, GPU utilization)
    - sensors     (CPU temp)
    - /proc/meminfo or psutil  (RAM)
    - df          (disk usage)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
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
class SystemSensorConfig:
    """Configuration for the system sensor.

    Attributes:
        poll_interval: Seconds between sensor polls.
        gpu_enabled: Whether to poll GPU via nvidia-smi.
        cpu_temp_enabled: Whether to poll CPU temp via sensors.
        disk_paths: Disk paths to check usage for.
    """

    poll_interval: float = 10.0
    gpu_enabled: bool = True
    cpu_temp_enabled: bool = True
    disk_paths: List[str] = field(default_factory=lambda: ["/", "/archive"])


# -----------------------------------------------------------------
# System Sensor
# -----------------------------------------------------------------


class SystemSensor:
    """Polls hardware sensors and publishes telemetry.

    Usage::

        sensor = SystemSensor()
        await sensor.start(fabric)
        # ... publishes every poll_interval seconds ...
        await sensor.stop()
    """

    def __init__(
        self,
        config: Optional[SystemSensorConfig] = None,
    ) -> None:
        self._config = config or SystemSensorConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, fabric: NatsEventFabric) -> None:
        """Start the polling loop."""
        self._fabric = fabric
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            "[sys-sensor] started, polling every %.0fs",
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
        logger.info("[sys-sensor] stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                reading = await self._collect_readings()
                event = Event.create(
                    source="env",
                    event_type="env.sensor.system",
                    payload=reading,
                )
                await self._fabric.publish("agi.env.sensor.system", event)
                logger.debug(
                    "[sys-sensor] published: gpu=%s cpu=%s ram=%.1f%%",
                    reading.get("gpu", {}).get("temperature", "n/a"),
                    reading.get("cpu_temp", "n/a"),
                    reading.get("ram", {}).get("percent_used", 0),
                )
            except Exception:
                logger.exception("[sys-sensor] error collecting readings")

            await asyncio.sleep(self._config.poll_interval)

    async def _collect_readings(self) -> Dict[str, Any]:
        """Collect all sensor readings."""
        loop = asyncio.get_event_loop()
        readings: Dict[str, Any] = {}

        # GPU readings
        if self._config.gpu_enabled:
            readings["gpu"] = await loop.run_in_executor(None, self._read_gpu)

        # CPU temperature
        if self._config.cpu_temp_enabled:
            readings["cpu_temp"] = await loop.run_in_executor(None, self._read_cpu_temp)

        # RAM usage
        readings["ram"] = await loop.run_in_executor(None, self._read_ram)

        # Disk usage
        readings["disk"] = await loop.run_in_executor(None, self._read_disk)

        return readings

    @staticmethod
    def _read_gpu() -> Dict[str, Any]:
        """Read GPU metrics via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,"
                    "utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return {"error": "nvidia-smi failed"}

            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "temperature": int(parts[2]),
                            "utilization_pct": int(parts[3]),
                            "memory_used_mb": int(parts[4]),
                            "memory_total_mb": int(parts[5]),
                        }
                    )
            return {"gpus": gpus, "count": len(gpus)}

        except FileNotFoundError:
            return {"error": "nvidia-smi not found"}
        except subprocess.TimeoutExpired:
            return {"error": "nvidia-smi timeout"}
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _read_cpu_temp() -> Any:
        """Read CPU temperature via sensors command."""
        try:
            result = subprocess.run(
                ["sensors", "-j"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                # Fallback: try parsing non-JSON output
                result2 = subprocess.run(
                    ["sensors"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # Extract first temperature reading
                for line in result2.stdout.split("\n"):
                    if "Core 0" in line and "+" in line:
                        temp_str = line.split("+")[1].split("°")[0]
                        return float(temp_str.strip())
                return "unknown"

            import json

            data = json.loads(result.stdout)
            # Extract package temperature from coretemp
            for chip_name, chip_data in data.items():
                if "coretemp" in chip_name.lower():
                    for sensor_name, sensor_data in chip_data.items():
                        if "Package" in sensor_name and isinstance(sensor_data, dict):
                            for key, val in sensor_data.items():
                                if "input" in key:
                                    return float(val)
            return "unknown"

        except FileNotFoundError:
            return "sensors not found"
        except Exception as exc:
            return str(exc)

    @staticmethod
    def _read_ram() -> Dict[str, Any]:
        """Read RAM usage from /proc/meminfo."""
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()

            info: Dict[str, int] = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])  # kB

            total_kb = info.get("MemTotal", 0)
            available_kb = info.get("MemAvailable", 0)
            used_kb = total_kb - available_kb

            return {
                "total_mb": round(total_kb / 1024, 0),
                "used_mb": round(used_kb / 1024, 0),
                "available_mb": round(available_kb / 1024, 0),
                "percent_used": (
                    round(used_kb / total_kb * 100, 1) if total_kb else 0.0
                ),
            }

        except Exception as exc:
            return {"error": str(exc)}

    def _read_disk(self) -> Dict[str, Any]:
        """Read disk usage for configured paths."""
        disks: Dict[str, Any] = {}
        for path in self._config.disk_paths:
            try:
                usage = shutil.disk_usage(path)
                disks[path] = {
                    "total_gb": round(usage.total / (1024**3), 1),
                    "used_gb": round(usage.used / (1024**3), 1),
                    "free_gb": round(usage.free / (1024**3), 1),
                    "percent_used": round(usage.used / usage.total * 100, 1),
                }
            except Exception as exc:
                disks[path] = {"error": str(exc)}
        return disks
