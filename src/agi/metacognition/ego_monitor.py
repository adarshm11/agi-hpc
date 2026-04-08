# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Ego Monitor — Read-only system awareness for the Ego (Gemma 4 E4B).

The Ego observes but never controls. This module collects system state
from all Atlas subsystems and presents it as a structured world model
that the Ego can reference during arbitration and training sessions.

Read-only data sources:
    - Thermal: CPU package temps, GPU temps, thermal headroom
    - Dashboard: Service health, uptime, request counts
    - Event log: Recent NATS events, message rates
    - Safety: Veto count, audit log size, reflex latency
    - Memory: Episode count, chunk count, wiki article count
    - Training: Last session scores, curriculum levels, domain coverage
    - Dreaming: Last cycle timestamp, articles created, certainty grades

Cognitive science grounding:
    - Interoception (Craig, 2002): Awareness of internal body state
    - Metacognitive monitoring (Flavell, 1979): Knowing what you know
    - Allostasis (Sterling, 2012): Predictive regulation of resources

Usage:
    monitor = EgoMonitor()
    state = monitor.observe()
    # state is a dict suitable for injection into LLM context
    summary = monitor.summarize()
    # summary is a human-readable string for the Ego's system prompt
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore[assignment]


@dataclass
class EgoMonitorConfig:
    """Configuration for the Ego monitor.

    All endpoints are read-only HTTP GETs or database SELECTs.
    The Ego never writes, mutates, or controls any system.

    Attributes:
        telemetry_url: Atlas RAG server telemetry endpoint.
        nats_monitor_url: NATS monitoring endpoint.
        db_dsn: PostgreSQL connection string (read-only queries).
        wiki_dir: Path to wiki directory for article counting.
        timeout: HTTP request timeout in seconds.
    """

    telemetry_url: str = "http://localhost:8081/api/telemetry"
    nats_monitor_url: str = "http://localhost:8222/varz"
    db_dsn: str = "dbname=atlas user=claude"
    wiki_dir: str = "/home/claude/agi-hpc/wiki"
    timeout: int = 5


class EgoMonitor:
    """Read-only system observer for the Ego.

    Collects telemetry from all Atlas subsystems and presents
    a unified world model. The Ego uses this for:
    - Adjusting training difficulty based on system load
    - Providing system context during arbitration
    - Self-awareness in conversations ("How are you doing?")

    This class NEVER modifies any state. All methods are read-only.
    """

    def __init__(self, config: Optional[EgoMonitorConfig] = None) -> None:
        self._config = config or EgoMonitorConfig()
        self._last_observation: Optional[Dict[str, Any]] = None
        self._last_observation_time: float = 0.0

    def observe(self, cache_ttl_s: float = 10.0) -> Dict[str, Any]:
        """Collect a full system observation.

        Returns a structured dict with all available telemetry.
        Results are cached for cache_ttl_s seconds to avoid
        hammering endpoints during rapid Ego calls.

        Args:
            cache_ttl_s: Cache time-to-live in seconds.

        Returns:
            Dict with keys: thermal, hemispheres, safety, memory,
            nats, training, dreaming, system.
        """
        now = time.monotonic()
        if (
            self._last_observation is not None
            and (now - self._last_observation_time) < cache_ttl_s
        ):
            return self._last_observation

        state: Dict[str, Any] = {
            "timestamp": time.time(),
            "thermal": self._read_thermal(),
            "hemispheres": self._read_hemispheres(),
            "safety": self._read_safety(),
            "memory": self._read_memory(),
            "nats": self._read_nats(),
            "training": self._read_training(),
            "dreaming": self._read_dreaming(),
            "system": self._read_system(),
        }

        self._last_observation = state
        self._last_observation_time = now
        return state

    def summarize(self) -> str:
        """Generate a human-readable summary for the Ego's context.

        This string can be injected into the Ego's system prompt
        so it has awareness of Atlas's current state.

        Returns:
            Multi-line string describing system state.
        """
        state = self.observe()

        lines = ["=== Atlas System State ==="]

        # Thermal
        thermal = state.get("thermal", {})
        if thermal.get("cpu_max_temp"):
            lines.append(
                f"CPU: {thermal['cpu_max_temp']}°C "
                f"(headroom: {thermal.get('thermal_headroom', '?')}°C)"
            )
        gpus = thermal.get("gpus", [])
        for g in gpus:
            lines.append(
                f"GPU {g['index']}: {g['temp']}°C, "
                f"{g['util']}% util, "
                f"{g['mem_used']}/{g['mem_total']} MB"
            )

        # Hemispheres
        hemispheres = state.get("hemispheres", {})
        for name, info in hemispheres.items():
            status = info.get("status", "unknown")
            model = info.get("model", "")
            lines.append(f"{name}: {status} ({model})")

        # Safety
        safety = state.get("safety", {})
        if safety:
            lines.append(
                f"Safety: {safety.get('input_checks', 0)} checks, "
                f"{safety.get('vetoes', 0)} vetoes, "
                f"{safety.get('avg_latency_ms', 0):.1f}ms avg"
            )

        # Memory
        memory = state.get("memory", {})
        if memory:
            lines.append(
                f"Memory: {memory.get('episodic_episodes', 0)} episodes, "
                f"{memory.get('semantic_chunks', 0)} chunks, "
                f"{memory.get('wiki_articles', 0)} wiki articles"
            )

        # Training
        training = state.get("training", {})
        if training.get("last_session_score"):
            lines.append(
                f"Training: last score "
                f"{training['last_session_score']:.2f}, "
                f"{training.get('total_sessions', 0)} sessions"
            )

        # Dreaming
        dreaming = state.get("dreaming", {})
        if dreaming.get("articles_created"):
            lines.append(
                f"Dreaming: {dreaming['articles_created']} " f"articles created"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Read-only data collectors (NEVER write or mutate)
    # ------------------------------------------------------------------

    def _read_thermal(self) -> Dict[str, Any]:
        """Read CPU and GPU temperatures."""
        thermal: Dict[str, Any] = {}

        # Try getting full telemetry from RAG server (it already collects GPU/CPU)
        try:
            r = requests.get(
                self._config.telemetry_url,
                timeout=self._config.timeout,
            )
            if r.ok:
                data = r.json()
                env = data.get("environment", {})
                cpu = env.get("cpu", {})
                gpus = env.get("gpu", [])

                thermal["cpu_max_temp"] = cpu.get("max_temp")
                thermal["cpu_package_temps"] = cpu.get("package_temps", [])
                thermal["gpus"] = gpus
                thermal["thermal_headroom"] = (
                    82 - cpu["max_temp"] if cpu.get("max_temp") else None
                )
        except Exception:
            pass

        return thermal

    def _read_hemispheres(self) -> Dict[str, Any]:
        """Read hemisphere health status."""
        try:
            r = requests.get(
                self._config.telemetry_url,
                timeout=self._config.timeout,
            )
            if r.ok:
                return r.json().get("hemispheres", {})
        except Exception:
            pass
        return {}

    def _read_safety(self) -> Dict[str, Any]:
        """Read safety gateway statistics."""
        try:
            r = requests.get(
                self._config.telemetry_url,
                timeout=self._config.timeout,
            )
            if r.ok:
                return r.json().get("safety", {})
        except Exception:
            pass
        return {}

    def _read_memory(self) -> Dict[str, Any]:
        """Read memory subsystem statistics."""
        memory: Dict[str, Any] = {}

        try:
            r = requests.get(
                self._config.telemetry_url,
                timeout=self._config.timeout,
            )
            if r.ok:
                memory = r.json().get("memory", {})
        except Exception:
            pass

        # Count wiki articles (the AGI's life story)
        try:
            from pathlib import Path

            wiki_path = Path(self._config.wiki_dir)
            if wiki_path.exists():
                articles = list(wiki_path.glob("dream-*.md"))
                memory["wiki_articles"] = len(articles)
                memory["wiki_dir"] = str(wiki_path)
        except Exception:
            pass

        return memory

    def _read_nats(self) -> Dict[str, Any]:
        """Read NATS event fabric statistics."""
        try:
            r = requests.get(
                self._config.nats_monitor_url,
                timeout=self._config.timeout,
            )
            if r.ok:
                data = r.json()
                return {
                    "status": "online",
                    "in_msgs": data.get("in_msgs", 0),
                    "out_msgs": data.get("out_msgs", 0),
                    "connections": data.get("connections", 0),
                    "uptime": data.get("uptime", ""),
                }
        except Exception:
            pass
        return {"status": "offline"}

    def _read_training(self) -> Dict[str, Any]:
        """Read training metrics from PostgreSQL."""
        if psycopg2 is None:
            return {}

        try:
            conn = psycopg2.connect(self._config.db_dsn)
            training: Dict[str, Any] = {}
            with conn.cursor() as cur:
                # Last training session score
                try:
                    cur.execute(
                        "SELECT AVG(score), COUNT(*), MAX(timestamp) "
                        "FROM training_results "
                        "ORDER BY timestamp DESC LIMIT 20"
                    )
                    row = cur.fetchone()
                    if row and row[1] > 0:
                        training["last_session_score"] = float(row[0] or 0)
                        training["total_sessions"] = int(row[1])
                        training["last_training"] = str(row[2])
                except Exception:
                    conn.rollback()

                # Curriculum levels by environment
                try:
                    cur.execute(
                        "SELECT env_name, "
                        "COUNT(*), "
                        "AVG(score)::float "
                        "FROM training_results "
                        "GROUP BY env_name"
                    )
                    training["environments"] = {
                        row[0]: {
                            "episodes": int(row[1]),
                            "avg_score": round(float(row[2]), 3),
                        }
                        for row in cur.fetchall()
                    }
                except Exception:
                    conn.rollback()
            conn.close()
            return training
        except Exception:
            return {}

    def _read_dreaming(self) -> Dict[str, Any]:
        """Read dreaming/consolidation statistics."""
        dreaming: Dict[str, Any] = {}

        # Check for unconsolidated episodes
        if psycopg2 is not None:
            try:
                conn = psycopg2.connect(self._config.db_dsn)
                with conn.cursor() as cur:
                    try:
                        cur.execute(
                            "SELECT COUNT(*) FROM episodes "
                            "WHERE metadata->>'consolidated' IS NULL "
                            "OR metadata->>'consolidated' = 'false'"
                        )
                        dreaming["unconsolidated_episodes"] = cur.fetchone()[0]
                    except Exception:
                        conn.rollback()

                    try:
                        cur.execute(
                            "SELECT COUNT(*) FROM episodes "
                            "WHERE metadata->>'consolidated' = 'true'"
                        )
                        dreaming["consolidated_episodes"] = cur.fetchone()[0]
                    except Exception:
                        conn.rollback()
                conn.close()
            except Exception:
                pass

        # Count wiki articles
        try:
            from pathlib import Path

            wiki_path = Path(self._config.wiki_dir)
            if wiki_path.exists():
                articles = list(wiki_path.glob("dream-*.md"))
                dreaming["articles_created"] = len(articles)
        except Exception:
            pass

        return dreaming

    def _read_system(self) -> Dict[str, Any]:
        """Read general system info."""
        try:
            r = requests.get(
                self._config.telemetry_url,
                timeout=self._config.timeout,
            )
            if r.ok:
                data = r.json()
                env = data.get("environment", {})
                return {
                    "ram": env.get("ram", {}),
                    "nats": data.get("nats", {}),
                    "dht": data.get("dht", {}),
                }
        except Exception:
            pass
        return {}
