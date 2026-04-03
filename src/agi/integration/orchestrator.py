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
Integration Orchestrator for AGI-HPC Phase 4.

The "conductor" that routes incoming user requests to the appropriate
hemisphere(s) and merges responses when both are engaged.

Subscribes to:
    agi.integration.route

Publishes to:
    agi.lh.request.{chat,plan,reason}
    agi.rh.request.{pattern,spatial,creative}
    agi.integration.merge (when both hemispheres respond)
    agi.memory.store.episodic (session tracking)
    agi.meta.monitor.integration (telemetry)

Query classification determines routing:
    - Analytical/precise queries -> LH only
    - Creative/pattern queries  -> RH only
    - Complex/ambiguous queries -> both hemispheres, then merge
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from agi.common.event import Event  # noqa: E402
from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig  # noqa: E402

# -----------------------------------------------------------------
# Hemisphere keywords (shared with lh.nats_service)
# -----------------------------------------------------------------

LH_KEYWORDS = {
    "explain",
    "debug",
    "error",
    "fix",
    "how does",
    "what is",
    "define",
    "analyze",
    "calculate",
    "prove",
    "implement",
    "code",
    "function",
    "syntax",
    "compile",
    "trace",
    "step by step",
    "specifically",
    "exact",
    "precise",
    "correct",
    "documentation",
    "api",
    "reference",
}

RH_KEYWORDS = {
    "brainstorm",
    "creative",
    "imagine",
    "what if",
    "pattern",
    "analogy",
    "design",
    "vision",
    "inspire",
    "explore",
    "possibilities",
    "connect",
    "themes",
    "big picture",
    "strategy",
    "reimagine",
    "innovate",
    "compare across",
    "similarities",
    "different angle",
    "metaphor",
}


def classify_query(text: str) -> str:
    """Route to 'lh', 'rh', or 'both' based on query content.

    Returns:
        'lh' for analytical queries, 'rh' for creative queries,
        'both' when both scores are significant or ambiguous.
    """
    lower = text.lower()
    lh_score = sum(1 for kw in LH_KEYWORDS if kw in lower)
    rh_score = sum(1 for kw in RH_KEYWORDS if kw in lower)

    # Both hemispheres if both have significant signal
    if lh_score >= 2 and rh_score >= 2:
        return "both"
    if rh_score > lh_score and rh_score >= 2:
        return "rh"
    return "lh"


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    """Configuration for the Integration Orchestrator.

    Attributes:
        nats_servers: NATS server URLs.
        merge_timeout: Seconds to wait for both hemisphere responses.
        default_lh_subject: Default LH request subject.
        default_rh_subject: Default RH request subject.
        enable_session_tracking: Whether to store episodes in memory.
        routing_threshold: Minimum keyword hits to trigger a hemisphere.
    """

    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    merge_timeout: float = 120.0
    default_lh_subject: str = "agi.lh.request.chat"
    default_rh_subject: str = "agi.rh.request.creative"
    enable_session_tracking: bool = True
    routing_threshold: int = 2

    @classmethod
    def from_yaml(cls, path: str) -> OrchestratorConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        integ = data.get("integration", data)
        nats_cfg = integ.get("nats", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            merge_timeout=integ.get("merge_timeout", 120.0),
            default_lh_subject=integ.get("default_lh_subject", "agi.lh.request.chat"),
            default_rh_subject=integ.get(
                "default_rh_subject", "agi.rh.request.creative"
            ),
            enable_session_tracking=integ.get("enable_session_tracking", True),
            routing_threshold=integ.get("routing_threshold", 2),
        )


# -----------------------------------------------------------------
# Telemetry
# -----------------------------------------------------------------


@dataclass
class OrchestratorTelemetry:
    """Accumulates orchestrator metrics."""

    requests_total: int = 0
    lh_routes: int = 0
    rh_routes: int = 0
    both_routes: int = 0
    merges_completed: int = 0
    merges_timeout: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_total": self.requests_total,
            "lh_routes": self.lh_routes,
            "rh_routes": self.rh_routes,
            "both_routes": self.both_routes,
            "merges_completed": self.merges_completed,
            "merges_timeout": self.merges_timeout,
            "errors": self.errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "hemisphere_ratio": (round(self.lh_routes / max(self.rh_routes, 1), 2)),
        }


# -----------------------------------------------------------------
# Pending merge tracker
# -----------------------------------------------------------------


@dataclass
class PendingMerge:
    """Tracks a pending dual-hemisphere request awaiting both responses."""

    trace_id: str
    prompt: str
    session_id: str
    t0: float
    lh_response: Optional[Dict[str, Any]] = None
    rh_response: Optional[Dict[str, Any]] = None

    @property
    def is_complete(self) -> bool:
        return self.lh_response is not None and self.rh_response is not None


# -----------------------------------------------------------------
# Integration Orchestrator
# -----------------------------------------------------------------


class IntegrationOrchestrator:
    """Routes incoming requests to the appropriate hemisphere(s).

    Acts as the central conductor of the cognitive architecture,
    classifying queries, dispatching to LH/RH, and merging
    dual-hemisphere responses.

    Usage::

        orch = IntegrationOrchestrator()
        await orch.start()
        # ... runs until stopped ...
        await orch.stop()
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self._config = config or OrchestratorConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._telemetry = OrchestratorTelemetry()
        self._pending_merges: Dict[str, PendingMerge] = {}
        self._running = False

    async def start(self) -> None:
        """Connect to NATS and subscribe to routing + response subjects."""
        logger.info("[orchestrator] starting Phase 4 Integration Orchestrator")

        fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
        self._fabric = NatsEventFabric(config=fabric_config)
        await self._fabric.connect()

        # Subscribe to incoming route requests
        await self._fabric.subscribe("agi.integration.route", self._handle_route)

        # Subscribe to hemisphere responses (for merge tracking)
        await self._fabric.subscribe("agi.lh.response.>", self._handle_lh_response)
        await self._fabric.subscribe("agi.rh.response.>", self._handle_rh_response)

        self._running = True
        logger.info("[orchestrator] ready -- listening on agi.integration.route")

    async def stop(self) -> None:
        """Disconnect and clean up."""
        self._running = False
        if self._fabric:
            await self._fabric.disconnect()
        logger.info("[orchestrator] stopped")

    async def _handle_route(self, event: Event) -> None:
        """Handle an incoming route request.

        Expected payload keys:
            prompt (str): The user query to route.
            session_id (str, optional): Session identifier.
            force_hemisphere (str, optional): Override routing ('lh', 'rh', 'both').
            config (dict, optional): InferenceConfig overrides.
        """
        t0 = time.perf_counter()
        prompt = event.payload.get("prompt", "")
        session_id = event.payload.get("session_id", str(uuid.uuid4())[:8])
        trace_id = event.trace_id

        logger.info(
            "[orchestrator] route request trace=%s session=%s prompt=%r",
            trace_id[:8],
            session_id,
            prompt[:60],
        )

        try:
            # Classify the query
            force = event.payload.get("force_hemisphere")
            if force in ("lh", "rh", "both"):
                hemisphere = force
            else:
                hemisphere = classify_query(prompt)

            self._telemetry.requests_total += 1

            # Build the dispatch payload
            dispatch_payload: Dict[str, Any] = {
                "prompt": prompt,
                "session_id": session_id,
            }
            user_config = event.payload.get("config")
            if user_config:
                dispatch_payload["config"] = user_config

            if hemisphere == "lh":
                self._telemetry.lh_routes += 1
                lh_event = Event.create(
                    source="integration",
                    event_type="lh.request.chat",
                    payload=dispatch_payload,
                    trace_id=trace_id,
                )
                await self._fabric.publish(self._config.default_lh_subject, lh_event)
                logger.info("[orchestrator] routed to LH trace=%s", trace_id[:8])

            elif hemisphere == "rh":
                self._telemetry.rh_routes += 1
                rh_event = Event.create(
                    source="integration",
                    event_type="rh.request.creative",
                    payload=dispatch_payload,
                    trace_id=trace_id,
                )
                await self._fabric.publish(self._config.default_rh_subject, rh_event)
                logger.info("[orchestrator] routed to RH trace=%s", trace_id[:8])

            else:
                # Both hemispheres
                self._telemetry.both_routes += 1

                # Track pending merge
                pending = PendingMerge(
                    trace_id=trace_id,
                    prompt=prompt,
                    session_id=session_id,
                    t0=t0,
                )
                self._pending_merges[trace_id] = pending

                # Dispatch to both
                lh_event = Event.create(
                    source="integration",
                    event_type="lh.request.chat",
                    payload=dispatch_payload,
                    trace_id=trace_id,
                )
                rh_event = Event.create(
                    source="integration",
                    event_type="rh.request.creative",
                    payload=dispatch_payload,
                    trace_id=trace_id,
                )
                await self._fabric.publish(self._config.default_lh_subject, lh_event)
                await self._fabric.publish(self._config.default_rh_subject, rh_event)

                # Schedule timeout cleanup
                asyncio.get_event_loop().call_later(
                    self._config.merge_timeout,
                    lambda tid=trace_id: asyncio.ensure_future(
                        self._timeout_merge(tid)
                    ),
                )

                logger.info(
                    "[orchestrator] routed to BOTH hemispheres trace=%s",
                    trace_id[:8],
                )

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry.total_latency_ms += latency_ms

            # Publish routing telemetry
            telemetry_event = Event.create(
                source="integration",
                event_type="meta.monitor.integration",
                payload={
                    **self._telemetry.to_dict(),
                    "last_routing": hemisphere,
                    "last_trace": trace_id[:8],
                },
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.meta.monitor.integration", telemetry_event)

        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[orchestrator] error routing request trace=%s",
                trace_id[:8],
            )

    async def _handle_lh_response(self, event: Event) -> None:
        """Handle LH response events for merge tracking."""
        trace_id = event.trace_id
        if trace_id not in self._pending_merges:
            return

        pending = self._pending_merges[trace_id]
        pending.lh_response = event.payload

        if pending.is_complete:
            await self._complete_merge(trace_id)

    async def _handle_rh_response(self, event: Event) -> None:
        """Handle RH response events for merge tracking."""
        trace_id = event.trace_id
        if trace_id not in self._pending_merges:
            return

        pending = self._pending_merges[trace_id]
        pending.rh_response = event.payload

        if pending.is_complete:
            await self._complete_merge(trace_id)

    async def _complete_merge(self, trace_id: str) -> None:
        """Merge dual-hemisphere responses and publish the result."""
        pending = self._pending_merges.pop(trace_id, None)
        if pending is None:
            return

        latency_ms = (time.perf_counter() - pending.t0) * 1000.0
        self._telemetry.merges_completed += 1

        lh_text = (pending.lh_response or {}).get("text", "")
        rh_text = (pending.rh_response or {}).get("text", "")

        # Merge strategy: concatenate with clear labels
        merged_text = (
            f"## Analytical Perspective (Left Hemisphere)\n\n{lh_text}\n\n"
            f"## Creative Perspective (Right Hemisphere)\n\n{rh_text}"
        )

        lh_tokens = (pending.lh_response or {}).get("tokens_used", 0)
        rh_tokens = (pending.rh_response or {}).get("tokens_used", 0)

        merge_event = Event.create(
            source="integration",
            event_type="integration.merge",
            payload={
                "text": merged_text,
                "lh_text": lh_text,
                "rh_text": rh_text,
                "lh_tokens": lh_tokens,
                "rh_tokens": rh_tokens,
                "total_tokens": lh_tokens + rh_tokens,
                "latency_ms": round(latency_ms, 1),
                "session_id": pending.session_id,
                "prompt": pending.prompt,
            },
            trace_id=trace_id,
        )
        await self._fabric.publish("agi.integration.merge", merge_event)

        # Store episode in memory
        if self._config.enable_session_tracking:
            episode_event = Event.create(
                source="integration",
                event_type="memory.store.episodic",
                payload={
                    "session_id": pending.session_id,
                    "user_message": pending.prompt,
                    "atlas_response": merged_text[:2000],
                    "hemisphere": "both",
                    "metadata": {
                        "lh_tokens": lh_tokens,
                        "rh_tokens": rh_tokens,
                        "latency_ms": round(latency_ms, 1),
                    },
                },
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.memory.store.episodic", episode_event)

        logger.info(
            "[orchestrator] merge complete trace=%s latency=%.0fms",
            trace_id[:8],
            latency_ms,
        )

    async def _timeout_merge(self, trace_id: str) -> None:
        """Handle merge timeout -- publish partial result if available."""
        pending = self._pending_merges.pop(trace_id, None)
        if pending is None:
            return  # already completed

        self._telemetry.merges_timeout += 1
        latency_ms = (time.perf_counter() - pending.t0) * 1000.0

        # Build partial response
        parts = []
        if pending.lh_response:
            parts.append(
                "## Analytical Perspective (Left Hemisphere)\n\n"
                + pending.lh_response.get("text", "")
            )
        if pending.rh_response:
            parts.append(
                "## Creative Perspective (Right Hemisphere)\n\n"
                + pending.rh_response.get("text", "")
            )

        if not parts:
            parts.append("Both hemispheres timed out without responding.")

        merge_event = Event.create(
            source="integration",
            event_type="integration.merge",
            payload={
                "text": "\n\n".join(parts),
                "partial": True,
                "timeout": True,
                "latency_ms": round(latency_ms, 1),
                "session_id": pending.session_id,
                "prompt": pending.prompt,
            },
            trace_id=trace_id,
        )
        await self._fabric.publish("agi.integration.merge", merge_event)

        logger.warning(
            "[orchestrator] merge timeout trace=%s latency=%.0fms",
            trace_id[:8],
            latency_ms,
        )

    @property
    def telemetry(self) -> OrchestratorTelemetry:
        """Return current telemetry snapshot."""
        return self._telemetry


# -----------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------


async def run_service(config_path: Optional[str] = None) -> None:
    """Run the Integration Orchestrator until interrupted."""
    if config_path:
        config = OrchestratorConfig.from_yaml(config_path)
    else:
        config = OrchestratorConfig()

    orch = IntegrationOrchestrator(config=config)
    await orch.start()

    try:
        while orch._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await orch.stop()


def main() -> None:
    """CLI entry point for the Integration Orchestrator."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AGI-HPC Integration Orchestrator (Phase 4)"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to integration_config.yaml",
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
        logger.info("[orchestrator] interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
