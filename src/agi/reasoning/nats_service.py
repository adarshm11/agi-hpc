# AGI-HPC Project — Divine Council NATS Service
# Copyright (c) 2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Divine Council NATS bridge — subscribes to agi.ego.deliberate,
publishes heartbeat to agi.meta.monitor.ego.

Uses the same NatsEventFabric as other AGI-HPC services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from agi.common.event import Event
from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig

logger = logging.getLogger(__name__)

EGO_URL = "http://localhost:8084"
HEARTBEAT_S = 30


async def _check_health() -> bool:
    import urllib.request

    try:
        r = urllib.request.urlopen(f"{EGO_URL}/health", timeout=3)
        return r.status == 200
    except Exception:
        return False


async def _handle_deliberate(event: Event) -> None:
    """Handle a deliberation request."""
    logger.info("[ego-nats] deliberation request: %s", str(event.payload)[:80])


async def run():
    fabric = NatsEventFabric(config=NatsFabricConfig(servers=["nats://localhost:4222"]))
    await fabric.connect()
    logger.info("[ego-nats] connected")

    await fabric.subscribe("agi.ego.deliberate", _handle_deliberate)
    logger.info("[ego-nats] subscribed to agi.ego.deliberate")

    # Heartbeat
    while True:
        healthy = await _check_health()
        hb = Event.create(
            "ego",
            "agi.meta.monitor.ego",
            {
                "service": "divine_council",
                "status": "online" if healthy else "offline",
                "ts": time.time(),
            },
        )
        await fabric.publish("agi.meta.monitor.ego", hb)
        await asyncio.sleep(HEARTBEAT_S)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger.info("[ego-nats] starting Divine Council NATS bridge")
    asyncio.run(run())
