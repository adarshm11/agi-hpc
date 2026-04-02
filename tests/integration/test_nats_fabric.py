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
# software is provided "AS IS", without WARRANTIES or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Phase 0 integration test: NATS Event Fabric pub/sub round-trip.

Requires a running NATS server with JetStream enabled at localhost:4222.
Start with:  nats-server --jetstream
"""

from __future__ import annotations

import asyncio
import logging
import sys

# Allow running directly from the repo root
sys.path.insert(0, "src")

from agi.common.event import Event
from agi.core.events.nats_fabric import NatsEventFabric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def test_publish_subscribe() -> bool:
    """Publish an event and verify the subscriber receives it."""
    fabric = NatsEventFabric(servers=["nats://localhost:4222"])
    await fabric.connect()

    received: list[Event] = []
    received_event = asyncio.Event()

    async def handler(event: Event) -> None:
        received.append(event)
        received_event.set()

    subject = "agi.lh.request.chat"
    await fabric.subscribe(subject, handler)

    # Allow subscription to propagate
    await asyncio.sleep(0.3)

    # Publish
    event = Event.create(
        source="test",
        event_type="lh.request.chat",
        payload={"prompt": "Hello from Phase 0!"},
    )
    await fabric.publish(subject, event)
    logger.info("Published event id=%s to %s", event.id, subject)

    # Wait for delivery
    try:
        await asyncio.wait_for(received_event.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.error("FAIL: Timed out waiting for event delivery")
        await fabric.disconnect()
        return False

    assert len(received) == 1, f"Expected 1 event, got {len(received)}"
    rx = received[0]
    assert rx.id == event.id, f"Event id mismatch: {rx.id} != {event.id}"
    assert rx.source == "test"
    assert rx.type == "lh.request.chat"
    assert rx.payload["prompt"] == "Hello from Phase 0!"
    assert rx.trace_id == event.trace_id

    logger.info(
        "PASS: Received event id=%s source=%s type=%s",
        rx.id,
        rx.source,
        rx.type,
    )

    await fabric.disconnect()
    return True


async def test_request_reply() -> bool:
    """Send a request and verify the reply round-trip."""
    fabric = NatsEventFabric(servers=["nats://localhost:4222"])
    await fabric.connect()

    subject = "agi.safety.check.input"

    # Set up a responder on core NATS using the _rpc. prefix
    # (fabric.request() automatically adds _rpc. to avoid JetStream capture)
    async def responder(msg) -> None:
        req = Event.from_bytes(msg.data)
        reply = Event.create(
            source="safety",
            event_type="safety.check.result",
            payload={"approved": True, "request_id": req.id},
            trace_id=req.trace_id,
        )
        await fabric._nc.publish(msg.reply, reply.to_bytes())

    await fabric._nc.subscribe(f"_rpc.{subject}", cb=responder)
    await asyncio.sleep(0.3)

    request_event = Event.create(
        source="lh",
        event_type="safety.check.input",
        payload={"action": "generate_text"},
    )

    reply = await fabric.request(subject, request_event, timeout=5.0)
    assert reply.payload["approved"] is True
    assert reply.payload["request_id"] == request_event.id
    assert reply.trace_id == request_event.trace_id

    logger.info(
        "PASS: Request-reply round-trip ok. reply_id=%s approved=%s",
        reply.id,
        reply.payload["approved"],
    )

    await fabric.disconnect()
    return True


async def test_wildcard_subscribe() -> bool:
    """Subscribe with wildcard and receive events from multiple subjects."""
    fabric = NatsEventFabric(servers=["nats://localhost:4222"])
    await fabric.connect()

    received: list[Event] = []
    all_received = asyncio.Event()
    expected_count = 3

    async def handler(event: Event) -> None:
        received.append(event)
        if len(received) >= expected_count:
            all_received.set()

    # Subscribe to all memory subjects
    await fabric.subscribe("agi.memory.>", handler)
    await asyncio.sleep(0.3)

    # Publish to three different memory subjects
    subjects = [
        "agi.memory.store.semantic",
        "agi.memory.query.episodic",
        "agi.memory.store.procedural",
    ]
    for subj in subjects:
        ev = Event.create(source="test", event_type=subj, payload={"subject": subj})
        await fabric.publish(subj, ev)

    try:
        await asyncio.wait_for(all_received.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.error(
            "FAIL: Timed out. Received %d/%d events", len(received), expected_count
        )
        await fabric.disconnect()
        return False

    assert len(received) >= expected_count
    logger.info(
        "PASS: Wildcard subscribe received %d events across memory subjects",
        len(received),
    )

    await fabric.disconnect()
    return True


async def main() -> int:
    """Run all Phase 0 integration tests."""
    tests = [
        ("publish_subscribe", test_publish_subscribe),
        ("request_reply", test_request_reply),
        ("wildcard_subscribe", test_wildcard_subscribe),
    ]

    results: dict[str, bool] = {}
    for name, test_fn in tests:
        logger.info("--- Running: %s ---", name)
        try:
            passed = await test_fn()
        except Exception:
            logger.exception("FAIL: %s raised an exception", name)
            passed = False
        results[name] = passed

    logger.info("=" * 60)
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %s: %s", status, name)
    logger.info("=" * 60)
    logger.info("Overall: %s", "ALL PASSED" if all_passed else "SOME FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
