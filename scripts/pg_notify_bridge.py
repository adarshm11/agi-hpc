#!/usr/bin/env python3
"""PostgreSQL LISTEN/NOTIFY -> NATS bridge.

Listens on PostgreSQL notification channels and publishes
corresponding events to NATS subjects.

Channels:
  - new_episode     -> agi.events.new_episode
  - new_chunk       -> agi.events.new_chunk
  - confidence_update -> agi.events.confidence_update

Usage:
  python3 pg_notify_bridge.py [--pg-dsn DSN] [--nats-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys

import psycopg2
import psycopg2.extensions

logger = logging.getLogger(__name__)

# Channel -> NATS subject mapping
CHANNEL_MAP = {
    "new_episode": "agi.events.new_episode",
    "new_chunk": "agi.events.new_chunk",
    "confidence_update": "agi.events.confidence_update",
}

DEFAULT_PG_DSN = "dbname=atlas user=claude host=localhost"
DEFAULT_NATS_URL = "nats://localhost:4222"


class PgNotifyBridge:
    """Bridge PostgreSQL LISTEN/NOTIFY to NATS."""

    def __init__(self, pg_dsn: str, nats_url: str) -> None:
        self.pg_dsn = pg_dsn
        self.nats_url = nats_url
        self._running = False
        self._nc = None
        self._conn = None

    def _connect_pg(self) -> psycopg2.extensions.connection:
        """Connect to PostgreSQL and subscribe to channels."""
        conn = psycopg2.connect(self.pg_dsn)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        for channel in CHANNEL_MAP:
            cur.execute(f"LISTEN {channel};")
            logger.info("Listening on PG channel: %s", channel)
        return conn

    async def _connect_nats(self):
        """Connect to NATS."""
        try:
            import nats

            self._nc = await nats.connect(self.nats_url)
            logger.info("Connected to NATS at %s", self.nats_url)
        except ImportError:
            logger.warning(
                "nats-py not installed; will log events instead of publishing"
            )
            self._nc = None
        except Exception as e:
            logger.warning("NATS connection failed (%s); will log events instead", e)
            self._nc = None

    async def _publish(self, subject: str, payload: str) -> None:
        """Publish a message to NATS or log it."""
        if self._nc is not None:
            await self._nc.publish(subject, payload.encode("utf-8"))
            logger.info("Published to %s: %s", subject, payload[:120])
        else:
            logger.info("[dry-run] %s: %s", subject, payload[:200])

    async def run(self) -> None:
        """Main event loop."""
        self._running = True
        self._conn = self._connect_pg()
        await self._connect_nats()

        logger.info("Bridge running. Ctrl+C to stop.")

        loop = asyncio.get_event_loop()

        while self._running:
            # Use select() to wait for notifications (non-blocking with timeout)
            if self._conn.notifies:
                while self._conn.notifies:
                    notify = self._conn.notifies.pop(0)
                    channel = notify.channel
                    payload = notify.payload
                    subject = CHANNEL_MAP.get(channel, f"agi.events.{channel}")
                    await self._publish(subject, payload)
            else:
                # Wait for data with a timeout so we can check _running flag
                import select

                readable, _, _ = select.select([self._conn], [], [], 1.0)
                if readable:
                    self._conn.poll()

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._nc is not None:
            await self._nc.close()
        if self._conn is not None:
            self._conn.close()
        logger.info("Bridge stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PG LISTEN/NOTIFY -> NATS bridge")
    parser.add_argument("--pg-dsn", default=DEFAULT_PG_DSN, help="PostgreSQL DSN")
    parser.add_argument("--nats-url", default=DEFAULT_NATS_URL, help="NATS URL")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    bridge = PgNotifyBridge(args.pg_dsn, args.nats_url)

    async def run_with_signal():
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig, lambda: asyncio.ensure_future(bridge.stop())
                )
            except NotImplementedError:
                pass  # Windows
        await bridge.run()

    try:
        asyncio.run(run_with_signal())
    except KeyboardInterrupt:
        logger.info("Interrupted.")


if __name__ == "__main__":
    main()
