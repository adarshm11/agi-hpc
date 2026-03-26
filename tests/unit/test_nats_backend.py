# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.core.events.nats_backend - NATS JetStream Backend."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from agi.core.events.nats_backend import (
    NatsBackendConfig,
    NatsStreamConfig,
)


class TestNatsBackendConfig:
    def test_default(self):
        cfg = NatsBackendConfig()
        assert isinstance(cfg.servers, list)
        assert len(cfg.servers) >= 1
        assert cfg.stream_name == "AGI_HPC_EVENTS"
        assert cfg.consumer_group == "agi-hpc"
        assert cfg.max_msgs == 1000000
        assert cfg.max_bytes == 1073741824
        assert cfg.ack_wait_seconds == 30
        assert cfg.max_deliver == 3
        assert cfg.batch_size == 10

    def test_custom(self):
        cfg = NatsBackendConfig(
            servers=["nats://custom:4222"],
            stream_name="CUSTOM",
            consumer_group="cg",
            batch_size=50,
        )
        assert cfg.servers == ["nats://custom:4222"]
        assert cfg.stream_name == "CUSTOM"
        assert cfg.batch_size == 50

    def test_consumer_name_default(self):
        assert NatsBackendConfig().consumer_name is None

    def test_max_age(self):
        assert NatsBackendConfig().max_age_seconds == 604800

    def test_fetch_timeout(self):
        assert NatsBackendConfig().fetch_timeout_seconds == 1.0


class TestNatsStreamConfig:
    def test_basic(self):
        cfg = NatsStreamConfig(name="TEST", subjects=["fabric.>"])
        assert cfg.name == "TEST"
        assert cfg.subjects == ["fabric.>"]
        assert cfg.retention == "limits"
        assert cfg.storage == "file"

    def test_custom(self):
        cfg = NatsStreamConfig(
            name="C",
            subjects=["e.>", "l.>"],
            retention="workqueue",
            storage="memory",
        )
        assert len(cfg.subjects) == 2
        assert cfg.retention == "workqueue"


def _make_nats_mocks():
    """Create properly configured NATS mock objects.

    ``nc.jetstream()`` is a sync method in the real nats-py library,
    so we use ``MagicMock`` for it rather than ``AsyncMock``.
    """
    nc = AsyncMock()
    js = AsyncMock()
    # jetstream() is synchronous — must return js directly, not a coroutine
    nc.jetstream = MagicMock(return_value=js)
    return nc, js


class TestNatsBackendInit:
    @patch("agi.core.events.nats_backend.nats", None)
    def test_no_nats_raises(self):
        from agi.core.events.nats_backend import NatsBackend

        with pytest.raises(RuntimeError, match="nats-py is required"):
            NatsBackend()

    @patch("agi.core.events.nats_backend.nats")
    def test_mocked_init(self, mock_nats):
        from agi.core.events.nats_backend import NatsBackend

        nc, js = _make_nats_mocks()
        mock_nats.connect = AsyncMock(return_value=nc)
        backend = NatsBackend(NatsBackendConfig(servers=["nats://test:4222"]))
        assert backend._nc is nc
        assert backend._js is js
        backend._loop.call_soon_threadsafe(backend._loop.stop)


class TestNatsBackendPublish:
    @patch("agi.core.events.nats_backend.nats")
    def test_publish(self, mock_nats):
        from agi.core.events.nats_backend import NatsBackend

        nc, js = _make_nats_mocks()
        ack = MagicMock(stream="S", seq=1)
        js.publish = AsyncMock(return_value=ack)
        mock_nats.connect = AsyncMock(return_value=nc)
        backend = NatsBackend()
        backend.publish("test.topic", {"key": "value"})
        js.publish.assert_called_once()
        args = js.publish.call_args[0]
        assert args[0] == "fabric.test.topic"
        assert json.loads(args[1].decode()) == {"key": "value"}
        backend._loop.call_soon_threadsafe(backend._loop.stop)


class TestNatsBackendSubscribe:
    @patch("agi.core.events.nats_backend.nats")
    def test_subscribe(self, mock_nats):
        from agi.core.events.nats_backend import NatsBackend

        nc, js = _make_nats_mocks()
        mock_nats.connect = AsyncMock(return_value=nc)
        backend = NatsBackend()
        # Bypass the async subscription setup which requires a real event loop
        with patch.object(backend, "_setup_subscription", new_callable=AsyncMock):
            handler = MagicMock()
            backend.subscribe("test.topic", handler)
            assert "test.topic" in backend._subscribers
            assert handler in backend._subscribers["test.topic"]
        backend._loop.call_soon_threadsafe(backend._loop.stop)

    @patch("agi.core.events.nats_backend.nats")
    def test_multiple_handlers(self, mock_nats):
        from agi.core.events.nats_backend import NatsBackend

        nc, js = _make_nats_mocks()
        mock_nats.connect = AsyncMock(return_value=nc)
        backend = NatsBackend()
        with patch.object(backend, "_setup_subscription", new_callable=AsyncMock):
            h1, h2 = MagicMock(), MagicMock()
            backend.subscribe("t", h1)
            backend.subscribe("t", h2)
            assert len(backend._subscribers["t"]) == 2
        backend._loop.call_soon_threadsafe(backend._loop.stop)
