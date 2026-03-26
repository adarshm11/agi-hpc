# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
DHT Observability infrastructure.

Provides:
- Prometheus metrics for DHT monitoring
- Structured logging with correlation IDs
- Distributed tracing with OpenTelemetry-compatible spans
- Decorators for tracking DHT operations
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

from agi.lh.observability import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed Tracing
# ---------------------------------------------------------------------------


@dataclass
class SpanContext:
    """OpenTelemetry-compatible span context for distributed tracing.

    Attributes:
        trace_id: Unique identifier for the trace
        span_id: Unique identifier for this span
        operation: Name of the operation being traced
        start_time: Monotonic start time of the span
        attributes: Key-value attributes attached to the span
        status: Span status (ok, error)
        end_time: Monotonic end time of the span (None if still active)
    """

    trace_id: str
    span_id: str
    operation: str
    start_time: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    end_time: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration of the span in milliseconds, or None if not yet ended."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class DHTTracer:
    """OpenTelemetry-compatible tracer for DHT operations.

    Provides span-based distributed tracing to track the lifecycle
    and performance of DHT operations across cluster nodes.

    Usage:
        tracer = DHTTracer(service_name="dht-node-1")

        with tracer.trace("put", key="mykey") as span:
            span.attributes["replicas"] = 3
    """

    def __init__(self, service_name: str = "dht") -> None:
        """Initialize the tracer.

        Args:
            service_name: Name of the service for trace attribution
        """
        self._service_name = service_name
        self._traces: List[SpanContext] = []
        self._max_traces = 10000

    def start_span(self, operation: str, **attributes: Any) -> SpanContext:
        """Start a new tracing span.

        Args:
            operation: Name of the operation being traced
            **attributes: Key-value attributes to attach to the span

        Returns:
            A new SpanContext representing the started span
        """
        span = SpanContext(
            trace_id=uuid.uuid4().hex[:32],
            span_id=uuid.uuid4().hex[:16],
            operation=operation,
            start_time=time.monotonic(),
            attributes={
                "service": self._service_name,
                **attributes,
            },
        )
        logger.debug(
            "[dht][trace] started span operation=%s trace_id=%s",
            operation,
            span.trace_id,
        )
        return span

    def end_span(self, span: SpanContext, status: str = "ok") -> None:
        """End a tracing span and record it.

        Args:
            span: The span to end
            status: Final status of the span (ok, error)
        """
        span.end_time = time.monotonic()
        span.status = status

        # Store completed span
        self._traces.append(span)
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces :]

        logger.debug(
            "[dht][trace] ended span operation=%s status=%s duration_ms=%.2f",
            span.operation,
            span.status,
            span.duration_ms or 0,
        )

    @contextmanager
    def trace(
        self, operation: str, **attributes: Any
    ) -> Generator[SpanContext, None, None]:
        """Context manager for tracing an operation.

        Automatically starts and ends a span, setting error status
        if an exception occurs.

        Args:
            operation: Name of the operation being traced
            **attributes: Key-value attributes to attach to the span

        Yields:
            The active SpanContext
        """
        span = self.start_span(operation, **attributes)
        try:
            yield span
        except Exception:
            self.end_span(span, status="error")
            raise
        else:
            self.end_span(span, status="ok")

    def get_traces(self, limit: int = 100) -> List[SpanContext]:
        """Get recently completed traces.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of completed SpanContext objects, most recent first
        """
        return list(reversed(self._traces[-limit:]))


# ---------------------------------------------------------------------------
# DHT Metrics Registry
# ---------------------------------------------------------------------------


class DHTMetrics:
    """
    Metrics registry for DHT service.

    Provides pre-defined Prometheus-compatible metrics for monitoring
    DHT health, performance, and operational characteristics.

    Usage:
        metrics = DHTMetrics()
        metrics.dht_operations_total.inc(operation="put", status="success")
        metrics.dht_operation_duration_seconds.observe(0.015, operation="put")
    """

    def __init__(self) -> None:
        # Operation metrics
        self.dht_operations_total = Counter(
            "dht_operations_total",
            "Total number of DHT operations",
            labels=["operation", "status"],
        )

        self.dht_operation_duration_seconds = Histogram(
            "dht_operation_duration_seconds",
            "Duration of DHT operations in seconds",
            labels=["operation"],
        )

        # Replication metrics
        self.dht_replication_total = Counter(
            "dht_replication_total",
            "Total number of replication events",
            labels=["status"],
        )

        self.dht_replication_lag_seconds = Histogram(
            "dht_replication_lag_seconds",
            "Replication lag in seconds",
        )

        # Cluster state metrics
        self.dht_keys_total = Gauge(
            "dht_keys_total",
            "Total number of keys stored in the DHT",
        )

        self.dht_nodes_total = Gauge(
            "dht_nodes_total",
            "Total number of nodes in the DHT cluster",
        )

        self.dht_storage_bytes = Gauge(
            "dht_storage_bytes",
            "Total storage used in bytes",
        )

        # Error metrics
        self.dht_errors_total = Counter(
            "dht_errors_total",
            "Total number of DHT errors by type",
            labels=["error_type"],
        )

    def to_prometheus_format(self) -> str:
        """Export all DHT metrics in Prometheus text exposition format.

        Returns:
            Prometheus-compatible metrics string
        """
        lines: List[str] = []

        def _format_labels(label_names: list, label_values: tuple) -> str:
            if not label_names:
                return "{}"
            pairs = [
                f'{n}="{v}"' for n, v in zip(label_names, label_values, strict=True)
            ]
            return "{" + ",".join(pairs) + "}"

        def format_metric(metric: Any, metric_type: str) -> None:
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} {metric_type}")

            if isinstance(metric, Counter):
                for labels, value in metric.collect().items():
                    label_str = _format_labels(metric.labels, labels)
                    lines.append(f"{metric.name}{label_str} {value}")

            elif isinstance(metric, Gauge):
                with metric._lock:
                    for labels, value in metric._values.items():
                        label_str = _format_labels(metric.labels, labels)
                        lines.append(f"{metric.name}{label_str} {value}")

            elif isinstance(metric, Histogram):
                with metric._lock:
                    for labels, buckets in metric._counts.items():
                        label_str = _format_labels(metric.labels, labels)
                        for bucket, count in buckets.items():
                            le = "+Inf" if bucket == float("inf") else str(bucket)
                            if label_str != "{}":
                                lines.append(
                                    f'{metric.name}_bucket{{{label_str[1:-1]},le="{le}"}} {count}'
                                )
                            else:
                                lines.append(
                                    f'{metric.name}_bucket{{le="{le}"}} {count}'
                                )
                        lines.append(
                            f"{metric.name}_sum{label_str} {metric._sums.get(labels, 0)}"
                        )
                        lines.append(
                            f"{metric.name}_count{label_str} {metric._totals.get(labels, 0)}"
                        )

        format_metric(self.dht_operations_total, "counter")
        format_metric(self.dht_operation_duration_seconds, "histogram")
        format_metric(self.dht_replication_total, "counter")
        format_metric(self.dht_replication_lag_seconds, "histogram")
        format_metric(self.dht_keys_total, "gauge")
        format_metric(self.dht_nodes_total, "gauge")
        format_metric(self.dht_storage_bytes, "gauge")
        format_metric(self.dht_errors_total, "counter")

        return "\n".join(lines)


# Global metrics instance
dht_metrics = DHTMetrics()

# Global tracer instance
dht_tracer = DHTTracer()


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def track_dht_operation(operation: str) -> Callable[[F], F]:
    """Decorator to track DHT operation metrics and tracing.

    Records operation count, duration, and error metrics. Also creates
    a trace span for the decorated operation.

    Usage:
        @track_dht_operation("put")
        async def put(self, key, value):
            ...

    Args:
        operation: Name of the DHT operation (e.g. "put", "get", "delete")

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            status = "success"

            span = dht_tracer.start_span(operation)

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                dht_metrics.dht_errors_total.inc(
                    error_type=type(e).__name__,
                )
                span.attributes["error"] = str(e)
                raise
            finally:
                duration = time.monotonic() - start_time
                dht_metrics.dht_operations_total.inc(
                    operation=operation,
                    status=status,
                )
                dht_metrics.dht_operation_duration_seconds.observe(
                    duration,
                    operation=operation,
                )
                dht_tracer.end_span(
                    span, status="ok" if status == "success" else "error"
                )

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            status = "success"

            span = dht_tracer.start_span(operation)

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                dht_metrics.dht_errors_total.inc(
                    error_type=type(e).__name__,
                )
                span.attributes["error"] = str(e)
                raise
            finally:
                duration = time.monotonic() - start_time
                dht_metrics.dht_operations_total.inc(
                    operation=operation,
                    status=status,
                )
                dht_metrics.dht_operation_duration_seconds.observe(
                    duration,
                    operation=operation,
                )
                dht_tracer.end_span(
                    span, status="ok" if status == "success" else "error"
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    return decorator
