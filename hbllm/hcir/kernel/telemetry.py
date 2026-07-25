"""
Kernel Telemetry Service — Unified performance and resource tracking.

Collects micro-benchmarks, scheduler latencies, bus dispatch metrics, and memory
retrieval timings across all Cognitive OS subsystems.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricWindow:
    """Sliding window of timing samples for a specific subsystem metric."""

    name: str
    max_samples: int = 1000
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    total_count: int = 0
    total_time_ms: float = 0.0

    def record(self, value_ms: float) -> None:
        """Record a single latency observation in milliseconds."""
        self.samples.append(value_ms)
        self.total_count += 1
        self.total_time_ms += value_ms

    @property
    def avg_ms(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]


class KernelTelemetry:
    """
    Centralized telemetry collector owned by KernelServices.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, MetricWindow] = defaultdict(lambda: MetricWindow(name=""))
        self._counters: dict[str, int] = defaultdict(int)

    def record_latency(self, subsystem: str, operation: str, duration_ms: float) -> None:
        """Record an operation latency sample."""
        key = f"{subsystem}.{operation}"
        if key not in self._metrics:
            self._metrics[key] = MetricWindow(name=key)
        self._metrics[key].record(duration_ms)

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment an integer counter metric."""
        self._counters[name] += value

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of all active telemetry metrics."""
        return {
            "counters": dict(self._counters),
            "latencies": {
                name: {
                    "count": win.total_count,
                    "avg_ms": round(win.avg_ms, 3),
                    "p95_ms": round(win.p95_ms, 3),
                }
                for name, win in self._metrics.items()
            },
        }
