"""Event Normalizer — noise filtering, debouncing, and throttling for reality.

This layer sits between the raw RealityEventBus and the WorldStateEngine.
It enforces modality-based budgets, deduplicates spam, and coalesces
micro-events to prevent the cognitive layer from collapsing under event storms.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from hbllm.perception.event_log import EventLog
from hbllm.perception.reality_bus import PerceptionEvent, PerceptionModality

logger = logging.getLogger(__name__)


class EventNormalizer:
    """Filters, deduplicates, and limits reality events before state fusion.

    Responsibilities:
    1. Budgeting (Modality limits)
    2. Deduplication (Same signature within time window)
    3. Routing (To EventLog and WorldStateEngine)
    """

    def __init__(self, event_log: EventLog | None = None) -> None:
        self.event_log = event_log
        self._downstream_subscribers: list[Callable[[PerceptionEvent], Any]] = []

        # Budgeting tracking
        self._modality_counts: dict[str, int] = defaultdict(int)
        self._last_budget_reset: float = time.time()

        # Deduplication cache: signature -> last_seen_time
        self._dedup_cache: dict[str, float] = {}

        # Limits (events per minute)
        self.BUDGETS = {
            PerceptionModality.SYSTEM: float("inf"),  # Unlimited (trusted)
            PerceptionModality.APP: 300,  # Throttled (5/sec avg)
            PerceptionModality.SENSOR: 60,  # Heavily sampled (1/sec avg)
            PerceptionModality.INFERRED: 10,  # Very limited
        }

        # Deduplication windows (seconds) by modality
        self.DEDUP_WINDOWS = {
            PerceptionModality.SYSTEM: 0.1,
            PerceptionModality.APP: 1.0,
            PerceptionModality.SENSOR: 5.0,
            PerceptionModality.INFERRED: 10.0,
        }

    def subscribe(self, callback: Callable[[PerceptionEvent], Any]) -> None:
        """Subscribe to the NORMALIZED stream."""
        if callback not in self._downstream_subscribers:
            self._downstream_subscribers.append(callback)

    async def handle_raw_event(self, event: PerceptionEvent) -> None:
        """Process a raw event from the RealityEventBus."""
        now = time.time()

        # 1. Reset budgets if minute has passed
        if now - self._last_budget_reset > 60.0:
            self._modality_counts.clear()
            self._last_budget_reset = now
            # Also cleanup dedup cache
            self._dedup_cache = {k: v for k, v in self._dedup_cache.items() if now - v < 60.0}

        # 2. Enforce Budgets
        mod = event.modality
        if self._modality_counts[mod.value] >= self.BUDGETS[mod]:
            logger.debug("Dropped event %s: Budget exceeded for %s", event.event_id, mod)
            return

        # 3. Deduplication (coalescing)
        # Signature based on entity, type, and subtype
        sig = f"{event.entity_id}:{event.event_type}:{event.sub_type}"
        last_seen = self._dedup_cache.get(sig, 0.0)
        window = self.DEDUP_WINDOWS.get(mod, 1.0)

        if now - last_seen < window:
            # We already saw this exact signature very recently. Drop it.
            return

        # Mark as seen
        self._dedup_cache[sig] = now
        self._modality_counts[mod.value] += 1

        # 4. Success — Write to Log (Truth Ledger)
        if self.event_log:
            self.event_log.append(event)

        # 5. Route to WorldStateEngine / AutonomyCore
        for sub in self._downstream_subscribers:
            try:
                res = sub(event)
                if asyncio.iscoroutine(res):
                    asyncio.create_task(res)
            except Exception as e:
                logger.error("Error in EventNormalizer downstream subscriber: %s", e)
