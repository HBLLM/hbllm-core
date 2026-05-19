"""Idempotency Tracking Engine.

Prevents duplicate actions if the cognitive runtime crashes or a delegation loop retries a task.
Generates an idempotency key and checks a lock-table before executing mutating OS actions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class IdempotencyEngine:
    """Manages idempotency keys to ensure exactly-once execution of mutating actions."""

    def __init__(self) -> None:
        # In-memory mock of a persistent lock-table (e.g. SQLite/Redis)
        self._executed_keys: dict[str, float] = {}

    def generate_key(self, goal_id: str, action: str, parameters: dict[str, Any]) -> str:
        """Generate a stable, unique hash for an action payload."""
        payload = json.dumps(
            {"goal_id": goal_id, "action": action, "parameters": parameters}, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def check_and_lock(self, idempotency_key: str) -> bool:
        """Check if an action has already been executed. If not, lock it.

        Returns True if safe to execute. Returns False if already executed (duplicate).
        """
        if idempotency_key in self._executed_keys:
            logger.warning(
                "Idempotency lock triggered for key %s. Skipping duplicate execution.",
                idempotency_key,
            )
            return False

        self._executed_keys[idempotency_key] = time.time()
        return True

    def clear_lock(self, idempotency_key: str) -> None:
        """Clear a lock if an action failed and needs to be legally retried."""
        if idempotency_key in self._executed_keys:
            del self._executed_keys[idempotency_key]
