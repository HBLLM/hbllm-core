"""
Circuit Breaker — prevents cascading failures by detecting unhealthy nodes.

States:
  CLOSED  → Normal operation, requests pass through
  OPEN    → Node is failing, requests are rejected immediately
  HALF_OPEN → Testing if node has recovered

Transitions:
  CLOSED → OPEN: After `failure_threshold` consecutive failures
  OPEN → HALF_OPEN: After `recovery_timeout` seconds
  HALF_OPEN → CLOSED: If test request succeeds
  HALF_OPEN → OPEN: If test request fails
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Coroutine
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    """States of a circuit breaker."""

    CLOSED = "closed"  # Normal — requests flow through
    OPEN = "open"  # Failing — requests rejected
    HALF_OPEN = "half_open"  # Testing — one request allowed through
    PARTIAL_OPEN = "partial_open"  # Recovering — partial traffic allowed


class CircuitOpenError(Exception):
    """Raised when a request is rejected because the circuit is open."""

    def __init__(self, node_id: str, time_until_retry: float):
        self.node_id = node_id
        self.time_until_retry = time_until_retry
        super().__init__(f"Circuit open for node '{node_id}'. Retry in {time_until_retry:.1f}s")


class CircuitBreaker:
    """
    Circuit breaker for a single node.

    Tracks failures and automatically opens the circuit when a threshold is hit.
    After a recovery timeout, allows a test request through (half-open state).
    """

    def __init__(
        self,
        node_id: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        max_recovery_timeout: float = 300.0,
        half_open_max_calls: int = 1,
        health_check: Callable[[], Coroutine[Any, Any, bool]] | None = None,
    ):
        self.node_id = node_id
        self.failure_threshold = failure_threshold
        self.base_recovery_timeout = recovery_timeout
        self.max_recovery_timeout = max_recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.health_check = health_check

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._current_recovery_timeout = self.base_recovery_timeout

        # Metrics
        self.total_failures = 0
        self.total_successes = 0
        self.last_state_change = time.time()

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic OPEN → HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._current_recovery_timeout:
                self._change_state(CircuitState.HALF_OPEN)
                self._half_open_calls = 0
                logger.info("Circuit for '%s': OPEN → HALF_OPEN (testing recovery)", self.node_id)
        return self._state

    def _change_state(self, new_state: CircuitState) -> None:
        self._state = new_state
        self.last_state_change = time.time()

    @property
    def time_until_retry(self) -> float:
        """Seconds until the circuit transitions to HALF_OPEN."""
        if self._state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self._current_recovery_timeout - elapsed)

    def can_execute(self) -> bool:
        """Check if a request can pass through the circuit."""
        state = self.state  # Triggers auto-transition
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        if state == CircuitState.PARTIAL_OPEN:
            # Allow 50% of traffic during recovery
            return random.random() < 0.5
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful request."""
        self.total_successes += 1
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._change_state(CircuitState.PARTIAL_OPEN)
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit for '%s': HALF_OPEN → PARTIAL_OPEN (recovering)", self.node_id)
        elif self._state == CircuitState.PARTIAL_OPEN:
            self._success_count += 1
            if self._success_count >= 5:  # fully recover after 5 successes
                self._change_state(CircuitState.CLOSED)
                self._failure_count = 0
                self._success_count = 0
                self._current_recovery_timeout = self.base_recovery_timeout
                logger.info("Circuit for '%s': PARTIAL_OPEN → CLOSED (recovered)", self.node_id)
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state in (CircuitState.HALF_OPEN, CircuitState.PARTIAL_OPEN):
            self._change_state(CircuitState.OPEN)
            self._apply_backoff()
            logger.warning("Circuit for '%s': %s → OPEN (still failing)", self.node_id, self._state)
        elif self._failure_count >= self.failure_threshold:
            self._change_state(CircuitState.OPEN)
            self._apply_backoff()
            logger.warning(
                "Circuit for '%s': CLOSED → OPEN (after %d failures)",
                self.node_id,
                self._failure_count,
            )

    def _apply_backoff(self) -> None:
        """Apply exponential backoff with jitter to recovery timeout."""
        # Double the timeout, cap at max
        new_timeout = min(self._current_recovery_timeout * 2.0, self.max_recovery_timeout)
        # Add jitter (up to 20%)
        jitter = new_timeout * 0.2 * random.random()
        self._current_recovery_timeout = new_timeout + jitter

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function through the circuit breaker.

        Raises CircuitOpenError if the circuit is open.
        """
        if not self.can_execute():
            raise CircuitOpenError(self.node_id, self.time_until_retry)

        if self._state == CircuitState.HALF_OPEN:
            if self.health_check is not None:
                is_healthy = await self.health_check()
                if not is_healthy:
                    self.record_failure()
                    raise CircuitOpenError(self.node_id, self.time_until_retry)
            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        self._change_state(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._current_recovery_timeout = self.base_recovery_timeout
        logger.info("Circuit for '%s' manually reset to CLOSED", self.node_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize circuit breaker state for persistence."""
        return {
            "node_id": self.node_id,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "half_open_calls": self._half_open_calls,
            "current_recovery_timeout": self._current_recovery_timeout,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_state_change": self.last_state_change,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        """Restore circuit breaker state from a persisted snapshot."""
        self._state = CircuitState(data["state"])
        self._failure_count = data.get("failure_count", 0)
        self._success_count = data.get("success_count", 0)
        self._last_failure_time = data.get("last_failure_time", 0.0)
        self._half_open_calls = data.get("half_open_calls", 0)
        self._current_recovery_timeout = data.get(
            "current_recovery_timeout", self.base_recovery_timeout
        )
        self.total_failures = data.get("total_failures", 0)
        self.total_successes = data.get("total_successes", 0)
        self.last_state_change = data.get("last_state_change", time.time())

    def __repr__(self) -> str:
        return (
            f"<CircuitBreaker node={self.node_id} state={self.state.value} "
            f"failures={self._failure_count}/{self.failure_threshold}>"
        )


class CircuitBreakerRegistry:
    """
    Manages circuit breakers for all nodes.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

    def get(self, node_id: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a node."""
        if node_id not in self._breakers:
            self._breakers[node_id] = CircuitBreaker(
                node_id=node_id,
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )
        return self._breakers[node_id]

    def get_open_circuits(self) -> list[str]:
        """Get node IDs with open circuits."""
        return [
            node_id
            for node_id, breaker in self._breakers.items()
            if breaker.state == CircuitState.OPEN
        ]

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def save_state(self, path: str | Path) -> None:
        """Persist all circuit breaker states to a JSON file."""
        state = {node_id: breaker.to_dict() for node_id, breaker in self._breakers.items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Saved %d circuit breaker states to %s", len(state), path)

    def load_state(self, path: str | Path) -> None:
        """Restore circuit breaker states from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.debug("No circuit breaker state file at %s, starting fresh", path)
            return

        try:
            with open(path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load circuit breaker state from %s: %s", path, e)
            return

        restored = 0
        for node_id, data in state.items():
            breaker = self.get(node_id)
            breaker.load_dict(data)
            restored += 1
        logger.info("Restored %d circuit breaker states from %s", restored, path)
