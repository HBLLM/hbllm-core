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

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"  # Normal — requests flow through
    OPEN = "open"  # Failing — requests rejected
    HALF_OPEN = "half_open"  # Testing — one request allowed through


class CircuitOpenError(Exception):
    """Raised when a request is rejected because the circuit is open."""

    def __init__(self, node_id: str, time_until_retry: float):
        self.node_id = node_id
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit open for node '{node_id}'. "
            f"Retry in {time_until_retry:.1f}s"
        )


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
        half_open_max_calls: int = 1,
    ):
        self.node_id = node_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic OPEN → HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit for '%s': OPEN → HALF_OPEN (testing recovery)", self.node_id)
        return self._state

    @property
    def time_until_retry(self) -> float:
        """Seconds until the circuit transitions to HALF_OPEN."""
        if self._state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self.recovery_timeout - elapsed)

    def can_execute(self) -> bool:
        """Check if a request can pass through the circuit."""
        state = self.state  # Triggers auto-transition
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit for '%s': HALF_OPEN → CLOSED (recovered)", self.node_id)
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit for '%s': HALF_OPEN → OPEN (still failing)", self.node_id)
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit for '%s': CLOSED → OPEN (after %d failures)",
                self.node_id,
                self._failure_count,
            )

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
            self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info("Circuit for '%s' manually reset to CLOSED", self.node_id)

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
