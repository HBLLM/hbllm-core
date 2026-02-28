"""Tests for CircuitBreaker — state transitions and failure detection."""

import pytest
import time

from hbllm.network.circuit_breaker import (
    CircuitBreaker, CircuitBreakerRegistry,
    CircuitState, CircuitOpenError,
)


# ── Single CircuitBreaker ────────────────────────────────────────────────────

def test_initial_state():
    cb = CircuitBreaker("node_1")
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute()


def test_closed_after_success():
    cb = CircuitBreaker("node_1")
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_open_after_failures():
    cb = CircuitBreaker("node_1", failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED  # Not yet
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert not cb.can_execute()


def test_circuit_open_error():
    cb = CircuitBreaker("node_1", failure_threshold=1, recovery_timeout=60.0)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    with pytest.raises(CircuitOpenError) as exc_info:
        import asyncio
        asyncio.run(cb.call(lambda: None))

    assert "node_1" in str(exc_info.value)
    assert exc_info.value.time_until_retry > 0


def test_half_open_transition():
    cb = CircuitBreaker("node_1", failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    time.sleep(0.02)  # Wait for recovery timeout
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.can_execute()


def test_half_open_to_closed():
    cb = CircuitBreaker("node_1", failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.02)
    assert cb.state == CircuitState.HALF_OPEN

    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_to_open():
    cb = CircuitBreaker("node_1", failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.02)
    assert cb.state == CircuitState.HALF_OPEN

    cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_success_resets_failure_count():
    cb = CircuitBreaker("node_1", failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    # Should still be closed — success reset the counter
    assert cb.state == CircuitState.CLOSED


def test_manual_reset():
    cb = CircuitBreaker("node_1", failure_threshold=1)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute()


@pytest.mark.asyncio
async def test_call_success():
    cb = CircuitBreaker("node_1")

    async def ok():
        return 42

    result = await cb.call(ok)
    assert result == 42
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_call_failure():
    cb = CircuitBreaker("node_1", failure_threshold=2)

    async def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await cb.call(fail)

    with pytest.raises(ValueError):
        await cb.call(fail)

    assert cb.state == CircuitState.OPEN


def test_time_until_retry():
    cb = CircuitBreaker("node_1", failure_threshold=1, recovery_timeout=5.0)
    cb.record_failure()
    assert 4.0 < cb.time_until_retry <= 5.0


def test_repr():
    cb = CircuitBreaker("node_1")
    r = repr(cb)
    assert "node_1" in r
    assert "closed" in r


# ── Registry ────────────────────────────────────────────────────────────────

def test_registry_get_or_create():
    reg = CircuitBreakerRegistry()
    cb = reg.get("node_a")
    assert cb.node_id == "node_a"
    assert reg.get("node_a") is cb  # Same instance


def test_registry_open_circuits():
    reg = CircuitBreakerRegistry(failure_threshold=1)
    reg.get("ok_node")
    reg.get("bad_node").record_failure()

    open_list = reg.get_open_circuits()
    assert "bad_node" in open_list
    assert "ok_node" not in open_list


def test_registry_reset_all():
    reg = CircuitBreakerRegistry(failure_threshold=1)
    reg.get("n1").record_failure()
    reg.get("n2").record_failure()
    assert len(reg.get_open_circuits()) == 2

    reg.reset_all()
    assert len(reg.get_open_circuits()) == 0
