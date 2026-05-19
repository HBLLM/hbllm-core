"""Tests for the Cognitive Watchdog."""

import asyncio

import pytest

from hbllm.brain.autonomy.watchdog import CognitiveWatchdog, RecursionDetector


def test_recursion_detector():
    detector = RecursionDetector(history_size=10)

    # Normal execution
    assert not detector.record_execution("task_1", "hash1")
    assert not detector.record_execution("task_2", "hash2")

    # A -> A -> A (Immediate Repetition)
    assert not detector.record_execution("task_A", "hashA")
    assert not detector.record_execution("task_A", "hashA")
    assert detector.record_execution("task_A", "hashA") is True

    detector = RecursionDetector(history_size=10)

    # A -> B -> A -> B (Oscillating)
    assert not detector.record_execution("task_A", "hashA")
    assert not detector.record_execution("task_B", "hashB")
    assert not detector.record_execution("task_A", "hashA")
    assert detector.record_execution("task_B", "hashB") is True


@pytest.mark.asyncio
async def test_watchdog_timeout():
    watchdog = CognitiveWatchdog(max_deliberation_ms=100)  # 100ms

    async def slow_task():
        await asyncio.sleep(0.5)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await watchdog.execute_with_guard("slow", "hash", slow_task())


@pytest.mark.asyncio
async def test_watchdog_success():
    watchdog = CognitiveWatchdog(max_deliberation_ms=1000)

    async def fast_task():
        await asyncio.sleep(0.01)
        return "done"

    result = await watchdog.execute_with_guard("fast", "hash", fast_task())
    assert result == "done"
