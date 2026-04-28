"""
Shared test fixtures and session-level cleanup.

Forces clean exit after all tests by ensuring no orphaned asyncio
tasks or event loops prevent pytest from terminating.
"""

from __future__ import annotations

import asyncio
import gc
import logging

import pytest
import pytest_asyncio

from hbllm.network.bus import InProcessBus

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _force_gc_after_test():
    """Force garbage collection after each test to clean up dangling references."""
    yield
    gc.collect()


@pytest_asyncio.fixture(autouse=True)
async def _force_task_cleanup():
    """Cancel all stray tasks at the end of each async test to prevent hanging."""
    yield

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    if tasks:
        # Use a timeout to prevent gather itself from hanging indefinitely
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, Exception):
            pass  # Tasks that won't cancel in 5s are orphaned — let them die


@pytest_asyncio.fixture
async def bus():
    """A standard InProcessBus that is cleanly shut down after the test."""
    test_bus = InProcessBus()
    await test_bus.start()
    yield test_bus
    await test_bus.stop()
