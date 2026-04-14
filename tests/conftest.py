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

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _force_gc_after_test():
    """Force garbage collection after each test to clean up dangling references."""
    yield
    gc.collect()


def pytest_sessionfinish(session, exitstatus):
    """Force-close any lingering asyncio event loops at session end."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop) if hasattr(asyncio, "all_tasks") else []
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
    except RuntimeError:
        pass  # No event loop to clean up
