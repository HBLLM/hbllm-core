"""Tests for Execution Verification Engine."""

import asyncio

import pytest

from hbllm.brain.embodiment.os_adapter import OSAdapter
from hbllm.brain.embodiment.verifier import ExecutionVerifier


class MockOSAdapter(OSAdapter):
    def __init__(self):
        self.file_exists = False

    def check_file_exists(self, filepath: str) -> bool:
        return self.file_exists


@pytest.mark.asyncio
async def test_verifier_polling_success():
    adapter = MockOSAdapter()
    verifier = ExecutionVerifier(adapter)

    # Background task to simulate physical delay
    async def simulate_file_creation():
        await asyncio.sleep(0.6)
        adapter.file_exists = True

    task = asyncio.create_task(simulate_file_creation())

    # Verify should poll and wait for the task to complete
    success = await verifier.verify_file_creation("fake.txt", max_wait_s=2.0)
    assert success is True

    await task


@pytest.mark.asyncio
async def test_verifier_polling_timeout():
    adapter = MockOSAdapter()
    verifier = ExecutionVerifier(adapter)

    # File never created
    success = await verifier.verify_file_creation("fake.txt", max_wait_s=0.5)
    assert success is False
