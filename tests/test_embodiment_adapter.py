"""Tests for Device Reality Integration (OS Adapter)."""

import os
import tempfile

import pytest

from hbllm.brain.embodiment.os_adapter import OSAdapter


@pytest.mark.asyncio
async def test_os_adapter_sensors():
    adapter = OSAdapter()

    # Basic sensor checks
    assert 0.0 <= adapter.read_battery_level() <= 1.0
    assert 0.0 <= adapter.read_cpu_load() <= 1.0


def test_os_adapter_file_exists():
    adapter = OSAdapter()

    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf_name = tf.name

    try:
        assert adapter.check_file_exists(tf_name) is True
        assert adapter.check_file_exists("/path/to/nowhere/fake_file.txt") is False
    finally:
        os.remove(tf_name)
