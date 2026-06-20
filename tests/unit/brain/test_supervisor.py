"""Tests for the External Process Supervisor."""

import time

from hbllm.brain.autonomy.supervisor import ProcessSupervisor


def test_supervisor_heartbeat_timeout():
    recovered = False

    def recovery_cb():
        nonlocal recovered
        recovered = True

    # Use a very short timeout for the test
    supervisor = ProcessSupervisor(heartbeat_timeout_s=0.2, recovery_callback=recovery_cb)
    supervisor.start()

    # Keep heartbeat alive
    time.sleep(0.1)
    supervisor.heartbeat()
    assert not recovered

    # Let it hang
    time.sleep(1.5)

    assert recovered is True
    supervisor.stop()
