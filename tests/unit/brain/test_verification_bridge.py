"""Unit tests for ActionVerificationBridge — execute → verify → correct loop."""

import pytest

from hbllm.brain.autonomy.verification_bridge import ActionVerificationBridge


class TestActionVerificationBridge:
    def test_init(self):
        bridge = ActionVerificationBridge(task_graph=None, world_state=None, bus=None)
        assert bridge.check_interval_s == 5.0
        assert bridge._running is False

    def test_init_custom_interval(self):
        bridge = ActionVerificationBridge(
            task_graph=None,
            world_state=None,
            bus=None,
            check_interval_s=10.0,
        )
        assert bridge.check_interval_s == 10.0

    def test_stats_initial(self):
        bridge = ActionVerificationBridge(task_graph=None, world_state=None, bus=None)
        stats = bridge.stats()
        assert stats["verification_checks"] == 0
        assert stats["successes"] == 0
        assert stats["corrections"] == 0

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        bridge = ActionVerificationBridge(task_graph=None, world_state=None, bus=None)
        # Should not raise
        await bridge.stop()

    def test_infer_verification_rule_iot(self):
        """Auto-generate verification rule from IoT action."""

        class MockTask:
            action_type = "iot.command"
            action_payload = {"device_id": "kitchen_light", "expected_state": "on"}

        rule = ActionVerificationBridge._infer_verification_rule(MockTask())
        assert rule is not None
        assert rule.entity_id == "kitchen_light"
        assert rule.property_name == "state"
        assert rule.expected_value == "on"

    def test_infer_verification_rule_no_iot(self):
        """No rule for non-IoT actions without explicit metadata."""

        class MockTask:
            action_type = "web.search"
            action_payload = {"query": "test"}

        rule = ActionVerificationBridge._infer_verification_rule(MockTask())
        assert rule is None

    def test_infer_verification_rule_empty(self):
        """No rule when action_type is empty."""

        class MockTask:
            action_type = ""
            action_payload = {}

        rule = ActionVerificationBridge._infer_verification_rule(MockTask())
        assert rule is None
