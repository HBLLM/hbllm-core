"""Tests for ActionTransaction and RollbackRegistry."""

import pytest

from hbllm.actions.rollback import (
    ActionTransaction,
    RollbackRegistry,
    StateSnapshot,
    TransactionStatus,
)


class TestRollbackRegistry:
    """Tests for action → undo mapping."""

    @pytest.fixture
    def registry(self):
        return RollbackRegistry()

    def test_reversible_actions_count(self, registry):
        reversible = registry.list_reversible()
        assert len(reversible) >= 10

    def test_file_create_undo(self, registry):
        undo = registry.get_undo("file.create")
        assert undo is not None
        assert undo["action"] == "file.delete"

    def test_light_on_off(self, registry):
        assert registry.is_reversible("iot.light.on")
        undo = registry.get_undo("iot.light.on")
        assert undo["action"] == "iot.light.off"

    def test_lock_undo(self, registry):
        undo = registry.get_undo("iot.lock.unlock")
        assert undo["action"] == "iot.lock.lock"

    def test_notification_irreversible(self, registry):
        assert not registry.is_reversible("system.notification.send")

    def test_unknown_action_none(self, registry):
        assert registry.get_undo("totally.unknown") is None
        assert not registry.is_reversible("totally.unknown")

    def test_custom_registration(self, registry):
        registry.register("custom.deploy", "custom.rollback")
        assert registry.is_reversible("custom.deploy")
        undo = registry.get_undo("custom.deploy")
        assert undo["action"] == "custom.rollback"

    def test_build_undo_params(self, registry):
        params = registry.build_undo_params(
            "iot.light.on",
            {"device_id": "living_room"},
        )
        assert params == {"device_id": "living_room"}

    def test_build_undo_params_with_state(self, registry):
        snap = StateSnapshot(key="temp", value=21)
        params = registry.build_undo_params(
            "iot.thermostat.set",
            {"device_id": "hvac"},
            pre_state=snap,
        )
        assert params is not None
        assert params.get("value") == 21

    def test_stats(self, registry):
        s = registry.stats()
        assert s["total_registered"] >= 12
        assert s["reversible"] >= 10


class TestActionTransaction:
    """Tests for multi-step transaction execution."""

    @pytest.fixture
    def registry(self):
        return RollbackRegistry()

    @pytest.mark.asyncio
    async def test_successful_transaction(self, registry):
        """All steps complete → COMPLETED status."""
        executed = []

        async def executor(action, params):
            executed.append(action)
            return {"ok": True}

        tx = ActionTransaction("tx-1", registry, executor=executor)
        tx.add_step("iot.light.on", {"device_id": "living_room"})
        tx.add_step("iot.light.on", {"device_id": "kitchen"})

        result = await tx.execute()
        assert result.status == TransactionStatus.COMPLETED
        assert len(executed) == 2

    @pytest.mark.asyncio
    async def test_failure_triggers_rollback(self, registry):
        """Step failure → previous steps are rolled back."""
        call_log = []

        async def executor(action, params):
            call_log.append(action)
            if action == "iot.thermostat.set":
                raise RuntimeError("Thermostat offline")
            return {"ok": True}

        tx = ActionTransaction("tx-2", registry, executor=executor)
        tx.add_step("iot.light.on", {"device_id": "lr"})
        tx.add_step("iot.thermostat.set", {"device_id": "hvac", "temp": 22})

        result = await tx.execute()
        assert result.status == TransactionStatus.ROLLED_BACK
        # Light should have been rolled back (light.off)
        assert "iot.light.off" in call_log

    @pytest.mark.asyncio
    async def test_dry_run_without_executor(self, registry):
        """No executor → dry run mode."""
        tx = ActionTransaction("tx-3", registry)
        tx.add_step("test.action", {"param": "value"})
        result = await tx.execute()
        assert result.status == TransactionStatus.COMPLETED
        assert result.steps[0].result == {"dry_run": True, "action": "test.action"}

    @pytest.mark.asyncio
    async def test_manual_rollback(self, registry):
        """Manual rollback of completed steps."""
        call_log = []

        async def executor(action, params):
            call_log.append(action)

        tx = ActionTransaction("tx-4", registry, executor=executor)
        tx.add_step("iot.light.on", {"device_id": "lr"})
        await tx.execute()

        assert tx.status == TransactionStatus.COMPLETED
        await tx.rollback_all()
        assert "iot.light.off" in call_log

    def test_to_dict(self, registry):
        tx = ActionTransaction("tx-5", registry)
        tx.add_step("test", {"a": 1})
        d = tx.to_dict()
        assert d["transaction_id"] == "tx-5"
        assert len(d["steps"]) == 1

    def test_stats(self, registry):
        tx = ActionTransaction("tx-6", registry)
        tx.add_step("test1")
        tx.add_step("test2")
        s = tx.stats()
        assert s["total_steps"] == 2
        assert s["status"] == "pending"
