"""Tests for RollbackRegistry and ActionTransaction."""

import pytest

from hbllm.actions.rollback import (
    ActionTransaction,
    RollbackRegistry,
)


@pytest.fixture
def registry():
    return RollbackRegistry()


# ── Registry Tests ───────────────────────────────────────────────────────────


class TestRegistration:
    def test_register_handler(self, registry):
        registry.register("file.create", description="Create a file")
        assert registry.has_handler("file.create")
        assert not registry.has_handler("file.delete")

    def test_stats(self, registry):
        registry.register("a")
        registry.register("b")
        stats = registry.stats()
        assert "a" in stats["registered_handlers"]
        assert "b" in stats["registered_handlers"]


# ── Execute with Rollback Tests ──────────────────────────────────────────────


class TestExecuteWithRollback:
    @pytest.mark.asyncio
    async def test_successful_execution(self, registry):
        executed = []

        async def do_action(**kwargs):
            executed.append(kwargs)

        result = await registry.execute_with_rollback(
            "test.action",
            do_action,
            {"key": "value"},
        )
        assert result.success is True
        assert result.rolled_back is False
        assert len(executed) == 1

    @pytest.mark.asyncio
    async def test_failed_execution_triggers_rollback(self, registry):
        undo_called = []

        async def failing_action(**kwargs):
            raise RuntimeError("boom")

        async def undo_fn(pre_state):
            undo_called.append(pre_state)
            return True

        registry.register(
            "risky.action",
            snapshot_fn=lambda params: {"backup": True},
            undo_fn=undo_fn,
        )

        result = await registry.execute_with_rollback(
            "risky.action",
            failing_action,
            {"x": 1},
        )
        assert result.success is False
        assert result.rolled_back is True
        assert result.error == "boom"
        assert len(undo_called) == 1
        assert undo_called[0] == {"backup": True}

    @pytest.mark.asyncio
    async def test_failed_execution_no_handler(self, registry):
        """Failure without a registered undo handler."""

        async def failing_action(**kwargs):
            raise RuntimeError("fail")

        result = await registry.execute_with_rollback(
            "unknown.action",
            failing_action,
        )
        assert result.success is False
        assert result.rolled_back is False

    @pytest.mark.asyncio
    async def test_snapshot_captured(self, registry):
        state = {"counter": 5}

        registry.register(
            "increment",
            snapshot_fn=lambda params: {"old_value": state["counter"]},
        )

        async def increment(**kwargs):
            state["counter"] += 1

        result = await registry.execute_with_rollback("increment", increment)
        assert result.snapshot is not None
        assert result.snapshot.pre_state == {"old_value": 5}


# ── Undo Last Tests ──────────────────────────────────────────────────────────


class TestUndoLast:
    @pytest.mark.asyncio
    async def test_undo_last_action(self, registry):
        undo_calls = []

        async def undo(pre_state):
            undo_calls.append(pre_state)
            return True

        registry.register("do.thing", undo_fn=undo)

        async def do_thing(**kwargs):
            pass

        await registry.execute_with_rollback("do.thing", do_thing, tenant_id="t1")
        success = await registry.undo_last(tenant_id="t1")
        assert success is True
        assert len(undo_calls) == 1

    @pytest.mark.asyncio
    async def test_undo_empty_history(self, registry):
        success = await registry.undo_last(tenant_id="empty")
        assert success is False

    @pytest.mark.asyncio
    async def test_no_double_undo(self, registry):
        undo_calls = []

        async def undo(pre_state):
            undo_calls.append(1)
            return True

        registry.register("act", undo_fn=undo)

        async def act(**kwargs):
            pass

        await registry.execute_with_rollback("act", act, tenant_id="t1")
        await registry.undo_last(tenant_id="t1")
        # Second undo should find nothing undoable
        success = await registry.undo_last(tenant_id="t1")
        assert success is False
        assert len(undo_calls) == 1


# ── History Tests ────────────────────────────────────────────────────────────


class TestHistory:
    @pytest.mark.asyncio
    async def test_history_tracked_per_tenant(self, registry):
        async def noop(**kwargs):
            pass

        await registry.execute_with_rollback("a", noop, tenant_id="t1")
        await registry.execute_with_rollback("b", noop, tenant_id="t2")

        h1 = registry.get_history("t1")
        h2 = registry.get_history("t2")
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0].action_name == "a"
        assert h2[0].action_name == "b"


# ── Transaction Context Manager Tests ────────────────────────────────────────


class TestActionTransaction:
    @pytest.mark.asyncio
    async def test_committed_transaction(self, registry):
        registry.register("create", snapshot_fn=lambda p: {"snap": True})

        async with ActionTransaction(registry, "create", {"path": "/tmp"}) as txn:
            # Simulate work
            txn.commit()

        history = registry.get_history()
        assert len(history) == 1
        assert history[0].committed is True

    @pytest.mark.asyncio
    async def test_auto_rollback_on_exception(self, registry):
        undo_calls = []

        async def undo(pre_state):
            undo_calls.append(pre_state)
            return True

        registry.register("risky", snapshot_fn=lambda p: {"x": 1}, undo_fn=undo)

        async with ActionTransaction(registry, "risky") as _txn:
            raise ValueError("something went wrong")

        # Exception should be suppressed and undo called
        assert len(undo_calls) == 1

    @pytest.mark.asyncio
    async def test_no_rollback_after_commit(self, registry):
        undo_calls = []

        async def undo(pre_state):
            undo_calls.append(1)
            return True

        registry.register("safe", undo_fn=undo)

        async with ActionTransaction(registry, "safe") as txn:
            txn.commit()

        assert len(undo_calls) == 0
