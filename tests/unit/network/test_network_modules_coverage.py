"""Unit tests for network modules — durable_bus, session_migration."""

import time

import pytest

from hbllm.network.durable_bus import DurableBus, DurableMessage, MessageStatus


class TestMessageStatus:
    def test_pending(self):
        assert MessageStatus.PENDING == "pending"

    def test_delivered(self):
        assert MessageStatus.DELIVERED == "delivered"

    def test_failed(self):
        assert MessageStatus.FAILED == "failed"


class TestDurableMessage:
    def test_creation(self):
        msg = DurableMessage(
            id="test-1",
            topic="test.topic",
            payload_json='{"key": "value"}',
            status=MessageStatus.PENDING,
        )
        assert msg.id == "test-1"
        assert msg.status == MessageStatus.PENDING

    def test_default_status(self):
        msg = DurableMessage(
            id="test-2",
            topic="test.topic",
            payload_json="{}",
        )
        assert msg.status == MessageStatus.PENDING
        assert msg.attempts == 0


class TestDurableBus:
    def test_init(self):
        bus = DurableBus()
        assert bus is not None

    def test_init_with_params(self):
        bus = DurableBus(max_retries=5, base_delay=2.0)
        assert bus is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, tmp_path):
        bus = DurableBus(db_path=str(tmp_path / "test.db"))
        await bus.start()
        await bus.stop()

    @pytest.mark.asyncio
    async def test_has_subscribers_empty(self, tmp_path):
        bus = DurableBus(db_path=str(tmp_path / "test.db"))
        await bus.start()
        assert bus.has_subscribers("no.topic") is False
        await bus.stop()

    @pytest.mark.asyncio
    async def test_stats(self, tmp_path):
        bus = DurableBus(db_path=str(tmp_path / "test.db"))
        await bus.start()
        stats = bus.stats()
        assert stats is not None
        await bus.stop()


# ── Session Migration ────────────────────────────────────────────────────────

from hbllm.network.session_migration import SessionMigrationManager, SessionSnapshot


class TestSessionSnapshot:
    def test_creation(self):
        snap = SessionSnapshot(
            id="sess-1",
            tenant_id="t1",
        )
        assert snap.id == "sess-1"
        assert snap.tenant_id == "t1"

    def test_compute_checksum(self):
        snap = SessionSnapshot(id="sess-1", tenant_id="t1")
        checksum = snap.compute_checksum()
        assert isinstance(checksum, str)
        assert len(checksum) > 0

    def test_verify_integrity(self):
        snap = SessionSnapshot(id="sess-1", tenant_id="t1")
        snap.checksum = snap.compute_checksum()
        assert snap.verify_integrity() is True

    def test_is_expired_true(self):
        snap = SessionSnapshot(
            id="sess-1",
            tenant_id="t1",
            expires_at=time.time() - 100,
        )
        assert snap.is_expired() is True

    def test_is_expired_false(self):
        snap = SessionSnapshot(
            id="sess-1",
            tenant_id="t1",
            expires_at=time.time() + 3600,
        )
        assert snap.is_expired() is False

    def test_to_dict(self):
        snap = SessionSnapshot(id="sess-1", tenant_id="t1")
        d = snap.to_dict()
        assert d["id"] == "sess-1"

    def test_from_dict(self):
        snap = SessionSnapshot(id="sess-1", tenant_id="t1")
        d = snap.to_dict()
        restored = SessionSnapshot.from_dict(d)
        assert restored.id == "sess-1"

    def test_json_round_trip(self):
        snap = SessionSnapshot(id="sess-1", tenant_id="t1")
        json_str = snap.to_json()
        restored = SessionSnapshot.from_json(json_str)
        assert restored.id == snap.id


class TestSessionMigrationManager:
    def test_init(self):
        mgr = SessionMigrationManager()
        assert mgr is not None
