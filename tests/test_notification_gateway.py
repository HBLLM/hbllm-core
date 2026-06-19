"""Tests for NotificationGateway — proactive notification system."""

import pytest

from hbllm.serving.notifications import (
    InMemoryBackend,
    Notification,
    NotificationCategory,
    NotificationGateway,
    NotificationPriority,
)


class TestNotification:
    def test_defaults(self):
        n = Notification()
        assert n.id
        assert not n.is_read
        assert not n.delivered

    def test_to_dict(self):
        n = Notification(title="test", priority=NotificationPriority.HIGH)
        d = n.to_dict()
        assert d["title"] == "test"
        assert d["priority"] == "high"

    def test_mark_read(self):
        import time

        n = Notification()
        assert not n.is_read
        n.read_at = time.time()
        assert n.is_read


class TestNotificationGateway:
    def test_push(self):
        gw = NotificationGateway()
        n = gw.push("t1", title="Hello", body="World")
        assert n.tenant_id == "t1"
        assert n.title == "Hello"

    def test_get_unread(self):
        gw = NotificationGateway()
        gw.push("t1", title="N1")
        gw.push("t1", title="N2")
        unread = gw.get_unread("t1")
        assert len(unread) == 2

    def test_get_unread_empty(self):
        gw = NotificationGateway()
        assert gw.get_unread("nonexistent") == []

    def test_mark_read(self):
        gw = NotificationGateway()
        n = gw.push("t1", title="test")
        assert gw.unread_count("t1") == 1
        gw.mark_read("t1", n.id)
        assert gw.unread_count("t1") == 0

    def test_mark_all_read(self):
        gw = NotificationGateway()
        gw.push("t1", title="N1")
        gw.push("t1", title="N2")
        gw.push("t1", title="N3")
        count = gw.mark_all_read("t1")
        assert count == 3
        assert gw.unread_count("t1") == 0

    def test_tenant_isolation(self):
        gw = NotificationGateway()
        gw.push("t1", title="T1 only")
        gw.push("t2", title="T2 only")
        assert gw.unread_count("t1") == 1
        assert gw.unread_count("t2") == 1

    def test_priority(self):
        gw = NotificationGateway()
        gw.push("t1", title="low", priority=NotificationPriority.SUGGESTION)
        gw.push("t1", title="high", priority=NotificationPriority.CRITICAL)
        unread = gw.get_unread("t1")
        assert len(unread) == 2

    def test_category_filter(self):
        gw = NotificationGateway()
        gw.push("t1", title="goal", category=NotificationCategory.GOAL)
        gw.push("t1", title="system", category=NotificationCategory.SYSTEM)
        goals = gw.get_unread("t1", category=NotificationCategory.GOAL)
        assert len(goals) == 1
        assert goals[0].title == "goal"

    def test_max_per_tenant(self):
        gw = NotificationGateway(max_per_tenant=5)
        for i in range(10):
            gw.push("t1", title=f"N{i}")
        all_notifs = gw.get_all("t1", limit=100)
        assert len(all_notifs) <= 5

    def test_callback(self):
        gw = NotificationGateway()
        received = []
        gw.on_notification("t1", lambda n: received.append(n))
        gw.push("t1", title="callback test")
        assert len(received) == 1
        assert received[0].title == "callback test"

    def test_remove_callback(self):
        gw = NotificationGateway()
        received = []
        cb = lambda n: received.append(n)  # noqa: E731
        gw.on_notification("t1", cb)
        gw.remove_callback("t1", cb)
        gw.push("t1", title="should not fire")
        assert len(received) == 0

    def test_clear(self):
        gw = NotificationGateway()
        gw.push("t1", title="test")
        gw.clear("t1")
        assert gw.unread_count("t1") == 0

    def test_stats(self):
        gw = NotificationGateway()
        gw.push("t1", title="N1")
        gw.push("t2", title="N2")
        stats = gw.stats()
        assert stats["tenant_count"] == 2
        assert stats["total_notifications"] == 2
        assert stats["total_unread"] == 2

    @pytest.mark.asyncio
    async def test_deliver_pending(self):
        gw = NotificationGateway(default_backend=InMemoryBackend())
        gw.push("t1", title="test")
        delivered = await gw.deliver_pending("t1")
        assert delivered == 1
