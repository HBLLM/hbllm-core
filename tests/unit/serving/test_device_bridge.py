"""Unit tests for DeviceBridge — cross-device session continuity."""

import pytest

from hbllm.serving.device_bridge import DeviceBridge, DeviceInfo, SessionHandoff


class TestDeviceInfo:
    def test_defaults(self):
        d = DeviceInfo(device_id="phone-1", tenant_id="user1")
        assert d.device_id == "phone-1"
        assert d.device_type == "unknown"
        assert d.is_active is True
        assert d.push_token is None

    def test_is_stale(self):
        import time

        d = DeviceInfo(device_id="old", tenant_id="user1")
        d.last_heartbeat = time.time() - 600  # 10 minutes ago
        assert d.is_stale is True

    def test_not_stale(self):
        d = DeviceInfo(device_id="fresh", tenant_id="user1")
        assert d.is_stale is False

    def test_age_seconds(self):
        d = DeviceInfo(device_id="test", tenant_id="user1")
        assert d.age_seconds < 1.0

    def test_with_capabilities(self):
        d = DeviceInfo(
            device_id="laptop",
            tenant_id="user1",
            device_type="desktop",
            capabilities=["audio", "display", "keyboard"],
        )
        assert "audio" in d.capabilities
        assert d.device_type == "desktop"


class TestSessionHandoff:
    def test_creation(self):
        h = SessionHandoff(
            session_id="sess-1",
            from_device="phone",
            to_device="laptop",
        )
        assert h.session_id == "sess-1"
        assert h.context_transferred is False
        assert h.timestamp > 0


class TestDeviceBridge:
    def test_init(self):
        bridge = DeviceBridge()
        assert len(bridge._devices) == 0

    def test_register_device(self):
        bridge = DeviceBridge()
        device = DeviceInfo(device_id="phone-1", tenant_id="user1", device_type="mobile")
        bridge.register_device(device)

        assert "phone-1" in bridge._devices
        assert "phone-1" in bridge._tenant_devices["user1"]

    def test_register_multiple_devices(self):
        bridge = DeviceBridge()
        bridge.register_device(DeviceInfo(device_id="phone", tenant_id="user1"))
        bridge.register_device(DeviceInfo(device_id="laptop", tenant_id="user1"))

        assert len(bridge._tenant_devices["user1"]) == 2

    def test_unregister_device(self):
        bridge = DeviceBridge()
        bridge.register_device(DeviceInfo(device_id="phone", tenant_id="user1"))
        bridge.unregister_device("phone")

        assert "phone" not in bridge._devices
        assert "phone" not in bridge._tenant_devices.get("user1", set())

    def test_heartbeat(self):
        import time

        bridge = DeviceBridge()
        device = DeviceInfo(device_id="phone", tenant_id="user1")
        device.last_heartbeat = time.time() - 100
        bridge.register_device(device)

        old_hb = device.last_heartbeat
        bridge.heartbeat("phone")
        assert bridge._devices["phone"].last_heartbeat > old_hb

    def test_heartbeat_unknown_device(self):
        bridge = DeviceBridge()
        # Should not raise
        bridge.heartbeat("nonexistent")

    def test_get_active_devices(self):
        bridge = DeviceBridge()
        bridge.register_device(DeviceInfo(device_id="a", tenant_id="user1"))
        bridge.register_device(DeviceInfo(device_id="b", tenant_id="user1"))
        bridge.register_device(DeviceInfo(device_id="c", tenant_id="user2"))

        active = bridge.get_active_devices("user1")
        assert len(active) == 2

    def test_get_active_devices_empty(self):
        bridge = DeviceBridge()
        assert bridge.get_active_devices("unknown") == []

    def test_get_best_device(self):
        import time

        bridge = DeviceBridge()
        d1 = DeviceInfo(device_id="old", tenant_id="user1")
        d1.last_heartbeat = time.time() - 60
        d2 = DeviceInfo(device_id="new", tenant_id="user1")

        bridge.register_device(d1)
        bridge.register_device(d2)

        best = bridge.get_best_device("user1")
        assert best is not None
        assert best.device_id == "new"  # Most recent heartbeat

    def test_get_best_device_with_capabilities(self):
        bridge = DeviceBridge()
        bridge.register_device(
            DeviceInfo(device_id="phone", tenant_id="u1", capabilities=["audio"])
        )
        bridge.register_device(
            DeviceInfo(device_id="laptop", tenant_id="u1", capabilities=["audio", "display"])
        )

        best = bridge.get_best_device("u1", required_capabilities=["display"])
        assert best is not None
        assert best.device_id == "laptop"

    def test_get_best_device_none(self):
        bridge = DeviceBridge()
        assert bridge.get_best_device("unknown") is None

    @pytest.mark.asyncio
    async def test_handoff_session(self):
        bridge = DeviceBridge()
        bridge.register_device(DeviceInfo(device_id="phone", tenant_id="user1"))
        bridge.register_device(DeviceInfo(device_id="laptop", tenant_id="user1"))

        # Set session on phone
        bridge._devices["phone"].current_session_id = "sess-1"

        result = await bridge.handoff_session("sess-1", "phone", "laptop")
        assert result is True
        assert bridge._devices["phone"].current_session_id is None
        assert bridge._devices["laptop"].current_session_id == "sess-1"
        assert len(bridge._handoff_log) == 1

    @pytest.mark.asyncio
    async def test_handoff_cross_tenant_denied(self):
        bridge = DeviceBridge()
        bridge.register_device(DeviceInfo(device_id="d1", tenant_id="user1"))
        bridge.register_device(DeviceInfo(device_id="d2", tenant_id="user2"))

        result = await bridge.handoff_session("sess-1", "d1", "d2")
        assert result is False

    @pytest.mark.asyncio
    async def test_handoff_unknown_device(self):
        bridge = DeviceBridge()
        result = await bridge.handoff_session("sess-1", "unknown1", "unknown2")
        assert result is False

    def test_stats(self):
        bridge = DeviceBridge()
        bridge.register_device(DeviceInfo(device_id="d1", tenant_id="user1"))
        stats = bridge.stats()
        assert stats["total_devices"] == 1
        assert stats["active_devices"] == 1
        assert stats["tenants"] == 1
        assert stats["handoffs"] == 0

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        bridge = DeviceBridge()
        # Should not raise
        await bridge.stop()
