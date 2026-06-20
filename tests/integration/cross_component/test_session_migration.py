"""Tests for SessionMigration — cross-device context handoff."""

import json
import time

from hbllm.network.session_migration import SessionMigrationManager, SessionSnapshot


class TestSessionSnapshot:
    def test_defaults(self):
        snap = SessionSnapshot(id="test1", tenant_id="t1")
        assert snap.id == "test1"
        assert not snap.is_expired()

    def test_checksum_computation(self):
        snap = SessionSnapshot(
            id="test1",
            tenant_id="t1",
            conversation_history=[{"role": "user", "content": "Hello"}],
        )
        snap.checksum = snap.compute_checksum()
        assert snap.verify_integrity()

    def test_checksum_tamper_detection(self):
        snap = SessionSnapshot(
            id="test1",
            tenant_id="t1",
            conversation_history=[{"role": "user", "content": "Hello"}],
        )
        snap.checksum = snap.compute_checksum()
        # Tamper with data
        snap.conversation_history.append({"role": "user", "content": "Injected"})
        assert not snap.verify_integrity()

    def test_expiry(self):
        snap = SessionSnapshot(
            id="test1",
            tenant_id="t1",
            expires_at=time.time() - 100,
        )
        assert snap.is_expired()

    def test_no_expiry(self):
        snap = SessionSnapshot(
            id="test1",
            tenant_id="t1",
            expires_at=0.0,
        )
        assert not snap.is_expired()

    def test_serialization(self):
        snap = SessionSnapshot(
            id="test1",
            tenant_id="t1",
            conversation_history=[{"role": "user", "content": "Hello"}],
            active_goals=[{"id": "g1", "name": "Deploy"}],
            persona_traits={"formality": 0.8},
        )
        snap.checksum = snap.compute_checksum()
        json_str = snap.to_json()
        snap2 = SessionSnapshot.from_json(json_str)
        assert snap2.id == "test1"
        assert len(snap2.conversation_history) == 1
        assert len(snap2.active_goals) == 1
        assert snap2.persona_traits["formality"] == 0.8

    def test_roundtrip_integrity(self):
        snap = SessionSnapshot(
            id="test1",
            tenant_id="t1",
            conversation_history=[{"role": "user", "content": "Hello"}],
        )
        snap.checksum = snap.compute_checksum()
        json_str = snap.to_json()
        snap2 = SessionSnapshot.from_json(json_str)
        assert snap2.verify_integrity()


class TestSessionMigrationManager:
    def test_export(self):
        mgr = SessionMigrationManager(node_id="server1")
        snap = mgr.export_session(
            tenant_id="t1",
            conversation_history=[{"role": "user", "content": "Hello"}],
        )
        assert snap.source_node_id == "server1"
        assert snap.tenant_id == "t1"
        assert snap.checksum

    def test_export_with_ttl(self):
        mgr = SessionMigrationManager(node_id="server1")
        snap = mgr.export_session("t1", ttl_seconds=60)
        assert snap.expires_at > time.time()

    def test_import_valid(self):
        mgr1 = SessionMigrationManager(node_id="server1")
        mgr2 = SessionMigrationManager(node_id="server2")

        snap = mgr1.export_session(
            tenant_id="t1",
            conversation_history=[{"role": "user", "content": "Hello"}],
        )
        json_str = snap.to_json()

        imported = mgr2.import_session(json_str)
        assert imported is not None
        assert imported.tenant_id == "t1"

    def test_import_invalid_json(self):
        mgr = SessionMigrationManager(node_id="server1")
        result = mgr.import_session("{invalid json")
        assert result is None

    def test_import_tampered(self):
        mgr1 = SessionMigrationManager(node_id="server1")
        mgr2 = SessionMigrationManager(node_id="server2")

        snap = mgr1.export_session(
            tenant_id="t1",
            conversation_history=[{"role": "user", "content": "Hello"}],
        )
        data = snap.to_dict()
        # Tamper
        data["conversation_history"].append({"role": "user", "content": "Injected"})
        json_str = json.dumps(data)

        imported = mgr2.import_session(json_str)
        assert imported is None  # Integrity check should fail

    def test_import_expired(self):
        mgr1 = SessionMigrationManager(node_id="server1")
        mgr2 = SessionMigrationManager(node_id="server2")

        snap = mgr1.export_session("t1", ttl_seconds=3600)
        # Manually expire the snapshot
        snap.expires_at = time.time() - 100
        snap.checksum = snap.compute_checksum()  # Recompute after mutation
        json_str = snap.to_json()

        imported = mgr2.import_session(json_str)
        assert imported is None

    def test_self_import(self):
        mgr = SessionMigrationManager(node_id="server1")
        snap = mgr.export_session("t1")
        json_str = snap.to_json()

        # Self-import should still return the snapshot but log a message
        imported = mgr.import_session(json_str)
        assert imported is not None

    def test_list_exported(self):
        mgr = SessionMigrationManager(node_id="server1")
        mgr.export_session("t1")
        mgr.export_session("t2")
        all_exports = mgr.list_exported()
        assert len(all_exports) == 2

    def test_list_exported_by_tenant(self):
        mgr = SessionMigrationManager(node_id="server1")
        mgr.export_session("t1")
        mgr.export_session("t2")
        t1_only = mgr.list_exported(tenant_id="t1")
        assert len(t1_only) == 1

    def test_cleanup_expired(self):
        mgr = SessionMigrationManager(node_id="server1")
        snap = mgr.export_session("t1", ttl_seconds=3600)
        # Manually expire the snapshot
        snap.expires_at = time.time() - 100
        count = mgr.cleanup_expired()
        assert count == 1
        assert len(mgr.list_exported()) == 0

    def test_stats(self):
        mgr = SessionMigrationManager(node_id="server1")
        mgr.export_session("t1")
        stats = mgr.stats()
        assert stats["exported_count"] == 1
        assert stats["node_id"] == "server1"
