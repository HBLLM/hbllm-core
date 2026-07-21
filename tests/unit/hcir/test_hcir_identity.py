"""Unit tests for HCIR Identity + Causality Layer."""

import pytest

from hbllm.hcir.identity import (
    CausalEvent,
    CausalGraph,
    CauseRelation,
    HCIRNamespace,
    HCIRObjectID,
    IDFactory,
)


# ═══════════════════════════════════════════════════════════════════════════
# HCIRObjectID Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHCIRObjectID:
    def test_defaults(self):
        oid = HCIRObjectID()
        assert oid.tenant_id == "default"
        assert oid.device_id == "local"
        assert oid.namespace == HCIRNamespace.KNOWLEDGE
        assert oid.object_type == "node"
        assert oid.version == 1
        assert len(oid.uuid) == 12

    def test_to_string(self):
        oid = HCIRObjectID(
            tenant_id="acme", device_id="robot_12",
            namespace=HCIRNamespace.MEMORY, object_type="belief",
            uuid="83fa1e2b3c4d", version=3,
        )
        s = oid.to_string()
        assert s == "acme:robot_12:memory:belief:83fa1e2b3c4d:3"

    def test_from_string_roundtrip(self):
        original = HCIRObjectID(
            tenant_id="acme", device_id="laptop_01",
            namespace=HCIRNamespace.EXECUTION, object_type="event",
            uuid="abc123def456", version=7,
        )
        s = original.to_string()
        restored = HCIRObjectID.from_string(s)
        assert restored.tenant_id == "acme"
        assert restored.device_id == "laptop_01"
        assert restored.namespace == HCIRNamespace.EXECUTION
        assert restored.object_type == "event"
        assert restored.uuid == "abc123def456"
        assert restored.version == 7

    def test_from_string_legacy_fallback(self):
        oid = HCIRObjectID.from_string("some-bare-uuid")
        assert oid.is_legacy
        assert oid.uuid == "some-bare-uuid"

    def test_next_version(self):
        oid = HCIRObjectID(version=5)
        next_oid = oid.next_version()
        assert next_oid.version == 6
        assert next_oid.uuid == oid.uuid

    def test_with_namespace(self):
        oid = HCIRObjectID(namespace=HCIRNamespace.KNOWLEDGE)
        sim_oid = oid.with_namespace(HCIRNamespace.SIMULATION)
        assert sim_oid.namespace == HCIRNamespace.SIMULATION
        assert sim_oid.uuid == oid.uuid

    def test_frozen(self):
        oid = HCIRObjectID()
        with pytest.raises(Exception):
            oid.tenant_id = "evil"  # type: ignore

    def test_str(self):
        oid = HCIRObjectID(tenant_id="t", device_id="d", uuid="u", version=1)
        assert str(oid) == oid.to_string()


class TestIDFactory:
    def test_node_id(self):
        factory = IDFactory(tenant_id="acme", device_id="laptop")
        nid = factory.node_id(namespace=HCIRNamespace.KNOWLEDGE, object_type="belief")
        assert nid.tenant_id == "acme"
        assert nid.device_id == "laptop"
        assert nid.object_type == "belief"
        assert nid.namespace == HCIRNamespace.KNOWLEDGE

    def test_event_id(self):
        factory = IDFactory(tenant_id="acme")
        eid = factory.event_id()
        assert eid.namespace == HCIRNamespace.SYSTEM
        assert eid.object_type == "event"

    def test_transaction_id(self):
        factory = IDFactory()
        tid = factory.transaction_id()
        assert tid.namespace == HCIRNamespace.EXECUTION
        assert tid.object_type == "transaction"

    def test_unique_uuids(self):
        factory = IDFactory()
        ids = {factory.node_id().uuid for _ in range(100)}
        assert len(ids) == 100  # All unique


# ═══════════════════════════════════════════════════════════════════════════
# Causal Event Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCausalEvent:
    def test_defaults(self):
        e = CausalEvent(event_type="observation")
        assert e.event_id.startswith("evt_")
        assert e.parent_event_ids == []
        assert e.timestamp > 0

    def test_seal_produces_hash(self):
        e = CausalEvent(event_type="observation", actor="sensor_1")
        e.seal()
        assert len(e.content_hash) == 16

    def test_hash_deterministic(self):
        e = CausalEvent(event_id="e1", event_type="test", actor="a")
        h1 = e.compute_hash()
        h2 = e.compute_hash()
        assert h1 == h2

    def test_different_events_different_hashes(self):
        e1 = CausalEvent(event_id="e1", event_type="test", actor="a")
        e2 = CausalEvent(event_id="e2", event_type="test", actor="b")
        assert e1.compute_hash() != e2.compute_hash()


class TestCausalGraph:
    def test_add_and_get_event(self):
        cg = CausalGraph()
        e = CausalEvent(event_id="e1", event_type="observation")
        cg.add_event(e)
        assert cg.get_event("e1") is e
        assert cg.event_count == 1

    def test_trace_causes_linear_chain(self):
        cg = CausalGraph()
        e1 = CausalEvent(event_id="e1", event_type="sensor")
        e2 = CausalEvent(event_id="e2", event_type="observation", parent_event_ids=["e1"])
        e3 = CausalEvent(event_id="e3", event_type="belief", parent_event_ids=["e2"])
        cg.add_event(e1)
        cg.add_event(e2)
        cg.add_event(e3)

        causes = cg.trace_causes("e3")
        cause_ids = [e.event_id for e in causes]
        assert cause_ids == ["e1", "e2"]

    def test_trace_causes_diamond(self):
        """Test diamond-shaped causal DAG."""
        cg = CausalGraph()
        e1 = CausalEvent(event_id="e1", event_type="root")
        e2 = CausalEvent(event_id="e2", event_type="branch_a", parent_event_ids=["e1"])
        e3 = CausalEvent(event_id="e3", event_type="branch_b", parent_event_ids=["e1"])
        e4 = CausalEvent(event_id="e4", event_type="merge", parent_event_ids=["e2", "e3"])
        for e in [e1, e2, e3, e4]:
            cg.add_event(e)

        causes = cg.trace_causes("e4")
        cause_ids = {e.event_id for e in causes}
        assert cause_ids == {"e1", "e2", "e3"}

    def test_trace_effects(self):
        cg = CausalGraph()
        e1 = CausalEvent(event_id="e1", event_type="root")
        e2 = CausalEvent(event_id="e2", event_type="child1", parent_event_ids=["e1"])
        e3 = CausalEvent(event_id="e3", event_type="child2", parent_event_ids=["e1"])
        e4 = CausalEvent(event_id="e4", event_type="grandchild", parent_event_ids=["e2"])
        for e in [e1, e2, e3, e4]:
            cg.add_event(e)

        effects = cg.trace_effects("e1")
        effect_ids = {e.event_id for e in effects}
        assert effect_ids == {"e2", "e3", "e4"}

    def test_trace_causes_no_parents(self):
        cg = CausalGraph()
        e1 = CausalEvent(event_id="e1", event_type="root")
        cg.add_event(e1)
        causes = cg.trace_causes("e1")
        assert causes == []
