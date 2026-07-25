"""Tests for Phase 6 (Executive Runtime + Services) & Phase 7 (Federation)."""

from __future__ import annotations

from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.transaction_sync import (
    IntentionSet,
    TransactionSyncProtocol,
)
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionOp,
    TransactionOperation,
)
from hbllm.hcir.workspace import HCIRWorkspaceState

# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: KernelServices Extension Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKernelServicesExtensions:
    """Verify KernelServices has HCIR Cognitive OS fields."""

    def test_hcir_fields_default_to_none(self) -> None:
        from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
        from hbllm.hcir.kernel.scheduler import CognitiveScheduler
        from hbllm.hcir.kernel.transaction_manager import TransactionManager

        ws = HCIRWorkspaceState()
        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
        )
        assert services.tiered_workspace is None
        assert services.cognitive_journal is None
        assert services.cognitive_event_log is None
        assert services.semantic_normalizer is None
        assert services.constitutional_verifier is None
        assert services.bus_bridge is None

    def test_hcir_fields_can_be_set(self) -> None:
        from hbllm.hcir.cognitive_journal import CognitiveJournal
        from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
        from hbllm.hcir.kernel.scheduler import CognitiveScheduler
        from hbllm.hcir.kernel.transaction_manager import TransactionManager
        from hbllm.hcir.stores import InMemoryEventStore
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        ws = HCIRWorkspaceState()
        journal = CognitiveJournal(store=InMemoryEventStore())
        tiered = TieredWorkspace()

        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
            tiered_workspace=tiered,
            cognitive_journal=journal,
        )
        assert services.tiered_workspace is tiered
        assert services.cognitive_journal is journal


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: IntentionSet Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIntentionSet:
    """Verify IntentionSet pre-commit coordination."""

    def test_declare_intention(self) -> None:
        intentions = IntentionSet(device_id="node_a")
        intent = intentions.declare("plan_goal", "Will create goal G1", ["g1"])
        assert intent.intention_id == "plan_goal"
        assert intent.target_node_ids == ["g1"]
        assert len(intentions.intentions) == 1

    def test_no_conflict(self) -> None:
        set_a = IntentionSet(device_id="node_a")
        set_a.declare("op1", "Modify g1", ["g1"])

        set_b = IntentionSet(device_id="node_b")
        set_b.declare("op2", "Modify g2", ["g2"])

        conflicts = set_a.check_conflict(set_b)
        assert conflicts == []

    def test_conflict_detection(self) -> None:
        set_a = IntentionSet(device_id="node_a")
        set_a.declare("op1", "Modify g1", ["g1", "g2"])

        set_b = IntentionSet(device_id="node_b")
        set_b.declare("op2", "Also modify g2", ["g2", "g3"])

        conflicts = set_a.check_conflict(set_b)
        assert conflicts == ["g2"]

    def test_to_dict(self) -> None:
        intentions = IntentionSet(device_id="node_a")
        intentions.declare("op1", "Test", ["n1"])
        d = intentions.to_dict()
        assert d["device_id"] == "node_a"
        assert len(d["intentions"]) == 1

    def test_clear(self) -> None:
        intentions = IntentionSet(device_id="node_a")
        intentions.declare("op1", "Test")
        intentions.clear()
        assert len(intentions.intentions) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: TransactionSyncProtocol Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTransactionSyncProtocol:
    """Verify TransactionSyncProtocol federation."""

    def setup_method(self) -> None:
        self.source_ws = HCIRWorkspaceState()
        self.target_ws = HCIRWorkspaceState()
        self.sync_a = TransactionSyncProtocol(device_id="node_a")
        self.sync_b = TransactionSyncProtocol(device_id="node_b")

    def test_export_transaction(self) -> None:
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g1",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Test goal",
                    },
                )
            ],
        )
        tx.status = "committed"

        envelope = self.sync_a.export_transaction(tx)
        assert envelope.source_device_id == "node_a"
        assert envelope.transaction_id == tx.id
        assert len(envelope.operations) == 1
        assert envelope.signature != ""

    def test_import_transaction(self) -> None:
        """Full round-trip: export from A, import to B's workspace."""
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g1",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Federated goal",
                    },
                )
            ],
        )

        envelope = self.sync_a.export_transaction(tx)
        success = self.sync_b.import_transaction(envelope, self.target_ws)
        assert success is True
        assert self.sync_b.synced_count == 1

        # Verify node exists in target workspace
        node = self.target_ws.get_node("g1")
        assert node is not None

    def test_deduplicate_import(self) -> None:
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g2",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Dup test",
                    },
                )
            ],
        )
        envelope = self.sync_a.export_transaction(tx)

        # Import twice
        self.sync_b.import_transaction(envelope, self.target_ws)
        self.sync_b.import_transaction(envelope, self.target_ws)

        # Should only count once
        assert self.sync_b.synced_count == 1

    def test_signature_verification_failure(self) -> None:
        tx = HCIRTransaction(author="test", operations=[])
        envelope = self.sync_a.export_transaction(tx)
        envelope.signature = "tampered_signature"

        success = self.sync_b.import_transaction(envelope, self.target_ws)
        assert success is False

    def test_multi_operation_sync(self) -> None:
        """Sync a multi-op transaction."""
        tx = HCIRTransaction(
            author="planner",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g_fed1",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Goal 1",
                    },
                ),
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data={
                        "id": "g_fed2",
                        "node_type": "goal",
                        "category": "planning",
                        "description": "Goal 2",
                    },
                ),
            ],
        )
        envelope = self.sync_a.export_transaction(tx)
        success = self.sync_b.import_transaction(envelope, self.target_ws)
        assert success is True
        assert self.target_ws.get_node("g_fed1") is not None
        assert self.target_ws.get_node("g_fed2") is not None
