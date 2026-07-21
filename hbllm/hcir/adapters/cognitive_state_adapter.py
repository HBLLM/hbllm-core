"""
Cognitive State Adapter — bridges CognitiveState ↔ HCIR graph.

Maps the existing event-sourced ``CognitiveStateSnapshot`` and
``CognitiveStateDelta`` model into HCIR graph operations,
enabling the cognitive runtime to track emotional, attentional,
and motivational state as first-class graph nodes.

Direction:
    CognitiveState → HCIR:   ``snapshot_to_nodes()``
    CognitiveStateDelta → HCIR:  ``delta_to_transaction()``
    HCIR → CognitiveState:   ``nodes_to_snapshot()``
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    HCIRNodeType,
    NodeLifecycle,
    ObservationNode,
)
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionOp,
    TransactionOperation,
)
from hbllm.hcir.types import Provenance, Scope, UncertaintyVector
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)

# Import existing cognitive state types
from hbllm.brain.core.cognitive_state import (
    CognitiveStateDelta,
    CognitiveStateSnapshot,
)


# Stable ID prefix for cognitive state observation nodes
_CS_NODE_PREFIX = "cs_"


class CognitiveStateAdapter:
    """Bidirectional adapter between CognitiveState and HCIR graph.

    The existing cognitive state system uses:
        - ``CognitiveStateSnapshot`` (frozen dataclass, 11 float fields)
        - ``CognitiveStateDelta`` (proposed changes)
        - ``CognitiveStateReducer`` (applies deltas)

    This adapter maps each cognitive variable into an HCIR
    ``ObservationNode`` within the workspace graph, enabling
    the cognitive runtime to reason about emotional and
    attentional state using the same query/transaction API.

    Usage::

        adapter = CognitiveStateAdapter()
        adapter.sync_to_workspace(snapshot, workspace)
        # Later...
        restored = adapter.read_from_workspace(workspace)
    """

    #: Cognitive variable names and their HCIR observation IDs.
    COGNITIVE_FIELDS = [
        "confidence", "uncertainty", "relevance", "novelty",
        "motivation", "valence", "arousal", "intention_strength",
        "fatigue", "curiosity", "stress",
    ]

    def snapshot_to_nodes(
        self,
        snapshot: CognitiveStateSnapshot,
        tenant_id: str = "default",
        author: str = "cognitive_state_adapter",
    ) -> list[ObservationNode]:
        """Convert a CognitiveStateSnapshot into HCIR ObservationNodes.

        Each cognitive variable becomes a separate observation node,
        enabling fine-grained querying and attention management.
        """
        nodes: list[ObservationNode] = []
        for field_name in self.COGNITIVE_FIELDS:
            value = getattr(snapshot, field_name, None)
            if value is None:
                continue
            node = ObservationNode(
                id=f"{_CS_NODE_PREFIX}{field_name}",
                lifecycle=NodeLifecycle.ACTIVE,
                payload={"variable": field_name, "value": float(value)},
                sensor_source="cognitive_state",
                provenance=Provenance(
                    created_by=author,
                    reasoning_step=snapshot.version,
                ),
                uncertainty=UncertaintyVector(
                    confidence=0.95,  # High confidence — direct internal state
                ),
                scope=Scope(tenant_id=tenant_id),
                tags=["cognitive_state", field_name],
            )
            nodes.append(node)

        # Focus target as a separate node
        if snapshot.focus_target:
            nodes.append(ObservationNode(
                id=f"{_CS_NODE_PREFIX}focus_target",
                lifecycle=NodeLifecycle.ACTIVE,
                payload={"variable": "focus_target", "value": snapshot.focus_target},
                sensor_source="cognitive_state",
                provenance=Provenance(created_by=author),
                scope=Scope(tenant_id=tenant_id),
                tags=["cognitive_state", "focus_target"],
            ))

        return nodes

    def sync_to_workspace(
        self,
        snapshot: CognitiveStateSnapshot,
        workspace: HCIRWorkspaceState,
        tenant_id: str = "default",
        author: str = "cognitive_state_adapter",
    ) -> None:
        """Sync a CognitiveStateSnapshot into the HCIR workspace.

        Uses upsert semantics — creates nodes on first call,
        updates them on subsequent calls.
        """
        nodes = self.snapshot_to_nodes(snapshot, tenant_id, author)
        for node in nodes:
            workspace.upsert_node(node, author=author)
        logger.debug("Synced %d cognitive state variables to workspace", len(nodes))

    def delta_to_transaction(
        self,
        delta: CognitiveStateDelta,
        tenant_id: str = "default",
    ) -> HCIRTransaction:
        """Convert a CognitiveStateDelta into an HCIRTransaction.

        Each field change becomes a MODIFY_NODE operation on the
        corresponding observation node.
        """
        operations: list[TransactionOperation] = []

        for field_name, value in delta.changes.items():
            node_id = f"{_CS_NODE_PREFIX}{field_name}"
            if field_name == "focus_target":
                payload = {"variable": "focus_target", "value": value}
            else:
                payload = {"variable": field_name, "value": float(value)}

            operations.append(TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data=ObservationNode(
                    id=node_id,
                    lifecycle=NodeLifecycle.ACTIVE,
                    payload=payload,
                    sensor_source="cognitive_state",
                    provenance=Provenance(
                        created_by=delta.source_node,
                    ),
                    scope=Scope(tenant_id=tenant_id),
                    tags=["cognitive_state", field_name],
                ).model_dump(),
            ))

        return HCIRTransaction(
            author=delta.source_node,
            operations=operations,
            provenance=Provenance(
                created_by=delta.source_node,
            ),
        )

    def read_from_workspace(
        self,
        workspace: HCIRWorkspaceState,
    ) -> dict[str, Any]:
        """Read cognitive state variables from the HCIR workspace.

        Returns a dict of variable_name → value, reconstructable
        into a CognitiveStateSnapshot.
        """
        result: dict[str, Any] = {}
        for field_name in self.COGNITIVE_FIELDS + ["focus_target"]:
            node = workspace.get_node(f"{_CS_NODE_PREFIX}{field_name}")
            if node is None:
                continue
            if isinstance(node, ObservationNode):
                result[field_name] = node.payload.get("value")
        return result
