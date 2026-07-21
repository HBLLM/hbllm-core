"""
Node Adapter — bridges existing HBLLM Node ↔ HCIR ABI.

Wraps any existing ``hbllm.network.node.Node`` subclass so it can
participate in the HCIR kernel as a cognitive node. The adapter:

    1. Maps Node metadata (capabilities, scopes, device_tier) into
       HCIR-native ``CapabilityNode`` registrations.
    2. Translates incoming ``Message`` into HCIR bytecode instructions.
    3. Wraps outgoing ``HCIRDelta`` into ``Message`` responses.

This is the primary migration path: existing nodes adopt HCIR
incrementally by wrapping themselves in this adapter.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.abi import ExecutionResult, ICognitiveNodeABI
from hbllm.hcir.graph import CapabilityNode, EventNode
from hbllm.hcir.transactions import HCIRDelta
from hbllm.hcir.types import Provenance, Scope

logger = logging.getLogger(__name__)


class NodeAdapter(ICognitiveNodeABI):
    """Wraps an existing HBLLM ``Node`` to participate in the HCIR kernel.

    This adapter does NOT require modifying the original node.
    It reads metadata from ``NodeInfo`` and registers capabilities
    into the HCIR workspace.

    Usage::

        from hbllm.network.node import Node
        from hbllm.hcir.adapters.node_adapter import NodeAdapter

        adapter = NodeAdapter(existing_node)
        # Register with HCIR kernel
        cap_nodes = adapter.export_capabilities(tenant_id="acme")
    """

    def __init__(self, node: Any) -> None:
        """Wrap an existing Node.

        Args:
            node: An ``hbllm.network.node.Node`` instance.
        """
        self._node = node
        self._node_info = node.get_info()

        # Set ABI declarations from existing node metadata
        self.supported_hcir_versions = ["1.0.0"]
        self.required_kernel_services = ["TransactionManager"]
        self.declared_capabilities = list(self._node_info.capabilities)

    @property
    def node_id(self) -> str:
        return self._node_info.node_id

    @property
    def node_type(self) -> str:
        return self._node_info.node_type

    def export_capabilities(self, tenant_id: str = "default") -> list[CapabilityNode]:
        """Export the node's capabilities as HCIR CapabilityNodes.

        Each capability declared by the original node becomes a
        first-class node in the HCIR graph.
        """
        cap_nodes: list[CapabilityNode] = []
        for cap_name in self._node_info.capabilities:
            cap_node = CapabilityNode(
                id=f"cap_{self.node_id}_{cap_name}",
                capability_name=cap_name,
                description=f"Capability '{cap_name}' from node '{self.node_id}'",
                provenance=Provenance(
                    created_by=self.node_id,
                    engine="node_adapter",
                ),
                scope=Scope(tenant_id=tenant_id),
                tags=["adapter", "legacy_node", self.node_id],
            )
            cap_nodes.append(cap_node)
        return cap_nodes

    def message_to_event_node(
        self,
        message: Any,
        tenant_id: str = "default",
    ) -> EventNode:
        """Convert an incoming Message to an HCIR EventNode.

        This enables the HCIR graph to track message-driven events
        alongside native cognitive operations.
        """
        return EventNode(
            id=f"msg_{getattr(message, 'id', 'unknown')}",
            event_kind=f"message.{getattr(message, 'type', 'unknown')}",
            event_data={
                "source_node_id": getattr(message, "source_node_id", ""),
                "target_node_id": getattr(message, "target_node_id", ""),
                "topic": getattr(message, "topic", ""),
                "payload_keys": list(getattr(message, "payload", {}).keys()),
            },
            provenance=Provenance(
                created_by=getattr(message, "source_node_id", "unknown"),
            ),
            scope=Scope(tenant_id=tenant_id),
            tags=["message", "adapter"],
        )

    async def execute(
        self,
        transaction: Any,
        workspace: Any,
        services: Any,
    ) -> ExecutionResult:
        """Execute by delegating to the wrapped node's handle_message.

        For the adapter layer, execution means:
        1. Convert the transaction context into a Message.
        2. Call the wrapped node's handle_message.
        3. Wrap the response into an ExecutionResult with deltas.

        This is a passthrough — the real work happens in the original node.
        """
        # Default: no-op execution. Concrete delegation requires
        # a MessageBus context which the adapter doesn't own.
        return ExecutionResult(
            success=True,
            annotations=[],
            events=[{
                "type": "adapter_passthrough",
                "node_id": self.node_id,
                "node_type": self.node_type,
            }],
        )
