"""World-State Overlay and Ephemeral SimulationBranch for A18.

Implements copy-on-write subgraph overlays over canonical HCIR CognitiveGraph.
Guarantees strict isolation: simulation mutations never leak into reality.
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.simulation.events import SimulationEvent, compute_state_hash
from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIRNode

logger = logging.getLogger(__name__)


@dataclass
class SimulationBranch:
    """An ephemeral, copy-on-write overlay representing a hypothetical future world."""

    branch_id: str = field(default_factory=lambda: f"branch_{uuid.uuid4().hex[:8]}")
    parent_branch_id: str | None = None
    base_graph: CognitiveGraph | None = None
    parent_branch: SimulationBranch | None = None
    base_revision: int = 1
    depth: int = 0
    created_at: float = field(default_factory=time.time)

    # Local copy-on-write overlays
    _modified_nodes: dict[str, HCIRNode] = field(default_factory=dict)
    _added_edges: list[HCIREdge] = field(default_factory=list)
    _removed_edges: set[tuple[str, str, str]] = field(default_factory=set)  # (src, edge_type, tgt)
    _events: list[SimulationEvent] = field(default_factory=list)

    # Epistemic metrics
    confidence_decay_rate: float = 0.95
    accumulated_risk: float = 0.0
    violated_constraints: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Effective confidence decayed by rollout depth."""
        return max(0.10, min(1.0, (self.confidence_decay_rate**self.depth)))

    @property
    def events(self) -> list[SimulationEvent]:
        return list(self._events)

    def record_event(self, event: SimulationEvent) -> None:
        self._events.append(event)
        self.accumulated_risk = max(self.accumulated_risk, event.risk)
        if event.violations:
            self.violated_constraints.extend(event.violations)

    # ── Node Resolution (Copy-on-Write) ───────────────────────────────

    def get_node(self, node_id: str) -> HCIRNode | None:
        """Resolve node: local overlay -> parent branch -> base canonical graph.

        Returns a deepcopy to ensure canonical reality can never be mutated in-place.
        """
        if node_id in self._modified_nodes:
            return copy.deepcopy(self._modified_nodes[node_id])

        if self.parent_branch is not None:
            parent_node = self.parent_branch.get_node(node_id)
            if parent_node is not None:
                return copy.deepcopy(parent_node)

        if self.base_graph is not None:
            base_node = self.base_graph.get_node(node_id)
            if base_node is not None:
                return copy.deepcopy(base_node)

        return None

    def upsert_node(self, node: HCIRNode) -> None:
        """Modify or add node strictly in local simulation overlay."""
        # Deepcopy to ensure complete mutation isolation
        node_copy = copy.deepcopy(node)
        # Tag with simulation branch provenance
        if hasattr(node_copy, "provenance") and node_copy.provenance is not None:
            node_copy.provenance.simulation_branch = self.branch_id
        self._modified_nodes[node.id] = node_copy

    def all_nodes(self) -> list[HCIRNode]:
        """Return all nodes in the effective world state."""
        node_map: dict[str, HCIRNode] = {}
        if self.base_graph is not None:
            for n in self.base_graph.all_nodes():
                node_map[n.id] = n

        if self.parent_branch is not None:
            for n in self.parent_branch.all_nodes():
                node_map[n.id] = n

        for nid, n in self._modified_nodes.items():
            node_map[nid] = n

        return list(node_map.values())

    # ── Edge Resolution ───────────────────────────────────────────────

    def add_edge(self, edge: HCIREdge) -> None:
        """Add edge in simulation overlay."""
        self._added_edges.append(edge)
        # Un-remove if previously removed
        for src in edge.sources:
            for tgt in edge.targets:
                self._removed_edges.discard((src, edge.edge_type.value, tgt))

    def remove_edge(self, source_id: str, edge_type: str, target_id: str) -> None:
        """Mark an edge as removed in simulation overlay."""
        self._removed_edges.add((source_id, edge_type, target_id))
        self._added_edges = [
            e
            for e in self._added_edges
            if not (
                source_id in e.sources and target_id in e.targets and e.edge_type.value == edge_type
            )
        ]

    def all_edges(self) -> list[HCIREdge]:
        """Return all active edges in the effective world state."""
        effective_edges: list[HCIREdge] = []
        base_edge_list: list[HCIREdge] = []

        if self.base_graph is not None:
            base_edge_list.extend(self.base_graph.all_edges())
        if self.parent_branch is not None:
            base_edge_list.extend(self.parent_branch._added_edges)

        for edge in base_edge_list:
            # Check if all source-target pairs are removed
            is_removed = any(
                (src, edge.edge_type.value, tgt) in self._removed_edges
                for src in edge.sources
                for tgt in edge.targets
            )
            if not is_removed:
                effective_edges.append(edge)

        effective_edges.extend(self._added_edges)
        return effective_edges

    def edges_from(self, source_id: str) -> list[HCIREdge]:
        return [e for e in self.all_edges() if source_id in e.sources]

    def edges_to(self, target_id: str) -> list[HCIREdge]:
        return [e for e in self.all_edges() if target_id in e.targets]

    # ── Branching & Hashing ───────────────────────────────────────────

    def fork_child(self, child_branch_id: str | None = None) -> SimulationBranch:
        """Create a nested simulation child branch."""
        cid = child_branch_id or f"{self.branch_id}_sub_{uuid.uuid4().hex[:6]}"
        return SimulationBranch(
            branch_id=cid,
            parent_branch_id=self.branch_id,
            base_graph=self.base_graph,
            parent_branch=self,
            base_revision=self.base_revision,
            depth=self.depth + 1,
        )

    def compute_current_state_hash(self) -> str:
        """Deterministic state hash of the overlayed world state."""
        node_tuples: list[tuple[str, str, dict[str, Any]]] = []
        for n in self.all_nodes():
            props = getattr(n, "properties", None) or getattr(n, "observed_properties", {}) or {}
            node_tuples.append((n.id, n.node_type.value, dict(props)))

        edge_tuples: list[tuple[str, str, str]] = []
        for e in self.all_edges():
            for src in e.sources:
                for tgt in e.targets:
                    edge_tuples.append((src, e.edge_type.value, tgt))

        return compute_state_hash(node_tuples, edge_tuples)
