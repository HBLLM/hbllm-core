"""Epistemic Safety Gate — Graph-Connectivity-Based Epistemic Invariant Enforcement.

Bridges the epistemic state in CognitiveGraph (ContradictionNodes, BeliefNodes,
Observations) with the GovernanceEngine execution policy.

CRITICAL DESIGN INVARIANTS:
1. Relevance by Graph Topology, NOT Text Keyword Matching:
   Determines relevance strictly through hyperedge path reachability (up to K hops)
   between contradiction conflict anchors (claim_a_id, claim_b_id, observation_ids,
   conflicting_belief_ids) and target entity nodes in CognitiveGraph.
   Zero claim text is inspected, making this completely language-independent.

2. Scoped Exclusively to Overriding Affirmative Clearance:
   GovernanceEngine already fails closed when safety context is absent or silent.
   The gate's specific responsibility is overriding an affirmatively-asserted safe
   state (e.g. workspace_cleared=True, human_in_workspace=False, authorized=True)
   when an active contradiction in the target entity's topological neighborhood
   disputes that clearance. On silence, the gate is a no-op.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    ContradictionNode,
    HCIREdgeType,
    HCIRNodeType,
    PhysicalEntityNode,
)

logger = logging.getLogger(__name__)

# Exclude non-domain meta-scoping edges that would falsely bridge unrelated entities across tenant
_EXCLUDED_EDGE_TYPES: set[HCIREdgeType] = {
    HCIREdgeType.TENANT_SCOPE,
}


class EpistemicSafetyGate:
    """Epistemic safety gate that overrides affirmative safety clearance

    when an active contradiction in the cognitive graph topologically connects
    to the target entity of a capability.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        max_hops: int = 4,
    ) -> None:
        self._graph = graph
        self._max_hops = max_hops

    @property
    def graph(self) -> CognitiveGraph:
        return self._graph

    def _resolve_target_nodes(
        self,
        target: str,
        grounded_entity_ids: list[str] | None = None,
    ) -> set[str]:
        """Resolve all node IDs in the graph corresponding to the capability target."""
        target_nodes: set[str] = set()

        # 1. Include explicit grounded entity IDs if provided
        if grounded_entity_ids:
            for gid in grounded_entity_ids:
                if self._graph.has_node(gid):
                    target_nodes.add(gid)

        clean_target = (target or "").lower().strip()
        if not clean_target:
            return target_nodes

        # 2. Direct ID matches
        if self._graph.has_node(target):
            target_nodes.add(target)
        if self._graph.has_node(clean_target):
            target_nodes.add(clean_target)

        # 3. Match against PhysicalEntityNodes and all graph nodes
        for node in self._graph.all_nodes():
            nid = node.id.lower()
            if nid == clean_target or (
                len(clean_target) >= 3 and (clean_target in nid or nid in clean_target)
            ):
                target_nodes.add(node.id)

            if isinstance(node, PhysicalEntityNode):
                ename = (node.entity_name or "").lower()
                etype = (node.entity_type or "").lower()
                if ename and (
                    ename == clean_target
                    or (len(clean_target) >= 3 and (clean_target in ename or ename in clean_target))
                ):
                    target_nodes.add(node.id)
                if etype and (
                    etype == clean_target
                    or (len(clean_target) >= 3 and (clean_target in etype or etype in clean_target))
                ):
                    target_nodes.add(node.id)

        return target_nodes

    def _is_topologically_connected(
        self,
        sources: set[str],
        targets: set[str],
    ) -> bool:
        """Bounded BFS traversal to check if any source connects to any target within max_hops."""
        if sources & targets:
            return True

        visited: set[str] = set(sources)
        queue: list[tuple[str, int]] = [(s, 0) for s in sources]

        while queue:
            curr, depth = queue.pop(0)
            if curr in targets:
                return True
            if depth >= self._max_hops:
                continue

            incident_edges = self._graph.edges_from(curr) + self._graph.edges_to(curr)
            for edge in incident_edges:
                if edge.edge_type in _EXCLUDED_EDGE_TYPES:
                    continue
                endpoints = set(edge.sources + edge.targets)
                for nxt in endpoints:
                    if nxt not in visited:
                        visited.add(nxt)
                        if nxt in targets:
                            return True
                        queue.append((nxt, depth + 1))

        return False

    def find_connected_contradictions(
        self,
        target_nodes: set[str],
    ) -> list[ContradictionNode]:
        """Find all active ContradictionNodes that have an edge-path to any target node."""
        if not target_nodes:
            return []

        connected: list[ContradictionNode] = []
        contra_nodes = [
            n
            for n in self._graph.nodes_by_type(HCIRNodeType.CONTRADICTION)
            if isinstance(n, ContradictionNode)
        ]

        for contra in contra_nodes:
            # Only active, unresolved contradictions can override affirmative claims
            if contra.resolution_status in ("resolved", "dismissed", "explained"):
                continue

            # Gather conflict anchor nodes
            anchors: set[str] = set()
            if contra.claim_a_id and self._graph.has_node(contra.claim_a_id):
                anchors.add(contra.claim_a_id)
            if contra.claim_b_id and self._graph.has_node(contra.claim_b_id):
                anchors.add(contra.claim_b_id)
            for oid in getattr(contra, "observation_ids", []):
                if self._graph.has_node(oid):
                    anchors.add(oid)
            for bid in getattr(contra, "conflicting_belief_ids", []):
                if self._graph.has_node(bid):
                    anchors.add(bid)

            # If contradiction node itself is directly linked via edges
            if self._graph.edges_from(contra.id) or self._graph.edges_to(contra.id):
                anchors.add(contra.id)

            if not anchors:
                continue

            if self._is_topologically_connected(anchors, target_nodes):
                connected.append(contra)

        return connected

    def evaluate_overrides(
        self,
        target: str,
        capability_name: str,
        context: dict[str, Any],
        grounded_entity_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate whether active contradictions dispute affirmative safety claims in context.

        Returns a modified copy of context with affirmative claims overridden if
        a connected contradiction is detected, or the original context if not.
        """
        # 1. Check if affirmative safety claims are present.
        # If no affirmative signals are present, the gate is a no-op;
        # the baseline fail-closed governance invariants handle absent clearance.
        has_affirmative_clearance = (
            context.get("workspace_cleared") is True
            or context.get("human_in_workspace") is False
            or context.get("area_cleared") is True
            or context.get("proximity_cleared") is True
            or context.get("clearance_verified") is True
            or context.get("authorized") is True
        )

        if not has_affirmative_clearance:
            return context

        # 2. Resolve target entities in the graph
        target_nodes = self._resolve_target_nodes(target, grounded_entity_ids)
        if not target_nodes:
            return context

        # 3. Find connected active contradictions
        connected_contras = self.find_connected_contradictions(target_nodes)
        if not connected_contras:
            return context

        # 4. Connected contradictions found: override affirmative claims
        logger.warning(
            "EpistemicSafetyGate: Active contradiction(s) %s topologically connected to target '%s'. "
            "Overriding affirmative safety clearance.",
            [c.id for c in connected_contras],
            target,
        )

        modified = dict(context)
        conflict_ids = [c.id for c in connected_contras]

        # Override actuator workspace clearance
        if modified.get("workspace_cleared") is True:
            modified["workspace_cleared"] = False
        if modified.get("human_in_workspace") is False:
            modified["human_in_workspace"] = True
        if modified.get("area_cleared") is True:
            modified["area_cleared"] = False
        if modified.get("proximity_cleared") is True:
            modified["proximity_cleared"] = False
        if modified.get("clearance_verified") is True:
            modified["clearance_verified"] = False

        # Override perimeter authorization
        if modified.get("authorized") is True:
            modified["authorized"] = False

        # Attach epistemic audit provenance
        modified["epistemic_contradiction_active"] = True
        modified["epistemic_conflict_ids"] = conflict_ids

        return modified
