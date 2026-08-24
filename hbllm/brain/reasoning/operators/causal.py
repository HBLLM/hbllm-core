"""
Causal Operator — wraps CausalGraph traversal as a ReasoningOperator.

Discovers causal chains in the HCIR graph by following CAUSES edges,
computing chain probabilities, and proposing new causal relationships.

This wraps the existing ``CausalGraph`` and ``CausalReasoner`` logic
but operates on the frozen HCIR view rather than the SQLite-backed
causal store directly.

Independence Level: L1 (no LLM)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    ProblemType,
    ProvenanceChain,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.hcir.graph import (
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)


class CausalOperator:
    """Multi-hop causal chain discovery over HCIR edges.

    Traverses CAUSES edges in the frozen view using BFS up to
    max_depth, computes chain probabilities, and proposes new
    causal relationships via transitivity.
    """

    def __init__(
        self,
        max_depth: int = 4,
        min_probability: float = 0.2,
        top_k: int = 5,
    ) -> None:
        self._max_depth = max_depth
        self._min_probability = min_probability
        self._top_k = top_k

    @property
    def operator_id(self) -> str:
        return "causal"

    @property
    def operator_name(self) -> str:
        return "Causal Chain Discovery Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.CAUSAL: 0.95,
            ProblemType.EXPLANATION: 0.6,
            ProblemType.DIAGNOSIS: 0.7,
            ProblemType.PREDICTION: 0.4,
            ProblemType.COUNTERFACTUAL: 0.3,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        view = context.graph_view
        # Count CAUSES edges
        causal_edge_count = sum(
            1
            for eid in view.all_edge_ids()
            if (e := view.get_edge(eid)) is not None and e.edge_type == HCIREdgeType.CAUSES
        )
        if causal_edge_count == 0:
            return 0.0

        return min(1.0, base + min(0.3, causal_edge_count * 0.05))

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = context.graph_view.node_count
        e = context.graph_view.edge_count
        return ResourceCost(
            wall_clock_ms=max(2.0, (n + e) * 0.2),
            nodes_read=n,
            edges_read=e,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Discover causal chains in the frozen HCIR view."""
        start = time.time()
        view = context.graph_view

        # ── Build causal adjacency from CAUSES edges ─────────────────
        causal_adj: dict[str, list[tuple[str, float]]] = {}
        for eid in view.all_edge_ids():
            edge = view.get_edge(eid)
            if edge is None or edge.edge_type != HCIREdgeType.CAUSES:
                continue
            for src in edge.sources:
                for tgt in edge.targets:
                    causal_adj.setdefault(src, []).append((tgt, edge.weight))

        if not causal_adj:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                metadata={"reason": "No causal edges in view"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Determine starting nodes ────────────────────────────────
        start_nodes = (
            set(problem.focus_node_ids) if problem.focus_node_ids else set(causal_adj.keys())
        )

        # ── BFS to discover chains ───────────────────────────────────
        all_chains: list[dict[str, Any]] = []

        for start_id in start_nodes:
            if start_id not in causal_adj:
                continue

            # BFS: (current_node, path, combined_probability)
            queue: deque[tuple[str, list[str], float]] = deque()
            queue.append((start_id, [start_id], 1.0))

            while queue:
                current, path, prob = queue.popleft()

                if len(path) > 1:
                    # Record this as a chain
                    all_chains.append(
                        {
                            "source": path[0],
                            "target": current,
                            "path": list(path),
                            "depth": len(path) - 1,
                            "probability": prob,
                        }
                    )

                if len(path) - 1 >= self._max_depth:
                    continue

                for next_node, edge_weight in causal_adj.get(current, []):
                    if next_node in path:  # Avoid cycles
                        continue
                    new_prob = prob * edge_weight
                    if new_prob >= self._min_probability:
                        queue.append((next_node, path + [next_node], new_prob))

        if not all_chains:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                conclusions={"chains_found": 0},
                metadata={"reason": "No causal chains above probability threshold"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # ── Sort by probability, take top-K ──────────────────────────
        all_chains.sort(key=lambda c: c["probability"], reverse=True)
        top_chains = all_chains[: self._top_k]

        # ── Propose transitive causal edges ──────────────────────────
        proposed_ops: list[TransactionOperation] = []
        provenance_chains: list[ProvenanceChain] = []
        evidence_refs: list[str] = []

        for chain in top_chains:
            # Propose a direct causal edge for multi-hop chains
            if chain["depth"] > 1:
                edge = HCIREdge(
                    edge_type=HCIREdgeType.CAUSES,
                    sources=[chain["source"]],
                    targets=[chain["target"]],
                    weight=chain["probability"],
                    properties={
                        "origin": "causal_transitivity",
                        "chain_depth": chain["depth"],
                        "path": chain["path"],
                    },
                    provenance=Provenance(
                        created_by=self.operator_id,
                        source_type="inferred",
                        reason=f"Transitive causal chain: {' → '.join(chain['path'])}",
                    ),
                )
                proposed_ops.append(
                    TransactionOperation(
                        op=TransactionOp.ADD_EDGE,
                        edge_data=edge.model_dump(),
                    )
                )

            provenance_chains.append(
                ProvenanceChain(
                    conclusion=f"{chain['source']} causes {chain['target']}",
                    evidence_node_ids=chain["path"],
                    operator_id=self.operator_id,
                    reasoning_steps=[
                        f"Chain: {' → '.join(chain['path'])}",
                        f"Depth: {chain['depth']}",
                        f"Probability: {chain['probability']:.3f}",
                    ],
                    confidence=chain["probability"],
                )
            )
            evidence_refs.extend(chain["path"])

        evidence_refs = list(dict.fromkeys(evidence_refs))
        elapsed_ms = (time.time() - start) * 1000

        return CognitiveResult(
            status=ResultStatus.SUCCESS,
            conclusions={
                "chains_found": len(all_chains),
                "top_chains": [
                    {
                        "source": c["source"],
                        "target": c["target"],
                        "depth": c["depth"],
                        "probability": round(c["probability"], 4),
                        "path": c["path"],
                    }
                    for c in top_chains
                ],
            },
            confidence=top_chains[0]["probability"] if top_chains else 0.0,
            evidence_refs=evidence_refs,
            proposed_transitions=proposed_ops,
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                edges_read=view.edge_count,
            ),
        )
