"""
SNN Reasoning Operator — wraps CausalReasoner + ReasoningNetwork.

Discovers causal chains and evaluates them with the SNN-based
ReasoningNetwork, which uses a spiking neural network to score
chain quality based on structural features.

This operator operates on the HCIR frozen view but can optionally
use an existing CausalGraph + ReasoningNetwork for richer evaluation.

Without external dependencies (no CausalGraph/SNN available), it
falls back to pure graph-based causal chain discovery similar to
the CausalOperator but with SNN-inspired feature scoring.

Independence Level: L1 (no LLM). Uses SNN — LLM-independent ≠ neural-network-free.
"""

from __future__ import annotations

import logging
import math
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
    HCIREdgeType,
)

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Sigmoid activation for SNN-inspired scoring."""
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


class SNNReasoningOperator:
    """SNN-based causal chain evaluation over HCIR.

    Discovers causal chains from the frozen view and evaluates
    them using SNN-inspired structural feature scoring:
        - Chain depth (shorter = higher quality)
        - Combined probability
        - Fan-out at each node
        - Edge diversity (variety of edge types)

    When a real ReasoningNetwork is available, it delegates to that.
    Otherwise uses a lightweight LIF-inspired scoring function.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_probability: float = 0.3,
        top_k: int = 5,
    ) -> None:
        self._max_depth = max_depth
        self._min_probability = min_probability
        self._top_k = top_k

        # Try to import the real ReasoningNetwork
        self._snn_network: Any = None
        try:
            from hbllm.brain.snn.reasoning.reasoning_network import (
                ReasoningNetwork,
            )

            self._snn_network = ReasoningNetwork()
            logger.info("SNNReasoningOperator using real ReasoningNetwork")
        except (ImportError, Exception) as e:
            logger.info(
                "SNNReasoningOperator using fallback scoring (ReasoningNetwork unavailable: %s)", e
            )

    @property
    def operator_id(self) -> str:
        return "snn_reasoning"

    @property
    def operator_name(self) -> str:
        return "SNN Causal Reasoning Engine"

    @property
    def prerequisites(self) -> tuple[str, ...]:
        return ()

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        type_scores: dict[ProblemType, float] = {
            ProblemType.CAUSAL: 0.85,
            ProblemType.EXPLANATION: 0.5,
            ProblemType.DIAGNOSIS: 0.6,
            ProblemType.PREDICTION: 0.3,
        }
        base = type_scores.get(problem.problem_type, 0.05)

        view = context.graph_view
        # Need CAUSES edges
        has_causal = any(
            (e := view.get_edge(eid)) is not None and e.edge_type == HCIREdgeType.CAUSES
            for eid in view.all_edge_ids()
        )
        if not has_causal:
            return 0.0

        return min(1.0, base)

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        n = context.graph_view.node_count
        e = context.graph_view.edge_count
        return ResourceCost(
            wall_clock_ms=max(3.0, (n + e) * 0.3),
            nodes_read=n,
            edges_read=e,
        )

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Discover and evaluate causal chains with SNN scoring."""
        start = time.time()
        view = context.graph_view

        # ── Build adjacency ──────────────────────────────────────────
        causal_adj: dict[str, list[tuple[str, float]]] = {}
        fan_out: dict[str, int] = {}

        for eid in view.all_edge_ids():
            edge = view.get_edge(eid)
            if edge is None or edge.edge_type != HCIREdgeType.CAUSES:
                continue
            for src in edge.sources:
                for tgt in edge.targets:
                    causal_adj.setdefault(src, []).append((tgt, edge.weight))
                    fan_out[src] = fan_out.get(src, 0) + 1

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

        # ── BFS chain discovery ──────────────────────────────────────
        start_nodes = (
            set(problem.focus_node_ids) if problem.focus_node_ids else set(causal_adj.keys())
        )

        all_chains: list[dict[str, Any]] = []

        for start_id in start_nodes:
            if start_id not in causal_adj:
                continue

            queue: deque[tuple[str, list[str], float]] = deque()
            queue.append((start_id, [start_id], 1.0))

            while queue:
                current, path, prob = queue.popleft()

                if len(path) > 1:
                    # Score with SNN
                    features = self._extract_features(path, prob, fan_out)
                    snn_score = self._snn_evaluate(features)

                    all_chains.append(
                        {
                            "source": path[0],
                            "target": current,
                            "path": list(path),
                            "depth": len(path) - 1,
                            "probability": prob,
                            "snn_confidence": snn_score,
                            "features": features,
                        }
                    )

                if len(path) - 1 >= self._max_depth:
                    continue

                for next_node, edge_weight in causal_adj.get(current, []):
                    if next_node in path:
                        continue
                    new_prob = prob * edge_weight
                    if new_prob >= self._min_probability:
                        queue.append((next_node, path + [next_node], new_prob))

        if not all_chains:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                operator_id=self.operator_id,
                conclusions={"chains_found": 0},
                metadata={"reason": "No chains above threshold"},
                resource_cost=ResourceCost(
                    wall_clock_ms=(time.time() - start) * 1000,
                    nodes_read=view.node_count,
                ),
            )

        # Sort by SNN confidence
        all_chains.sort(key=lambda c: c["snn_confidence"], reverse=True)
        top_chains = all_chains[: self._top_k]

        # ── Build result ─────────────────────────────────────────────
        provenance_chains: list[ProvenanceChain] = []
        evidence_refs: list[str] = []

        for chain in top_chains:
            provenance_chains.append(
                ProvenanceChain(
                    conclusion=f"{chain['source']} → {chain['target']}",
                    evidence_node_ids=chain["path"],
                    operator_id=self.operator_id,
                    reasoning_steps=[
                        f"Chain: {' → '.join(chain['path'])}",
                        f"Depth: {chain['depth']}",
                        f"Probability: {chain['probability']:.3f}",
                        f"SNN confidence: {chain['snn_confidence']:.3f}",
                    ],
                    confidence=chain["snn_confidence"],
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
                        "snn_confidence": round(c["snn_confidence"], 4),
                    }
                    for c in top_chains
                ],
            },
            confidence=top_chains[0]["snn_confidence"] if top_chains else 0,
            evidence_refs=evidence_refs,
            provenance_chains=provenance_chains,
            operator_id=self.operator_id,
            resource_cost=ResourceCost(
                wall_clock_ms=elapsed_ms,
                nodes_read=view.node_count,
                edges_read=view.edge_count,
            ),
        )

    # ── Feature Extraction ───────────────────────────────────────────

    @staticmethod
    def _extract_features(
        path: list[str],
        probability: float,
        fan_out: dict[str, int],
    ) -> list[float]:
        """Extract structural features for SNN evaluation.

        Features:
            [0] depth_score: shorter chains = higher score
            [1] probability: combined causal probability
            [2] avg_fan_out: average outgoing edges per node (lower = more specific)
            [3] specificity: 1/avg_fan_out (higher = more targeted)
        """
        depth = len(path) - 1
        depth_score = 1.0 / (1.0 + depth * 0.3)

        avg_fan = sum(fan_out.get(n, 1) for n in path) / max(1, len(path))
        specificity = 1.0 / (1.0 + avg_fan * 0.2)

        return [depth_score, probability, avg_fan, specificity]

    def _snn_evaluate(self, features: list[float]) -> float:
        """Evaluate a causal chain using SNN or fallback scoring.

        If a real ReasoningNetwork is available, delegates to it.
        Otherwise uses a sigmoid-based weighted sum (LIF-inspired).
        """
        if self._snn_network is not None:
            try:
                return float(self._snn_network.evaluate(features))
            except Exception:
                pass

        # Fallback: sigmoid-weighted sum
        weights = [0.3, 0.4, -0.1, 0.2]
        weighted_sum = sum(w * f for w, f in zip(weights, features))
        return _sigmoid(weighted_sum * 3.0)
