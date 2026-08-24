"""
A12 — Cognitive Execution Model: Core Types & Protocol.

This module defines the execution contract for HBLLM's reasoning runtime.
It is the CPU of HBLLM cognition — the layer that makes reasoning
operators composable, deterministic, and resource-aware over immutable
HCIR state.

Architectural Invariants:
    1. A reasoning operator NEVER receives mutable references to HCIR
       state.  It receives frozen snapshots and returns proposed
       transitions.  Only the transaction layer commits.

    2. Given the same HCIR snapshot and the same ReasoningProblem, the
       runtime produces the same operator trace and equivalent cognitive
       result.  (Determinism.)

    3. Every CognitiveResult carries full provenance: evidence → operator
       → input state → inference.  There is no unattributed knowledge.

    4. Operators expose a multi-dimensional selection surface (applicability,
       prerequisites, cost, expected info gain, reliability) that A19 can
       eventually learn to optimize.

Design note (from roadmap):
    LLM-independent ≠ neural-network-free.  SNN, learned models, and
    differentiable components are welcome inside operators — provided they
    obey: model produces result → HCIR transaction proposal → commit.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIRNode,
    HCIRNodeType,
)
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionOperation,
)
from hbllm.hcir.types import (
    Scope,
)

# ═══════════════════════════════════════════════════════════════════════════
# Reasoning Problem Types
# ═══════════════════════════════════════════════════════════════════════════


class ProblemType(StrEnum):
    """Classification of reasoning problems.

    Each type hints to the registry which operators are likely applicable,
    but operators self-select via ``can_handle()``.
    """

    EXPLANATION = "explanation"  # Why did X happen?
    PREDICTION = "prediction"  # What will happen if X?
    CONTRADICTION = "contradiction"  # X and Y conflict — resolve
    CLASSIFICATION = "classification"  # What category is X?
    PLANNING = "planning"  # How to achieve goal G?
    DIAGNOSIS = "diagnosis"  # What caused failure F?
    ANALOGY = "analogy"  # X is like Y because...
    CONSTRAINT = "constraint"  # Find X satisfying constraints C
    TEMPORAL = "temporal"  # What order did events occur?
    SPATIAL = "spatial"  # Where is X relative to Y?
    CAUSAL = "causal"  # Does X cause Y?
    COUNTERFACTUAL = "counterfactual"  # What if X had not happened?
    GENERALIZATION = "generalization"  # What pattern does X suggest?
    CUSTOM = "custom"  # Operator-specific


@dataclass(frozen=True)
class ReasoningProblem:
    """An immutable description of what needs to be reasoned about.

    Created by the cognitive loop when a reasoning need arises
    (e.g., contradiction detected, prediction needed, goal formed).

    Attributes:
        problem_id: Unique identifier for this problem instance.
        problem_type: Classification of the problem.
        description: Human-readable description (for trace/debug only).
        goal_node_ids: HCIR goal nodes this problem serves.
        evidence_node_ids: HCIR evidence nodes relevant to the problem.
        constraint_node_ids: HCIR constraints that must be respected.
        focus_node_ids: Primary nodes to reason about.
        parameters: Additional problem-specific parameters.
        originator: What subsystem created this problem.
        timestamp: When the problem was formulated.
    """

    problem_id: str = field(default_factory=lambda: f"prob_{uuid.uuid4().hex[:12]}")
    problem_type: ProblemType = ProblemType.CUSTOM
    description: str = ""
    goal_node_ids: tuple[str, ...] = ()
    evidence_node_ids: tuple[str, ...] = ()
    constraint_node_ids: tuple[str, ...] = ()
    focus_node_ids: tuple[str, ...] = ()
    parameters: tuple[tuple[str, Any], ...] = ()  # Immutable key-value pairs
    originator: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def param_dict(self) -> dict[str, Any]:
        """Convenience accessor for parameters as a dict."""
        return dict(self.parameters)


# ═══════════════════════════════════════════════════════════════════════════
# Resource Budget
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReasoningBudget:
    """Resource constraints for a reasoning execution.

    Eventually the system must choose between cheap deduction vs
    expensive simulation vs physical experiment vs waiting.
    A19 can learn these choices by optimizing over this surface.

    All limits are soft ceilings — operators should check remaining
    budget before expensive operations and gracefully degrade.

    Attributes:
        compute_ms: Maximum wall-clock time for operator execution.
        memory_bytes: Maximum memory budget (0 = unlimited).
        operator_depth: Maximum chain depth in a pipeline.
        simulation_steps: Max world-model simulation steps.
        uncertainty_tolerance: Accept results at this uncertainty
            level or better.  Higher = accept more uncertain results.
        max_operators: Maximum number of operators to invoke.
    """

    compute_ms: int = 5000
    memory_bytes: int = 0
    operator_depth: int = 10
    simulation_steps: int = 100
    uncertainty_tolerance: float = 0.7
    max_operators: int = 5


# ═══════════════════════════════════════════════════════════════════════════
# Immutable Cognitive Context (frozen HCIR view)
# ═══════════════════════════════════════════════════════════════════════════


class FrozenGraphView:
    """An immutable, read-only snapshot of a CognitiveGraph.

    Operators receive this instead of the live graph.  All mutation
    methods are absent.  This makes it architecturally impossible for
    an operator to corrupt HCIR state.

    The snapshot is taken at a specific point in time.  The content_hash
    identifies the exact state for deterministic replay.

    Internals use deep-copied node/edge dicts so that even if an
    operator mutates the returned Pydantic models, the original graph
    is unaffected.
    """

    def __init__(
        self,
        nodes: dict[str, HCIRNode],
        edges: dict[str, HCIREdge],
        snapshot_sequence: int = 0,
    ) -> None:
        # Deep-copy via Pydantic serialization for true immutability
        self._nodes: dict[str, HCIRNode] = {
            nid: node.model_copy(deep=True) for nid, node in nodes.items()
        }
        self._edges: dict[str, HCIREdge] = {
            eid: edge.model_copy(deep=True) for eid, edge in edges.items()
        }
        self._snapshot_sequence = snapshot_sequence
        self._content_hash = self._compute_hash()

    @classmethod
    def from_graph(
        cls,
        graph: CognitiveGraph,
        node_ids: set[str] | None = None,
        snapshot_sequence: int = 0,
    ) -> FrozenGraphView:
        """Create a frozen view from a live CognitiveGraph.

        Args:
            graph: The live graph to snapshot.
            node_ids: Optional subset of node IDs to include.
                If None, includes all nodes.
            snapshot_sequence: Event-log sequence number at snapshot time.

        Returns:
            An immutable view of the specified graph state.
        """
        if node_ids is not None:
            nodes = {
                nid: node
                for nid, node in ((nid, graph.get_node(nid)) for nid in node_ids)
                if node is not None
            }
            # Include edges where both endpoints are in the view
            relevant_edge_ids: set[str] = set()
            for nid in node_ids:
                for edge in graph.edges_from(nid):
                    if all(t in node_ids for t in edge.targets):
                        relevant_edge_ids.add(edge.id)
                for edge in graph.edges_to(nid):
                    if all(s in node_ids for s in edge.sources):
                        relevant_edge_ids.add(edge.id)
            edges = {
                eid: edge
                for eid, edge in ((eid, graph.get_edge(eid)) for eid in relevant_edge_ids)
                if edge is not None
            }
        else:
            nodes = {n.id: n for n in graph.all_nodes()}
            edges = {e.id: e for e in graph.all_edges()}

        return cls(nodes, edges, snapshot_sequence)

    def _compute_hash(self) -> str:
        """Content-addressable hash of the graph state."""
        # Sort by ID for deterministic ordering
        node_data = sorted((nid, n.model_dump_json()) for nid, n in self._nodes.items())
        edge_data = sorted((eid, e.model_dump_json()) for eid, e in self._edges.items())
        hasher = hashlib.sha256()
        for nid, data in node_data:
            hasher.update(nid.encode())
            hasher.update(data.encode())
        for eid, data in edge_data:
            hasher.update(eid.encode())
            hasher.update(data.encode())
        return hasher.hexdigest()

    # ── Read-only accessors ──────────────────────────────────────────

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the graph state at snapshot time."""
        return self._content_hash

    @property
    def snapshot_sequence(self) -> int:
        return self._snapshot_sequence

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def get_node(self, node_id: str) -> HCIRNode | None:
        """Retrieve a deep-copied node by ID."""
        node = self._nodes.get(node_id)
        if node is not None:
            return node.model_copy(deep=True)
        return None

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_edge(self, edge_id: str) -> HCIREdge | None:
        edge = self._edges.get(edge_id)
        if edge is not None:
            return edge.model_copy(deep=True)
        return None

    def all_node_ids(self) -> frozenset[str]:
        return frozenset(self._nodes.keys())

    def all_edge_ids(self) -> frozenset[str]:
        return frozenset(self._edges.keys())

    def nodes_by_type(self, node_type: HCIRNodeType) -> list[HCIRNode]:
        """Return deep-copied nodes of the given type."""
        return [n.model_copy(deep=True) for n in self._nodes.values() if n.node_type == node_type]

    def edges_from(self, node_id: str) -> list[HCIREdge]:
        """Edges where node_id is a source."""
        return [e.model_copy(deep=True) for e in self._edges.values() if node_id in e.sources]

    def edges_to(self, node_id: str) -> list[HCIREdge]:
        """Edges where node_id is a target."""
        return [e.model_copy(deep=True) for e in self._edges.values() if node_id in e.targets]

    def __repr__(self) -> str:
        return (
            f"FrozenGraphView(nodes={self.node_count}, "
            f"edges={self.edge_count}, "
            f"hash={self._content_hash[:12]}…)"
        )


@dataclass(frozen=True)
class CognitiveContext:
    """Immutable context provided to a reasoning operator.

    This is the ONLY interface through which operators access HCIR state.
    No mutable references.  No live graph access.

    Attributes:
        graph_view: Frozen, read-only snapshot of relevant HCIR state.
        problem: The reasoning problem to solve.
        budget: Resource constraints for this execution.
        scope: Tenant/workspace/user isolation context.
        cognitive_mode: Current reasoning mode (standard, discovery, etc.).
        timestamp: When this context was created.
        parent_trace_id: If this is a sub-invocation, the parent trace.
    """

    graph_view: FrozenGraphView
    problem: ReasoningProblem
    budget: ReasoningBudget = field(default_factory=ReasoningBudget)
    scope: Scope = field(default_factory=Scope)
    cognitive_mode: str = "standard"
    timestamp: float = field(default_factory=time.time)
    parent_trace_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Result — operator output with provenance
# ═══════════════════════════════════════════════════════════════════════════


class ResultStatus(StrEnum):
    """Outcome of an operator execution."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some conclusions reached, not all
    NO_RESULT = "no_result"  # Operator could not produce conclusions
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"


@dataclass
class ProvenanceChain:
    """Full provenance for a single conclusion.

    Enables answering: "Why does HBLLM believe this?"

    Chain: evidence → operator → input state → intermediate inference
           → prediction → observation.
    """

    conclusion: str
    evidence_node_ids: list[str] = field(default_factory=list)
    input_node_ids: list[str] = field(default_factory=list)
    operator_id: str = ""
    reasoning_steps: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class ResourceCost:
    """Actual resources consumed by an operator execution."""

    wall_clock_ms: float = 0.0
    operators_invoked: int = 0
    simulation_steps_used: int = 0
    nodes_read: int = 0
    edges_read: int = 0


@dataclass
class CognitiveResult:
    """The typed output of a reasoning operator or pipeline.

    Operators produce these; the runtime collects them and proposes
    HCIR transactions.  A CognitiveResult NEVER directly modifies HCIR.

    Attributes:
        result_id: Unique identifier.
        status: Outcome status.
        conclusions: Structured conclusions keyed by label.
        confidence: Overall confidence in the result [0.0, 1.0].
        evidence_refs: HCIR node IDs used as evidence.
        assumptions: Explicit assumptions made during reasoning.
        proposed_transitions: HCIR transaction operations to commit.
        provenance_chains: Full provenance for each conclusion.
        operator_id: Which operator produced this.
        resource_cost: Actual resources consumed.
        metadata: Additional operator-specific data.
        timestamp: When the result was produced.
    """

    result_id: str = field(default_factory=lambda: f"res_{uuid.uuid4().hex[:12]}")
    status: ResultStatus = ResultStatus.SUCCESS
    conclusions: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    evidence_refs: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    proposed_transitions: list[TransactionOperation] = field(default_factory=list)
    provenance_chains: list[ProvenanceChain] = field(default_factory=list)
    operator_id: str = ""
    resource_cost: ResourceCost = field(default_factory=ResourceCost)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
# Operator Trace — deterministic replay record
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OperatorInvocation:
    """Record of a single operator's execution within a trace."""

    operator_id: str
    problem_id: str
    context_hash: str  # FrozenGraphView.content_hash at invocation time
    result: CognitiveResult
    started_at: float = 0.0
    finished_at: float = 0.0
    selection_score: float = 0.0


@dataclass
class OperatorTrace:
    """Complete deterministic trace of a reasoning execution.

    Given the same CognitiveContext, replaying through this trace
    should produce the same sequence of invocations and equivalent
    results.

    Attributes:
        trace_id: Unique trace identifier.
        problem: The original problem.
        context_hash: Content hash of the initial HCIR snapshot.
        invocations: Ordered list of operator invocations.
        final_result: Merged result from the pipeline.
        proposed_transaction: The HCIR transaction assembled from results.
        total_wall_clock_ms: Total execution time.
        timestamp: When the trace was created.
    """

    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    problem: ReasoningProblem | None = None
    context_hash: str = ""
    invocations: list[OperatorInvocation] = field(default_factory=list)
    final_result: CognitiveResult | None = None
    proposed_transaction: HCIRTransaction | None = None
    total_wall_clock_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════
# Operator Selection Score — multi-dimensional control surface
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OperatorSelectionScore:
    """Multi-dimensional score for operator selection.

    A12 exposes this surface.  A19 eventually learns to optimize it.

    Final selection score:
        applicability × reliability × expected_info_gain
        × expected_utility × budget_fit × prerequisite_satisfaction

    Attributes:
        operator_id: Which operator this score is for.
        applicability: How well the operator matches the problem [0, 1].
        reliability: Historical success rate on similar problems [0, 1].
        expected_info_gain: How much uncertainty reduction is expected [0, 1].
        expected_utility: Expected usefulness of result [0, 1].
        budget_fit: How well the operator fits the resource budget [0, 1].
        prerequisite_satisfaction: How well prerequisites are met [0, 1].
        composite_score: Computed composite selection score.
    """

    operator_id: str = ""
    applicability: float = 0.0
    reliability: float = 0.5
    expected_info_gain: float = 0.5
    expected_utility: float = 0.5
    budget_fit: float = 1.0
    prerequisite_satisfaction: float = 1.0
    composite_score: float = 0.0

    def compute_composite(self) -> float:
        """Compute the composite selection score.

        Simple multiplicative model.  A19 can eventually learn custom
        weighting or a non-linear combination.
        """
        self.composite_score = (
            self.applicability
            * self.reliability
            * self.expected_info_gain
            * self.expected_utility
            * self.budget_fit
            * self.prerequisite_satisfaction
        )
        return self.composite_score


# ═══════════════════════════════════════════════════════════════════════════
# Reasoning Operator Protocol
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ReasoningOperator(Protocol):
    """Protocol for all cognitive reasoning operators.

    A reasoning operator:
    1. Receives an immutable CognitiveContext (frozen HCIR view).
    2. Produces a typed CognitiveResult with provenance.
    3. Proposes state transitions as TransactionOperations.
    4. NEVER directly mutates HCIR.

    Implementation notes:
    - ``can_handle`` returns a float applicability score [0.0, 1.0].
      Return 0.0 if the operator cannot handle this problem at all.
    - ``execute`` is deterministic given the same CognitiveContext.
    - ``prerequisites`` declares what other operators must run first.
    - ``estimated_cost`` helps the budget system make tradeoffs.
    """

    @property
    def operator_id(self) -> str:
        """Unique, stable identifier for this operator (e.g., 'deduction')."""
        ...

    @property
    def operator_name(self) -> str:
        """Human-readable name (e.g., 'Formal Deduction Engine')."""
        ...

    @property
    def prerequisites(self) -> tuple[str, ...]:
        """Operator IDs that must run before this operator.

        Empty tuple means no prerequisites.
        """
        ...

    def can_handle(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> float:
        """Score how applicable this operator is to the problem.

        Returns:
            Applicability score [0.0, 1.0].
            0.0 = cannot handle at all.
            1.0 = ideal match.
        """
        ...

    def estimated_cost(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> ResourceCost:
        """Estimate resources this operator will consume.

        Used by the budget system to decide between cheap vs expensive
        approaches.  Operators should be conservative (overestimate).
        """
        ...

    def execute(
        self,
        problem: ReasoningProblem,
        context: CognitiveContext,
    ) -> CognitiveResult:
        """Execute reasoning over immutable context.

        Must be deterministic: same context → same result.
        Must not have side effects beyond the returned CognitiveResult.
        """
        ...
