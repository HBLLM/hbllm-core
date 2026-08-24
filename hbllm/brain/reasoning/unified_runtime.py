"""
A12 — Unified Reasoning Runtime.

The runtime is the CPU of HBLLM cognition.  It:
1. Builds an immutable CognitiveContext from live HCIR state.
2. Queries the OperatorRegistry for applicable operators.
3. Executes operators (single or pipeline).
4. Collects CognitiveResults and assembles a deterministic OperatorTrace.
5. Proposes an HCIRTransaction from the results — NEVER directly mutates HCIR.

Acceptance criterion:
    Given the same HCIR snapshot and the same ReasoningProblem, the runtime
    produces the same operator trace and equivalent cognitive result.

Design notes:
    - Pipeline composition: result of operator A feeds operator B by
      enriching the CognitiveContext with intermediate conclusions.
    - The runtime respects ReasoningBudget at every step.
    - The OperatorTrace enables deterministic replay and provenance
      reconstruction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from hbllm.brain.reasoning.operators.base import (
    CognitiveContext,
    CognitiveResult,
    FrozenGraphView,
    OperatorInvocation,
    OperatorTrace,
    ProvenanceChain,
    ReasoningBudget,
    ReasoningProblem,
    ResourceCost,
    ResultStatus,
)
from hbllm.brain.reasoning.operators.registry import OperatorRegistry
from hbllm.hcir.graph import CognitiveGraph
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionOperation,
)
from hbllm.hcir.types import Provenance, Scope

logger = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    """Configuration for the reasoning runtime."""

    # Maximum operators to invoke in a single reasoning pass
    max_operators_per_pass: int = 5

    # Minimum composite score to consider an operator
    min_selection_score: float = 0.001

    # Whether to stop pipeline on first NO_RESULT
    stop_on_no_result: bool = False

    # Whether to merge results or keep only the best
    merge_results: bool = True


class UnifiedReasoningRuntime:
    """The cognitive execution model — A12 core.

    Usage::

        runtime = UnifiedReasoningRuntime(registry)

        # Build context from live graph
        trace = runtime.reason(
            graph=live_graph,
            problem=problem,
            budget=budget,
            scope=scope,
        )

        # Inspect trace
        for inv in trace.invocations:
            print(f"  {inv.operator_id}: {inv.result.status}")

        # The trace contains a proposed transaction — commit via
        # TransactionManager (not the runtime's responsibility).
        if trace.proposed_transaction:
            tx_manager.submit(trace.proposed_transaction)
    """

    def __init__(
        self,
        registry: OperatorRegistry,
        config: RuntimeConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or RuntimeConfig()

    @property
    def registry(self) -> OperatorRegistry:
        return self._registry

    # ── Primary Entry Point ──────────────────────────────────────────

    def reason(
        self,
        graph: CognitiveGraph,
        problem: ReasoningProblem,
        budget: ReasoningBudget | None = None,
        scope: Scope | None = None,
        cognitive_mode: str = "standard",
        node_ids: set[str] | None = None,
        snapshot_sequence: int = 0,
    ) -> OperatorTrace:
        """Execute reasoning over HCIR state.

        This is the main entry point.  It:
        1. Creates an immutable FrozenGraphView from the live graph.
        2. Builds a CognitiveContext.
        3. Selects and executes operators.
        4. Assembles an OperatorTrace with a proposed transaction.

        Args:
            graph: The live CognitiveGraph (read-only access — a frozen
                snapshot is taken immediately).
            problem: What to reason about.
            budget: Resource constraints.
            scope: Tenant/workspace isolation.
            cognitive_mode: Reasoning mode (standard, discovery, etc.).
            node_ids: Optional subset of HCIR nodes to include in the
                frozen view.  If None, relevant nodes are auto-selected
                from the problem definition.
            snapshot_sequence: Event-log sequence number for the snapshot.

        Returns:
            A complete OperatorTrace with deterministic replay data and
            a proposed HCIRTransaction (if any conclusions were reached).
        """
        start_time = time.time()

        _budget = budget or ReasoningBudget()
        _scope = scope or Scope()

        # ── Step 1: Build bounded frozen view ────────────────────────
        view_node_ids = self._compute_view_boundary(graph, problem, node_ids)
        frozen_view = FrozenGraphView.from_graph(graph, view_node_ids, snapshot_sequence)

        # ── Step 2: Build immutable context ──────────────────────────
        context = CognitiveContext(
            graph_view=frozen_view,
            problem=problem,
            budget=_budget,
            scope=_scope,
            cognitive_mode=cognitive_mode,
        )

        # ── Step 3: Select operators ─────────────────────────────────
        selection = self._registry.select(
            problem, context, min_applicability=self._config.min_selection_score
        )

        if not selection:
            trace = OperatorTrace(
                problem=problem,
                context_hash=frozen_view.content_hash,
                total_wall_clock_ms=(time.time() - start_time) * 1000,
            )
            logger.warning("No operators selected for problem '%s'", problem.problem_id)
            return trace

        # ── Step 4: Execute operators (pipeline) ─────────────────────
        trace = OperatorTrace(
            problem=problem,
            context_hash=frozen_view.content_hash,
        )

        results: list[CognitiveResult] = []
        remaining_budget_ms = _budget.compute_ms
        operators_run = 0

        for score in selection:
            if operators_run >= self._config.max_operators_per_pass:
                logger.info(
                    "Reached max operators per pass (%d)",
                    self._config.max_operators_per_pass,
                )
                break

            if operators_run >= _budget.max_operators:
                logger.info(
                    "Reached budget operator limit (%d)",
                    _budget.max_operators,
                )
                break

            operator = self._registry.get(score.operator_id)
            if operator is None:
                continue

            # Check prerequisites
            if not self._prerequisites_satisfied(operator.prerequisites, trace):
                logger.debug(
                    "Skipping operator '%s' — prerequisites not met",
                    score.operator_id,
                )
                continue

            # Execute
            inv_start = time.time()
            try:
                result = operator.execute(problem, context)
                result.operator_id = score.operator_id
            except Exception as e:
                logger.error(
                    "Operator '%s' raised during execute: %s",
                    score.operator_id,
                    e,
                    exc_info=True,
                )
                result = CognitiveResult(
                    status=ResultStatus.ERROR,
                    operator_id=score.operator_id,
                    metadata={"error": str(e)},
                )
                self._registry.record_outcome(score.operator_id, False)
            else:
                success = result.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)
                self._registry.record_outcome(score.operator_id, success)

            inv_end = time.time()
            inv_ms = (inv_end - inv_start) * 1000

            invocation = OperatorInvocation(
                operator_id=score.operator_id,
                problem_id=problem.problem_id,
                context_hash=frozen_view.content_hash,
                result=result,
                started_at=inv_start,
                finished_at=inv_end,
                selection_score=score.composite_score,
            )
            trace.invocations.append(invocation)
            results.append(result)
            operators_run += 1

            remaining_budget_ms -= inv_ms
            if remaining_budget_ms <= 0:
                logger.info("Budget exhausted after %d operators", operators_run)
                break

            if self._config.stop_on_no_result and result.status == ResultStatus.NO_RESULT:
                break

        # ── Step 5: Merge results ────────────────────────────────────
        final_result = self._merge_results(results, problem)
        trace.final_result = final_result

        # ── Step 6: Assemble proposed transaction ────────────────────
        if final_result.proposed_transitions:
            transaction = HCIRTransaction(
                author=f"reasoning_runtime:{problem.problem_id}",
                parent_snapshot_hash=frozen_view.content_hash,
                operations=final_result.proposed_transitions,
                provenance=Provenance(
                    created_by="unified_reasoning_runtime",
                    trace_id=trace.trace_id,
                    reason=f"Reasoning result for {problem.problem_type}",
                    source_type="inferred",
                ),
            )
            trace.proposed_transaction = transaction

        trace.total_wall_clock_ms = (time.time() - start_time) * 1000

        logger.info(
            "Reasoning complete: problem='%s' operators=%d wall_clock=%.1fms conclusions=%d",
            problem.problem_id,
            operators_run,
            trace.total_wall_clock_ms,
            len(final_result.conclusions),
        )

        return trace

    # ── View Boundary Computation ────────────────────────────────────

    @staticmethod
    def _compute_view_boundary(
        graph: CognitiveGraph,
        problem: ReasoningProblem,
        explicit_ids: set[str] | None,
    ) -> set[str] | None:
        """Determine which HCIR nodes to include in the frozen view.

        If explicit_ids is provided, use those.
        Otherwise, auto-select from problem definition.
        If no node IDs are specified anywhere, return None (full graph).
        """
        if explicit_ids is not None:
            return explicit_ids

        # Auto-select from problem definition
        auto_ids: set[str] = set()
        auto_ids.update(problem.goal_node_ids)
        auto_ids.update(problem.evidence_node_ids)
        auto_ids.update(problem.constraint_node_ids)
        auto_ids.update(problem.focus_node_ids)

        if not auto_ids:
            return None  # Full graph view

        # Expand to include 1-hop neighbors for context
        expanded: set[str] = set(auto_ids)
        for nid in auto_ids:
            for edge in graph.edges_from(nid):
                expanded.update(edge.targets)
                expanded.add(edge.id)  # Not a node, but keep edge refs
            for edge in graph.edges_to(nid):
                expanded.update(edge.sources)

        # Filter to actual nodes
        return {nid for nid in expanded if graph.has_node(nid)}

    # ── Pipeline Helpers ─────────────────────────────────────────────

    @staticmethod
    def _prerequisites_satisfied(
        prerequisites: tuple[str, ...],
        trace: OperatorTrace,
    ) -> bool:
        """Check if all prerequisite operators have already run successfully."""
        if not prerequisites:
            return True

        completed = {
            inv.operator_id
            for inv in trace.invocations
            if inv.result.status in (ResultStatus.SUCCESS, ResultStatus.PARTIAL)
        }
        return all(p in completed for p in prerequisites)

    def _merge_results(
        self,
        results: list[CognitiveResult],
        problem: ReasoningProblem,
    ) -> CognitiveResult:
        """Merge multiple operator results into a final result.

        Strategy:
        - Conclusions from all operators are merged (later overrides earlier
          for same key).
        - Confidence is the weighted average by applicability.
        - Evidence refs are unioned.
        - Assumptions are unioned.
        - Proposed transitions are concatenated.
        - Provenance chains are collected.
        - Resource costs are summed.
        """
        if not results:
            return CognitiveResult(
                status=ResultStatus.NO_RESULT,
                metadata={"reason": "No operator results to merge"},
            )

        if len(results) == 1:
            return results[0]

        # Merge
        merged_conclusions: dict[str, Any] = {}
        merged_evidence: list[str] = []
        merged_assumptions: list[str] = []
        merged_transitions: list[TransactionOperation] = []
        merged_provenance: list[ProvenanceChain] = []
        total_cost = ResourceCost()
        confidences: list[float] = []

        seen_evidence: set[str] = set()
        seen_assumptions: set[str] = set()

        for result in results:
            if result.status == ResultStatus.ERROR:
                continue

            merged_conclusions.update(result.conclusions)
            confidences.append(result.confidence)

            for ref in result.evidence_refs:
                if ref not in seen_evidence:
                    merged_evidence.append(ref)
                    seen_evidence.add(ref)

            for assumption in result.assumptions:
                if assumption not in seen_assumptions:
                    merged_assumptions.append(assumption)
                    seen_assumptions.add(assumption)

            merged_transitions.extend(result.proposed_transitions)
            merged_provenance.extend(result.provenance_chains)

            total_cost.wall_clock_ms += result.resource_cost.wall_clock_ms
            total_cost.operators_invoked += 1
            total_cost.simulation_steps_used += result.resource_cost.simulation_steps_used
            total_cost.nodes_read += result.resource_cost.nodes_read
            total_cost.edges_read += result.resource_cost.edges_read

        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Determine status
        statuses = [r.status for r in results]
        if all(s == ResultStatus.SUCCESS for s in statuses):
            final_status = ResultStatus.SUCCESS
        elif any(s == ResultStatus.SUCCESS for s in statuses):
            final_status = ResultStatus.PARTIAL
        elif all(s == ResultStatus.NO_RESULT for s in statuses):
            final_status = ResultStatus.NO_RESULT
        else:
            final_status = ResultStatus.PARTIAL

        return CognitiveResult(
            status=final_status,
            conclusions=merged_conclusions,
            confidence=avg_confidence,
            evidence_refs=merged_evidence,
            assumptions=merged_assumptions,
            proposed_transitions=merged_transitions,
            provenance_chains=merged_provenance,
            operator_id="unified_runtime_merged",
            resource_cost=total_cost,
        )
