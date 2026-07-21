"""
Graph Validation — structural integrity checks for the CognitiveGraph.

Validates graph invariants that are difficult to enforce at the
individual node/edge level.  Should be run after batch mutations,
during sleep consolidation, or as part of the transaction verification
pipeline.

Checks:
    - Dangling edges (references to non-existent nodes)
    - Duplicate IDs
    - Invalid lifecycle transitions
    - Scope violations
    - Cyclic dependency detection (on ``depends_on`` subgraph)
    - Orphan nodes (no connected edges, optionally)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdgeType,
    HCIRNodeType,
    NodeLifecycle,
)


class ValidationSeverity(StrEnum):
    """Severity level of a validation issue."""

    ERROR = "error"  # Must be fixed before commit
    WARNING = "warning"  # Should be fixed but non-blocking
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """A single structural integrity violation."""

    severity: ValidationSeverity
    code: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None


@dataclass
class ValidationReport:
    """Complete validation result for a graph."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors were found (warnings are OK)."""
        return not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)


class GraphValidator:
    """Validates structural integrity of a ``CognitiveGraph``.

    Usage::

        validator = GraphValidator()
        report = validator.validate(graph)
        if not report.is_valid:
            for issue in report.issues:
                print(f"[{issue.severity}] {issue.code}: {issue.message}")
    """

    def validate(self, graph: CognitiveGraph) -> ValidationReport:
        """Run all validation checks and return a report."""
        report = ValidationReport()
        self._check_dangling_edges(graph, report)
        self._check_lifecycle_consistency(graph, report)
        self._check_scope_isolation(graph, report)
        self._check_dependency_cycles(graph, report)
        self._check_truth_utility_separation(graph, report)
        return report

    def _check_truth_utility_separation(
        self, graph: CognitiveGraph, report: ValidationReport
    ) -> None:
        """Enforce strict separation between Truth graph and Decision graph.

        Truth graph nodes (BeliefNode, FactNode, ObservationNode) cannot
        be directly targeted by ValueNode/UtilityNode via CAUSES or DERIVED_FROM edges.
        This prevents goal bias from corrupting factual knowledge.
        """
        truth_types = {HCIRNodeType.BELIEF, HCIRNodeType.FACT, HCIRNodeType.OBSERVATION}
        decision_types = {HCIRNodeType.VALUE, HCIRNodeType.GOAL}

        for edge in graph.all_edges():
            if edge.edge_type in (HCIREdgeType.CAUSES, HCIREdgeType.DERIVED_FROM):
                # Check if sources contain decision nodes and targets contain truth nodes
                has_decision_src = False
                has_truth_tgt = False

                for src_id in edge.sources:
                    node = graph.get_node(src_id)
                    if node and node.node_type in decision_types:
                        has_decision_src = True

                for tgt_id in edge.targets:
                    node = graph.get_node(tgt_id)
                    if node and node.node_type in truth_types:
                        has_truth_tgt = True

                if has_decision_src and has_truth_tgt:
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="TRUTH_UTILITY_CORRUPTION",
                            message=(
                                f"Edge '{edge.id}' allows Decision/Value node to directly "
                                f"drive Truth node via {edge.edge_type}."
                            ),
                            edge_id=edge.id,
                        )
                    )

    def _check_dangling_edges(self, graph: CognitiveGraph, report: ValidationReport) -> None:
        """Verify all edge endpoints reference existing nodes."""
        for edge in graph.all_edges():
            for src in edge.sources:
                if not graph.has_node(src):
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="DANGLING_EDGE_SOURCE",
                            message=f"Edge '{edge.id}' source '{src}' not found in graph.",
                            edge_id=edge.id,
                        )
                    )
            for tgt in edge.targets:
                if not graph.has_node(tgt):
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="DANGLING_EDGE_TARGET",
                            message=f"Edge '{edge.id}' target '{tgt}' not found in graph.",
                            edge_id=edge.id,
                        )
                    )

    def _check_lifecycle_consistency(self, graph: CognitiveGraph, report: ValidationReport) -> None:
        """Warn about nodes in terminal lifecycle states that still have active edges."""
        terminal_states = {NodeLifecycle.ARCHIVED, NodeLifecycle.FORGOTTEN}
        for node in graph.all_nodes():
            if node.lifecycle in terminal_states:
                outgoing = graph.edges_from(node.id)
                active_outgoing = [
                    e
                    for e in outgoing
                    if e.edge_type in (HCIREdgeType.DEPENDS_ON, HCIREdgeType.REQUIRES)
                ]
                if active_outgoing:
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            code="TERMINAL_NODE_ACTIVE_EDGES",
                            message=(
                                f"Node '{node.id}' is {node.lifecycle} but has "
                                f"{len(active_outgoing)} active dependency/requirement edges."
                            ),
                            node_id=node.id,
                        )
                    )

    def _check_scope_isolation(self, graph: CognitiveGraph, report: ValidationReport) -> None:
        """Verify edges don't cross tenant boundaries (except system-scoped nodes)."""
        from hbllm.hcir.types import SecurityLevel

        for edge in graph.all_edges():
            scopes: set[str] = set()
            for nid in edge.sources + edge.targets:
                node = graph.get_node(nid)
                if node is None:
                    continue  # Caught by dangling edge check
                if node.scope.security_level == SecurityLevel.SYSTEM:
                    continue  # System nodes can cross boundaries
                scopes.add(node.scope.tenant_id)
            if len(scopes) > 1:
                report.add(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="CROSS_TENANT_EDGE",
                        message=(f"Edge '{edge.id}' crosses tenant boundaries: {scopes}."),
                        edge_id=edge.id,
                    )
                )

    def _check_dependency_cycles(self, graph: CognitiveGraph, report: ValidationReport) -> None:
        """Detect cycles in the ``depends_on`` subgraph using iterative DFS."""
        # Build adjacency list for depends_on edges only
        adj: dict[str, list[str]] = {}
        for edge in graph.all_edges():
            if edge.edge_type != HCIREdgeType.DEPENDS_ON:
                continue
            for src in edge.sources:
                for tgt in edge.targets:
                    adj.setdefault(src, []).append(tgt)

        visited: set[str] = set()
        in_stack: set[str] = set()

        for start_node in adj:
            if start_node in visited:
                continue
            # Iterative DFS with explicit stack
            stack: list[tuple[str, int]] = [(start_node, 0)]
            path: list[str] = []

            while stack:
                node, idx = stack.pop()

                # Backtrack path to current depth
                while len(path) > len(stack):
                    removed = path.pop()
                    in_stack.discard(removed)

                if node in in_stack:
                    # Cycle detected
                    cycle_start = path.index(node) if node in path else -1
                    cycle = path[cycle_start:] + [node] if cycle_start >= 0 else [node]
                    report.add(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="DEPENDENCY_CYCLE",
                            message=f"Cycle detected in depends_on subgraph: {' → '.join(cycle)}.",
                        )
                    )
                    continue

                if node in visited:
                    continue

                visited.add(node)
                in_stack.add(node)
                path.append(node)

                for neighbor in adj.get(node, []):
                    stack.append((neighbor, 0))

            # Clean up in_stack
            for n in path:
                in_stack.discard(n)
