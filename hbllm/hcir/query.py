"""
Graph Query API — HCIR §5 (Refined).

Provides a declarative query interface for the CognitiveGraph.
Callers build ``GraphQuery`` objects with filter predicates, and the
query engine resolves them against the graph's secondary indexes.

Future backends (SQL, Cypher, Gremlin) can implement the same
``IQueryEngine`` interface without changing callers.

Usage::

    from hbllm.hcir.query import GraphQuery

    results = graph.query(
        GraphQuery(
            node_type=HCIRNodeType.GOAL,
            lifecycle=NodeLifecycle.ACTIVE,
            scope_tenant="tenant_alpha",
            min_confidence=0.7,
        )
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hbllm.hcir.graph import (
    CognitiveCategory,
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    HCIRNode,
    HCIRNodeType,
    NodeLifecycle,
)

# ═══════════════════════════════════════════════════════════════════════════
# Query Specification
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GraphQuery:
    """Declarative query specification for cognitive graph nodes.

    All filter fields are optional.  When multiple fields are set,
    results must satisfy ALL of them (logical AND).
    """

    # Type filters
    node_type: HCIRNodeType | None = None
    category: CognitiveCategory | None = None
    lifecycle: NodeLifecycle | None = None

    # Scope filters
    scope_tenant: str | None = None

    # Tag filter
    tags: list[str] | None = None  # Node must have ALL listed tags

    # Confidence / attention filters
    min_confidence: float | None = None
    min_salience: float | None = None

    # Text search (substring match on common text fields)
    text_contains: str | None = None

    # Limit
    limit: int = 100


@dataclass
class EdgeQuery:
    """Declarative query specification for hyperedges."""

    edge_type: HCIREdgeType | None = None
    source_id: str | None = None
    target_id: str | None = None
    limit: int = 100


# ═══════════════════════════════════════════════════════════════════════════
# Query Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class QueryResult:
    """Result set from a graph query."""

    nodes: list[HCIRNode] = field(default_factory=list)
    edges: list[HCIREdge] = field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Query Engine Interface
# ═══════════════════════════════════════════════════════════════════════════


class IQueryEngine(ABC):
    """Abstract query engine.

    Allows future backends (SQL, Cypher, etc.) to implement the same
    query API without changing callers.
    """

    @abstractmethod
    def query_nodes(self, query: GraphQuery) -> QueryResult:
        """Execute a node query and return matching results."""
        ...

    @abstractmethod
    def query_edges(self, query: EdgeQuery) -> QueryResult:
        """Execute an edge query and return matching results."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# In-Memory Query Engine
# ═══════════════════════════════════════════════════════════════════════════


_QUERY_TEXT_ATTRS: tuple[str, ...] = (
    "claim",
    "description",
    "label",
    "summary",
    "skill_name",
    "procedure_name",
    "intent",
    "capability_name",
    "name",
    "expression",
)
_CLASS_TEXT_ATTRS_CACHE: dict[type, tuple[str, ...]] = {}


def _text_attrs_for_class(cls: type) -> tuple[str, ...]:
    """Return cached tuple of searchable text attributes present on *cls*."""
    cached = _CLASS_TEXT_ATTRS_CACHE.get(cls)
    if cached is not None:
        return cached
    model_fields = getattr(cls, "model_fields", None)
    if isinstance(model_fields, dict):
        attrs = tuple(a for a in _QUERY_TEXT_ATTRS if a in model_fields)
    else:
        attrs = tuple(a for a in _QUERY_TEXT_ATTRS if hasattr(cls, a))
    _CLASS_TEXT_ATTRS_CACHE[cls] = attrs
    return attrs


class InMemoryQueryEngine(IQueryEngine):
    """Query engine that operates over an in-memory ``CognitiveGraph``.

    Uses the graph's secondary indexes for efficient filtering,
    then applies remaining predicates as post-filters.
    """

    def __init__(self, graph: CognitiveGraph) -> None:
        self._graph = graph

    def query_nodes(self, query: GraphQuery) -> QueryResult:
        """Execute a node query using index-first filtering."""
        # Start with the smallest candidate set from indexes
        candidates: set[str] | None = None

        if query.node_type is not None:
            ids = {n.id for n in self._graph.nodes_by_type(query.node_type)}
            candidates = ids if candidates is None else candidates & ids

        if query.category is not None:
            ids = {n.id for n in self._graph.nodes_by_category(query.category)}
            candidates = ids if candidates is None else candidates & ids

        if query.lifecycle is not None:
            ids = {n.id for n in self._graph.nodes_by_lifecycle(query.lifecycle)}
            candidates = ids if candidates is None else candidates & ids

        if query.scope_tenant is not None:
            ids = {n.id for n in self._graph.nodes_by_scope(query.scope_tenant)}
            candidates = ids if candidates is None else candidates & ids

        if query.tags:
            for tag in query.tags:
                ids = {n.id for n in self._graph.nodes_by_tag(tag)}
                candidates = ids if candidates is None else candidates & ids

        # If no index was used, scan all nodes
        if candidates is None:
            candidate_nodes = list(self._graph.all_nodes())
        else:
            candidate_nodes = [
                self._graph.get_node(nid) for nid in candidates if self._graph.has_node(nid)
            ]
            candidate_nodes = [n for n in candidate_nodes if n is not None]

        # Post-filter
        text_lower = query.text_contains.lower() if query.text_contains is not None else None
        results: list[HCIRNode] = []
        for node in candidate_nodes:
            if query.min_confidence is not None:
                if node.uncertainty.confidence < query.min_confidence:
                    continue
            if query.min_salience is not None:
                if node.attention.salience < query.min_salience:
                    continue
            if text_lower is not None:
                found = False
                node_dict = node.__dict__
                for attr_name in _text_attrs_for_class(type(node)):
                    val = node_dict.get(attr_name)
                    if isinstance(val, str) and val and text_lower in val.lower():
                        found = True
                        break
                if not found:
                    extra = node_dict.get("__pydantic_extra__")
                    if isinstance(extra, dict) and extra:
                        for attr_name in _QUERY_TEXT_ATTRS:
                            val = extra.get(attr_name)
                            if isinstance(val, str) and val and text_lower in val.lower():
                                found = True
                                break
                if not found:
                    continue

            results.append(node)
            if len(results) >= query.limit:
                return QueryResult(
                    nodes=results,
                    total_matches=len(results),
                    truncated=True,
                )

        return QueryResult(nodes=results, total_matches=len(results))

    def query_edges(self, query: EdgeQuery) -> QueryResult:
        """Execute an edge query."""
        results: list[HCIREdge] = []

        if query.source_id is not None:
            candidate_edges = self._graph.edges_from(query.source_id)
        elif query.target_id is not None:
            candidate_edges = self._graph.edges_to(query.target_id)
        else:
            candidate_edges = list(self._graph.all_edges())

        for edge in candidate_edges:
            if query.edge_type is not None and edge.edge_type != query.edge_type:
                continue
            if query.source_id is not None and query.source_id not in edge.sources:
                continue
            if query.target_id is not None and query.target_id not in edge.targets:
                continue
            results.append(edge)
            if len(results) >= query.limit:
                return QueryResult(edges=results, total_matches=len(results), truncated=True)

        return QueryResult(edges=results, total_matches=len(results))
