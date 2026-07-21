"""
Typed Cognitive Hypergraph — HCIR §3.

The core data structure of the HCIR workspace.  A single graph instance
holds all cognitive entities (nodes) and semantic relationships (edges).
Different subsystems access it through **views** (filtered projections)
rather than separate graph objects:

    Graph
      ├── Knowledge View   (Facts, Beliefs, Concepts, Procedures)
      ├── Execution View   (Goals, Actions, active threads)
      ├── Simulation View  (Forked branches)
      └── Memory View      (Episodes, Skills, Values)

Graph nodes are **typed subclasses**, not untyped dictionaries.
Each subclass carries typed payload fields validated at construction.

Design invariants:
    - Node IDs are globally unique within a graph instance.
    - Edge IDs are globally unique within a graph instance.
    - Edges are *hyper*-edges: they can link multiple source/target nodes.
    - The graph never owns persistence.  Storage is delegated to ``IGraphStore``.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from hbllm.hcir.types import (
    Attention,
    Confidence,
    CostMetric,
    Priority,
    Provenance,
    Scope,
    TimeDuration,
    UncertaintyVector,
)

# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════


class HCIRNodeType(StrEnum):
    """All valid cognitive node types in HCIR."""

    # --- Directives ---
    GOAL = "goal"
    CONSTRAINT = "constraint"
    INTENT = "intent"

    # --- Epistemology hierarchy ---
    OBSERVATION = "observation"
    FACT = "fact"
    BELIEF = "belief"
    HYPOTHESIS = "hypothesis"
    PREDICTION = "prediction"

    # --- Execution ---
    ACTION = "action"
    EVENT = "event"
    RESOURCE = "resource"
    CAPABILITY = "capability"

    # --- Memory classes ---
    EPISODE = "episode"
    CONCEPT = "concept"
    SKILL = "skill"
    PROCEDURE = "procedure"
    VALUE = "value"
    EXTERNAL_KNOWLEDGE = "external_knowledge"

    # --- World Model & Predictive Cognitive Runtime ---
    WORLD_VARIABLE = "world_variable"
    PHYSICAL_ENTITY = "physical_entity"
    ENVIRONMENT_STATE = "environment_state"


class HCIREdgeType(StrEnum):
    """All valid typed edge relationships in HCIR."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    REQUIRES = "requires"
    CAUSES = "causes"
    TENANT_SCOPE = "tenant_scope"

    # Temporal relations
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    EXPIRES = "expires"
    VALID_UNTIL = "valid_until"

    # Structural
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"
    CREATED_BY = "created_by"
    OWNED_BY = "owned_by"
    PART_OF = "part_of"


class CognitiveCategory(StrEnum):
    """High-level cognitive function classification.

    Enables schedulers and attention engines to prioritize by cognitive
    function rather than concrete node class.
    """

    PERCEPTION = "perception"
    MEMORY = "memory"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    VALUE = "value"
    COMMUNICATION = "communication"


class NodeLifecycle(StrEnum):
    """Generic lifecycle states for any cognitive node."""

    CREATED = "created"
    OBSERVED = "observed"
    VALIDATED = "validated"
    ACTIVE = "active"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"


class GoalLifecycle(StrEnum):
    """Specialized lifecycle states for goal nodes."""

    CREATED = "created"
    PLANNED = "planned"
    EXECUTING = "executing"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CONSOLIDATED = "consolidated"


# ═══════════════════════════════════════════════════════════════════════════
# Node Base & Typed Subclasses
# ═══════════════════════════════════════════════════════════════════════════


def _new_id(prefix: str = "n") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class HCIRNode(BaseModel):
    """Base class for all typed cognitive graph nodes.

    Subclasses carry domain-specific typed payload fields.
    The base provides identity, provenance, uncertainty, attention,
    scope, and lifecycle — shared by every cognitive entity.
    """

    id: str = Field(default_factory=lambda: _new_id("n"))
    node_type: HCIRNodeType
    category: CognitiveCategory
    lifecycle: NodeLifecycle = NodeLifecycle.CREATED
    provenance: Provenance = Field(default_factory=Provenance)
    uncertainty: UncertaintyVector = Field(default_factory=UncertaintyVector)
    attention: Attention = Field(default_factory=Attention)
    scope: Scope = Field(default_factory=Scope)
    tags: list[str] = Field(default_factory=list)


# ── Directive Nodes ──────────────────────────────────────────────────────


class IntentNode(HCIRNode):
    """A user's high-level intent.  Survives even if plans fail."""

    node_type: HCIRNodeType = HCIRNodeType.INTENT
    category: CognitiveCategory = CognitiveCategory.PLANNING
    description: str = ""
    resolved: bool = False


class GoalNode(HCIRNode):
    """An active objective or target state to achieve."""

    node_type: HCIRNodeType = HCIRNodeType.GOAL
    category: CognitiveCategory = CognitiveCategory.PLANNING
    goal_lifecycle: GoalLifecycle = GoalLifecycle.CREATED
    description: str = ""
    priority: Priority = 0.5
    resolved: bool = False


class ConstraintNode(HCIRNode):
    """A rule, boundary, or policy restriction."""

    node_type: HCIRNodeType = HCIRNodeType.CONSTRAINT
    category: CognitiveCategory = CognitiveCategory.REASONING
    name: str = ""
    expression: str = ""
    enforcement: str = "HARD"  # "HARD" | "SOFT"


# ── Epistemology Hierarchy ───────────────────────────────────────────────


class ObservationNode(HCIRNode):
    """Raw grounded telemetry from environment, sensors, or user."""

    node_type: HCIRNodeType = HCIRNodeType.OBSERVATION
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    payload: dict[str, Any] = Field(default_factory=dict)
    sensor_source: str = ""


class FactNode(HCIRNode):
    """A sensor-verified, grounded observation."""

    node_type: HCIRNodeType = HCIRNodeType.FACT
    category: CognitiveCategory = CognitiveCategory.REASONING
    claim: str = ""


class BeliefNode(HCIRNode):
    """An integrated assertion held by the system."""

    node_type: HCIRNodeType = HCIRNodeType.BELIEF
    category: CognitiveCategory = CognitiveCategory.REASONING
    claim: str = ""
    belief_type: str = "factual"  # factual, causal, procedural, strategic
    evidence_sources: list[str] = Field(default_factory=list)


class HypothesisNode(HCIRNode):
    """A candidate claim under evaluation."""

    node_type: HCIRNodeType = HCIRNodeType.HYPOTHESIS
    category: CognitiveCategory = CognitiveCategory.REASONING
    claim: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)


class PredictionNode(HCIRNode):
    """A simulated or predicted future state."""

    node_type: HCIRNodeType = HCIRNodeType.PREDICTION
    category: CognitiveCategory = CognitiveCategory.REASONING
    claim: str = ""
    predicted_outcome: str = ""
    time_horizon_ms: TimeDuration = 0


# ── Execution Nodes ──────────────────────────────────────────────────────


class ActionNode(HCIRNode):
    """A declarative action, independent of specific tools/plugins.

    Actions declare *intent* and *requirements*.  The CapabilityResolver
    binds them to concrete implementations at runtime.
    """

    node_type: HCIRNodeType = HCIRNodeType.ACTION
    category: CognitiveCategory = CognitiveCategory.EXECUTION
    intent: str = ""
    requirements: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    estimated_cost: CostMetric = 0
    permissions: list[str] = Field(default_factory=list)


class EventNode(HCIRNode):
    """A first-class chronological event tick.

    Examples: UserSpoke, ToolFailed, GoalCompleted, SimulationFinished.
    """

    node_type: HCIRNodeType = HCIRNodeType.EVENT
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    event_kind: str = ""
    event_data: dict[str, Any] = Field(default_factory=dict)
    event_timestamp: float = Field(default_factory=time.time)


class ResourceNode(HCIRNode):
    """A budget or resource allocation node (cpu, tokens, time, etc.)."""

    node_type: HCIRNodeType = HCIRNodeType.RESOURCE
    category: CognitiveCategory = CognitiveCategory.EXECUTION
    resource_type: str = ""  # "cpu", "tokens", "time", "battery", "attention"
    allocated: float = 0.0
    consumed: float = 0.0
    limit: float = 0.0
    is_hard: bool = True  # Hard = physical limit; Soft = cognitive budget


class CapabilityNode(HCIRNode):
    """Declares what the system or a plugin can do."""

    node_type: HCIRNodeType = HCIRNodeType.CAPABILITY
    category: CognitiveCategory = CognitiveCategory.EXECUTION
    capability_name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


# ── Memory Class Nodes ───────────────────────────────────────────────────


class EpisodeNode(HCIRNode):
    """A recalled episode from conversation or experience history."""

    node_type: HCIRNodeType = HCIRNodeType.EPISODE
    category: CognitiveCategory = CognitiveCategory.MEMORY
    summary: str = ""
    outcome: str = ""
    reward: float = 0.0


class ConceptNode(HCIRNode):
    """A long-term factual concept in semantic memory."""

    node_type: HCIRNodeType = HCIRNodeType.CONCEPT
    category: CognitiveCategory = CognitiveCategory.MEMORY
    label: str = ""
    definition: str = ""
    domain: str = ""


class SkillNode(HCIRNode):
    """A reusable, learned skill with success tracking."""

    node_type: HCIRNodeType = HCIRNodeType.SKILL
    category: CognitiveCategory = CognitiveCategory.MEMORY
    skill_name: str = ""
    description: str = ""
    success_rate: Confidence = 0.0
    invocation_count: int = 0


class ProcedureNode(HCIRNode):
    """A parameterized, reusable bytecode subroutine."""

    node_type: HCIRNodeType = HCIRNodeType.PROCEDURE
    category: CognitiveCategory = CognitiveCategory.MEMORY
    procedure_name: str = ""
    parameters: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    # Bytecode instructions are stored by reference, not inline


class ValueNode(HCIRNode):
    """An emotional or alignment preference marker."""

    node_type: HCIRNodeType = HCIRNodeType.VALUE
    category: CognitiveCategory = CognitiveCategory.VALUE
    dimension: str = ""  # e.g., "utility", "safety", "curiosity"
    weight: Confidence = 0.5


class ExternalKnowledgeNode(HCIRNode):
    """A reference to knowledge from an external source."""

    node_type: HCIRNodeType = HCIRNodeType.EXTERNAL_KNOWLEDGE
    category: CognitiveCategory = CognitiveCategory.MEMORY
    source_uri: str = ""
    content_hash: str = ""
    summary: str = ""


# ── World Model & Predictive Nodes ───────────────────────────────────────


class WorldVariableNode(HCIRNode):
    """An environmental parameter in the world model (e.g. temperature, humidity, market demand)."""

    node_type: HCIRNodeType = HCIRNodeType.WORLD_VARIABLE
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    variable_name: str = ""
    value: Any = None
    unit: str = ""
    min_value: float | None = None
    max_value: float | None = None


class PhysicalEntityNode(HCIRNode):
    """A physical asset, component, or system in the physical world."""

    node_type: HCIRNodeType = HCIRNodeType.PHYSICAL_ENTITY
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    entity_name: str = ""
    entity_type: str = ""
    status: str = "operational"
    properties: dict[str, Any] = Field(default_factory=dict)


class EnvironmentStateNode(HCIRNode):
    """A macro snapshot of environmental state."""

    node_type: HCIRNodeType = HCIRNodeType.ENVIRONMENT_STATE
    category: CognitiveCategory = CognitiveCategory.PERCEPTION
    environment_name: str = ""
    active_variables: list[str] = Field(default_factory=list)
    overall_status: str = "nominal"


# ═══════════════════════════════════════════════════════════════════════════
# Node Type Registry — for deserialization & validation
# ═══════════════════════════════════════════════════════════════════════════

#: Maps ``HCIRNodeType`` enum values to their typed subclass.
NODE_TYPE_REGISTRY: dict[HCIRNodeType, type[HCIRNode]] = {
    HCIRNodeType.INTENT: IntentNode,
    HCIRNodeType.GOAL: GoalNode,
    HCIRNodeType.CONSTRAINT: ConstraintNode,
    HCIRNodeType.OBSERVATION: ObservationNode,
    HCIRNodeType.FACT: FactNode,
    HCIRNodeType.BELIEF: BeliefNode,
    HCIRNodeType.HYPOTHESIS: HypothesisNode,
    HCIRNodeType.PREDICTION: PredictionNode,
    HCIRNodeType.ACTION: ActionNode,
    HCIRNodeType.EVENT: EventNode,
    HCIRNodeType.RESOURCE: ResourceNode,
    HCIRNodeType.CAPABILITY: CapabilityNode,
    HCIRNodeType.EPISODE: EpisodeNode,
    HCIRNodeType.CONCEPT: ConceptNode,
    HCIRNodeType.SKILL: SkillNode,
    HCIRNodeType.PROCEDURE: ProcedureNode,
    HCIRNodeType.VALUE: ValueNode,
    HCIRNodeType.EXTERNAL_KNOWLEDGE: ExternalKnowledgeNode,
    HCIRNodeType.WORLD_VARIABLE: WorldVariableNode,
    HCIRNodeType.PHYSICAL_ENTITY: PhysicalEntityNode,
    HCIRNodeType.ENVIRONMENT_STATE: EnvironmentStateNode,
}


# ═══════════════════════════════════════════════════════════════════════════
# Hyperedge
# ═══════════════════════════════════════════════════════════════════════════


class HCIREdge(BaseModel):
    """A typed hyperedge connecting multiple source and target nodes.

    Hyperedges allow representations like "Goal G1 depends on
    Constraint C1, Resource R1, and Capability Cap1" as a single
    relationship.
    """

    id: str = Field(default_factory=lambda: _new_id("e"))
    edge_type: HCIREdgeType
    sources: list[str]  # Node IDs
    targets: list[str]  # Node IDs
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    weight: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Graph — single graph with views
# ═══════════════════════════════════════════════════════════════════════════


class CognitiveGraph:
    """The unified typed cognitive hypergraph.

    A single graph instance holds all cognitive entities.  Different
    subsystems access filtered *views* (knowledge, execution, memory,
    simulation) rather than separate graph objects.

    Internally maintains hash-map indexes for O(1) lookups by ID,
    and secondary indexes by type, category, lifecycle, scope, and
    capability for efficient querying.
    """

    def __init__(self) -> None:
        # ── Primary storage ──
        self._nodes: dict[str, HCIRNode] = {}
        self._edges: dict[str, HCIREdge] = {}

        # ── Secondary indexes (type → set of node IDs) ──
        self._idx_by_type: dict[HCIRNodeType, set[str]] = {t: set() for t in HCIRNodeType}
        self._idx_by_category: dict[CognitiveCategory, set[str]] = {
            c: set() for c in CognitiveCategory
        }
        self._idx_by_lifecycle: dict[NodeLifecycle, set[str]] = {l: set() for l in NodeLifecycle}
        self._idx_by_scope: dict[str, set[str]] = {}  # tenant_id → node IDs
        self._idx_by_tag: dict[str, set[str]] = {}  # tag → node IDs

        # ── Edge indexes ──
        self._edges_by_source: dict[str, set[str]] = {}  # node_id → edge IDs
        self._edges_by_target: dict[str, set[str]] = {}  # node_id → edge IDs

    # ── Node Operations ──────────────────────────────────────────────

    def add_node(self, node: HCIRNode) -> None:
        """Add a node to the graph.  Raises ValueError on duplicate ID."""
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node ID: {node.id}")
        self._nodes[node.id] = node
        self._index_node(node)

    def upsert_node(self, node: HCIRNode) -> None:
        """Add or replace a node in the graph."""
        old = self._nodes.get(node.id)
        if old is not None:
            self._deindex_node(old)
        self._nodes[node.id] = node
        self._index_node(node)

    def remove_node(self, node_id: str) -> HCIRNode | None:
        """Remove a node and all its connected edges.  Returns the removed node."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return None
        self._deindex_node(node)
        # Remove edges connected to this node
        connected_edge_ids = set()
        connected_edge_ids.update(self._edges_by_source.pop(node_id, set()))
        connected_edge_ids.update(self._edges_by_target.pop(node_id, set()))
        for eid in connected_edge_ids:
            edge = self._edges.pop(eid, None)
            if edge:
                # Clean up reverse indexes for other endpoints
                for src in edge.sources:
                    if src != node_id:
                        self._edges_by_source.get(src, set()).discard(eid)
                for tgt in edge.targets:
                    if tgt != node_id:
                        self._edges_by_target.get(tgt, set()).discard(eid)
        return node

    def get_node(self, node_id: str) -> HCIRNode | None:
        """Retrieve a node by ID.  O(1)."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def all_nodes(self) -> Iterator[HCIRNode]:
        """Iterate over all nodes."""
        yield from self._nodes.values()

    # ── Edge Operations ──────────────────────────────────────────────

    def add_edge(self, edge: HCIREdge) -> None:
        """Add an edge.  Raises ValueError on duplicate ID or dangling refs."""
        if edge.id in self._edges:
            raise ValueError(f"Duplicate edge ID: {edge.id}")
        for nid in edge.sources + edge.targets:
            if nid not in self._nodes:
                raise ValueError(f"Dangling edge reference: node '{nid}' not in graph")
        self._edges[edge.id] = edge
        for src in edge.sources:
            self._edges_by_source.setdefault(src, set()).add(edge.id)
        for tgt in edge.targets:
            self._edges_by_target.setdefault(tgt, set()).add(edge.id)

    def remove_edge(self, edge_id: str) -> HCIREdge | None:
        """Remove an edge by ID."""
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return None
        for src in edge.sources:
            self._edges_by_source.get(src, set()).discard(edge_id)
        for tgt in edge.targets:
            self._edges_by_target.get(tgt, set()).discard(edge_id)
        return edge

    def get_edge(self, edge_id: str) -> HCIREdge | None:
        return self._edges.get(edge_id)

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self._edges

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def all_edges(self) -> Iterator[HCIREdge]:
        yield from self._edges.values()

    def edges_from(self, node_id: str) -> list[HCIREdge]:
        """All edges where ``node_id`` is a source."""
        return [
            self._edges[eid]
            for eid in self._edges_by_source.get(node_id, set())
            if eid in self._edges
        ]

    def edges_to(self, node_id: str) -> list[HCIREdge]:
        """All edges where ``node_id`` is a target."""
        return [
            self._edges[eid]
            for eid in self._edges_by_target.get(node_id, set())
            if eid in self._edges
        ]

    # ── Indexed Queries ──────────────────────────────────────────────

    def nodes_by_type(self, node_type: HCIRNodeType) -> list[HCIRNode]:
        """O(k) where k is the number of nodes of that type."""
        return [
            self._nodes[nid]
            for nid in self._idx_by_type.get(node_type, set())
            if nid in self._nodes
        ]

    def nodes_by_category(self, category: CognitiveCategory) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_category.get(category, set())
            if nid in self._nodes
        ]

    def nodes_by_lifecycle(self, lifecycle: NodeLifecycle) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_lifecycle.get(lifecycle, set())
            if nid in self._nodes
        ]

    def nodes_by_scope(self, tenant_id: str) -> list[HCIRNode]:
        return [
            self._nodes[nid]
            for nid in self._idx_by_scope.get(tenant_id, set())
            if nid in self._nodes
        ]

    def nodes_by_tag(self, tag: str) -> list[HCIRNode]:
        return [self._nodes[nid] for nid in self._idx_by_tag.get(tag, set()) if nid in self._nodes]

    # ── Views ────────────────────────────────────────────────────────

    def knowledge_view(self) -> list[HCIRNode]:
        """Facts, Beliefs, Concepts, Procedures — slowly changing knowledge."""
        knowledge_types = {
            HCIRNodeType.FACT,
            HCIRNodeType.BELIEF,
            HCIRNodeType.CONCEPT,
            HCIRNodeType.PROCEDURE,
            HCIRNodeType.EXTERNAL_KNOWLEDGE,
        }
        result: list[HCIRNode] = []
        for t in knowledge_types:
            result.extend(self.nodes_by_type(t))
        return result

    def execution_view(self) -> list[HCIRNode]:
        """Goals, Actions, Resources — rapidly changing execution state."""
        exec_types = {
            HCIRNodeType.GOAL,
            HCIRNodeType.INTENT,
            HCIRNodeType.ACTION,
            HCIRNodeType.RESOURCE,
            HCIRNodeType.CONSTRAINT,
        }
        result: list[HCIRNode] = []
        for t in exec_types:
            result.extend(self.nodes_by_type(t))
        return result

    def memory_view(self) -> list[HCIRNode]:
        """Episodes, Skills, Values — episodic and procedural memory."""
        mem_types = {
            HCIRNodeType.EPISODE,
            HCIRNodeType.SKILL,
            HCIRNodeType.VALUE,
        }
        result: list[HCIRNode] = []
        for t in mem_types:
            result.extend(self.nodes_by_type(t))
        return result

    # ── Internal Indexing ────────────────────────────────────────────

    def _index_node(self, node: HCIRNode) -> None:
        self._idx_by_type[node.node_type].add(node.id)
        self._idx_by_category[node.category].add(node.id)
        self._idx_by_lifecycle[node.lifecycle].add(node.id)
        self._idx_by_scope.setdefault(node.scope.tenant_id, set()).add(node.id)
        for tag in node.tags:
            self._idx_by_tag.setdefault(tag, set()).add(node.id)

    def _deindex_node(self, node: HCIRNode) -> None:
        self._idx_by_type[node.node_type].discard(node.id)
        self._idx_by_category[node.category].discard(node.id)
        self._idx_by_lifecycle[node.lifecycle].discard(node.id)
        scope_set = self._idx_by_scope.get(node.scope.tenant_id)
        if scope_set:
            scope_set.discard(node.id)
        for tag in node.tags:
            tag_set = self._idx_by_tag.get(tag)
            if tag_set:
                tag_set.discard(node.id)
