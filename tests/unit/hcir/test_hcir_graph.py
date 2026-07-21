"""Unit tests for CognitiveGraph, typed nodes, edges, indexes, and views."""

import pytest

from hbllm.hcir.graph import (
    NODE_TYPE_REGISTRY,
    ActionNode,
    BeliefNode,
    CapabilityNode,
    CognitiveCategory,
    CognitiveGraph,
    ConceptNode,
    ConstraintNode,
    EpisodeNode,
    FactNode,
    GoalLifecycle,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    IntentNode,
    NodeLifecycle,
    ProcedureNode,
    ResourceNode,
    SkillNode,
    ValueNode,
)

# ═══════════════════════════════════════════════════════════════════════════
# Typed Node Subclasses
# ═══════════════════════════════════════════════════════════════════════════


class TestTypedNodes:
    """Test that each typed node subclass initialises with correct defaults."""

    def test_goal_node(self):
        g = GoalNode(description="Build solar dehydrator", priority=0.9)
        assert g.node_type == HCIRNodeType.GOAL
        assert g.category == CognitiveCategory.PLANNING
        assert g.description == "Build solar dehydrator"
        assert g.priority == 0.9
        assert g.goal_lifecycle == GoalLifecycle.CREATED
        assert g.resolved is False

    def test_belief_node(self):
        b = BeliefNode(claim="Python is good for ML", belief_type="factual")
        assert b.node_type == HCIRNodeType.BELIEF
        assert b.claim == "Python is good for ML"

    def test_action_node(self):
        a = ActionNode(intent="Search web", requirements=["internet"])
        assert a.node_type == HCIRNodeType.ACTION
        assert a.requirements == ["internet"]

    def test_constraint_node(self):
        c = ConstraintNode(name="budget_limit", expression="cost < 1000", enforcement="HARD")
        assert c.enforcement == "HARD"

    def test_fact_node(self):
        f = FactNode(claim="Earth orbits the Sun")
        assert f.node_type == HCIRNodeType.FACT

    def test_concept_node(self):
        c = ConceptNode(label="Machine Learning", domain="AI")
        assert c.domain == "AI"

    def test_skill_node(self):
        s = SkillNode(skill_name="code_review", success_rate=0.85)
        assert s.success_rate == 0.85

    def test_episode_node(self):
        e = EpisodeNode(summary="User asked about weather", reward=0.7)
        assert e.reward == 0.7

    def test_capability_node(self):
        c = CapabilityNode(capability_name="execute_python")
        assert c.capability_name == "execute_python"

    def test_resource_node(self):
        r = ResourceNode(resource_type="tokens", limit=10000.0, is_hard=True)
        assert r.resource_type == "tokens"

    def test_all_node_types_registered(self):
        """Verify every HCIRNodeType has a registered class."""
        for nt in HCIRNodeType:
            assert nt in NODE_TYPE_REGISTRY, f"Missing registration for {nt}"

    def test_node_auto_id(self):
        """Nodes get unique auto-generated IDs."""
        g1 = GoalNode(description="a")
        g2 = GoalNode(description="b")
        assert g1.id != g2.id
        assert g1.id.startswith("n_")


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveGraph Operations
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveGraphNodes:
    """Test node CRUD and indexing."""

    def test_add_and_get_node(self):
        graph = CognitiveGraph()
        node = GoalNode(id="g1", description="Test goal")
        graph.add_node(node)
        assert graph.get_node("g1") is node
        assert graph.has_node("g1")
        assert graph.node_count == 1

    def test_add_duplicate_raises(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        with pytest.raises(ValueError, match="Duplicate node ID"):
            graph.add_node(GoalNode(id="g1", description="b"))

    def test_upsert_node(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="v1"))
        graph.upsert_node(GoalNode(id="g1", description="v2"))
        node = graph.get_node("g1")
        assert isinstance(node, GoalNode)
        assert node.description == "v2"
        assert graph.node_count == 1

    def test_remove_node(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        removed = graph.remove_node("g1")
        assert removed is not None
        assert not graph.has_node("g1")
        assert graph.node_count == 0

    def test_remove_nonexistent_returns_none(self):
        graph = CognitiveGraph()
        assert graph.remove_node("nonexistent") is None

    def test_remove_node_also_removes_connected_edges(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        graph.add_node(ConstraintNode(id="c1", name="budget"))
        edge = HCIREdge(id="e1", edge_type=HCIREdgeType.DEPENDS_ON, sources=["g1"], targets=["c1"])
        graph.add_edge(edge)
        assert graph.edge_count == 1
        graph.remove_node("g1")
        assert graph.edge_count == 0


class TestCognitiveGraphEdges:
    """Test edge CRUD and referential integrity."""

    def test_add_and_get_edge(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        graph.add_node(FactNode(id="f1", claim="b"))
        edge = HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["f1"], targets=["g1"])
        graph.add_edge(edge)
        assert graph.get_edge("e1") is edge
        assert graph.edge_count == 1

    def test_add_edge_dangling_source_raises(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        edge = HCIREdge(
            id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["missing"], targets=["g1"]
        )
        with pytest.raises(ValueError, match="Dangling edge reference"):
            graph.add_edge(edge)

    def test_add_edge_dangling_target_raises(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        edge = HCIREdge(
            id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["g1"], targets=["missing"]
        )
        with pytest.raises(ValueError, match="Dangling edge reference"):
            graph.add_edge(edge)

    def test_edges_from_and_to(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        graph.add_node(FactNode(id="f1", claim="b"))
        edge = HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["f1"], targets=["g1"])
        graph.add_edge(edge)
        assert len(graph.edges_from("f1")) == 1
        assert len(graph.edges_to("g1")) == 1
        assert len(graph.edges_from("g1")) == 0

    def test_remove_edge(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        graph.add_node(FactNode(id="f1", claim="b"))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["f1"], targets=["g1"])
        )
        removed = graph.remove_edge("e1")
        assert removed is not None
        assert graph.edge_count == 0


class TestCognitiveGraphIndexes:
    """Test secondary index lookups."""

    def _build_graph(self) -> CognitiveGraph:
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="goal1", tags=["urgent"]))
        graph.add_node(GoalNode(id="g2", description="goal2"))
        graph.add_node(BeliefNode(id="b1", claim="belief1"))
        graph.add_node(FactNode(id="f1", claim="fact1"))
        graph.add_node(ConceptNode(id="c1", label="concept1"))
        graph.add_node(EpisodeNode(id="ep1", summary="episode1"))
        graph.add_node(SkillNode(id="s1", skill_name="skill1"))
        return graph

    def test_nodes_by_type(self):
        graph = self._build_graph()
        goals = graph.nodes_by_type(HCIRNodeType.GOAL)
        assert len(goals) == 2
        assert all(isinstance(g, GoalNode) for g in goals)

    def test_nodes_by_category(self):
        graph = self._build_graph()
        planning = graph.nodes_by_category(CognitiveCategory.PLANNING)
        assert len(planning) == 2  # 2 goals

    def test_nodes_by_lifecycle(self):
        graph = self._build_graph()
        created = graph.nodes_by_lifecycle(NodeLifecycle.CREATED)
        assert len(created) == 7  # all default to CREATED

    def test_nodes_by_scope(self):
        graph = self._build_graph()
        default_scope = graph.nodes_by_scope("default")
        assert len(default_scope) == 7

    def test_nodes_by_tag(self):
        graph = self._build_graph()
        urgent = graph.nodes_by_tag("urgent")
        assert len(urgent) == 1
        assert urgent[0].id == "g1"


class TestCognitiveGraphViews:
    """Test filtered view projections."""

    def _build_graph(self) -> CognitiveGraph:
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="goal"))
        graph.add_node(IntentNode(id="i1", description="intent"))
        graph.add_node(ActionNode(id="a1", intent="action"))
        graph.add_node(FactNode(id="f1", claim="fact"))
        graph.add_node(BeliefNode(id="b1", claim="belief"))
        graph.add_node(ConceptNode(id="c1", label="concept"))
        graph.add_node(ProcedureNode(id="p1", procedure_name="proc"))
        graph.add_node(EpisodeNode(id="ep1", summary="episode"))
        graph.add_node(SkillNode(id="s1", skill_name="skill"))
        graph.add_node(ValueNode(id="v1", dimension="utility"))
        return graph

    def test_knowledge_view(self):
        graph = self._build_graph()
        kv = graph.knowledge_view()
        node_ids = {n.id for n in kv}
        assert "f1" in node_ids  # Fact
        assert "b1" in node_ids  # Belief
        assert "c1" in node_ids  # Concept
        assert "p1" in node_ids  # Procedure
        assert "g1" not in node_ids  # Goal is execution

    def test_execution_view(self):
        graph = self._build_graph()
        ev = graph.execution_view()
        node_ids = {n.id for n in ev}
        assert "g1" in node_ids  # Goal
        assert "i1" in node_ids  # Intent
        assert "a1" in node_ids  # Action
        assert "f1" not in node_ids  # Fact is knowledge

    def test_memory_view(self):
        graph = self._build_graph()
        mv = graph.memory_view()
        node_ids = {n.id for n in mv}
        assert "ep1" in node_ids  # Episode
        assert "s1" in node_ids  # Skill
        assert "v1" in node_ids  # Value
        assert "f1" not in node_ids
