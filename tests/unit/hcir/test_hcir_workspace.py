"""Unit tests for HCIRWorkspaceState — graph ops, resources, forking, queries."""

import pytest

from hbllm.hcir.graph import (
    BeliefNode,
    FactNode,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    NodeLifecycle,
)
from hbllm.hcir.query import GraphQuery
from hbllm.hcir.workspace import HCIRWorkspaceState, ResourceBudget


class TestWorkspaceNodeOperations:
    """Test event-logged graph operations."""

    def test_add_and_get_node(self):
        ws = HCIRWorkspaceState()
        node = GoalNode(id="g1", description="Test goal")
        ws.add_node(node)
        assert ws.get_node("g1") is not None
        assert ws.graph.node_count == 1

    def test_upsert_node(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="v1"))
        ws.upsert_node(GoalNode(id="g1", description="v2"))
        node = ws.get_node("g1")
        assert isinstance(node, GoalNode)
        assert node.description == "v2"

    def test_remove_node(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="a"))
        removed = ws.remove_node("g1")
        assert removed is not None
        assert ws.get_node("g1") is None

    def test_add_and_remove_edge(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="a"))
        ws.add_node(FactNode(id="f1", claim="b"))
        edge = HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["f1"], targets=["g1"])
        ws.add_edge(edge)
        assert ws.get_edge("e1") is not None
        ws.remove_edge("e1")
        assert ws.get_edge("e1") is None


class TestWorkspaceEventLogging:
    """Test that operations produce events in the event store."""

    def test_events_recorded(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="a"))
        ws.add_node(FactNode(id="f1", claim="b"))
        ws.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["f1"], targets=["g1"])
        )
        ws.remove_node("g1")
        # Should have: snapshot_created(bootstrap) + node_added + node_added + edge_added + node_removed
        events = ws.event_store.get_events()
        assert len(events) >= 3  # At least the three graph mutation events


class TestWorkspaceQuery:
    """Test declarative query API."""

    def test_query_by_type(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="a"))
        ws.add_node(GoalNode(id="g2", description="b"))
        ws.add_node(BeliefNode(id="b1", claim="c"))
        result = ws.query(GraphQuery(node_type=HCIRNodeType.GOAL))
        assert result.total_matches == 2

    def test_query_empty(self):
        ws = HCIRWorkspaceState()
        result = ws.query(GraphQuery(node_type=HCIRNodeType.SKILL))
        assert result.total_matches == 0


class TestWorkspaceResources:
    """Test resource budget tracking."""

    def test_register_and_consume(self):
        ws = HCIRWorkspaceState()
        budget = ResourceBudget(name="tokens", limit=1000.0, is_hard=True)
        ws.register_resource(budget)
        assert ws.consume_resource("tokens", 500.0) is True
        assert ws.get_resource("tokens").remaining == 500.0

    def test_hard_budget_exceeds(self):
        ws = HCIRWorkspaceState()
        budget = ResourceBudget(name="tokens", limit=100.0, is_hard=True)
        ws.register_resource(budget)
        assert ws.consume_resource("tokens", 150.0) is False

    def test_unknown_resource_unconstrained(self):
        ws = HCIRWorkspaceState()
        assert ws.consume_resource("nonexistent", 99999.0) is True


class TestWorkspaceForking:
    """Test simulation branch forking."""

    def test_fork_creates_independent_copy(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="main goal"))
        forked = ws.fork("simulation_1")
        # Add node to fork, should not appear in main
        forked.add_node(GoalNode(id="g2", description="sim goal"))
        assert forked.get_node("g2") is not None
        assert ws.get_node("g2") is None

    def test_fork_preserves_original_nodes(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="main goal"))
        forked = ws.fork("sim")
        assert forked.get_node("g1") is not None

    def test_fork_duplicate_name_raises(self):
        ws = HCIRWorkspaceState()
        ws.fork("sim")
        with pytest.raises(ValueError, match="already exists"):
            ws.fork("sim")

    def test_drop_branch(self):
        ws = HCIRWorkspaceState()
        ws.fork("sim")
        assert ws.drop_branch("sim") is True
        assert ws.drop_branch("nonexistent") is False

    def test_branch_names(self):
        ws = HCIRWorkspaceState()
        ws.fork("a")
        ws.fork("b")
        assert set(ws.branch_names) == {"a", "b"}


class TestWorkspaceActiveGoals:
    """Test convenience view for active goals."""

    def test_active_goals(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="active", lifecycle=NodeLifecycle.ACTIVE))
        ws.add_node(GoalNode(id="g2", description="archived", lifecycle=NodeLifecycle.ARCHIVED))
        ws.add_node(GoalNode(id="g3", description="created"))
        active = ws.active_goals()
        ids = [g.id for g in active]
        assert "g1" in ids
        assert "g3" in ids
        assert "g2" not in ids


class TestWorkspaceSnapshots:
    """Test snapshot creation and versioning."""

    def test_initial_version(self):
        ws = HCIRWorkspaceState()
        assert ws.current_version == 1  # Bootstrap snapshot

    def test_create_snapshot_increments_version(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g1", description="a"))
        snap = ws.create_snapshot()
        assert snap.version == 2
        assert ws.current_version == 2
