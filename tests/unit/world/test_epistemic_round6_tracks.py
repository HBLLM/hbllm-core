from __future__ import annotations

from hbllm.hcir.graph import (
    CognitiveGraph,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.native_adapter import (
    NativeGraphAdapter,
)

"""Unit tests for Round 6 Epistemic & Cortical Neuro-Symbolic Enhancements.

Track 1: Dual Neuromodulation (VTA Dopamine RPE vs. LC Noradrenaline Adaptive Gain)
Track 2: Retrosplenial Egocentric-to-Allocentric Spatial Coordinate Frame Transform
Track 3: Pearl Level-3 Counterfactual Simulation & Regret Minimization
Track 4: Native Structural Graph Multi-Source Parallel Reachability Cones
"""


class TestTrack2RetrosplenialSpatialCoordinateTransform:
    """Track 2: Retrosplenial Egocentric-to-Allocentric transformations."""

    def test_native_adapter_parallel_reachability(self):
        adapter = NativeGraphAdapter()
        adapter.add_node("subgoal_a", "goal")
        adapter.add_node("subgoal_b", "goal")
        adapter.add_node("door_1", "barrier")
        adapter.add_node("door_2", "barrier")
        adapter.add_node("terminal", "goal")

        adapter.add_edge("e1", "depends_on", ["subgoal_a"], ["door_1"])
        adapter.add_edge("e2", "depends_on", ["door_1"], ["terminal"])
        adapter.add_edge("e3", "depends_on", ["subgoal_b"], ["door_2"])

        cones = adapter.parallel_reachability_cones(
            seed_nodes=["subgoal_a", "subgoal_b"], max_depth=5
        )

        assert "subgoal_a" in cones
        assert "subgoal_b" in cones
        assert cones["subgoal_a"] == {"subgoal_a", "door_1", "terminal"}
        assert cones["subgoal_b"] == {"subgoal_b", "door_2"}

    def test_cognitive_graph_parallel_reachability(self):
        graph = CognitiveGraph()
        n1 = GoalNode(id="g1", description="G1")
        n2 = GoalNode(id="g2", description="G2")
        n3 = GoalNode(id="g3", description="G3")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        graph.add_edge(
            HCIREdge(id="e12", edge_type=HCIREdgeType.DEPENDS_ON, sources=["g1"], targets=["g2"])
        )
        graph.add_edge(
            HCIREdge(id="e23", edge_type=HCIREdgeType.DEPENDS_ON, sources=["g2"], targets=["g3"])
        )

        cones = graph.parallel_reachability_cones(seed_nodes=["g1", "g2"])
        assert cones["g1"] == {"g1", "g2", "g3"}
        assert cones["g2"] == {"g2", "g3"}
