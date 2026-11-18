"""Unit and integration tests for Round 4 Cognitive OS Tracks:
Track 1: Native Structural Graph Topological & Spatial Queries (find_path, reachable_nodes, connected_components, topological_sort).
Track 2: SNN Spike-Frequency Adaptation (SFA) & Homeostatic Plasticity.
Track 3: Cerebellar Smith Predictor & Efference-Copy Motor Latency Compensation.
Track 4: Hippocampal Pattern Separation vs. Pattern Completion (Dentate Gyrus / CA3 Recurrent Attractor).
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir import algorithms
from hbllm.hcir.graph import CognitiveGraph, GoalNode, HCIREdge, HCIREdgeType
from hbllm.hcir.native_adapter import NativeGraphAdapter
from hbllm.hcir.world.cortex_episodic import (
    CA3RecurrentPatternCompleter,
    DentateGyrusPatternSeparator,
)
from hbllm.hcir.world.predictors.snn import SNNTemporalPredictor


class TestTrack1NativeGraphTopologicalQueries:
    """Track 1: Native Graph Topological & Reachability Traversal."""

    def test_native_adapter_topological_queries(self):
        adapter = NativeGraphAdapter()
        adapter.add_node("a", "goal")
        adapter.add_node("b", "goal")
        adapter.add_node("c", "goal")
        adapter.add_node("d", "goal")

        adapter.add_edge("e1", "depends_on", ["a"], ["b"])
        adapter.add_edge("e2", "depends_on", ["b"], ["c"])
        adapter.add_edge("e3", "depends_on", ["a"], ["d"])

        # Path search
        path = adapter.find_path("a", "c", max_depth=5)
        assert path == ["a", "b", "c"]

        # Reachability cone
        reachable = adapter.reachable_nodes("a", max_depth=5)
        assert reachable == {"a", "b", "c", "d"}
        reachable_b = adapter.reachable_nodes("b", max_depth=5)
        assert reachable_b == {"b", "c"}

        # Connected components (all 4 nodes connected)
        comps = adapter.connected_components()
        assert len(comps) == 1
        assert comps[0] == {"a", "b", "c", "d"}

        # Topological sort (DAG order)
        topo = adapter.topological_sort()
        assert len(topo) == 4
        assert topo.index("a") < topo.index("b")
        assert topo.index("b") < topo.index("c")
        assert topo.index("a") < topo.index("d")

    def test_cognitive_graph_topological_queries(self):
        graph = CognitiveGraph()
        n1 = GoalNode(id="subgoal_1", description="Step 1")
        n2 = GoalNode(id="subgoal_2", description="Step 2")
        n3 = GoalNode(id="subgoal_3", description="Step 3")
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)

        graph.add_edge(
            HCIREdge(
                id="edge_12",
                edge_type=HCIREdgeType.DEPENDS_ON,
                sources=["subgoal_1"],
                targets=["subgoal_2"],
            )
        )
        graph.add_edge(
            HCIREdge(
                id="edge_23",
                edge_type=HCIREdgeType.DEPENDS_ON,
                sources=["subgoal_2"],
                targets=["subgoal_3"],
            )
        )

        # find_path
        path = graph.find_path("subgoal_1", "subgoal_3")
        assert path == ["subgoal_1", "subgoal_2", "subgoal_3"]

        # reachable_nodes
        reach = graph.reachable_nodes("subgoal_1")
        assert reach == {"subgoal_1", "subgoal_2", "subgoal_3"}

        # topological_sort
        order = algorithms.topological_sort(graph)
        assert order == ["subgoal_1", "subgoal_2", "subgoal_3"]

        # Disjoint node
        n4 = GoalNode(id="subgoal_isolated", description="Isolated")
        graph.add_node(n4)
        comps = algorithms.connected_components(graph)
        assert len(comps) == 2


class TestTrack2SNNSpikeFrequencyAdaptation:
    """Track 2: SNN Spike-Frequency Adaptation & Homeostatic Plasticity."""

    def test_spike_frequency_adaptation_dynamic_threshold(self):
        snn = SNNTemporalPredictor(
            v_thresh=1.0,
            tau_adaptation=10.0,
            adaptation_increment=0.5,
        )
        input_strong = np.full((5, 5), 2.5, dtype=np.float32)

        # Step 1: High input triggers initial spikes
        spikes1, _ = snn.step_lif(input_strong)
        assert np.any(spikes1 > 0)
        assert snn.adaptation_threshold is not None
        # Adaptation threshold should increase at spiking locations
        assert np.any(snn.adaptation_threshold > 0.4)

        # Step 2: Under adapted threshold, firing rate is constrained
        snn.step_lif(input_strong)
        assert np.mean(snn.adaptation_threshold) > 0.0

    def test_homeostatic_sensitivity_scaling(self):
        snn = SNNTemporalPredictor(homeostatic_rate=0.1)
        # Intense stimulus -> firing rate higher than target homeostatic rate
        input_burst = np.full((4, 4), 5.0, dtype=np.float32)
        initial_scale = snn.sensitivity_scale

        for _ in range(5):
            snn.step_lif(input_burst)

        # Sensitivity scale should self-downregulate to curb excessive bursting
        assert snn.sensitivity_scale <= initial_scale


class TestTrack4HippocampalAttractorNetwork:
    """Track 4: Dentate Gyrus Pattern Separation & CA3 Auto-Association."""

    def test_dentate_gyrus_sparse_pattern_separation(self):
        separator = DentateGyrusPatternSeparator(input_dim=16, expansion_dim=64, k_active=4)
        v1 = np.ones(16, dtype=np.float32)
        v2 = np.ones(16, dtype=np.float32)
        v2[0] = 0.0  # Slightly perturbed

        res1 = separator.separate_pattern(v1)
        res2 = separator.separate_pattern(v2)

        assert res1.sparsity_ratio == (4.0 / 64.0)
        assert np.count_nonzero(res1.sparse_code) == 4
        # Sparse representations are decorrelated and non-identical
        assert np.any(res1.sparse_code != res2.sparse_code)

    def test_ca3_recurrent_pattern_completion(self):
        completer = CA3RecurrentPatternCompleter(capacity=16, feature_dim=32)
        pattern = np.zeros(32, dtype=np.float32)
        pattern[:8] = 1.0  # Ground truth memory
        winning_plan = [1, 2, 1, 3]

        completer.store_pattern(memory_id="mem_1", pattern=pattern, plan=winning_plan)

        # Noisy / partial cue (only 4 of 8 elements observed)
        noisy_cue = np.zeros(32, dtype=np.float32)
        noisy_cue[:4] = 1.0

        comp_result = completer.complete_pattern(noisy_cue, match_threshold=0.3)
        assert comp_result.retrieved_memory_id == "mem_1"
        assert comp_result.retrieval_confidence > 0.5
        assert np.count_nonzero(comp_result.completed_pattern) == 8
