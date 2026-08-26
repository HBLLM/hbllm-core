"""Parity and Verification Oracle for NativeCognitiveRuntime and DAG Transition Memoization.

Verifies:
1. Resident state persistence and in-memory mutation.
2. Cold vs Warm cache execution, hit/miss tracking, and prefix reuse.
3. Cost-model adaptive dispatch routing logic.
"""

from hbllm.native.registry import WorkloadProfile, native


class TestResidentRuntimeParityOracle:
    """Level 4: Resident Native Cognitive Runtime & DAG Memoization."""

    def test_resident_runtime_state_and_hash(self):
        """Verify NativeCognitiveRuntime maintains state and computes BLAKE3 canonical hash."""
        import hbllm_simulation_engine

        runtime = hbllm_simulation_engine.NativeCognitiveRuntime()
        assert runtime.node_count() == 0
        assert runtime.edge_count() == 0

        runtime.add_node("table_1", "Entity", "ACTIVE", {"material": "wood"}, 100.0)
        runtime.add_node("box_1", "Entity", "ACTIVE", {"weight": "5.0"}, 101.0)
        runtime.add_edge("edge_1", "SUPPORTS", ["table_1"], ["box_1"], 1.0, {}, 102.0)

        assert runtime.node_count() == 2
        assert runtime.edge_count() == 1
        h1 = runtime.canonical_hash()
        assert len(h1) == 64

    def test_dag_transition_cache_hits_and_prefix_reuse(self):
        """Verify cold cache records misses, warm cache records hits, and shared prefixes hit."""
        import hbllm_simulation_engine

        runtime = hbllm_simulation_engine.NativeCognitiveRuntime()
        runtime.add_node("table_1", "Entity", "ACTIVE", {}, 100.0)

        branch_a = {
            "branch_id": 1,
            "actions": [
                {"operator": "MOVE", "subject": "box_1", "target": "table_1", "parameters": {}},
                {"operator": "STACK", "subject": "box_2", "target": "box_1", "parameters": {}},
            ],
            "initial_risk": 0.0,
            "initial_cost": 0.0,
            "max_steps": 10,
        }

        branch_b = {
            "branch_id": 2,
            "actions": [
                {"operator": "MOVE", "subject": "box_1", "target": "table_1", "parameters": {}},
                {"operator": "PROBE", "subject": "box_1", "target": "sensor", "parameters": {}},
            ],
            "initial_risk": 0.0,
            "initial_cost": 0.0,
            "max_steps": 10,
        }

        # 1. Cold execution: Run branch A, then branch B with shared prefix
        results_a, stats_a = runtime.evaluate_rollouts([branch_a], "SEED_EXPERIMENT")
        assert len(results_a) == 1
        assert stats_a["cache_misses"] == 2
        assert stats_a["cache_hits"] == 0

        # Branch B evaluates: Action 1 hits cache from branch A!
        results_b, stats_b = runtime.evaluate_rollouts([branch_b], "SEED_EXPERIMENT")
        assert len(results_b) == 1
        assert stats_b["cache_hits"] == 1
        assert stats_b["cache_misses"] == 1
        assert runtime.cache_size() == 3

        # 2. Warm execution: Repeated evaluation of both branches -> 4 hits, 0 misses!
        results_2, stats_2 = runtime.evaluate_rollouts([branch_a, branch_b], "SEED_EXPERIMENT")
        assert len(results_2) == 2
        assert stats_2["cache_hits"] == 4
        assert stats_2["cache_misses"] == 0

        # Bit-for-bit final state hash determinism
        assert results_a[0]["final_state_hash"] == results_2[0]["final_state_hash"]
        assert results_b[0]["final_state_hash"] == results_2[1]["final_state_hash"]

    def test_cost_model_adaptive_dispatch_routing(self):
        """Verify AdaptiveDispatcher accurately routes based on workload size and complexity."""
        # 1. Tiny non-geometric workload (e.g. 5 simple branches) -> Python preferred
        small_profile = WorkloadProfile(
            batch_size=5,
            state_node_count=20,
            has_geometric_collision=False,
            action_complexity=1,
            estimated_prefix_sharing=0.0,
        )
        assert native.should_execute_native("simulation", small_profile) is False

        # 2. Large complex workload (e.g. 500 branches, high complexity) -> Native Rust preferred
        large_profile = WorkloadProfile(
            batch_size=500,
            state_node_count=100,
            has_geometric_collision=False,
            action_complexity=5,
            estimated_prefix_sharing=0.5,
        )
        assert native.should_execute_native("simulation", large_profile) is True

        # 3. Geometric physics workload -> Native Rust preferred
        geom_profile = WorkloadProfile(
            batch_size=30,
            state_node_count=20,
            has_geometric_collision=True,
            action_complexity=2,
            estimated_prefix_sharing=0.0,
        )
        assert native.should_execute_native("simulation", geom_profile) is True

        # 4. Large graph snapshot -> Native Rust preferred
        snapshot_profile = WorkloadProfile(state_node_count=500)
        assert native.should_execute_native("hcir_graph", snapshot_profile) is True

    def test_deterministic_cache_equivalence(self):
        """Invariant: Memoization changes execution cost, never cognitive semantics.

        Asserts that Cache Hit result is bit-for-bit identical to Cache Miss result
        across final state hash, risk, cost, steps executed, and terminal status.
        """
        import hbllm_simulation_engine

        runtime = hbllm_simulation_engine.NativeCognitiveRuntime()
        runtime.add_node("base_stand", "Entity", "ACTIVE", {"material": "metal"}, 100.0)

        complex_branch = {
            "branch_id": 42,
            "actions": [
                {
                    "operator": "MOVE",
                    "subject": "sensor_pod",
                    "target": "base_stand",
                    "parameters": {},
                },
                {
                    "operator": "STACK",
                    "subject": "antenna",
                    "target": "sensor_pod",
                    "parameters": {},
                },
                {
                    "operator": "PUSH",
                    "subject": "base_stand",
                    "target": "alignment_mark",
                    "parameters": {},
                },
                {"operator": "PROBE", "subject": "antenna", "target": "receiver", "parameters": {}},
            ],
            "initial_risk": 0.05,
            "initial_cost": 1.0,
            "max_steps": 15,
        }

        # 1. First execution (Cold / Cache Miss)
        runtime.clear_cache()
        cold_results, cold_stats = runtime.evaluate_rollouts([complex_branch], "SEED_DETERMINISM")
        assert len(cold_results) == 1
        assert cold_stats["cache_misses"] == 4
        assert cold_stats["cache_hits"] == 0
        cold_res = cold_results[0]

        # 2. Second execution (Warm / Cache Hit)
        warm_results, warm_stats = runtime.evaluate_rollouts([complex_branch], "SEED_DETERMINISM")
        assert len(warm_results) == 1
        assert warm_stats["cache_hits"] == 4
        assert warm_stats["cache_misses"] == 0
        warm_res = warm_results[0]

        # 3. Equivalence assertions: Bit-for-bit semantic identity
        assert cold_res["final_state_hash"] == warm_res["final_state_hash"]
        assert abs(cold_res["success_probability"] - warm_res["success_probability"]) < 1e-12
        assert abs(cold_res["risk_score"] - warm_res["risk_score"]) < 1e-12
        assert abs(cold_res["trajectory_cost"] - warm_res["trajectory_cost"]) < 1e-12
        assert cold_res["steps_executed"] == warm_res["steps_executed"]
        assert cold_res["terminal_status"] == warm_res["terminal_status"]

        # 4. Cleared cache execution (Re-computation verification)
        runtime.clear_cache()
        fresh_results, fresh_stats = runtime.evaluate_rollouts([complex_branch], "SEED_DETERMINISM")
        assert fresh_stats["cache_misses"] == 4
        fresh_res = fresh_results[0]

        assert cold_res["final_state_hash"] == fresh_res["final_state_hash"]
        assert cold_res["trajectory_cost"] == fresh_res["trajectory_cost"]
        assert cold_res["terminal_status"] == fresh_res["terminal_status"]

    def test_multi_step_trajectory_cache_equivalence(self):
        """Invariant: Downstream multi-step branching trajectories remain bit-for-bit identical with DAG cache enabled.

        Evaluates an entire counterfactual branching tree across 5 distinct branches
        where branches share intermediate prefix paths, asserting 100% semantic identity
        between cold execution and warm DAG cached execution.
        """
        import hbllm_simulation_engine

        runtime = hbllm_simulation_engine.NativeCognitiveRuntime()
        runtime.add_node("hub", "Entity", "ACTIVE", {}, 100.0)

        # 5 branches with hierarchical prefix sharing
        branches = [
            {
                "branch_id": i,
                "actions": [
                    {"operator": "MOVE", "subject": "core_pod", "target": "hub", "parameters": {}},
                    {
                        "operator": "STACK",
                        "subject": f"branch_mod_{i // 2}",
                        "target": "core_pod",
                        "parameters": {},
                    },
                    {
                        "operator": "PROBE",
                        "subject": f"leaf_{i}",
                        "target": f"branch_mod_{i // 2}",
                        "parameters": {},
                    },
                ],
                "initial_risk": 0.01 * i,
                "initial_cost": 0.0,
                "max_steps": 10,
            }
            for i in range(5)
        ]

        # 1. Cold execution (all misses or partial intra-batch misses)
        runtime.clear_cache()
        cold_results, _ = runtime.evaluate_rollouts(branches, "SEED_TRAJECTORY")
        assert len(cold_results) == 5

        # 2. Warm execution (all hits from cached simulation DAG)
        warm_results, warm_stats = runtime.evaluate_rollouts(branches, "SEED_TRAJECTORY")
        assert len(warm_results) == 5
        assert warm_stats["cache_hits"] == 15
        assert warm_stats["cache_misses"] == 0

        # 3. Assert complete trajectory-level identity across all 5 branches
        for b_idx in range(5):
            c_res = cold_results[b_idx]
            w_res = warm_results[b_idx]
            assert c_res["branch_id"] == w_res["branch_id"]
            assert c_res["final_state_hash"] == w_res["final_state_hash"]
            assert abs(c_res["risk_score"] - w_res["risk_score"]) < 1e-12
            assert abs(c_res["trajectory_cost"] - w_res["trajectory_cost"]) < 1e-12
            assert c_res["steps_executed"] == w_res["steps_executed"]
            assert c_res["terminal_status"] == w_res["terminal_status"]
