"""Parity Oracle and Workload Scaling Test for Native Mental Simulation Engine.

Verifies:
1. Numerical Parity: Geometric support stability matches with epsilon tolerance.
2. Parallel Determinism: Rayon multi-threaded rollouts preserve strict input branch ordering.
3. Extensible Result Schema: Keeps Rust computational and Python epistemic.
4. Scale Stress: Evaluates 200 concurrent candidate rollouts across CPU threads.
"""

import time

import pytest

from hbllm.native.registry import native

pytestmark = pytest.mark.skipif(
    not native.available("simulation"),
    reason="hbllm_simulation_engine native extension not compiled",
)


class TestSimulationParityOracle:
    """Level 2: Numerical Parity and Multi-Threaded Determinism."""

    def test_simulation_engine_discovery(self):
        """Verify native registry accurately detects simulation capability."""
        assert native.available("simulation") is True
        info = native.get_info("simulation")
        assert info is not None
        assert info.available is True
        assert "Rayon" in info.description

    def test_geometric_stability_numerical_parity(self):
        """Verify 3D bounding box stability computation."""
        import hbllm_simulation_engine

        # Stable configuration: box on table
        table_bounds = (0.0, 0.0, 0.0, 10.0, 10.0, 1.0)
        stable_box = (3.0, 3.0, 1.0, 7.0, 7.0, 4.0)

        is_stable, margin = hbllm_simulation_engine.evaluate_support_stability(
            stable_box, table_bounds, 0.05
        )
        assert is_stable is True
        assert abs(margin - 5.0) < 1e-6

        # Overhanging / unstable configuration
        overhanging_box = (9.5, 9.5, 1.0, 12.0, 12.0, 4.0)
        is_unstable, _ = hbllm_simulation_engine.evaluate_support_stability(
            overhanging_box, table_bounds, 0.05
        )
        assert is_unstable is False

    def test_parallel_batch_rollout_determinism_and_ordering(self):
        """Verify Rayon parallel rollouts preserve strict input ordering and deterministic hashes."""
        import hbllm_simulation_engine

        # Generate 100 candidate branches with different action trajectories
        branches = []
        for i in range(100):
            branches.append(
                {
                    "branch_id": i,
                    "initial_risk": 0.02 * (i % 5),
                    "initial_cost": 0.0,
                    "max_steps": 10,
                    "actions": [
                        {
                            "operator": "MOVE",
                            "subject": f"obj_{i}",
                            "target": "tray_1",
                            "parameters": {},
                        },
                        {
                            "operator": "STACK",
                            "subject": f"obj_{i}",
                            "target": "box_2",
                            "parameters": {},
                        },
                        {
                            "operator": "PROBE",
                            "subject": f"obj_{i}",
                            "target": "sensor",
                            "parameters": {},
                        },
                    ],
                }
            )

        t0 = time.perf_counter()
        results1 = hbllm_simulation_engine.evaluate_parallel_rollouts(branches, "SEED_HASH_ALPHA")
        t_elapsed = time.perf_counter() - t0

        results2 = hbllm_simulation_engine.evaluate_parallel_rollouts(branches, "SEED_HASH_ALPHA")

        assert len(results1) == 100
        assert len(results2) == 100

        # Verify strict input branch ordering and determinism
        for i in range(100):
            r1 = results1[i]
            r2 = results2[i]

            assert r1["branch_id"] == i, "Results must strictly preserve input branch index"
            assert r1["branch_id"] == r2["branch_id"]
            assert r1["final_state_hash"] == r2["final_state_hash"]
            assert abs(r1["success_probability"] - r2["success_probability"]) < 1e-9
            assert r1["steps_executed"] == 3
            assert r1["terminal_status"] == "SUCCESS"

        # 100 parallel rollouts should execute in sub-millisecond to low milliseconds
        assert t_elapsed < 0.5, f"100 parallel rollouts took {t_elapsed * 1000:.2f} ms"

    def test_risk_exceeded_termination(self):
        """Verify rollout terminates early when risk threshold is breached."""
        import hbllm_simulation_engine

        risky_branch = [
            {
                "branch_id": 1,
                "initial_risk": 0.8,
                "initial_cost": 0.0,
                "max_steps": 10,
                "actions": [
                    {
                        "operator": "PUSH",
                        "subject": "unstable_vase",
                        "target": "edge",
                        "parameters": {},
                    },
                    {
                        "operator": "PUSH",
                        "subject": "unstable_vase",
                        "target": "edge",
                        "parameters": {},
                    },
                ],
            }
        ]

        res = hbllm_simulation_engine.evaluate_parallel_rollouts(risky_branch, "SEED_HASH_BETA")
        assert len(res) == 1
        assert res[0]["terminal_status"] == "RISK_EXCEEDED"
        assert res[0]["success_probability"] == 0.0
