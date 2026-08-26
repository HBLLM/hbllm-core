"""A18 Embodied Simulation & Counterfactual Mental Sandbox Benchmark Suite (19 Scenarios).

Evaluates copy-on-write simulation branches, deterministic state-transition operators,
geometric support reasoning, multi-branch planning, and the flagship obstacle course trial.
"""

from __future__ import annotations

from typing import Any

from hbllm.brain.learning.error_classifier import ErrorClassifier, ErrorContext
from hbllm.brain.simulation import (
    ExecutionPlan,
    MentalSandbox,
    SimulationBranch,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)

# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: A18-01 Branch Isolation & Canonical Immutability
# ═══════════════════════════════════════════════════════════════════════════


class TestBranchIsolation:
    """Mutations inside SimulationBranch do not leak to canonical reality."""

    def test_canonical_immutability_during_simulation(self) -> None:
        graph = CognitiveGraph()
        box = PhysicalEntityNode(id="box_1", entity_type="box", properties={"x": 0.0, "y": 0.0})
        graph.add_node(box)

        sandbox = MentalSandbox()
        branch = sandbox.fork_branch(graph, branch_id="b1")

        # Simulate push in branch
        res = sandbox.simulate_action(branch, "PUSH", {"target_id": "box_1", "dx": 5.0, "dy": 0.0})
        assert res.is_success

        # Branch entity moved to x=5.0
        branch_box = branch.get_node("box_1")
        assert isinstance(branch_box, PhysicalEntityNode)
        assert branch_box.properties["x"] == 5.0

        # Canonical entity remains unchanged at x=0.0
        canonical_box = graph.get_node("box_1")
        assert isinstance(canonical_box, PhysicalEntityNode)
        assert canonical_box.properties["x"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: A18-02 Nested Branch Hierarchy
# ═══════════════════════════════════════════════════════════════════════════


class TestNestedBranches:
    """Nested child branches B1 -> B1A, B1B isolate mutations independently."""

    def test_nested_branch_independence(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup_1", entity_type="cup", properties={"x": 0.0, "y": 0.0})
        graph.add_node(cup)

        sandbox = MentalSandbox()
        b1 = sandbox.fork_branch(graph, branch_id="b1")
        sandbox.simulate_action(b1, "PUSH", {"target_id": "cup_1", "dx": 2.0, "dy": 0.0})

        # Fork sub-branches from b1
        b1a = b1.fork_child("b1a")
        b1b = b1.fork_child("b1b")

        sandbox.simulate_action(
            b1a, "PUSH", {"target_id": "cup_1", "dx": 3.0, "dy": 0.0}
        )  # x becomes 5.0
        sandbox.simulate_action(
            b1b, "PUSH", {"target_id": "cup_1", "dx": -1.0, "dy": 0.0}
        )  # x becomes 1.0

        assert b1a.get_node("cup_1").properties["x"] == 5.0  # type: ignore[union-attr]
        assert b1b.get_node("cup_1").properties["x"] == 1.0  # type: ignore[union-attr]
        assert b1.get_node("cup_1").properties["x"] == 2.0  # type: ignore[union-attr]
        assert graph.get_node("cup_1").properties["x"] == 0.0  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: A18-03 Deterministic Simulation Replay
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterministicReplay:
    """Same base graph + same actions produce identical state hashes across runs."""

    def test_simulation_replay_state_hash(self) -> None:
        def _run_sim() -> str:
            g = CognitiveGraph()
            box = PhysicalEntityNode(id="b_1", entity_type="box", properties={"x": 1.0, "y": 2.0})
            table = PhysicalEntityNode(
                id="t_1", entity_type="table", properties={"x": 0.0, "y": 0.0}
            )
            g.add_node(box)
            g.add_node(table)

            sandbox = MentalSandbox()
            actions: list[tuple[str, dict[str, Any]]] = [
                ("PUSH", {"target_id": "b_1", "dx": 2.0, "dy": 1.0}),
                ("STACK", {"item_id": "b_1", "base_id": "t_1"}),
            ]
            branch, _ = sandbox.simulate_trajectory(g, actions, branch_id="fixed_b1")
            return branch.compute_current_state_hash()

        hash1 = _run_sim()
        hash2 = _run_sim()

        assert hash1 == hash2
        assert len(hash1) == 16


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: A18-04 Spatial Transition & Displacement
# ═══════════════════════════════════════════════════════════════════════════


class TestSpatialTransition:
    """PUSH operator computes correct spatial displacement."""

    def test_push_operator_displacement(self) -> None:
        graph = CognitiveGraph()
        ball = PhysicalEntityNode(id="ball_1", entity_type="ball", properties={"x": 10.0, "y": 5.0})
        graph.add_node(ball)

        sandbox = MentalSandbox()
        branch = sandbox.fork_branch(graph)
        res = sandbox.simulate_action(
            branch, "PUSH", {"target_id": "ball_1", "dx": -4.0, "dy": 3.0}
        )

        assert res.is_success
        node = branch.get_node("ball_1")
        assert isinstance(node, PhysicalEntityNode)
        assert node.properties["x"] == 6.0
        assert node.properties["y"] == 8.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: A18-05 Support Stability Reasoning
# ═══════════════════════════════════════════════════════════════════════════


class TestSupportStability:
    """Flat support (table) is stable; convex curved support (sphere) is unstable."""

    def test_geometric_stability_box_vs_ball(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup_1", entity_type="cup", properties={"shape": "cylinder"})
        table = PhysicalEntityNode(
            id="table_1", entity_type="table", properties={"geometry": "flat"}
        )
        ball = PhysicalEntityNode(id="ball_1", entity_type="ball", properties={"shape": "sphere"})
        graph.add_node(cup)
        graph.add_node(table)
        graph.add_node(ball)

        sandbox = MentalSandbox()

        # Branch 1: Stack cup on flat table -> STABLE
        b_flat = sandbox.fork_branch(graph, "b_flat")
        res_flat = sandbox.simulate_action(
            b_flat, "STACK", {"item_id": "cup_1", "base_id": "table_1"}
        )
        assert res_flat.is_success
        assert res_flat.risk < 0.10

        # Branch 2: Stack cup on convex ball -> UNSTABLE / FALL
        b_curved = sandbox.fork_branch(graph, "b_curved")
        res_curved = sandbox.simulate_action(
            b_curved, "STACK", {"item_id": "cup_1", "base_id": "ball_1"}
        )
        assert not res_curved.is_success
        assert "unstable_support_fall" in res_curved.violations
        assert res_curved.risk >= 0.80


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: A18-06 Containment & Capacity Reasoning
# ═══════════════════════════════════════════════════════════════════════════


class TestContainmentReasoning:
    """PUT_IN succeeds on open container, rejected when container is closed."""

    def test_put_in_and_closed_container_rejection(self) -> None:
        graph = CognitiveGraph()
        ball = PhysicalEntityNode(id="ball_1", entity_type="ball")
        open_box = PhysicalEntityNode(
            id="box_open", entity_type="box", properties={"is_closed": False}
        )
        closed_box = PhysicalEntityNode(
            id="box_closed", entity_type="box", properties={"is_closed": True}
        )
        graph.add_node(ball)
        graph.add_node(open_box)
        graph.add_node(closed_box)

        sandbox = MentalSandbox()

        # Open box -> success
        b1 = sandbox.fork_branch(graph)
        res_open = sandbox.simulate_action(
            b1, "PUT_IN", {"item_id": "ball_1", "container_id": "box_open"}
        )
        assert res_open.is_success

        # Closed box -> rejected precondition
        b2 = sandbox.fork_branch(graph)
        res_closed = sandbox.simulate_action(
            b2, "PUT_IN", {"item_id": "ball_1", "container_id": "box_closed"}
        )
        assert not res_closed.is_success
        assert "container_closed" in res_closed.violations


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: A18-07 Precondition Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestPreconditionValidation:
    """Action fails gracefully when non-existent entity is targeted."""

    def test_missing_entity_precondition_failure(self) -> None:
        graph = CognitiveGraph()
        sandbox = MentalSandbox()
        branch = sandbox.fork_branch(graph)

        res = sandbox.simulate_action(branch, "PUSH", {"target_id": "ghost_object", "dx": 1.0})
        assert not res.is_success
        assert "entity_not_found" in res.violations


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: A18-08 Derived Consequences & Chain Reactions
# ═══════════════════════════════════════════════════════════════════════════


class TestDerivedConsequences:
    """Unstable placement triggers derived falling event recorded in branch."""

    def test_unstable_curvature_triggers_fall(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup", entity_type="cup")
        sphere = PhysicalEntityNode(
            id="sphere", entity_type="ball", properties={"surface": "convex"}
        )
        graph.add_node(cup)
        graph.add_node(sphere)

        sandbox = MentalSandbox()
        branch = sandbox.fork_branch(graph)
        res = sandbox.simulate_action(branch, "STACK", {"item_id": "cup", "base_id": "sphere"})

        assert not res.is_success
        assert any("fell_off" in c for c in res.consequences)
        assert len(branch.events) == 1
        assert branch.events[0].risk >= 0.80


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: A18-09 Multi-Step Trajectory Rollout
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiStepRollout:
    """Forward trajectory t0 -> t1 -> t2 -> t3 projects composite outcome."""

    def test_multi_step_trajectory_projection(self) -> None:
        graph = CognitiveGraph()
        box = PhysicalEntityNode(id="box_1", entity_type="box", properties={"x": 0.0, "y": 0.0})
        table = PhysicalEntityNode(
            id="table_1", entity_type="table", properties={"geometry": "flat"}
        )
        graph.add_node(box)
        graph.add_node(table)

        sandbox = MentalSandbox()
        trajectory: list[tuple[str, dict[str, Any]]] = [
            ("PUSH", {"target_id": "box_1", "dx": 2.0, "dy": 0.0}),
            ("PUSH", {"target_id": "box_1", "dx": 1.0, "dy": 0.0}),
            ("STACK", {"item_id": "box_1", "base_id": "table_1"}),
        ]

        branch, results = sandbox.simulate_trajectory(graph, trajectory)
        assert len(results) == 3
        assert all(r.is_success for r in results)
        assert branch.depth == 3
        assert branch.get_node("box_1").properties["x"] == 3.0  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: A18-10 Epistemic Uncertainty Decay
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicUncertaintyDecay:
    """Simulation confidence decays smoothly with rollout depth."""

    def test_confidence_decay_over_depth(self) -> None:
        graph = CognitiveGraph()
        box = PhysicalEntityNode(id="b1", entity_type="box", properties={"x": 0.0, "y": 0.0})
        graph.add_node(box)

        sandbox = MentalSandbox()
        b = sandbox.fork_branch(graph)

        c0 = b.confidence
        assert c0 == 1.0

        # Step 1
        sandbox.simulate_action(b, "PUSH", {"target_id": "b1", "dx": 1.0})
        b.depth += 1
        c1 = b.confidence
        assert c1 < c0

        # Step 5
        b.depth = 5
        c5 = b.confidence
        assert c5 < c1


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11: A18-11 Epistemic Source Tagging
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicSourceTagging:
    """Nodes modified in simulation carry branch ID in provenance."""

    def test_simulation_branch_provenance(self) -> None:
        graph = CognitiveGraph()
        box = PhysicalEntityNode(id="b1", entity_type="box", properties={"x": 0.0, "y": 0.0})
        graph.add_node(box)

        sandbox = MentalSandbox()
        branch = sandbox.fork_branch(graph, branch_id="b_provenance_test")
        sandbox.simulate_action(branch, "PUSH", {"target_id": "b1", "dx": 2.0})

        sim_node = branch.get_node("b1")
        assert sim_node is not None
        assert sim_node.provenance.simulation_branch == "b_provenance_test"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12: A18-12 Multi-Branch Counterfactual Search
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiBranchSearch:
    """Evaluates competing rollout branches B1..B4 and selects the optimal path."""

    def test_parallel_branch_evaluation(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup", entity_type="cup")
        box = PhysicalEntityNode(id="box", entity_type="box", properties={"geometry": "flat"})
        ball = PhysicalEntityNode(id="ball", entity_type="ball", properties={"geometry": "convex"})
        graph.add_node(cup)
        graph.add_node(box)
        graph.add_node(ball)

        sandbox = MentalSandbox()

        # Branch 1: Stack on ball (unstable)
        # Branch 2: Stack on box (stable)
        trajectories = [
            [("STACK", {"item_id": "cup", "base_id": "ball"})],
            [("STACK", {"item_id": "cup", "base_id": "box"})],
        ]

        def goal_stacked(b: SimulationBranch) -> bool:
            return len(b.edges_from("cup")) > 0 and b.accumulated_risk < 0.2

        winner, all_res = sandbox.multi_branch_search(
            graph, trajectories, goal_predicate=goal_stacked
        )
        assert winner is not None
        assert winner.branch_id == "plan_b2"
        assert winner.risk_score < 0.10


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 13: A18-13 Risk Pruning & Safety Gating
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskPruning:
    """Rejects high-risk simulation branches even if goal is superficially reached."""

    def test_high_risk_branch_rejection(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup", entity_type="cup")
        ball = PhysicalEntityNode(id="ball", entity_type="ball", properties={"surface": "convex"})
        graph.add_node(cup)
        graph.add_node(ball)

        sandbox = MentalSandbox()
        res = sandbox.evaluate_counterfactual(
            graph,
            [("STACK", {"item_id": "cup", "base_id": "ball"})],
        )

        assert not res.goal_achieved
        assert res.risk_score >= 0.80


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 14: A18-14 Goal Attainment Verification
# ═══════════════════════════════════════════════════════════════════════════


class TestGoalAttainment:
    """Verifies arbitrary goal predicates over simulated world states."""

    def test_goal_predicate_fulfillment(self) -> None:
        graph = CognitiveGraph()
        box = PhysicalEntityNode(id="b1", entity_type="box", properties={"x": 0.0, "y": 0.0})
        graph.add_node(box)

        sandbox = MentalSandbox()

        def goal_reached_x5(b: SimulationBranch) -> bool:
            node = b.get_node("b1")
            return isinstance(node, PhysicalEntityNode) and node.properties.get("x", 0.0) >= 5.0

        res = sandbox.evaluate_counterfactual(
            graph,
            [
                ("PUSH", {"target_id": "b1", "dx": 3.0}),
                ("PUSH", {"target_id": "b1", "dx": 3.0}),
            ],
            goal_predicate=goal_reached_x5,
        )

        assert res.goal_achieved
        assert res.final_predicted_state.active_relations is not None


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 15: A18-15 Canonical Plan Commitment
# ═══════════════════════════════════════════════════════════════════════════


class TestExecutionPlanProduction:
    """Winning trajectory emits an ExecutionPlan; simulation facts stay outside reality."""

    def test_action_plan_produced_not_simulation_facts(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup_1", entity_type="cup", properties={"x": 0.0})
        graph.add_node(cup)

        sandbox = MentalSandbox()
        res = sandbox.evaluate_counterfactual(
            graph,
            [("PUSH", {"target_id": "cup_1", "dx": 4.0})],
        )

        plan = sandbox.produce_execution_plan(res)
        assert isinstance(plan, ExecutionPlan)
        assert plan.validated_actions == [("PUSH", {"target_id": "cup_1", "dx": 4.0})]
        assert plan.status == "READY_FOR_EXECUTION"

        # Canonical reality still at x=0.0 (until physical actuator runs)
        assert graph.get_node("cup_1").properties["x"] == 0.0  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 16: A18-16 The Flagship Obstacle Course Counterfactual
# ═══════════════════════════════════════════════════════════════════════════


class TestObstacleCourseCounterfactual:
    """The Flagship Acceptance Gate: Agent navigates a multi-step obstacle course

    in mental simulation, detects collisions/blocks, rejects unsafe paths,
    selects the winning trajectory, and emits the real-world action plan.
    """

    def test_obstacle_course_navigation_and_rejection(self) -> None:
        graph = CognitiveGraph()
        agent = PhysicalEntityNode(id="agent", entity_type="agent", properties={"x": 0.0, "y": 0.0})
        wall = PhysicalEntityNode(
            id="wall_1",
            entity_type="wall",
            properties={"x": 4.0, "y": 0.0, "width": 2.0, "depth": 6.0, "is_obstacle": True},
        )
        target = PhysicalEntityNode(
            id="target_zone", entity_type="zone", properties={"x": 10.0, "y": 0.0}
        )
        graph.add_node(agent)
        graph.add_node(wall)
        graph.add_node(target)

        sandbox = MentalSandbox()

        # Path 1: Straight through wall -> Collision! (Blocked)
        # Path 2: Flank around wall via (0,0) -> (4, 8) -> (10, 0) -> Success!
        path_blocked = [
            ("MOVE", {"entity_id": "agent", "target_x": 5.0, "target_y": 0.0}),
            ("MOVE", {"entity_id": "agent", "target_x": 10.0, "target_y": 0.0}),
        ]
        path_flank = [
            ("MOVE", {"entity_id": "agent", "target_x": 0.0, "target_y": 8.0}),
            ("MOVE", {"entity_id": "agent", "target_x": 8.0, "target_y": 8.0}),
            ("MOVE", {"entity_id": "agent", "target_x": 10.0, "target_y": 0.0}),
        ]

        def goal_reached_target(b: SimulationBranch) -> bool:
            node = b.get_node("agent")
            if not isinstance(node, PhysicalEntityNode):
                return False
            props = (
                getattr(node, "properties", None) or getattr(node, "observed_properties", {}) or {}
            )
            return props.get("x", 0.0) == 10.0 and props.get("y", 0.0) == 0.0

        winner, all_results = sandbox.multi_branch_search(
            graph,
            [path_blocked, path_flank],
            goal_predicate=goal_reached_target,
        )

        assert winner is not None
        assert winner.branch_id == "plan_b2"  # Flanking path won
        assert winner.risk_score < 0.10
        assert winner.goal_achieved

        # First path was blocked
        assert all_results[0].risk_score >= 0.80
        assert any("path_blocked" in v for v in all_results[0].violations)

        # Produce real-world execution plan
        exec_plan = sandbox.produce_execution_plan(winner)
        assert len(exec_plan.validated_actions) == 3

        # Canonical reality agent is still at start pos (0.0, 0.0)
        assert graph.get_node("agent").properties["x"] == 0.0  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 17: A18-17 Zero-LLM Invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Mental sandbox runs with 100% deterministic code and zero neural/LLM imports."""

    def test_zero_llm_imports(self) -> None:
        import subprocess
        import sys

        check_code = """
import sys
import hbllm.brain.simulation

llm_markers = ["openai", "anthropic", "litellm", "langchain", "transformers"]
loaded = set(sys.modules.keys())
for marker in llm_markers:
    assert marker not in loaded, f"LLM module loaded in simulation runtime: {marker}"
"""
        res = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
        assert res.returncode == 0, f"Zero-LLM verification failed:\n{res.stderr}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 18: A18-18 Simulation/Reality Non-Equivalence
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulationRealityNonEquivalence:
    """Simulation predicts cup falls; canonical reality remains cup on table."""

    def test_simulation_mutation_does_not_affect_reality(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup", entity_type="cup")
        table = PhysicalEntityNode(id="table", entity_type="table")
        ball = PhysicalEntityNode(id="ball", entity_type="ball", properties={"surface": "convex"})
        graph.add_node(cup)
        graph.add_node(table)
        graph.add_node(ball)
        init_edge = HCIREdge(
            edge_type=HCIREdgeType.LOCATED_IN, sources=[cup.id], targets=[table.id]
        )
        graph.add_edge(init_edge)

        sandbox = MentalSandbox()
        # Simulated hypothetical: what if cup was stacked on ball?
        res = sandbox.evaluate_counterfactual(
            graph, [("STACK", {"item_id": "cup", "base_id": "ball"})]
        )

        # Simulation predicts fall
        assert not res.goal_achieved
        assert "unstable_support_fall" in res.violations

        # Canonical reality still has cup LOCATED_ON table!
        canonical_edges = graph.edges_from(cup.id)
        assert len(canonical_edges) == 1
        assert canonical_edges[0].targets == [table.id]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 19: A18-19 Simulation Prediction Error Feedback to A14
# ═══════════════════════════════════════════════════════════════════════════


class TestSimulationPredictionErrorFeedback:
    """Simulation prediction error when actual outcome contradicts simulation routes to A14."""

    def test_simulated_prediction_error_routes_to_a14(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup", entity_type="cup", properties={"x": 0.0})
        graph.add_node(cup)

        sandbox = MentalSandbox()
        # Simulation predicts after PUSH(dx=4.0), x=4.0
        res = sandbox.evaluate_counterfactual(graph, [("PUSH", {"target_id": "cup", "dx": 4.0})])
        predicted_x = 4.0

        # Physical execution occurs, but real-world obstacle or actuator slippage results in actual x=2.0
        actual_x = 2.0
        prediction_error_magnitude = abs(predicted_x - actual_x) / 4.0  # 0.50

        # A14 ErrorClassifier classifies prediction error
        classifier = ErrorClassifier()
        context = ErrorContext(
            error_magnitude=prediction_error_magnitude,
            prediction_confidence=res.confidence,
            historical_error_rate=0.20,
            temporal_pattern="recurring",
            cross_entity_correlation=0.1,
            recency_weighted_frequency=0.25,
            prediction_domain="physics_simulation",
        )
        classification = classifier.classify(context)
        assert classification.model_error > 0.30
