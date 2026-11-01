from __future__ import annotations

from hbllm.hcir.world.cortex_perception import (
    AffordanceRole,
)
from hbllm.hcir.world.prefrontal_working_memory import (
    PBWMWorkingMemoryBuffer,
)

"""Unit tests for Round 5 Epistemic & Cortical Neuro-Symbolic Enhancements.

Track 1: Basal Ganglia Tri-Pathway Action Arbitration (Direct D1 vs. Indirect D2 vs. Hyperdirect STN)
Track 2: Active Inference Expected Free Energy (EFE) with Epistemic vs. Pragmatic Decomposition
Track 3: Prefrontal Working Memory Bistable Attractors & Distractor Filtering (PBWM Gating)
Track 4: Gibsonian Affordance Fields & Closed Topological Boundary Contours
"""

import numpy as np

from hbllm.hcir.world.active_inference_efe import (
    ExpectedFreeEnergyEvaluator,
)
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_deliberation import (
    BasalGangliaGatingCircuit,
    BasalGangliaPathway,
)
from hbllm.hcir.world.cortex_perception import (
    GibsonianAffordanceExtractor,
)
from hbllm.hcir.world.experimental import restore_all as _restore_all


class TestTrack1BasalGangliaTriPathwayArbitration:
    """Track 1: Basal Ganglia D1/D2/STN arbitration."""

    def test_direct_go_pathway_selection(self):
        bg = BasalGangliaGatingCircuit()
        utilities = {1: 0.9, 2: 0.2, 3: 0.1}

        winning, all_dec = bg.arbitrate_candidates(
            candidate_utilities=utilities,
            dopamine_level=1.2,
            acc_conflict=0.1,
            threat_looming=0.0,
        )

        assert winning is not None
        assert winning.action == 1
        assert winning.pathway == BasalGangliaPathway.DIRECT_GO
        assert not winning.is_suppressed
        assert winning.d1_go_drive > winning.d2_nogo_drive
        assert winning.net_drive > 0.0

    def test_indirect_nogo_pathway_suppression(self):
        bg = BasalGangliaGatingCircuit()
        utilities = {1: 0.8, 2: 0.6}
        # Action 1 was lethal / punished in episodic memory
        hab_penalties = {1: 1.0, 2: 0.0}

        winning, all_dec = bg.arbitrate_candidates(
            candidate_utilities=utilities,
            dopamine_level=0.5,  # Low dopamine elevates D2 sensitivity
            acc_conflict=0.1,
            threat_looming=0.0,
            habenular_inhibitions=hab_penalties,
        )

        dec_1 = next(d for d in all_dec if d.action == 1)
        assert dec_1.pathway == BasalGangliaPathway.INDIRECT_NOGO
        assert dec_1.is_suppressed
        assert dec_1.d2_nogo_drive > dec_1.d1_go_drive

        # Action 2 should win as it was not punished
        assert winning is not None
        assert winning.action == 2

    def test_hyperdirect_stn_emergency_brake(self):
        bg = BasalGangliaGatingCircuit(brake_threshold=0.65)
        utilities = {1: 0.9, 2: 0.8}

        # Acute conflict from ACC and looming collision threat
        winning, all_dec = bg.arbitrate_candidates(
            candidate_utilities=utilities,
            dopamine_level=1.0,
            acc_conflict=0.9,
            threat_looming=0.85,
        )

        # STN brake should trigger global halt
        assert all(d.pathway == BasalGangliaPathway.HYPERDIRECT_BRAKE for d in all_dec)
        assert all(d.is_suppressed for d in all_dec)
        assert winning is not None
        assert winning.pathway == BasalGangliaPathway.HYPERDIRECT_BRAKE

    def test_engine_basal_ganglia_integration(self):
        engine = AutonomousEpistemicEngine()

        _restore_all(engine)
        utilities = {1: 0.7, 2: 0.3}

        winning, all_dec = engine.arbitrate_basal_ganglia_action(
            candidate_utilities=utilities,
            acc_conflict=0.2,
            threat_looming=0.0,
        )

        assert winning is not None
        assert winning.action == 1
        assert winning.pathway == BasalGangliaPathway.DIRECT_GO
        assert winning.to_causal_predicate().variable == "basal_ganglia:pathway"
        assert len(winning.to_transaction_operations()) > 0


class TestTrack2ActiveInferenceExpectedFreeEnergy:
    """Track 2: Friston Expected Free Energy policy evaluation."""

    def test_dynamic_beta_schedule(self):
        evaluator = ExpectedFreeEnergyEvaluator(beta_base=1.0)

        # High uncertainty, zero goal confidence -> High curiosity
        beta_explore = evaluator.compute_dynamic_beta(
            world_theory_uncertainty=0.9, goal_confidence=0.0
        )
        # Low uncertainty, high goal confidence -> Low curiosity
        beta_exploit = evaluator.compute_dynamic_beta(
            world_theory_uncertainty=0.1, goal_confidence=0.9
        )

        assert beta_explore > 1.5
        assert beta_exploit < 0.4
        assert beta_explore > beta_exploit * 3.0

    def test_efe_pragmatic_vs_epistemic_decomposition(self):
        evaluator = ExpectedFreeEnergyEvaluator()
        grid = np.zeros((10, 10), dtype=int)
        goals = {(8, 8)}
        visit_counts = {(6, 6): 10, (1, 1): 0}

        # Candidate A: Towards goal (8, 8), but heavily visited
        score_a = evaluator.evaluate_action(
            action=1,
            current_pos=(5, 5),
            target_pos=(6, 6),
            grid=grid,
            goal_positions=goals,
            position_visit_counts=visit_counts,
            goal_confidence=0.8,
            world_theory_uncertainty=0.2,
        )

        # Candidate B: Away from goal, but completely unvisited frontier cell
        score_b = evaluator.evaluate_action(
            action=2,
            current_pos=(5, 5),
            target_pos=(1, 1),
            grid=grid,
            goal_positions=goals,
            position_visit_counts=visit_counts,
            goal_confidence=0.0,
            world_theory_uncertainty=0.9,
        )

        assert score_a.pragmatic_value > score_b.pragmatic_value
        assert score_b.epistemic_value > score_a.epistemic_value
        assert score_a.expected_free_energy == -score_a.total_expected_value
        assert score_b.to_causal_predicate().variable == "active_inference:efe"

    def test_engine_evaluate_policy_expected_free_energy(self):
        engine = AutonomousEpistemicEngine()
        _restore_all(engine)
        grid = np.zeros((10, 10), dtype=int)
        engine.avatar_pos = (2, 2)
        engine.learned_goal_positions.add((8, 8))

        candidates = {
            1: (3, 2),  # South (towards goal)
            2: (1, 2),  # North (away from goal)
        }

        ranked = engine.evaluate_policy_expected_free_energy(
            candidate_transitions=candidates,
            grid=grid,
            goal_confidence=0.9,
        )

        assert len(ranked) == 2
        # Under high goal confidence, pragmatic step towards goal should rank first
        assert ranked[0].action == 1
        assert ranked[0].total_expected_value > ranked[1].total_expected_value

    def test_striatal_gating_distractor_filtering(self):
        buffer = PBWMWorkingMemoryBuffer(gate_threshold=0.5)

        # 1. Distractor input with weak gating signal (< 0.5) should be ignored
        filtered = buffer.update_with_striatal_gate(
            slot_id="subgoal",
            content="distractor_noise",
            striatal_gate_signal=0.3,
        )
        assert not filtered
        assert buffer.get_slot("subgoal") is None

        # 2. Intentional executive signal (>= 0.5) opens the striatal gate
        gated = buffer.update_with_striatal_gate(
            slot_id="subgoal",
            content="collect_red_key",
            striatal_gate_signal=0.8,
            tag="active_subgoal",
        )
        assert gated
        slot = buffer.get_slot("subgoal")
        assert slot is not None
        assert slot.content == "collect_red_key"
        assert slot.is_bistable
        assert slot.activation_energy == 1.0

    def test_bistable_attractor_persistence_and_clearance(self):
        buffer = PBWMWorkingMemoryBuffer(gate_threshold=0.5)
        buffer.update_with_striatal_gate(
            slot_id="active_key",
            content=4,
            striatal_gate_signal=0.9,
        )

        # Step dynamics with sub-threshold visual noise
        buffer.step_attractor_dynamics(distractor_perturbation=0.2)
        slot = buffer.get_slot("active_key")
        assert slot is not None
        assert slot.is_bistable
        assert slot.stability_counter >= 1

        # Explicit clearance upon transactional commit
        cleared = buffer.clear_slot("active_key")
        assert cleared
        assert buffer.get_slot("active_key") is None

    def test_gibsonian_affordance_map_generation(self):
        extractor = GibsonianAffordanceExtractor()
        grid = np.zeros((6, 6), dtype=int)
        grid[1, 1] = 5  # Barrier
        grid[2, 2] = 2  # Lethal hazard
        grid[4, 4] = 3  # Goal

        affordance_map = extractor.extract_affordance_map(
            grid=grid,
            bg_color=0,
            goal_features={3},
            barrier_features={5},
            lethal_features={2},
        )

        assert affordance_map[0, 0] == AffordanceRole.TRAVERSABLE.value
        assert affordance_map[1, 1] == AffordanceRole.STATIC_BARRIER.value
        assert affordance_map[2, 2] == AffordanceRole.LETHAL_HAZARD.value
        assert affordance_map[4, 4] == AffordanceRole.COLLECTIBLE_GOAL.value

    def test_topological_contour_and_cavity_detection(self):
        extractor = GibsonianAffordanceExtractor()
        # Create a hollow enclosed square (donut) with an interior cavity
        grid = np.zeros((7, 7), dtype=int)
        # 5x5 outer ring of feature 4
        for r in range(1, 6):
            for c in range(1, 6):
                if r == 1 or r == 5 or c == 1 or c == 5:
                    grid[r, c] = 4
                else:
                    grid[r, c] = 0  # Interior cavity

        contours = extractor.extract_topological_contours(grid=grid, bg_color=0)
        assert len(contours) == 1
        contour = contours[0]
        assert contour.feature_id == 4
        assert contour.perimeter_length == 16  # 5x5 outer boundary cells
        assert contour.is_enclosed
        assert len(contour.cavity_loops) == 1
        # Cavity should contain the 3x3 interior (9 cells)
        assert len(contour.cavity_loops[0]) == 9
        assert contour.to_causal_predicate().variable == "contour:feature_4"

    def test_engine_affordance_and_contour_integration(self):
        engine = AutonomousEpistemicEngine()
        _restore_all(engine)
        grid = np.zeros((8, 8), dtype=int)
        grid[0, :] = 1  # Top wall
        grid[5, 5] = 8  # Salient object

        affordances = engine.extract_affordance_field(grid, bg_color=0)
        assert affordances.shape == (8, 8)
        assert affordances[4, 4] == AffordanceRole.TRAVERSABLE.value

        contours = engine.extract_topological_contours(grid, bg_color=0)
        assert len(contours) >= 2
        feat_ids = {c.feature_id for c in contours}
        assert 1 in feat_ids
        assert 8 in feat_ids
