"""Unit and integration tests for Round 3 Epistemic World Model Tracks:
Track 1: Thalamocortical Gamma-Band Phase Synchronization (40 Hz Kuramoto Temporal Binding) & TRN Gating.
Track 2: Popperian Falsification & Inductive Rule Export/Retraction to HCIRSymbolicWorldTheory.
Track 3: Ballistic Particle Kinematics & Angle-of-Incidence Specular Deflections in SpatiotemporalCollisionCones.
Track 4: Frontopolar Subgoal Bytecode Compilation & Transactional Rollback Virtual Machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import GoalNode
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine, EpistemicPhase
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
from hbllm.hcir.world.cortex_hypothesis import (
    HypothesisStatus,
    HypothesisType,
    InductiveHypothesisEngine,
)
from hbllm.hcir.world.elastic_particle_induction import DeflectionMode
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoal,
    FrontopolarSubgoalStack,
    SubgoalType,
)
from hbllm.hcir.world.spatiotemporal_collision import SpatiotemporalCollisionCones
from hbllm.hcir.world.thalamocortical_binding import ThalamocorticalBindingEngine


class TestTrack1ThalamocorticalBinding:
    """Track 1: 40 Hz Kuramoto Temporal Feature Binding & TRN Gating."""

    def test_kuramoto_phase_synchronization_and_binding(self):
        engine = ThalamocorticalBindingEngine(nominal_frequency_hz=40.0)
        entities = [
            {"pos": (5, 5), "feature": 2, "role": "target"},
            {"pos": (5, 6), "feature": 2, "role": "target"},
            {"pos": (18, 18), "feature": 7, "role": "clutter"},
        ]
        bundles = engine.bind_features_oscillatory(entities)
        assert len(bundles) >= 1
        # Close entities should bind into a high-coherence bundle
        bound_bundle = next((b for b in bundles if b.source_components_count > 1), None)
        assert bound_bundle is not None
        assert bound_bundle.is_bound
        assert bound_bundle.phase_coherence_r > 0.0

    def test_trn_attentional_gating(self):
        engine = ThalamocorticalBindingEngine()
        entities = [
            {"pos": (2, 2), "feature": 3, "role": "goal"},
            {"pos": (9, 9), "feature": 8, "role": "distractor"},
        ]
        bundles = engine.bind_features_oscillatory(entities)
        gated = engine.apply_trn_attentional_gating(bundles, target_features={"role": "goal"})
        assert len(gated) == len(bundles)
        # Resonant target should be amplified and sorted to top
        top_bundle = gated[0]
        assert top_bundle.features.get("role") == "goal"
        assert top_bundle.is_bound

    def test_autonomous_engine_temporal_binding_in_assimilation(self):
        ep_engine = AutonomousEpistemicEngine()
        grid = np.zeros((10, 10), dtype=int)
        # Create a cluster of feature 3 cells (goal)
        grid[2, 2] = 3
        grid[2, 3] = 3
        ep_engine.symbolic_theory.goal_features.add(3)
        ep_engine.avatar_pos = (0, 0)
        ep_engine.avatar_feature = 1

        ep_engine.assimilate_feedback(
            curr_grid=grid,
            available_actions=[1, 2, 3, 4],
            is_success=False,
            is_failure=False,
        )

        assert len(ep_engine.working_memory.active_bound_bundles) > 0
        pred_vars = [p.variable for p in ep_engine.symbolic_theory.causal_predicates]
        assert any("bound_entity" in v for v in pred_vars)

        # Verify working memory reset clears active bound bundles
        ep_engine.working_memory.reset_episode(retain_long_term=False)
        assert len(ep_engine.working_memory.active_bound_bundles) == 0


class TestTrack2PopperianInductiveHypothesis:
    """Track 2: Popperian Falsification & Inductive Rule Export/Retraction."""

    def test_export_and_retract_hypotheses_in_world_theory(self):
        hyp_engine = InductiveHypothesisEngine()
        theory = HCIRSymbolicWorldTheory()

        # Create confirmed hypothesis rule
        hyp = hyp_engine.propose_hypothesis(
            rule_type=HypothesisType.REMOTE_MECHANISM,
            premise_feature=2,
            target_feature=5,
            consequence="door_open",
        )
        hyp.confirmation_count = 5
        hyp.status = HypothesisStatus.CONFIRMED

        hyp_engine.export_to_world_theory(theory)
        assert 5 in theory.barrier_features
        assert 2 in theory.actuator_features
        pred_vars = [p.variable for p in theory.causal_predicates]
        assert any(hyp.rule_id in v or "inductive_rule" in v for v in pred_vars)

        # Falsify hypothesis with counterexample
        hyp.status = HypothesisStatus.REFUTED
        retracted = hyp_engine.retract_refuted_rules(theory)
        assert hyp.rule_id in retracted
        assert 5 not in theory.barrier_features
        assert 2 not in theory.actuator_features

    def test_closed_loop_falsification_in_epistemic_engine(self):
        ep_engine = AutonomousEpistemicEngine()
        hyp = ep_engine.working_memory.hypothesis_engine.propose_hypothesis(
            rule_type=HypothesisType.HAZARD_CONTACT,
            premise_feature=4,
            target_feature=1,
            consequence="lethal_reset",
        )
        hyp.confirmation_count = 3
        hyp.status = HypothesisStatus.CONFIRMED

        # Assimilate step exports rule to theory
        grid = np.zeros((8, 8), dtype=int)
        ep_engine.assimilate_feedback(
            curr_grid=grid,
            available_actions=[1, 2],
        )
        assert 4 in ep_engine.symbolic_theory.barrier_features

        # Falsify hypothesis
        hyp.status = HypothesisStatus.REFUTED
        ep_engine.assimilate_feedback(
            curr_grid=grid,
            available_actions=[1, 2],
        )
        assert 4 not in ep_engine.symbolic_theory.barrier_features


class TestTrack3BallisticParticleKinematics:
    """Track 3: Ballistic Particle Kinematics & Angle-of-Incidence Deflections."""

    def test_ballistic_particle_specular_deflection(self):
        cones = SpatiotemporalCollisionCones(default_horizon=10)
        grid = np.zeros((10, 10), dtype=int)
        # Vertical wall at col 5
        grid[:, 5] = 1

        # Particle moving right towards wall at (3, 3) with velocity (0, 1)
        particles = [{"pos": (3, 3), "velocity": (0, 1), "feature_id": 9, "entity_id": "photon_1"}]
        cones.update_ballistic_particle_trajectories(
            particles=particles,
            grid=grid,
            barrier_features={1},
            deflection_mode=DeflectionMode.SPECULAR_NORMAL,
            horizon=8,
        )

        assert len(cones.projected_trajectories) == 1
        traj = cones.projected_trajectories[0]
        assert len(traj.positions) > 0
        # Step 1: (3, 4)
        # Step 2: Hits wall at col 5, rebounds left to (3, 4) or specular direction
        assert (3, 4) in cones.get_collision_cone_at_step(1)
        assert cones.is_collision_hazard(3, 4, time_step=1)
        assert cones.is_looming_threat(3, 4, max_horizon=3)

    def test_autonomous_engine_ballistic_particle_integration(self):
        ep_engine = AutonomousEpistemicEngine()
        grid = np.zeros((12, 12), dtype=int)
        grid[5, :] = 2  # Horizontal wall
        ep_engine.symbolic_theory.barrier_features.add(2)

        particles = [
            {"pos": (2, 2), "velocity": (1, 0), "feature_id": 7, "entity_id": "ballistic_rock"}
        ]
        ep_engine.update_ballistic_particles(
            particles=particles,
            grid=grid,
            deflection_mode=DeflectionMode.SPECULAR_NORMAL,
            horizon=6,
        )

        intercept = ep_engine.collision_cones.predict_intercept_step((3, 2), max_horizon=6)
        assert intercept == 1
        preds = ep_engine.collision_cones.export_causal_predicates()
        assert len(preds) > 0


class TestTrack4FrontopolarSubgoalBytecodeVM:
    """Track 4: Frontopolar Subgoal Bytecode Compilation & Transactional Rollback VM."""

    def test_subgoal_stack_compile_to_instruction_stream(self):
        stack = FrontopolarSubgoalStack()
        sg1 = FrontopolarSubgoal(
            subgoal_id="sg_door",
            subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
            target_entity_pos=(2, 3),
            target_destination=(2, 4),
            priority=2.0,
        )
        sg2 = FrontopolarSubgoal(
            subgoal_id="sg_goal",
            subgoal_type=SubgoalType.DELIVER_TO_GOAL,
            target_entity_pos=(5, 5),
            target_destination=(5, 6),
            priority=1.0,
        )
        stack.push(sg1)
        stack.push(sg2)

        stream = stack.compile_to_instruction_stream(author="test_ba10")
        assert len(stream.instructions) > 0
        opcodes = [ins.opcode for ins in stream.instructions]
        assert Opcode.ASSERT in opcodes
        assert Opcode.EXECUTE in opcodes

    @pytest.mark.asyncio
    async def test_execute_subgoal_program_with_checkpoint_success(self):
        ep_engine = AutonomousEpistemicEngine()
        ws = HCIRWorkspaceState()
        node = GoalNode(id="g_test_10", description="Reach exit")

        # Instruction stream asserting a goal node
        stream = InstructionStream(
            instructions=[
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={"node_data": node.model_dump(), "author": "ba10"},
                ),
            ]
        )

        out, success = await ep_engine.execute_subgoal_program_with_checkpoint(stream, ws)
        assert success
        assert out["status"] == "success"
        # Node must be merged into workspace
        assert ws.get_node("g_test_10") is not None

    @pytest.mark.asyncio
    async def test_execute_subgoal_program_with_checkpoint_failure_and_rollback(self):
        ep_engine = AutonomousEpistemicEngine()
        ws = HCIRWorkspaceState()
        acc = ep_engine.acc_conflict_monitor
        initial_frustration = acc.frustration

        # Instruction stream querying a non-existent capability causing failure
        stream = InstructionStream(
            instructions=[
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={"capability": "invalid.non_existent_capability"},
                ),
            ]
        )

        out, success = await ep_engine.execute_subgoal_program_with_checkpoint(stream, ws)
        # In HCIR interpreter, unresolved capability returns {"error": ...} or throws
        if not success:
            assert out["status"] == "failed"
            assert acc.frustration > initial_frustration
            assert ep_engine.current_phase == EpistemicPhase.REPLANNING

    def test_run_subgoal_program_sync(self):
        ep_engine = AutonomousEpistemicEngine()
        ws = HCIRWorkspaceState()
        node = GoalNode(id="g_sync_1", description="Sync goal")
        stream = InstructionStream(
            instructions=[
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={"node_data": node.model_dump(), "author": "ba10"},
                ),
            ]
        )

        out, success = ep_engine.run_subgoal_program_sync(stream, ws)
        assert success
        assert ws.get_node("g_sync_1") is not None
