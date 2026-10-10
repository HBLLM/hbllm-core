"""Unit tests for human-like cognition primitives: morphology, motor calibration, and blackbox learning."""

from __future__ import annotations

from hbllm.drivers.base import DriverAction, DriverFeedback
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.world.morphology import MorphologicalConcept, ShapeArchetype
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel, StateMutationModel


class TestShapeArchetypeAndMorphology:
    def test_shape_archetype_rotation_invariance(self) -> None:
        coords_1 = {(0, 0), (0, 1), (1, 0)}  # L-tromino
        coords_2 = {(0, 0), (1, 0), (1, 1)}  # Rotated L-tromino

        arch1 = ShapeArchetype.from_coords(coords_1)
        arch2 = ShapeArchetype.from_coords(coords_2)

        assert arch1.area == 3
        assert arch2.area == 3
        assert arch1.is_rotation_of(arch2)
        assert arch1.canonical_id == arch2.canonical_id

    def test_morphological_concept_trial_and_error_learning(self) -> None:
        coords = {(0, 0), (1, 0)}
        arch = ShapeArchetype.from_coords(coords)
        concept = MorphologicalConcept(
            canonical_id=arch.canonical_id,
            archetype=arch,
            canonical_name="domino",
            inferred_role=EntityRole.MANIPULABLE,
        )

        assert not concept.has_affordance("pushable")
        assert concept.get_affordance("pushable") is None

        # Empirical trial-and-error learning
        concept.learn_affordance("pushable", True)
        assert concept.has_affordance("pushable")
        assert concept.get_affordance("pushable") is True

        concept.record_observation("metallic")
        concept.record_observation("blue")
        assert "metallic" in concept.observed_features
        assert "blue" in concept.observed_features

        concept.record_interaction()
        assert concept.interaction_count == 1

    def test_morphological_concept_backward_compat(self) -> None:
        coords = {(0, 0)}
        arch = ShapeArchetype.from_coords(coords)
        concept = MorphologicalConcept(
            canonical_id=arch.canonical_id,
            archetype=arch,
            canonical_name="pixel",
            inferred_role=EntityRole.OBSTACLE,
            observed_colors={3, 4},
            is_rotatable=True,
            rotation_trigger=5,
            is_color_switch=True,
            passable_colors={0},
            barrier_colors={7, 8},
        )

        assert concept.is_rotatable is True
        assert concept.rotation_trigger == 5
        assert concept.is_color_switch is True
        assert 3 in concept.observed_colors
        assert 0 in concept.passable_colors
        assert 7 in concept.barrier_colors

        # Mutate via properties
        concept.is_rotatable = False
        assert concept.is_rotatable is False
        assert concept.learned_affordances["rotatable"] is False

        concept.barrier_colors.add(9)
        assert 9 in concept.barrier_colors
        assert 9 in concept.learned_affordances["obstacle_features"]


class TestMotorCalibrationAndCausalLearning:
    def test_action_dynamics_trial_and_error_learning(self) -> None:
        model = ActionDynamicsModel(action_id=1, delta_r=0, delta_c=0, confidence=0.5)

        # Probing trial: action 1 moves (-1, 0)
        model.update_from_trial(observed_delta=(-1, 0), success=True, learning_rate=1.0)
        assert model.delta_r == -1
        assert model.delta_c == 0
        assert model.confidence > 0.5
        assert model.probes_tested == 1

        # Failed probe decays confidence
        model.update_from_trial(observed_delta=(0, 0), success=False)
        assert model.probes_tested == 2
        assert model.confidence < 0.6

    def test_state_mutation_causal_discovery(self) -> None:
        rule = StateMutationModel(
            trigger_type="CONTACT",
            trigger_pos=(3, 3),
            trigger_feature="red_switch",
            mutation_type="GATE_OPEN",
            prior_value="closed",
            posterior_value="open",
            confidence=0.5,
        )

        assert rule.trigger_color == "red_switch"
        assert rule.mutation_type == "GATE_OPEN"

        # Observation corroborates rule
        rule.record_observation("open")
        assert rule.occurrences == 2
        assert rule.confidence > 0.5

        # Inconsistent observation weakens confidence
        rule.record_observation("still_closed")
        assert rule.occurrences == 3
        assert rule.posterior_value == "still_closed"

    def test_conditional_action_dynamics_branching(self) -> None:
        """Verify that condition-tagged updates maintain separate branches and prevent EMA corruption."""
        model = ActionDynamicsModel(action_id=1, delta_r=0, delta_c=0, confidence=0.5)

        # Baseline empty-handed movement: action 1 moves (-1, 0)
        model.update_from_trial(
            observed_delta=(-1, 0), success=True, learning_rate=1.0, condition="default"
        )
        assert model.delta_r == -1
        assert model.delta_c == 0

        # Movement when carrying heavy item: action 1 moves (0, 0) or different direction
        model.update_from_trial(observed_delta=(0, 0), success=True, condition="carrying")
        assert model.delta_r == -1  # Default branch unaffected!
        assert model.delta_c == 0

        # Carrying branch has (0, 0)
        carrying_dyn = model.get_dynamics("carrying")
        assert carrying_dyn.delta_r == 0
        assert carrying_dyn.delta_c == 0
        assert model.get_displacement("carrying") == (0, 0)
        assert model.get_displacement("default") == (-1, 0)

    def test_multimodal_contradiction_forking(self) -> None:
        """Verify that high-confidence model encountering conflicting observations forks rather than corrupts EMA."""
        model = ActionDynamicsModel(
            action_id=2, delta_r=1, delta_c=0, confidence=0.9, probes_tested=5
        )

        # Contradictory observation without explicit condition (e.g. mode changed in environment)
        model.update_from_trial(observed_delta=(0, 1), success=True)

        # Baseline is preserved
        assert model.delta_r == 1
        assert model.delta_c == 0

        # An alternative branch was forked
        assert "cond_0_1" in model.conditional_branches
        branch = model.conditional_branches["cond_0_1"]
        assert branch.delta_r == 0
        assert branch.delta_c == 1

    def test_conditioned_action_dynamics_container(self) -> None:
        """Verify ConditionedActionDynamics dict-like access and fallback."""
        from hbllm.hcir.world.motor_calibration import ConditionedActionDynamics

        cad = ConditionedActionDynamics()
        m_base = ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=0.9)
        m_carry = ActionDynamicsModel(action_id=1, delta_r=0, delta_c=0, confidence=0.8)

        cad.set(1, None, m_base)
        cad.set(1, "carrying", m_carry)

        assert cad.get(1) == m_base
        assert cad.get(1, "carrying") == m_carry
        assert cad.get(1, "unknown_mode") == m_base  # Fallback to implicit default

        cad.set_active_condition("carrying")
        assert cad[1] == m_carry
        cad.set_active_condition(None)
        assert cad[1] == m_base


class TestCognitiveBlackboxLearning:
    def test_blackbox_learns_from_feedback_trial_and_error(self) -> None:
        blackbox = CognitiveBlackbox()
        action = DriverAction(action_id=2)

        # Feedback includes observed displacement and traversed feature
        feedback = DriverFeedback(
            success=True,
            reward=0.1,
            terminated=False,
            info={
                "observed_delta": (1, 0),
                "traversed_feature": 0,
            },
        )

        blackbox.update(action, feedback)
        state = blackbox.get_state()

        assert 2 in state.action_models
        assert state.action_models[2].delta_r == 1
        assert state.action_models[2].delta_c == 0
        assert 0 in state.learned_traversable_features

        # Collision with obstacle
        col_feedback = DriverFeedback(
            success=False,
            reward=-0.1,
            info={
                "collision_feature": 8,
            },
        )
        blackbox.update(action, col_feedback)
        assert 8 in state.learned_obstacle_features

    def test_blackbox_memory_retention(self) -> None:
        blackbox = CognitiveBlackbox()
        state = blackbox.get_state()
        state.learned_obstacle_features.add("wall")
        state.learned_target_features.add("gem")

        # Sync to workspace
        blackbox.sync_state_to_workspace()
        assert blackbox.workspace.graph.get_node("var_obstacle_features") is not None
        assert blackbox.workspace.graph.get_node("var_target_features") is not None

        # Reset with retain_memory=True
        blackbox.reset(retain_memory=True)
        new_state = blackbox.get_state()
        assert "wall" in new_state.learned_obstacle_features
        assert "gem" in new_state.learned_target_features

        # Reset with retain_memory=False
        blackbox.reset(retain_memory=False)
        wiped_state = blackbox.get_state()
        assert len(wiped_state.learned_obstacle_features) == 0
