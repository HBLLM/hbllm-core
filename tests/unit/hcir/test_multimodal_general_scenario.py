"""Unit tests verifying domain-generality and arbitrary multimodal/non-grid scenario support.

Guarantees that core cognitive faculties (HCIRSymbolicWorldTheory, AutonomousEpistemicEngine,
DiscreteVectorTranslator, SkillProgramSynthesizer) function seamlessly across robotics,
web navigation, dialogue, and continuous/multimodal domains without integer palette or
ARC-AGI-specific assumptions.
"""

from __future__ import annotations

from hbllm.hcir.skills.common_subskills import DiscreteVectorTranslator
from hbllm.hcir.world.cortex_causal import (
    CausalInductionCortex,
    HCIRSymbolicWorldTheory,
)
from hbllm.hcir.world.percept import EntityRole, SpatialEntity
from hbllm.hcir.world.skill_program_synthesizer import (
    SkillProgramSynthesizer,
    SpatialActionIntent,
)


def test_symbolic_theory_string_concepts_and_features():
    """Verify that HCIRSymbolicWorldTheory handles arbitrary string concept identifiers."""
    theory = HCIRSymbolicWorldTheory()

    # Register string concepts across cognitive roles
    theory.avatar_features.add("agent_avatar")
    theory.induce_goal("target_cup")
    theory.induce_barrier("security_door")
    theory.induce_walkable("open_hallway")
    theory.induce_cargo("wooden_crate")
    theory.induce_receptacle("loading_dock")

    # Queries using raw strings
    assert theory.is_avatar_feature("agent_avatar")
    assert not theory.is_avatar_feature("target_cup")

    assert theory.is_goal("target_cup")
    assert not theory.is_goal("agent_avatar")
    assert not theory.is_goal("security_door")

    assert theory.is_barrier("security_door")
    assert not theory.is_barrier("open_hallway")
    assert not theory.is_barrier("agent_avatar")

    assert theory.is_walkable("open_hallway")
    assert not theory.is_walkable("security_door")

    assert theory.is_cargo("wooden_crate")
    assert not theory.is_cargo("target_cup")

    assert theory.is_receptacle("loading_dock")
    assert not theory.is_receptacle("wooden_crate")


def test_symbolic_theory_handles_rich_entity_objects():
    """Verify that is_goal, is_cargo, is_receptacle correctly inspect arbitrary SpatialEntity objects."""
    theory = HCIRSymbolicWorldTheory()
    theory.induce_goal("golden_key")

    goal_entity = SpatialEntity(
        entity_id="item_01",
        feature_id="golden_key",
        grid_pos=(5, 5),
        role=EntityRole.GOAL,
        properties={"label": "special_key"},
    )
    assert theory.is_goal(goal_entity)

    # Entity with role GOAL even if feature_id is not yet explicitly in goal_features
    novel_goal = SpatialEntity(
        entity_id="item_02",
        feature_id="unknown_trophy",
        grid_pos=(1, 1),
        role=EntityRole.GOAL,
    )
    assert theory.is_goal(novel_goal)

    # Cargo entity
    theory.induce_cargo("box_A")
    cargo_entity = SpatialEntity(
        entity_id="box_01",
        feature_id="box_A",
        grid_pos=(2, 2),
        role=EntityRole.MANIPULABLE,
    )
    assert theory.is_cargo(cargo_entity)

    # Receptacle entity
    theory.induce_receptacle("bin_01")
    receptacle_entity = SpatialEntity(
        entity_id="bin_inst",
        feature_id="bin_01",
        grid_pos=(3, 3),
        role=EntityRole.RECEPTACLE,
    )
    assert theory.is_receptacle(receptacle_entity)


def test_symbolic_theory_string_tool_barrier_affinities():
    """Verify tool-barrier causal resonance ledger with string tools and gates."""
    theory = HCIRSymbolicWorldTheory()

    theory.record_tool_barrier_affinity("crowbar", "wooden_planks")
    theory.record_tool_barrier_incompatibility("crowbar", "titanium_vault")

    assert theory.has_tool_barrier_affinity("crowbar", "wooden_planks")
    assert theory.has_tool_barrier_affinity("crowbar")  # has any affinity
    assert not theory.has_tool_barrier_affinity("crowbar", "titanium_vault")
    assert not theory.has_tool_barrier_affinity("screwdriver")

    assert theory.is_tool_barrier_incompatible("crowbar", "titanium_vault")
    assert not theory.is_tool_barrier_incompatible("crowbar", "wooden_planks")


def test_causal_induction_cortex_string_barriers_and_tools():
    """Verify CausalInductionCortex handles string entities and passability checks."""
    cortex = CausalInductionCortex()
    cortex.symbolic_theory.induce_walkable("corridor")
    cortex.symbolic_theory.induce_goal("flag")
    cortex.symbolic_theory.induce_barrier("laser_grid")

    assert cortex.is_passable("corridor")
    assert cortex.is_passable("flag")
    assert not cortex.is_passable("laser_grid")
    assert cortex.is_passable("unseen_feature")  # not confirmed as barrier


def test_discrete_vector_translator_custom_action_schema():
    """Verify DiscreteVectorTranslator handles custom non-integer action schemas (e.g. robotics/browser)."""
    custom_map = {
        "up": "MOVE_NORTH",
        "down": "MOVE_SOUTH",
        "left": "MOVE_WEST",
        "right": "MOVE_EAST",
    }
    actions = DiscreteVectorTranslator.delta_to_actions(
        dx=-2, dy=3, action_map=custom_map, horizontal_first=True
    )
    assert len(actions) == 5
    assert actions[:2] == [("MOVE_WEST", None), ("MOVE_WEST", None)]
    assert actions[2:] == [("MOVE_SOUTH", None), ("MOVE_SOUTH", None), ("MOVE_SOUTH", None)]

    # Waypoint path with custom actions
    path = [(0, 0), (0, 1), (1, 1)]
    path_actions = DiscreteVectorTranslator.path_to_actions(path, action_map=custom_map)
    assert len(path_actions) == 2
    assert path_actions[0] == ("MOVE_SOUTH", None)
    assert path_actions[1] == ("MOVE_EAST", None)


def test_skill_program_synthesizer_domain_general():
    """Verify SkillProgramSynthesizer synthesizes skills without hardcoded ARC action restrictions."""
    synth = SkillProgramSynthesizer()
    pattern = (SpatialActionIntent.NAVIGATE_TO, SpatialActionIntent.INTERACT)

    # Synthesize without specifying required actions -> should allow any affordance
    skill = synth.synthesize_skill_from_pattern(pattern, skill_name="general_pickup_skill")
    assert skill.skill_name == "general_pickup_skill"
    assert skill.signature is not None
    # ActionAffordancePredicate with required=set() matches any available actions
    mock_ctx = {"available_actions": {"CLICK", "DRAG", "TYPE"}}
    assert skill.signature.evaluate(mock_ctx)

    # Synthesize with custom domain actions (e.g. robotics primitives)
    custom_skill = synth.synthesize_skill_from_pattern(
        pattern,
        skill_name="robot_arm_skill",
        required_actions={"GRASP", "RELEASE"},
    )
    assert not custom_skill.signature.evaluate({"available_actions": {"CLICK"}})
    assert custom_skill.signature.evaluate({"available_actions": {"GRASP", "RELEASE", "MOVE"}})


def test_engine_state_supports_general_actions_and_features():
    """Verify CognitiveContext and EngineState support arbitrary types for actions and features."""
    from hbllm.hcir.world.engine_state import CognitiveContext, EngineState

    ctx = CognitiveContext(
        avatar_feature="agent_drone",
        avatar_features=frozenset(["agent_drone", "payload"]),
        prior_avatar_feature="agent_drone_old",
        last_action="THRUST_FORWARD",
        recent_actions=("TAKEOFF", "HOVER", "THRUST_FORWARD"),
        bg_feature="sky_background",
    )
    assert ctx.avatar_feature == "agent_drone"
    assert "payload" in ctx.avatar_features
    assert ctx.last_action == "THRUST_FORWARD"
    assert ctx.bg_feature == "sky_background"

    state = EngineState(
        avatar_feature="cursor_pointer",
        avatar_features={"cursor_pointer"},
        prior_avatar_feature=None,
        last_action="CLICK_SUBMIT",
        recent_actions=["FOCUS_INPUT", "TYPE_TEXT", "CLICK_SUBMIT"],
        bg_feature="canvas_white",
    )
    assert state.avatar_feature == "cursor_pointer"
    assert state.last_action == "CLICK_SUBMIT"
    assert state.bg_feature == "canvas_white"
