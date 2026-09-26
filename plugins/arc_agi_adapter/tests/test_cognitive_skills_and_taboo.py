from hbllm.drivers.base import DriverAction
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox, StateMutationModel
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.subgoal_decomposer import HCIRSkill


def test_mutation_harvests_procedural_sub_skill() -> None:
    blackbox = CognitiveBlackbox()
    state = blackbox.get_state("test")

    # Simulate recent actions leading to a mutation
    state.recent_action_history = [1, 1, 2, 3]
    state.recent_action_data_history = [None, None, None, None]

    mutation = StateMutationModel(
        trigger_type="CONTACT",
        trigger_pos=(4, 5),
        trigger_feature=7,
        mutation_type="COLOR_REMAP",
        prior_value=2,
        posterior_value=8,
        confidence=0.95,
    )
    blackbox.record_state_mutation(mutation, source_id="test")

    assert len(state.learned_skills) == 1
    skill = next(iter(state.learned_skills.values()))
    assert skill.action_sequence == [1, 1, 2, 3]
    assert skill.expected_effect["mutation_type"] == "COLOR_REMAP"
    assert skill.expected_effect["posterior_value"] == 8


def test_reset_preserves_skills_and_accumulates_failed_trajectories() -> None:
    blackbox = CognitiveBlackbox()
    state = blackbox.get_state("test")

    state.learned_skills["skill_1"] = HCIRSkill(
        skill_id="skill_1",
        action_sequence=[1, 2],
    )
    state.recent_action_history = [1, 2, 3, 4]
    state.last_attempt_won = False

    # Retry reset: should record [1, 2, 3, 4] in failed_trajectories and keep skill_1
    blackbox.reset(source_id="test", retain_memory=True, is_retry=True)

    new_state = blackbox.get_state("test")
    assert "skill_1" in new_state.learned_skills
    assert [1, 2, 3, 4] in new_state.failed_trajectories
    assert len(new_state.recent_action_history) == 0


def test_active_skill_queue_execution() -> None:
    blackbox = CognitiveBlackbox()
    state = blackbox.get_state("test")

    state.active_skill_queue = [
        DriverAction(action_id=3),
        DriverAction(action_id=4),
    ]

    available = [DriverAction(action_id=1), DriverAction(action_id=2), DriverAction(action_id=3)]
    chosen = blackbox.decide(available, source_id="test")

    assert chosen.action_id == 3
    assert len(state.active_skill_queue) == 1
    assert state.active_skill_queue[0].action_id == 4


def test_taboo_path_branching_away_from_failed_trajectory() -> None:
    blackbox = CognitiveBlackbox()
    state = blackbox.get_state("test")

    # Suppose a previous attempt failed with trajectory [1, 1, 2]
    state.failed_trajectories = [[1, 1, 2]]
    # Currently, agent has executed [1]
    state.recent_action_history = [1]

    # Available actions are 1 (replays [1, 1] which matches failed prefix) and 2 (branches)
    available = [
        DriverAction(action_id=1, semantic_intent=SpatialActionIntent.NAVIGATE),
        DriverAction(action_id=2, semantic_intent=SpatialActionIntent.NAVIGATE),
    ]

    # Mock action_models so spatial navigation doesn't think 2 is blocked
    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

    state.action_models[1] = ActionDynamicsModel(action_id=1, delta_r=-1, delta_c=0, confidence=1.0)
    state.action_models[2] = ActionDynamicsModel(action_id=2, delta_r=1, delta_c=0, confidence=1.0)

    # Spatial plan wants to do step that would pick action 1:
    # If the chosen action would be 1, taboo branching intercepts and picks 2
    from hbllm.hcir.spatial_planner import SequencePlanStep

    state.current_plan = [
        SequencePlanStep(
            target_entity_id="goal", target_pos=(0, 5), action_type=SpatialActionIntent.NAVIGATE
        )
    ]
    from hbllm.hcir.spatial_planner import EntityGraph, EntityRole, SpatialEntity

    eg = EntityGraph(grid_shape=(10, 10))
    eg.avatar = SpatialEntity(
        id="av",
        role=EntityRole.AGENT,
        grid_pos=(1, 5),
        centroid=(1.0, 5.0),
        area=1,
        bounding_box=(1, 1, 5, 5),
    )
    blackbox._source_entity_graphs["test"] = eg

    chosen = blackbox.decide(available, source_id="test", entity_graph=eg)
    # The taboo branching must avoid replaying action 1
    assert chosen.action_id == 2
