"""Unit tests for CognitiveBlackbox KnowledgeGraph integration and persistence."""

import tempfile
from pathlib import Path

from hbllm.drivers.base import DriverAction, DriverFeedback
from hbllm.drivers.cognitive_blackbox import AgentPhase, AgentState, CognitiveBlackbox


def test_agent_state_serialization_roundtrip():
    """Verify that AgentState to_dict and from_dict preserve all properties."""
    state = AgentState(
        phase=AgentPhase.EXPLOITATION,
        step_count=42,
        total_reward=10.5,
        avatar_feature=14,
        step_size=4,
        learned_obstacle_features={3, 5},
        learned_target_features={9},
        learned_traversable_features={0},
        explored_entity_positions={(10, 10), (20, 20)},
        explored_entity_ids={"ent_1", "ent_2"},
        delivered_positions={(30, 30)},
    )

    data = state.to_dict()
    restored = AgentState.from_dict(data)

    assert restored.phase == AgentPhase.EXPLOITATION
    assert restored.step_count == 42
    assert restored.total_reward == 10.5
    assert restored.avatar_feature == 14
    assert restored.step_size == 4
    assert restored.learned_obstacle_features == {3, 5}
    assert restored.learned_target_features == {9}
    assert restored.learned_traversable_features == {0}
    assert restored.explored_entity_positions == {(10, 10), (20, 20)}
    assert restored.explored_entity_ids == {"ent_1", "ent_2"}
    assert restored.delivered_positions == {(30, 30)}


def test_cognitive_blackbox_knowledge_graph_projection():
    """Verify that observe and update populate the core KnowledgeGraph."""
    blackbox = CognitiveBlackbox()

    # 1. Update with motor action outcome
    act = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={
            "observed_delta": [-4, 0],
            "collision_feature": 3,
            "traversed_feature": 0,
        },
    )
    blackbox.get_state("test_env").avatar_feature = 14
    blackbox.update(act, feedback, source_id="test_env")

    kg = blackbox.get_knowledge_graph("test_env")
    assert kg.entity_count > 0
    assert kg.relation_count > 0

    # Verify action entity
    act_ent = kg.get_entity("action_1")
    assert act_ent is not None
    assert act_ent.entity_type == "motor_action"
    assert act_ent.attributes["delta_r"] == -4

    # Verify obstacle relation
    feat_ent = kg.get_entity("feat_3")
    assert feat_ent is not None
    assert feat_ent.attributes["feature_id"] == 3

    # Verify avatar relation
    av_ent = kg.get_entity("feat_14")
    assert av_ent is not None


def test_cognitive_blackbox_save_and_load_knowledge():
    """Verify that save_knowledge and load_knowledge preserve world models."""
    blackbox1 = CognitiveBlackbox()
    state1 = blackbox1.get_state("game_x")
    state1.avatar_feature = 7
    state1.step_size = 3
    state1.learned_obstacle_features.add(2)
    state1.learned_target_features.add(9)
    state1.learned_traversable_features.add(0)

    # Record motor feedback
    act = DriverAction(action_id=2)
    feedback = DriverFeedback(
        success=True,
        reward=1.0,
        terminated=True,
        info={"observed_delta": [3, 0], "reached_feature": 9},
    )
    blackbox1.update(act, feedback, source_id="game_x")

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        saved_file = blackbox1.save_knowledge(p, source_id="game_x")
        assert saved_file.exists()

        # Load into fresh blackbox
        blackbox2 = CognitiveBlackbox()
        loaded = blackbox2.load_knowledge(p, source_id="game_x")
        assert loaded is True

        state2 = blackbox2.get_state("game_x")
        assert state2.avatar_feature == 7
        assert 2 in state2.learned_obstacle_features
        assert 9 in state2.learned_target_features
        assert 0 in state2.learned_traversable_features
        assert 2 in state2.action_models
        assert state2.action_models[2].delta_r == 3
        assert state2.step_size == 3


def test_agent_state_state_mutations_serialization():
    """Verify that AgentState to_dict and from_dict preserve state_mutations and active_condition."""
    from hbllm.hcir.world.motor_calibration import StateMutationModel

    mut = StateMutationModel(
        trigger_type="CONTACT",
        trigger_pos=(4, 5),
        trigger_feature=3,
        mutation_type="COLOR_REMAP",
        prior_value=3,
        posterior_value=7,
        confidence=0.95,
        occurrences=2,
    )
    state = AgentState(
        avatar_feature=7,
        state_mutations=[mut],
        active_condition="carrying",
    )
    data = state.to_dict()
    restored = AgentState.from_dict(data)

    assert len(restored.state_mutations) == 1
    rmut = restored.state_mutations[0]
    assert rmut.trigger_type == "CONTACT"
    assert rmut.trigger_pos == (4, 5)
    assert rmut.mutation_type == "COLOR_REMAP"
    assert rmut.prior_value == 3
    assert rmut.posterior_value == 7
    assert rmut.confidence == 0.95
    assert restored.active_condition == "carrying"
    assert restored.get_active_condition() == "carrying"


def test_cognitive_blackbox_causal_induction():
    """Verify that update() induces StateMutationModels on color remap, barrier opening, and carrying change."""
    blackbox = CognitiveBlackbox()
    state = blackbox.get_state("env_mut")
    state.avatar_feature = 3

    # 1. Color remap upon stepping on tile
    act = DriverAction(action_id=1)
    fb1 = DriverFeedback(
        success=True,
        reward=0.0,
        terminated=False,
        info={
            "observed_delta": [-1, 0],
            "traversed_feature": 6,
            "new_avatar_feature": 7,
        },
    )
    blackbox.update(act, fb1, source_id="env_mut")
    assert state.avatar_feature == 7
    assert len(state.state_mutations) == 1
    m1 = state.state_mutations[0]
    assert m1.mutation_type == "COLOR_REMAP"
    assert m1.prior_value == 3
    assert m1.posterior_value == 7

    # 2. Barrier opened by switch interaction
    act_interact = DriverAction(action_id=5)
    fb2 = DriverFeedback(
        success=True,
        reward=0.5,
        terminated=False,
        info={"barrier_opened": [(2, 2), (2, 3)]},
    )
    blackbox.update(act_interact, fb2, source_id="env_mut")
    assert len(state.state_mutations) == 2
    m2 = state.state_mutations[1]
    assert m2.mutation_type == "BARRIER_OPEN"
    assert m2.posterior_value == "open"

    # 3. Carrying payload pickup
    state.carrying.holding = True
    fb3 = DriverFeedback(
        success=True,
        reward=0.0,
        terminated=False,
        info={"holding_change": True},
    )
    blackbox.update(act_interact, fb3, source_id="env_mut")
    assert len(state.state_mutations) == 3
    m3 = state.state_mutations[2]
    assert m3.mutation_type == "HOLDING_CHANGE"
    assert m3.posterior_value is True


def test_cognitive_blackbox_decide_load_bearing_state_mutation():
    """Verify that decide() prioritizes navigating to a switch trigger when goal is obstructed by barriers."""
    from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel, StateMutationModel

    blackbox = CognitiveBlackbox()
    state = blackbox.get_state("test_env")
    state.avatar_feature = 1

    # Calibrate actions
    state.action_models[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=0.95, probes_tested=2
    )
    state.action_models[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=0.95, probes_tested=2
    )
    state.action_models[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=0.95, probes_tested=2
    )
    state.action_models[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=0.95, probes_tested=2
    )

    # Learned mutation: stepping on switch tile at (2, 1) opens the barrier
    switch_mut = StateMutationModel(
        trigger_type="CONTACT",
        trigger_pos=(2, 1),
        trigger_feature=8,
        mutation_type="BARRIER_OPEN",
        prior_value="blocked",
        posterior_value="open",
        confidence=0.95,
    )
    state.state_mutations.append(switch_mut)

    # Construct an EntityGraph:
    # Avatar is at (0, 0)
    # Goal is at (0, 5)
    # Wall/barriers at (0, 3), (1, 3), (2, 3), (3, 3) blocking direct path to goal
    # But path to switch (2, 1) is open!
    avatar = SpatialEntity(
        id="avatar",
        role=EntityRole.AGENT,
        centroid=(0.0, 0.0),
        grid_pos=(0, 0),
        area=1,
        bounding_box=(0, 0, 0, 0),
        color=1,
    )
    goal = SpatialEntity(
        id="goal",
        role=EntityRole.GOAL,
        centroid=(0.0, 5.0),
        grid_pos=(0, 5),
        area=1,
        bounding_box=(0, 0, 5, 5),
        color=9,
    )
    # A solid dividing wall across column 3 blocks all paths from col 0-2 to col 4-9
    barriers = {(r, 3) for r in range(10)}

    eg = blackbox.spatial_planner.construct_entity_graph(
        entities=[avatar, goal],
        barriers=barriers,
        grid_shape=(10, 10),
        step_size=1,
    )

    available_actions = [
        DriverAction(action_id=1),
        DriverAction(action_id=2),
        DriverAction(action_id=3),
        DriverAction(action_id=4),
    ]

    action = blackbox.decide(available_actions, source_id="test_env", entity_graph=eg)

    # Goal is obstructed by wall at column 3.
    # decide() should chain the state mutation at (2, 1), planning steps toward (2, 1)!
    # Moving toward (2, 1) from (0, 0) requires moving down (+r -> action 2) or right (+c -> action 4).
    assert action.action_id in (2, 4)
    assert len(state.current_plan) > 0
    assert "mutation_BARRIER_OPEN" in state.current_plan[0].target_entity_id
    assert state.current_plan[0].target_pos == (2, 1)
