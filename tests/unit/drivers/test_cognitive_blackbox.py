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
