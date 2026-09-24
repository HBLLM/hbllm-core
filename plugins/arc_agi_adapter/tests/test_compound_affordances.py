import numpy as np

from hbllm.drivers.base import DriverAction, DriverFeedback
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
from hbllm.hcir.spatial_planner import (
    EntityGraph,
    EntityRole,
    HCIRSpatialEntityPlanner,
    ObjectAffordanceRule,
    SpatialActionIntent,
    SpatialEntity,
)
from plugins.arc_agi_adapter.arc_perception import ARCPerceptualLifter


def test_spatial_entity_compound_signatures() -> None:
    # 2x2 square item of color 3
    key_ent = SpatialEntity(
        id="key_1",
        role=EntityRole.UNKNOWN,
        centroid=(5.5, 5.5),
        grid_pos=(5, 5),
        area=4,
        bounding_box=(5, 6, 5, 6),
        color=3,
    )
    sig_key = key_ent.get_signature_key()
    assert sig_key == "f3_srect_zsmall"

    # Moving key to a different position does not change its signature key
    key_ent_moved = SpatialEntity(
        id="key_2",
        role=EntityRole.UNKNOWN,
        centroid=(18.5, 22.5),
        grid_pos=(18, 22),
        area=4,
        bounding_box=(18, 19, 22, 23),
        color=3,
    )
    assert key_ent_moved.get_signature_key() == "f3_srect_zsmall"

    # 1x15 vertical wall of the same color 3
    wall_ent = SpatialEntity(
        id="wall_1",
        role=EntityRole.UNKNOWN,
        centroid=(7.0, 10.0),
        grid_pos=(0, 10),
        area=15,
        bounding_box=(0, 14, 10, 10),
        color=3,
    )
    sig_wall = wall_ent.get_signature_key()
    assert sig_wall == "f3_sline_v_zmedium"

    # Clean disambiguation despite identical color
    assert sig_key != sig_wall


def test_object_affordance_rule_serialization() -> None:
    rule = ObjectAffordanceRule(
        signature_key="f3_srect_zsmall",
        role=EntityRole.MANIPULABLE,
        preferred_action=5,
        action_intent=SpatialActionIntent.PICKUP,
        outcomes=["pickup_success"],
        confidence=0.95,
        times_confirmed=3,
    )
    d = rule.to_dict()
    assert d["signature_key"] == "f3_srect_zsmall"
    assert d["role"] == "manipulable"
    assert d["preferred_action"] == 5
    assert d["times_confirmed"] == 3

    hydrated = ObjectAffordanceRule.from_dict(d)
    assert hydrated.signature_key == rule.signature_key
    assert hydrated.role == rule.role
    assert hydrated.preferred_action == 5
    assert hydrated.outcomes == ["pickup_success"]
    assert hydrated.times_confirmed == 3


def test_agent_state_signature_and_affordance_persistence() -> None:
    blackbox = CognitiveBlackbox()
    state = blackbox.get_state("test")

    state.learned_cargo_signatures.add("f3_srect_zsmall")
    state.learned_obstacle_signatures.add("f3_sline_v_zmedium")
    state.learned_target_signatures.add("f8_spoint_zpoint")
    state.learned_affordance_rules["f3_srect_zsmall"] = ObjectAffordanceRule(
        signature_key="f3_srect_zsmall",
        role=EntityRole.MANIPULABLE,
        preferred_action=5,
        action_intent=SpatialActionIntent.PICKUP,
        outcomes=["pickup_success"],
    )

    # Test serialization
    state_dict = state.to_dict()
    assert "f3_srect_zsmall" in state_dict["learned_cargo_signatures"]
    assert "f3_sline_v_zmedium" in state_dict["learned_obstacle_signatures"]
    assert "f8_spoint_zpoint" in state_dict["learned_target_signatures"]
    assert "f3_srect_zsmall" in state_dict["learned_affordance_rules"]

    # Test reset with retain_memory
    blackbox.reset(source_id="test", retain_memory=True, is_retry=False)
    new_state = blackbox.get_state("test")
    assert "f3_srect_zsmall" in new_state.learned_cargo_signatures
    assert "f3_sline_v_zmedium" in new_state.learned_obstacle_signatures
    assert "f8_spoint_zpoint" in new_state.learned_target_signatures
    assert "f3_srect_zsmall" in new_state.learned_affordance_rules
    assert new_state.learned_affordance_rules["f3_srect_zsmall"].role == EntityRole.MANIPULABLE


def test_cognitive_blackbox_update_learns_signatures() -> None:
    blackbox = CognitiveBlackbox()
    state = blackbox.get_state("test")

    # Setup avatar and an adjacent item
    avatar = SpatialEntity(
        id="agent",
        role=EntityRole.AGENT,
        centroid=(2.0, 2.0),
        grid_pos=(2, 2),
        area=1,
        bounding_box=(2, 2, 2, 2),
        color=1,
    )
    key = SpatialEntity(
        id="key_1",
        role=EntityRole.UNKNOWN,
        centroid=(2.0, 3.0),
        grid_pos=(2, 3),
        area=4,
        bounding_box=(2, 3, 3, 4),
        color=3,
    )
    eg = EntityGraph(entities={"agent": avatar, "key_1": key}, agent=avatar)
    blackbox._source_entity_graphs["test"] = eg

    # Action 5 triggers pickup
    action = DriverAction(action_id=5, semantic_intent=SpatialActionIntent.PICKUP)
    state.carrying.holding = True
    feedback = DriverFeedback(
        success=True,
        reward=0.0,
        terminated=False,
        info={"holding_change": True},
    )
    blackbox.update(action, feedback, source_id="test")

    assert "f3_srect_zsmall" in state.learned_cargo_signatures
    assert "f3_srect_zsmall" in state.learned_affordance_rules
    rule = state.learned_affordance_rules["f3_srect_zsmall"]
    assert rule.role == EntityRole.MANIPULABLE
    assert rule.preferred_action == 5


def test_arc_perception_signature_priority_over_color() -> None:
    grid = np.zeros((20, 20), dtype=np.int32)
    # 2x2 key of color 3 at (2, 2)
    grid[2:4, 2:4] = 3
    # 1x12 barrier wall of color 3 at row 10
    grid[10:11, 2:14] = 3

    # Mock raw segmented objects
    class MockObj:
        def __init__(
            self,
            color: int,
            min_r: int,
            max_r: int,
            min_c: int,
            max_c: int,
            area: int,
        ):
            self.color = color
            self.min_r = min_r
            self.max_r = max_r
            self.min_c = min_c
            self.max_c = max_c
            self.area = area
            self.centroid = ((min_r + max_r) * 0.5, (min_c + max_c) * 0.5)
            self.width = max_c - min_c + 1
            self.height = max_r - min_r + 1
            self.coords = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]

    raw_objects = [
        MockObj(color=3, min_r=2, max_r=3, min_c=2, max_c=3, area=4),
        MockObj(color=3, min_r=10, max_r=10, min_c=2, max_c=13, area=12),
    ]

    # Lift with signature affordances:
    # 2x2 square is cargo (MANIPULABLE), horizontal line is barrier (OBSTACLE)
    learned_cargo_sigs = {"f3_srect_zsmall"}
    learned_obstacle_sigs = {"f3_sline_h_zmedium"}

    entities, barriers = ARCPerceptualLifter.lift(
        grid=grid,
        raw_objects=raw_objects,
        avatar_color=1,
        learned_cargo_signatures=learned_cargo_sigs,
        learned_obstacle_signatures=learned_obstacle_sigs,
    )

    ent_map = {e.get_signature_key(): e for e in entities}
    assert "f3_srect_zsmall" in ent_map
    assert ent_map["f3_srect_zsmall"].role == EntityRole.MANIPULABLE
    assert "f3_sline_h_zmedium" in ent_map
    assert ent_map["f3_sline_h_zmedium"].role == EntityRole.OBSTACLE

    # Verify that the key's cells are NOT in barriers, but the wall's cells ARE
    assert (2, 2) not in barriers
    assert (3, 3) not in barriers
    assert (10, 5) in barriers


def test_spatial_planner_signature_candidate_filtering() -> None:
    planner = HCIRSpatialEntityPlanner()
    avatar = SpatialEntity(
        id="agent",
        role=EntityRole.AGENT,
        centroid=(0.0, 0.0),
        grid_pos=(0, 0),
        area=1,
        bounding_box=(0, 0, 0, 0),
        color=1,
    )
    # Key with signature f3_srect_zsmall
    key = SpatialEntity(
        id="key_1",
        role=EntityRole.MANIPULABLE,
        centroid=(2.0, 2.0),
        grid_pos=(2, 2),
        area=4,
        bounding_box=(2, 3, 2, 3),
        color=3,
    )
    # Barrier that was marked manipulable by mistake, with signature f3_sline_v_zmedium
    fake_item = SpatialEntity(
        id="wall_fake",
        role=EntityRole.MANIPULABLE,
        centroid=(5.0, 5.0),
        grid_pos=(5, 5),
        area=15,
        bounding_box=(0, 14, 5, 5),
        color=3,
    )
    receptacle = SpatialEntity(
        id="goal_zone",
        role=EntityRole.RECEPTACLE,
        centroid=(10.0, 10.0),
        grid_pos=(10, 10),
        area=9,
        bounding_box=(9, 11, 9, 11),
        color=8,
    )

    eg = EntityGraph(
        entities={
            "agent": avatar,
            "key_1": key,
            "wall_fake": fake_item,
            "goal_zone": receptacle,
        },
        agent=avatar,
        grid_shape=(20, 20),
    )

    # When planning with cargo_signatures and obstacle_signatures:
    plan = planner.plan_sequence(
        eg=eg,
        cargo_signatures={"f3_srect_zsmall"},
        obstacle_signatures={"f3_sline_v_zmedium"},
    )

    assert len(plan) > 0
    target_ids = [step.target_entity_id for step in plan]
    # The valid key must be targeted for pickup
    assert "key_1" in target_ids
    # The fake item matching obstacle signature must NEVER be targeted
    assert "wall_fake" not in target_ids
