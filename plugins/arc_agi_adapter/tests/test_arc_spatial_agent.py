"""Unit and regression tests for ARC3SpatialCognitiveAgent (thin blackbox adapter)."""

import numpy as np

from hbllm.drivers import BaseDriver, DriverManager
from hbllm.drivers.base import DriverAction, DriverFeedback
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
from hbllm.hcir.world.morphology import ShapeArchetype
from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
from plugins.arc_agi_adapter.arc_driver import ArcadeDriver
from plugins.arc_agi_adapter.arc_memory import HCIRCrossGameMemory
from plugins.arc_agi_adapter.arc_spatial_agent import (
    ARC3SpatialCognitiveAgent,
)


def test_driver_management_registration() -> None:
    """Verify ArcadeDriver registers with DriverManager."""
    manager = DriverManager()
    driver = ArcadeDriver()
    assert isinstance(driver, BaseDriver)
    manager.register(driver)
    retrieved = manager.get_driver("arcade_driver")
    assert retrieved is driver
    assert manager.get_driver("arcade_driver").name == "arcade_driver"


def test_agent_creates_blackbox() -> None:
    """Verify ARC3SpatialCognitiveAgent wraps a CognitiveBlackbox."""
    agent = ARC3SpatialCognitiveAgent()
    assert isinstance(agent.blackbox, CognitiveBlackbox)


def test_agent_plan_returns_valid_action() -> None:
    """Verify plan_next_action returns an action from the available list."""
    agent = ARC3SpatialCognitiveAgent()
    grid = np.zeros((16, 16), dtype=np.uint8)
    avail = [1, 2, 3, 4, 5]
    act, conf = agent.plan_next_action(grid, avail)
    assert act in avail or act == 0
    assert 0.0 <= conf <= 1.0


def test_agent_update_does_not_crash() -> None:
    """Verify update_causal_dynamics handles grid diffs without error."""
    agent = ARC3SpatialCognitiveAgent()
    prev = np.zeros((16, 16), dtype=np.uint8)
    curr = np.zeros((16, 16), dtype=np.uint8)
    curr[5, 5] = 3  # Some change
    agent.update_causal_dynamics(1, prev, curr)


def test_agent_update_mismatched_shapes() -> None:
    """Verify update_causal_dynamics gracefully handles mismatched grid shapes."""
    agent = ARC3SpatialCognitiveAgent()
    prev = np.zeros((16, 16), dtype=np.uint8)
    curr = np.zeros((20, 20), dtype=np.uint8)
    agent.update_causal_dynamics(1, prev, curr)  # Should not crash


def test_agent_reset_episode() -> None:
    """Verify reset_episode creates fresh state."""
    agent = ARC3SpatialCognitiveAgent()
    # Do a plan to increment step count
    grid = np.zeros((16, 16), dtype=np.uint8)
    agent.plan_next_action(grid, [1, 2, 3, 4])
    agent.reset_episode()
    # After reset, blackbox state is fresh
    state = agent.blackbox.get_state("default")
    assert state.step_count == 0


def test_cross_game_memory_export_import() -> None:
    """Verify cross-game memory serialization round-trip."""
    memory = HCIRCrossGameMemory()
    memory.avatar_color = 14
    memory.step_size = 4
    memory.action_5_affordance = "PICKUP_DROP"
    memory.learned_item_colors.add(4)
    memory.learned_receptacle_colors.add(9)
    memory.learned_walkable_colors.update([0, 1])
    memory.action_models[1] = ActionDynamicsModel(
        action_id=1, delta_r=-4, delta_c=0, confidence=0.95, probes_tested=5
    )

    exported = memory.export_dict()
    assert exported["avatar_color"] == 14
    assert exported["step_size"] == 4

    new_memory = HCIRCrossGameMemory()
    new_memory.import_dict(exported)
    assert new_memory.avatar_color == 14
    assert new_memory.step_size == 4
    assert new_memory.action_5_affordance == "PICKUP_DROP"
    assert 4 in new_memory.learned_item_colors
    assert 1 in new_memory.action_models
    assert new_memory.action_models[1].confidence >= 0.9


def test_agent_record_episode_outcome() -> None:
    """Verify record_episode_outcome writes to cross-game memory."""
    memory = HCIRCrossGameMemory()
    agent = ARC3SpatialCognitiveAgent(shared_memory=memory)
    agent.record_episode_outcome(completed=True)
    assert len(memory.successful_episodes) == 1
    assert memory.successful_episodes[0]["score"] == 1.0


def test_shape_archetype_translation_and_rotational_orbits() -> None:
    """Verify ShapeArchetype canonicalization and 90/180/270 degree rotation orbit matching."""
    # Horizontal bar of 3 pixels at (5, 5), (5, 6), (5, 7)
    h_bar_coords = {(5, 5), (5, 6), (5, 7)}
    arch_h = ShapeArchetype.from_coords(h_bar_coords)
    assert arch_h.height == 1
    assert arch_h.width == 3
    assert arch_h.area == 3
    assert arch_h.canonical_id == ((0, 0), (0, 1), (0, 2))

    # Translated horizontal bar at (20, 10), (20, 11), (20, 12)
    h_bar_translated = {(20, 10), (20, 11), (20, 12)}
    arch_h_trans = ShapeArchetype.from_coords(h_bar_translated)
    assert arch_h == arch_h_trans  # Exactly equal canonical archetype

    # Vertical bar of 3 pixels (90 degree rotation of horizontal bar)
    v_bar_coords = {(8, 10), (9, 10), (10, 10)}
    arch_v = ShapeArchetype.from_coords(v_bar_coords)
    assert arch_v.height == 3
    assert arch_v.width == 1
    assert arch_h.is_rotation_of(arch_v)
    assert arch_v.is_rotation_of(arch_h)


def test_cognitive_blackbox_observe_decide_update_loop() -> None:
    """Verify the full observe→decide→update loop works end-to-end."""
    blackbox = CognitiveBlackbox()

    available = [DriverAction(action_id=a) for a in [1, 2, 3, 4]]
    action = blackbox.decide(available, source_id="test")
    assert isinstance(action, DriverAction)
    assert action.action_id in [0, 1, 2, 3, 4]

    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"observed_delta": [0, 1]},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert action.action_id in state.action_models


def test_cognitive_blackbox_learns_obstacles_from_feedback() -> None:
    """Verify blackbox learns obstacle features from collision feedback."""
    blackbox = CognitiveBlackbox()
    action = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"collision_feature": 5},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert 5 in state.learned_obstacle_features


def test_cognitive_blackbox_learns_traversable_from_feedback() -> None:
    """Verify blackbox learns traversable features from successful movement."""
    blackbox = CognitiveBlackbox()
    action = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"traversed_feature": 3},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert 3 in state.learned_traversable_features


def test_cognitive_blackbox_reset_preserves_memory() -> None:
    """Verify blackbox reset with retain_memory keeps learned knowledge."""
    blackbox = CognitiveBlackbox()

    # Teach it something
    action = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"collision_feature": 7, "observed_delta": [-2, 0]},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert 7 in state.learned_obstacle_features
    assert 1 in state.action_models

    # Reset with memory retention
    blackbox.reset(source_id="test", retain_memory=True)

    state = blackbox.get_state("test")
    assert 7 in state.learned_obstacle_features
    assert 1 in state.action_models
    assert state.step_count == 0  # Step count reset


def test_cognitive_blackbox_reset_clears_memory() -> None:
    """Verify blackbox reset without retain_memory clears everything."""
    blackbox = CognitiveBlackbox()

    action = DriverAction(action_id=1)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=False,
        info={"collision_feature": 7},
    )
    blackbox.update(action, feedback, source_id="test")

    blackbox.reset(source_id="test", retain_memory=False)

    state = blackbox.get_state("test")
    assert len(state.learned_obstacle_features) == 0
    assert len(state.action_models) == 0


def test_cognitive_blackbox_learns_target_feature_on_success() -> None:
    """Verify blackbox learns target feature empirically from win/reward."""
    blackbox = CognitiveBlackbox()
    action = DriverAction(action_id=2)
    feedback = DriverFeedback(
        success=True,
        reward=1.0,
        terminated=True,
        info={"reached_feature": 3},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert 3 in state.learned_target_features
    assert 3 not in state.learned_obstacle_features


def test_cognitive_blackbox_learns_hazard_feature_on_failure() -> None:
    """Verify blackbox learns hazard feature empirically from loss/termination."""
    blackbox = CognitiveBlackbox()
    action = DriverAction(action_id=3)
    feedback = DriverFeedback(
        success=False,
        reward=0.0,
        terminated=True,
        info={"hazard_feature": 8},
    )
    blackbox.update(action, feedback, source_id="test")

    state = blackbox.get_state("test")
    assert 8 in state.learned_obstacle_features
    assert 8 not in state.learned_target_features


def test_default_grid_2d_lifter_and_zero_shot_planning() -> None:
    """Verify default_grid_2d_lifter segments grid and plans zero-shot to empirical goal."""
    from hbllm.drivers.base import DriverInput, DriverModality
    from hbllm.hcir.spatial_planner import EntityRole

    agent = ARC3SpatialCognitiveAgent()

    # Step 1: In Episode 1, agent receives feedback that feature 3 is the goal, 7 is a barrier
    agent.avatar_color = 2
    feedback1 = DriverFeedback(
        success=True,
        reward=1.0,
        terminated=True,
        info={
            "reached_feature": 3,
            "collision_feature": 7,
            "observed_delta": [0, 1],
        },
    )
    agent.blackbox.update(DriverAction(action_id=4), feedback1, source_id="arc_agi")

    # Add calibration for remaining actions
    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

    agent.action_models[1] = ActionDynamicsModel(
        action_id=1, delta_r=-1, delta_c=0, confidence=0.9, probes_tested=1
    )
    agent.action_models[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=0.9, probes_tested=1
    )
    agent.action_models[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-1, confidence=0.9, probes_tested=1
    )
    agent.action_models[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=0.9, probes_tested=1
    )

    # Step 2: Reset for Episode 2 (retaining empirical memory)
    agent.reset_episode(retain_dynamics=True)
    assert 3 in agent.learned_item_colors
    assert 7 in agent.learned_barrier_colors
    assert agent.avatar_color == 2

    # Step 3: Observe a new grid layout for Episode 2:
    # Avatar (2) at (1, 1), Wall (7) blocking direct path at (1, 2), Goal (3) at (1, 3)
    grid2 = np.zeros((5, 5), dtype=np.uint8)
    grid2[1, 1] = 2  # Avatar
    grid2[1, 2] = 7  # Wall
    grid2[1, 3] = 3  # Goal

    # Call observe through DriverInput
    driver_input = DriverInput(
        raw_data=grid2,
        modality=DriverModality.GRID_2D,
        source_id="arc_agi",
        metadata={"grid_shape": (5, 5), "step_size": 1},
    )
    eg = agent.blackbox.observe(driver_input, perception_data={"grid": grid2, "avatar_color": 2})

    assert eg is not None
    assert eg.avatar is not None
    assert eg.avatar.grid_pos == (1, 1)
    assert (1, 2) in eg.barriers

    # Verify goal entity was lifted with role GOAL dynamically from empirical memory
    goals = [e for e in eg.entities.values() if e.role == EntityRole.GOAL]
    assert len(goals) == 1
    assert goals[0].grid_pos == (1, 3)

    # Step 4: Plan next action — agent avoids the wall at (1, 2) by going down (action 2)
    act, conf = agent.plan_next_action(grid2, [1, 2, 3, 4])
    # The A* safe path must go around the wall: either down (dr=+1) or up (dr=-1), NOT right (action 4)
    assert act in [1, 2]
    assert conf >= 0.85


def test_multi_cell_stride_auto_detection() -> None:
    """Verify ARC3SpatialCognitiveAgent detects block-stride avatar translations (e.g. wa30)."""
    agent = ARC3SpatialCognitiveAgent()

    # Create prev_grid with a 4x4 avatar at (16, 16)
    prev = np.zeros((32, 32), dtype=np.uint8)
    prev[16:20, 16:20] = 5  # 4x4 avatar block

    # Action 1 moves avatar up by 4 units to (12, 16)
    curr = np.zeros((32, 32), dtype=np.uint8)
    curr[12:16, 16:20] = 5

    agent.update_causal_dynamics(action_id=1, prev_grid=prev, curr_grid=curr)

    assert agent.avatar_color == 5
    assert agent.step_size == 4
    model = agent.action_models.get(1)
    assert model is not None
    assert model.delta_r == -4
    assert model.delta_c == 0


def test_cross_attempt_frontier_exploration() -> None:
    """Verify SpatialPlanner de-prioritizes explored entities and explores frontiers on retry."""
    from hbllm.hcir.graph import WorldVariableNode
    from hbllm.hcir.spatial_planner import EntityRole, HCIRSpatialEntityPlanner, SpatialEntity
    from hbllm.hcir.workspace import HCIRWorkspaceState

    planner = HCIRSpatialEntityPlanner()
    ws = HCIRWorkspaceState()

    # Avatar at (1, 1), Candidate 1 at (1, 3), Candidate 2 at (3, 1)
    avatar = SpatialEntity(
        id="agent",
        role=EntityRole.AGENT,
        centroid=(1.0, 1.0),
        grid_pos=(1, 1),
        area=1,
        bounding_box=(1, 1, 1, 1),
    )
    cand1 = SpatialEntity(
        id="cand_explored",
        role=EntityRole.UNKNOWN,
        centroid=(1.0, 3.0),
        grid_pos=(1, 3),
        area=1,
        bounding_box=(1, 1, 3, 3),
    )
    cand2 = SpatialEntity(
        id="cand_new",
        role=EntityRole.UNKNOWN,
        centroid=(3.0, 1.0),
        grid_pos=(3, 1),
        area=1,
        bounding_box=(3, 3, 1, 1),
    )

    eg = planner.construct_entity_graph(
        entities=[avatar, cand1, cand2],
        barriers=set(),
        grid_shape=(10, 10),
        step_size=1,
    )

    # In workspace, mark cand1 as already explored in prior attempt
    ws.upsert_node(
        WorldVariableNode(
            id="var_explored_entity_ids",
            variable_name="var_explored_entity_ids",
            value=["cand_explored"],
        )
    )
    ws.upsert_node(
        WorldVariableNode(
            id="var_explored_entity_positions",
            variable_name="var_explored_entity_positions",
            value=[[1, 3]],
        )
    )

    plan = planner.plan_sequence(eg=eg, workspace=ws)
    assert len(plan) > 0
    # The planner must prioritize the unexplored candidate (cand_new at (3, 1))
    assert plan[0].target_entity_id == "cand_new"
    assert plan[0].target_pos == (3, 1)
