"""Unit and regression tests for ARC3SpatialCognitiveAgent and Driver Management Layer."""

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

from hbllm.drivers import BaseDriver, DriverManager
from plugins.arc_agi_adapter.arc_driver import ArcadeDriver
from plugins.arc_agi_adapter.arc_spatial_agent import (
    ActionDynamicsModel,
    AgentPhase,
    ARC3SpatialCognitiveAgent,
    HCIRCrossGameMemory,
    ShapeArchetype,
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


def test_arc_spatial_agent_wa30_all_levels() -> None:
    """Verify ARC3SpatialCognitiveAgent solves wa30 across all 3 levels."""
    client = Arcade()
    env = client.make("wa30", render_mode=None)
    fd = env.reset()
    agent = ARC3SpatialCognitiveAgent()

    for lvl in range(3):
        completed = False
        for s in range(110):
            if getattr(fd, "levels_completed", 0) > lvl:
                completed = True
                break
            if not hasattr(fd, "frame") or len(fd.frame) == 0:
                break
            avail = getattr(fd, "available_actions", [1, 2, 3, 4, 5])
            curr_grid = fd.frame[-1]
            act, conf = agent.plan_next_action(curr_grid, avail)
            prev_grid = curr_grid
            fd = env.step(getattr(GameAction, f"ACTION{act}"))
            curr_grid = fd.frame[-1] if len(fd.frame) > 0 else prev_grid
            agent.update_causal_dynamics(act, prev_grid, curr_grid)

        if not completed and getattr(fd, "levels_completed", 0) > lvl:
            completed = True
        assert completed, f"Level {lvl} failed to complete within step budget"
        agent.reset_episode(retain_dynamics=True)
        try:
            fd = env.step(GameAction.ACTION5)
        except Exception:
            pass

    assert getattr(fd, "levels_completed", 0) == 3


def test_arc_spatial_agent_autonomous_learning_and_soft_restart() -> None:
    """Verify epistemic exploration deduces task, soft-restarts via RESET, and achieves optimal score."""
    client = Arcade()
    env = client.make("wa30", render_mode=None)
    fd = env.reset()
    custom_memory = HCIRCrossGameMemory()
    agent = ARC3SpatialCognitiveAgent(enable_soft_restart=True, shared_memory=custom_memory)

    assert agent.phase == AgentPhase.EPISTEMIC_LEARNING
    subgoals_deduced = False
    restarted = False

    for s in range(120):
        if getattr(fd, "levels_completed", 0) > 0:
            break
        curr_grid = fd.frame[-1]
        avail = getattr(fd, "available_actions", [1, 2, 3, 4, 5])
        act, conf = agent.plan_next_action(curr_grid, avail, allow_soft_restart=True)
        prev_grid = curr_grid

        if act == 0 or agent.should_soft_restart:
            restarted = True
            subgoals_deduced = len(agent.optimal_task_plan) > 0
            fd = env.step(GameAction.RESET)
            curr_grid = fd.frame[-1]
            agent.soft_restart()
            assert agent.phase == AgentPhase.OPTIMAL_EXECUTION
            continue

        fd = env.step(getattr(GameAction, f"ACTION{act}"))
        curr_grid = fd.frame[-1] if len(fd.frame) > 0 else prev_grid
        agent.update_causal_dynamics(act, prev_grid, curr_grid)

    assert restarted, "Agent did not trigger soft restart after task deduction"
    assert subgoals_deduced, "Agent did not deduce optimal task plan"
    assert getattr(fd, "levels_completed", 0) >= 1, "Failed to complete Level 1 after soft restart"

    # Verify episode outcome recording
    agent.record_episode_outcome(completed=True)
    assert agent.phase == AgentPhase.COMPLETED
    assert len(custom_memory.successful_episodes) == 1
    assert custom_memory.successful_episodes[0]["score"] == 1.0


def test_arc_spatial_agent_cross_game_knowledge_transfer() -> None:
    """Verify exported cross-game memory transfers motor dynamics and concepts zero-shot to a new agent."""
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
    memory.action_models[2] = ActionDynamicsModel(
        action_id=2, delta_r=4, delta_c=0, confidence=0.95, probes_tested=5
    )
    memory.action_models[3] = ActionDynamicsModel(
        action_id=3, delta_r=0, delta_c=-4, confidence=0.95, probes_tested=5
    )
    memory.action_models[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=4, confidence=0.95, probes_tested=5
    )

    exported = memory.export_dict()

    new_agent = ARC3SpatialCognitiveAgent(shared_memory=HCIRCrossGameMemory())
    assert new_agent.avatar_color is None
    assert new_agent.step_size == 1
    assert len(new_agent.action_models) == 0

    new_agent.import_knowledge(exported)

    assert new_agent.avatar_color == 14
    assert new_agent.step_size == 4
    assert new_agent.action_5_affordance == "PICKUP_DROP"
    assert 4 in new_agent.learned_item_colors
    assert 9 in new_agent.learned_receptacle_colors
    assert all(a in new_agent.action_models for a in [1, 2, 3, 4])
    assert all(new_agent.action_models[a].confidence >= 0.9 for a in [1, 2, 3, 4])


def test_arc_spatial_agent_failure_constraint_induction() -> None:
    """Verify motor collision generates HCIR negative constraints and BeliefNode in cognitive workspace."""
    memory = HCIRCrossGameMemory()
    agent = ARC3SpatialCognitiveAgent(shared_memory=memory)

    agent.avatar_color = 14
    agent.step_size = 2
    agent.spatial_planner.step_size = 2
    agent.avatar_centroid = (10.0, 10.0)
    agent.action_models[1] = ActionDynamicsModel(
        action_id=1, delta_r=-2, delta_c=0, confidence=0.95, probes_tested=3
    )

    grid = np.zeros((30, 30), dtype=np.uint8)
    grid[10, 10] = 14
    grid[8, 10] = 5

    prev_grid = grid.copy()
    curr_grid = grid.copy()

    agent.update_causal_dynamics(1, prev_grid, curr_grid)

    assert (8, 10) in agent.spatial_planner._learned_barriers
    assert (8, 10) in memory.negative_constraints
    assert 5 in agent.learned_barrier_colors

    nodes = list(agent.workspace.graph._nodes.values())
    belief_nodes = [
        n
        for n in nodes
        if hasattr(n, "properties") and n.properties.get("negative_constraint") is True
    ]
    assert len(belief_nodes) >= 1
    assert belief_nodes[0].properties.get("position") == (8, 10)


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


def test_morphological_concept_learning_and_transfer() -> None:
    """Verify MorphologicalConcept preserves shape archetype across color mutations and transfers in memory."""
    memory = HCIRCrossGameMemory()
    agent = ARC3SpatialCognitiveAgent(shared_memory=memory)

    # Agent observes red door shape (color 2)
    coords = {(4, 10), (4, 11), (4, 12)}
    arch = ShapeArchetype.from_coords(coords)
    concept = agent._get_or_create_shape_concept(
        arch, color=2, role=None, name_prefix="sliding_door"
    )
    concept.observed_colors.add(2)
    concept.barrier_colors.add(2)

    # Same shape seen in next level with blue color (color 8)
    concept_level2 = agent._get_or_create_shape_concept(arch, color=8)
    assert concept_level2 is concept
    assert 2 in concept.observed_colors and 8 in concept.observed_colors

    # Export to memory and import to a new agent
    exp = agent.export_knowledge()
    assert "shape_concepts" in exp

    agent2 = ARC3SpatialCognitiveAgent()
    agent2.import_knowledge(exp)
    assert arch.canonical_id in agent2.shape_concepts
    restored = agent2.shape_concepts[arch.canonical_id]
    assert 2 in restored.observed_colors and 8 in restored.observed_colors


def test_rotating_door_automatic_discovery_and_unblocking() -> None:
    """Verify online detection of rotating doors: unblocks vacated passage cells and blocks new cells."""
    agent = ARC3SpatialCognitiveAgent()
    agent.avatar_color = 14
    agent.avatar_centroid = (5.0, 5.0)

    # Setup initial 20x20 grid with a vertical door barrier (color 7) blocking row 10, cols 10..12
    prev_g = np.zeros((20, 20), dtype=np.uint8)
    prev_g[5, 5] = 14  # Avatar
    door_v = {(10, 10), (11, 10), (12, 10)}
    for r, c in door_v:
        prev_g[r, c] = 7

    # Initialize known barriers with the vertical door
    agent.known_barriers = np.zeros((20, 20), dtype=bool)
    agent.learned_barrier_colors.add(7)
    for r, c in door_v:
        agent.known_barriers[r, c] = True
        agent.spatial_planner.record_collision_barrier((r, c))

    # Action 5 triggers 90 degree door rotation: vertical door becomes horizontal door at row 10, cols 10..12
    curr_g = np.zeros((20, 20), dtype=np.uint8)
    curr_g[5, 5] = 14  # Avatar
    door_h = {(10, 10), (10, 11), (10, 12)}
    for r, c in door_h:
        curr_g[r, c] = 7

    agent.update_causal_dynamics(action_id=5, prev_grid=prev_g, curr_grid=curr_g)

    # Vacated cells (11, 10) and (12, 10) must now be UNBLOCKED in known_barriers & spatial_planner
    assert not agent.known_barriers[11, 10]
    assert not agent.known_barriers[12, 10]
    assert (11, 10) not in agent.spatial_planner._learned_barriers
    assert (12, 10) not in agent.spatial_planner._learned_barriers

    # Newly occupied cells (10, 11) and (10, 12) must now be BLOCKED
    assert agent.known_barriers[10, 11]
    assert agent.known_barriers[10, 12]

    # Morphological concept must record rotation trigger
    v_arch = ShapeArchetype.from_coords({(11, 10), (12, 10)})
    assert v_arch.canonical_id in agent.shape_concepts
    door_concept = agent.shape_concepts[v_arch.canonical_id]
    assert door_concept.is_rotatable
    assert door_concept.rotation_trigger == 5


def test_color_transition_door_opening() -> None:
    """Verify online detection of color-changing door opening into passable floor."""
    agent = ARC3SpatialCognitiveAgent()
    agent.avatar_color = 14

    prev_g = np.zeros((20, 20), dtype=np.uint8)
    door_cells = {(8, 5), (8, 6)}
    for r, c in door_cells:
        prev_g[r, c] = 9  # Barrier door color 9

    agent.known_barriers = np.zeros((20, 20), dtype=bool)
    for r, c in door_cells:
        agent.known_barriers[r, c] = True
        agent.spatial_planner.record_collision_barrier((r, c))

    # Stepping or interacting turns color 9 door to color 0 (floor)
    curr_g = np.zeros((20, 20), dtype=np.uint8)  # All 0

    agent.update_causal_dynamics(action_id=2, prev_grid=prev_g, curr_grid=curr_g)

    # Door cells must now be unblocked!
    for r, c in door_cells:
        assert not agent.known_barriers[r, c]
        assert (r, c) not in agent.spatial_planner._learned_barriers

    arch = ShapeArchetype.from_coords(door_cells)
    assert arch.canonical_id in agent.shape_concepts
    concept = agent.shape_concepts[arch.canonical_id]
    assert concept.is_color_switch
    assert 0 in concept.passable_colors
