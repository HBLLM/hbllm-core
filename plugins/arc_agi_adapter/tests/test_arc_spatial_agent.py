"""Unit and regression tests for ARC3SpatialCognitiveAgent and Driver Management Layer."""

import numpy as np
from arc_agi import Arcade
from arcengine import GameAction

from hbllm.drivers import BaseDriver, DriverManager
from plugins.arc_agi_adapter.arc_driver import ArcadeDriver
from plugins.arc_agi_adapter.arc_spatial_agent import (
    AgentPhase,
    ARC3SpatialCognitiveAgent,
    HCIRCrossGameMemory,
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
    from plugins.arc_agi_adapter.arc_agi_3_runner import ActionDynamicsModel

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
    from plugins.arc_agi_adapter.arc_agi_3_runner import ActionDynamicsModel

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
