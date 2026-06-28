import asyncio

import pytest

from hbllm.brain.autonomy.task_graph import Goal, GoalStatus
from hbllm.brain.executive_cortex import CognitiveExecutiveController
from hbllm.brain.intentional_workspace import IntentionalWorkspace
from hbllm.brain.self_model import SelfModel
from hbllm.network.messages import Message, MessageType


def test_intentional_workspace_operations(tmp_path):
    ws = IntentionalWorkspace(data_dir=str(tmp_path))

    # 1. Goal management
    g1 = Goal(goal_id="g1", name="Goal 1", description="desc 1", status=GoalStatus.ACTIVE)
    g2 = Goal(goal_id="g2", name="Goal 2", description="desc 2", status=GoalStatus.PENDING)

    ws.add_goal(g1)
    ws.add_goal(g2)

    assert len(ws.get_active_goals()) == 1
    assert ws.get_active_goals()[0].goal_id == "g1"

    # Update status
    ws.update_goal_status("g2", GoalStatus.ACTIVE)
    assert len(ws.get_active_goals()) == 2

    # Defer a goal
    ws.update_goal_status("g1", GoalStatus.PAUSED)
    assert len(ws.get_active_goals()) == 1
    assert len(ws.get_deferred_goals()) == 1
    assert ws.get_deferred_goals()[0].goal_id == "g1"

    # 2. Curiosity goals
    ws.add_curiosity_goal("curiosity 1")
    ws.add_curiosity_goal("curiosity 1")  # Dupe check
    ws.add_curiosity_goal("curiosity 2")

    assert len(ws.get_curiosity_goals()) == 2
    assert "curiosity 1" in ws.get_curiosity_goals()

    ws.remove_curiosity_goal("curiosity 1")
    assert len(ws.get_curiosity_goals()) == 1
    assert ws.get_curiosity_goals()[0] == "curiosity 2"

    # 3. Opportunities & Threats
    ws.add_opportunity("opp1", "Desc opp", {"meta": "val"})
    opps = ws.get_opportunities()
    assert len(opps) == 1
    assert opps[0]["id"] == "opp1"
    assert opps[0]["metadata"] == {"meta": "val"}

    ws.add_threat("t1", "Desc threat 1", 0.5)
    ws.add_threat("t2", "Desc threat 2", 0.9)
    threats = ws.get_threats()
    assert len(threats) == 2
    # Sorted by severity DESC
    assert threats[0]["id"] == "t2"
    assert threats[1]["id"] == "t1"


@pytest.mark.asyncio
async def test_executive_orchestration_new_goal(tmp_path, bus):
    intentional_ws = IntentionalWorkspace(data_dir=str(tmp_path))
    self_model = SelfModel(data_dir=str(tmp_path))

    controller = CognitiveExecutiveController(
        node_id="executive", intentional_workspace=intentional_ws, self_model=self_model
    )

    # Run the controller node on our test bus
    await controller.start(bus)

    # Keep track of published state changes
    state_changes = []

    async def track_state_change(msg):
        state_changes.append(msg)

    await bus.subscribe("workspace.cognition.state_change", track_state_change)

    # Trigger a new goal event
    goal_msg = Message(
        type=MessageType.EVENT,
        source_node_id="user_interface",
        topic="workspace.cognition.goal",
        correlation_id="corr_123",
        payload={
            "goal_id": "test_goal_1",
            "name": "Solve equations",
            "description": "Calculate simple parameters",
            "domain": "math",
        },
    )

    await bus.publish("workspace.cognition.goal", goal_msg)

    # Wait briefly for execution
    await asyncio.sleep(0.5)

    # Verify: Goal is registered
    active_goals = intentional_ws.get_active_goals()
    assert len(active_goals) == 1
    assert active_goals[0].goal_id == "test_goal_1"

    # Verify: State version 1 was initialized and published
    assert len(state_changes) == 1
    state_dict = state_changes[0].payload["state"]
    assert state_dict["version"] == 1
    assert state_dict["goal"]["goal_id"] == "test_goal_1"
    assert (
        state_dict["policy"]["reasoning_strategy"] == "CoT"
    )  # Default since math is not a weakness yet

    # Stop controller node
    await controller.stop()


@pytest.mark.asyncio
async def test_executive_state_derivation_and_finalization(tmp_path, bus):
    intentional_ws = IntentionalWorkspace(data_dir=str(tmp_path))
    self_model = SelfModel(data_dir=str(tmp_path))

    controller = CognitiveExecutiveController(
        node_id="executive", intentional_workspace=intentional_ws, self_model=self_model
    )
    await controller.start(bus)

    # 1. Initialize goal state
    goal_msg = Message(
        type=MessageType.EVENT,
        source_node_id="user_interface",
        topic="workspace.cognition.goal",
        correlation_id="corr_final_test",
        payload={
            "goal_id": "final_goal_id",
            "name": "Consolidate databases",
            "description": "Merge state tables",
            "domain": "database",
        },
    )
    await bus.publish("workspace.cognition.goal", goal_msg)
    await asyncio.sleep(0.2)

    current_state = controller.active_states["corr_final_test"]
    assert current_state.version == 1

    # 2. Simulate workspace node updating the state (Version 2)
    # Derive a new state snapshot
    derived_state = current_state.derive_state(
        retrieved_memory=[{"fact": "DB uses SQLite"}], working_memory={"engine": "sqlite"}
    )

    update_msg = Message(
        type=MessageType.EVENT,
        source_node_id="planner",
        topic="workspace.cognition.state_change",
        correlation_id="corr_final_test",
        payload={"state": derived_state.to_dict()},
    )
    await bus.publish("workspace.cognition.state_change", update_msg)
    await asyncio.sleep(0.2)

    # Verify state version is updated to 2 in controller
    assert controller.active_states["corr_final_test"].version == 2
    assert controller.active_states["corr_final_test"].working_memory == {"engine": "sqlite"}

    # 3. Simulate evaluation node finalising the state to COMPLETED (Version 3)
    final_goal = Goal(
        goal_id=derived_state.goal.goal_id,
        tenant_id=derived_state.goal.tenant_id,
        name=derived_state.goal.name,
        description=derived_state.goal.description,
        status=GoalStatus.COMPLETED,
        priority=derived_state.goal.priority,
        created_at=derived_state.goal.created_at,
        metadata=derived_state.goal.metadata,
    )
    final_state = derived_state.derive_state(goal=final_goal)

    final_msg = Message(
        type=MessageType.EVENT,
        source_node_id="evaluation",
        topic="workspace.cognition.state_change",
        correlation_id="corr_final_test",
        payload={"state": final_state.to_dict()},
    )
    await bus.publish("workspace.cognition.state_change", final_msg)
    await asyncio.sleep(0.2)

    # Verify: Active state is removed after finalization
    assert "corr_final_test" not in controller.active_states

    # Verify: SelfModel registered a success outcome for database domain
    cap = self_model.get_capability("database")
    assert cap is not None
    assert cap.score == 1.0

    await controller.stop()
