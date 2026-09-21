"""Unit tests for HCIRSpatialEntityPlanner."""

from __future__ import annotations

from hbllm.hcir.spatial_planner import (
    EntityRole,
    HCIRSpatialEntityPlanner,
    SpatialEntity,
)
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_partition_detection_and_cutsets() -> None:
    planner = HCIRSpatialEntityPlanner(step_size=1)

    # Avatar at (2, 2)
    # Wall at col 5 from row 0 to 9 with a doorway gap at (4, 5)
    # Goal at (2, 8)
    avatar = SpatialEntity(
        id="agent_1",
        role=EntityRole.AGENT,
        centroid=(2.0, 2.0),
        grid_pos=(2, 2),
        area=1,
        bounding_box=(2, 2, 2, 2),
    )
    goal = SpatialEntity(
        id="goal_1",
        role=EntityRole.GOAL,
        centroid=(2.0, 8.0),
        grid_pos=(2, 8),
        area=1,
        bounding_box=(2, 2, 8, 8),
    )
    # Solid wall at col 5 from row 0 to 9 separating agent and goal
    barriers = {(r, 5) for r in range(10)}

    eg = planner.construct_entity_graph(
        entities=[avatar, goal],
        barriers=barriers,
        grid_shape=(10, 10),
        step_size=1,
    )

    assert eg.avatar is not None
    assert eg.avatar.role == EntityRole.AGENT
    assert len(eg.cut_sets) == 1
    assert eg.cut_sets[0].is_partitioned

    doorways = [
        e for e in eg.entities.values() if e.role in (EntityRole.PORTAL, EntityRole.DOORWAY)
    ]
    assert len(doorways) == 1
    assert doorways[0].grid_pos == (2, 5)


def test_native_memory_constraint_recording_and_recall() -> None:
    planner = HCIRSpatialEntityPlanner(step_size=1)
    workspace = HCIRWorkspaceState()

    # Record failure
    planner.record_failure(
        workspace=workspace,
        session_id="test_session_1",
        failed_action=4,
        failure_pos=(5, 5),
        reason="collision",
        attempted_sequence=["step_1", "step_2"],
    )

    # Verify nodes created in HCIR workspace graph
    from hbllm.hcir.graph import HCIRNodeType

    episodes = workspace.graph.nodes_by_type(HCIRNodeType.EPISODE)
    beliefs = workspace.graph.nodes_by_type(HCIRNodeType.BELIEF)

    assert len(episodes) == 1
    assert episodes[0].reward == -1.0
    assert episodes[0].outcome == "collision"

    assert len(beliefs) == 1
    assert beliefs[0].properties.get("negative_constraint") is True
    assert beliefs[0].properties.get("position") == (5, 5)

    # Recall constraints
    impassable, energy = planner._recall_memory_constraints(workspace)
    assert (5, 5) in impassable


def test_domain_agnostic_construct_entity_graph() -> None:
    """Verifies that HCIR spatial planning works with pure abstract entities without any color information."""
    planner = HCIRSpatialEntityPlanner(step_size=1)

    # Pure abstract entities: Agent, Manipulable Item, and Receptacle (no colors!)
    agent = SpatialEntity(
        id="robot_agent",
        role=EntityRole.AGENT,
        centroid=(2.0, 2.0),
        grid_pos=(2, 2),
        area=1,
        bounding_box=(2, 2, 2, 2),
    )
    item = SpatialEntity(
        id="box_item",
        role=EntityRole.MANIPULABLE,
        centroid=(3.0, 3.0),
        grid_pos=(3, 3),
        area=1,
        bounding_box=(3, 3, 3, 3),
    )
    receptacle = SpatialEntity(
        id="delivery_bin",
        role=EntityRole.RECEPTACLE,
        centroid=(8.0, 8.0),
        grid_pos=(8, 8),
        area=4,
        bounding_box=(7, 8, 7, 8),
    )

    barriers = {(r, 5) for r in range(10)}

    eg = planner.construct_entity_graph(
        entities=[agent, item, receptacle],
        barriers=barriers,
        grid_shape=(10, 10),
        step_size=1,
    )

    assert eg.agent is not None
    assert eg.agent.id == "robot_agent"
    assert eg.agent.role == EntityRole.AGENT
    assert len(eg.cut_sets) == 1
    assert eg.cut_sets[0].is_partitioned

    portals = [e for e in eg.entities.values() if e.role == EntityRole.PORTAL]
    assert len(portals) == 1

    # Plan sequence across partitioned portal
    plan = planner.plan_sequence(eg)
    assert len(plan) == 2
    assert plan[0].action_type == "PICKUP"
    assert plan[0].target_entity_id == "box_item"
    assert plan[1].action_type == "DROP"
