"""
Safety-Gymnasium Perception Adapter.

Projects continuous 2D positions, LiDAR distances, hazard zones,
and dynamic gremlins into CognitiveGraph and risk-weighted spatial fields.
"""

from __future__ import annotations

import logging
import math

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    GoalNode,
    PhysicalEntityNode,
)
from hbllm.perception import EpistemicSpatialGrid

from .types import (
    SafetyObservation,
)

logger = logging.getLogger(__name__)


class SafetyGymPerceptionAdapter:
    """
    Ingests continuous arena observations, dynamically identifies risk contours,
    and updates CognitiveGraph with obstacles, hazards, and goal geometry.
    """

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()
        self.grid = EpistemicSpatialGrid(self.graph)

    def ingest_observation(self, obs: SafetyObservation) -> CognitiveGraph:
        """Translate SafetyObservation into typed CognitiveGraph nodes."""
        ax, ay = obs.agent_pos

        # Update Agent Node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="point_agent",
            entity_type="agent",
            properties={
                "x": ax,
                "y": ay,
                "heading": obs.agent_heading,
                "velocity": obs.agent_vel,
                "current_cost": obs.current_cost,
                "cumulative_cost": obs.cumulative_cost,
                "step_count": obs.step_count,
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node("agent"):
            ex = self.graph.get_node("agent")
            if isinstance(ex, PhysicalEntityNode):
                ex.properties.update(agent_node.properties)
        else:
            self.graph.add_node(agent_node)

        # Update Goal Node
        gx, gy = obs.goal_pos
        dist_to_goal = math.hypot(ax - gx, ay - gy)
        goal_node = PhysicalEntityNode(
            id="goal",
            entity_name="target_goal",
            entity_type="goal",
            properties={"x": gx, "y": gy, "distance": dist_to_goal},
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node("goal"):
            ex = self.graph.get_node("goal")
            if isinstance(ex, PhysicalEntityNode):
                ex.properties.update(goal_node.properties)
        else:
            self.graph.add_node(goal_node)

        # Ingest Hazards (with safety clearance buffer)
        for h in obs.hazards:
            dist = math.hypot(ax - h.x, ay - h.y)
            h_node = PhysicalEntityNode(
                id=h.id,
                entity_name="static_hazard",
                entity_type="hazard",
                properties={
                    "x": h.x,
                    "y": h.y,
                    "radius": h.radius,
                    "safe_radius": h.radius + 0.35,  # safety margin
                    "distance": dist,
                },
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            if self.graph.has_node(h.id):
                ex = self.graph.get_node(h.id)
                if isinstance(ex, PhysicalEntityNode):
                    ex.properties.update(h_node.properties)
            else:
                self.graph.add_node(h_node)

        # Ingest Dynamic Gremlins (with predictive forward projection)
        for g in obs.gremlins:
            dist = math.hypot(ax - g.x, ay - g.y)
            # Project gremlin position 3 steps ahead
            proj_x = g.x + 3 * g.vx
            proj_y = g.y + 3 * g.vy
            g_node = PhysicalEntityNode(
                id=g.id,
                entity_name="dynamic_gremlin",
                entity_type="gremlin",
                properties={
                    "x": g.x,
                    "y": g.y,
                    "vx": g.vx,
                    "vy": g.vy,
                    "projected_x": proj_x,
                    "projected_y": proj_y,
                    "radius": g.radius,
                    "safe_radius": g.radius + 0.45,
                    "distance": dist,
                },
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            if self.graph.has_node(g.id):
                ex = self.graph.get_node(g.id)
                if isinstance(ex, PhysicalEntityNode):
                    ex.properties.update(g_node.properties)
            else:
                self.graph.add_node(g_node)

        return self.graph

    def ingest_goal(self, obs: SafetyObservation | None = None) -> GoalNode:
        """Create active GoalNode requiring reaching the target goal safely."""
        goal_node = GoalNode(
            id="goal_active",
            properties={"target_conditions": ["near(goal)"]},
        )
        if self.graph.has_node("goal_active"):
            ex = self.graph.get_node("goal_active")
            if isinstance(ex, GoalNode):
                ex.properties.update(goal_node.properties)
        else:
            self.graph.add_node(goal_node)
        return goal_node
