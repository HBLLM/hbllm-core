"""
Stochastic Perception Adapter.

Provides epistemic belief maintenance, object permanence filtering under sensory
dropout, sensorimotor surprise detection, and HCIR CognitiveGraph generation.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    GoalNode,
    PhysicalEntityNode,
)

from .types import EpistemicEntityBelief, StochasticObservation

logger = logging.getLogger(__name__)


class StochasticPerceptionAdapter:
    """Maintains epistemic state tracking and surprise detection under noise."""

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()
        self.target_belief: EpistemicEntityBelief | None = None
        self.expected_player_pos: tuple[int, int] | None = None
        self.known_obstacles: set[tuple[int, int]] = set()

    def reset(self) -> None:
        """Reset belief tracking and graph."""
        self.graph = CognitiveGraph()
        self.target_belief = None
        self.expected_player_pos = None
        self.known_obstacles.clear()

    def register_expected_transition(self, expected_pos: tuple[int, int]) -> None:
        """Register forward model expectation for next step."""
        self.expected_player_pos = expected_pos

    def ingest_observation(self, obs: StochasticObservation) -> CognitiveGraph:
        """Ingest noisy observation and build/update HCIR CognitiveGraph."""
        surprise_detected = False
        if self.expected_player_pos is not None:
            if obs.player_pos != self.expected_player_pos:
                surprise_detected = True

        # Accumulate obstacle knowledge
        for obs_pos in obs.obstacles:
            self.known_obstacles.add(obs_pos)

        # Target belief tracking
        if obs.target_pos is not None:
            self.target_belief = EpistemicEntityBelief(
                entity_id="target_goal",
                estimated_pos=obs.target_pos,
                confidence=1.0,
                last_seen_step=obs.step_count,
                occluded=False,
            )
        elif self.target_belief is not None:
            steps_since = obs.step_count - self.target_belief.last_seen_step
            decayed_conf = max(0.2, 1.0 - (0.05 * steps_since))
            self.target_belief.confidence = decayed_conf
            self.target_belief.occluded = True

        # 1. Agent node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="player",
            entity_type="agent",
            properties={
                "pos": obs.player_pos,
                "x": obs.player_pos[0],
                "y": obs.player_pos[1],
                "step_count": obs.step_count,
                "surprise_detected": surprise_detected,
                "was_slipped": obs.was_slipped,
                "was_occluded": obs.was_occluded,
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node("agent"):
            ex = self.graph.get_node("agent")
            if isinstance(ex, PhysicalEntityNode):
                ex.properties.update(agent_node.properties)
        else:
            self.graph.add_node(agent_node)

        # 2. Target Goal node
        if self.target_belief is not None:
            tgt_node = PhysicalEntityNode(
                id="target_goal",
                entity_name="target",
                entity_type="target",
                properties={
                    "pos": self.target_belief.estimated_pos,
                    "x": self.target_belief.estimated_pos[0],
                    "y": self.target_belief.estimated_pos[1],
                    "confidence": self.target_belief.confidence,
                    "occluded": self.target_belief.occluded,
                    "last_seen": self.target_belief.last_seen_step,
                },
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            if self.graph.has_node("target_goal"):
                ex = self.graph.get_node("target_goal")
                if isinstance(ex, PhysicalEntityNode):
                    ex.properties.update(tgt_node.properties)
            else:
                self.graph.add_node(tgt_node)

        # 3. Obstacle nodes
        for r, c in self.known_obstacles:
            obs_id = f"obstacle_{r}_{c}"
            if not self.graph.has_node(obs_id):
                obs_node = PhysicalEntityNode(
                    id=obs_id,
                    entity_name=f"obstacle_{r}_{c}",
                    entity_type="obstacle",
                    properties={"pos": (r, c), "x": r, "y": c},
                    entity_lifecycle=EntityLifecycle.TRACKED,
                )
                self.graph.add_node(obs_node)

        return self.graph

    def ingest_goal(self, goal_spec: str | None = None) -> GoalNode:
        """Create active GoalNode representing navigation objective."""
        conditions = [goal_spec] if goal_spec else ["at_target(agent, target_goal)"]
        goal_node = GoalNode(
            id="goal_active",
            properties={
                "target_conditions": conditions,
            },
        )
        if self.graph.has_node("goal_active"):
            ex = self.graph.get_node("goal_active")
            if isinstance(ex, GoalNode):
                ex.properties.update(goal_node.properties)
        else:
            self.graph.add_node(goal_node)
        return goal_node

    def process_observation(self, obs: StochasticObservation) -> dict[str, Any]:
        """Ingest noisy observation and perform epistemic belief update."""
        self.ingest_observation(obs)

        surprise_detected = False
        if self.expected_player_pos is not None:
            if obs.player_pos != self.expected_player_pos:
                surprise_detected = True

        return {
            "player_pos": obs.player_pos,
            "target_belief": self.target_belief,
            "target_pos": self.target_belief.estimated_pos if self.target_belief else None,
            "obstacles": set(self.known_obstacles),
            "surprise_detected": surprise_detected,
            "was_slipped": obs.was_slipped,
            "was_occluded": obs.was_occluded,
            "grid": obs.grid,
            "step_count": obs.step_count,
        }
