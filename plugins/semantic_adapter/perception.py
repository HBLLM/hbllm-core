"""
Semantic Ambiguity Perception Adapter.

Parses natural language directives, estimates linguistic grounding confidence,
detects epistemic ambiguity, and constructs HCIR CognitiveGraphs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    GoalNode,
    PhysicalEntityNode,
)

from .types import SemanticObservation

logger = logging.getLogger(__name__)

AMBIGUITY_MARKERS = [
    r"\bscratch that\b",
    r"\bwait\b",
    r"\bactually\b",
    r"\btidy up\b",
    r"\bprepare\b",
    r"\bclean\b",
    r"\bmake ready\b",
]


class SemanticPerceptionAdapter:
    """Evaluates linguistic ambiguity and builds epistemic CognitiveGraph for semantic directives."""

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()

    def reset(self) -> None:
        """Reset internal buffers and graph."""
        self.graph = CognitiveGraph()

    def ingest_observation(self, obs: SemanticObservation) -> CognitiveGraph:
        """Ingest semantic observation into HCIR CognitiveGraph."""
        text = obs.instruction.lower()

        ambiguity_detected = False
        for pattern in AMBIGUITY_MARKERS:
            if re.search(pattern, text):
                ambiguity_detected = True
                break

        is_canonical = (
            bool(re.search(r"\b(pick up|unlock|place)\b", text)) and not ambiguity_detected
        )
        confidence = 1.0 if is_canonical else (0.4 if ambiguity_detected else 0.8)

        # 1. Agent node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="semantic_agent",
            entity_type="agent",
            properties={
                "instruction": obs.instruction,
                "is_canonical": is_canonical,
                "ambiguity_detected": ambiguity_detected,
                "grounding_confidence": confidence,
                "completed_subgoals": list(obs.completed_subgoals),
                "pending_subgoals": list(obs.pending_subgoals),
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

        # 2. Scene Object nodes
        for obj_name, obj_data in obs.scene_objects.items():
            node_id = f"obj_{obj_name}"
            obj_node = PhysicalEntityNode(
                id=node_id,
                entity_name=obj_name,
                entity_type="object",
                properties={"name": obj_name, **obj_data}
                if isinstance(obj_data, dict)
                else {"name": obj_name, "val": obj_data},
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            if self.graph.has_node(node_id):
                ex = self.graph.get_node(node_id)
                if isinstance(ex, PhysicalEntityNode):
                    ex.properties.update(obj_node.properties)
            else:
                self.graph.add_node(obj_node)

        return self.graph

    def ingest_goal(self, goal_spec: str | None = None) -> GoalNode:
        """Create active GoalNode representing semantic directive completion."""
        conditions = [goal_spec] if goal_spec else ["semantic_directive_completed"]
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

    def process_observation(self, obs: SemanticObservation) -> dict[str, Any]:
        """Classify instruction clarity and extract candidate entities."""
        self.ingest_observation(obs)

        text = obs.instruction.lower()
        ambiguity_detected = False
        for pattern in AMBIGUITY_MARKERS:
            if re.search(pattern, text):
                ambiguity_detected = True
                break

        is_canonical = (
            bool(re.search(r"\b(pick up|unlock|place)\b", text)) and not ambiguity_detected
        )
        confidence = 1.0 if is_canonical else (0.4 if ambiguity_detected else 0.8)

        return {
            "instruction": obs.instruction,
            "is_canonical": is_canonical,
            "ambiguity_detected": ambiguity_detected,
            "grounding_confidence": confidence,
            "scene_objects": dict(obs.scene_objects),
            "step_count": obs.step_count,
        }
