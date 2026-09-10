"""
ALFWorld Perception Adapter.

Translates ALFWorld text observations and mission instructions into
hierarchical CognitiveGraph representation (Rooms, Receptacles, Objects, Affordances).
"""

from __future__ import annotations

import logging
import re

from hbllm.hcir.graph import (
    CognitiveGraph,
    EntityLifecycle,
    GoalNode,
    PhysicalEntityNode,
)

from .types import (
    ALFWorldGoal,
    ALFWorldObservation,
    ALFWorldTaskType,
)

logger = logging.getLogger(__name__)


class ALFWorldPerceptionAdapter:
    """
    Ingests ALFWorld natural language observations and maintains an epistemic CognitiveGraph
    of household topology, containers, object bindings, and current state.
    """

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()

    def parse_instruction(self, instruction: str) -> ALFWorldGoal:
        """Decompose natural language mission instruction into typed ALFWorldGoal."""
        inst = instruction.lower().strip()

        # 1. Clean and place
        m = re.search(r"clean (.+?) with (.+?) and put (?:it )?(?:on|in) (.+)", inst)
        if m:
            return ALFWorldGoal(
                task_type=ALFWorldTaskType.CLEAN_AND_PLACE,
                target_object_type=m.group(1).strip(),
                apparatus_receptacle_type=m.group(2).strip(),
                target_receptacle_type=m.group(3).strip(),
                raw_instruction=instruction,
            )

        # 2. Heat and place
        m = re.search(r"heat (.+?) with (.+?) and put (?:it )?(?:on|in) (.+)", inst)
        if m:
            return ALFWorldGoal(
                task_type=ALFWorldTaskType.HEAT_AND_PLACE,
                target_object_type=m.group(1).strip(),
                apparatus_receptacle_type=m.group(2).strip(),
                target_receptacle_type=m.group(3).strip(),
                raw_instruction=instruction,
            )

        # 3. Cool and place
        m = re.search(r"cool (.+?) with (.+?) and put (?:it )?(?:on|in) (.+)", inst)
        if m:
            return ALFWorldGoal(
                task_type=ALFWorldTaskType.COOL_AND_PLACE,
                target_object_type=m.group(1).strip(),
                apparatus_receptacle_type=m.group(2).strip(),
                target_receptacle_type=m.group(3).strip(),
                raw_instruction=instruction,
            )

        # 4. Examine in light
        m = re.search(r"examine (.+?) with (.+)", inst)
        if m:
            return ALFWorldGoal(
                task_type=ALFWorldTaskType.EXAMINE_IN_LIGHT,
                target_object_type=m.group(1).strip(),
                apparatus_receptacle_type=m.group(2).strip(),
                raw_instruction=instruction,
            )

        # 5. Pick two and place
        m = re.search(r"put two (.+?)s? (?:on|in) (.+)", inst)
        if m:
            return ALFWorldGoal(
                task_type=ALFWorldTaskType.PICK_TWO_AND_PLACE,
                target_object_type=m.group(1).strip(),
                target_receptacle_type=m.group(2).strip(),
                count_required=2,
                raw_instruction=instruction,
            )

        # 6. Pick and place
        m = re.search(r"put (?:a|an) (.+?) (?:on|in) (.+)", inst)
        if m:
            return ALFWorldGoal(
                task_type=ALFWorldTaskType.PICK_AND_PLACE,
                target_object_type=m.group(1).strip(),
                target_receptacle_type=m.group(2).strip(),
                raw_instruction=instruction,
            )

        # Default fallback
        return ALFWorldGoal(
            task_type=ALFWorldTaskType.PICK_AND_PLACE,
            target_object_type="object",
            target_receptacle_type="countertop 1",
            raw_instruction=instruction,
        )

    def ingest_observation(self, obs: ALFWorldObservation) -> CognitiveGraph:
        """Update CognitiveGraph with agent status, visible objects, and receptacles."""
        # Update agent node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="agent",
            entity_type="agent",
            properties={
                "current_location": obs.current_location,
                "inventory": list(obs.inventory),
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

        # Parse text observation for seen objects
        # e.g., "On/in the cabinet 1, you see: soapbar 1, pen 2."
        m = re.search(r"On/in the (.+?), you see: (.+?)\.", obs.text_obs)
        if m:
            rec_name = m.group(1).strip()
            obj_list_str = m.group(2).strip()
            obj_ids = [o.strip() for o in obj_list_str.split(",") if o.strip()]

            # Record receptacle node
            rec_node = PhysicalEntityNode(
                id=f"rec_{rec_name}",
                entity_name=rec_name,
                entity_type="receptacle",
                properties={"name": rec_name, "contained": obj_ids},
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            if self.graph.has_node(rec_node.id):
                ex = self.graph.get_node(rec_node.id)
                if isinstance(ex, PhysicalEntityNode):
                    ex.properties.update(rec_node.properties)
            else:
                self.graph.add_node(rec_node)

            # Record contained object nodes
            for oid in obj_ids:
                obj_node = PhysicalEntityNode(
                    id=f"obj_{oid}",
                    entity_name=oid,
                    entity_type="object",
                    properties={"name": oid, "parent_receptacle": rec_name},
                    entity_lifecycle=EntityLifecycle.TRACKED,
                )
                if self.graph.has_node(obj_node.id):
                    ex = self.graph.get_node(obj_node.id)
                    if isinstance(ex, PhysicalEntityNode):
                        ex.properties.update(obj_node.properties)
                else:
                    self.graph.add_node(obj_node)

        return self.graph

    def ingest_goal(self, goal: ALFWorldGoal) -> GoalNode:
        """Create active GoalNode representing household task objective."""
        target_conditions: list[str] = []
        if goal.target_receptacle_type and goal.target_object_type:
            target_conditions.append(
                f"inside({goal.target_object_type}, {goal.target_receptacle_type})"
            )
        elif goal.target_object_type:
            target_conditions.append(f"holds({goal.target_object_type})")
        else:
            target_conditions.append("goal_achieved")

        goal_node = GoalNode(
            id="goal_active",
            properties={
                "target_conditions": target_conditions,
                "task_type": goal.task_type.value,
                "target_object_type": goal.target_object_type,
                "target_receptacle_type": goal.target_receptacle_type,
            },
        )
        if self.graph.has_node("goal_active"):
            ex = self.graph.get_node("goal_active")
            if isinstance(ex, GoalNode):
                ex.properties.update(goal_node.properties)
        else:
            self.graph.add_node(goal_node)
        return goal_node
