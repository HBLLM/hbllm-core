"""
Digital Agent Perception Adapter.

Parses stdout/stderr streams, virtual filesystem entries, and DOM trees into
structured epistemic states, diagnostic features, and HCIR CognitiveGraph representations.
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

from .types import DigitalObservation, DOMNode

logger = logging.getLogger(__name__)


class DigitalPerceptionAdapter:
    """Extracts digital environment features, errors, and UI affordances into CognitiveGraph."""

    def __init__(self, graph: CognitiveGraph | None = None) -> None:
        self.graph = graph if graph is not None else CognitiveGraph()

    def reset(self) -> None:
        """Reset internal states and graph."""
        self.graph = CognitiveGraph()

    def ingest_observation(self, obs: DigitalObservation) -> CognitiveGraph:
        """Ingest digital observation into HCIR CognitiveGraph."""
        has_error = obs.exit_code != 0 or bool(obs.stderr)
        is_assertion_error = "AssertionError" in obs.stderr

        # 1. Agent node
        agent_node = PhysicalEntityNode(
            id="agent",
            entity_name="digital_agent",
            entity_type="agent",
            properties={
                "stdout": obs.stdout,
                "stderr": obs.stderr,
                "exit_code": obs.exit_code,
                "has_error": has_error,
                "is_assertion_error": is_assertion_error,
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

        # 2. Filesystem nodes
        for path, content in obs.filesystem.items():
            f_node = PhysicalEntityNode(
                id=f"file_{path}",
                entity_name=path,
                entity_type="file",
                properties={"path": path, "content": content, "length": len(content)},
                entity_lifecycle=EntityLifecycle.TRACKED,
            )
            if self.graph.has_node(f_node.id):
                ex = self.graph.get_node(f_node.id)
                if isinstance(ex, PhysicalEntityNode):
                    ex.properties.update(f_node.properties)
            else:
                self.graph.add_node(f_node)

        # 3. DOM interactive nodes
        if obs.active_dom:
            self._ingest_dom_node(obs.active_dom)

        return self.graph

    def _ingest_dom_node(self, node: DOMNode) -> None:
        """Recursively add DOM elements to CognitiveGraph."""
        dom_node = PhysicalEntityNode(
            id=f"dom_{node.node_id}",
            entity_name=node.node_id,
            entity_type="dom_element",
            properties={
                "tag": node.tag,
                "node_id": node.node_id,
                "clickable": node.clickable,
                "text": node.text,
                "value": node.value,
                "attributes": node.attributes,
            },
            entity_lifecycle=EntityLifecycle.TRACKED,
        )
        if self.graph.has_node(dom_node.id):
            ex = self.graph.get_node(dom_node.id)
            if isinstance(ex, PhysicalEntityNode):
                ex.properties.update(dom_node.properties)
        else:
            self.graph.add_node(dom_node)

        for child in node.children:
            self._ingest_dom_node(child)

    def ingest_goal(self, goal_spec: str | None = None) -> GoalNode:
        """Create active GoalNode representing digital operational goal."""
        conditions = [goal_spec] if goal_spec else ["task_completed"]
        goal_node = GoalNode(
            id="goal_active",
            properties={
                "target_conditions": conditions,
                "goal_spec": goal_spec,
            },
        )
        if self.graph.has_node("goal_active"):
            ex = self.graph.get_node("goal_active")
            if isinstance(ex, GoalNode):
                ex.properties.update(goal_node.properties)
        else:
            self.graph.add_node(goal_node)
        return goal_node

    def process_observation(self, obs: DigitalObservation) -> dict[str, Any]:
        """Convert observation into structured perceptual state."""
        self.ingest_observation(obs)

        has_error = obs.exit_code != 0 or bool(obs.stderr)
        is_assertion_error = "AssertionError" in obs.stderr

        dom_inputs = []
        dom_buttons = []
        if obs.active_dom:
            self._extract_dom_affordances(obs.active_dom, dom_inputs, dom_buttons)

        return {
            "stdout": obs.stdout,
            "stderr": obs.stderr,
            "exit_code": obs.exit_code,
            "has_error": has_error,
            "is_assertion_error": is_assertion_error,
            "files": list(obs.filesystem.keys()),
            "dom_inputs": dom_inputs,
            "dom_buttons": dom_buttons,
            "step_count": obs.step_count,
        }

    def _extract_dom_affordances(
        self,
        node: DOMNode,
        inputs: list[str],
        buttons: list[str],
    ) -> None:
        """Recursively collect interactive form elements."""
        if node.tag == "input":
            inputs.append(node.node_id)
        elif node.tag == "button" or node.clickable:
            buttons.append(node.node_id)
        for child in node.children:
            self._extract_dom_affordances(child, inputs, buttons)
