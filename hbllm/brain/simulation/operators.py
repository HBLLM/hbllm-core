"""State-Transition Action Operators for A18.

Implements deterministic state transitions:
Precondition Check -> State Transition -> Derived Consequences -> Constraint Violations -> New State.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.simulation.branch import SimulationBranch
from hbllm.brain.simulation.events import SimulationEvent
from hbllm.brain.simulation.geometry import (
    BoundingBox,
    evaluate_support_stability,
    is_path_clear,
)
from hbllm.hcir.graph import HCIREdge, HCIREdgeType, PhysicalEntityNode

logger = logging.getLogger(__name__)


@dataclass
class OperatorExecutionResult:
    """Result of applying a state-transition operator on a SimulationBranch."""

    operator_name: str
    is_success: bool
    pre_state_hash: str
    post_state_hash: str
    consequences: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    risk: float = 0.0
    confidence: float = 1.0
    reason: str = ""


class ActionOperator(ABC):
    """Abstract base class for deterministic action operators in simulation."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(
        self, branch: SimulationBranch, params: dict[str, Any], step: int = 0
    ) -> OperatorExecutionResult:
        pass


# ── 1. Push Operator ──────────────────────────────────────────────────


class PushOperator(ActionOperator):
    """Pushes an entity along a 2D vector, detecting obstacle collisions."""

    @property
    def name(self) -> str:
        return "PUSH"

    def execute(
        self, branch: SimulationBranch, params: dict[str, Any], step: int = 0
    ) -> OperatorExecutionResult:
        pre_hash = branch.compute_current_state_hash()
        target_id = str(params.get("target_id", ""))
        dx = float(params.get("dx", 0.0))
        dy = float(params.get("dy", 0.0))

        entity = branch.get_node(target_id)
        if not isinstance(entity, PhysicalEntityNode):
            return OperatorExecutionResult(
                operator_name=self.name,
                is_success=False,
                pre_state_hash=pre_hash,
                post_state_hash=pre_hash,
                violations=["entity_not_found"],
                risk=0.8,
                reason=f"Target entity {target_id} not found",
            )

        props = dict(
            entity.properties
            if hasattr(entity, "properties") and isinstance(entity.properties, dict)
            else entity.observed_properties
        )
        curr_x = float(props.get("x", 0.0))
        curr_y = float(props.get("y", 0.0))
        new_x = curr_x + dx
        new_y = curr_y + dy

        # Check collision with other obstacles in branch
        obstacles: list[tuple[str, BoundingBox]] = []
        for node in branch.all_nodes():
            if node.id != target_id and isinstance(node, PhysicalEntityNode):
                nprops = (
                    getattr(node, "properties", None)
                    or getattr(node, "observed_properties", {})
                    or {}
                )
                if nprops.get("is_obstacle", False) or node.entity_type in (
                    "wall",
                    "obstacle",
                    "barrier",
                ):
                    ox = float(nprops.get("x", 0.0))
                    oy = float(nprops.get("y", 0.0))
                    w = float(nprops.get("width", 1.0))
                    d = float(nprops.get("depth", 1.0))
                    obstacles.append((node.id, BoundingBox(x=ox, y=oy, width=w, depth=d)))

        clear, colliding_id = is_path_clear((curr_x, curr_y), (new_x, new_y), obstacles)
        if not clear:
            return OperatorExecutionResult(
                operator_name=self.name,
                is_success=False,
                pre_state_hash=pre_hash,
                post_state_hash=pre_hash,
                violations=[f"collision_with_{colliding_id}"],
                risk=0.85,
                reason=f"Push blocked: collision with obstacle {colliding_id}",
            )

        # Apply state transition
        props["x"] = new_x
        props["y"] = new_y
        entity.properties = props
        branch.upsert_node(entity)

        post_hash = branch.compute_current_state_hash()
        event = SimulationEvent(
            branch_id=branch.branch_id,
            step=step,
            operator=self.name,
            parameters=params,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            consequences=[f"{target_id}_displaced_to_({new_x},{new_y})"],
            confidence=branch.confidence,
            risk=0.05,
        )
        branch.record_event(event)

        return OperatorExecutionResult(
            operator_name=self.name,
            is_success=True,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            consequences=event.consequences,
            confidence=branch.confidence,
            risk=0.05,
            reason="push_completed",
        )


# ── 2. Stack Operator ─────────────────────────────────────────────────


class StackOperator(ActionOperator):
    """Places an object on top of a base object, evaluating geometric support stability."""

    @property
    def name(self) -> str:
        return "STACK"

    def execute(
        self, branch: SimulationBranch, params: dict[str, Any], step: int = 0
    ) -> OperatorExecutionResult:
        pre_hash = branch.compute_current_state_hash()
        item_id = str(params.get("item_id", ""))
        base_id = str(params.get("base_id", ""))

        item = branch.get_node(item_id)
        base = branch.get_node(base_id)

        if not isinstance(item, PhysicalEntityNode) or not isinstance(base, PhysicalEntityNode):
            return OperatorExecutionResult(
                operator_name=self.name,
                is_success=False,
                pre_state_hash=pre_hash,
                post_state_hash=pre_hash,
                violations=["entity_not_found"],
                risk=0.9,
                reason="Item or base entity not found",
            )

        item_props = (
            getattr(item, "properties", None) or getattr(item, "observed_properties", {}) or {}
        )
        base_props = (
            getattr(base, "properties", None) or getattr(base, "observed_properties", {}) or {}
        )

        # Evaluate support geometry stability
        is_stable, stability_score, reason = evaluate_support_stability(
            supported_props=item_props,
            supporting_props=base_props,
            supporting_type=base.entity_type,
        )

        consequences = []
        violations = []
        risk = 0.05

        if is_stable:
            # Add spatial support edge
            edge = HCIREdge(edge_type=HCIREdgeType.LOCATED_IN, sources=[item_id], targets=[base_id])
            branch.add_edge(edge)
            consequences.append(f"{item_id}_stable_on_{base_id}")
        else:
            # Derived consequence: object rolls/falls off unstable curved surface
            violations.append("unstable_support_fall")
            consequences.append(f"{item_id}_fell_off_{base_id}")
            risk = 0.92  # High risk of falling

        post_hash = branch.compute_current_state_hash()
        event = SimulationEvent(
            branch_id=branch.branch_id,
            step=step,
            operator=self.name,
            parameters=params,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            consequences=consequences,
            violations=violations,
            confidence=branch.confidence,
            risk=risk,
        )
        branch.record_event(event)

        return OperatorExecutionResult(
            operator_name=self.name,
            is_success=is_stable,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            consequences=consequences,
            violations=violations,
            confidence=branch.confidence,
            risk=risk,
            reason=reason,
        )


# ── 3. PutIn Operator ─────────────────────────────────────────────────


class PutInOperator(ActionOperator):
    """Places an object inside a container, validating containment constraints."""

    @property
    def name(self) -> str:
        return "PUT_IN"

    def execute(
        self, branch: SimulationBranch, params: dict[str, Any], step: int = 0
    ) -> OperatorExecutionResult:
        pre_hash = branch.compute_current_state_hash()
        item_id = str(params.get("item_id", ""))
        container_id = str(params.get("container_id", ""))

        item = branch.get_node(item_id)
        container = branch.get_node(container_id)

        if not isinstance(item, PhysicalEntityNode) or not isinstance(
            container, PhysicalEntityNode
        ):
            return OperatorExecutionResult(
                operator_name=self.name,
                is_success=False,
                pre_state_hash=pre_hash,
                post_state_hash=pre_hash,
                violations=["entity_not_found"],
                risk=0.8,
                reason="Item or container entity not found",
            )

        cont_props = (
            getattr(container, "properties", None)
            or getattr(container, "observed_properties", {})
            or {}
        )
        # Precondition check: container must be open
        is_closed = cont_props.get("is_closed", False)
        if is_closed:
            return OperatorExecutionResult(
                operator_name=self.name,
                is_success=False,
                pre_state_hash=pre_hash,
                post_state_hash=pre_hash,
                violations=["container_closed"],
                risk=0.7,
                reason=f"Container {container_id} is closed",
            )

        # Apply containment edge
        edge = HCIREdge(
            edge_type=HCIREdgeType.LOCATED_IN, sources=[item_id], targets=[container_id]
        )
        branch.add_edge(edge)

        post_hash = branch.compute_current_state_hash()
        event = SimulationEvent(
            branch_id=branch.branch_id,
            step=step,
            operator=self.name,
            parameters=params,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            consequences=[f"{item_id}_contained_in_{container_id}"],
            confidence=branch.confidence,
            risk=0.02,
        )
        branch.record_event(event)

        return OperatorExecutionResult(
            operator_name=self.name,
            is_success=True,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            consequences=event.consequences,
            confidence=branch.confidence,
            risk=0.02,
            reason="contained_successfully",
        )


# ── 4. Move Operator ──────────────────────────────────────────────────


class MoveOperator(ActionOperator):
    """Navigates an agent or entity along a waypoint path to a target position."""

    @property
    def name(self) -> str:
        return "MOVE"

    def execute(
        self, branch: SimulationBranch, params: dict[str, Any], step: int = 0
    ) -> OperatorExecutionResult:
        pre_hash = branch.compute_current_state_hash()
        entity_id = str(params.get("entity_id", "agent"))
        target_pos = (float(params.get("target_x", 0.0)), float(params.get("target_y", 0.0)))

        entity = branch.get_node(entity_id)
        if not isinstance(entity, PhysicalEntityNode):
            # Create agent proxy if moving self
            entity = PhysicalEntityNode(
                id=entity_id, entity_type="agent", properties={"x": 0.0, "y": 0.0}
            )
            branch.upsert_node(entity)

        props = dict(
            getattr(entity, "properties", None) or getattr(entity, "observed_properties", {}) or {}
        )
        start_pos = (float(props.get("x", 0.0)), float(props.get("y", 0.0)))

        # Collect obstacles in simulation branch
        obstacles: list[tuple[str, BoundingBox]] = []
        for node in branch.all_nodes():
            if node.id != entity_id and isinstance(node, PhysicalEntityNode):
                nprops = (
                    getattr(node, "properties", None)
                    or getattr(node, "observed_properties", {})
                    or {}
                )
                if nprops.get("is_obstacle", True) and node.entity_type in (
                    "wall",
                    "box",
                    "obstacle",
                ):
                    ox = float(nprops.get("x", 0.0))
                    oy = float(nprops.get("y", 0.0))
                    w = float(nprops.get("width", 1.0))
                    d = float(nprops.get("depth", 1.0))
                    obstacles.append((node.id, BoundingBox(x=ox, y=oy, width=w, depth=d)))

        # Evaluate path clearance
        clear, colliding_id = is_path_clear(start_pos, target_pos, obstacles)
        if not clear:
            return OperatorExecutionResult(
                operator_name=self.name,
                is_success=False,
                pre_state_hash=pre_hash,
                post_state_hash=pre_hash,
                violations=[f"path_blocked_by_{colliding_id}"],
                risk=0.90,
                reason=f"Trajectory blocked by obstacle {colliding_id}",
            )

        # Apply position update
        props["x"] = target_pos[0]
        props["y"] = target_pos[1]
        entity.properties = props
        branch.upsert_node(entity)

        post_hash = branch.compute_current_state_hash()
        event = SimulationEvent(
            branch_id=branch.branch_id,
            step=step,
            operator=self.name,
            parameters=params,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            consequences=[f"{entity_id}_moved_to_{target_pos}"],
            confidence=branch.confidence,
            risk=0.02,
        )
        branch.record_event(event)

        return OperatorExecutionResult(
            operator_name=self.name,
            is_success=True,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            consequences=event.consequences,
            confidence=branch.confidence,
            risk=0.02,
            reason="move_completed",
        )
