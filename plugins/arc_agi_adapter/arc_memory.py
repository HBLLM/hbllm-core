"""ARC-AGI Cross-Game Memory — Persistent knowledge transfer across episodes and environments.

This module manages the accumulation and transfer of learned knowledge (action dynamics,
barrier colors, shape archetypes, walkable cells) across game instances. It enables
zero-shot transfer to new environments within the same ARC-AGI session.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies — these are resolved at runtime
# when methods are actually called, not at module import time.


class AgentPhase(str, Enum):
    """Cognitive lifecycle phases for autonomous spatial reasoning."""

    EPISTEMIC_LEARNING = "EPISTEMIC_LEARNING"
    SOFT_RESTART_PENDING = "SOFT_RESTART_PENDING"
    OPTIMAL_EXECUTION = "OPTIMAL_EXECUTION"
    COMPLETED = "COMPLETED"


@dataclass
class HCIRCrossGameMemory:
    """Persistent cross-game and cross-level HCIR memory store.

    Accumulates learned knowledge (action dynamics, color classifications,
    shape concepts, collision constraints) and transfers them zero-shot
    to new or reset agents.
    """

    action_models: dict[int, Any] = field(default_factory=dict)
    avatar_color: int | None = None
    step_size: int = 1
    action_5_affordance: str = "UNKNOWN"
    learned_barrier_colors: set[int] = field(default_factory=set)
    learned_walkable_colors: set[int] = field(default_factory=set)
    learned_item_colors: set[int] = field(default_factory=set)
    learned_receptacle_colors: set[int] = field(default_factory=set)
    negative_constraints: set[tuple[int, int]] = field(default_factory=set)
    shape_concepts: dict[tuple[tuple[int, int], ...], Any] = field(default_factory=dict)
    successful_episodes: list[dict[str, Any]] = field(default_factory=list)
    failed_episodes: list[dict[str, Any]] = field(default_factory=list)

    def transfer_to_agent(self, agent: Any) -> None:
        """Transfer accumulated knowledge zero-shot to a new or reset agent."""
        from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

        if self.avatar_color is not None and agent.avatar_color is None:
            agent.avatar_color = self.avatar_color
        if self.step_size > 1:
            agent.step_size = max(agent.step_size, self.step_size)
            agent.spatial_planner.step_size = agent.step_size
        for a, m in self.action_models.items():
            if a not in agent.action_models:
                agent.action_models[a] = ActionDynamicsModel(
                    action_id=m.action_id,
                    delta_r=m.delta_r,
                    delta_c=m.delta_c,
                    confidence=m.confidence,
                    probes_tested=m.probes_tested,
                )
        if self.action_5_affordance != "UNKNOWN":
            agent.action_5_affordance = self.action_5_affordance
        agent.learned_barrier_colors.update(self.learned_barrier_colors)
        agent.learned_walkable_colors.update(self.learned_walkable_colors)
        agent.learned_item_colors.update(self.learned_item_colors)
        agent.learned_receptacle_colors.update(self.learned_receptacle_colors)
        for p in self.negative_constraints:
            agent.spatial_planner.record_collision_barrier(p)
            agent.spatial_planner.record_failure(
                workspace=agent.workspace,
                session_id="transfer",
                failed_action=0,
                failure_pos=p,
                reason="transferred_constraint",
            )
        if hasattr(agent, "shape_concepts"):
            agent.shape_concepts.update(copy.deepcopy(self.shape_concepts))

    def update_from_agent(self, agent: Any) -> None:
        """Harvest newly discovered concepts and affordances from agent."""
        from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

        if agent.avatar_color is not None:
            self.avatar_color = agent.avatar_color
        if agent.step_size > 1:
            self.step_size = agent.step_size
        for a, m in agent.action_models.items():
            if m.confidence >= 0.8:
                self.action_models[a] = ActionDynamicsModel(
                    action_id=m.action_id,
                    delta_r=m.delta_r,
                    delta_c=m.delta_c,
                    confidence=m.confidence,
                    probes_tested=m.probes_tested,
                )
        if agent.action_5_affordance != "UNKNOWN":
            self.action_5_affordance = agent.action_5_affordance
        self.learned_barrier_colors.update(agent.learned_barrier_colors)
        self.learned_walkable_colors.update(agent.learned_walkable_colors)
        self.learned_item_colors.update(agent.learned_item_colors)
        self.learned_receptacle_colors.update(agent.learned_receptacle_colors)
        for p in getattr(agent.spatial_planner, "_learned_barriers", set()):
            self.negative_constraints.add(p)
        if hasattr(agent, "shape_concepts"):
            self.shape_concepts.update(copy.deepcopy(agent.shape_concepts))

    def record_success(
        self,
        session_id: str,
        score: float = 1.0,
        steps: int = 0,
        metrics: dict[str, Any] | None = None,
        tasks: list[str] | None = None,
    ) -> None:
        """Register a validated winning trajectory into persistent episodic memory."""
        self.successful_episodes.append(
            {
                "session_id": session_id,
                "score": score,
                "steps": steps,
                "tasks": tasks or [],
                "metrics": metrics or {},
                "avatar_color": self.avatar_color,
                "step_size": self.step_size,
            }
        )

    def record_failure(
        self,
        session_id: str,
        reason: str,
        failed_action: int,
        failure_pos: tuple[int, int],
    ) -> None:
        """Register failure and update collision/negative constraints."""
        self.failed_episodes.append(
            {
                "session_id": session_id,
                "reason": reason,
                "failed_action": failed_action,
                "failure_pos": failure_pos,
            }
        )
        if reason == "collision" and failure_pos != (0, 0):
            self.negative_constraints.add(failure_pos)

    def export_dict(self) -> dict[str, Any]:
        """Export serialized dictionary for cross-game transfer."""
        return {
            "avatar_color": self.avatar_color,
            "step_size": self.step_size,
            "action_5_affordance": self.action_5_affordance,
            "learned_barrier_colors": list(self.learned_barrier_colors),
            "learned_walkable_colors": list(self.learned_walkable_colors),
            "learned_item_colors": list(self.learned_item_colors),
            "learned_receptacle_colors": list(self.learned_receptacle_colors),
            "negative_constraints": [list(p) for p in self.negative_constraints],
            "action_models": {
                a: {
                    "action_id": m.action_id,
                    "delta_r": m.delta_r,
                    "delta_c": m.delta_c,
                    "confidence": m.confidence,
                    "probes_tested": m.probes_tested,
                }
                for a, m in self.action_models.items()
            },
            "shape_concepts": {
                str(list(k)): {
                    "canonical_id": list(k),
                    "canonical_name": v.canonical_name,
                    "inferred_role": v.inferred_role.value
                    if hasattr(v.inferred_role, "value")
                    else str(v.inferred_role),
                    "observed_colors": list(v.observed_colors),
                    "is_rotatable": v.is_rotatable,
                    "rotation_trigger": v.rotation_trigger,
                    "is_color_switch": v.is_color_switch,
                    "passable_colors": list(v.passable_colors),
                    "barrier_colors": list(v.barrier_colors),
                }
                for k, v in self.shape_concepts.items()
            },
            "successful_episodes": self.successful_episodes,
            "failed_episodes": self.failed_episodes,
        }

    def import_dict(self, data: dict[str, Any]) -> None:
        """Import knowledge from serialized dictionary."""
        from hbllm.hcir.spatial_planner import EntityRole
        from hbllm.hcir.world.morphology import MorphologicalConcept, ShapeArchetype
        from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

        self.avatar_color = data.get("avatar_color", self.avatar_color)
        self.step_size = data.get("step_size", self.step_size)
        self.action_5_affordance = data.get("action_5_affordance", self.action_5_affordance)
        self.learned_barrier_colors.update(data.get("learned_barrier_colors", []))
        self.learned_walkable_colors.update(data.get("learned_walkable_colors", []))
        self.learned_item_colors.update(data.get("learned_item_colors", []))
        self.learned_receptacle_colors.update(data.get("learned_receptacle_colors", []))
        self.negative_constraints.update(tuple(p) for p in data.get("negative_constraints", []))
        for a_str, md in data.get("action_models", {}).items():
            a = int(a_str)
            self.action_models[a] = ActionDynamicsModel(
                action_id=md["action_id"],
                delta_r=md["delta_r"],
                delta_c=md["delta_c"],
                confidence=md["confidence"],
                probes_tested=md.get("probes_tested", 1),
            )
        for sc_data in data.get("shape_concepts", {}).values():
            cid = tuple(tuple(p) for p in sc_data["canonical_id"])
            role_str = sc_data.get("inferred_role", "unknown")
            try:
                role = EntityRole(role_str)
            except ValueError:
                role = EntityRole.UNKNOWN
            arch = ShapeArchetype.from_coords(set(cid))
            self.shape_concepts[cid] = MorphologicalConcept(
                canonical_id=cid,
                archetype=arch,
                canonical_name=sc_data.get("canonical_name", "shape"),
                inferred_role=role,
                observed_colors=set(sc_data.get("observed_colors", [])),
                is_rotatable=sc_data.get("is_rotatable", False),
                rotation_trigger=sc_data.get("rotation_trigger"),
                is_color_switch=sc_data.get("is_color_switch", False),
                passable_colors=set(sc_data.get("passable_colors", [])),
                barrier_colors=set(sc_data.get("barrier_colors", [])),
            )
        self.successful_episodes.extend(data.get("successful_episodes", []))
        self.failed_episodes.extend(data.get("failed_episodes", []))
