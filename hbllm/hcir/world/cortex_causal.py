"""Ventromedial Prefrontal Cortex (vmPFC) & Inferior Parietal Causal Induction Faculty.

Biologically modeled on mammalian vmPFC distal causal credit assignment (Pearl's do(a) intervention)
and parietal tool affordance resonance (Iriki-Maravita body schema plasticity):
1. Interventional Action-Effect Binding:
   - Evaluates do(a) interventions: correlates intentional actions at origin/target locations
     with local and distal state mutations (diffs).
   - Generates and verifies causal hypotheses for switches, levers, plates, and keys.
2. Neuro-Symbolic World Theory:
   - Grounds first-order causal predicates for goals, barriers, walkable areas, and pushable cargo.
   - Enforces the HCIR Attractor Invariant across diverse environments without heuristics.
3. Parietal Tool-Barrier Resonance:
   - Tracks held objects in the extended body schema and evaluates barrier permeability.
4. Remote Mechanism Reverse Affordance Lookup:
   - When an objective is blocked by a barrier, queries vmPFC for governing distal triggers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import EpistemicObservationDiff
from hbllm.hcir.world.extended_body_schema import ExtendedBodySchema
from hbllm.hcir.world.remote_causal_attribution import (
    RemoteCausalAttributor,
)

logger = logging.getLogger(__name__)


@dataclass
class ActionAffordance:
    """Declared or empirically discovered motor affordance for an embodied action.

    Decouples cognitive reasoning from specific game environments, hardware drivers,
    cameras, depth sensors, or LiDAR peripherals.
    """

    action_id: Any
    name: str = ""
    requires_spatial_target: bool = False
    target_param_keys: tuple[str, ...] = ("x", "y")
    is_displacement: bool = False
    delta: tuple[int, ...] | None = None
    is_state_transform: bool = False
    is_focus_switch: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HCIRSymbolicWorldTheory:
    """Neuro-symbolic domain-agnostic world theory grounded in HCIR CausalPredicates.

    Maintains intensional causal rules and first-order predicates across level transitions,
    eliminating fragile procedural heuristics.
    """

    goal_predicates: list[CausalPredicate] = field(default_factory=list)
    barrier_predicates: list[CausalPredicate] = field(default_factory=list)
    walkable_predicates: list[CausalPredicate] = field(default_factory=list)
    cargo_predicates: list[CausalPredicate] = field(default_factory=list)
    receptacle_predicates: list[CausalPredicate] = field(default_factory=list)

    # Fast-lookup cached sets
    goal_features: set[int] = field(default_factory=set)
    candidate_goal_features: set[int] = field(default_factory=set)
    barrier_features: set[int] = field(default_factory=set)
    walkable_features: set[int] = field(default_factory=set)
    cargo_features: set[int] = field(default_factory=set)
    receptacle_features: set[int] = field(default_factory=set)

    def is_goal(self, entity: SpatialEntity | int) -> bool:
        """True if entity or feature satisfies any induced goal predicate or confirmed goal feature."""
        if isinstance(entity, (int, np.integer)):
            feat = int(entity)
            if feat in self.goal_features or feat in self.candidate_goal_features:
                return True
            props = {"feature_id": feat, "visual_id": feat}
            return any(pred.evaluate(props) for pred in self.goal_predicates)

        if entity.role == EntityRole.GOAL:
            return True
        if (
            entity.feature_id in self.goal_features
            or entity.feature_id in self.candidate_goal_features
        ):
            return True
        props = getattr(entity, "properties", {})
        return any(pred.evaluate(props) for pred in self.goal_predicates)

    def is_goal_feature(self, feature_id: int) -> bool:
        """True if feature satisfies any induced goal predicate or confirmed goal feature."""
        return self.is_goal(feature_id)

    def is_barrier(self, feature_id: int) -> bool:
        """True if feature satisfies any induced barrier predicate (enforcing HCIR Attractor Invariant)."""
        if feature_id in self.goal_features or feature_id in self.candidate_goal_features:
            return False
        if feature_id in self.barrier_features:
            return True
        props = {"feature_id": feature_id, "visual_id": feature_id}
        return any(pred.evaluate(props) for pred in self.barrier_predicates)

    def is_walkable(self, feature_id: int) -> bool:
        """True if feature satisfies any induced walkable space predicate."""
        if feature_id in self.walkable_features:
            return True
        props = {"feature_id": feature_id, "visual_id": feature_id}
        return any(pred.evaluate(props) for pred in self.walkable_predicates)

    def is_cargo(self, entity: SpatialEntity | int) -> bool:
        """True if entity or feature satisfies any pushable cargo predicate."""
        if isinstance(entity, (int, np.integer)):
            feat = int(entity)
            if feat in self.cargo_features:
                return True
            props = {"feature_id": feat, "visual_id": feat}
            return any(pred.evaluate(props) for pred in self.cargo_predicates)

        if entity.feature_id in self.cargo_features:
            return True
        props = getattr(entity, "properties", {})
        return any(pred.evaluate(props) for pred in self.cargo_predicates)

    def induce_goal(self, feature_id: int) -> None:
        """Induce universal goal predicate with minimum description length."""
        self.barrier_features.discard(feature_id)
        if feature_id not in self.goal_features:
            self.goal_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.goal_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Goal Rule: %s", pred.describe())
        self.walkable_features.add(feature_id)

    def induce_barrier(self, feature_id: int) -> None:
        """Induce universal barrier predicate (enforcing HCIR Attractor Invariant)."""
        if feature_id in self.goal_features or feature_id in self.candidate_goal_features:
            return
        if feature_id in self.walkable_features:
            return
        if feature_id not in self.barrier_features:
            self.barrier_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.barrier_predicates.append(pred)
            self.walkable_features.discard(feature_id)
            logger.info("HCIRSymbolicWorldTheory: Induced Barrier Rule: %s", pred.describe())

    def induce_walkable(self, feature_id: int) -> None:
        """Induce universal walkable space predicate."""
        if feature_id in self.barrier_features:
            return
        if feature_id not in self.walkable_features:
            self.walkable_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.walkable_predicates.append(pred)

    def induce_cargo(self, feature_id: int) -> None:
        """Induce pushable cargo predicate."""
        if feature_id in self.barrier_features:
            return
        self.walkable_features.discard(feature_id)
        if feature_id not in self.cargo_features:
            self.cargo_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.cargo_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Cargo Rule: %s", pred.describe())

    def clear(self) -> None:
        """Clear all intensional predicates when resetting across distinct games."""
        self.goal_predicates.clear()
        self.barrier_predicates.clear()
        self.walkable_predicates.clear()
        self.cargo_predicates.clear()
        self.receptacle_predicates.clear()
        self.goal_features.clear()
        self.candidate_goal_features.clear()
        self.barrier_features.clear()
        self.walkable_features.clear()
        self.cargo_features.clear()
        self.receptacle_features.clear()


class CausalInductionCortex:
    """vmPFC & Parietal Cortex causal induction engine.

    Unifies interventional do(a) action-effect binding, symbolic world theory,
    distal trigger-barrier attribution, and tool-barrier resonance.
    """

    def __init__(
        self,
        symbolic_theory: HCIRSymbolicWorldTheory | None = None,
        causal_attributor: RemoteCausalAttributor | None = None,
        body_schema: ExtendedBodySchema | None = None,
    ) -> None:
        self.symbolic_theory = symbolic_theory or HCIRSymbolicWorldTheory()
        self.causal_attributor = causal_attributor or RemoteCausalAttributor()
        self.body_schema = body_schema or ExtendedBodySchema()

        # Empirical interventional transition observations: (origin_pos, action) -> outcomes
        self.observed_transitions: dict[
            tuple[tuple[int, int], Any], list[EpistemicObservationDiff]
        ] = {}

    def bind_intervention(
        self,
        action: Any,
        origin_pos: tuple[int, int] | None,
        target_pos: tuple[int, int] | None,
        obs_diff: EpistemicObservationDiff,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        bg_feature: int = 0,
    ) -> None:
        """Bind intentional do(a) action to local and distal environmental mutations."""
        if origin_pos is not None:
            self.observed_transitions.setdefault((origin_pos, action), []).append(obs_diff)

        # 1. Distal Causal Attribution via vmPFC
        interaction_loci: list[tuple[int, int]] = []
        if origin_pos is not None:
            interaction_loci.append(origin_pos)
        if target_pos is not None and target_pos != origin_pos:
            interaction_loci.append(target_pos)

        for loc in interaction_loci:
            self.causal_attributor.record_transition(
                prev_grid=prev_grid,
                curr_grid=curr_grid,
                action_pos=loc,
                background_feature=bg_feature,
            )

    def query_barrier_clearance(
        self,
        barrier_pos: tuple[int, int],
        barrier_feature: int,
    ) -> tuple[tuple[int, int] | None, int | None]:
        """Query if a barrier can be cleared by a remote trigger or held tool.

        Returns:
            (trigger_pos, required_tool_feature)
        """
        # 1. Check if extended body schema holds a tool that unlocks this barrier
        if self.body_schema.is_barrier_permeable(barrier_feature):
            for tool in self.body_schema.held_tools:
                if barrier_feature in tool.unlocked_barriers or tool.feature_id == barrier_feature:
                    return (None, tool.feature_id)

        # 2. Check if vmPFC has discovered a remote trigger for this barrier pos
        affordance = self.causal_attributor.get_trigger_for_barrier(
            barrier_pos,
            barrier_feature=barrier_feature,
        )
        if affordance is not None:
            return (affordance.trigger_pos, None)

        return (None, None)

    def is_passable(self, feature_id: int, pos: tuple[int, int] | None = None) -> bool:
        """Determine if a grid tile is passable considering symbolic theory and tools."""
        if self.symbolic_theory.is_walkable(feature_id):
            return True
        if self.symbolic_theory.is_goal(feature_id):
            return True
        if self.body_schema.is_barrier_permeable(feature_id):
            return True
        if not self.symbolic_theory.is_barrier(feature_id):
            # Not confirmed as barrier
            return True
        return False

    def reset_episode(self, retain_dynamics: bool = True) -> None:
        """Reset episode state, optionally preserving discovered causal affordances."""
        self.body_schema.reset_episode(retain_dynamics=retain_dynamics)
        if hasattr(self.causal_attributor, "reset_episode"):
            self.causal_attributor.reset_episode(retain_long_term=retain_dynamics)
        elif hasattr(self.causal_attributor, "reset"):
            if not retain_dynamics:
                self.causal_attributor.reset()
        if not retain_dynamics:
            self.symbolic_theory.clear()
            self.observed_transitions.clear()

    def reset(self) -> None:
        """Reset all causal induction state, symbolic theories, and observed transitions."""
        self.reset_episode(retain_dynamics=False)
