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
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from hbllm.hcir.graph import ConceptNode
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import EpistemicObservationDiff
from hbllm.hcir.world.extended_body_schema import ExtendedBodySchema
from hbllm.hcir.world.remote_causal_attribution import (
    RemoteCausalAttributor,
)

logger = logging.getLogger(__name__)


class SpelkePhysicalAxiom(StrEnum):
    """Elizabeth Spelke's infant/primate Core Knowledge physical principles."""

    FREE_SPACE = "free_space"
    COHESION = "cohesion"
    SOLIDITY_CONTINUITY = "solidity_continuity"
    CONTACT_CAUSALITY = "contact_causality"
    DIRECTIONAL_INERTIA = "directional_inertia"


class AffordanceType(StrEnum):
    """Categorical object and area affordance roles in neuro-symbolic world theory."""

    CAN_PICKUP = "can_pickup"
    CAN_CARRY = "can_carry"
    RECEPTACLE = "receptacle"
    SOLID_BARRIER = "solid_barrier"
    WALKABLE = "walkable"
    GOAL = "goal"


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
    is_undo: bool = False
    is_value_mutator: bool = False
    is_index_selector: bool = False
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
    domain_predicates: list[CausalPredicate] = field(default_factory=list)

    # Fast-lookup cached sets
    goal_features: set[int] = field(default_factory=set)
    candidate_goal_features: set[int] = field(default_factory=set)
    barrier_features: set[int] = field(default_factory=set)
    walkable_features: set[int] = field(default_factory=set)
    cargo_features: set[int] = field(default_factory=set)
    receptacle_features: set[int] = field(default_factory=set)
    actuator_features: set[int] = field(default_factory=set)

    # Functional action categorization
    undo_actions: set[Any] = field(default_factory=set)
    focus_switch_actions: set[Any] = field(default_factory=set)
    index_selector_actions: set[Any] = field(default_factory=set)
    value_mutator_actions: set[Any] = field(default_factory=set)
    interaction_actions: set[Any] = field(default_factory=set)
    primary_interaction_action: Any | None = None
    action_cycle_periods: dict[Any, int] = field(default_factory=dict)
    mutator_state_history: dict[Any, list[int]] = field(default_factory=dict)
    active_manipulandum_id: str | None = None
    last_effector_target_coord: tuple[int, int] | None = None
    last_effector_target_entity_id: str | None = None

    # Telic global board goal induction
    telic_goal_type: str = ""  # e.g. "deplete", "navigate", "cargo"
    telic_target_feature: int | None = None

    # Elizabeth Spelke's Core Knowledge physical axioms
    spelke_axioms: dict[SpelkePhysicalAxiom, CausalPredicate] = field(default_factory=dict)

    # Tool-Barrier Causal Resonance (persistent across level resets)
    tool_barrier_affinities: dict[int, set[int]] = field(default_factory=dict)

    def record_tool_barrier_affinity(self, tool_feature: int, barrier_feature: int) -> None:
        """Record causal affinity: tool_feature acts upon / clears barrier_feature."""
        if tool_feature == barrier_feature:
            return
        aff_set = self.tool_barrier_affinities.setdefault(tool_feature, set())
        if barrier_feature not in aff_set:
            aff_set.add(barrier_feature)
            pred = CausalPredicate(
                variable="clears_barrier",
                operator="==",
                value=barrier_feature,
                target_entity=str(tool_feature),
            )
            self.register_predicate(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Learned Causal Affinity: Tool/Trigger %d -> Barrier %d",
                tool_feature,
                barrier_feature,
            )

    def get_causal_affinity(
        self,
        entity_feature: int,
        target_barrier_features: set[int] | Sequence[int] | None = None,
    ) -> float:
        """Compute causal mutual information / affinity score P(Delta CutSet | do(O)) > 0."""
        known_barriers = self.tool_barrier_affinities.get(entity_feature)
        if not known_barriers:
            return 0.0
        if not target_barrier_features:
            return 1.0
        if known_barriers.intersection(target_barrier_features):
            return 1.0
        return 0.0

    def has_affordance(
        self,
        entity_or_feature: SpatialEntity | int,
        affordance: AffordanceType | str,
    ) -> bool:
        """Query whether an entity or feature ID affords a specific functional property."""
        aff = str(affordance).lower()
        if "can_pickup" in aff or "cargo" in aff or "portable" in aff:
            return self.is_cargo(entity_or_feature)
        elif "receptacle" in aff or "drop_zone" in aff or "container" in aff:
            return self.is_receptacle(entity_or_feature)
        elif "barrier" in aff or "solid" in aff:
            feat = (
                entity_or_feature
                if isinstance(entity_or_feature, int)
                else entity_or_feature.feature_id
            )
            return self.is_barrier(feat)
        elif "walkable" in aff:
            feat = (
                entity_or_feature
                if isinstance(entity_or_feature, int)
                else entity_or_feature.feature_id
            )
            return self.is_walkable(feat)
        elif "goal" in aff:
            return self.is_goal(entity_or_feature)
        return False

    def induce_interaction_action(self, action: Any) -> None:
        """Induce universal empirical interaction / state-transformation affordance."""
        self.interaction_actions.add(action)
        self.undo_actions.discard(action)
        self.primary_interaction_action = action
        pred = CausalPredicate(
            variable="action_role",
            operator="==",
            value="interaction",
            target_entity=str(action),
        )
        self.register_predicate(pred)
        logger.info(
            "HCIRSymbolicWorldTheory: Induced Empirical Interaction Affordance: Action %s", action
        )

    def is_goal(self, entity: SpatialEntity | int) -> bool:
        """True if entity or feature satisfies any induced goal predicate or confirmed goal feature."""
        if isinstance(entity, (int, np.integer)):
            feat = int(entity)
            if feat in self.goal_features:
                return True
            if not self.goal_features and feat in self.candidate_goal_features:
                return True
            props = {"feature_id": feat, "visual_id": feat}
            return any(pred.evaluate(props) for pred in self.goal_predicates)

        if entity.role == EntityRole.GOAL:
            return True
        if entity.feature_id in self.goal_features:
            return True
        if not self.goal_features and entity.feature_id in self.candidate_goal_features:
            return True
        props = {
            "feature_id": entity.feature_id,
            "visual_id": entity.feature_id,
            "area": getattr(entity, "area", 1),
            "role": str(entity.role),
            **getattr(entity, "properties", {}),
        }
        return any(pred.evaluate(props) for pred in self.goal_predicates)

    def is_goal_feature(self, feature_id: int) -> bool:
        """True if feature satisfies any induced goal predicate or confirmed goal feature."""
        return self.is_goal(feature_id)

    def is_barrier(self, feature_id: int) -> bool:
        """True if feature satisfies any induced barrier predicate (enforcing HCIR Attractor Invariant)."""
        if feature_id in self.goal_features:
            return False
        if not self.goal_features and feature_id in self.candidate_goal_features:
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
        props = {
            "feature_id": entity.feature_id,
            "visual_id": entity.feature_id,
            "area": getattr(entity, "area", 1),
            "role": str(entity.role),
            **getattr(entity, "properties", {}),
        }
        return any(pred.evaluate(props) for pred in self.cargo_predicates)

    def is_receptacle(self, entity: SpatialEntity | int) -> bool:
        """True if entity or feature satisfies any receptacle predicate."""
        if isinstance(entity, (int, np.integer)):
            feat = int(entity)
            if feat in self.receptacle_features:
                return True
            props = {"feature_id": feat, "visual_id": feat}
            return any(pred.evaluate(props) for pred in self.receptacle_predicates)

        if entity.feature_id in self.receptacle_features:
            return True
        props = {
            "feature_id": entity.feature_id,
            "visual_id": entity.feature_id,
            "area": getattr(entity, "area", 1),
            "role": str(entity.role),
            **getattr(entity, "properties", {}),
        }
        return any(pred.evaluate(props) for pred in self.receptacle_predicates)

    def induce_receptacle(self, feature_id: int) -> None:
        """Induce universal receptacle predicate."""
        if feature_id not in self.receptacle_features:
            self.receptacle_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.receptacle_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Receptacle Rule: %s", pred.describe())

    def induce_goal(self, feature_id: int) -> None:
        """Induce universal goal predicate with minimum description length."""
        if feature_id == 0 or feature_id in self.cargo_features:
            return
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
        if feature_id in self.cargo_features:
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
        if feature_id in self.receptacle_features or feature_id in self.goal_features:
            return
        self.walkable_features.discard(feature_id)
        self.goal_features.discard(feature_id)
        self.candidate_goal_features.discard(feature_id)
        if feature_id not in self.cargo_features:
            self.cargo_features.add(feature_id)
            pred = CausalPredicate(variable="feature_id", operator="==", value=feature_id)
            self.cargo_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Cargo Rule: %s", pred.describe())

    def induce_depletion_goal(self, feature_id: int) -> None:
        """Induce a global board depletion goal (clear board of this feature)."""
        if feature_id in self.cargo_features:
            return
        self.telic_goal_type = "deplete"
        self.telic_target_feature = feature_id
        self.induce_goal(feature_id)
        logger.info(
            "HCIRSymbolicWorldTheory: Induced Telic Depletion Goal for feature %d", feature_id
        )

    def induce_undo_action(self, action: Any) -> None:
        """Induce universal empirical state-rollback / undo affordance."""
        if action in self.interaction_actions:
            return
        if action not in self.undo_actions:
            self.undo_actions.add(action)
            pred = CausalPredicate(
                variable="action_role",
                operator="==",
                value="undo",
                target_entity=str(action),
            )
            self.register_predicate(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Induced Empirical UNDO Operator: Action %s", action
            )

    def induce_index_selector(self, action: Any) -> None:
        """Induce dial / cursor index navigation affordance."""
        if action not in self.index_selector_actions:
            self.index_selector_actions.add(action)
            pred = CausalPredicate(
                variable="action_role",
                operator="==",
                value="index_selector",
                target_entity=str(action),
            )
            self.register_predicate(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Induced Empirical Index Selector: Action %s", action
            )

    def induce_value_mutator(self, action: Any) -> None:
        """Induce dial / state value mutation affordance."""
        if action not in self.value_mutator_actions:
            self.value_mutator_actions.add(action)
            pred = CausalPredicate(
                variable="action_role",
                operator="==",
                value="value_mutator",
                target_entity=str(action),
            )
            self.register_predicate(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Induced Empirical Value Mutator: Action %s", action
            )

    def record_state_mutation(
        self,
        action: Any,
        state_sig: int,
        prior_state_sig: int | None = None,
    ) -> int | None:
        """Record observed state signature under an empirical value-mutator action.

        Dynamically learns the recurrence period k of parameter/dial spaces (s_{t+k} = s_t)
        via Pearl's do(a) causal credit assignment, replacing magic dial limits.
        """
        history = self.mutator_state_history.setdefault(action, [])
        if not history and prior_state_sig is not None:
            history.append(prior_state_sig)

        if history and history[-1] == state_sig:
            return self.action_cycle_periods.get(action)

        history.append(state_sig)

        # Check if state_sig was seen before in history (s_{t+k} = s_t)
        if len(history) > 1 and state_sig in history[:-1]:
            prev_idx = len(history) - 2 - history[:-1][::-1].index(state_sig)
            k = len(history) - 1 - prev_idx
            if k >= 2:
                self.action_cycle_periods[action] = k
                logger.info(
                    "HCIRSymbolicWorldTheory: Learned Dynamic Recurrence Period k=%d for action %s",
                    k,
                    action,
                )
                return k
        return self.action_cycle_periods.get(action)

    def get_cycle_period(self, action: Any, default: int = 4) -> int:
        """Return the dynamically learned recurrence period k for action, or fallback."""
        return self.action_cycle_periods.get(action, default)

    def register_predicate(self, predicate: CausalPredicate) -> None:
        """Register a generic or learned causal predicate into the symbolic world theory."""
        if predicate not in self.domain_predicates:
            self.domain_predicates.append(predicate)

    def seed_core_knowledge(self, bg_feature: int = 0) -> None:
        """Seed domain-general Spelke core cognitive knowledge (Physics & Free Space axioms).

        Grounds infant/primate Core Knowledge systems (Spelke 1990, Carey 2009):
        1. FREE_SPACE: Environmental background affords uninhibited locomotion.
        2. SOLIDITY_CONTINUITY: Solid entity boundaries resist traversal unless actuated;
           spatiotemporal paths cannot pass through solid obstacles.
        3. CONTACT_CAUSALITY: State changes and displacements require direct spatial contact
           or grounded distal interventions.
        4. COHESION: Physical entities maintain cohesive bounded spatial extents.
        5. DIRECTIONAL_INERTIA: Displacements maintain directional continuity unless
           interrupted by collisions or external forces.
        """
        self.induce_walkable(bg_feature)

        # 1. Free Space
        self.spelke_axioms[SpelkePhysicalAxiom.FREE_SPACE] = CausalPredicate(
            variable="feature_id",
            operator="==",
            value=bg_feature,
        )

        # 2. Solidity & Spatiotemporal Continuity
        self.spelke_axioms[SpelkePhysicalAxiom.SOLIDITY_CONTINUITY] = CausalPredicate(
            variable="is_solid",
            operator="==",
            value=True,
        )

        # 3. Contact Causality
        self.spelke_axioms[SpelkePhysicalAxiom.CONTACT_CAUSALITY] = CausalPredicate(
            variable="requires_contact",
            operator="==",
            value=True,
        )

        # 4. Cohesion / Boundedness
        self.spelke_axioms[SpelkePhysicalAxiom.COHESION] = CausalPredicate(
            variable="is_cohesive",
            operator="==",
            value=True,
        )

        # 5. Directional Inertia
        self.spelke_axioms[SpelkePhysicalAxiom.DIRECTIONAL_INERTIA] = CausalPredicate(
            variable="maintains_inertia",
            operator="==",
            value=True,
        )

    def evaluates_spelke_continuity(
        self,
        origin: tuple[int, int],
        destination: tuple[int, int],
        barrier_set: set[tuple[int, int]],
        grid_bounds: tuple[int, int] | None = None,
    ) -> bool:
        """Evaluate Spelke spatiotemporal continuity.

        True if there exists a barrier-free topological conduit from origin to destination.
        Returns False if solid boundaries block continuous motion without actuation.
        """
        if origin == destination:
            return True
        if origin in barrier_set or destination in barrier_set:
            return False

        r_min = min(origin[0], destination[0]) - 2
        r_max = max(origin[0], destination[0]) + 2
        c_min = min(origin[1], destination[1]) - 2
        c_max = max(origin[1], destination[1]) + 2
        if grid_bounds:
            r_min = max(0, r_min)
            c_min = max(0, c_min)
            r_max = min(grid_bounds[0] - 1, r_max)
            c_max = min(grid_bounds[1] - 1, c_max)

        from collections import deque

        queue: deque[tuple[int, int]] = deque([origin])
        visited: set[tuple[int, int]] = {origin}

        while queue:
            curr = queue.popleft()
            if curr == destination:
                return True
            r, c = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if r_min <= nr <= r_max and c_min <= nc <= c_max:
                    if nxt not in visited and nxt not in barrier_set:
                        visited.add(nxt)
                        queue.append(nxt)
        return False

    def evaluates_spelke_contact(
        self,
        agent_pos: tuple[int, int],
        target_pos: tuple[int, int],
        adjacency: Literal["4-way", "8-way"] | bool = "4-way",
        stride: int = 1,
    ) -> bool:
        """Evaluate Spelke mechanical contact causality.

        True if agent is adjacent to the target entity affording direct mechanical actuation.
        """
        dr = abs(agent_pos[0] - target_pos[0])
        dc = abs(agent_pos[1] - target_pos[1])
        s = max(1, stride)
        if adjacency == "8-way":
            return max(dr, dc) <= s and max(dr, dc) > 0
        return (dr + dc) <= s and (dr + dc) > 0

    def evaluates_spelke_cohesion(
        self,
        entity_cells: set[tuple[int, int]],
        displacement: tuple[int, int],
        barrier_set: set[tuple[int, int]],
        grid_bounds: tuple[int, int] | None = None,
    ) -> bool:
        """Evaluate Spelke Cohesion & Rigid Body Motion.

        Objects move as cohesive wholes with rigid connectivity.
        True if all constituent cells under displacement (dr, dc):
        1. Maintain mutual relative connectivity without tearing.
        2. Remain strictly inside grid_bounds.
        3. Do not intersect any static barriers (excluding cells currently vacated by the entity).
        """
        if not entity_cells:
            return True
        dr, dc = displacement
        displaced_cells = {(r + dr, c + dc) for r, c in entity_cells}

        if grid_bounds:
            H, W = grid_bounds
            for r, c in displaced_cells:
                if not (0 <= r < H and 0 <= c < W):
                    return False

        # Collision check with external barriers
        external_barriers = barrier_set - entity_cells
        if any(cell in external_barriers for cell in displaced_cells):
            return False

        return True

    def evaluates_spelke_inertia(
        self,
        position: tuple[int, int],
        velocity: tuple[int, int],
        steps: int = 1,
        obstacles: set[tuple[int, int]] | None = None,
        bounce: bool = False,
        grid_bounds: tuple[int, int] | None = None,
    ) -> list[tuple[int, int]]:
        """Evaluate Spelke Directional Inertia & Momentum Conservation.

        Objects maintain directional trajectory and velocity across time steps
        unless deflected or halted by external boundaries or obstacles.

        Returns projected sequence of coordinates [(r_1, c_1), ..., (r_n, c_n)].
        """
        trajectory: list[tuple[int, int]] = []
        curr_r, curr_c = position
        vr, vc = velocity
        obs = obstacles or set()

        for _ in range(steps):
            next_r, next_c = curr_r + vr, curr_c + vc
            hit = False

            if grid_bounds:
                H, W = grid_bounds
                if not (0 <= next_r < H and 0 <= next_c < W):
                    hit = True

            if (next_r, next_c) in obs:
                hit = True

            if hit:
                if bounce:
                    vr, vc = -vr, -vc
                    next_r, next_c = curr_r + vr, curr_c + vc
                    if grid_bounds:
                        H, W = grid_bounds
                        next_r = max(0, min(H - 1, next_r))
                        next_c = max(0, min(W - 1, next_c))
                else:
                    break

            curr_r, curr_c = next_r, next_c
            trajectory.append((curr_r, curr_c))

        return trajectory

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert all symbolic world predicates and ontology into atomic HCIR graph transactions."""
        ops: list[TransactionOperation] = []

        # Ground Elizabeth Spelke physical axioms
        for axiom, pred in self.spelke_axioms.items():
            node_id = f"spelke_axiom_{axiom.value}"
            node = ConceptNode(
                id=node_id,
                label=f"spelke_{axiom.value}",
                definition=pred.describe(),
                domain="spelke_core_knowledge",
                properties={
                    "axiom": axiom.value,
                    "predicate": pred.describe(),
                    "variable": pred.variable,
                    "operator": pred.operator,
                    "value": str(pred.value),
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )

        for role, feats in [
            ("goal", self.goal_features),
            ("barrier", self.barrier_features),
            ("walkable", self.walkable_features),
            ("cargo", self.cargo_features),
            ("receptacle", self.receptacle_features),
        ]:
            for feat in feats:
                node_id = f"ontology_{role}_{feat}"
                node = ConceptNode(
                    id=node_id,
                    label=f"{role}_{feat}",
                    definition=f"Learned intensional {role} entity with feature {feat}",
                    domain="spatial_ontology",
                    properties={
                        "role": role,
                        "feature_id": feat,
                        "is_intensional": True,
                    },
                )
                ops.append(
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=node_id,
                        node_data=node.model_dump(),
                    )
                )
        if self.telic_goal_type:
            node_id = "ontology_telic_goal"
            node = ConceptNode(
                id=node_id,
                label=f"telic_goal_{self.telic_goal_type}",
                definition=f"Global telic goal: {self.telic_goal_type} targeting {self.telic_target_feature}",
                domain="spatial_ontology",
                properties={
                    "telic_goal_type": self.telic_goal_type,
                    "telic_target_feature": self.telic_target_feature,
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )

        for act, period in self.action_cycle_periods.items():
            node_id = f"causal_recurrence_action_{act}"
            node = ConceptNode(
                id=node_id,
                label=f"recurrence_{act}_{period}",
                definition=f"Learned dynamic recurrence period k={period} for action {act}",
                domain="causal_dynamics",
                properties={"action": act, "period": period},
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )
        return ops

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
        self.spelke_axioms.clear()
        self.action_cycle_periods.clear()
        self.mutator_state_history.clear()
        self.telic_goal_type = ""
        self.telic_target_feature = None


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
        action_locus: tuple[int, int] | None = None,
        avatar_features: set[int] | None = None,
    ) -> None:
        """Bind intentional do(a) action to local and distal environmental mutations."""
        if origin_pos is not None:
            self.observed_transitions.setdefault((origin_pos, action), []).append(obs_diff)

        # 1. Distal Causal Attribution via vmPFC
        interaction_loci: list[tuple[int, int]] = []
        if action_locus is not None:
            interaction_loci.append(action_locus)
        else:
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
                avatar_features=avatar_features,
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

        # 3. Check if body schema knows a tool that unlocks this barrier
        tools = self.body_schema.get_tools_for_barrier(barrier_feature)
        if tools:
            for t_feat in tools:
                if t_feat in self.body_schema.tool_barrier_affinities:
                    return (None, t_feat)

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
        self.causal_attributor.reset_episode(retain_long_term=retain_dynamics)
        if not retain_dynamics:
            self.causal_attributor.reset()
        if not retain_dynamics:
            self.symbolic_theory.clear()
            self.observed_transitions.clear()

    def reset(self) -> None:
        """Reset all causal induction state, symbolic theories, and observed transitions."""
        self.reset_episode(retain_dynamics=False)

    def plan_epistemic_intervention(
        self,
        avatar_pos: tuple[int, int] | None,
        candidate_entities: Sequence[SpatialEntity],
        available_actions: Sequence[Any],
        bg_feature: int = 0,
    ) -> EpistemicInterventionProposal | None:
        """Pearl's do(a) active causal calculus:
        Computes expected information gain of targeted interventions to disambiguate unknown entities.
        """
        if not candidate_entities or not available_actions:
            return None

        # Filter ambiguous entities (not confirmed goal, barrier, or background)
        ambiguous = [
            e
            for e in candidate_entities
            if e.feature_id != bg_feature
            and (avatar_pos is None or e.grid_pos != avatar_pos)
            and not self.symbolic_theory.is_goal(e.feature_id)
            and not self.symbolic_theory.is_barrier(e.feature_id)
            and not self.symbolic_theory.is_walkable(e.feature_id)
        ]
        if not ambiguous:
            return None

        best_entity: SpatialEntity | None = None
        best_score = -1.0
        best_hypothesis = ""

        for e in ambiguous:
            dist = (
                abs(e.grid_pos[0] - avatar_pos[0]) + abs(e.grid_pos[1] - avatar_pos[1])
                if avatar_pos
                else 1
            )
            is_unprobed = e.feature_id not in self.symbolic_theory.walkable_features
            entropy = 1.5 if is_unprobed else 0.5
            info_gain = entropy / max(1, dist)

            if info_gain > best_score:
                best_score = info_gain
                best_entity = e
                best_hypothesis = f"test_affordance_feature_{e.feature_id}_at_{e.grid_pos}"

        if best_entity is None:
            return None

        chosen_action = available_actions[0]
        return EpistemicInterventionProposal(
            action=chosen_action,
            target_coord=best_entity.grid_pos,
            target_feature=best_entity.feature_id,
            expected_information_gain=best_score,
            hypothesis=best_hypothesis,
        )


@dataclass
class EpistemicInterventionProposal:
    """Proposed Pearl do(a) intervention designed to maximize causal information gain."""

    action: Any
    target_coord: tuple[int, int] | None = None
    target_feature: int | None = None
    expected_information_gain: float = 0.0
    hypothesis: str = ""

    def to_causal_predicate(self) -> CausalPredicate:
        return CausalPredicate(
            variable="epistemic_intervention",
            operator="==",
            value=str(self.action),
            relation=f"info_gain:{self.expected_information_gain:.3f}",
            target_entity=str(self.target_coord),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="active_epistemic_intervention",
                node_data={
                    "action": self.action,
                    "target_coord": list(self.target_coord) if self.target_coord else None,
                    "target_feature": self.target_feature,
                    "expected_information_gain": self.expected_information_gain,
                    "hypothesis": self.hypothesis,
                },
            )
        ]
