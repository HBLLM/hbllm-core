"""Declarative Spatial Affordances & Spelke Physical Constraints.

Replaces ad-hoc procedural role checks with universal, declarative affordance
predicates adhering to Pearl's do(a) causal calculus and Spelke physical continuity.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.world.causal.transition_kb import TransitionOutcome
from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpatialAffordanceSnapshot:
    """Immutable evaluation of spatial affordances across a sensory lattice.

    Provides declarative query methods for cognitive operators (OFC, BA10, vmPFC)
    so planners do not rely on level-specific heuristics or raw feature identities.
    """

    passable_coords: frozenset[tuple[int, int]]
    solid_barriers: frozenset[tuple[int, int]]
    permeable_barriers: frozenset[tuple[int, int]]
    cargo_coords: frozenset[tuple[int, int]]
    goal_coords: frozenset[tuple[int, int]]
    lethal_coords: frozenset[tuple[int, int]]
    actuator_coords: frozenset[tuple[int, int]]
    grid_shape: tuple[int, int]
    barrier_features: frozenset[int] = field(default_factory=frozenset)
    cargo_features: frozenset[int] = field(default_factory=frozenset)
    goal_features: frozenset[int] = field(default_factory=frozenset)
    lethal_features: frozenset[int] = field(default_factory=frozenset)
    tool_barrier_affinities: dict[int, set[int]] = field(default_factory=dict)

    def is_passable(self, pos: tuple[int, int]) -> bool:
        """Return True if coordinate pos is mechanically passable."""
        return pos in self.passable_coords and pos not in self.solid_barriers

    def is_solid_barrier(self, pos: tuple[int, int]) -> bool:
        """Return True if pos is an impassable solid (Spelke solidarity)."""
        return pos in self.solid_barriers

    def is_permeable_barrier(self, pos: tuple[int, int]) -> bool:
        """Return True if pos is a barrier that can be opened/unlocked via tool/switch."""
        return pos in self.permeable_barriers

    def is_cargo(self, pos: tuple[int, int]) -> bool:
        """Return True if pos contains pushable/movable cargo."""
        return pos in self.cargo_coords

    def is_goal(self, pos: tuple[int, int]) -> bool:
        """Return True if pos is an active telic goal."""
        return pos in self.goal_coords

    def is_lethal(self, pos: tuple[int, int]) -> bool:
        """Return True if pos is a lethal hazard or Habenular threat."""
        return pos in self.lethal_coords

    def is_actuator(self, pos: tuple[int, int]) -> bool:
        """Return True if pos contains a distal switch, trigger, or actuator."""
        return pos in self.actuator_coords

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export active affordance invariants to causal theory."""
        preds: list[CausalPredicate] = []
        for feat in self.barrier_features:
            preds.append(
                CausalPredicate(
                    variable="affordance:solid",
                    operator="==",
                    value=feat,
                    relation="passable:False",
                )
            )
        for feat in self.goal_features:
            preds.append(
                CausalPredicate(
                    variable="affordance:telic_goal",
                    operator="==",
                    value=feat,
                    relation="reward:terminal",
                )
            )
        return preds


class SpatialAffordanceModel:
    """Evaluates spatial affordances across environmental lattices.

    Bridges HCIRSymbolicWorldTheory, Spelke physical continuity, and
    PrefrontalWorkingMemory into a unified, declarative snapshot.
    """

    @classmethod
    def evaluate(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        bg: int = 0,
        av_feats: set[int] | None = None,
        goals: Sequence[tuple[int, int]] | None = None,
        entities: Sequence[Any] | None = None,
    ) -> SpatialAffordanceSnapshot:
        """Extract a complete declarative affordance snapshot for the current sensory state."""
        H, W = curr_grid.shape
        av_set = av_feats or {getattr(engine, "avatar_feature", 1)}
        theory = getattr(engine, "symbolic_theory", None)
        wm = getattr(engine, "working_memory", None)

        held_items = set(getattr(wm, "held_feature_ids", set()) or ())
        agent_state = {"inventory": held_items}

        # 1. Base feature classifications
        barrier_feats: set[int] = set()
        walkable_feats: set[int] = set()
        cargo_feats: set[int] = set()
        goal_feats: set[int] = set()
        lethal_feats: set[int] = set()

        if theory is not None:
            barrier_feats.update(int(f) for f in theory.barrier_features if isinstance(f, int))
            walkable_feats.update(int(f) for f in theory.walkable_features if isinstance(f, int))
            cargo_feats.update(int(f) for f in theory.cargo_features if isinstance(f, int))
            goal_feats.update(int(f) for f in theory.goal_features if isinstance(f, int))

        hazard_tracker = getattr(engine, "hazard_tracker", None)
        if hazard_tracker is not None:
            lethal_feats.update(
                int(f)
                for f in hazard_tracker.known_lethal_features
                if f not in getattr(engine, "mobile_threat_features", set()) and isinstance(f, int)
            )

        tool_affinities: dict[int, set[int]] = {}
        if wm is not None:
            tool_affinities = dict(wm.tool_barrier_affinities)

        # 2. Lattice coordinate sets
        passable_coords: set[tuple[int, int]] = set()
        solid_barriers: set[tuple[int, int]] = set()
        permeable_barriers: set[tuple[int, int]] = set()
        cargo_coords: set[tuple[int, int]] = set()
        goal_coords: set[tuple[int, int]] = set(goals or ())
        lethal_coords: set[tuple[int, int]] = set()
        actuator_coords: set[tuple[int, int]] = set()

        if hazard_tracker is not None:
            lethal_coords.update(hazard_tracker.static_lethal_positions)

        # 3. Scan spatial grid
        for r in range(H):
            for c in range(W):
                coord = (r, c)
                val = int(curr_grid[r, c])

                if val in av_set:
                    passable_coords.add(coord)
                    continue

                ev_cell = (
                    theory.evaluate_transition(val, locus=coord) if theory is not None else None
                )
                if val == bg or val in walkable_feats:
                    passable_coords.add(coord)
                elif (
                    ev_cell is not None
                    and ev_cell.traversable
                    and ev_cell.confidence > 0.0
                    and ev_cell.risk_cost < 40.0
                ):
                    passable_coords.add(coord)
                elif theory is not None and theory.is_walkable(val, agent_state=agent_state):
                    passable_coords.add(coord)

                is_bar_cell = (
                    theory is not None
                    and (
                        (
                            ev_cell is not None
                            and (not ev_cell.traversable or ev_cell.risk_cost >= 40.0)
                        )
                        or theory.is_barrier(val, agent_state=agent_state)
                    )
                ) or val in barrier_feats
                if (
                    ev_cell is not None
                    and ev_cell.traversable
                    and ev_cell.confidence > 0.0
                    and ev_cell.risk_cost < 40.0
                    and theory is not None
                    and theory.transition_kb._locus_beliefs
                    and theory.transition_kb.get_belief(coord, scope="locus") is not None
                ):
                    is_bar_cell = False

                if is_bar_cell:
                    # Check if permeable via held inventory (wm affinities or KB causal rules)
                    is_permeable = False
                    for h in held_items:
                        if h in tool_affinities and val in tool_affinities[h]:
                            is_permeable = True
                            break
                        if theory is not None:
                            for rule in theory.transition_kb.query_causal_rules(h):
                                if rule.effect_signature in (
                                    f"unlocks:{val}",
                                    f"unlocks:{coord}",
                                ):
                                    is_permeable = True
                                    break
                        if is_permeable:
                            break
                    if is_permeable:
                        permeable_barriers.add(coord)
                    else:
                        solid_barriers.add(coord)

                if val in cargo_feats or (
                    theory is not None
                    and (
                        theory.transition_kb.has_feature_outcome(
                            val,
                            TransitionOutcome.DISPLACED_OTHER,
                        )
                        or theory.is_cargo(val)
                    )
                ):
                    cargo_coords.add(coord)

                if (
                    val in goal_feats
                    or (ev_cell is not None and ev_cell.goal_value > 0.5)
                    or (theory is not None and theory.is_goal_feature(val))
                ):
                    goal_coords.add(coord)

                if val in lethal_feats or (ev_cell is not None and ev_cell.risk_cost >= 40.0):
                    lethal_coords.add(coord)

                if theory is not None and (
                    (ev_cell is not None and ev_cell.mutates_state)
                    or theory.transition_kb.has_feature_outcome(
                        val,
                        TransitionOutcome.MUTATED_STATE,
                    )
                    or theory.is_actuator(val)
                ):
                    actuator_coords.add(coord)

        # 4. Integrate discrete entities
        if entities:
            for e in entities:
                pos = getattr(e, "grid_pos", None)
                if pos is None:
                    continue
                role = getattr(e, "role", None)
                feat = getattr(e, "feature_id", int(curr_grid[pos[0], pos[1]]))

                if role == EntityRole.OBSTACLE and feat != bg and feat not in av_set:
                    solid_barriers.add((int(pos[0]), int(pos[1])))
                elif role == EntityRole.GOAL:
                    goal_coords.add((int(pos[0]), int(pos[1])))
                elif role in (EntityRole.MANIPULABLE, "cargo"):
                    cargo_coords.add((int(pos[0]), int(pos[1])))
                elif role == EntityRole.ACTUATOR:
                    actuator_coords.add((int(pos[0]), int(pos[1])))

        return SpatialAffordanceSnapshot(
            passable_coords=frozenset(passable_coords),
            solid_barriers=frozenset(solid_barriers),
            permeable_barriers=frozenset(permeable_barriers),
            cargo_coords=frozenset(cargo_coords),
            goal_coords=frozenset(goal_coords),
            lethal_coords=frozenset(lethal_coords),
            actuator_coords=frozenset(actuator_coords),
            grid_shape=(H, W),
            barrier_features=frozenset(barrier_feats),
            cargo_features=frozenset(cargo_feats),
            goal_features=frozenset(goal_feats),
            lethal_features=frozenset(lethal_feats),
            tool_barrier_affinities=tool_affinities,
        )
