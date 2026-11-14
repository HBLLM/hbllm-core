"""Orbitofrontal Cortex (OFC) & Prefrontal Mental Simulation Planner.

Biologically modeled on mammalian OFC counterfactual preplay and frontopolar (BA10) cognitive branching:
1. Counterfactual Forward Simulation:
   - Rollouts candidate actions within internal mental models prior to physical motor commitment.
   - Evaluates multi-body kinematics (pushing cargo, falling under gravity, dynamic patrollers & chasers).
2. Topological Deadlock Pruning:
   - Rejects action trajectories leading to irreversible state deadlocks (corner traps, wall freezes, 2x2 blocks).
3. Dynamic Adversary Kinematics & Theory of Mind:
   - Predicts directional patroller trajectories and greedy chasing agents.
   - Computes evasive or stealth approach paths.
4. BA10 Hierarchical Subgoal Scheduling:
   - Decomposes locked barrier pathways into recursive intermediate subgoals (collect key, trip switch, clear door).
"""

from __future__ import annotations


def get_sub_attr(target: Any, attr: str, default: Any = None) -> Any:
    return getattr(target, attr, default) if target is not None else default


import heapq
import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import OrientedThreat, PerceptionEngine
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.frontopolar_subgoal_stack import FrontopolarSubgoal, SubgoalType
from hbllm.hcir.world.prefrontal_working_memory import HeldItem
from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.engine_state import CognitiveContext

logger = logging.getLogger(__name__)


@dataclass
class MentalSimulationStep:
    """A step planned purely within internal mental simulation."""

    action: Any
    action_data: dict[str, Any] | None = None
    predicted_avatar_pos: tuple[int, int] = (0, 0)
    expected_mutation: str | None = None
    is_deterministic_geodesic: bool = False

    def to_causal_predicate(self) -> CausalPredicate:
        """Export simulation step as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"simulation_step:{self.action}",
            operator="==",
            value=f"({self.predicted_avatar_pos[0]},{self.predicted_avatar_pos[1]})",
            relation=f"geodesic:{self.is_deterministic_geodesic}",
            target_entity=str(self.action),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording simulation step."""
        node_id = (
            f"sim_step_{self.action}_{self.predicted_avatar_pos[0]}_{self.predicted_avatar_pos[1]}"
        )
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "id": node_id,
                    "node_type": "action",
                    "category": "execution",
                    "action": self.action,
                    "predicted_pos": self.predicted_avatar_pos,
                    "expected_mutation": self.expected_mutation,
                    "is_deterministic_geodesic": self.is_deterministic_geodesic,
                },
            )
        ]


@dataclass
class EpistemicGoalProposal:
    """Immutable proposal for candidate goals and associated epistemic state mutations.

    Adheres to the Reasoning Invariant: Operators calculate proposals;
    HCIR / AutonomousEpistemicEngine commits state changes atomically.
    """

    goals: list[tuple[int, int]]
    confirmed_goals: list[tuple[int, int]]
    new_learned_goal_positions: set[tuple[int, int]] = field(default_factory=set)
    new_candidate_features: set[int] = field(default_factory=set)
    subgoal_to_pop: bool = False
    clear_exhausted_goals: bool = False
    satisfied_when: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    transient_blocked_cells: set[tuple[int, int]] = field(default_factory=set)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export goal proposal as a declarative causal predicate."""
        return CausalPredicate(
            variable="epistemic_goal_proposal",
            operator="==",
            value=len(self.goals),
            relation=f"confirmed:{len(self.confirmed_goals)}",
            target_entity=f"subgoal_pop:{self.subgoal_to_pop}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording goal proposal."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="goal_proposal_summary",
                node_data={
                    "goals_count": len(self.goals),
                    "confirmed_count": len(self.confirmed_goals),
                    "new_learned_positions": [list(p) for p in self.new_learned_goal_positions],
                    "new_candidate_features": sorted(self.new_candidate_features),
                },
            )
        ]


class EpistemicGoalScheduler:
    """Apex orbitofrontal & frontopolar goal priority scheduler (BA10 / OFC).

    Evaluates potential attractor states across a principled epistemic hierarchy:
    1. Causal Confirmed Goals (empirical evidence from past episodes).
    2. BA10 Cognitive Branching Subgoals (active obstacle-clearing / switch subgoals).
    3. Structural & Inductive Inferences (geometric target zones, exit markers).
    4. Exploratory Entities (unverified candidate attractors).
    """

    @staticmethod
    @staticmethod
    def compute_reversible_approach_stances(
        engine: Any,
        curr_grid: np.ndarray,
        H: int,
        W: int,
        bg: int,
        avatar_pos: tuple[int, int] | None,
        entity_pos: tuple[int, int],
        feature_id: int,
        entity_area: int,
        has_cargo_context: bool,
    ) -> list[tuple[int, int]] | None:
        """Return safe open-push-through approach stances if some approach sides would pin an unverified interior entity."""
        th = engine.symbolic_theory
        er, ec = entity_pos
        should_check_reversible = (
            not has_cargo_context
            and avatar_pos is not None
            and int(curr_grid[er, ec]) == int(feature_id)
            and not th.is_walkable(feature_id)
            and feature_id not in th.goal_features
            and entity_pos not in engine.learned_goal_positions
            and not th.is_barrier(feature_id)
            and 1 <= entity_area <= 4
            and 0 < er < H - 1
            and 0 < ec < W - 1
        )
        if not should_check_reversible or avatar_pos is None:
            return None
        goal_like = th.goal_features | th.candidate_goal_features | th.receptacle_features
        safe_stances: list[tuple[int, int]] = []
        unsafe_stances: list[tuple[int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ap = (er + dr, ec + dc)
            if not (0 <= ap[0] < H and 0 <= ap[1] < W):
                continue
            if ap in engine.learned_barriers or ap in engine.current_level_lethal_positions:
                continue
            ap_feat = int(curr_grid[ap[0], ap[1]])
            if not (ap_feat == bg or th.is_walkable(ap_feat) or ap == avatar_pos):
                continue
            b1 = (er - dr, ec - dc)
            b2 = (er - 2 * dr, ec - 2 * dc)
            b1_open = False
            b2_open = False
            if 0 <= b1[0] < H and 0 <= b1[1] < W:
                f1 = int(curr_grid[b1[0], b1[1]])
                b1_open = (
                    b1 not in engine.learned_barriers
                    and b1 not in engine.current_level_lethal_positions
                    and not th.is_barrier(f1)
                    and (
                        f1 == bg
                        or b1 == avatar_pos
                        or th.is_avatar_feature(f1)
                        or th.is_walkable(f1)
                        or f1 in goal_like
                    )
                )
            if 0 <= b2[0] < H and 0 <= b2[1] < W:
                f2 = int(curr_grid[b2[0], b2[1]])
                b2_open = (
                    b2 not in engine.learned_barriers
                    and not th.is_barrier(f2)
                    and (
                        f2 == bg
                        or b2 == avatar_pos
                        or th.is_avatar_feature(f2)
                        or th.is_walkable(f2)
                        or f2 in goal_like
                    )
                )
            if b1_open and b2_open:
                safe_stances.append(ap)
            else:
                unsafe_stances.append(ap)
        if safe_stances and unsafe_stances and avatar_pos not in safe_stances:
            return safe_stances
        return None

    @staticmethod
    def _enqueue_walkable_or_reversible_approach(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        H: int,
        W: int,
        bg: int,
        avatar_pos: tuple[int, int] | None,
        entity_pos: tuple[int, int],
        feature_id: int,
        entity_area: int,
        has_cargo_context: bool,
        target_goals: list[tuple[int, int]],
        satisfied_when: dict[tuple[int, int], dict[str, Any]],
        *,
        set_contact_meta: bool = False,
        transient_blocked: set[tuple[int, int]] | None = None,
    ) -> None:
        """Approach unverified small interior entities from an open push-through side when possible."""
        th = engine.symbolic_theory
        safe_stances = EpistemicGoalScheduler.compute_reversible_approach_stances(
            engine=engine,
            curr_grid=curr_grid,
            H=H,
            W=W,
            bg=bg,
            avatar_pos=avatar_pos,
            entity_pos=entity_pos,
            feature_id=feature_id,
            entity_area=entity_area,
            has_cargo_context=has_cargo_context,
        )
        if safe_stances is not None:
            unexplored_safe = [
                ap
                for ap in safe_stances
                if ap not in engine.exhausted_candidate_goals and not th.is_position_explored(ap)
            ]
            if unexplored_safe:
                for ap in unexplored_safe:
                    if ap not in target_goals:
                        target_goals.append(ap)
                        satisfied_when[ap] = {"max_visits": 1, "target_feature": feature_id}
                if transient_blocked is not None:
                    transient_blocked.add(entity_pos)
                return
        if entity_pos not in target_goals:
            target_goals.append(entity_pos)
            if set_contact_meta:
                satisfied_when[entity_pos] = {"feature": feature_id, "max_visits": 2}

    @staticmethod
    def _enqueue_adjacent_or_contact(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        H: int,
        W: int,
        bg: int,
        avatar_pos: tuple[int, int] | None,
        entity_pos: tuple[int, int],
        feature_id: int,
        target_goals: list[tuple[int, int]],
        satisfied_when: dict[tuple[int, int], dict[str, Any]],
        *,
        set_contact_meta: bool = False,
        transient_blocked: set[tuple[int, int]] | None = None,
    ) -> None:
        """Queue unprobed adjacent stances first; escalate to contact only once all fail."""
        adj_walkable = [
            (entity_pos[0] + dr, entity_pos[1] + dc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= entity_pos[0] + dr < H
            and 0 <= entity_pos[1] + dc < W
            and (
                curr_grid[entity_pos[0] + dr, entity_pos[1] + dc] == bg
                or engine.symbolic_theory.is_walkable(
                    int(curr_grid[entity_pos[0] + dr, entity_pos[1] + dc])
                )
                or (entity_pos[0] + dr, entity_pos[1] + dc) == avatar_pos
            )
        ]
        last_act = getattr(engine, "last_action", None)
        acc = getattr(engine, "acc_conflict_monitor", None)
        osc = getattr(acc, "_motor_oscillation_monitor", acc) if acc is not None else None
        acc_pos_hist = getattr(osc, "position_history", None) if osc is not None else None
        just_probed_here = (
            avatar_pos is not None
            and last_act is not None
            and not engine.is_displacement_action(last_act)
            and not engine.is_spatial_effector(last_act)
            and bool(acc_pos_hist)
            and acc_pos_hist[-1] == avatar_pos
        )
        added_adj = 0
        for ap in adj_walkable:
            if ap == avatar_pos and just_probed_here:
                engine.symbolic_theory.mark_position_explored(ap, reason="actuator_probed")
                engine.exhausted_candidate_goals.add(ap)
                continue
            if ap in engine.exhausted_candidate_goals or (
                engine.symbolic_theory.is_position_explored(ap) and ap != avatar_pos
            ):
                continue
            added_adj += 1
            if ap not in target_goals:
                target_goals.append(ap)
                satisfied_when[ap] = {
                    "feature": feature_id,
                    "entity_pos": entity_pos,
                    "max_visits": 2,
                }
        any_adj_probed = any(
            engine.symbolic_theory.is_position_actuator_probed(ap) for ap in adj_walkable
        )
        is_singleton = int(np.count_nonzero(curr_grid == feature_id)) == 1
        allow_contact_escalation = added_adj == 0 or (
            any_adj_probed and (is_singleton or feature_id in engine.symbolic_theory.goal_features)
        )
        if (
            allow_contact_escalation
            and adj_walkable
            and entity_pos not in engine.exhausted_candidate_goals
            and not engine.symbolic_theory.is_position_explored(entity_pos)
            and entity_pos not in engine.learned_barriers
            and entity_pos not in engine.current_level_lethal_positions
            and not engine.symbolic_theory.is_barrier(feature_id)
        ):
            if entity_pos not in target_goals:
                target_goals.append(entity_pos)
                if set_contact_meta or (any_adj_probed and is_singleton):
                    satisfied_when[entity_pos] = {"feature": feature_id, "max_visits": 2}

    @staticmethod
    def _filter_reachable_and_unlocked(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        H: int,
        W: int,
        bg: int,
        av_feats: set[int],
        avatar_pos: tuple[int, int],
        goals: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
        """Filter candidate goals to currently reachable stances and detect newly unlocked compartments."""
        th = engine.symbolic_theory
        hz = getattr(engine, "hazard_tracker", None)
        lethal_feats = set(getattr(th, "lethal_features", frozenset()) or ())
        if hz is not None:
            lethal_feats.update(getattr(hz, "known_lethal_features", ()) or ())
        opened_portals: set[tuple[int, int]] = set()
        for mut in getattr(engine, "state_mutations", ()):
            mut_meta = getattr(mut, "metadata", None) or {}
            for mr, mc in mut_meta.get("mutated_positions", ()):
                if 0 <= mr < H and 0 <= mc < W:
                    mv = int(curr_grid[mr, mc])
                    if mv == bg or th.is_walkable(mv):
                        opened_portals.add((mr, mc))

        if len(goals) <= 1 and not opened_portals:
            return goals, set()

        static_lethal = getattr(
            getattr(engine, "hazard_tracker", None), "static_lethal_positions", set()
        )
        hard_blocked: set[tuple[int, int]] = set(engine.learned_barriers) | set(static_lethal)
        unverified_blocked: set[tuple[int, int]] = set(hard_blocked)
        non_bg_rs, non_bg_cs = np.nonzero(curr_grid != bg)
        for r_np, c_np in zip(non_bg_rs.tolist(), non_bg_cs.tolist(), strict=False):
            if (r_np, c_np) == avatar_pos:
                continue
            val = int(curr_grid[r_np, c_np])
            if val in av_feats:
                continue
            if th.is_barrier(val) or val in lethal_feats:
                hard_blocked.add((r_np, c_np))
                unverified_blocked.add((r_np, c_np))
            elif not th.is_walkable(val) and val not in th.goal_features:
                unverified_blocked.add((r_np, c_np))

        opened_portals.difference_update(unverified_blocked)
        reachable_comp = TopologicalCutSetAnalyzer.get_reachable_component(
            avatar_pos, unverified_blocked, (H, W)
        )
        newly_unlocked: set[tuple[int, int]] = set()
        if opened_portals:
            comp_before_unlock = TopologicalCutSetAnalyzer.get_reachable_component(
                avatar_pos, unverified_blocked | opened_portals, (H, W)
            )
            newly_unlocked = reachable_comp - comp_before_unlock

        if len(goals) > 1:
            reachable_goals = [
                g
                for g in goals
                if g in reachable_comp
                or (
                    g not in hard_blocked
                    and any(
                        (g[0] + dr, g[1] + dc) in reachable_comp
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    )
                )
            ]
            if reachable_goals:
                goals = reachable_goals

        return goals, newly_unlocked

    @staticmethod
    def _role_scores_from_priors(
        role_priors: dict[int, dict[str, float]],
        feat_id: int,
        has_cargo_ctx: bool,
    ) -> tuple[float, float, float]:
        """Return (preferred_role_score, barrier_score, hazard_score) in [0, 1]."""
        priors = role_priors.get(feat_id)
        if not priors:
            return (0.0, 0.0, 0.0)
        barrier_score = float(priors.get("barrier", 0.0))
        hazard_score = float(priors.get("hazard", 0.0))
        goal_score = max(
            float(priors.get("goal", 0.0)),
            float(priors.get("receptacle", 0.0)),
        )
        cargo_score = float(priors.get("cargo", 0.0))
        pref_score = goal_score if has_cargo_ctx else max(goal_score, cargo_score)
        return (pref_score, barrier_score, hazard_score)

    @classmethod
    def _is_prior_hazard_feat(
        cls,
        engine: AutonomousEpistemicEngine,
        feat_id: int,
    ) -> bool:
        """Return True if unverified perceptual role priors classify ``feat_id`` as hazardous."""
        th = engine.symbolic_theory
        if feat_id in th.goal_features or feat_id in th.cargo_features or th.is_walkable(feat_id):
            return False
        wm_priors: dict[int, dict[str, float]] = (
            getattr(getattr(engine, "working_memory", None), "concept_role_priors", None) or {}
        )
        has_cargo_ctx = bool(th.cargo_features or engine.learned_cargo_features)
        pref_s, _, haz_s = cls._role_scores_from_priors(wm_priors, int(feat_id), has_cargo_ctx)
        return haz_s >= 0.65 and pref_s < 0.45

    @classmethod
    def _high_confidence_prior_hazards(
        cls,
        engine: AutonomousEpistemicEngine,
        bg: int,
        av_feats: set[int],
    ) -> set[int]:
        """Return unverified feature IDs whose perceptual priors strongly indicate hazard."""
        wm_priors: dict[int, dict[str, float]] = (
            getattr(getattr(engine, "working_memory", None), "concept_role_priors", None) or {}
        )
        return {
            int(f)
            for f in wm_priors
            if int(f) != bg and int(f) not in av_feats and cls._is_prior_hazard_feat(engine, int(f))
        }

    @classmethod
    def schedule_goals(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        entities: Sequence[Any],
        bg: int,
        av_feats: set[int],
        target_role: EntityRole = EntityRole.GOAL,
        available_actions: Sequence[int] = (),
    ) -> EpistemicGoalProposal:
        H, W = curr_grid.shape
        avatar_pos = engine.avatar_pos
        new_learned_positions: set[tuple[int, int]] = set()
        new_candidate_feats: set[int] = set()
        subgoal_to_pop = False
        clear_exhausted = False
        satisfied_when: dict[tuple[int, int], dict[str, Any]] = {}
        transient_blocked: set[tuple[int, int]] = set()

        # 1. Confirmed goals — causal evidence from past wins
        confirmed_goals: list[tuple[int, int]] = []
        has_cargo_context = bool(
            engine.symbolic_theory.cargo_features or engine.learned_cargo_features
        )
        for e in entities:
            if avatar_pos is not None and e.grid_pos == avatar_pos and not has_cargo_context:
                continue
            if e.feature_id == bg:
                continue
            if (
                e.role == EntityRole.OBSTACLE
                or engine.symbolic_theory.is_barrier(e.feature_id)
                or e.grid_pos in engine.learned_barriers
            ):
                continue
            if (
                e.feature_id in engine.mobile_threat_features
                or e.feature_id in engine.hazard_tracker.known_lethal_features
            ):
                continue
            if e.feature_id in av_feats:
                if (
                    avatar_pos is not None
                    and (abs(e.grid_pos[0] - avatar_pos[0]) + abs(e.grid_pos[1] - avatar_pos[1]))
                    > 4
                    and e.role == EntityRole.GOAL
                ):
                    confirmed_goals.append(e.grid_pos)
                continue
            # Causal Disambiguation: Pervasive walkable substrates (e.g. tracks, floors)
            # that recur in large quantities cannot be the singular exit/goal.
            feat_total_cells = int(np.count_nonzero(curr_grid == e.feature_id))
            feat_instance_count = sum(1 for ent in entities if ent.feature_id == e.feature_id)
            max_entity_area = max(64, int(H * W * 0.15))
            is_pervasive_substrate = (
                e.feature_id in engine.symbolic_theory.walkable_features
            ) and (feat_instance_count > 2 or feat_total_cells > max_entity_area)
            if is_pervasive_substrate:
                engine.symbolic_theory.retract_goal(e.feature_id, reason="pervasive_substrate")
                continue

            has_discrete_interact_cg = any(
                not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
                for a in available_actions
            )
            is_goal_candidate = (
                e.feature_id in engine.symbolic_theory.goal_features
                and (
                    1 <= e.area <= max_entity_area
                    or not engine.symbolic_theory.is_walkable(e.feature_id)
                )
            ) or (e.role == EntityRole.GOAL and not has_discrete_interact_cg)
            if is_goal_candidate:
                if e.grid_pos not in confirmed_goals:
                    confirmed_goals.append(e.grid_pos)
                    if e.feature_id in engine.symbolic_theory.goal_features:
                        new_learned_positions.add(e.grid_pos)

        # Object Permanence: Retain known goal positions if currently occluded by mobile entities
        for gp in engine.learned_goal_positions:
            if (
                0 <= gp[0] < H
                and 0 <= gp[1] < W
                and gp not in engine.learned_barriers
                and (gp != avatar_pos or has_cargo_context)
            ):
                if av_feats and int(curr_grid[gp[0], gp[1]]) in av_feats and not has_cargo_context:
                    continue
                if gp not in confirmed_goals:
                    confirmed_goals.append(gp)

        # Goal Arbiter active goal integration (confirmed extrinsic targets, or live valid
        # epistemic targets when not yet in block-delivery mode)
        arbitrated = getattr(engine, "active_arbitrated_goal", None)
        if (
            arbitrated is not None
            and isinstance(arbitrated.target, tuple)
            and len(arbitrated.target) == 2
        ):
            target_coord = (int(arbitrated.target[0]), int(arbitrated.target[1]))
            if (
                0 <= target_coord[0] < H
                and 0 <= target_coord[1] < W
                and target_coord != avatar_pos
                and target_coord not in engine.learned_barriers
                and not engine.symbolic_theory.is_position_explored(target_coord)
            ):
                tc_feat = int(curr_grid[target_coord[0], target_coord[1]])
                arb_meta = getattr(arbitrated, "metadata", None) or {}
                exp_feat = arb_meta.get("feature_id")
                arb_type = getattr(arbitrated, "goal_type", "")
                # Discard stale epistemic targets whose object moved, vanished into bg,
                # or turned out to be cargo / barrier rather than a goal.
                is_stale_epistemic = arb_type == "epistemic_exploration" and (
                    (exp_feat is not None and tc_feat != int(exp_feat))
                    or tc_feat == bg
                    or tc_feat in engine.symbolic_theory.cargo_features
                    or engine.symbolic_theory.is_cargo(tc_feat)
                    or engine.symbolic_theory.is_feature_explored(tc_feat)
                )
                is_tc_barrier = (
                    engine.symbolic_theory.is_barrier(tc_feat) or tc_feat in engine.learned_barriers
                )
                has_discrete_interact = any(
                    not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
                    for a in available_actions
                )
                is_extrinsic_arb = (
                    tc_feat in engine.symbolic_theory.goal_features
                    or target_coord in engine.learned_goal_positions
                    or arb_type in ("confirmed_extrinsic", "social_instruction")
                    or (
                        not has_discrete_interact
                        and arb_type != "epistemic_exploration"
                        and not has_cargo_context
                    )
                )
                if not is_stale_epistemic and is_extrinsic_arb:
                    if not is_tc_barrier and not (
                        has_discrete_interact and not engine.symbolic_theory.is_walkable(tc_feat)
                    ):
                        target_area = next(
                            (e.area for e in entities if e.grid_pos == target_coord),
                            1,
                        )
                        cls._enqueue_walkable_or_reversible_approach(
                            engine,
                            curr_grid,
                            H,
                            W,
                            bg,
                            avatar_pos,
                            target_coord,
                            tc_feat,
                            target_area,
                            has_cargo_context,
                            confirmed_goals,
                            satisfied_when,
                            set_contact_meta=False,
                            transient_blocked=transient_blocked,
                        )
                        if target_coord in confirmed_goals:
                            confirmed_goals.remove(target_coord)
                            confirmed_goals.insert(0, target_coord)
                    else:
                        # Unwalkable arbitrated target: target closest adjacent walkable stance
                        adj_walkable = [
                            (target_coord[0] + dr, target_coord[1] + dc)
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            if 0 <= target_coord[0] + dr < H
                            and 0 <= target_coord[1] + dc < W
                            and (
                                not engine.symbolic_theory.is_position_explored(
                                    (target_coord[0] + dr, target_coord[1] + dc)
                                )
                                or (target_coord[0] + dr, target_coord[1] + dc) == avatar_pos
                            )
                            and (
                                curr_grid[target_coord[0] + dr, target_coord[1] + dc] == bg
                                or engine.symbolic_theory.is_walkable(
                                    int(curr_grid[target_coord[0] + dr, target_coord[1] + dc])
                                )
                                or (target_coord[0] + dr, target_coord[1] + dc) == avatar_pos
                            )
                        ]
                        if adj_walkable:
                            if avatar_pos is not None:
                                adj_walkable.sort(
                                    key=lambda p: (
                                        abs(p[0] - avatar_pos[0]) + abs(p[1] - avatar_pos[1])
                                    )
                                )
                            chosen_stance = adj_walkable[0]
                            if chosen_stance in confirmed_goals:
                                confirmed_goals.remove(chosen_stance)
                            confirmed_goals.insert(0, chosen_stance)
                            satisfied_when[chosen_stance] = {
                                "feature": tc_feat,
                                "entity_pos": target_coord,
                                "max_visits": 2,
                            }

        goals: list[tuple[int, int]] = []
        if confirmed_goals:
            goals = list(confirmed_goals)

        # Causal Prerequisite Subgoal Scheduling via Conditional Affordances (BA10 / OFC)
        if (
            goals
            and avatar_pos is not None
            and engine.symbolic_theory.conditional_affordances
            and not engine.subgoal_stack.has_pending_goals
        ):
            cond_affs = engine.symbolic_theory.conditional_affordances
            held_ids = getattr(engine.working_memory, "held_feature_ids", set())
            hz = getattr(engine, "hazard_tracker", None)
            lethal_cond_feats = set(getattr(hz, "known_lethal_features", ()) or ()) if hz else set()
            unfulfilled = [
                aff
                for aff in cond_affs.values()
                if aff.prerequisite_feature is not None
                and aff.prerequisite_feature != aff.feature
                and aff.prerequisite_feature != bg
                and aff.prerequisite_feature not in av_feats
                and aff.prerequisite_feature not in lethal_cond_feats
                and aff.prerequisite_feature not in held_ids
                and np.any(curr_grid == aff.feature)
            ]
            if unfulfilled:
                aff_features = {a.feature for a in unfulfilled}
                primary_target = goals[0]
                q_decomp: deque[tuple[int, int]] = deque([avatar_pos])
                visited_decomp: set[tuple[int, int]] = {avatar_pos}
                parent_decomp: dict[tuple[int, int], tuple[int, int]] = {}
                reached_target = False
                while q_decomp:
                    cp = q_decomp.popleft()
                    if cp == primary_target:
                        reached_target = True
                        break
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cp[0] + dr, cp[1] + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited_decomp:
                            cell_v = int(curr_grid[nr, nc])
                            if (
                                cell_v != bg
                                and engine.symbolic_theory.is_barrier(cell_v)
                                and cell_v not in aff_features
                            ):
                                continue
                            visited_decomp.add((nr, nc))
                            parent_decomp[(nr, nc)] = cp
                            q_decomp.append((nr, nc))

                if reached_target:
                    curr_track = primary_target
                    path_to_tgt: list[tuple[int, int]] = []
                    while curr_track in parent_decomp:
                        path_to_tgt.append(curr_track)
                        curr_track = parent_decomp[curr_track]
                    path_to_tgt.reverse()

                    blocking_aff: Any | None = None
                    barrier_pos_in_path: tuple[int, int] | None = None
                    for cell in path_to_tgt:
                        cv = int(curr_grid[cell[0], cell[1]])
                        if cv in aff_features:
                            blocking_aff = next((a for a in unfulfilled if a.feature == cv), None)
                            barrier_pos_in_path = cell
                            break

                    guarded_barrier_pos: tuple[int, int] | None = barrier_pos_in_path
                    known_trigger_pos: tuple[int, int] | None = None
                    if (
                        barrier_pos_in_path is not None
                        and hasattr(engine, "remote_causal")
                        and blocking_aff is not None
                    ):
                        r_affs = engine.remote_causal.remote_actuators.get(barrier_pos_in_path, [])
                        for ra in r_affs:
                            if ra.trigger_feature == blocking_aff.prerequisite_feature:
                                if (
                                    0 <= ra.trigger_pos[0] < H
                                    and 0 <= ra.trigger_pos[1] < W
                                    and int(curr_grid[ra.trigger_pos[0], ra.trigger_pos[1]])
                                    == ra.trigger_feature
                                ):
                                    known_trigger_pos = ra.trigger_pos
                                    break
                    # Distal Causal Gatekeeper: Check if primary_target is guarded by an unactivated remote barrier
                    if blocking_aff is None and hasattr(engine, "remote_causal"):
                        for rem_pos, aff_list in engine.remote_causal.remote_actuators.items():
                            if (
                                abs(rem_pos[0] - primary_target[0])
                                + abs(rem_pos[1] - primary_target[1])
                                <= 1
                            ):
                                if 0 <= rem_pos[0] < H and 0 <= rem_pos[1] < W:
                                    cur_rf = int(curr_grid[rem_pos[0], rem_pos[1]])
                                    unfulfilled_prereqs = {
                                        a.prerequisite_feature
                                        for a in unfulfilled
                                        if a.prerequisite_feature is not None
                                    }
                                    ordered_affs = sorted(
                                        aff_list,
                                        key=lambda ra: (
                                            0 if ra.trigger_feature in unfulfilled_prereqs else 1,
                                            -getattr(ra, "confidence", 0.0),
                                        ),
                                    )
                                    for r_aff in ordered_affs:
                                        # Reject stale links whose trigger cell no longer
                                        # holds the trigger feature on this lattice.
                                        if not (
                                            0 <= r_aff.trigger_pos[0] < H
                                            and 0 <= r_aff.trigger_pos[1] < W
                                            and int(
                                                curr_grid[
                                                    r_aff.trigger_pos[0], r_aff.trigger_pos[1]
                                                ]
                                            )
                                            == r_aff.trigger_feature
                                        ):
                                            continue
                                        if (
                                            cur_rf == r_aff.original_remote_feature
                                            and r_aff.trigger_feature is not None
                                        ):
                                            # Skip avatar's own feature
                                            if (
                                                engine.avatar_feature is not None
                                                and r_aff.trigger_feature == engine.avatar_feature
                                            ):
                                                continue
                                            blocking_aff = next(
                                                (
                                                    a
                                                    for a in unfulfilled
                                                    if a.prerequisite_feature
                                                    == r_aff.trigger_feature
                                                ),
                                                None,
                                            )
                                            if blocking_aff is None:
                                                from hbllm.hcir.world.causal_discovery import (
                                                    CausalPredicate,
                                                )
                                                from hbllm.hcir.world.cortex_causal import (
                                                    ConditionalAffordance,
                                                )

                                                blocking_aff = ConditionalAffordance(
                                                    feature=r_aff.original_remote_feature,
                                                    condition=CausalPredicate(
                                                        variable="trigger",
                                                        operator="==",
                                                        value=r_aff.trigger_feature,
                                                    ),
                                                    prerequisite_feature=r_aff.trigger_feature,
                                                )
                                            guarded_barrier_pos = rem_pos
                                            if (
                                                0 <= r_aff.trigger_pos[0] < H
                                                and 0 <= r_aff.trigger_pos[1] < W
                                                and int(
                                                    curr_grid[
                                                        r_aff.trigger_pos[0], r_aff.trigger_pos[1]
                                                    ]
                                                )
                                                == r_aff.trigger_feature
                                            ):
                                                known_trigger_pos = r_aff.trigger_pos
                                            break
                                if blocking_aff is not None:
                                    break

                    # Also check interior cells adjacent to primary_target for unfulfilled conditional affordances
                    if blocking_aff is None and unfulfilled:
                        for dr_g, dc_g in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr_g, nc_g = primary_target[0] + dr_g, primary_target[1] + dc_g
                            if 1 <= nr_g < H - 1 and 1 <= nc_g < W - 1:
                                cv_g = int(curr_grid[nr_g, nc_g])
                                if cv_g in aff_features:
                                    blocking_aff = next(
                                        (a for a in unfulfilled if a.feature == cv_g), None
                                    )
                                    guarded_barrier_pos = (nr_g, nc_g)
                                    break

                    if blocking_aff is not None and blocking_aff.prerequisite_feature is not None:
                        prereq_feat = blocking_aff.prerequisite_feature

                        def _walkable_neighbors(coord: tuple[int, int]) -> int:
                            return sum(
                                1
                                for dr_w, dc_w in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                if 0 <= coord[0] + dr_w < H
                                and 0 <= coord[1] + dc_w < W
                                and (
                                    int(curr_grid[coord[0] + dr_w, coord[1] + dc_w]) == bg
                                    or engine.symbolic_theory.is_walkable(
                                        int(curr_grid[coord[0] + dr_w, coord[1] + dc_w])
                                    )
                                    or (coord[0] + dr_w, coord[1] + dc_w) == avatar_pos
                                )
                            )

                        # Relational Cue Resonance: If the conditional affordance was induced from a
                        # relational cue pair (count == 2) and prereq_feat no longer has count == 2
                        # on this lattice, bind prereq_feat to the current cue-matched interior actuator.
                        is_rel_aff = (
                            str(getattr(blocking_aff, "evidence", "")).startswith("relational_cue:")
                            or getattr(blocking_aff.condition, "relation", "")
                            == "cue_actuator_resonance"
                        )
                        if not is_rel_aff and hasattr(engine, "working_memory"):
                            hyp_eng = getattr(engine.working_memory, "hypothesis_engine", None)
                            if hyp_eng is not None:
                                is_rel_aff = any(
                                    h.target_feature == blocking_aff.feature
                                    and h.rule_id.startswith("relational_cue_")
                                    for h in hyp_eng.hypotheses.values()
                                )
                        if is_rel_aff and int(np.count_nonzero(curr_grid == prereq_feat)) != 2:
                            for e_cand in entities:
                                cf = e_cand.feature_id
                                if (
                                    cf != bg
                                    and cf not in av_feats
                                    and cf != blocking_aff.feature
                                    and not engine.symbolic_theory.is_barrier(cf)
                                    and e_cand.grid_pos not in engine.quiescent_actuation_targets
                                    and int(np.count_nonzero(curr_grid == cf)) == 2
                                    and _walkable_neighbors(e_cand.grid_pos) == 4
                                ):
                                    prereq_feat = cf
                                    known_trigger_pos = None
                                    break

                        best_prereq_pos = known_trigger_pos
                        if best_prereq_pos is None:
                            prereq_coords = [
                                e.grid_pos
                                for e in entities
                                if e.feature_id == prereq_feat
                                and e.grid_pos not in engine.quiescent_actuation_targets
                            ]
                            if not prereq_coords:
                                y_idxs, x_idxs = np.where(curr_grid == prereq_feat)
                                prereq_coords = [
                                    (int(y), int(x))
                                    for y, x in zip(y_idxs, x_idxs)
                                    if (int(y), int(x)) not in engine.quiescent_actuation_targets
                                ]

                            if prereq_coords:
                                prereq_coords.sort(
                                    key=lambda c: (
                                        -_walkable_neighbors(c),
                                        abs(c[0] - avatar_pos[0]) + abs(c[1] - avatar_pos[1]),
                                    )
                                )
                                best_prereq_pos = prereq_coords[0]

                        if best_prereq_pos is not None:
                            has_discrete_interact = any(
                                not engine.is_displacement_action(a)
                                and not engine.is_spatial_effector(a)
                                for a in available_actions
                            )
                            is_prereq_walkable = engine.symbolic_theory.is_walkable(prereq_feat)
                            is_adjacent_actuator = has_discrete_interact and not is_prereq_walkable
                            nav_target_pos = best_prereq_pos
                            if is_adjacent_actuator:
                                # Find closest walkable adjacent cell to the actuator/switch
                                adj_candidates: list[tuple[int, int]] = []
                                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                    ar, ac = best_prereq_pos[0] + dr, best_prereq_pos[1] + dc
                                    if 0 <= ar < H and 0 <= ac < W:
                                        af = int(curr_grid[ar, ac])
                                        if (
                                            af == bg
                                            or engine.symbolic_theory.is_walkable(af)
                                            or (ar, ac) == avatar_pos
                                        ):
                                            adj_candidates.append((ar, ac))
                                if adj_candidates:

                                    def _other_adj_entities(stance: tuple[int, int]) -> int:
                                        cnt = 0
                                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                            nr2, nc2 = stance[0] + dr2, stance[1] + dc2
                                            if (
                                                0 <= nr2 < H
                                                and 0 <= nc2 < W
                                                and (nr2, nc2) != best_prereq_pos
                                            ):
                                                nf = int(curr_grid[nr2, nc2])
                                                if (
                                                    nf != bg
                                                    and nf not in av_feats
                                                    and not engine.symbolic_theory.is_walkable(nf)
                                                    and not engine.symbolic_theory.is_barrier(nf)
                                                ):
                                                    cnt += 1
                                        return cnt

                                    adj_candidates.sort(
                                        key=lambda c: (
                                            _other_adj_entities(c),
                                            abs(c[0] - avatar_pos[0]) + abs(c[1] - avatar_pos[1]),
                                        )
                                    )
                                    nav_target_pos = adj_candidates[0]

                            sg_type = (
                                SubgoalType.UNLOCK_REMOTE_MECHANISM
                                if is_adjacent_actuator
                                else SubgoalType.NAVIGATE_TO
                            )
                            engine.schedule_causal_subgoal_chain(
                                target_destination=primary_target,
                                target_type=SubgoalType.DELIVER_TO_GOAL,
                                causal_prerequisites=[
                                    (
                                        f"ACTUATE_{prereq_feat}"
                                        if is_adjacent_actuator
                                        else f"NAVIGATE_TO_{prereq_feat}",
                                        nav_target_pos,
                                        sg_type,
                                        prereq_feat,
                                    )
                                ],
                                target_id=f"target_goal_{primary_target[0]}_{primary_target[1]}",
                            )
                            if engine.subgoal_stack.has_pending_goals:
                                cur_sg = engine.subgoal_stack.current_subgoal
                                if cur_sg is not None:
                                    cur_sg.target_entity_pos = best_prereq_pos
                                    if guarded_barrier_pos is not None:
                                        cur_sg.metadata["barrier_pos"] = guarded_barrier_pos
                                        cur_sg.metadata["barrier_feat"] = blocking_aff.feature

                            if is_adjacent_actuator:
                                satisfied_when[nav_target_pos] = {
                                    "feature": prereq_feat,
                                    "entity_pos": best_prereq_pos,
                                    "max_visits": 5,
                                    "role": "actuator",
                                }

                            goals = [nav_target_pos]

        # Faculty: Prefrontal Working Memory Active Schema & Procedural Skill Prioritization (dlPFC / Habitual Replay)
        wm = get_engine_state_attr(engine, "working_memory", None)
        if wm is not None:
            if wm.active_schema is not None and not wm.active_schema.is_finished:
                if wm.active_schema.is_current_stage_satisfied(
                    avatar_pos=avatar_pos,
                    held_items=wm.held_items,
                    unlocked_barriers=wm.unlocked_barriers,
                ):
                    wm.active_schema.advance()

                curr_st = wm.active_schema.current_stage
                if curr_st is not None and curr_st.target_positions:
                    for sp in reversed(list(curr_st.target_positions)):
                        if 0 <= sp[0] < H and 0 <= sp[1] < W and sp != avatar_pos:
                            if sp in goals:
                                goals.remove(sp)
                            goals.insert(0, sp)
            elif wm.active_schema is None and wm.procedural_library:
                matched = wm.find_matching_skill(
                    available_actions=available_actions,
                    context={"current_level": get_engine_state_attr(engine, "current_level", 0)},
                )
                if matched is not None:
                    skill_name, _ = matched
                    schema = wm.instantiate_skill_as_schema(skill_name)
                    if schema and schema.current_stage and schema.current_stage.target_positions:
                        valid_schema_targets = [
                            sp
                            for sp in schema.current_stage.target_positions
                            if 0 <= sp[0] < H
                            and 0 <= sp[1] < W
                            and sp != avatar_pos
                            and (
                                not confirmed_goals
                                or sp in confirmed_goals
                                or any(e.grid_pos == sp for e in entities)
                            )
                        ]
                        for sp in reversed(valid_schema_targets):
                            if sp in goals:
                                goals.remove(sp)
                            goals.insert(0, sp)

        # Faculty: Brodmann Area 10 (aPFC) Frontopolar Subgoal Prioritization
        if engine.subgoal_stack.has_pending_goals:
            cur_sg = engine.subgoal_stack.current_subgoal
            held_ids = getattr(engine.working_memory, "held_feature_ids", set())
            is_sg_sat = (
                engine.subgoal_stack.is_subgoal_satisfied(
                    cur_sg,
                    set(),
                    avatar_pos=avatar_pos,
                    held_features=held_ids,
                    grid=curr_grid,
                    background_feature=bg,
                )
                if cur_sg is not None
                else False
            )
            if cur_sg is not None and cur_sg.target_destination != avatar_pos and not is_sg_sat:
                sg_dest = cur_sg.target_destination
                if 0 <= sg_dest[0] < H and 0 <= sg_dest[1] < W:
                    goals = [sg_dest]
                    if (
                        cur_sg.subgoal_type == SubgoalType.UNLOCK_REMOTE_MECHANISM
                        and cur_sg.target_entity_pos is not None
                    ):
                        satisfied_when[sg_dest] = {
                            "feature": cur_sg.required_feature,
                            "entity_pos": cur_sg.target_entity_pos,
                            "max_visits": 2,
                            "role": "actuator",
                        }
            elif cur_sg is not None and (
                is_sg_sat
                or (
                    cur_sg.target_destination == avatar_pos
                    and cur_sg.subgoal_type != SubgoalType.UNLOCK_REMOTE_MECHANISM
                )
            ):
                subgoal_to_pop = True
                if len(engine.subgoal_stack.stack) > 1:
                    next_sg_dest = engine.subgoal_stack.stack[-2].target_destination
                    if 0 <= next_sg_dest[0] < H and 0 <= next_sg_dest[1] < W:
                        if next_sg_dest in goals:
                            goals.remove(next_sg_dest)
                        goals.insert(0, next_sg_dest)
            elif cur_sg is not None and cur_sg.target_destination == avatar_pos:
                # Avatar is at stance waiting for unlock actuation
                goals = [avatar_pos]

        # Panel coordinates filter
        panel_coords: set[tuple[int, int]] = set()
        if any(engine.is_spatial_effector(a) for a in available_actions):
            panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
            for p in panels:
                for item in p["items"]:
                    r0, r1, c0, c1 = item.bounding_box
                    for rr in range(max(0, r0 - 2), min(H, r1 + 3)):
                        for cc in range(max(0, c0 - 2), min(W, c1 + 3)):
                            panel_coords.add((rr, cc))

        # (b) Hypothesized goals from previous level knowledge or role inference
        if not goals:
            hypothesized_goals: list[tuple[int, int]] = []
            for e in entities:
                if avatar_pos is not None and e.grid_pos == avatar_pos and not has_cargo_context:
                    continue
                if e.feature_id == bg:
                    continue
                if e.grid_pos in engine.learned_barriers and (
                    e.feature_id in engine.symbolic_theory.goal_features
                    or e.grid_pos in engine.learned_goal_positions
                ):
                    continue
                if (
                    engine.symbolic_theory.is_barrier(e.feature_id)
                    and e.feature_id not in engine.symbolic_theory.candidate_goal_features
                    and e.feature_id not in engine.symbolic_theory.goal_features
                ) or (
                    e.role == EntityRole.OBSTACLE
                    and e.feature_id not in engine.symbolic_theory.candidate_goal_features
                    and e.feature_id not in engine.symbolic_theory.goal_features
                ):
                    continue
                is_cargo_entity = (
                    e.feature_id in engine.symbolic_theory.cargo_features
                    or e.feature_id in engine.learned_cargo_features
                )
                if engine.symbolic_theory.is_position_explored(e.grid_pos) and (
                    not has_cargo_context or is_cargo_entity
                ):
                    continue
                if e.grid_pos in engine.exhausted_candidate_goals and (
                    not has_cargo_context or is_cargo_entity
                ):
                    continue
                if (
                    e.feature_id in engine.hazard_tracker.known_lethal_features
                    or e.feature_id in engine.mobile_threat_features
                    or cls._is_prior_hazard_feat(engine, int(e.feature_id))
                ):
                    continue
                if avatar_pos is not None and e.feature_id in av_feats:
                    if (
                        abs(e.grid_pos[0] - avatar_pos[0]) <= 3
                        and abs(e.grid_pos[1] - avatar_pos[1]) <= 3
                    ):
                        continue
                elif e.feature_id in av_feats:
                    continue
                if e.grid_pos in panel_coords:
                    continue
                max_entity_area = max(64, int(H * W * 0.15))
                if (
                    e.feature_id in engine.symbolic_theory.goal_features
                    and (
                        1 <= e.area <= max_entity_area
                        or not engine.symbolic_theory.is_walkable(e.feature_id)
                    )
                ) or (
                    e.feature_id in engine.symbolic_theory.candidate_goal_features
                    and (
                        1 <= e.area <= max_entity_area
                        or not engine.symbolic_theory.is_walkable(e.feature_id)
                    )
                ):
                    has_discrete_interact = any(
                        not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
                        for a in available_actions
                    )
                    is_walk = (
                        engine.symbolic_theory.is_walkable(e.feature_id)
                        or (
                            not has_discrete_interact
                            and not engine.symbolic_theory.is_barrier(e.feature_id)
                        )
                        or e.feature_id in engine.symbolic_theory.goal_features
                        or e.grid_pos in engine.learned_goal_positions
                    ) and e.grid_pos not in engine.learned_barriers
                    if is_walk:
                        cls._enqueue_walkable_or_reversible_approach(
                            engine,
                            curr_grid,
                            H,
                            W,
                            bg,
                            avatar_pos,
                            e.grid_pos,
                            e.feature_id,
                            e.area,
                            has_cargo_context,
                            hypothesized_goals,
                            satisfied_when,
                            set_contact_meta=False,
                            transient_blocked=transient_blocked,
                        )
                    else:
                        cls._enqueue_adjacent_or_contact(
                            engine,
                            curr_grid,
                            H,
                            W,
                            bg,
                            avatar_pos,
                            e.grid_pos,
                            e.feature_id,
                            hypothesized_goals,
                            satisfied_when,
                        )
                elif (
                    e.role == target_role
                    or e.feature_id in engine.learned_receptacle_features
                    or e.grid_pos in engine.learned_goal_positions
                    or e.grid_pos in engine.learned_receptacle_positions
                ):
                    has_discrete_interact = any(
                        not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
                        for a in available_actions
                    )
                    is_walk = (
                        engine.symbolic_theory.is_walkable(e.feature_id)
                        or (
                            not has_discrete_interact
                            and not engine.symbolic_theory.is_barrier(e.feature_id)
                        )
                        or e.feature_id in engine.symbolic_theory.goal_features
                        or e.grid_pos in engine.learned_goal_positions
                    ) and e.grid_pos not in engine.learned_barriers
                    if is_walk:
                        cls._enqueue_walkable_or_reversible_approach(
                            engine,
                            curr_grid,
                            H,
                            W,
                            bg,
                            avatar_pos,
                            e.grid_pos,
                            e.feature_id,
                            e.area,
                            has_cargo_context,
                            hypothesized_goals,
                            satisfied_when,
                            set_contact_meta=False,
                            transient_blocked=transient_blocked,
                        )
                    else:
                        cls._enqueue_adjacent_or_contact(
                            engine,
                            curr_grid,
                            H,
                            W,
                            bg,
                            avatar_pos,
                            e.grid_pos,
                            e.feature_id,
                            hypothesized_goals,
                            satisfied_when,
                        )
            if hypothesized_goals:
                goals = hypothesized_goals

        # (c) Real Unvisited Novel / Interactive Entities on the Grid (Epistemic Curiosity)
        has_discrete_interact = any(
            not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
            for a in available_actions
        )
        if not goals:
            novel_entity_goals: list[tuple[int, int]] = []
            for e in entities:
                if avatar_pos is not None and e.grid_pos == avatar_pos and not has_cargo_context:
                    continue
                if e.feature_id == bg:
                    continue
                if e.feature_id in av_feats:
                    continue
                if e.grid_pos in panel_coords:
                    continue
                is_isolated_interior = (
                    1 <= e.area <= 2
                    and 1 <= e.grid_pos[0] < H - 1
                    and 1 <= e.grid_pos[1] < W - 1
                    and has_discrete_interact
                    and not engine.symbolic_theory.is_position_explored(e.grid_pos)
                    and e.feature_id not in engine.symbolic_theory.goal_features
                )
                if (
                    (engine.symbolic_theory.is_barrier(e.feature_id) and not is_isolated_interior)
                    or (e.role == EntityRole.OBSTACLE and not is_isolated_interior)
                    or e.feature_id in engine.hazard_tracker.known_lethal_features
                    or e.feature_id in engine.mobile_threat_features
                    or cls._is_prior_hazard_feat(engine, int(e.feature_id))
                ):
                    continue
                is_cargo_entity = (
                    e.feature_id in engine.symbolic_theory.cargo_features
                    or e.feature_id in engine.learned_cargo_features
                )
                is_potential_cargo_receptor = (
                    has_cargo_context
                    and not is_cargo_entity
                    and (
                        e.feature_id in engine.symbolic_theory.goal_features
                        or e.feature_id in engine.symbolic_theory.candidate_goal_features
                        or e.feature_id in engine.symbolic_theory.receptacle_features
                        or (
                            engine.symbolic_theory.is_walkable(e.feature_id)
                            and int(np.count_nonzero(curr_grid == e.feature_id)) <= 4
                        )
                    )
                    and e.grid_pos not in engine.learned_barriers
                    and e.grid_pos not in engine.current_level_lethal_positions
                )
                if (
                    engine.symbolic_theory.is_feature_explored(e.feature_id)
                    and not is_isolated_interior
                    and not is_potential_cargo_receptor
                ):
                    continue
                if engine.symbolic_theory.is_position_explored(e.grid_pos) and (
                    not has_cargo_context or is_cargo_entity
                ):
                    continue
                if e.grid_pos in engine.exhausted_candidate_goals and (
                    not has_cargo_context or is_cargo_entity
                ):
                    continue
                if not (1 <= e.area <= 64):
                    continue
                is_walkable = (
                    engine.symbolic_theory.is_walkable(e.feature_id)
                    or (
                        not has_discrete_interact
                        and not engine.symbolic_theory.is_barrier(e.feature_id)
                    )
                    or e.feature_id == bg
                )
                if is_walkable:
                    cls._enqueue_walkable_or_reversible_approach(
                        engine,
                        curr_grid,
                        H,
                        W,
                        bg,
                        avatar_pos,
                        e.grid_pos,
                        e.feature_id,
                        e.area,
                        has_cargo_context,
                        novel_entity_goals,
                        satisfied_when,
                        set_contact_meta=True,
                        transient_blocked=transient_blocked,
                    )
                else:
                    # Non-walkable interactive entity: probe adjacent stances, then contact if exhausted
                    cls._enqueue_adjacent_or_contact(
                        engine,
                        curr_grid,
                        H,
                        W,
                        bg,
                        avatar_pos,
                        e.grid_pos,
                        e.feature_id,
                        novel_entity_goals,
                        satisfied_when,
                        set_contact_meta=True,
                        transient_blocked=transient_blocked,
                    )
                if is_walkable:
                    new_candidate_feats.add(e.feature_id)

            if novel_entity_goals:
                goals = novel_entity_goals

        if avatar_pos is not None and goals and satisfied_when:
            th = engine.symbolic_theory
            role_priors: dict[int, dict[str, float]] = (
                getattr(getattr(engine, "working_memory", None), "concept_role_priors", None) or {}
            )
            has_cargo_ctx = bool(th.cargo_features or engine.learned_cargo_features)
            goals, newly_unlocked = cls._filter_reachable_and_unlocked(
                engine, curr_grid, H, W, bg, av_feats, avatar_pos, goals
            )

            def _role_score_for_feature(feat_id: int) -> tuple[float, float, float]:
                """Return (preferred_role_score, barrier_score, hazard_score) in [0, 1]."""
                return cls._role_scores_from_priors(role_priors, feat_id, has_cargo_ctx)

            def _stance_priority(p: tuple[int, int]) -> tuple[int, float, int]:
                dist = abs(p[0] - avatar_pos[0]) + abs(p[1] - avatar_pos[1])
                if p == avatar_pos and p in satisfied_when and "entity_pos" in satisfied_when[p]:
                    return (0, 0.0, 0)
                if newly_unlocked and (
                    p in newly_unlocked
                    or any(
                        (p[0] + dr, p[1] + dc) in newly_unlocked
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    )
                ):
                    return (1, 0.0, dist)
                meta = satisfied_when.get(p)
                if meta and "entity_pos" in meta:
                    ep = meta["entity_pos"]
                    raw_ef = meta.get("feature")
                    ef = int(raw_ef) if raw_ef is not None else -1
                    pref_score, barrier_score, hazard_score = (
                        _role_score_for_feature(ef) if ef >= 0 else (0.0, 0.0, 0.0)
                    )
                    if ef >= 0:
                        if hazard_score >= 0.65 and pref_score < 0.45:
                            return (8, hazard_score, dist)
                        if barrier_score >= 0.72 and pref_score < 0.40:
                            return (7, barrier_score, dist)
                        walk_n = sum(
                            1
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            if 0 <= ep[0] + dr < H
                            and 0 <= ep[1] + dc < W
                            and (
                                curr_grid[ep[0] + dr, ep[1] + dc] == bg
                                or th.is_walkable(int(curr_grid[ep[0] + dr, ep[1] + dc]))
                                or (ep[0] + dr, ep[1] + dc) == avatar_pos
                            )
                        )
                        is_interior_switch = walk_n == 4
                        cnt = int(np.count_nonzero(curr_grid == ef))
                        if cnt == 2 and is_interior_switch:
                            locs = [
                                (int(r), int(c))
                                for r, c in zip(*np.where(curr_grid == ef), strict=False)
                            ]
                            if hasattr(th, "induce_relational_cue_pair"):
                                th.induce_relational_cue_pair(ef, locs)
                            return (2, -round(pref_score, 3), dist)
                        if is_interior_switch:
                            return (3, -round(pref_score, 3), dist)
                        if pref_score >= 0.55:
                            return (4, -round(pref_score, 3), dist)
                        return (5 if cnt == 1 else 6, -round(pref_score, 3), dist)
                p_feat = int(curr_grid[p[0], p[1]])
                target_feat = (
                    int(meta["target_feature"]) if (meta and "target_feature" in meta) else p_feat
                )
                pref_score, barrier_score, hazard_score = (
                    _role_score_for_feature(target_feat) if target_feat != bg else (0.0, 0.0, 0.0)
                )
                if (
                    hazard_score >= 0.65
                    and pref_score < 0.45
                    and target_feat not in th.goal_features
                ):
                    return (8, hazard_score, dist)
                if (
                    barrier_score >= 0.72
                    and pref_score < 0.40
                    and target_feat not in th.goal_features
                ):
                    return (7, barrier_score, dist)
                p_cnt = int(np.count_nonzero(curr_grid == p_feat)) if p_feat != bg else 0
                if (
                    p_feat != bg
                    and not th.is_barrier(p_feat)
                    and (p_cnt == 1 or p_feat in th.goal_features)
                    and any(
                        0 <= p[0] + dr < H
                        and 0 <= p[1] + dc < W
                        and th.is_position_actuator_probed((p[0] + dr, p[1] + dc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                ):
                    return (1, -round(pref_score, 3), dist)
                if pref_score >= 0.55:
                    return (4, -round(pref_score, 3), dist)
                return (5 if p_cnt == 1 else 6, -round(pref_score, 3), dist)

            priorities = {g: _stance_priority(g) for g in goals}
            min_tier = min(pr[0] for pr in priorities.values())
            if min_tier <= 3:
                goals = [g for g in goals if priorities[g][0] == min_tier]
            goals.sort(key=lambda g: priorities[g])

        # (d) Structural inference: geometric patterns suggesting goals
        if not goals and engine.learned_goal_positions:
            goals = [
                g
                for g in engine.learned_goal_positions
                if 0 <= g[0] < H
                and 0 <= g[1] < W
                and g not in panel_coords
                and (
                    avatar_pos is None
                    or (
                        g != avatar_pos
                        and (abs(g[0] - avatar_pos[0]) > 1 or abs(g[1] - avatar_pos[1]) > 1)
                    )
                )
            ]
        if not goals:
            wm_priors_d: dict[int, dict[str, float]] = (
                getattr(getattr(engine, "working_memory", None), "concept_role_priors", None) or {}
            )
            has_cargo_d = bool(
                engine.symbolic_theory.cargo_features or engine.learned_cargo_features
            )
            structural_goals = engine.detect_structural_goals(curr_grid, bg=bg)
            primary_sgs = [
                sg
                for sg in structural_goals
                if sg.get("type")
                in (
                    "target_zone",
                    "exit_marker",
                    "reflection_target",
                    "relational_alignment",
                    "optical_mirror_target",
                    "constellation_target",
                )
            ]
            cand_sgs = primary_sgs if primary_sgs else structural_goals

            def _is_prior_blocked_feat(feat_val: int) -> bool:
                if feat_val in engine.symbolic_theory.goal_features:
                    return False
                p_s, b_s, h_s = cls._role_scores_from_priors(wm_priors_d, feat_val, has_cargo_d)
                return (h_s >= 0.65 and p_s < 0.45) or (b_s >= 0.72 and p_s < 0.40)

            for sg in cand_sgs:
                feat = sg.get("feature")
                if sg.get("is_midpoint") or sg.get("type") == "relational_alignment":
                    feat = None
                if feat is not None and (
                    (feat in av_feats and not sg.get("is_midpoint"))
                    or feat in engine.hazard_tracker.known_lethal_features
                    or (
                        engine.symbolic_theory.is_barrier(feat)
                        and feat not in engine.symbolic_theory.goal_features
                        and feat not in engine.symbolic_theory.candidate_goal_features
                    )
                    or _is_prior_blocked_feat(int(feat))
                ):
                    continue
                if (
                    feat is not None
                    and feat != bg
                    and feat not in av_feats
                    and (engine.symbolic_theory.is_walkable(feat) or not has_discrete_interact)
                ):
                    new_candidate_feats.add(feat)
                is_high_conf = sg.get("confidence", 0.0) >= 0.7 and (
                    avatar_pos is not None or bool(av_feats)
                )
                cells = sg.get("cells")
                if cells:
                    for cp in cells:
                        if (
                            0 <= cp[0] < H
                            and 0 <= cp[1] < W
                            and cp != avatar_pos
                            and cp not in panel_coords
                            and int(curr_grid[cp[0], cp[1]]) != bg
                        ):
                            cp_feat = int(curr_grid[cp[0], cp[1]])
                            if (
                                cp_feat in engine.hazard_tracker.known_lethal_features
                                or _is_prior_blocked_feat(cp_feat)
                            ):
                                continue
                            if cp not in goals:
                                goals.append(cp)
                            if is_high_conf and (
                                feat is not None and feat in engine.symbolic_theory.goal_features
                            ):
                                new_learned_positions.add(cp)
                else:
                    pos = sg.get("position")
                    if (
                        pos
                        and 0 <= pos[0] < H
                        and 0 <= pos[1] < W
                        and pos != avatar_pos
                        and pos not in panel_coords
                        and int(curr_grid[pos[0], pos[1]]) != bg
                    ):
                        pos_feat = int(curr_grid[pos[0], pos[1]])
                        if (
                            pos_feat in engine.hazard_tracker.known_lethal_features
                            or _is_prior_blocked_feat(pos_feat)
                        ):
                            continue
                        if pos not in goals:
                            goals.append(pos)
                        if is_high_conf and (
                            feat is not None and feat in engine.symbolic_theory.goal_features
                        ):
                            new_learned_positions.add(pos)

            if len(goals) > 1 and avatar_pos is not None:

                def _structural_priority(g: tuple[int, int]) -> tuple[int, float, int]:
                    gf = int(curr_grid[g[0], g[1]])
                    pref_g, bar_g, haz_g = (
                        cls._role_scores_from_priors(wm_priors_d, gf, has_cargo_d)
                        if gf != bg
                        else (0.0, 0.0, 0.0)
                    )
                    dist_g = abs(g[0] - avatar_pos[0]) + abs(g[1] - avatar_pos[1])
                    if gf in engine.symbolic_theory.goal_features:
                        return (0, -round(pref_g, 3), dist_g)
                    if haz_g >= 0.65 and pref_g < 0.45:
                        return (8, haz_g, dist_g)
                    if bar_g >= 0.72 and pref_g < 0.40:
                        return (7, bar_g, dist_g)
                    if pref_g >= 0.55:
                        return (1, -round(pref_g, 3), dist_g)
                    if pref_g >= 0.35:
                        return (2, -round(pref_g, 3), dist_g)
                    cnt_g = int(np.count_nonzero(curr_grid == gf)) if gf != bg else 0
                    return (3 if cnt_g == 1 else 4, -round(pref_g, 3), dist_g)

                goals.sort(key=_structural_priority)

        # (e) Frontier Exploration (P2-9): Nearest reachable boundary between visited and unvisited
        prior_hazard_feats = cls._high_confidence_prior_hazards(engine, bg, av_feats)
        if not goals:
            visited = set(getattr(engine.working_memory, "visited_cells", set()))
            if avatar_pos is not None:
                visited.add(avatar_pos)

            reachable_passable: set[tuple[int, int]] = set()
            if avatar_pos is not None:
                q_frontier: deque[tuple[int, int]] = deque([avatar_pos])
                reachable_passable.add(avatar_pos)
                while q_frontier:
                    curr_cell = q_frontier.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_cell[0] + dr, curr_cell[1] + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in reachable_passable:
                            c_val = int(curr_grid[nr, nc])
                            if (
                                (c_val == bg or engine.symbolic_theory.is_walkable(c_val))
                                and not engine.symbolic_theory.is_barrier(c_val)
                                and c_val not in engine.hazard_tracker.known_lethal_features
                                and c_val not in prior_hazard_feats
                            ):
                                reachable_passable.add((nr, nc))
                                q_frontier.append((nr, nc))

            frontier_candidates = [
                cp
                for cp in reachable_passable
                if cp not in visited
                and cp not in panel_coords
                and any(
                    (cp[0] + dr, cp[1] + dc) in visited
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                )
            ]
            if not frontier_candidates:
                frontier_candidates = [
                    cp for cp in reachable_passable if cp not in visited and cp not in panel_coords
                ]

            if frontier_candidates and avatar_pos is not None:
                frontier_candidates.sort(
                    key=lambda p: abs(p[0] - avatar_pos[0]) + abs(p[1] - avatar_pos[1])
                )
                nearest_frontier = frontier_candidates[0]
                goals.append(nearest_frontier)
                satisfied_when[nearest_frontier] = {"max_visits": 1}

        if avatar_pos is not None:
            goals = [
                g
                for g in goals
                if (
                    (g != avatar_pos or (g in satisfied_when and "entity_pos" in satisfied_when[g]))
                    and (g in confirmed_goals or g not in panel_coords)
                )
            ]

        all_cargo_feats = set(engine.symbolic_theory.cargo_features) | set(
            engine.learned_cargo_features
        )
        unexhausted_goals = [
            g
            for g in goals
            if (
                (
                    g not in engine.exhausted_candidate_goals
                    or (has_cargo_context and int(curr_grid[g[0], g[1]]) not in all_cargo_feats)
                )
                and (
                    not engine.symbolic_theory.is_position_explored(g)
                    or has_cargo_context
                    or g in confirmed_goals
                )
            )
            or (
                g in satisfied_when
                and "entity_pos" in satisfied_when[g]
                and not engine.symbolic_theory.is_position_explored(satisfied_when[g]["entity_pos"])
                and (not engine.symbolic_theory.is_position_explored(g) or g == avatar_pos)
            )
        ]
        if unexhausted_goals:
            goals = unexhausted_goals
        elif goals:
            clear_exhausted = True

        # Exclude all known static lethal coordinates from candidate attractors
        all_lethals = engine.current_level_lethal_positions
        goals = [g for g in goals if g not in all_lethals]

        # Reachability Gating: Discard goals physically disconnected from avatar
        if avatar_pos is not None and goals:
            barrier_feats = engine.symbolic_theory.barrier_features
            lethal_feats = engine.hazard_tracker.known_lethal_features
            learned_b = engine.learned_barriers

            # Vectorized 4-connected flood fill over passable cells —
            # identical closure to the old per-cell BFS expansion.
            blocked = np.zeros((H, W), dtype=bool)
            if learned_b:
                lb_idx = np.array([(r, c) for r, c in learned_b if 0 <= r < H and 0 <= c < W])
                if lb_idx.size:
                    blocked[lb_idx[:, 0], lb_idx[:, 1]] = True
            bad_feats = set(barrier_feats) | set(lethal_feats) | prior_hazard_feats
            if bad_feats:
                blocked |= np.isin(curr_grid, list(bad_feats))
            passable = ~blocked
            reach = np.zeros((H, W), dtype=bool)
            reach[avatar_pos[0], avatar_pos[1]] = True
            prev_count = -1
            while True:
                grown = reach.copy()
                grown[1:, :] |= reach[:-1, :]
                grown[:-1, :] |= reach[1:, :]
                grown[:, 1:] |= reach[:, :-1]
                grown[:, :-1] |= reach[:, 1:]
                grown &= passable
                grown[avatar_pos[0], avatar_pos[1]] = True
                count = int(grown.sum())
                if count == prev_count:
                    break
                reach = grown
                prev_count = count
            reachable_cells = {(int(r), int(c)) for r, c in np.argwhere(reach)}

            reachable_goals = [
                g
                for g in goals
                if g in reachable_cells
                or any(
                    (g[0] + dr, g[1] + dc) in reachable_cells
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
            ]
            if reachable_goals:
                goals = reachable_goals

        # DreamCoder LibraryLearner proposal prior guidance over unconfirmed exploratory goals
        if (
            len(goals) > 1
            and not confirmed_goals
            and getattr(engine, "library_learner", None)
            and getattr(engine.library_learner, "proposal_distribution", None) is not None
        ):
            ll_ctx = {
                "has_goal": bool(engine.symbolic_theory.goal_features),
                "has_cargo": bool(engine.symbolic_theory.cargo_features),
                "has_switches": bool(
                    engine.symbolic_theory.conditional_affordances
                    or getattr(getattr(engine, "remote_causal", None), "remote_actuators", None)
                ),
                "has_threats": bool(getattr(engine.hazard_tracker, "known_lethal_features", None)),
                "n_actions": len(available_actions),
            }
            ll_proposals = engine.library_learner.sample_proposals(ll_ctx, top_k=5)
            if len(ll_proposals) >= 2 and ll_proposals[0][1] > ll_proposals[1][1] + 0.01:
                top_intent = ll_proposals[0][0].upper()
                actuator_feats = getattr(engine.symbolic_theory, "actuator_features", set())
                cargo_feats = getattr(engine.symbolic_theory, "cargo_features", set())
                goal_feats = (
                    engine.symbolic_theory.goal_features
                    | engine.symbolic_theory.candidate_goal_features
                )

                def _ll_priority(g: tuple[int, int]) -> tuple[int, int]:
                    feat = int(curr_grid[g[0], g[1]])
                    sw_feat = satisfied_when.get(g, {}).get("feature")
                    dist = (
                        abs(g[0] - avatar_pos[0]) + abs(g[1] - avatar_pos[1])
                        if avatar_pos is not None
                        else 0
                    )
                    if "INTERACT" in top_intent or "ACTIVATE" in top_intent:
                        tier = 0 if (feat in actuator_feats or sw_feat in actuator_feats) else 1
                    elif "PICKUP" in top_intent:
                        tier = 0 if feat in cargo_feats else 1
                    elif "NAV" in top_intent:
                        tier = 0 if feat in goal_feats else 1
                    else:
                        tier = 0
                    return (tier, dist)

                goals.sort(key=_ll_priority)

        return EpistemicGoalProposal(
            goals=goals,
            confirmed_goals=confirmed_goals,
            new_learned_goal_positions=new_learned_positions,
            new_candidate_features=new_candidate_feats,
            subgoal_to_pop=subgoal_to_pop,
            clear_exhausted_goals=clear_exhausted,
            satisfied_when=satisfied_when,
            transient_blocked_cells=transient_blocked,
        )

    @classmethod
    def to_causal_predicate(cls, view: Any = None) -> CausalPredicate:
        """Export goal scheduler capability and runtime subgoal stack state as a declarative causal predicate."""
        val = "ACTIVE"
        rel = "ofc_ba10_prioritization"
        if view is not None:
            sgs = len(getattr(view, "subgoal_stack", []) or [])
            goals = len(getattr(view, "learned_goal_positions", set()) or set())
            rel = f"ofc_ba10_prioritization:goals:{goals}:subgoals:{sgs}"
        return CausalPredicate(
            variable="epistemic_goal_scheduler",
            operator="==",
            value=val,
            relation=rel,
            target_entity="cortex_planner",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering goal scheduler."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="goal_scheduler",
                node_data={
                    "faculty": "epistemic_goal_scheduler",
                    "status": "ACTIVE",
                },
            )
        ]


class ContinuousKinematicPlanner:
    """Metric continuous trajectory planner over ReferenceFrame coordinates (WP-C1)."""

    @staticmethod
    def plan_continuous_trajectory(
        engine: AutonomousEpistemicEngine,
        available_actions: Sequence[Any],
        curr_grid: np.ndarray | None = None,
        goal_radius: float = 0.55,
        max_expansions: int = 512,
    ) -> list[MentalSimulationStep] | None:
        """Plan a multi-step kinematic trajectory in continuous metric space using grounded velocity vectors."""
        wm = getattr(engine, "working_memory", None)
        if wm is None:
            return None
        c_pos = getattr(wm, "continuous_agent_pos", None)
        c_goal = getattr(wm, "continuous_goal_pos", None)
        rf = getattr(wm, "reference_frame", None)
        if c_pos is None or len(c_pos) < 2:
            return None

        # Gather grounded continuous velocity vectors
        vel_map: dict[Any, tuple[float, float]] = {}
        for act in available_actions:
            dyn = engine.action_dynamics.get(act) if hasattr(engine, "action_dynamics") else None
            if dyn is not None and hasattr(dyn, "get_continuous_velocity"):
                vx, vy = dyn.get_continuous_velocity()
                if (vx * vx + vy * vy) > 1e-6:
                    vel_map[act] = (float(vx), float(vy))

        if not vel_map or c_goal is None or len(c_goal) < 2:
            return None

        bounds = getattr(rf, "bounds", ((0.0, 10.0), (0.0, 10.0)))
        x_min, x_max = (
            (float(bounds[0][0]), float(bounds[0][1])) if len(bounds) >= 1 else (0.0, 10.0)
        )
        y_min, y_max = (
            (float(bounds[1][0]), float(bounds[1][1])) if len(bounds) >= 2 else (0.0, 10.0)
        )
        origin_desc = (
            getattr(rf, "origin_description", "bottom_left_metric") if rf else "bottom_left_metric"
        )

        gx, gy = float(c_goal[0]), float(c_goal[1])
        sx, sy = float(c_pos[0]), float(c_pos[1])

        def _to_lattice(x: float, y: float) -> tuple[int, int]:
            if curr_grid is None or curr_grid.ndim < 2:
                return (0, 0)
            H, W = curr_grid.shape
            c_idx = int(np.clip(round((x - x_min) / max(1e-6, x_max - x_min) * W - 0.5), 0, W - 1))
            if "bottom_left" in origin_desc:
                r_idx = int(
                    np.clip(round((y_max - y) / max(1e-6, y_max - y_min) * H - 0.5), 0, H - 1)
                )
            else:
                r_idx = int(
                    np.clip(round((y - y_min) / max(1e-6, y_max - y_min) * H - 0.5), 0, H - 1)
                )
            return (r_idx, c_idx)

        def _is_blocked(x: float, y: float) -> bool:
            if not (x_min + 0.4 <= x <= x_max - 0.4 and y_min + 0.4 <= y <= y_max - 0.4):
                return True
            if curr_grid is not None and curr_grid.ndim >= 2:
                lr, lc = _to_lattice(x, y)
                feat = int(curr_grid[lr, lc])
                if engine.symbolic_theory.is_barrier(feat) or (lr, lc) in engine.learned_barriers:
                    return True
            return False

        # Continuous A* with 0.25m spatial hashing
        start_key = (round(sx * 4), round(sy * 4))
        pq: list[tuple[float, int, float, float, list[MentalSimulationStep]]] = [
            (math.hypot(gx - sx, gy - sy), 0, sx, sy, [])
        ]
        visited: dict[tuple[int, int], int] = {start_key: 0}
        expansions = 0

        while pq and expansions < max_expansions:
            _, cost, cx, cy, path = heapq.heappop(pq)
            expansions += 1
            if math.hypot(gx - cx, gy - cy) <= goal_radius and path:
                return path

            for act, (vx, vy) in vel_map.items():
                nx, ny = cx + vx, cy + vy
                if _is_blocked(nx, ny):
                    continue
                nkey = (round(nx * 4), round(ny * 4))
                ncost = cost + 1
                if nkey in visited and visited[nkey] <= ncost:
                    continue
                visited[nkey] = ncost
                lat_p = _to_lattice(nx, ny)
                step_obj = MentalSimulationStep(
                    action=int(act) if isinstance(act, (int, np.integer)) else act,
                    predicted_avatar_pos=lat_p,
                    expected_mutation="continuous_kinematic_step",
                )
                h = math.hypot(gx - nx, gy - ny) / max(0.25, math.hypot(vx, vy))
                heapq.heappush(pq, (ncost + h, ncost, nx, ny, [*path, step_obj]))

        return None


class MentalSimulationPlanner:
    """Simulates candidate action sequences internally in imagination using learned world theory."""

    @classmethod
    def to_causal_predicate(cls, view: Any = None) -> CausalPredicate:
        """Export mental simulation planner capability and active plan horizon as a declarative causal predicate."""
        val = "ACTIVE"
        rel = "internal_counterfactual_rollout"
        if view is not None:
            plen = len(getattr(view, "mental_plan", []) or [])
            rel = f"internal_counterfactual_rollout:plan_len:{plen}"
        return CausalPredicate(
            variable="mental_simulation_planner",
            operator="==",
            value=val,
            relation=rel,
            target_entity="cortex_planner",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering mental simulation planner."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="simulation_planner",
                node_data={
                    "faculty": "mental_simulation_planner",
                    "status": "ACTIVE",
                },
            )
        ]

    @staticmethod
    def is_corner_deadlock(
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        H: int,
        W: int,
    ) -> bool:
        """Check whether a cargo block at pos is trapped in an irreversible corner deadlock."""
        return CounterfactualDeadlockDetector.is_corner_deadlock(
            pos, goals, static_barriers, (H, W)
        )

    @staticmethod
    def is_wall_deadlock(
        pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers_or_h: set[tuple[int, int]] | int | None = None,
        H_or_w: int | None = None,
        W: int | None = None,
    ) -> bool:
        """Check whether a cargo block is trapped along a boundary wall with no goals on it."""
        if W is None:
            # Called as is_wall_deadlock(pos, goals, H, W)
            static_barriers: set[tuple[int, int]] = set()
            grid_h = static_barriers_or_h if isinstance(static_barriers_or_h, int) else 10
            grid_w = H_or_w if isinstance(H_or_w, int) else 10
        else:
            # Called as is_wall_deadlock(pos, goals, static_barriers, H, W)
            static_barriers = (
                static_barriers_or_h if isinstance(static_barriers_or_h, set) else set()
            )
            grid_h = H_or_w if isinstance(H_or_w, int) else 10
            grid_w = int(W)
        return CounterfactualDeadlockDetector.is_wall_deadlock(
            pos, goals, static_barriers, (grid_h, grid_w)
        )

    @staticmethod
    def _compute_static_barriers(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        bg: int,
        av_feats: set[int],
        goals: Sequence[tuple[int, int]],
        blocked_cells: set[tuple[int, int]] | None = None,
        entities: Sequence[Any] | None = None,
    ) -> tuple[set[tuple[int, int]], bool]:
        """Compute static physical barriers and void background boundaries."""
        H, W = curr_grid.shape
        static_barriers: set[tuple[int, int]] = set()

        has_discrete_interact_sb = any(
            not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
            for a in (getattr(engine, "action_dynamics", None) or {})
        )
        if entities:
            for e in entities:
                if (
                    (
                        e.role == EntityRole.OBSTACLE
                        or (
                            has_discrete_interact_sb
                            and e.feature_id not in engine.symbolic_theory.goal_features
                            and not engine.symbolic_theory.is_cargo(e)
                        )
                    )
                    and e.feature_id != bg
                    and e.feature_id not in av_feats
                    and not engine.symbolic_theory.is_walkable(e.feature_id)
                ):
                    for cr, cc in e.properties.get("cells", [e.grid_pos]):
                        if (int(cr), int(cc)) not in goals:
                            static_barriers.add((int(cr), int(cc)))

        for r, c in engine.learned_barriers:
            if 0 <= r < H and 0 <= c < W:
                val = int(curr_grid[r, c])
                if (r, c) in engine.hazard_tracker.static_lethal_positions or (
                    val != bg and not engine.symbolic_theory.is_walkable(val)
                ):
                    static_barriers.add((r, c))

        agent_state = {"inventory": getattr(engine.working_memory, "held_feature_ids", set())}
        prior_hazard_feats = EpistemicGoalScheduler._high_confidence_prior_hazards(
            engine, bg, av_feats
        )
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if (
                    engine.symbolic_theory.is_barrier(val, agent_state=agent_state)
                    or (
                        val in engine.hazard_tracker.known_lethal_features
                        and val not in engine.mobile_threat_features
                    )
                    or (val in prior_hazard_feats and (r, c) not in engine.learned_goal_positions)
                ):
                    static_barriers.add((r, c))

        # Explicitly ground all static lethal positions from hazard tracker and level memory
        for r, c in engine.current_level_lethal_positions:
            if 0 <= r < H and 0 <= c < W:
                static_barriers.add((r, c))

        # Empirical Barrier & Void Gating: Only treat background as void if empirically confirmed
        bg_is_impassable_void = (
            engine.symbolic_theory.is_barrier(bg)
            or bg in engine.hazard_tracker.known_lethal_features
        )
        if bg_is_impassable_void:
            for r in range(H):
                for c in range(W):
                    if int(curr_grid[r, c]) == bg and (r, c) not in goals:
                        static_barriers.add((r, c))

        # Enforce HCIR Attractor Invariant: Non-lethal goals must never be static barriers
        all_lethals = engine.current_level_lethal_positions
        safe_goals = {
            g
            for g in goals
            if g not in all_lethals and int(curr_grid[g[0], g[1]]) not in prior_hazard_feats
        }
        static_barriers.difference_update(safe_goals)
        goal_features = engine.symbolic_theory.goal_features | (
            engine.symbolic_theory.candidate_goal_features - prior_hazard_feats
        )
        for gf in goal_features:
            if gf != bg and gf not in av_feats:
                g_pts = np.argwhere(curr_grid == gf)
                for gr_idx, gc_idx in g_pts:
                    pt = (int(gr_idx), int(gc_idx))
                    if pt in safe_goals or (
                        pt not in engine.learned_barriers and not has_discrete_interact_sb
                    ):
                        static_barriers.discard(pt)
        if engine.avatar_pos is not None:
            ar, ac = engine.avatar_pos
            av_footprint = set()
            for dr_f in range(-3, 4):
                for dc_f in range(-3, 4):
                    r_f, c_f = ar + dr_f, ac + dc_f
                    if 0 <= r_f < H and 0 <= c_f < W:
                        if int(curr_grid[r_f, c_f]) in av_feats:
                            av_footprint.add((r_f, c_f))
            static_barriers.difference_update(av_footprint)

        # Empirical Walkability Invariant: Verified walkable/safe features cannot be static barriers
        for feat_w in engine.symbolic_theory.walkable_features:
            if feat_w != bg:
                w_pts = np.argwhere(curr_grid == feat_w)
                for wr, wc in w_pts:
                    static_barriers.discard((int(wr), int(wc)))

        if blocked_cells:
            static_barriers.update(blocked_cells)

        return static_barriers, bg_is_impassable_void

    @staticmethod
    def _extract_mutation_triggers(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        bg: int,
    ) -> dict[tuple[int, int], set[tuple[int, int]]]:
        """Extract verified state mutation triggers (switches that toggle doors) via Pearl's causal calculus."""
        H, W = curr_grid.shape
        mutation_triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}

        # vmPFC Remote Causal Triggers (Pearl's do(a) causal attribution)
        if hasattr(engine.working_memory, "remote_causal"):
            for trig_pos, aff_list in engine.working_memory.remote_causal.trigger_registry.items():
                for aff in aff_list:
                    if (
                        aff.confidence >= 0.70
                        and 0 <= aff.remote_pos[0] < H
                        and 0 <= aff.remote_pos[1] < W
                    ):
                        mutation_triggers.setdefault(trig_pos, set()).add(aff.remote_pos)

        # Historical state mutations
        if hasattr(engine, "state_mutations"):
            for m in engine.state_mutations:
                opened_cells: set[tuple[int, int]] = set()
                for r in range(H):
                    for c in range(W):
                        if int(curr_grid[r, c]) == m.prior_value and (
                            m.posterior_value == bg or m.posterior_value == 0
                        ):
                            opened_cells.add((r, c))
                if not opened_cells:
                    continue

                if (
                    m.trigger_pos is not None
                    and len(m.trigger_pos) >= 2
                    and 0 <= m.trigger_pos[0] < H
                    and 0 <= m.trigger_pos[1] < W
                ):
                    tr_p = (int(m.trigger_pos[0]), int(m.trigger_pos[1]))
                    mutation_triggers.setdefault(tr_p, set()).update(opened_cells)

                if (
                    m.trigger_feature is not None
                    and m.trigger_feature != bg
                    and m.trigger_feature != engine.bg_feature
                    and not engine.symbolic_theory.is_walkable(m.trigger_feature)
                ):
                    for r in range(H):
                        for c in range(W):
                            if int(curr_grid[r, c]) == m.trigger_feature:
                                mutation_triggers.setdefault((r, c), set()).update(opened_cells)

        return mutation_triggers

    @staticmethod
    def compute_swr_backward_penalties(
        failed_trajectories: list[list[Any]],
        horizon: int = 5,
        initial_penalty: float = 40.0,
        decay: float = 0.7,
    ) -> dict[tuple[tuple[int, int], Any], float]:
        """Hippocampal Sharp-Wave Ripple (SWR) Backward Credit Assignment.

        Replays fatal episodes in reverse, assigning decaying penalties backward
        from the failure branch point along the tail of the trajectory.
        """
        penalties: dict[tuple[tuple[int, int], Any], float] = {}
        for ep in failed_trajectories[-5:]:
            valid_transitions = [
                (getattr(t, "avatar_pos", None), getattr(t, "action_id", None))
                for t in ep
                if getattr(t, "avatar_pos", None) is not None
            ]
            if not valid_transitions:
                continue
            tail = valid_transitions[-horizon:]
            curr_p = initial_penalty
            for pos, act in reversed(tail):
                if pos is not None and act is not None:
                    key = (pos, act)
                    penalties[key] = max(penalties.get(key, 0.0), curr_p)
                curr_p *= decay
        return penalties

    @staticmethod
    def _select_stance_actuation(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        satisfied_when: dict[tuple[int, int], dict[str, Any]],
        H: int,
        W: int,
        check_start_explored: bool = False,
        mark_probed_pos: bool = True,
    ) -> list[MentalSimulationStep] | None:
        """Select a stationary actuation if standing adjacent to a pending probe/actuator target."""
        start_pos = engine.avatar_pos
        if start_pos is None:
            return None

        cur_sg = (
            engine.subgoal_stack.current_subgoal if engine.subgoal_stack.has_pending_goals else None
        )
        probe_target_pos: tuple[int, int] | None = None
        if start_pos in satisfied_when and "entity_pos" in satisfied_when[start_pos]:
            probe_target_pos = satisfied_when[start_pos]["entity_pos"]
        elif cur_sg is not None and cur_sg.target_entity_pos is not None:
            probe_target_pos = cur_sg.target_entity_pos

        if probe_target_pos is None:
            return None

        if not engine.subgoal_stack.has_pending_goals and (
            start_pos in engine.exhausted_candidate_goals
            or (
                (not check_start_explored or start_pos not in satisfied_when)
                and (
                    (
                        check_start_explored
                        and engine.symbolic_theory.is_position_explored(start_pos)
                    )
                    or engine.symbolic_theory.is_position_explored(probe_target_pos)
                )
            )
        ):
            return None

        te_pos = probe_target_pos
        te_feat = (
            int(curr_grid[te_pos[0], te_pos[1]])
            if (0 <= te_pos[0] < H and 0 <= te_pos[1] < W)
            else None
        )
        if not (
            te_feat is not None
            and not engine.symbolic_theory.is_walkable(te_feat)
            and abs(start_pos[0] - te_pos[0]) + abs(start_pos[1] - te_pos[1]) <= 1
        ):
            return None

        is_unlock_sg = (
            cur_sg is not None and cur_sg.subgoal_type == SubgoalType.UNLOCK_REMOTE_MECHANISM
        )
        act_id: int | None = None
        confirmed_stationary = False
        for m in engine.state_mutations:
            if m.trigger_pos == te_pos or (
                m.trigger_feature is not None and te_feat == m.trigger_feature
            ):
                if m.trigger_type == "ACTION":
                    aid = m.metadata.get("action_id")
                    if aid is not None and aid in available_actions:
                        act_id = int(aid)
                        confirmed_stationary = True
                        break

        if act_id is None:
            discrete_acts = [
                a
                for a in available_actions
                if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
            ]
            tried = (
                cur_sg.metadata.get("tried_actions", set())
                if is_unlock_sg and cur_sg is not None
                else set()
            )
            candidates = (
                [a for a in discrete_acts if a not in tried] if is_unlock_sg else discrete_acts
            )
            if not candidates:
                if is_unlock_sg and cur_sg is not None:
                    engine.symbolic_theory.mark_position_explored(
                        te_pos, reason="actuator_exhausted"
                    )
                    engine.exhausted_candidate_goals.add(start_pos)
                    cur_sg.metadata["triggered"] = True
                return None

            dynamics = getattr(engine, "action_dynamics", {})
            candidates.sort(
                key=lambda a: (
                    0 if engine.inhibited_actions.get(a, 0) == 0 else 1,
                    0
                    if (
                        a in dynamics
                        and float(getattr(dynamics[a], "delta_r", 0)) == 0.0
                        and float(getattr(dynamics[a], "delta_c", 0)) == 0.0
                    )
                    else 1,
                )
            )
            act_id = int(candidates[0])
            confirmed_stationary = (
                act_id in dynamics
                and float(getattr(dynamics[act_id], "delta_r", 0)) == 0.0
                and float(getattr(dynamics[act_id], "delta_c", 0)) == 0.0
            )

        if act_id is None:
            return None

        if is_unlock_sg and cur_sg is not None:
            cur_sg.metadata.setdefault("tried_actions", set()).add(act_id)
            if not confirmed_stationary:
                # Defer "triggered" confirmation until AssimilationStage verifies
                # whether the candidate action actually triggered or merely displaced the avatar.
                logger.info(
                    "AutonomousEpistemicEngine: Stance adjacent to actuator %s reached. Probing candidate action %d.",
                    te_pos,
                    act_id,
                )
                return [
                    MentalSimulationStep(
                        action=act_id,
                        predicted_avatar_pos=start_pos,
                        expected_mutation="ACTION_TRIGGER",
                    )
                ]

        if mark_probed_pos:
            engine.symbolic_theory.mark_position_explored(start_pos, reason="actuator_probed")
            engine.exhausted_candidate_goals.add(start_pos)
        if cur_sg is not None:
            cur_sg.metadata["triggered"] = True
        logger.info(
            "AutonomousEpistemicEngine: Stance adjacent to actuator %s reached. Triggering action %d.",
            te_pos,
            act_id,
        )
        return [
            MentalSimulationStep(
                action=act_id,
                predicted_avatar_pos=start_pos,
                expected_mutation="ACTION_TRIGGER",
            )
        ]

    @staticmethod
    def simulate_in_mind(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        target_role: EntityRole = EntityRole.GOAL,
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> list[MentalSimulationStep] | None:
        """Simulate candidate action sequences internally in memory without taking physical steps."""
        if not isinstance(curr_grid, np.ndarray):
            curr_grid = np.asarray(curr_grid, dtype=int)
        if curr_grid.ndim == 3 and len(curr_grid) > 0:
            curr_grid = curr_grid[-1]

        if engine.avatar_pos is None and (
            engine.avatar_features
            or engine.avatar_feature is not None
            or engine.prior_avatar_feature is not None
        ):
            known_av = set(engine.avatar_features)
            if engine.avatar_feature is not None:
                known_av.add(engine.avatar_feature)
            elif engine.prior_avatar_feature is not None:
                known_av.add(engine.prior_avatar_feature)
            engine._update_avatar_position_from_grid(curr_grid, known_av)

        if engine.avatar_pos is None and engine.working_memory.body_schema.has_manipulandum():
            manip = get_sub_attr(engine.working_memory.body_schema, "active_manipulandum", None)
            if manip is not None and getattr(manip, "pos", None) is not None:
                engine.state.avatar_pos = manip.pos

        if not engine.is_motor_grounded() or engine.avatar_pos is None:
            return None

        _budget = engine.compute_budget if engine is not None else None

        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        entities = engine.extract_entities(curr_grid, bg)

        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        # 1. Epistemic Goal Scheduling (BA10 / OFC priority hierarchy)
        proposal = EpistemicGoalScheduler.schedule_goals(
            engine=engine,
            curr_grid=curr_grid,
            entities=entities,
            bg=bg,
            av_feats=av_feats,
            target_role=target_role,
            available_actions=available_actions,
        )

        # Commit proposed state updates atomically
        for gp in proposal.new_learned_goal_positions:
            engine.record_learned_goal(gp)
        for feat in proposal.new_candidate_features:
            engine.symbolic_theory.induce_candidate_goal(feat, evidence="planner:proposal")
        if proposal.subgoal_to_pop:
            engine.subgoal_stack.pop_subgoal()
        if proposal.clear_exhausted_goals:
            engine.exhausted_candidate_goals.clear()
        for av_f in av_feats:
            engine.symbolic_theory.retract_goal(av_f, reason="avatar_feature")
            engine.symbolic_theory.retract_candidate_goal(av_f, reason="avatar_feature")
        if proposal.satisfied_when:
            engine.working_memory.active_epistemic_conditions.update(proposal.satisfied_when)

        goals = proposal.goals
        satisfied_when = proposal.satisfied_when

        # Immediate Actuator Stance Execution (if already adjacent to interactive target entity)
        start_pos = engine.avatar_pos
        imm_stance_plan = MentalSimulationPlanner._select_stance_actuation(
            engine=engine,
            curr_grid=curr_grid,
            available_actions=available_actions,
            satisfied_when=satisfied_when,
            H=H,
            W=W,
            check_start_explored=False,
            mark_probed_pos=True,
        )
        if imm_stance_plan is not None:
            return imm_stance_plan

        # Faculty D: Bilateral Convergent Coordinate Frames
        if engine.bilateral_state is not None and len(engine.action_dynamics) >= 2:
            passable_mask = np.ones((H, W), dtype=bool)
            for r in range(H):
                for c in range(W):
                    feat = int(curr_grid[r, c])
                    if (
                        engine.symbolic_theory.is_barrier(feat)
                        or feat in engine.hazard_tracker.known_lethal_features
                        or (r, c) in engine.learned_barriers
                    ):
                        passable_mask[r, c] = False

            action_deltas = engine.action_deltas
            bilateral_plan = engine.bilateral_integrator.plan_convergence_sequence(
                pos1=engine.bilateral_state.pos1,
                pos2=engine.bilateral_state.pos2,
                axis=engine.bilateral_state.symmetry_axis,
                passable_mask=passable_mask,
                action_deltas=action_deltas,
                target_positions=goals if goals else None,
            )
            if bilateral_plan:
                logger.info(
                    "MentalSimulationPlanner: Formulated Bilateral Callosal Convergence plan (%d steps)",
                    len(bilateral_plan),
                )
                curr_b_pos = engine.avatar_pos or engine.bilateral_state.pos1
                steps = []
                for act in bilateral_plan:
                    dr_b, dc_b = action_deltas.get(act, (0, 0))
                    curr_b_pos = (curr_b_pos[0] + dr_b, curr_b_pos[1] + dc_b)
                    steps.append(
                        MentalSimulationStep(
                            action=act,
                            predicted_avatar_pos=curr_b_pos,
                            expected_mutation="bilateral_convergence",
                        )
                    )
                return steps

        # 2. Identify candidate pushable blocks / cargo
        pushable_blocks: list[tuple[int, int]] = [
            e.grid_pos
            for e in entities
            if e.feature_id not in av_feats
            and e.feature_id != bg
            and (e.grid_pos not in proposal.confirmed_goals)
            and (
                engine.symbolic_theory.is_cargo(e)
                or e.feature_id in engine.learned_cargo_features
                or (
                    e.role == EntityRole.MANIPULABLE
                    and 1 <= e.area <= 64
                    and e.feature_id not in engine.mobile_threat_features
                )
            )
            and e.feature_id not in engine.hazard_tracker.known_lethal_features
            and not engine.symbolic_theory.is_barrier(e.feature_id)
        ]

        if engine.symbolic_theory.cargo_features or engine.learned_cargo_features:
            delivery_goals = [g for g in goals if g not in pushable_blocks]
            if delivery_goals:
                goals = delivery_goals
            else:
                cargo_set = set(engine.symbolic_theory.cargo_features) | set(
                    engine.learned_cargo_features
                )
                receptor_set = (
                    set(engine.symbolic_theory.goal_features)
                    | set(engine.symbolic_theory.candidate_goal_features)
                    | set(engine.symbolic_theory.receptacle_features)
                )
                cand_receptors = [
                    e.grid_pos
                    for e in entities
                    if e.feature_id in receptor_set
                    and e.feature_id not in cargo_set
                    and e.grid_pos not in pushable_blocks
                    and e.grid_pos not in engine.learned_barriers
                    and e.grid_pos not in engine.current_level_lethal_positions
                ]
                if not cand_receptors:
                    cand_receptors = [
                        e.grid_pos
                        for e in entities
                        if e.feature_id != bg
                        and e.feature_id not in av_feats
                        and e.feature_id not in cargo_set
                        and not engine.symbolic_theory.is_barrier(e.feature_id)
                        and engine.symbolic_theory.is_walkable(e.feature_id)
                        and int(np.count_nonzero(curr_grid == e.feature_id)) <= 4
                        and e.grid_pos not in pushable_blocks
                        and e.grid_pos not in engine.learned_barriers
                        and e.grid_pos not in engine.current_level_lethal_positions
                    ]
                if cand_receptors:
                    wm_priors: dict[int, dict[str, float]] = (
                        getattr(
                            getattr(engine, "working_memory", None), "concept_role_priors", None
                        )
                        or {}
                    )
                    if wm_priors:
                        cand_receptors.sort(
                            key=lambda pos: (
                                -max(
                                    float(
                                        wm_priors.get(int(curr_grid[pos[0], pos[1]]), {}).get(
                                            "receptacle", 0.0
                                        )
                                    ),
                                    float(
                                        wm_priors.get(int(curr_grid[pos[0], pos[1]]), {}).get(
                                            "goal", 0.0
                                        )
                                    ),
                                )
                            )
                        )
                    goals = cand_receptors

        if not goals:
            return None

        # 3. Static barriers & corridor continuity
        effective_blocked = (set(blocked_cells) if blocked_cells else set()) | (
            proposal.transient_blocked_cells - set(goals)
        )
        static_barriers, bg_is_impassable_void = MentalSimulationPlanner._compute_static_barriers(
            engine=engine,
            curr_grid=curr_grid,
            bg=bg,
            av_feats=av_feats,
            goals=goals,
            blocked_cells=effective_blocked or None,
            entities=entities,
        )

        # 4. State mutation triggers
        mutation_triggers = MentalSimulationPlanner._extract_mutation_triggers(
            engine=engine,
            curr_grid=curr_grid,
            bg=bg,
        )

        # 4. Available directional movements in mental model
        movable_actions: list[tuple[Any, int, int]] = []
        action_deltas = engine.action_deltas
        for act in available_actions:
            if engine.is_spatial_effector(act):
                continue
            if act in engine.immobile_entity_actions:
                continue
            if act in action_deltas:
                dr, dc = action_deltas[act]
                if abs(dc) >= 2 * abs(dr):
                    dr = 0
                elif abs(dr) >= 2 * abs(dc):
                    dc = 0
                if dr != 0 or dc != 0:
                    movable_actions.append((act, dr, dc))

        if not movable_actions:
            return None

        # Unknown solid entity cells: risk-aware path planning
        unknown_entity_cells: set[tuple[int, int]] = set()
        for e in entities:
            if (
                e.feature_id != bg
                and e.feature_id not in av_feats
                and not engine.symbolic_theory.is_walkable(e.feature_id)
                and e.feature_id not in engine.symbolic_theory.goal_features
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and not engine.symbolic_theory.is_cargo(e)
                and e.feature_id not in engine.learned_cargo_features
                and e.role != EntityRole.MANIPULABLE
                and e.grid_pos not in goals
            ):
                for cell in e.properties.get("cells", [e.grid_pos]):
                    unknown_entity_cells.add(cell)

        # Determine movement displacement scale
        step_size = 1
        for act, dr, dc in movable_actions:
            dist = max(abs(dr), abs(dc))
            if dist > step_size:
                step_size = dist

        # Oriented threats
        cargos_known = set(engine.symbolic_theory.cargo_features | engine.learned_cargo_features)
        oriented_threats = PerceptionEngine.detect_oriented_threats(
            curr_grid,
            entities,
            bg=bg,
            step_size=step_size,
            avatar_pos=engine.avatar_pos,
            avatar_features=av_feats,
            goals=set(goals),
            cargo_features=cargos_known,
        )

        half_step = max(1, step_size // 2)

        _barrier_feats = engine.symbolic_theory.barrier_features
        _grid_walls: set[tuple[int, int]] = set(static_barriers)
        for rr in range(H):
            for cc in range(W):
                if (rr, cc) in goals:
                    continue
                cell_val = int(curr_grid[rr, cc])
                if cell_val in _barrier_feats or (bg_is_impassable_void and cell_val == bg):
                    _grid_walls.add((rr, cc))

        # Note: Do not artificially inject topological cul-de-sacs as solid walls into _grid_walls,
        # as doing so locks the agent out of evasive nooks, hiding alcoves, and side rooms
        # needed for pacing, threat evasion, and item retrieval. Space-Time A* with grid_dist_field
        # naturally prioritizes shortest geodesic paths without artificial topological mutilation.

        def _corridor_blocked(p: tuple[int, int], f: tuple[int, int]) -> bool:
            probe = (p[0] + f[0] * half_step, p[1] + f[1] * half_step)
            dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
            for q in (probe, dest):
                if not (0 <= q[0] < H and 0 <= q[1] < W) or q in _grid_walls:
                    return True
            return False

        static_threats = [
            t for t in oriented_threats if t.feature_id not in engine.mobile_threat_features
        ]
        threat_cells: set[tuple[int, int]] = set()
        threat_map: dict[tuple[int, int], OrientedThreat] = {}
        for t in static_threats:
            threat_cells.add(t.pos)
            threat_map[t.pos] = t
            tr, tc = t.pos
            for dr_t in (-1, 0, 1):
                for dc_t in (-1, 0, 1):
                    rr_t, cc_t = tr + dr_t, tc + dc_t
                    if 0 <= rr_t < H and 0 <= cc_t < W:
                        val_t = int(curr_grid[rr_t, cc_t])
                        if val_t == t.feature_id or (
                            val_t != bg
                            and not engine.symbolic_theory.is_barrier(val_t)
                            and val_t not in av_feats
                            and (rr_t, cc_t) not in goals
                        ):
                            threat_cells.add((rr_t, cc_t))
                            threat_map[(rr_t, cc_t)] = t

        # Oriented dynamic hazards are governed by approach angles rather than static immobility;
        # remove them from static barriers and grid walls so tactical flanking can neutralize them.
        static_barriers.difference_update(threat_cells)
        _grid_walls.difference_update(threat_cells)
        unknown_entity_cells.difference_update(threat_cells)

        init_threats = frozenset(threat_cells)
        trajectory_features = {
            proj.feature_id for proj in engine.collision_cones.projected_trajectories
        }
        patrol_items: set[tuple[tuple[int, int], tuple[int, int]]] = {
            (t.pos, t.facing)
            for t in oriented_threats
            if t.feature_id in engine.mobile_threat_features
            and t.facing != (0, 0)
            and t.feature_id not in trajectory_features
        }
        for proj in engine.collision_cones.projected_trajectories:
            if proj.feature_id in cargos_known:
                continue
            if (
                proj.feature_id in engine.mobile_threat_features
                or proj.feature_id in engine.hazard_tracker.known_lethal_features
            ):
                vr, vc = proj.velocity
                if vr != 0 or vc != 0:
                    f = (int(np.sign(vr)), int(np.sign(vc)))
                    patrol_items.add((proj.initial_pos, f))
        init_patrols: frozenset[tuple[tuple[int, int], tuple[int, int]]] = frozenset(patrol_items)

        def advance_patrols(
            patrols: frozenset[tuple[tuple[int, int], tuple[int, int]]],
            avatar_new: tuple[int, int],
            avatar_old: tuple[int, int] | None = None,
        ) -> frozenset[tuple[tuple[int, int], tuple[int, int]]] | None:
            nxt: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for p, f in patrols:
                dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W) or dest in _grid_walls:
                    new_f = (-f[0], -f[1])
                    nxt.add((p, new_f))
                    if p == avatar_new:
                        return None
                    continue
                if dest == avatar_new:
                    return None
                if avatar_old is not None and p == avatar_new and dest == avatar_old:
                    return None
                new_f = f
                if _corridor_blocked(dest, f):
                    new_f = (-f[0], -f[1])
                nxt.add((dest, new_f))
            return frozenset(nxt)

        def _patrol_next_cells(
            patrols: frozenset[tuple[tuple[int, int], tuple[int, int]]],
        ) -> set[tuple[int, int]]:
            cells: set[tuple[int, int]] = set()
            for p, f in patrols:
                dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                if 0 <= dest[0] < H and 0 <= dest[1] < W and dest not in _grid_walls:
                    cells.add(dest)
                else:
                    cells.add(p)
            return cells

        start_pos = engine.avatar_pos
        init_blocks = frozenset(pushable_blocks)
        init_open: frozenset[tuple[int, int]] = frozenset()

        oriented_threat_positions = {t.pos for t in oriented_threats}
        init_chasers: frozenset[tuple[int, int]] = frozenset(
            e.grid_pos
            for e in entities
            if not engine.symbolic_theory.is_cargo(e.feature_id)
            and (
                e.feature_id in engine.mobile_threat_features
                or (
                    e.role == EntityRole.AGENT
                    and e.grid_pos != start_pos
                    and e.feature_id not in av_feats
                    and e.feature_id != bg
                )
            )
            and e.grid_pos not in init_threats
            and not any(p[0] == e.grid_pos for p in init_patrols)
            and not any(
                abs(e.grid_pos[0] - tp[0]) + abs(e.grid_pos[1] - tp[1]) <= step_size
                for tp in oriented_threat_positions
            )
        )

        def advance_chasers(
            chasers: frozenset[tuple[int, int]],
            avatar_target: tuple[int, int],
        ) -> frozenset[tuple[int, int]] | None:
            nxt: set[tuple[int, int]] = set()
            for cp in chasers:
                if cp == avatar_target:
                    return None
                dr = 1 if avatar_target[0] > cp[0] else (-1 if avatar_target[0] < cp[0] else 0)
                dc = 1 if avatar_target[1] > cp[1] else (-1 if avatar_target[1] < cp[1] else 0)

                dest = cp
                if abs(avatar_target[0] - cp[0]) >= abs(avatar_target[1] - cp[1]) and dr != 0:
                    cand = (cp[0] + dr, cp[1])
                    if cand not in _grid_walls:
                        dest = cand
                    elif dc != 0 and (cp[0], cp[1] + dc) not in _grid_walls:
                        dest = (cp[0], cp[1] + dc)
                elif dc != 0:
                    cand = (cp[0], cp[1] + dc)
                    if cand not in _grid_walls:
                        dest = cand
                    elif dr != 0 and (cp[0] + dr, cp[1]) not in _grid_walls:
                        dest = (cp[0] + dr, cp[1])

                if dest == avatar_target:
                    return None
                nxt.add(dest)
            return frozenset(nxt)

        has_confirmed_cargo = bool(
            engine.symbolic_theory.cargo_features or engine.learned_cargo_features
        )
        is_block_delivery = bool(
            has_confirmed_cargo
            and 1 <= len(pushable_blocks) <= 16
            and any(
                (
                    engine.symbolic_theory.is_cargo(e)
                    or e.feature_id in engine.learned_cargo_features
                )
                and e.feature_id not in av_feats
                and e.grid_pos in pushable_blocks
                for e in entities
            )
        )

        has_periodic = (
            bool(engine.hazard_tracker.periodic_cells)
            and engine.hazard_tracker.environmental_period >= 2
        )
        has_temporal = (
            bool(init_patrols)
            or bool(init_chasers)
            or has_periodic
            or bool(engine.collision_cones.projected_trajectories)
        )

        # ─────────────────────────────────────────────────────────────────────
        # Fast Topological Geodesic & Kinematics Router (Substrate Layer)
        # For static navigation tasks (no blocks to push and no temporal threats),
        # route deterministic shortest action paths over traversable manifolds.
        # When temporal threats (patrols, chasers) exist, defer to Space-Time A*.
        # ─────────────────────────────────────────────────────────────────────
        if (
            goals
            and not is_block_delivery
            and not has_temporal
            and not mutation_triggers
            and not pushable_blocks
            and start_pos is not None
        ):
            geodesic_movable_actions = list(movable_actions)
            action_plan = (
                GeodesicManifoldRouter.route_geodesic(
                    curr_grid=curr_grid,
                    start_pos=start_pos,
                    goals=goals,
                    static_barriers=static_barriers,
                    grid_walls=_grid_walls,
                    unknown_entity_cells=unknown_entity_cells,
                    oriented_threats=oriented_threats,
                    movable_actions=geodesic_movable_actions,
                    bg=bg,
                    barrier_features=set(_barrier_feats),
                    avatar_features=av_feats,
                    max_candidate_goals=5,
                )
                if geodesic_movable_actions
                else None
            )
            if action_plan:
                plan_valid = True
                curr_p = start_pos
                for act, next_p in action_plan:
                    next_is_open = (
                        0 <= next_p[0] < H
                        and 0 <= next_p[1] < W
                        and (
                            int(curr_grid[next_p[0], next_p[1]]) == bg
                            or engine.symbolic_theory.is_walkable(
                                int(curr_grid[next_p[0], next_p[1]])
                            )
                            or next_p in goals
                        )
                    )
                    is_failed = (curr_p, act) in engine.failed_transitions and not next_is_open
                    if is_failed or engine.working_memory.habenular_ior.is_action_inhibited(
                        curr_p, act
                    ):
                        plan_valid = False
                        break
                    curr_p = next_p
                if (
                    plan_valid
                    and action_plan
                    and (
                        engine.inhibited_actions.get(action_plan[0][0], 0) > 0
                        or engine.working_memory.habenular_ior.is_action_inhibited(
                            start_pos, action_plan[0][0]
                        )
                    )
                ):
                    plan_valid = False

                if plan_valid:
                    topo_steps = [
                        MentalSimulationStep(
                            action=act,
                            predicted_avatar_pos=next_pos,
                            expected_mutation="topological_geodesic_step",
                            is_deterministic_geodesic=True,
                        )
                        for act, next_pos in action_plan
                    ]
                    logger.info(
                        "MentalSimulationPlanner: Action-Space Geodesic BFS plan dispatched to %s (%d steps)",
                        action_plan[-1][1],
                        len(topo_steps),
                    )
                    return topo_steps

        acc_monitor = engine.working_memory.acc_conflict
        acc_weight = acc_monitor.compute_heuristic_weight(1.0) if acc_monitor else 1.0
        if acc_monitor and acc_monitor.active_conflict and start_pos is not None:
            traversable_cells: set[tuple[int, int]] = set()
            for r_t in range(H):
                for c_t in range(W):
                    if (r_t, c_t) not in static_barriers and (r_t, c_t) not in _grid_walls:
                        traversable_cells.add((r_t, c_t))
            first_goal = list(goals)[0] if goals else start_pos
            acc_monitor.synthesize_perimeter_detour(
                current_pos=start_pos,
                goal_pos=first_goal,
                grid_shape=(H, W),
                traversable=traversable_cells,
            )
        # Gather all physical cells of current target goal entities
        all_goal_cells: set[tuple[int, int]] = set(goals)
        known_goal_features: set[int] = (
            engine.symbolic_theory.goal_features | engine.symbolic_theory.candidate_goal_features
        )
        for e in entities:
            if e.grid_pos in goals:
                for cr, cc in e.properties.get("cells", [e.grid_pos]):
                    if (int(cr), int(cc)) not in engine.learned_barriers:
                        all_goal_cells.add((int(cr), int(cc)))
        if not all_goal_cells:
            for gf in known_goal_features:
                if gf != bg and gf not in av_feats:
                    for gr_idx, gc_idx in np.argwhere(curr_grid == gf):
                        all_goal_cells.add((int(gr_idx), int(gc_idx)))

        # Precompute Vectorized 2D Geodesic Distance Field Matrix for O(1) heuristic lookup
        grid_dist_field = np.full((H, W), 999999.0, dtype=np.float32)
        if goals:
            b_step = step_size if step_size > 1 else 1
            bfs_q: deque[tuple[int, int]] = deque()
            for gr, gc in all_goal_cells:
                if 0 <= gr < H and 0 <= gc < W:
                    grid_dist_field[gr, gc] = 0.0
                    bfs_q.append((gr, gc))

            while bfs_q:
                cr, cc = bfs_q.popleft()
                cd = grid_dist_field[cr, cc]
                for dr_b, dc_b in [(-b_step, 0), (b_step, 0), (0, -b_step), (0, b_step)]:
                    nr_b, nc_b = cr + dr_b, cc + dc_b
                    if 0 <= nr_b < H and 0 <= nc_b < W:
                        if (nr_b, nc_b) in _grid_walls and (nr_b, nc_b) not in all_goal_cells:
                            continue
                        if grid_dist_field[nr_b, nc_b] > cd + 1.0:
                            grid_dist_field[nr_b, nc_b] = cd + 1.0
                            bfs_q.append((nr_b, nc_b))

            unreached = grid_dist_field >= 999999.0
            if np.any(unreached):
                r_indices = np.arange(H, dtype=np.int32)[:, None]
                c_indices = np.arange(W, dtype=np.int32)[None, :]
                for gr, gc in all_goal_cells:
                    d_mat = (np.abs(r_indices - gr) + np.abs(c_indices - gc)) / float(b_step)
                    grid_dist_field[unreached] = np.minimum(
                        grid_dist_field[unreached], d_mat[unreached]
                    )

        def heuristic(pos: tuple[int, int], blocks: frozenset[tuple[int, int]]) -> float:
            if is_block_delivery and blocks:
                b_list = list(blocks)
                if len(goals) == 1:
                    goal_dist = float(min(grid_dist_field[b[0], b[1]] for b in b_list))
                else:
                    goal_dist = 0.0
                    for g in goals:
                        goal_dist += min(abs(g[0] - b[0]) + abs(g[1] - b[1]) for b in b_list)
                avatar_to_b = min(abs(pos[0] - b[0]) + abs(pos[1] - b[1]) for b in b_list)
                base_h = float(goal_dist * 2.0 + avatar_to_b)
            elif goals:
                base_h = float(grid_dist_field[pos[0], pos[1]])
            else:
                base_h = 0.0

            if acc_monitor and acc_monitor.active_conflict:
                choke_pen = acc_monitor.get_chokepoint_penalty(pos)
                detour = acc_monitor.detour_target
                detour_dist = (abs(detour[0] - pos[0]) + abs(detour[1] - pos[1])) if detour else 0.0
                return float(base_h * acc_weight + choke_pen + detour_dist * 1.5)
            elif acc_monitor:
                choke_pen = acc_monitor.get_chokepoint_penalty(pos)
                return float(base_h + choke_pen)
            return base_h

        counter = 0
        has_periodic = (
            bool(engine.hazard_tracker.periodic_cells)
            and engine.hazard_tracker.environmental_period >= 2
        )
        has_temporal = (
            bool(init_patrols)
            or bool(init_chasers)
            or has_periodic
            or bool(engine.collision_cones.projected_trajectories)
        )
        meta_depth = get_sub_attr(engine.prediction_budget.counterfactual_budget, "max_depth", None)
        meta_mode = (
            getattr(
                engine.working_memory.last_metacognitive_assessment,
                "recommended_exploration_mode",
                None,
            )
            if get_sub_attr(engine.working_memory, "last_metacognitive_assessment", None)
            else None
        )
        max_expansions_budget = engine.prediction_budget.compute_search_budget(
            has_temporal=has_temporal,
            has_threats=bool(init_threats),
            consecutive_failures=engine.consecutive_simulation_failures,
            is_under_strict_budget=engine.is_under_strict_step_budget(),
            remaining_steps=engine.get_remaining_step_budget(),
            metacognitive_depth=meta_depth,
            exploration_mode=meta_mode,
        )

        non_disp_wait_action: Any | None = None
        if has_temporal:
            for a in available_actions:
                if a in engine.action_dynamics:
                    if not engine.action_dynamics[a].is_displacement_action():
                        non_disp_wait_action = a
                        break
                elif not engine.is_displacement_action(a) and not engine.is_spatial_effector(a):
                    non_disp_wait_action = a
                    break

        # SWR Backward Horizon Penalties from Hippocampal Episodic Memory
        swr_penalties: dict[tuple[tuple[int, int], Any], float] = {}
        if hasattr(engine, "episodic_cortex") and engine.episodic_cortex.failed_trajectories:
            swr_penalties = MentalSimulationPlanner.compute_swr_backward_penalties(
                engine.episodic_cortex.failed_trajectories
            )

        def _search_spacetime_astar(
            _relaxed_safety: bool,
        ) -> tuple[list[MentalSimulationStep] | None, bool]:
            nonlocal counter
            parent_map: dict[int, tuple[int, MentalSimulationStep]] = {}

            def _reconstruct_path(node_id: int) -> list[MentalSimulationStep]:
                curr = node_id
                rev: list[MentalSimulationStep] = []
                while curr in parent_map:
                    curr, step = parent_map[curr]
                    rev.append(step)
                rev.reverse()
                return rev

            open_set: list[
                tuple[
                    float,
                    float,
                    int,
                    tuple[int, int],
                    frozenset[tuple[int, int]],
                    frozenset[tuple[int, int]],
                    frozenset[tuple[int, int]],
                    frozenset[tuple[tuple[int, int], tuple[int, int]]],
                    frozenset[tuple[int, int]],
                    int,
                    int,
                ]
            ] = []
            h0 = heuristic(start_pos, init_blocks)
            heapq.heappush(
                open_set,
                (
                    h0,
                    0,
                    counter,
                    start_pos,
                    init_blocks,
                    init_open,
                    init_threats,
                    init_patrols,
                    init_chasers,
                    0,
                    0,
                ),
            )

            visited_states: set[
                tuple[
                    tuple[int, int],
                    frozenset[tuple[int, int]],
                    frozenset[tuple[int, int]],
                    frozenset[tuple[int, int]],
                    frozenset[tuple[tuple[int, int], tuple[int, int]]],
                    frozenset[tuple[int, int]],
                    int,
                ]
            ] = set()
            # WP0: budget-aware search — scale expansions from remaining work
            # units, charge 1 unit ("cheap" class) per rollout expansion, and
            # return best-so-far if the budget cuts the loop.
            best_g: dict[tuple[Any, ...], float] = {}
            max_expansions = max_expansions_budget
            if _budget is not None:
                max_expansions = min(max_expansions, _budget.remaining_units)
            best_node_id: int | None = None
            best_f: float = float("inf")
            budget_cutoff = False
            had_relaxable_block = False

            while open_set and max_expansions > 0:
                (
                    f_score,
                    g_cost,
                    node_id,
                    cur_pos,
                    cur_blocks,
                    cur_open,
                    cur_threats,
                    cur_patrols,
                    cur_chasers,
                    path_len,
                    consecutive_waits,
                ) = heapq.heappop(open_set)

                time_mod = (
                    path_len % engine.hazard_tracker.environmental_period if has_periodic else 0
                )
                state_key = (
                    cur_pos,
                    cur_blocks,
                    cur_open,
                    cur_threats,
                    cur_patrols,
                    cur_chasers,
                    time_mod,
                )
                if state_key in visited_states:
                    continue
                if _budget is not None and not _budget.allows("cheap"):
                    budget_cutoff = True
                    break
                if _budget is not None:
                    _budget.charge(1)
                max_expansions -= 1
                if f_score < best_f:
                    best_f = f_score
                    best_node_id = node_id
                visited_states.add(state_key)

                if is_block_delivery:
                    if all(g in cur_blocks for g in goals) and path_len > 0:
                        path = _reconstruct_path(node_id)
                        logger.info(
                            "AutonomousEpistemicEngine: Compound Mental Simulation SUCCEEDED! Synthesized %d-step block delivery plan.",
                            len(path),
                        )
                        engine.state.current_simulated_goal = list(goals)[0] if goals else None
                        return path, had_relaxable_block
                elif (cur_pos in goals or cur_pos in all_goal_cells) and path_len > 0:
                    path = _reconstruct_path(node_id)
                    # Check if cur_pos is an adjacent interaction stance for an actuator subgoal or probe entity
                    probe_target_pos: tuple[int, int] | None = None
                    if cur_pos in satisfied_when and "entity_pos" in satisfied_when[cur_pos]:
                        probe_target_pos = satisfied_when[cur_pos]["entity_pos"]
                    elif engine.subgoal_stack.has_pending_goals:
                        cur_sg = engine.subgoal_stack.current_subgoal
                        if cur_sg is not None and cur_sg.target_entity_pos is not None:
                            probe_target_pos = cur_sg.target_entity_pos

                    if probe_target_pos is not None:
                        te_pos = probe_target_pos
                        te_feat = (
                            int(curr_grid[te_pos[0], te_pos[1]])
                            if (0 <= te_pos[0] < H and 0 <= te_pos[1] < W)
                            else None
                        )
                        if (
                            te_feat is not None
                            and not engine.symbolic_theory.is_walkable(te_feat)
                            and abs(cur_pos[0] - te_pos[0]) + abs(cur_pos[1] - te_pos[1]) <= 1
                        ):
                            act_id_term: int | None = None
                            for m in engine.state_mutations:
                                if m.trigger_pos == te_pos or (
                                    m.trigger_feature is not None and te_feat == m.trigger_feature
                                ):
                                    if m.trigger_type == "ACTION":
                                        aid = m.metadata.get("action_id")
                                        if aid is not None and aid in available_actions:
                                            act_id_term = int(aid)
                                            break
                            if act_id_term is None:
                                discrete_acts = [
                                    a
                                    for a in available_actions
                                    if not engine.is_displacement_action(a)
                                    and not engine.is_spatial_effector(a)
                                ]
                                if discrete_acts:
                                    uninhib = [
                                        a
                                        for a in discrete_acts
                                        if engine.inhibited_actions.get(a, 0) == 0
                                    ]
                                    act_id_term = (
                                        int(uninhib[0]) if uninhib else int(discrete_acts[0])
                                    )
                            if act_id_term is not None:
                                path.append(
                                    MentalSimulationStep(
                                        action=act_id_term,
                                        predicted_avatar_pos=cur_pos,
                                        expected_mutation="ACTION_TRIGGER",
                                    )
                                )
                    logger.info(
                        "AutonomousEpistemicEngine: Mental Simulation SUCCEEDED! Synthesized %d-step path to goal %s.",
                        len(path),
                        cur_pos,
                    )
                    engine.state.current_simulated_goal = cur_pos
                    return path, had_relaxable_block

                candidate_actions: list[tuple[Any, int, int]] = list(movable_actions)
                max_consecutive_waits = max(4, min(16, engine.hazard_tracker.environmental_period))

                # Temporal Evasion: Determine safe waiting action (explicit wait or wall-bump in alcove)
                bump_wait_action: Any | None = non_disp_wait_action
                if bump_wait_action is None and (cur_patrols or has_periodic or has_temporal):
                    for a_disp in available_actions:
                        dyn = engine.action_dynamics.get(a_disp)
                        if dyn is not None and dyn.is_displacement_action():
                            dr_b, dc_b = dyn.delta_r, dyn.delta_c
                            br, bc = cur_pos[0] + dr_b, cur_pos[1] + dc_b
                            if (
                                not (0 <= br < H and 0 <= bc < W)
                                or (br, bc) in _grid_walls
                                or (br, bc) in engine.hazard_tracker.static_lethal_positions
                            ):
                                bump_wait_action = a_disp
                                break

                if (
                    (cur_patrols or has_periodic or has_temporal)
                    and bump_wait_action is not None
                    and consecutive_waits < max_consecutive_waits
                ):
                    candidate_actions.append(("__WAIT__", 0, 0))

                for act, dr, dc in candidate_actions:
                    is_wait = act == "__WAIT__"
                    if is_wait:
                        real_act = bump_wait_action
                        if real_act is None:
                            continue
                    else:
                        real_act = act
                    if (
                        not is_wait
                        and not has_temporal
                        and path_len == 0
                        and engine.inhibited_actions.get(act, 0) > 0
                    ):
                        continue

                    if not is_wait and (
                        (cur_pos, act) in engine.failed_transitions
                        or engine.working_memory.habenular_ior.is_action_inhibited(cur_pos, act)
                    ):
                        dest_r, dest_c = cur_pos[0] + dr, cur_pos[1] + dc
                        is_temporal_dynamic_cell = (
                            (
                                has_periodic
                                and (dest_r, dest_c) in engine.hazard_tracker.periodic_cells
                            )
                            or bool(cur_patrols)
                            or bool(cur_chasers)
                        )
                        if not is_temporal_dynamic_cell:
                            continue
                    nr, nc = cur_pos[0] + dr, cur_pos[1] + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue

                    # Core Knowledge Physics: Evaluate Spelke Continuity along candidate path step
                    if hasattr(engine.symbolic_theory, "evaluates_spelke_continuity"):
                        active_barriers = static_barriers - cur_open
                        if not engine.symbolic_theory.evaluates_spelke_continuity(
                            cur_pos, (nr, nc), active_barriers, (H, W)
                        ):
                            continue

                    step_r = int(np.sign(dr))
                    step_c = int(np.sign(dc))
                    dist = max(abs(dr), abs(dc))
                    ray_blocked = False
                    for k in range(1, dist + 1):
                        kr = cur_pos[0] + step_r * k
                        kc = cur_pos[1] + step_c * k
                        is_barrier_cell = (kr, kc) in static_barriers and (kr, kc) not in cur_open
                        if is_barrier_cell:
                            cell_feat = (
                                int(curr_grid[kr, kc]) if 0 <= kr < H and 0 <= kc < W else -1
                            )
                            if engine.working_memory.body_schema.is_barrier_permeable(cell_feat):
                                pass
                            elif cell_feat in known_goal_features or (kr, kc) in all_goal_cells:
                                pass
                            else:
                                ray_blocked = True
                                break
                        if bg_is_impassable_void and 0 <= kr < H and 0 <= kc < W:
                            if (
                                int(curr_grid[kr, kc]) == bg
                                and (kr, kc) not in goals
                                and (kr, kc) not in all_goal_cells
                            ):
                                ray_blocked = True
                                break
                    if ray_blocked:
                        continue

                    sim_step = path_len + 1
                    if engine.is_under_strict_step_budget():
                        rem_budget = engine.get_remaining_step_budget()
                        if sim_step > rem_budget + 8:
                            continue

                    walkable_feats = set(engine.symbolic_theory.walkable_features)
                    if engine.hazard_tracker.is_hazard_at(
                        nr,
                        nc,
                        sim_step,
                        bg,
                        avatar_features=av_feats,
                        walkable_features=walkable_feats,
                    ):
                        continue
                    collision_cone_penalty = 0.0
                    if engine.collision_cones.is_collision_hazard(nr, nc, sim_step):
                        if not _relaxed_safety:
                            had_relaxable_block = True
                            continue
                        else:
                            collision_cone_penalty = 30.0
                    is_flanked_step = (
                        (nr, nc) in cur_threats
                        and threat_map.get((nr, nc)) is not None
                        and cur_pos != threat_map[(nr, nc)].gaze_pos
                    )
                    is_neutralized_step = (nr, nc) in (init_threats - cur_threats)
                    if (
                        sim_step <= 1
                        and not is_flanked_step
                        and not is_neutralized_step
                        and int(curr_grid[nr, nc]) in engine.hazard_tracker.known_lethal_features
                        and int(curr_grid[nr, nc]) not in engine.mobile_threat_features
                    ):
                        continue

                    living_gaze: set[tuple[int, int]] = set()
                    for t_pos in cur_threats:
                        t = threat_map.get(t_pos)
                        if t is not None:
                            living_gaze.add(t.gaze_pos)

                    if (nr, nc) in living_gaze:
                        continue

                    threat_penalty = 0.0
                    new_threats = cur_threats
                    if (nr, nc) in cur_threats:
                        flanked_t = threat_map.get((nr, nc))
                        if flanked_t is not None and cur_pos != flanked_t.gaze_pos:
                            # Directional hazard neutralization: approaching oriented hazard outside its line of sight clears it
                            threat_cells_to_remove = {
                                p
                                for p, ot in threat_map.items()
                                if ot is flanked_t or p == (nr, nc)
                            }
                            new_threats = cur_threats - threat_cells_to_remove
                            threat_penalty = 5.0
                        else:
                            # Direct approach into active hazard line of sight or unmapped hazard is lethal
                            continue

                    new_patrols = cur_patrols
                    patrol_proximity_cost = 0.0
                    if cur_patrols:
                        predicted = advance_patrols(cur_patrols, (nr, nc), avatar_old=cur_pos)
                        if predicted is None:
                            continue
                        new_patrols = predicted

                        upcoming = _patrol_next_cells(cur_patrols)
                        if (nr, nc) in upcoming:
                            if _relaxed_safety:
                                patrol_proximity_cost += 15.0
                            else:
                                had_relaxable_block = True
                                continue
                        for pp, _pf in cur_patrols:
                            dist_to_patrol = abs(nr - pp[0]) + abs(nc - pp[1])
                            if 0 < dist_to_patrol <= step_size:
                                moving_towards = False
                                if _pf[0] != 0 and (nr - pp[0]) * _pf[0] > 0:
                                    moving_towards = True
                                elif _pf[1] != 0 and (nc - pp[1]) * _pf[1] > 0:
                                    moving_towards = True
                                if moving_towards:
                                    continue
                                patrol_proximity_cost += 10.0

                    new_chasers = cur_chasers
                    chaser_proximity_cost = 0.0
                    if cur_chasers:
                        predicted_chasers = advance_chasers(cur_chasers, (nr, nc))
                        if predicted_chasers is None:
                            continue
                        new_chasers = predicted_chasers
                        for cp in new_chasers:
                            dist_to_chaser = abs(nr - cp[0]) + abs(nc - cp[1])
                            if dist_to_chaser <= 1:
                                chaser_proximity_cost += 20.0
                            elif dist_to_chaser <= 2:
                                chaser_proximity_cost += 5.0

                    epistemic_cost = 0.0
                    for k in range(1, dist + 1):
                        kr = cur_pos[0] + step_r * k
                        kc = cur_pos[1] + step_c * k
                        if 0 <= kr < H and 0 <= kc < W:
                            cell_feat = int(curr_grid[kr, kc])
                            if cell_feat in engine.hazard_tracker.known_lethal_features:
                                if cell_feat in engine.mobile_threat_features:
                                    pass
                                elif (kr, kc) in (init_threats - cur_threats):
                                    pass
                                else:
                                    is_flanked_threat = (
                                        (kr, kc) in cur_threats
                                        and (kr, kc) == (nr, nc)
                                        and threat_map.get((kr, kc)) is not None
                                        and cur_pos != threat_map[(kr, kc)].gaze_pos
                                    )
                                    if not is_flanked_threat:
                                        epistemic_cost = float("inf")
                                        break
                            if engine.is_feature_unverified(cell_feat, bg):
                                epistemic_cost += engine.UNVERIFIED_FEATURE_COST
                            if (kr, kc) in unknown_entity_cells:
                                epistemic_cost += 10.0
                    if epistemic_cost == float("inf"):
                        continue

                    new_blocks = cur_blocks
                    new_pos = (nr, nc)

                    if (nr, nc) in cur_blocks:
                        # Core Knowledge Physics: Evaluate Spelke Contact Causality (must be adjacent)
                        if hasattr(engine.symbolic_theory, "evaluates_spelke_contact"):
                            if not engine.symbolic_theory.evaluates_spelke_contact(
                                cur_pos, (nr, nc), adjacency="4-way"
                            ):
                                continue

                        pushed_r = nr + step_r * step_size
                        pushed_c = nc + step_c * step_size
                        if not (0 <= pushed_r < H and 0 <= pushed_c < W):
                            continue
                        if (pushed_r, pushed_c) in _grid_walls:
                            continue
                        if (pushed_r, pushed_c) in cur_blocks:
                            continue

                        # Core Knowledge Physics: Evaluate Spelke Continuity for pushed cargo trajectory
                        if hasattr(engine.symbolic_theory, "evaluates_spelke_continuity"):
                            if not engine.symbolic_theory.evaluates_spelke_continuity(
                                (nr, nc), (pushed_r, pushed_c), _grid_walls, (H, W)
                            ):
                                continue
                        push_ray_blocked = False
                        for pk in range(1, step_size + 1):
                            pkr = nr + step_r * pk
                            pkc = nc + step_c * pk
                            if (pkr, pkc) in _grid_walls:
                                push_ray_blocked = True
                                break
                            if (
                                bg_is_impassable_void
                                and int(curr_grid[pkr, pkc]) == bg
                                and (pkr, pkc) not in goals
                            ):
                                push_ray_blocked = True
                                break
                        if push_ray_blocked:
                            continue

                        if engine.hazard_tracker.is_hazard_at(
                            pushed_r,
                            pushed_c,
                            sim_step,
                            bg,
                            avatar_features=av_feats,
                            walkable_features=walkable_feats,
                        ):
                            continue

                        if engine.physics_engine.has_gravity:
                            if not engine.physics_engine.is_supported(
                                pushed_r,
                                pushed_c,
                                curr_grid,
                                set(engine.symbolic_theory.barrier_features),
                                static_barriers,
                            ):
                                land_b, _, is_b_lethal = engine.physics_engine.project_fall(
                                    pushed_r,
                                    pushed_c,
                                    curr_grid,
                                    set(engine.symbolic_theory.barrier_features),
                                    static_barriers,
                                )
                                if is_b_lethal:
                                    continue
                                pushed_r, pushed_c = land_b

                        if (is_block_delivery or bool(cur_blocks)) and (
                            pushed_r,
                            pushed_c,
                        ) not in goals:
                            candidate_block_set = (set(cur_blocks) - {(nr, nc)}) | {
                                (pushed_r, pushed_c)
                            }
                            dl_eval = CounterfactualDeadlockDetector.evaluate_deadlock(
                                (pushed_r, pushed_c),
                                set(goals),
                                static_barriers,
                                candidate_block_set,
                                (H, W),
                            )
                            if dl_eval.is_deadlock:
                                continue

                        block_set = set(cur_blocks)
                        block_set.remove((nr, nc))
                        block_set.add((pushed_r, pushed_c))
                        new_blocks = frozenset(block_set)

                    new_open = cur_open
                    if new_pos in mutation_triggers:
                        new_open = cur_open | frozenset(mutation_triggers[new_pos])

                    wait_cost = 2.0 if is_wait else 0.0
                    swr_penalty = swr_penalties.get((cur_pos, real_act), 0.0)

                    habenular_penalty = 0.0
                    if hasattr(engine, "working_memory") and hasattr(
                        engine.working_memory, "habenular_ior"
                    ):
                        habenular_penalty = (
                            engine.working_memory.habenular_ior.get_repulsion_penalty(
                                cur_pos, real_act
                            )
                        )
                    elif hasattr(engine, "habenular_ior"):
                        habenular_penalty = engine.habenular_ior.get_repulsion_penalty(
                            cur_pos, real_act
                        )

                    step_cost = (
                        1
                        + epistemic_cost
                        + patrol_proximity_cost
                        + chaser_proximity_cost
                        + wait_cost
                        + threat_penalty
                        + collision_cone_penalty
                        + swr_penalty
                        + habenular_penalty
                    )
                    new_g = g_cost + step_cost
                    new_time_mod = (
                        (path_len + 1) % engine.hazard_tracker.environmental_period
                        if has_periodic
                        else 0
                    )
                    new_key = (
                        new_pos,
                        new_blocks,
                        new_open,
                        new_threats,
                        new_patrols,
                        new_chasers,
                        new_time_mod,
                    )
                    if new_key in visited_states or new_g >= best_g.get(new_key, float("inf")):
                        continue
                    best_g[new_key] = new_g
                    new_h = heuristic(new_pos, new_blocks)
                    new_step = MentalSimulationStep(action=real_act, predicted_avatar_pos=new_pos)
                    counter += 1
                    parent_map[counter] = (node_id, new_step)
                    new_waits = consecutive_waits + 1 if (is_wait or new_pos == cur_pos) else 0
                    heapq.heappush(
                        open_set,
                        (
                            new_g + new_h,
                            new_g,
                            counter,
                            new_pos,
                            new_blocks,
                            new_open,
                            new_threats,
                            new_patrols,
                            new_chasers,
                            path_len + 1,
                            new_waits,
                        ),
                    )

            if budget_cutoff and best_node_id is not None:
                return _reconstruct_path(best_node_id), had_relaxable_block
            return None, had_relaxable_block

        # Stance Check: If already standing adjacent to an unfulfilled actuator or probe entity, execute interaction immediately
        post_stance_plan = MentalSimulationPlanner._select_stance_actuation(
            engine=engine,
            curr_grid=curr_grid,
            available_actions=available_actions,
            satisfied_when=satisfied_when,
            H=H,
            W=W,
            check_start_explored=True,
            mark_probed_pos=False,
        )
        if post_stance_plan is not None:
            return post_stance_plan

        # Run primary safety-first A* pass
        plan, had_relaxable_block = _search_spacetime_astar(_relaxed_safety=False)
        if plan is not None:
            return plan

        # Relaxed Safety Retry: If primary safe search failed and gaze/threat constraints were encountered
        if had_relaxable_block and goals:
            logger.info(
                "MentalSimulationPlanner: Primary safe A* blocked by gaze/threat constraints. "
                "Retrying with relaxed safety constraints."
            )
            relaxed_plan, _ = _search_spacetime_astar(_relaxed_safety=True)
            if relaxed_plan is not None:
                return relaxed_plan

        # Hierarchical Subgoal Decomposition Fallback
        if (not is_block_delivery or bool(mutation_triggers)) and (mutation_triggers or goals):
            subgoal_plan = MentalSimulationPlanner._plan_hierarchical_subgoals(
                engine,
                curr_grid,
                available_actions,
                start_pos,
                set(goals),
                static_barriers,
                mutation_triggers,
                movable_actions,
            )
            if subgoal_plan:
                # Prefrontal Safety Gate: Validate first step does not step into a collision hazard
                first_pos = subgoal_plan[0].predicted_avatar_pos
                if first_pos is not None and engine.collision_cones.is_collision_hazard(
                    first_pos[0], first_pos[1], 1
                ):
                    logger.info(
                        "MentalSimulationPlanner: Hierarchical static plan rejected - first step collides with dynamic hazard at %s",
                        first_pos,
                    )
                    subgoal_plan = None
                else:
                    return subgoal_plan

        # Teleological Backward Cut-Set Decomposition Fallback (Human BA10 Reasoning)
        if (not is_block_delivery or bool(mutation_triggers)) and goals and start_pos is not None:
            teleo_plan = MentalSimulationPlanner._plan_teleological_cutset_subgoals(
                engine,
                curr_grid,
                available_actions,
                start_pos,
                set(goals),
                static_barriers,
                movable_actions,
                oriented_threats=oriented_threats,
            )
            if teleo_plan:
                first_pos = teleo_plan[0].predicted_avatar_pos
                if first_pos is not None and engine.collision_cones.is_collision_hazard(
                    first_pos[0], first_pos[1], 1
                ):
                    logger.info(
                        "MentalSimulationPlanner: Teleological static plan rejected - first step collides with dynamic hazard at %s",
                        first_pos,
                    )
                    teleo_plan = None
                else:
                    return teleo_plan

        return None

    @staticmethod
    def _plan_hierarchical_subgoals(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        start_pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        mutation_triggers: dict[tuple[int, int], set[tuple[int, int]]],
        movable_actions: list[tuple[int, int, int]],
    ) -> list[MentalSimulationStep] | None:
        """Recursive multi-stage subgoal decomposition for locked barrier doors and remote switches."""
        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        walkable_and_safe = engine.symbolic_theory.walkable_features
        bg_is_impassable_void = len(walkable_and_safe) > 0 and bg not in walkable_and_safe
        cur_pos = start_pos
        open_barriers: set[tuple[int, int]] = set()
        accumulated_plan: list[MentalSimulationStep] = []
        max_stages = 8

        def find_shortest_path(
            s_pos: tuple[int, int],
            target_positions: set[tuple[int, int]],
            active_barriers: set[tuple[int, int]],
        ) -> list[MentalSimulationStep] | None:
            q: deque[tuple[tuple[int, int], list[MentalSimulationStep]]] = deque([(s_pos, [])])
            visited = {s_pos}
            while q:
                p, path = q.popleft()
                if p in target_positions:
                    return path
                for act, dr, dc in movable_actions:
                    if (p, act) in engine.failed_transitions:
                        continue
                    nr, nc = p[0] + dr, p[1] + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in visited:
                        continue

                    step_r = int(np.sign(dr))
                    step_c = int(np.sign(dc))
                    dist = max(abs(dr), abs(dc))
                    ray_blocked = False
                    for k in range(1, dist + 1):
                        kr = p[0] + step_r * k
                        kc = p[1] + step_c * k
                        if (kr, kc) in active_barriers or (
                            kr,
                            kc,
                        ) in engine.current_level_lethal_positions:
                            ray_blocked = True
                            break
                        if bg_is_impassable_void and 0 <= kr < H and 0 <= kc < W:
                            if (
                                int(curr_grid[kr, kc]) == bg
                                and (kr, kc) not in goals
                                and (kr, kc) not in target_positions
                            ):
                                ray_blocked = True
                                break
                        if 0 <= kr < H and 0 <= kc < W:
                            if (
                                int(curr_grid[kr, kc])
                                in engine.hazard_tracker.known_lethal_features
                            ):
                                ray_blocked = True
                                break
                    if ray_blocked:
                        continue

                    visited.add((nr, nc))
                    step = MentalSimulationStep(action=act, predicted_avatar_pos=(nr, nc))
                    q.append(((nr, nc), path + [step]))
            return None

        buffered_subgoals: list[FrontopolarSubgoal] = []
        visited_triggers: set[tuple[int, int]] = set()
        for _ in range(max_stages):
            effective_barriers = static_barriers - open_barriers
            goal_path = find_shortest_path(cur_pos, goals, effective_barriers)
            if goal_path is not None:
                accumulated_plan.extend(goal_path)
                logger.info(
                    "AutonomousEpistemicEngine: Hierarchical Subgoal Decomposition SUCCEEDED with %d total steps!",
                    len(accumulated_plan),
                )
                for sg in reversed(buffered_subgoals):
                    engine.subgoal_stack.push_subgoal(sg)
                return accumulated_plan

            candidate_triggers: list[
                tuple[tuple[int, int], list[MentalSimulationStep], int, set[tuple[int, int]]]
            ] = []

            # 1. Discover reachable spatial manifold from cur_pos avoiding effective_barriers
            reachable_cells = {cur_pos}
            bfs_q = deque([cur_pos])
            while bfs_q:
                cp = bfs_q.popleft()
                for act, dr, dc in movable_actions:
                    np_pos = (cp[0] + dr, cp[1] + dc)
                    if 0 <= np_pos[0] < H and 0 <= np_pos[1] < W and np_pos not in reachable_cells:
                        if np_pos not in effective_barriers:
                            reachable_cells.add(np_pos)
                            bfs_q.append(np_pos)

            # 2. Identify boundary barrier cells bordering the reachable manifold
            boundary_barriers: set[tuple[int, int]] = set()
            for r, c in reachable_cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nbr = (r + dr, c + dc)
                    if nbr in effective_barriers:
                        boundary_barriers.add(nbr)

            # 3. Known vmPFC Distal Causal Attributions & Body Schema Tool Resonance
            bg = engine.estimate_background(curr_grid)
            entities = engine.extract_entities(curr_grid, bg)
            av_feats = engine.avatar_features or (
                {engine.avatar_feature} if engine.avatar_feature is not None else set()
            )

            for b_pos in boundary_barriers:
                b_feat = int(curr_grid[b_pos[0], b_pos[1]])
                trig_pos, req_tool = engine.causal_cortex.query_barrier_clearance(b_pos, b_feat)
                if trig_pos is not None:
                    if trig_pos in visited_triggers:
                        continue
                    tr_path = find_shortest_path(cur_pos, {trig_pos}, effective_barriers)
                    if tr_path is not None:
                        candidate_triggers.append((trig_pos, tr_path, 100, {b_pos}))
                elif req_tool is not None:
                    if engine.working_memory.is_holding(req_tool):
                        open_barriers.add(b_pos)
                    else:
                        tool_found = False
                        for e in entities:
                            if e.feature_id == req_tool and e.grid_pos in reachable_cells:
                                t_path = find_shortest_path(
                                    cur_pos, {e.grid_pos}, effective_barriers
                                )
                                if t_path is not None:
                                    candidate_triggers.append((e.grid_pos, t_path, 80, {b_pos}))
                                    tool_found = True

                        # Multi-Hop Backward-Chaining: If required tool is blocked behind another barrier,
                        # deduce the intervening cut-set barrier and resolve its clearance trigger!
                        if not tool_found:
                            unreached_tools = [
                                e.grid_pos
                                for e in entities
                                if e.feature_id == req_tool and e.grid_pos not in reachable_cells
                            ]
                            for ut_pos in unreached_tools:
                                unconstrained_path = find_shortest_path(cur_pos, {ut_pos}, set())
                                if unconstrained_path:
                                    intervening_barrier = None
                                    for step in unconstrained_path:
                                        if step.predicted_avatar_pos in effective_barriers:
                                            intervening_barrier = step.predicted_avatar_pos
                                            break
                                    if intervening_barrier is not None:
                                        ib_feat = int(
                                            curr_grid[
                                                intervening_barrier[0], intervening_barrier[1]
                                            ]
                                        )
                                        ib_trig, ib_tool = (
                                            engine.causal_cortex.query_barrier_clearance(
                                                intervening_barrier, ib_feat
                                            )
                                        )
                                        if ib_trig and ib_trig in reachable_cells:
                                            tr_path = find_shortest_path(
                                                cur_pos, {ib_trig}, effective_barriers
                                            )
                                            if tr_path is not None:
                                                candidate_triggers.append(
                                                    (
                                                        ib_trig,
                                                        tr_path,
                                                        95,
                                                        {intervening_barrier, b_pos},
                                                    )
                                                )
                                        elif ib_tool is not None:
                                            for e2 in entities:
                                                if (
                                                    e2.feature_id == ib_tool
                                                    and e2.grid_pos in reachable_cells
                                                ):
                                                    t_path2 = find_shortest_path(
                                                        cur_pos, {e2.grid_pos}, effective_barriers
                                                    )
                                                    if t_path2 is not None:
                                                        candidate_triggers.append(
                                                            (
                                                                e2.grid_pos,
                                                                t_path2,
                                                                90,
                                                                {intervening_barrier, b_pos},
                                                            )
                                                        )

            # 4. Known state mutation triggers (switches and pressure plates)
            for tr_pos, opened_set in mutation_triggers.items():
                if tr_pos in visited_triggers:
                    continue
                unopened = (opened_set & effective_barriers) - open_barriers
                if not unopened:
                    continue

                tr_feat = (
                    int(curr_grid[tr_pos[0], tr_pos[1]])
                    if (0 <= tr_pos[0] < H and 0 <= tr_pos[1] < W)
                    else None
                )
                is_standable = (
                    tr_feat is not None
                    and engine.symbolic_theory.is_walkable(tr_feat)
                    and tr_pos not in effective_barriers
                )
                if is_standable:
                    tr_path = find_shortest_path(cur_pos, {tr_pos}, effective_barriers)
                else:
                    adj_cells = {
                        (tr_pos[0] + dr, tr_pos[1] + dc)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        if 0 <= tr_pos[0] + dr < H
                        and 0 <= tr_pos[1] + dc < W
                        and (tr_pos[0] + dr, tr_pos[1] + dc) not in effective_barriers
                    }
                    if cur_pos in adj_cells:
                        tr_path = []
                    elif adj_cells:
                        tr_path = find_shortest_path(cur_pos, adj_cells, effective_barriers)
                    else:
                        tr_path = None

                if tr_path is not None:
                    candidate_triggers.append((tr_pos, tr_path, len(unopened) * 10, unopened))

            # 5. Epistemic Affordance Probing: If no confirmed triggers, test reachable objects
            if not candidate_triggers:
                if engine.working_memory.held_items:
                    adjacent_barrier_cells: set[tuple[int, int]] = set()
                    for br, bc in boundary_barriers:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ar, ac = br + dr, bc + dc
                            if (ar, ac) in reachable_cells:
                                adjacent_barrier_cells.add((ar, ac))
                    if adjacent_barrier_cells:
                        approach_path = find_shortest_path(
                            cur_pos, adjacent_barrier_cells, effective_barriers
                        )
                        if approach_path:
                            accumulated_plan.extend(approach_path)
                            for sg in buffered_subgoals:
                                engine.subgoal_stack.push_subgoal(sg)
                            return accumulated_plan

                candidate_tools = [
                    e.grid_pos
                    for e in entities
                    if e.role
                    in (
                        EntityRole.MANIPULABLE,
                        EntityRole.RESOURCE,
                        EntityRole.UNKNOWN,
                        EntityRole.ACTUATOR,
                    )
                    and e.feature_id not in av_feats
                    and e.grid_pos not in goals
                    and e.grid_pos != cur_pos
                    and (
                        e.grid_pos in reachable_cells
                        or any(
                            (e.grid_pos[0] + dr, e.grid_pos[1] + dc) in reachable_cells
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        )
                    )
                ]
                for t_pos in candidate_tools:
                    t_feat = (
                        int(curr_grid[t_pos[0], t_pos[1]])
                        if 0 <= t_pos[0] < H and 0 <= t_pos[1] < W
                        else None
                    )
                    t_standable = (
                        t_feat is not None
                        and engine.symbolic_theory.is_walkable(t_feat)
                        and t_pos not in effective_barriers
                    )
                    if t_standable:
                        t_path = find_shortest_path(cur_pos, {t_pos}, effective_barriers)
                    else:
                        adj_cells = {
                            (t_pos[0] + dr, t_pos[1] + dc)
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            if 0 <= t_pos[0] + dr < H
                            and 0 <= t_pos[1] + dc < W
                            and (t_pos[0] + dr, t_pos[1] + dc) in reachable_cells
                        }
                        if cur_pos in adj_cells:
                            t_path = []
                        elif adj_cells:
                            t_path = find_shortest_path(cur_pos, adj_cells, effective_barriers)
                        else:
                            t_path = None

                    if t_path is not None:
                        score = 1
                        if t_feat is not None:
                            # Relational salience: boost if matching a singleton cue
                            if int(np.sum(curr_grid == t_feat)) == 2:
                                score = 5
                        candidate_triggers.append((t_pos, t_path, score, set()))

            if not candidate_triggers:
                break

            if len(candidate_triggers) > 1:
                c_subgoals = [
                    FrontopolarSubgoal(
                        subgoal_id=f"cand_trig_{c_pos[0]}_{c_pos[1]}",
                        subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
                        target_entity_pos=c_pos,
                        target_destination=c_pos,
                        priority=float(score),
                    )
                    for c_pos, _, score, _ in candidate_triggers
                ]
                held_set: set[int] = set()
                if hasattr(engine.working_memory, "held_feature_ids"):
                    held_set = set(engine.working_memory.held_feature_ids)
                elif (
                    hasattr(engine.working_memory, "held_items")
                    and engine.working_memory.held_items
                ):
                    items = engine.working_memory.held_items
                    if isinstance(items, dict):
                        held_set = {
                            int(k)
                            if isinstance(k, (int, np.integer))
                            else getattr(v, "feature_id", 0)
                            for k, v in items.items()
                        }
                    else:
                        held_set = {
                            item.feature_id if isinstance(item, HeldItem) else int(item)
                            for item in items
                        }
                mcts_res = FrontopolarSubgoalMCTSPlanner.plan_subgoal_mcts(
                    avatar_pos=cur_pos,
                    candidate_subgoals=c_subgoals,
                    goal_positions=goals,
                    barriers=effective_barriers,
                    grid_bounds=(H, W),
                    held_items=held_set,
                    lethal_positions=engine.current_level_lethal_positions,
                    max_iterations=32,
                    body_schema=get_sub_attr(engine.working_memory, "body_schema", None),
                    tool_barrier_affinities=getattr(
                        engine.working_memory, "tool_barrier_affinities", None
                    ),
                    curr_grid=curr_grid,
                )
                if mcts_res.best_subgoals:
                    best_sg = mcts_res.best_subgoals[0]
                    matching = [c for c in candidate_triggers if c[0] == best_sg.target_entity_pos]
                    if matching:
                        candidate_triggers = matching

            candidate_triggers.sort(key=lambda x: (-x[2], len(x[1])))
            chosen_pos, switch_path, _, opened_cells = candidate_triggers[0]
            visited_triggers.add(chosen_pos)

            new_sg = FrontopolarSubgoal(
                subgoal_id=f"trigger_{chosen_pos[0]}_{chosen_pos[1]}",
                subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
                target_entity_pos=chosen_pos,
                target_destination=chosen_pos,
                priority=2.0,
            )
            if buffered_subgoals:
                new_sg.prerequisite_subgoal_ids = [buffered_subgoals[-1].subgoal_id]
            buffered_subgoals.append(new_sg)

            accumulated_plan.extend(switch_path)
            if switch_path:
                cur_pos = switch_path[-1].predicted_avatar_pos
            open_barriers.update(opened_cells)
            open_barriers.update(mutation_triggers.get(chosen_pos, set()))

            matching_mutations = [
                m
                for m in engine.state_mutations
                if (
                    m.trigger_pos == chosen_pos
                    or (
                        m.trigger_feature is not None
                        and int(curr_grid[chosen_pos[0], chosen_pos[1]]) == m.trigger_feature
                    )
                )
            ]
            if matching_mutations and matching_mutations[0].trigger_type == "ACTION":
                act_id = matching_mutations[0].metadata.get("action_id")
                if act_id is not None and act_id in available_actions:
                    accumulated_plan.append(
                        MentalSimulationStep(
                            action=int(act_id),
                            predicted_avatar_pos=cur_pos,
                            expected_mutation="ACTION_TRIGGER",
                        )
                    )
            else:
                ch_feat = (
                    int(curr_grid[chosen_pos[0], chosen_pos[1]])
                    if 0 <= chosen_pos[0] < H and 0 <= chosen_pos[1] < W
                    else None
                )
                if ch_feat is not None and not engine.symbolic_theory.is_walkable(ch_feat):
                    discrete_acts = [
                        a
                        for a in available_actions
                        if not engine.is_displacement_action(a)
                        and not engine.is_spatial_effector(a)
                    ]
                    if discrete_acts:
                        accumulated_plan.append(
                            MentalSimulationStep(
                                action=discrete_acts[0],
                                predicted_avatar_pos=cur_pos,
                                expected_mutation="INTERACT_TRIGGER",
                            )
                        )

        return None

    @staticmethod
    def _plan_teleological_cutset_subgoals(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        start_pos: tuple[int, int],
        goals: set[tuple[int, int]],
        static_barriers: set[tuple[int, int]],
        movable_actions: list[tuple[int, int, int]],
        oriented_threats: Sequence[OrientedThreat] = (),
    ) -> list[MentalSimulationStep] | None:
        """Teleological Retrocausal Decomposition (Human BA10 Reasoning):
        When direct path to goal is blocked by barriers/doors:
        1. Find relaxed geodesic from start_pos to closest goal ignoring static barriers.
        2. Identify the first blocking barrier cut-set along this trajectory.
        3. Backward-chain to discover potential keys, switches, levers, or matching affordances:
           - Matching-color entity (key of same feature as barrier)
           - Reachable manipulable entities / actuators
           - Gateway interaction stances adjacent to the barrier
        4. Push a high-priority FrontopolarSubgoal and return path to candidate key/trigger.
        """
        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        lethal_positions = engine.current_level_lethal_positions
        lethal_features = engine.hazard_tracker.known_lethal_features

        # Living sentry sightlines and physical threat footprints are never relaxable
        living_gaze: set[tuple[int, int]] = {
            t.gaze_pos for t in oriented_threats if t.gaze_pos is not None
        }
        threat_cells: set[tuple[int, int]] = set()
        for t in oriented_threats:
            threat_cells.add(t.pos)
            tr, tc = t.pos
            for dr_t in (-1, 0, 1):
                for dc_t in (-1, 0, 1):
                    threat_cells.add((tr + dr_t, tc + dc_t))

        # 1. Relaxed BFS to find trajectory towards goal ignoring static barriers
        q: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque([(start_pos, [start_pos])])
        visited_relaxed = {start_pos}
        relaxed_path: list[tuple[int, int]] | None = None

        while q:
            p, path = q.popleft()
            if p in goals:
                relaxed_path = path
                break
            for _, dr, dc in movable_actions:
                nr, nc = p[0] + dr, p[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in visited_relaxed or (nr, nc) in lethal_positions:
                    continue
                if (nr, nc) in living_gaze or (nr, nc) in threat_cells:
                    continue
                if int(curr_grid[nr, nc]) in lethal_features:
                    continue
                visited_relaxed.add((nr, nc))
                q.append(((nr, nc), path + [(nr, nc)]))

        if not relaxed_path:
            return None

        # 2. Identify the first blocking barrier cell along the relaxed path
        blocking_barrier_pos: tuple[int, int] | None = None
        for step_p in relaxed_path:
            if step_p in static_barriers or engine.symbolic_theory.is_barrier(
                int(curr_grid[step_p[0], step_p[1]])
            ):
                blocking_barrier_pos = step_p
                break

        if blocking_barrier_pos is None:
            return None

        # Sentry lines of fire and creature bodies are threats, not unlockable doors
        if blocking_barrier_pos in living_gaze or blocking_barrier_pos in threat_cells:
            return None

        # 3. Discover reachable spatial manifold from start_pos avoiding static_barriers
        reachable_cells = {start_pos}
        reach_q = deque([start_pos])
        parent_map: dict[tuple[int, int], tuple[tuple[int, int], Any]] = {}

        while reach_q:
            cp = reach_q.popleft()
            for act, dr, dc in movable_actions:
                if (cp, act) in engine.failed_transitions:
                    continue
                np_pos = (cp[0] + dr, cp[1] + dc)
                if not (0 <= np_pos[0] < H and 0 <= np_pos[1] < W):
                    continue
                if np_pos in reachable_cells:
                    continue
                if np_pos in static_barriers or np_pos in lethal_positions:
                    continue
                if np_pos in living_gaze or np_pos in threat_cells:
                    continue
                if engine.symbolic_theory.is_barrier(int(curr_grid[np_pos[0], np_pos[1]])):
                    continue
                if int(curr_grid[np_pos[0], np_pos[1]]) in lethal_features:
                    continue
                reachable_cells.add(np_pos)
                parent_map[np_pos] = (cp, act)
                reach_q.append(np_pos)

        def reconstruct_path(dest: tuple[int, int]) -> list[MentalSimulationStep]:
            steps: list[MentalSimulationStep] = []
            curr = dest
            while curr in parent_map:
                prev_p, act = parent_map[curr]
                steps.append(MentalSimulationStep(action=act, predicted_avatar_pos=curr))
                curr = prev_p
            steps.reverse()
            return steps

        # Extract separating barrier cut-set cells on the perimeter of the reachable manifold
        boundary_barriers: set[tuple[int, int]] = set()
        for r, c in reachable_cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nbr = (r + dr, c + dc)
                if nbr not in reachable_cells and 0 <= nbr[0] < H and 0 <= nbr[1] < W:
                    if nbr in static_barriers or engine.symbolic_theory.is_barrier(
                        int(curr_grid[nbr[0], nbr[1]])
                    ):
                        # Must border an unreached cell to be a separating cut-set boundary (not an isolated token on floor)
                        borders_unreached = False
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nbr2 = (nbr[0] + dr2, nbr[1] + dc2)
                            if (
                                0 <= nbr2[0] < H
                                and 0 <= nbr2[1] < W
                                and nbr2 not in reachable_cells
                                and nbr2 not in static_barriers
                                and not engine.symbolic_theory.is_barrier(
                                    int(curr_grid[nbr2[0], nbr2[1]])
                                )
                            ):
                                borders_unreached = True
                                break
                        if borders_unreached:
                            boundary_barriers.add(nbr)

        boundary_barrier_feats = {
            int(curr_grid[br, bc])
            for br, bc in boundary_barriers
            if (br, bc) not in living_gaze and (br, bc) not in threat_cells
        }

        # 4. Search for teleological unlocking keys or interactable entities
        entities = engine.extract_entities(curr_grid, bg)
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        candidate_subgoal_targets: list[
            tuple[tuple[int, int], float, str, list[MentalSimulationStep]]
        ] = []

        discrete_acts_teleo = [
            a
            for a in engine.action_dynamics
            if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
        ]

        for e in entities:
            if e.grid_pos == start_pos or e.feature_id in av_feats or e.feature_id == bg:
                continue
            if engine.symbolic_theory.is_position_explored(e.grid_pos):
                continue
            if e.grid_pos in goals:
                continue
            if e.grid_pos in living_gaze or e.grid_pos in threat_cells:
                continue
            if e.feature_id in lethal_features:
                continue
            if e.grid_pos in boundary_barriers:
                continue

            # Determine path to candidate entity
            target_path: list[MentalSimulationStep] | None = None
            if e.grid_pos in reachable_cells and engine.symbolic_theory.is_walkable(e.feature_id):
                target_path = reconstruct_path(e.grid_pos)
            else:
                # Entity pos might be tentatively marked as barrier or non-walkable actuator; check if approachable from adjacent manifold
                best_adj_p: list[MentalSimulationStep] | None = None
                for act, dr, dc in movable_actions:
                    adj_p = (e.grid_pos[0] - dr, e.grid_pos[1] - dc)
                    if adj_p in reachable_cells:
                        p_to_adj = reconstruct_path(adj_p)
                        if discrete_acts_teleo and not engine.symbolic_theory.is_walkable(
                            e.feature_id
                        ):
                            interact_step = MentalSimulationStep(
                                action=int(discrete_acts_teleo[0]),
                                predicted_avatar_pos=adj_p,
                                expected_mutation="INTERACT_TRIGGER",
                            )
                            full_p = p_to_adj + [interact_step]
                        else:
                            step_into = MentalSimulationStep(
                                action=act, predicted_avatar_pos=e.grid_pos
                            )
                            full_p = p_to_adj + [step_into]
                        if best_adj_p is None or len(full_p) < len(best_adj_p):
                            best_adj_p = full_p
                target_path = best_adj_p

            if not target_path:
                continue

            # A. Color-matching Key / Affordance (Feature match with door in boundary cut-set)
            if e.feature_id in boundary_barrier_feats and e.area <= 16:
                candidate_subgoal_targets.append(
                    (e.grid_pos, 200.0, f"color_match_key_{e.feature_id}", target_path)
                )
            # B. Small manipulable / pickup / switch items
            elif (
                e.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE, EntityRole.ACTUATOR)
                and e.area <= 16
            ):
                candidate_subgoal_targets.append(
                    (e.grid_pos, 100.0, f"interactive_entity_{e.feature_id}", target_path)
                )
            # C. Minority pop-out or isolated token
            elif 1 <= e.area <= 9:
                candidate_subgoal_targets.append(
                    (e.grid_pos, 60.0, f"salient_token_{e.feature_id}", target_path)
                )

        # Stance immediately adjacent to the blocking barrier
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (blocking_barrier_pos[0] + dr, blocking_barrier_pos[1] + dc)
            if (
                adj in reachable_cells
                and adj != start_pos
                and adj not in living_gaze
                and adj not in threat_cells
            ):
                st_path = reconstruct_path(adj)
                if st_path:
                    candidate_subgoal_targets.append(
                        (adj, 40.0, f"barrier_approach_stance_{blocking_barrier_pos}", st_path)
                    )

        if not candidate_subgoal_targets:
            return None

        if len(candidate_subgoal_targets) > 1:
            c_subgoals = [
                FrontopolarSubgoal(
                    subgoal_id=f"teleo_cand_{t_pos[0]}_{t_pos[1]}",
                    subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
                    target_entity_pos=t_pos,
                    target_destination=t_pos,
                    priority=float(score),
                )
                for t_pos, score, _, _ in candidate_subgoal_targets
            ]
            held_set: set[int] = set()
            if hasattr(engine.working_memory, "held_feature_ids"):
                held_set = set(engine.working_memory.held_feature_ids)
            elif hasattr(engine.working_memory, "held_items") and engine.working_memory.held_items:
                items = engine.working_memory.held_items
                if isinstance(items, dict):
                    held_set = {
                        int(k) if isinstance(k, (int, np.integer)) else getattr(v, "feature_id", 0)
                        for k, v in items.items()
                    }
                else:
                    held_set = {
                        item.feature_id if isinstance(item, HeldItem) else int(item)
                        for item in items
                    }
            mcts_res = FrontopolarSubgoalMCTSPlanner.plan_subgoal_mcts(
                avatar_pos=start_pos,
                candidate_subgoals=c_subgoals,
                goal_positions=goals,
                barriers=static_barriers,
                grid_bounds=(H, W),
                held_items=held_set,
                lethal_positions=engine.current_level_lethal_positions,
                max_iterations=32,
                body_schema=get_sub_attr(engine.working_memory, "body_schema", None),
                tool_barrier_affinities=getattr(
                    engine.working_memory, "tool_barrier_affinities", None
                ),
                curr_grid=curr_grid,
            )
            if mcts_res.best_subgoals:
                best_sg = mcts_res.best_subgoals[0]
                matching = [
                    c for c in candidate_subgoal_targets if c[0] == best_sg.target_entity_pos
                ]
                if matching:
                    rest = [
                        c for c in candidate_subgoal_targets if c[0] != best_sg.target_entity_pos
                    ]
                    candidate_subgoal_targets = matching + rest

        # Sort candidates by priority score descending, then by path length
        candidate_subgoal_targets.sort(key=lambda x: (-x[1], len(x[3])))

        # Teleological backward chaining requires a concrete candidate affordance (key, trigger, token)
        # or an item in inventory to apply; approaching an impassable blank barrier without tools is a dead-end
        if candidate_subgoal_targets[0][1] < 60.0 and not engine.working_memory.held_items:
            return None

        best_target, _, intent_name, path_to_target = candidate_subgoal_targets[0]

        logger.info(
            "MentalSimulationPlanner: Formulated Teleological Backward Subgoal [%s at %s] "
            "to unlock barrier %s (%d steps).",
            intent_name,
            best_target,
            blocking_barrier_pos,
            len(path_to_target),
        )

        # Push to BA10 Subgoal Stack
        teleo_sg = FrontopolarSubgoal(
            subgoal_id=f"teleo_{intent_name}_{best_target[0]}_{best_target[1]}",
            subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
            target_entity_pos=best_target,
            target_destination=best_target,
            priority=2.5,
        )
        engine.subgoal_stack.push_subgoal(teleo_sg)

        return path_to_target


@dataclass
class ReplanResult:
    """W140: Outcome of unexpected state divergence replanning."""

    replanned: bool
    new_plan: list[MentalSimulationStep]
    divergence_reason: str
    replan_count: int

    def to_causal_predicate(self) -> CausalPredicate:
        """Export replan result as a declarative causal predicate."""
        return CausalPredicate(
            variable="replan_result",
            operator="==",
            value=self.replanned,
            relation=f"count:{self.replan_count}",
            target_entity=self.divergence_reason,
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording replan result."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"replan_{self.replan_count}_{self.replanned}",
                node_data={
                    "replanned": self.replanned,
                    "plan_len": len(self.new_plan),
                    "divergence_reason": self.divergence_reason,
                    "replan_count": self.replan_count,
                },
            )
        ]


class DynamicReplanner:
    """W140: Triggers dynamic plan repair and topological rerouting upon unexpected observation divergences."""

    @staticmethod
    def replan_on_divergence(
        current_actual_pos: tuple[int, int],
        expected_pos: tuple[int, int],
        remaining_plan: list[MentalSimulationStep],
        goal_positions: set[tuple[int, int]],
        barriers: set[tuple[int, int]],
        grid_shape: tuple[int, int],
    ) -> ReplanResult:
        """Repairs or generates a new plan from actual position to goal."""
        if current_actual_pos == expected_pos and remaining_plan:
            return ReplanResult(
                replanned=False,
                new_plan=remaining_plan,
                divergence_reason="nominal",
                replan_count=0,
            )

        target = list(goal_positions)[0] if goal_positions else expected_pos
        H, W = grid_shape
        queue = [(current_actual_pos, [])]
        visited = {current_actual_pos}
        found_path: list[tuple[tuple[int, int], int]] | None = None

        while queue:
            curr, path = queue.pop(0)
            if curr == target or curr in goal_positions:
                found_path = path
                break
            for dr, dc, act in [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                nxt = (nr, nc)
                if 0 <= nr < H and 0 <= nc < W and nxt not in barriers and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [(nxt, act)]))

        if found_path is not None:
            new_steps = [
                MentalSimulationStep(action=act, predicted_avatar_pos=pos)
                for pos, act in found_path
            ]
            return ReplanResult(
                replanned=True,
                new_plan=new_steps,
                divergence_reason=f"diverged_from_{expected_pos}_to_{current_actual_pos}",
                replan_count=1,
            )

        return ReplanResult(
            replanned=False,
            new_plan=[],
            divergence_reason="unreachable_after_divergence",
            replan_count=1,
        )

    @classmethod
    def to_causal_predicate(cls, view: Any = None) -> CausalPredicate:
        """Export dynamic replanner capability and runtime divergence count as a declarative causal predicate."""
        val = "ACTIVE"
        rel = "divergence_rerouting"
        if view is not None:
            fails = getattr(view, "consecutive_plan_failures", 0)
            rel = f"divergence_rerouting:plan_failures:{fails}"
        return CausalPredicate(
            variable="dynamic_replanner",
            operator="==",
            value=val,
            relation=rel,
            target_entity="cortex_planner",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering dynamic replanner."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="dynamic_replanner",
                node_data={
                    "faculty": "dynamic_replanner",
                    "status": "ACTIVE",
                },
            )
        ]


@dataclass
class FrontopolarSubgoalMCTSNode:
    """MCTS node representing an abstract subgoal state in Frontopolar BA10 tree search."""

    subgoal: FrontopolarSubgoal | None
    state_avatar_pos: tuple[int, int]
    state_held_items: frozenset[int]
    state_cleared_barriers: frozenset[tuple[int, int]]
    parent: FrontopolarSubgoalMCTSNode | None = None
    children: list[FrontopolarSubgoalMCTSNode] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    untried_subgoals: list[FrontopolarSubgoal] = field(default_factory=list)
    depth: int = 0

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, c: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        if not self.parent or self.parent.visits == 0:
            return self.q_value
        return self.q_value + c * float(np.sqrt(np.log(self.parent.visits) / self.visits))


@dataclass
class FrontopolarSubgoalMCTSResult:
    """Outcome of BA10 Frontopolar Subgoal Monte Carlo Tree Search."""

    best_subgoals: list[FrontopolarSubgoal]
    total_iterations: int
    tree_depth: int
    expected_value: float
    pruned_dead_ends: int

    def to_causal_predicate(self) -> CausalPredicate:
        """Export MCTS plan outcome as a declarative causal predicate."""
        return CausalPredicate(
            variable="frontopolar_subgoal_mcts",
            operator="==",
            value=len(self.best_subgoals),
            relation=f"val:{self.expected_value:.2f}:pruned:{self.pruned_dead_ends}",
            target_entity="cortex_planner",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions registering the planned subgoal sequence."""
        ops: list[TransactionOperation] = []
        for idx, sg in enumerate(self.best_subgoals):
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=f"mcts_subgoal_{idx}_{sg.subgoal_id}",
                    node_data={
                        "order": idx,
                        "subgoal_id": sg.subgoal_id,
                        "subgoal_type": str(sg.subgoal_type),
                        "target_pos": list(sg.target_entity_pos) if sg.target_entity_pos else None,
                        "priority": sg.priority,
                    },
                )
            )
        return ops


class FrontopolarSubgoalMCTSPlanner:
    """BA10 Frontopolar Subgoal Monte Carlo Tree Search Planner.

    Executes hierarchical tree search over abstract subgoals rather than primitive steps:
    1. Selection: UCB1 exploration-exploitation over abstract subgoal nodes.
    2. Expansion & Physical/Habenular Dead-End Pruning:
       - Prunes branches requiring movement across impermeable barriers without required keys (Spelke continuity).
       - Prunes paths entering grounded Habenular lethal zones or death loci.
    3. Rollout Simulation:
       - Evaluates path connectivity from current state through candidates to terminal goal.
    4. Backpropagation:
       - Backpropagates path efficiency and reward up the subgoal hierarchy.
    """

    @staticmethod
    def _is_reachable(
        start: tuple[int, int],
        goal: tuple[int, int],
        effective_barriers: set[tuple[int, int]],
        grid_bounds: tuple[int, int],
    ) -> bool:
        """Check 4-way BFS topological reachability from start to goal."""
        if start == goal:
            return True
        H, W = grid_bounds
        queue = deque([start])
        visited = {start}

        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True
            cr, cc = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                nxt = (nr, nc)
                if 0 <= nr < H and 0 <= nc < W and nxt not in visited:
                    if nxt == goal or nxt not in effective_barriers:
                        visited.add(nxt)
                        queue.append(nxt)
        return False

    @classmethod
    def plan_subgoal_mcts(
        cls,
        avatar_pos: tuple[int, int],
        candidate_subgoals: Sequence[FrontopolarSubgoal],
        goal_positions: set[tuple[int, int]],
        barriers: set[tuple[int, int]],
        grid_bounds: tuple[int, int],
        held_items: set[int] | set[HeldItem] | Sequence[HeldItem] | None = None,
        lethal_positions: set[tuple[int, int]] | None = None,
        max_iterations: int = 48,
        exploration_weight: float = 1.414,
        max_depth: int = 5,
        body_schema: Any | None = None,
        tool_barrier_affinities: dict[int, set[int]] | None = None,
        curr_grid: np.ndarray | None = None,
    ) -> FrontopolarSubgoalMCTSResult:
        """Run Frontopolar Subgoal MCTS with Spelke/Habenular pruning and Extended Body Schema dynamics."""
        if not candidate_subgoals:
            return FrontopolarSubgoalMCTSResult(
                best_subgoals=[],
                total_iterations=0,
                tree_depth=0,
                expected_value=0.0,
                pruned_dead_ends=0,
            )

        lethal = lethal_positions or set()
        normalized_held: set[int] = set()
        if held_items:
            for item in held_items:
                if isinstance(item, HeldItem):
                    normalized_held.add(item.feature_id)
                elif hasattr(item, "feature_id"):
                    normalized_held.add(int(item.feature_id))
                else:
                    normalized_held.add(int(item))

        root = FrontopolarSubgoalMCTSNode(
            subgoal=None,
            state_avatar_pos=avatar_pos,
            state_held_items=frozenset(normalized_held),
            state_cleared_barriers=frozenset(),
            untried_subgoals=list(candidate_subgoals),
            depth=0,
        )

        pruned_count = 0

        for _ in range(max_iterations):
            node = root

            # 1. Selection
            while node.children and not node.untried_subgoals:
                node = max(node.children, key=lambda c: c.ucb1(exploration_weight))

            # 2. Expansion
            if node.untried_subgoals and node.depth < max_depth:
                cand_sg = node.untried_subgoals.pop(0)
                cand_target = cand_sg.target_entity_pos or cand_sg.target_destination

                # Habenular Lethal Pruning
                if cand_target in lethal:
                    pruned_count += 1
                    continue

                # Spelke Continuity / Barrier Reachability Pruning with Extended Body Schema Permeability
                effective_barriers = set(barriers) - set(node.state_cleared_barriers)
                if curr_grid is not None and (tool_barrier_affinities or body_schema):
                    permeable_feats: set[int] = set()
                    for h_feat in node.state_held_items:
                        if tool_barrier_affinities and h_feat in tool_barrier_affinities:
                            permeable_feats.update(tool_barrier_affinities[h_feat])
                        if body_schema and hasattr(body_schema, "tool_barrier_affinities"):
                            permeable_feats.add(h_feat)
                            permeable_feats.update(
                                body_schema.tool_barrier_affinities.get(h_feat, set())
                            )
                    if permeable_feats:
                        H_b, W_b = grid_bounds
                        effective_barriers = {
                            b
                            for b in effective_barriers
                            if not (
                                0 <= b[0] < H_b
                                and 0 <= b[1] < W_b
                                and int(curr_grid[b[0], b[1]]) in permeable_feats
                            )
                        }

                # Prerequisite checking: clearing obstacles or unlocking mechanisms requires held key/tool
                if cand_sg.subgoal_type in [
                    SubgoalType.CLEAR_OBSTACLE,
                    SubgoalType.UNLOCK_REMOTE_MECHANISM,
                ]:
                    req_f = getattr(cand_sg, "required_feature", None)
                    if req_f is not None and req_f not in node.state_held_items:
                        pruned_count += 1
                        continue

                if cand_target is not None:
                    if not cls._is_reachable(
                        node.state_avatar_pos,
                        cand_target,
                        effective_barriers,
                        grid_bounds,
                    ):
                        pruned_count += 1
                        continue

                # Transition state
                next_pos = cand_target if cand_target is not None else node.state_avatar_pos
                next_held = set(node.state_held_items)
                next_cleared = set(node.state_cleared_barriers)

                if cand_sg.subgoal_type in [SubgoalType.COLLECT_ITEM, SubgoalType.ACQUIRE_TOOL]:
                    feat = getattr(
                        cand_sg, "required_feature", getattr(cand_sg, "target_feature", None)
                    )
                    if feat is not None:
                        next_held.add(feat)
                        # Predict resonant barrier clearing upon tool acquisition
                        if curr_grid is not None:
                            unlocked_f: set[int] = set()
                            if tool_barrier_affinities and feat in tool_barrier_affinities:
                                unlocked_f.update(tool_barrier_affinities[feat])
                            if body_schema and hasattr(body_schema, "tool_barrier_affinities"):
                                unlocked_f.add(feat)
                                unlocked_f.update(
                                    body_schema.tool_barrier_affinities.get(feat, set())
                                )
                            if unlocked_f:
                                H_b, W_b = grid_bounds
                                for br, bc in barriers:
                                    if (
                                        0 <= br < H_b
                                        and 0 <= bc < W_b
                                        and int(curr_grid[br, bc]) in unlocked_f
                                    ):
                                        next_cleared.add((br, bc))
                elif cand_sg.subgoal_type in [
                    SubgoalType.CLEAR_OBSTACLE,
                    SubgoalType.UNLOCK_REMOTE_MECHANISM,
                ]:
                    if cand_sg.target_destination:
                        next_cleared.add(cand_sg.target_destination)
                    if cand_sg.target_entity_pos:
                        next_cleared.add(cand_sg.target_entity_pos)

                remaining_cands = [
                    s for s in candidate_subgoals if s.subgoal_id != cand_sg.subgoal_id
                ]
                child = FrontopolarSubgoalMCTSNode(
                    subgoal=cand_sg,
                    state_avatar_pos=next_pos,
                    state_held_items=frozenset(next_held),
                    state_cleared_barriers=frozenset(next_cleared),
                    parent=node,
                    untried_subgoals=remaining_cands,
                    depth=node.depth + 1,
                )
                node.children.append(child)
                target_node = child
            else:
                target_node = node

            # 3. Simulation / Value Evaluation
            sim_pos = target_node.state_avatar_pos
            if goal_positions and sim_pos in goal_positions:
                rollout_val = 100.0 - target_node.depth * 5.0
            elif goal_positions:
                min_d = min(
                    abs(gr - sim_pos[0]) + abs(gc - sim_pos[1]) for gr, gc in goal_positions
                )
                rollout_val = 50.0 / (1.0 + min_d) - target_node.depth * 2.0
            else:
                rollout_val = 10.0 - target_node.depth

            # 4. Backpropagation
            curr = target_node
            while curr is not None:
                curr.visits += 1
                curr.total_value += rollout_val
                curr = curr.parent

        # Extract optimal trajectory from tree
        best_path: list[FrontopolarSubgoal] = []
        curr = root
        while curr.children:
            curr = max(curr.children, key=lambda c: c.visits)
            if curr.subgoal is not None:
                best_path.append(curr.subgoal)

        return FrontopolarSubgoalMCTSResult(
            best_subgoals=best_path,
            total_iterations=max_iterations,
            tree_depth=len(best_path),
            expected_value=root.q_value,
            pruned_dead_ends=pruned_count,
        )


@dataclass
class SimulationStage:
    """Cortical Stage 3: Counterfactual Simulation & Subgoal Scheduling.

    OFC mental simulation rollouts; dead-end pruning; BA10 recursive subgoal decomposition.
    """

    name: str = "simulation_stage"
    engine: Any | None = None

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        engine = getattr(self, "engine", None)
        if engine is not None or ctx.signals.get("_unified_deliberation"):
            plan_len = len(getattr(engine, "mental_plan", ctx.mental_plan) or ())
            sg_obj = getattr(engine, "subgoal_stack", None) or ctx.subgoal_stack
            sg_depth = len(getattr(sg_obj, "stack", ()) or ())
            active_goals = len(
                getattr(engine, "learned_goal_positions", ctx.learned_goal_positions) or ()
            )
            return StageResult(
                author=self.name,
                signals={
                    "simulation_summary": {
                        "plan_len": plan_len,
                        "subgoal_stack_depth": sg_depth,
                        "active_goals": active_goals,
                    }
                },
            )
        proposed = None
        if ctx.mental_plan:
            first_step = ctx.mental_plan[0]
            proposed = first_step.action if hasattr(first_step, "action") else first_step

        ops: list[TransactionOperation] = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="simulation_plan",
                node_data={
                    "stage": self.name,
                    "plan_len": len(ctx.mental_plan),
                    "step": ctx.step_counter,
                },
            )
        ]
        signals = {"mental_plan_action": proposed} if proposed is not None else {}
        return StageResult(
            author=self.name,
            operations=ops,
            signals=signals,
            proposed_action=proposed,
        )
