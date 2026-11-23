"""Apex orbitofrontal & frontopolar goal priority scheduler (BA10 / OFC)."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import label as _scipy_label

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal.transition_kb import TransitionOutcome
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.frontopolar_subgoal_stack import SubgoalType
from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


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
                    "new_learned_positions": [
                        list(p) if isinstance(p, tuple) else p
                        for p in self.new_learned_goal_positions
                    ],
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
        f_ev = th.evaluate_transition(feature_id, locus=entity_pos)
        should_check_reversible = (
            not has_cargo_context
            and avatar_pos is not None
            and int(curr_grid[er, ec]) == int(feature_id)
            and not (f_ev.traversable and f_ev.confidence > 0.0)
            and f_ev.goal_value <= 0.5
            and feature_id not in th.goal_features
            and entity_pos not in engine.learned_goal_positions
            and f_ev.traversable
            and f_ev.risk_cost < 40.0
            and 1 <= entity_area <= 4
            and 0 < er < H - 1
            and 0 < ec < W - 1
        )
        if not should_check_reversible or avatar_pos is None:
            return None
        goal_like = th.goal_features | th.candidate_goal_features | th.receptacle_features
        safe_stances: list[tuple[int, int]] = []
        unsafe_stances: list[tuple[int, int]] = []
        kb = th.transition_kb
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ap = (er + dr, ec + dc)
            if not (0 <= ap[0] < H and 0 <= ap[1] < W):
                continue
            if ap in engine.current_context_hazards:
                continue
            ap_feat = int(curr_grid[ap[0], ap[1]])
            ev_ap = th.evaluate_transition(ap_feat, locus=ap)
            if not ev_ap.traversable or ev_ap.risk_cost >= 40.0:
                continue
            if ap in engine.learned_barriers and not (
                kb.get_belief(ap, scope="locus") is not None and ev_ap.confidence > 0.0
            ):
                continue
            if not (
                ap_feat == bg or (ev_ap.traversable and ev_ap.confidence > 0.0) or ap == avatar_pos
            ):
                continue
            b1 = (er - dr, ec - dc)
            b2 = (er - 2 * dr, ec - 2 * dc)
            b1_open = False
            b2_open = False
            if 0 <= b1[0] < H and 0 <= b1[1] < W:
                f1 = int(curr_grid[b1[0], b1[1]])
                ev1 = th.evaluate_transition(f1, locus=b1)
                b1_locus = kb.get_belief(b1, scope="locus")
                b1_bar = b1 in engine.learned_barriers and not (
                    b1_locus is not None and ev1.traversable and ev1.confidence > 0.0
                )
                b1_open = (
                    not b1_bar
                    and b1 not in engine.current_context_hazards
                    and ev1.traversable
                    and ev1.risk_cost < 40.0
                    and (
                        f1 == bg
                        or b1 == avatar_pos
                        or th.is_avatar_feature(f1)
                        or ev1.confidence > 0.0
                        or f1 in goal_like
                    )
                )
            if 0 <= b2[0] < H and 0 <= b2[1] < W:
                f2 = int(curr_grid[b2[0], b2[1]])
                ev2 = th.evaluate_transition(f2, locus=b2)
                b2_locus = kb.get_belief(b2, scope="locus")
                b2_bar = b2 in engine.learned_barriers and not (
                    b2_locus is not None and ev2.traversable and ev2.confidence > 0.0
                )
                b2_open = (
                    not b2_bar
                    and ev2.traversable
                    and ev2.risk_cost < 40.0
                    and (
                        f2 == bg
                        or b2 == avatar_pos
                        or th.is_avatar_feature(f2)
                        or ev2.confidence > 0.0
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
        if engine.object_planner.ledger.is_in_newly_reachable({entity_pos}):
            safe_stances = None
        else:
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
                if ap not in engine.exhausted_candidate_goals and not th.is_locus_explored(ap)
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
        th = engine.symbolic_theory
        adj_walkable: list[tuple[int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ap = (entity_pos[0] + dr, entity_pos[1] + dc)
            if not (0 <= ap[0] < H and 0 <= ap[1] < W):
                continue
            ap_val = int(curr_grid[ap[0], ap[1]])
            if ap_val == bg or ap == avatar_pos:
                adj_walkable.append(ap)
                continue
            ap_ev = th.evaluate_transition(ap_val, locus=ap)
            if ap_ev.traversable and ap_ev.confidence > 0.0 and ap_ev.risk_cost < 40.0:
                adj_walkable.append(ap)
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
        any_adj_probed = any(
            engine.symbolic_theory.is_locus_actuator_probed(ap) for ap in adj_walkable
        )
        added_adj = 0
        if not any_adj_probed:
            for ap in adj_walkable:
                if ap == avatar_pos and just_probed_here:
                    engine.symbolic_theory.mark_locus_explored(ap, reason="actuator_probed")
                    engine.exhausted_candidate_goals.add(ap)
                    continue
                if ap in engine.exhausted_candidate_goals or (
                    engine.symbolic_theory.is_locus_explored(ap) and ap != avatar_pos
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
        is_singleton = int(np.count_nonzero(curr_grid == feature_id)) == 1
        allow_contact_escalation = added_adj == 0 or (
            any_adj_probed and (is_singleton or feature_id in engine.symbolic_theory.goal_features)
        )
        feat_ev = th.evaluate_transition(feature_id, locus=entity_pos)
        ent_locus_b = th.transition_kb.get_belief(entity_pos, scope="locus")
        in_learned_bar = entity_pos in engine.learned_barriers and not (
            ent_locus_b is not None and feat_ev.traversable and feat_ev.confidence > 0.0
        )
        if (
            allow_contact_escalation
            and adj_walkable
            and entity_pos not in engine.exhausted_candidate_goals
            and not engine.symbolic_theory.is_locus_explored(entity_pos)
            and not in_learned_bar
            and entity_pos not in engine.current_context_hazards
            and feat_ev.traversable
            and feat_ev.risk_cost < 40.0
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
        hz = engine.hazard_tracker
        lethal_feats = set(getattr(th, "lethal_features", frozenset()) or ())
        lethal_feats.update(EpistemicGoalScheduler._effective_lethal_features(engine))
        opened_portals: set[tuple[int, int]] = set()
        for mut in engine.state_mutations:
            mut_meta = mut.metadata or {}
            for mr, mc in mut_meta.get("mutated_positions", ()):
                if 0 <= mr < H and 0 <= mc < W:
                    mv = int(curr_grid[mr, mc])
                    if mv == bg or th.is_walkable(mv):
                        opened_portals.add((mr, mc))

        if len(goals) <= 1 and not opened_portals:
            return goals, set()

        static_lethal = hz.static_lethal_positions
        hard_blocked: set[tuple[int, int]] = set(engine.learned_barriers) | set(static_lethal)
        unverified_blocked: set[tuple[int, int]] = set(hard_blocked)
        non_bg_rs, non_bg_cs = np.nonzero(curr_grid != bg)
        for r_np, c_np in zip(non_bg_rs.tolist(), non_bg_cs.tolist(), strict=False):
            if (r_np, c_np) == avatar_pos:
                continue
            val = int(curr_grid[r_np, c_np])
            if val in av_feats:
                continue
            ev_cell = th.evaluate_transition(val, locus=(r_np, c_np))
            if not ev_cell.traversable or ev_cell.risk_cost >= 40.0 or val in lethal_feats:
                hard_blocked.add((r_np, c_np))
                unverified_blocked.add((r_np, c_np))
            elif ev_cell.confidence == 0.0:
                unverified_blocked.add((r_np, c_np))
        for locus_pos in th.transition_kb._locus_beliefs:
            lr_l, lc_l = int(locus_pos[0]), int(locus_pos[1])
            if not (0 <= lr_l < H and 0 <= lc_l < W) or (lr_l, lc_l) == avatar_pos:
                continue
            lev = th.evaluate_transition(int(curr_grid[lr_l, lc_l]), locus=(lr_l, lc_l))
            if not lev.traversable or lev.risk_cost >= 40.0:
                hard_blocked.add((lr_l, lc_l))
                unverified_blocked.add((lr_l, lc_l))
            elif lev.traversable and lev.confidence > 0.0:
                hard_blocked.discard((lr_l, lc_l))
                unverified_blocked.discard((lr_l, lc_l))

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
        wm_priors: dict[int, dict[str, float]] = engine.working_memory.concept_role_priors
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
        wm_priors: dict[int, dict[str, float]] = engine.working_memory.concept_role_priors
        return {
            int(f)
            for f in wm_priors
            if int(f) != bg and int(f) not in av_feats and cls._is_prior_hazard_feat(engine, int(f))
        }

    @classmethod
    def _effective_lethal_features(cls, engine: Any) -> set[int]:
        """Return lethal features combining hazard tracker and TransitionKB fatal beliefs."""
        th = engine.symbolic_theory
        lethal: set[int] = {int(f) for f in engine.hazard_tracker.known_lethal_features}
        for f in th.transition_kb.features_with_outcome(TransitionOutcome.TERMINAL_FAIL):
            if isinstance(f, int) and th.evaluate_transition(f).risk_cost >= 40.0:
                lethal.add(int(f))
        return lethal

    @classmethod
    def compute_frontier_candidates(
        cls,
        engine: Any,
        curr_grid: np.ndarray,
        bg: int,
        av_feats: set[int],
        interactive_cluster_loci: set[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
    ) -> list[tuple[int, int]]:
        """Return reachable unvisited frontier cells ordered by Manhattan distance from avatar."""
        prior_hazard_feats = cls._high_confidence_prior_hazards(engine, bg, av_feats)
        frontiers = GeodesicManifoldRouter.compute_reachable_frontiers(
            engine=engine,
            curr_grid=curr_grid,
            bg=bg,
            prior_hazard_feats=prior_hazard_feats,
            interactive_cluster_loci=interactive_cluster_loci,
            scipy_label_fn=_scipy_label,
        )
        wm = engine.working_memory
        return [
            p
            for p in frontiers
            if not wm.is_exogenous_locus(p, lattice=curr_grid, bg=bg, include_outer_edge=False)
        ]

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
        wm_exog = engine.working_memory
        entities = [
            e
            for e in entities
            if not wm_exog.is_exogenous_locus(
                e.grid_pos, lattice=curr_grid, bg=bg, include_outer_edge=False
            )
        ]
        has_4d_disp = (
            H > 1
            and W > 1
            and any(d.is_displacement_action() for d in engine.action_dynamics.values())
        )
        has_discrete_interact = (len(available_actions) > 4 or not has_4d_disp) and any(
            not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
            for a in available_actions
        )
        prior_hazard_feats = cls._high_confidence_prior_hazards(engine, bg, av_feats)
        lethal_feats = cls._effective_lethal_features(engine)
        if curr_grid.size > 0 and np.issubdtype(curr_grid.dtype, np.integer):
            _amin, _amax = int(curr_grid.min()), int(curr_grid.max())
            if 0 <= _amin and _amax < 256:
                _bc = np.bincount(curr_grid.ravel(), minlength=_amax + 1)
                _nz = np.flatnonzero(_bc)
                feat_counts = {int(f): int(_bc[f]) for f in _nz.tolist()}
            else:
                _uv, _uc = np.unique(curr_grid, return_counts=True)
                feat_counts = {int(f): int(c) for f, c in zip(_uv.tolist(), _uc.tolist())}
        else:
            _uv, _uc = np.unique(curr_grid, return_counts=True)
            feat_counts = {int(f): int(c) for f, c in zip(_uv.tolist(), _uc.tolist())}
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
                or e.feature_id in lethal_feats
                or (
                    e.feature_id in engine.hazard_tracker.mobile_entities
                    and e.feature_id not in engine.symbolic_theory.cargo_features
                    and e.feature_id not in engine.learned_cargo_features
                )
            ):
                continue
            if e.role == EntityRole.AGENT or (avatar_pos is None and e.feature_id in av_feats):
                continue
            # Causal Disambiguation: Pervasive walkable substrates (e.g. tracks, floors)
            # that recur in large quantities cannot be the singular exit/goal.
            feat_total_cells = feat_counts.get(int(e.feature_id), 0)
            feat_instance_count = sum(1 for ent in entities if ent.feature_id == e.feature_id)
            max_entity_area = max(64, int(H * W * 0.15))
            is_pervasive_substrate = (
                e.feature_id in engine.symbolic_theory.walkable_features
            ) and (feat_instance_count > 2 or feat_total_cells > max_entity_area)
            if is_pervasive_substrate:
                engine.symbolic_theory.retract_goal(e.feature_id, reason="pervasive_substrate")
                continue

            is_goal_candidate = (
                e.feature_id in engine.symbolic_theory.goal_features
                and (
                    1 <= e.area <= max_entity_area
                    or not engine.symbolic_theory.is_walkable(e.feature_id)
                )
            ) or (
                e.role == EntityRole.GOAL
                and not has_discrete_interact
                and not (
                    engine.symbolic_theory.is_locus_explored(e.grid_pos)
                    and e.feature_id not in engine.symbolic_theory.goal_features
                )
                and e.grid_pos not in engine.exhausted_candidate_goals
            )
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
                and not engine.symbolic_theory.is_locus_explored(target_coord)
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
                                not engine.symbolic_theory.is_locus_explored(
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
            held_ids = engine.working_memory.held_feature_ids
            lethal_cond_feats = lethal_feats
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
                    if barrier_pos_in_path is not None and blocking_aff is not None:
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
                    if blocking_aff is None:
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
                    context={"context_epoch": get_engine_state_attr(engine, "context_epoch", 0)},
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

        # Non-navigation interactive control clusters filter
        interactive_cluster_loci: set[tuple[int, int]] = set()
        if any(engine.is_spatial_effector(a) for a in available_actions):
            clusters = PerceptionEngine.detect_affordance_clusters(entities, curr_grid, bg=bg)
            for p in clusters:
                for item in p["items"]:
                    r0, r1, c0, c1 = item.bounding_box
                    for rr in range(max(0, r0 - 2), min(H, r1 + 3)):
                        for cc in range(max(0, c0 - 2), min(W, c1 + 3)):
                            interactive_cluster_loci.add((rr, cc))

        # (b) Hypothesized goals from prior context knowledge or role inference
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
                    and e.feature_id not in engine.symbolic_theory.goal_features
                ):
                    continue
                is_cargo_entity = (
                    e.feature_id in engine.symbolic_theory.cargo_features
                    or e.feature_id in engine.learned_cargo_features
                )
                if engine.symbolic_theory.is_locus_explored(e.grid_pos) and (
                    not has_cargo_context or is_cargo_entity
                ):
                    continue
                if e.grid_pos in engine.exhausted_candidate_goals and (
                    not has_cargo_context or is_cargo_entity
                ):
                    continue
                if (
                    e.feature_id in lethal_feats
                    or e.feature_id in engine.mobile_threat_features
                    or (
                        e.feature_id in engine.hazard_tracker.mobile_entities
                        and not is_cargo_entity
                    )
                    or int(e.feature_id) in prior_hazard_feats
                ):
                    continue
                if e.role == EntityRole.AGENT or (avatar_pos is None and e.feature_id in av_feats):
                    continue
                if e.grid_pos in interactive_cluster_loci:
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
        if not goals:
            novel_entity_goals: list[tuple[int, int]] = []
            for e in entities:
                if avatar_pos is not None and e.grid_pos == avatar_pos and not has_cargo_context:
                    continue
                if e.feature_id == bg:
                    continue
                is_cargo_cand = (
                    e.feature_id in engine.symbolic_theory.cargo_features
                    or e.feature_id in engine.learned_cargo_features
                )
                if (
                    e.role == EntityRole.AGENT
                    or (avatar_pos is None and e.feature_id in av_feats)
                    or e.feature_id == engine.working_memory.shroud_feature
                    or (e.feature_id in engine.hazard_tracker.mobile_entities and not is_cargo_cand)
                    or e.grid_pos in interactive_cluster_loci
                ):
                    continue
                is_isolated_interior = (
                    1 <= e.area <= 2
                    and 1 <= e.grid_pos[0] < H - 1
                    and 1 <= e.grid_pos[1] < W - 1
                    and has_discrete_interact
                    and not engine.symbolic_theory.is_locus_explored(e.grid_pos)
                    and e.feature_id not in engine.symbolic_theory.goal_features
                )
                if (
                    (engine.symbolic_theory.is_barrier(e.feature_id) and not is_isolated_interior)
                    or (e.role == EntityRole.OBSTACLE and not is_isolated_interior)
                    or e.feature_id in lethal_feats
                    or e.feature_id in engine.mobile_threat_features
                    or int(e.feature_id) in prior_hazard_feats
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
                            and feat_counts.get(int(e.feature_id), 0) <= 4
                        )
                    )
                    and e.grid_pos not in engine.learned_barriers
                    and e.grid_pos not in engine.current_context_hazards
                )
                if (
                    engine.symbolic_theory.is_feature_explored(e.feature_id)
                    and not is_isolated_interior
                    and not is_potential_cargo_receptor
                ):
                    continue
                if engine.symbolic_theory.is_locus_explored(e.grid_pos) and (
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

        if (
            avatar_pos is not None
            and goals
            and (satisfied_when or (not confirmed_goals and len(goals) > 1))
        ):
            from hbllm.hcir.world.object_state_graph import compute_stance_priority

            th = engine.symbolic_theory
            ledger = engine.object_planner.ledger
            ledger.observe_grid_connectivity(
                curr_grid, avatar_pos, bg, set(th.walkable_features), av_feats
            )
            role_priors: dict[int, dict[str, float]] = engine.working_memory.concept_role_priors
            has_cargo_ctx = bool(th.cargo_features or engine.learned_cargo_features)
            goals, newly_unlocked = cls._filter_reachable_and_unlocked(
                engine, curr_grid, H, W, bg, av_feats, avatar_pos, goals
            )
            if ledger.newly_reachable_cells:
                newly_unlocked = set(newly_unlocked) | set(ledger.newly_reachable_cells)

            def _role_score_for_feature(feat_id: int) -> tuple[float, float, float]:
                """Return (preferred_role_score, barrier_score, hazard_score) in [0, 1]."""
                return cls._role_scores_from_priors(role_priors, feat_id, has_cargo_ctx)

            priorities = {
                g: compute_stance_priority(
                    engine,
                    curr_grid,
                    H,
                    W,
                    bg,
                    avatar_pos,
                    g,
                    satisfied_when,
                    newly_unlocked,
                    _role_score_for_feature,
                )
                for g in goals
            }
            min_tier = min(pr[0] for pr in priorities.values())
            if min_tier <= 3:
                goals = [g for g in goals if priorities[g][0] == min_tier]
            goals.sort(key=lambda g: priorities[g])

        if avatar_pos is not None:
            goals = [
                g
                for g in goals
                if (
                    (g != avatar_pos or (g in satisfied_when and "entity_pos" in satisfied_when[g]))
                    and (g in confirmed_goals or g not in interactive_cluster_loci)
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
                    not engine.symbolic_theory.is_locus_explored(g)
                    or has_cargo_context
                    or g in confirmed_goals
                )
            )
            or (
                g in satisfied_when
                and "entity_pos" in satisfied_when[g]
                and not engine.symbolic_theory.is_locus_explored(satisfied_when[g]["entity_pos"])
                and (not engine.symbolic_theory.is_locus_explored(g) or g == avatar_pos)
            )
        ]
        goals = unexhausted_goals

        # Tier 3. Frontier Exploration: Nearest reachable boundary between visited and unvisited
        # (Activated when confirmed goals and novel entities are either exhausted or absent)
        if not goals:
            frontier_candidates = cls.compute_frontier_candidates(
                engine=engine,
                curr_grid=curr_grid,
                bg=bg,
                av_feats=av_feats,
                interactive_cluster_loci=interactive_cluster_loci,
            )
            if frontier_candidates:
                nearest_frontier = frontier_candidates[0]
                goals.append(nearest_frontier)
                satisfied_when[nearest_frontier] = {"max_visits": 1}

        # Frontier Exhaustion Epistemic Contradiction:
        # If no reachable targets remain while unverified prior-context blacklists
        # (known_lethal_features from earlier levels) hide interior entities on the grid,
        # invalidate the stale prior labels so exploration can proceed.
        if (
            not goals
            and avatar_pos is not None
            and engine.symbolic_theory.has_prior_context_transition
            and engine.working_memory.consecutive_simulation_failures >= 2
        ):
            th = engine.symbolic_theory
            verified_lethal = th.context_verified_features("lethal")
            unverified_lethals = set(lethal_feats) - verified_lethal
            if unverified_lethals and any(f in unverified_lethals for f in feat_counts):
                from hbllm.hcir.world.cortex_causal import invalidate_stale_context_ontology

                invalidate_stale_context_ontology(
                    th,
                    hazard_tracker=engine.hazard_tracker,
                    episodic_cortex=engine.episodic_cortex,
                    working_memory=engine.working_memory,
                    engine_or_view=engine,
                    bg_feature=bg,
                    preserve_current_context=True,
                    reason="frontier_exhausted_over_prior_blacklist",
                )
                lethal_feats = cls._effective_lethal_features(engine)
                for e_blk in entities:
                    if (
                        e_blk.feature_id != bg
                        and e_blk.feature_id not in av_feats
                        and e_blk.grid_pos != avatar_pos
                        and not th.is_barrier(e_blk.feature_id)
                        and e_blk.feature_id not in lethal_feats
                        and e_blk.grid_pos not in engine.learned_barriers
                        and e_blk.grid_pos not in engine.current_context_hazards
                    ):
                        goals.append(e_blk.grid_pos)
                        satisfied_when[e_blk.grid_pos] = {"max_visits": 2}

        # If both entities AND frontiers are completely exhausted:
        if not goals and engine.exhausted_candidate_goals:
            clear_exhausted = True

        # Exclude all known static lethal and exogenous coordinates from candidate attractors
        all_lethals = engine.current_context_hazards
        goals = [
            g
            for g in goals
            if g not in all_lethals
            and int(curr_grid[g[0], g[1]]) not in lethal_feats
            and not wm_exog.is_exogenous_locus(
                g, lattice=curr_grid, bg=bg, include_outer_edge=False
            )
        ]
        new_learned_positions = {
            p
            for p in new_learned_positions
            if p not in all_lethals
            and int(curr_grid[p[0], p[1]]) not in lethal_feats
            and not wm_exog.is_exogenous_locus(
                p, lattice=curr_grid, bg=bg, include_outer_edge=False
            )
        }

        # Reachability Gating: Discard goals physically disconnected from avatar
        if avatar_pos is not None and goals:
            barrier_feats = engine.symbolic_theory.barrier_features
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
            for locus_pos in engine.symbolic_theory.transition_kb._locus_beliefs:
                lr_l, lc_l = int(locus_pos[0]), int(locus_pos[1])
                if not (0 <= lr_l < H and 0 <= lc_l < W):
                    continue
                l_ev = engine.symbolic_theory.evaluate_transition(
                    int(curr_grid[lr_l, lc_l]), locus=(lr_l, lc_l)
                )
                if not l_ev.traversable or l_ev.risk_cost >= 40.0:
                    blocked[lr_l, lc_l] = True
                elif l_ev.traversable and l_ev.confidence > 0.0:
                    blocked[lr_l, lc_l] = False
            passable = ~blocked
            passable[avatar_pos[0], avatar_pos[1]] = True
            labeled_reach, _ = _scipy_label(passable)
            av_lbl = int(labeled_reach[avatar_pos[0], avatar_pos[1]])

            reach_offsets = (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                *engine.action_deltas.values(),
            )
            reachable_goals = [
                g
                for g in goals
                if int(labeled_reach[g[0], g[1]]) == av_lbl
                or any(
                    0 <= g[0] + dr < H
                    and 0 <= g[1] + dc < W
                    and int(labeled_reach[g[0] + dr, g[1] + dc]) == av_lbl
                    for dr, dc in reach_offsets
                )
            ]
            if reachable_goals:
                goals = reachable_goals

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
