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

import heapq
import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_perception import OrientedThreat, PerceptionEngine
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.frontopolar_subgoal_stack import FrontopolarSubgoal, SubgoalType
from hbllm.hcir.world.prefrontal_working_memory import HeldItem
from hbllm.hcir.world.topological_graph_induction import GeodesicManifoldRouter

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

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
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"sim_step_{self.action}_{self.predicted_avatar_pos[0]}_{self.predicted_avatar_pos[1]}",
                node_data={
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

        # 1. Confirmed goals — causal evidence from past wins
        confirmed_goals: list[tuple[int, int]] = []
        cargos_known_goals = engine.symbolic_theory.cargo_features | getattr(
            engine, "learned_cargo_features", set()
        )
        for e in entities:
            if avatar_pos is not None and e.grid_pos == avatar_pos:
                continue
            if e.feature_id == bg:
                continue
            if e.role == EntityRole.OBSTACLE or engine.symbolic_theory.is_barrier(e.feature_id):
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
            is_pervasive_substrate = (
                e.feature_id in engine.symbolic_theory.walkable_features
            ) and (feat_instance_count > 2 or feat_total_cells > 64)
            if is_pervasive_substrate:
                engine.symbolic_theory.goal_features.discard(e.feature_id)
                continue

            is_goal_candidate = (
                e.feature_id in engine.symbolic_theory.goal_features
                and e.feature_id not in cargos_known_goals
                and not engine.symbolic_theory.has_affordance(e, "can_pickup")
                and (1 <= e.area <= 64 or not engine.symbolic_theory.is_walkable(e.feature_id))
            ) or (
                e.role == EntityRole.GOAL
                and e.feature_id not in cargos_known_goals
                and not engine.symbolic_theory.has_affordance(e, "can_pickup")
            )
            if is_goal_candidate:
                if e.grid_pos not in confirmed_goals:
                    confirmed_goals.append(e.grid_pos)
                    if e.feature_id in engine.symbolic_theory.goal_features:
                        new_learned_positions.add(e.grid_pos)

        # Object Permanence: Retain known goal positions if currently occluded by mobile entities
        for gp in engine.learned_goal_positions:
            if 0 <= gp[0] < H and 0 <= gp[1] < W and gp != avatar_pos:
                if gp not in confirmed_goals:
                    confirmed_goals.append(gp)

        goals: list[tuple[int, int]] = []
        if confirmed_goals:
            goals = list(confirmed_goals)
        # Faculty: Prefrontal Working Memory Active Schema & Procedural Skill Prioritization (dlPFC / Habitual Replay)
        wm = getattr(engine, "working_memory", None)
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
                        if 0 <= sp[0] < H and 0 <= sp[1] < W:
                            if sp in goals:
                                goals.remove(sp)
                            goals.insert(0, sp)
            elif wm.active_schema is None and wm.procedural_library:
                matched = wm.find_matching_skill(available_actions=available_actions)
                if matched is not None:
                    skill_name, skill = matched
                    grounded_targets: dict[str, set[tuple[int, int]]] = {}
                    for sg in getattr(getattr(skill, "program", None), "subgoals", []):
                        q = getattr(sg, "target_query", {})
                        st_id = q.get("stage_id", "")
                        role = q.get("role", "")
                        req_f = q.get("required_feature")
                        if role in ("tool", "key") or req_f is not None:
                            tool_cells = set()
                            for e in entities:
                                if (req_f is not None and e.feature_id == req_f) or (
                                    req_f is None and e.role in ("tool", "key")
                                ):
                                    tool_cells.add(e.grid_pos)
                            if tool_cells:
                                grounded_targets[st_id] = tool_cells
                        elif role == "barrier":
                            if engine.learned_barriers:
                                grounded_targets[st_id] = set(engine.learned_barriers)
                        elif role == "goal":
                            if goals:
                                grounded_targets[st_id] = set(goals)
                    schema = wm.instantiate_skill_as_schema(
                        skill_name, target_positions_override=grounded_targets
                    )
                    if schema and schema.current_stage and schema.current_stage.target_positions:
                        for sp in reversed(list(schema.current_stage.target_positions)):
                            if 0 <= sp[0] < H and 0 <= sp[1] < W:
                                if sp in goals:
                                    goals.remove(sp)
                                goals.insert(0, sp)

        # Faculty: Brodmann Area 10 (aPFC) Frontopolar Subgoal Prioritization
        if engine.subgoal_stack.has_pending_goals:
            cur_sg = engine.subgoal_stack.current_subgoal
            if (
                cur_sg is not None
                and cur_sg.target_destination != avatar_pos
                and not engine.subgoal_stack.is_subgoal_satisfied(
                    cur_sg, set(), avatar_pos=avatar_pos
                )
            ):
                sg_dest = cur_sg.target_destination
                if 0 <= sg_dest[0] < H and 0 <= sg_dest[1] < W:
                    if sg_dest in goals:
                        goals.remove(sg_dest)
                    goals.insert(0, sg_dest)
            elif cur_sg is not None and cur_sg.target_destination == avatar_pos:
                subgoal_to_pop = True

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
                if avatar_pos is not None and e.grid_pos == avatar_pos:
                    continue
                if e.feature_id == bg:
                    continue
                if engine.symbolic_theory.is_barrier(e.feature_id) or e.role == EntityRole.OBSTACLE:
                    continue
                if e.feature_id in cargos_known_goals or engine.symbolic_theory.has_affordance(
                    e, "can_pickup"
                ):
                    continue
                if (
                    e.feature_id in engine.hazard_tracker.known_lethal_features
                    or e.feature_id in engine.mobile_threat_features
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
                if (
                    e.feature_id in engine.symbolic_theory.goal_features
                    and e.feature_id != 0
                    and (1 <= e.area <= 64 or not engine.symbolic_theory.is_walkable(e.feature_id))
                ) or (
                    e.feature_id in engine.symbolic_theory.candidate_goal_features
                    and (1 <= e.area <= 64 or not engine.symbolic_theory.is_walkable(e.feature_id))
                ):
                    if e.grid_pos not in hypothesized_goals:
                        hypothesized_goals.append(e.grid_pos)
                elif (
                    e.role == target_role
                    or e.feature_id in engine.learned_receptacle_features
                    or e.grid_pos in engine.learned_goal_positions
                    or e.grid_pos in engine.learned_receptacle_positions
                ):
                    if e.grid_pos not in hypothesized_goals:
                        hypothesized_goals.append(e.grid_pos)
            if hypothesized_goals:
                goals = hypothesized_goals

        # (c) Structural inference: geometric patterns suggesting goals
        if not goals and engine.learned_goal_positions:
            goals = [
                g
                for g in engine.learned_goal_positions
                if 0 <= g[0] < H and 0 <= g[1] < W and g not in panel_coords
            ]
        if not goals:
            structural_goals = engine.detect_structural_goals(curr_grid, bg=bg)
            if hasattr(engine.working_memory, "structural_goals"):
                engine.working_memory.structural_goals = structural_goals
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
                    "target_constellation",
                    "receptacle_dock",
                    "symmetry_completion",
                )
            ]
            matched_constellations = [
                sg
                for sg in structural_goals
                if sg.get("type") == "target_constellation"
                and (sg.get("feature") in av_feats if av_feats else True)
            ]
            matched_docking = (
                matched_constellations
                if matched_constellations
                else [
                    sg
                    for sg in structural_goals
                    if sg.get("type") in ("target_constellation", "receptacle_dock")
                    and (sg.get("feature") in av_feats if av_feats else True)
                ]
            )
            cand_sgs = (
                matched_docking
                if matched_docking
                else (primary_sgs if primary_sgs else structural_goals)
            )
            cand_sgs = sorted(cand_sgs, key=lambda s: s.get("confidence", 0.5), reverse=True)
            for sg in cand_sgs:
                feat = sg.get("feature")
                sg_type = sg.get("type", "")
                is_docking = sg_type in ("receptacle_dock", "target_constellation")
                if feat is not None:
                    if not is_docking and feat in av_feats and not sg.get("is_midpoint"):
                        continue
                    if (
                        feat in engine.hazard_tracker.known_lethal_features
                        or engine.symbolic_theory.is_barrier(feat)
                    ):
                        continue
                pos = sg.get("position")
                if (
                    pos
                    and 0 <= pos[0] < H
                    and 0 <= pos[1] < W
                    and pos != avatar_pos
                    and pos not in panel_coords
                ):
                    if pos not in goals:
                        goals.append(pos)
                cells = sg.get("cells")
                if cells and sg_type != "target_constellation":
                    for cp in cells:
                        if (
                            0 <= cp[0] < H
                            and 0 <= cp[1] < W
                            and cp != avatar_pos
                            and cp not in panel_coords
                        ):
                            if cp not in goals:
                                goals.append(cp)

        # (d) Exploratory guesses: unknown entities
        if not goals:
            exploratory_goals = [
                e.grid_pos
                for e in entities
                if e.grid_pos not in panel_coords
                and e.role
                in (
                    EntityRole.UNKNOWN,
                    EntityRole.GOAL,
                    EntityRole.PORTAL,
                    EntityRole.RECEPTACLE,
                    EntityRole.AGENT,
                    EntityRole.ACTUATOR,
                    EntityRole.MANIPULABLE,
                    EntityRole.RESOURCE,
                )
                and 1 <= e.area <= 64
                and e.feature_id != bg
                and (
                    e.role
                    in (
                        EntityRole.GOAL,
                        EntityRole.PORTAL,
                        EntityRole.RECEPTACLE,
                        EntityRole.RESOURCE,
                    )
                    or not engine.symbolic_theory.is_walkable(e.feature_id)
                )
                and e.feature_id not in engine.hazard_tracker.known_lethal_features
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and (
                    e.feature_id not in av_feats
                    or (
                        avatar_pos is not None
                        and (
                            abs(e.grid_pos[0] - avatar_pos[0]) + abs(e.grid_pos[1] - avatar_pos[1])
                        )
                        > 4
                    )
                )
            ]
            if exploratory_goals:
                goals = exploratory_goals
                for e in entities:
                    if (
                        e.grid_pos in goals
                        and e.feature_id not in av_feats
                        and e.grid_pos not in panel_coords
                        and e.feature_id not in engine.mobile_threat_features
                        and e.feature_id not in engine.hazard_tracker.known_lethal_features
                    ):
                        new_candidate_feats.add(e.feature_id)

        if avatar_pos is not None:
            goals = [
                g
                for g in goals
                if g != avatar_pos and (g in confirmed_goals or g not in panel_coords)
            ]

        unexhausted_goals = [g for g in goals if g not in engine.exhausted_candidate_goals]
        if unexhausted_goals:
            goals = unexhausted_goals
        elif goals:
            clear_exhausted = True

        # Exclude all known static lethal coordinates from candidate attractors
        all_lethals = engine.current_level_lethal_positions
        goals = [g for g in goals if g not in all_lethals]

        # Extended avatar projection
        goals = cls._project_extended_avatar_goals(curr_grid, engine, goals, H, W)

        return EpistemicGoalProposal(
            goals=goals,
            confirmed_goals=confirmed_goals,
            new_learned_goal_positions=new_learned_positions,
            new_candidate_features=new_candidate_feats,
            subgoal_to_pop=subgoal_to_pop,
            clear_exhausted_goals=clear_exhausted,
        )

    @classmethod
    def _project_extended_avatar_goals(
        cls,
        curr_grid: np.ndarray,
        engine: AutonomousEpistemicEngine,
        goals: list[tuple[int, int]],
        H: int,
        W: int,
    ) -> list[tuple[int, int]]:
        if not goals or engine.avatar_pos is None or engine.avatar_feature is None:
            return goals
        av_pts = np.argwhere(curr_grid == engine.avatar_feature)
        if len(av_pts) < 15:
            return goals
        min_ar, min_ac = av_pts.min(axis=0)
        max_ar, max_ac = av_pts.max(axis=0)
        h_span = max_ar - min_ar + 1
        w_span = max_ac - min_ac + 1
        is_extended_vert = (h_span >= 20) and (w_span <= 5)
        is_extended_horiz = (w_span >= 20) and (h_span <= 5)
        if is_extended_vert:
            vert_goals = []
            for g in goals:
                proj = (engine.avatar_pos[0], g[1])
                if proj != engine.avatar_pos and 0 <= proj[1] < W and proj not in vert_goals:
                    vert_goals.append(proj)
            if vert_goals:
                return vert_goals
        elif is_extended_horiz:
            horiz_goals = []
            for g in goals:
                proj = (g[0], engine.avatar_pos[1])
                if proj != engine.avatar_pos and 0 <= proj[0] < H and proj not in horiz_goals:
                    horiz_goals.append(proj)
            if horiz_goals:
                return horiz_goals
        return goals

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export goal scheduler capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="epistemic_goal_scheduler",
            operator="==",
            value="ACTIVE",
            relation="ofc_ba10_prioritization",
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


class MentalSimulationPlanner:
    """Simulates candidate action sequences internally in imagination using learned world theory."""

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export mental simulation planner capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="mental_simulation_planner",
            operator="==",
            value="ACTIVE",
            relation="internal_counterfactual_rollout",
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
    ) -> tuple[set[tuple[int, int]], bool]:
        """Compute static physical barriers and void background boundaries."""
        H, W = curr_grid.shape
        static_barriers: set[tuple[int, int]] = set()

        for r, c in engine.learned_barriers:
            if 0 <= r < H and 0 <= c < W:
                val = int(curr_grid[r, c])
                if (r, c) in engine.hazard_tracker.static_lethal_positions or (
                    val != bg and not engine.symbolic_theory.is_walkable(val)
                ):
                    static_barriers.add((r, c))

        u_vals, u_counts = np.unique(curr_grid, return_counts=True)
        feat_counts = dict(zip(u_vals, u_counts))
        for r in range(H):
            for c in range(W):
                val = int(curr_grid[r, c])
                if engine.symbolic_theory.is_barrier(val) or (
                    val in engine.hazard_tracker.known_lethal_features
                    and feat_counts.get(val, 0) >= 25
                ):
                    static_barriers.add((r, c))

        # General Intelligence & Physics Grounding:
        # Physical surfaces (roads, tracks, rails, corridors, bridges) can have narrow elongated
        # geometries in 2D lattices or 3D LiDAR/occupancy scans. Empirical traversal evidence
        # strictly supersedes geometric bounding-box priors.
        for feat in np.unique(curr_grid):
            feat_int = int(feat)
            if (
                feat_int == bg
                or feat_int in av_feats
                or feat_int in engine.symbolic_theory.walkable_features
            ):
                continue
            if engine.avatar_pos is not None:
                ar, ac = engine.avatar_pos
                if 0 <= ar < H and 0 <= ac < W and int(curr_grid[ar, ac]) == feat_int:
                    continue
            pts = np.argwhere(curr_grid == feat_int)
            if 15 <= len(pts) <= H * W * 0.15:
                min_r, min_c = pts.min(axis=0)
                max_r, max_c = pts.max(axis=0)
                h_span = max_r - min_r + 1
                w_span = max_c - min_c + 1
                if (w_span <= 3 and h_span >= 20 and len(pts) >= h_span * 0.7) or (
                    h_span <= 3 and w_span >= 20 and len(pts) >= w_span * 0.7
                ):
                    for pr, pc in pts:
                        static_barriers.add((int(pr), int(pc)))

        # Explicitly ground all static lethal positions from hazard tracker and level memory
        for r, c in engine.current_level_lethal_positions:
            if 0 <= r < H and 0 <= c < W:
                static_barriers.add((r, c))

        # Corridor Manifold Continuity
        walkable_and_safe = engine.symbolic_theory.walkable_features
        other_walkables = {f for f in walkable_and_safe if f != bg and f != 0}
        bg_is_impassable_void = len(other_walkables) > 0 and bg not in walkable_and_safe
        if not bg_is_impassable_void:
            bg_count = int(feat_counts.get(bg, 0))
            bg_ratio = bg_count / max(1, H * W)
            ar, ac = (
                (engine.avatar_pos[0], engine.avatar_pos[1])
                if (
                    engine.avatar_pos
                    and 0 <= engine.avatar_pos[0] < H
                    and 0 <= engine.avatar_pos[1] < W
                )
                else (-1, -1)
            )
            stride = max(1, getattr(engine, "lattice_stride", 1) or 1)
            av_span = max(stride + 1, 4)
            avatar_on_bg = ar >= 0 and (
                int(curr_grid[ar, ac]) == bg
                or any(
                    0 <= ar + dr_a < H
                    and 0 <= ac + dc_a < W
                    and int(curr_grid[ar + dr_a, ac + dc_a]) == bg
                    for dr_a, dc_a in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-av_span, 0),
                        (av_span, 0),
                        (0, -av_span),
                        (0, av_span),
                    ]
                )
            )
            if bg_ratio >= 0.20 and not avatar_on_bg:
                bg_is_impassable_void = True

        if bg_is_impassable_void:
            for r in range(H):
                for c in range(W):
                    if int(curr_grid[r, c]) == bg and (r, c) not in goals:
                        static_barriers.add((r, c))

        # Enforce HCIR Attractor Invariant: Non-lethal goals must never be static barriers
        all_lethals = engine.current_level_lethal_positions
        safe_goals = set(goals) - all_lethals
        static_barriers.difference_update(safe_goals)
        goal_features = (
            engine.symbolic_theory.goal_features | engine.symbolic_theory.candidate_goal_features
        )
        for gf in goal_features:
            if gf != bg and gf not in av_feats:
                g_pts = np.argwhere(curr_grid == gf)
                for gr_idx, gc_idx in g_pts:
                    static_barriers.discard((int(gr_idx), int(gc_idx)))
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
        """Extract state mutation triggers (switches that toggle doors) and remote causal triggers."""
        H, W = curr_grid.shape
        mutation_triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}

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

        # vmPFC Remote Causal Triggers
        for trig_pos, aff_list in engine.working_memory.remote_causal.trigger_registry.items():
            for aff in aff_list:
                if (
                    aff.confidence >= 0.70
                    and 0 <= aff.remote_pos[0] < H
                    and 0 <= aff.remote_pos[1] < W
                ):
                    mutation_triggers.setdefault(trig_pos, set()).add(aff.remote_pos)

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
    def simulate_in_mind(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        target_role: EntityRole = EntityRole.GOAL,
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> list[MentalSimulationStep] | None:
        """Simulate candidate action sequences internally in memory without taking physical steps."""
        curr_grid = np.asarray(curr_grid)
        if curr_grid.ndim == 3:
            if curr_grid.shape[0] < min(curr_grid.shape[1], curr_grid.shape[2]):
                curr_grid = curr_grid[-1]
            elif curr_grid.shape[2] in (1, 3, 4):
                curr_grid = curr_grid[:, :, 0]
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
            active_manip = engine.working_memory.body_schema.active_manipulandum
            if active_manip is not None and getattr(active_manip, "role", "") != "cargo":
                engine.avatar_pos = active_manip.pos

        if not engine.is_motor_grounded() or engine.avatar_pos is None:
            return None

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
            engine.symbolic_theory.candidate_goal_features.add(feat)
        if proposal.subgoal_to_pop:
            engine.subgoal_stack.pop_subgoal()
        if proposal.clear_exhausted_goals:
            engine.exhausted_candidate_goals.clear()
        engine.symbolic_theory.goal_features.difference_update(av_feats)

        goals = proposal.goals

        # Faculty D: Bilateral Convergent Coordinate Frames
        cargos_known_d = engine.symbolic_theory.cargo_features | getattr(
            engine, "learned_cargo_features", set()
        )
        recs_known_d = engine.symbolic_theory.receptacle_features
        is_bilat_valid = (
            engine.bilateral_state is not None
            and len(engine.action_dynamics) >= 2
            and not engine.working_memory.body_schema.has_manipulandum()
            and not bool(engine.bilateral_state.features & cargos_known_d)
            and not bool(engine.bilateral_state.features & recs_known_d)
            and not (cargos_known_d and (recs_known_d or engine.learned_goal_positions))
            and (
                abs(engine.bilateral_state.pos1[0] - engine.bilateral_state.pos2[0])
                + abs(engine.bilateral_state.pos1[1] - engine.bilateral_state.pos2[1])
                > 6
            )
        )
        if is_bilat_valid:
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

        has_receptacle = bool(
            engine.learned_goal_positions
            or getattr(engine, "learned_receptacle_positions", None)
            or any(
                engine.symbolic_theory.has_affordance(e, "receptacle")
                or getattr(e, "role", "") in (EntityRole.RECEPTACLE, EntityRole.GOAL)
                for e in entities
            )
        )
        if not goals and not has_receptacle:
            return None

        # 2. Identify candidate pushable blocks / cargo
        pushable_blocks: list[tuple[int, int]] = [
            e.grid_pos
            for e in entities
            if e.feature_id not in av_feats
            and e.feature_id != bg
            and e.grid_pos not in goals
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
        if engine.working_memory.body_schema.has_manipulandum():
            active_manip = engine.working_memory.body_schema.active_manipulandum
            if (
                active_manip is not None
                and getattr(active_manip, "role", "") in ("cargo", "manipulandum")
                and active_manip.pos not in pushable_blocks
                and active_manip.pos not in goals
            ):
                pushable_blocks.append(active_manip.pos)

        unattached_cargo = [
            e
            for e in entities
            if (
                engine.symbolic_theory.is_cargo(e)
                or e.feature_id in engine.learned_cargo_features
                or (
                    e.role == EntityRole.MANIPULABLE
                    and 1 <= e.area <= 64
                    and e.feature_id not in engine.mobile_threat_features
                )
            )
            and e.feature_id not in av_feats
            and e.feature_id != bg
            and e.grid_pos not in goals
            and e.grid_pos not in engine.learned_goal_positions
            and e.grid_pos not in engine.learned_receptacle_positions
        ]

        # 3. Static barriers & corridor continuity
        static_barriers, bg_is_impassable_void = MentalSimulationPlanner._compute_static_barriers(
            engine=engine,
            curr_grid=curr_grid,
            bg=bg,
            av_feats=av_feats,
            goals=goals,
            blocked_cells=blocked_cells,
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
        enclave_border_feats: set[int] = set()
        if hasattr(engine, "working_memory") and hasattr(engine.working_memory, "structural_goals"):
            for sg in engine.working_memory.structural_goals:
                for bf in sg.get("border_features", []):
                    enclave_border_feats.add(bf)
        for e in entities:
            if (
                e.feature_id != bg
                and e.feature_id != 0
                and e.feature_id not in av_feats
                and e.feature_id not in enclave_border_feats
                and e.role not in (EntityRole.MANIPULABLE, EntityRole.GOAL, EntityRole.RECEPTACLE)
                and not engine.symbolic_theory.is_walkable(e.feature_id)
                and not engine.symbolic_theory.is_goal(e)
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and not engine.symbolic_theory.is_cargo(e)
                and e.feature_id not in engine.learned_cargo_features
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
        cargos_known = engine.symbolic_theory.cargo_features | engine.learned_cargo_features
        oriented_threats = PerceptionEngine.detect_oriented_threats(
            curr_grid,
            entities,
            bg=bg,
            step_size=step_size,
            avatar_pos=engine.avatar_pos,
            avatar_features=av_feats,
            goals=set(goals),
            cargo_features=cargos_known,
            barrier_features=engine.symbolic_theory.barrier_features,
            walkable_features=engine.symbolic_theory.walkable_features,
        )

        half_step = max(1, step_size // 2)

        _barrier_feats = engine.symbolic_theory.barrier_features
        _grid_walls: set[tuple[int, int]] = set(static_barriers)
        for rr in range(H):
            for cc in range(W):
                cell_val = int(curr_grid[rr, cc])
                if cell_val in _barrier_feats or (
                    bg_is_impassable_void and cell_val == bg and (rr, cc) not in goals
                ):
                    _grid_walls.add((rr, cc))

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
                        if int(curr_grid[rr_t, cc_t]) == t.feature_id:
                            threat_cells.add((rr_t, cc_t))
                            threat_map[(rr_t, cc_t)] = t

        init_threats = frozenset(threat_cells)
        patrol_items: set[tuple[tuple[int, int], tuple[int, int]]] = {
            (t.pos, t.facing)
            for t in oriented_threats
            if t.feature_id in engine.mobile_threat_features and t.facing != (0, 0)
        }
        for proj in engine.collision_cones.projected_trajectories:
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
        ) -> frozenset[tuple[tuple[int, int], tuple[int, int]]] | None:
            nxt: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for p, f in patrols:
                if p == avatar_new:
                    return None
                dest = (p[0] + f[0] * step_size, p[1] + f[1] * step_size)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W) or dest in _grid_walls:
                    # Check for 90-degree rotational turn at corners before reversing
                    turned = False
                    for rot_f in [(f[1], -f[0]), (-f[1], f[0])]:
                        rot_dest = (p[0] + rot_f[0] * step_size, p[1] + rot_f[1] * step_size)
                        if (
                            0 <= rot_dest[0] < H
                            and 0 <= rot_dest[1] < W
                            and rot_dest not in _grid_walls
                        ):
                            nxt.add((rot_dest, rot_f))
                            turned = True
                            break
                    if not turned:
                        new_f = (-f[0], -f[1])
                        nxt.add((p, new_f))
                    continue
                if dest == avatar_new:
                    return None
                new_f = f
                if _corridor_blocked(dest, f):
                    turned = False
                    for rot_f in [(f[1], -f[0]), (-f[1], f[0])]:
                        rot_dest = (dest[0] + rot_f[0] * step_size, dest[1] + rot_f[1] * step_size)
                        if (
                            0 <= rot_dest[0] < H
                            and 0 <= rot_dest[1] < W
                            and rot_dest not in _grid_walls
                        ):
                            new_f = rot_f
                            turned = True
                            break
                    if not turned:
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

        is_block_delivery = bool(
            1 <= len(pushable_blocks) <= 16
            and any(
                (
                    engine.symbolic_theory.is_cargo(e)
                    or e.feature_id in engine.learned_cargo_features
                    or getattr(e, "role", None) == EntityRole.MANIPULABLE
                )
                and e.feature_id not in av_feats
                and e.grid_pos in pushable_blocks
                for e in entities
            )
        )

        oriented_threat_positions = {t.pos for t in oriented_threats}
        bilat_feats = engine.bilateral_state.features if engine.bilateral_state else set()
        init_chasers: frozenset[tuple[int, int]] = frozenset(
            e.grid_pos
            for e in entities
            if (
                e.feature_id in engine.mobile_threat_features
                or (
                    e.role == EntityRole.AGENT
                    and e.grid_pos != start_pos
                    and e.feature_id not in av_feats
                    and e.feature_id not in bilat_feats
                    and e.feature_id != bg
                    and (
                        e.feature_id in engine.hazard_tracker.known_lethal_features
                        or any(
                            proj.feature_id == e.feature_id
                            for proj in engine.collision_cones.projected_trajectories
                        )
                    )
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

        is_cargo_attached = False
        attached_cargo_offset: tuple[int, int] | None = None
        if engine.working_memory.body_schema.has_manipulandum():
            active_manip = engine.working_memory.body_schema.active_manipulandum
            if (
                active_manip is not None
                and getattr(active_manip, "role", "") in ("cargo", "manipulandum")
                and start_pos is not None
            ):
                dr_m = active_manip.pos[0] - start_pos[0]
                dc_m = active_manip.pos[1] - start_pos[1]
                stride = max(1, getattr(engine, "lattice_stride", 1) or 1)
                if abs(dr_m) + abs(dc_m) <= max(2, stride * 2):
                    is_cargo_attached = True
                    attached_cargo_offset = (dr_m, dc_m)

        has_affordance_manip = bool(
            is_cargo_attached
            or unattached_cargo
            or any(
                engine.symbolic_theory.has_affordance(e, "can_pickup")
                or e.feature_id in engine.symbolic_theory.cargo_features
                or getattr(e, "role", "") == EntityRole.MANIPULABLE
                for e in entities
                if e.feature_id not in av_feats and e.feature_id != bg and e.feature_id != 0
            )
        )

        cargos_known_temp = engine.symbolic_theory.cargo_features | getattr(
            engine, "learned_cargo_features", set()
        )
        has_periodic = (
            bool(engine.hazard_tracker.periodic_cells)
            and engine.hazard_tracker.environmental_period >= 2
            and any(
                not (p.hazardous_values <= (cargos_known_temp | av_feats | {bg, 0}))
                for p in engine.hazard_tracker.periodic_cells.values()
            )
        )
        has_temporal = (
            bool(init_patrols)
            or bool(init_chasers)
            or has_periodic
            or any(
                (
                    proj.feature_id in engine.mobile_threat_features
                    or proj.feature_id in engine.hazard_tracker.known_lethal_features
                )
                and proj.feature_id not in cargos_known_temp
                and proj.feature_id not in av_feats
                for proj in engine.collision_cones.projected_trajectories
            )
        )

        # ─────────────────────────────────────────────────────────────────────
        # Affordance-Driven Subgoal Sequencing (Brodmann Area 10 Apex Executive)
        # When objects with CAN_PICKUP / CARGO affordances and a RECEPTACLE exist,
        # or when an object is currently incorporated in the ExtendedBodySchema,
        # formulate clean sequential subgoals:
        #   [Navigate to Cargo -> Attach -> Compound Navigate to Receptacle -> Detach]
        # Bypasses heuristic space-time search with deterministic modular geodesic routing.
        # ─────────────────────────────────────────────────────────────────────
        if has_affordance_manip and start_pos is not None:
            dyn_obs = oriented_threat_positions | set(init_chasers) | {p[0] for p in init_patrols}
            affordance_plan = MentalSimulationPlanner._plan_affordance_subgoals(
                engine=engine,
                curr_grid=curr_grid,
                entities=entities,
                bg=bg,
                av_feats=av_feats,
                available_actions=available_actions,
                movable_actions=movable_actions,
                static_barriers=static_barriers,
                goals=goals,
                dynamic_obstacles=dyn_obs,
            )
            if affordance_plan:
                return affordance_plan

        # ─────────────────────────────────────────────────────────────────────
        # Fast Topological Geodesic & Kinematics Router (Substrate Layer)
        # For static navigation tasks (no temporal threats),
        # route deterministic shortest action paths over traversable manifolds.
        # When temporal threats (patrols, chasers) exist, defer to Space-Time A*.
        # ─────────────────────────────────────────────────────────────────────
        if goals and not has_affordance_manip and not has_temporal and start_pos is not None:
            geodesic_movable_actions = [
                (act, dr, dc)
                for act, dr, dc in movable_actions
                if (start_pos, act) not in engine.failed_transitions
                and not engine.working_memory.habenular_ior.is_action_inhibited(start_pos, act)
                and engine.inhibited_actions.get(act, 0) == 0
            ]
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
                    if (
                        (curr_p, act) in engine.failed_transitions
                        or engine.working_memory.habenular_ior.is_action_inhibited(curr_p, act)
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
        # Gather all physical cells of confirmed/candidate goal entities
        all_goal_cells: set[tuple[int, int]] = set(goals)
        goal_features = (
            engine.symbolic_theory.goal_features | engine.symbolic_theory.candidate_goal_features
        )
        for gf in goal_features:
            if gf != bg and gf not in av_feats:
                for gr_idx, gc_idx in np.argwhere(curr_grid == gf):
                    all_goal_cells.add((int(gr_idx), int(gc_idx)))
        for e in entities:
            if (
                engine.symbolic_theory.is_goal(e)
                or e.role == EntityRole.GOAL
                or e.feature_id in goal_features
            ):
                for cr, cc in e.properties.get("cells", [e.grid_pos]):
                    all_goal_cells.add((int(cr), int(cc)))

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

        def heuristic(
            pos: tuple[int, int], blocks: frozenset[tuple[int, int]] | None = None
        ) -> float:
            if is_block_delivery and blocks:
                b_list = list(blocks)
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
        cargos_known_temp2 = engine.symbolic_theory.cargo_features | getattr(
            engine, "learned_cargo_features", set()
        )
        has_periodic = (
            bool(engine.hazard_tracker.periodic_cells)
            and engine.hazard_tracker.environmental_period >= 2
            and any(
                not (p.hazardous_values <= (cargos_known_temp2 | av_feats | {bg, 0}))
                for p in engine.hazard_tracker.periodic_cells.values()
            )
        )
        has_temporal = (
            bool(init_patrols)
            or bool(init_chasers)
            or has_periodic
            or any(
                (
                    proj.feature_id in engine.mobile_threat_features
                    or proj.feature_id in engine.hazard_tracker.known_lethal_features
                )
                and proj.feature_id not in cargos_known_temp2
                and proj.feature_id not in av_feats
                for proj in engine.collision_cones.projected_trajectories
            )
        )
        max_expansions_budget = engine.prediction_budget.compute_search_budget(
            has_temporal=has_temporal,
            has_threats=bool(init_threats),
            consecutive_failures=engine.consecutive_simulation_failures,
            is_under_strict_budget=engine.is_under_strict_step_budget(),
            remaining_steps=engine.get_remaining_step_budget(),
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
            max_expansions = max_expansions_budget
            had_relaxable_block = False

            sim_start_time = time.monotonic()
            max_sim_wall_seconds = 0.045
            best_frontier_node: int | None = None
            best_frontier_h: float = h0
            expansions_done = 0
            timed_out = False

            while open_set and max_expansions > 0:
                expansions_done += 1
                if (expansions_done & 31 == 0) and (
                    time.monotonic() - sim_start_time > max_sim_wall_seconds
                ):
                    timed_out = True
                    logger.debug(
                        "MentalSimulationPlanner: Any-time RTA* deadline exceeded (%.3fs, %d expansions). Returning best partial path.",
                        time.monotonic() - sim_start_time,
                        expansions_done,
                    )
                    break
                max_expansions -= 1
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
                visited_states.add(state_key)

                if (
                    not is_cargo_attached
                    and not is_block_delivery
                    and (cur_pos in goals or cur_pos in all_goal_cells)
                    and path_len > 0
                ):
                    path = _reconstruct_path(node_id)
                    if unattached_cargo:
                        stride = max(1, getattr(engine, "lattice_stride", 1) or 1)
                        if any(
                            abs(cur_pos[0] - uc.grid_pos[0]) + abs(cur_pos[1] - uc.grid_pos[1])
                            <= stride * 2
                            for uc in unattached_cargo
                        ):
                            interact_act = next(
                                (
                                    a
                                    for a in available_actions
                                    if a not in [m[0] for m in movable_actions]
                                    and (
                                        engine.is_spatial_effector(a)
                                        or a == 5
                                        or getattr(engine, "interaction_action", None) == a
                                    )
                                ),
                                None,
                            )
                            if interact_act is not None:
                                path.append(
                                    MentalSimulationStep(
                                        action=interact_act,
                                        predicted_avatar_pos=cur_pos,
                                        expected_mutation="attach_cargo",
                                    )
                                )
                    logger.info(
                        "AutonomousEpistemicEngine: Mental Simulation SUCCEEDED! Synthesized %d-step path to goal %s.",
                        len(path),
                        cur_pos,
                    )
                    engine.current_simulated_goal = cur_pos
                    return path, had_relaxable_block

                if is_block_delivery and all(g in cur_blocks for g in goals) and path_len > 0:
                    path = _reconstruct_path(node_id)
                    logger.info(
                        "AutonomousEpistemicEngine: Compound Mental Simulation SUCCEEDED! Synthesized %d-step block delivery plan.",
                        len(path),
                    )
                    engine.current_simulated_goal = list(goals)[0] if goals else None
                    return path, had_relaxable_block

                candidate_actions: list[tuple[Any, int, int]] = list(movable_actions)
                max_consecutive_waits = max(4, min(16, engine.hazard_tracker.environmental_period))

                living_gaze: set[tuple[int, int]] = set()
                seen_threats: set[int] = set()
                for t_pos in cur_threats:
                    t = threat_map.get(t_pos)
                    if t is not None and id(t) not in seen_threats:
                        seen_threats.add(id(t))
                        living_gaze.add(t.gaze_pos)
                        if t.facing != (0, 0):
                            d_r, d_c = t.facing
                            step_r = int(np.sign(d_r))
                            step_c = int(np.sign(d_c))
                            dist_gaze = max(
                                abs(t.gaze_pos[0] - t.pos[0]), abs(t.gaze_pos[1] - t.pos[1])
                            )
                            for k in range(1, dist_gaze + 1):
                                living_gaze.add((t.pos[0] + step_r * k, t.pos[1] + step_c * k))

                if cur_patrols:
                    for pp, pf in cur_patrols:
                        if pf != (0, 0):
                            for k in range(1, 8):
                                gr = pp[0] + pf[0] * k
                                gc = pp[1] + pf[1] * k
                                if not (0 <= gr < H and 0 <= gc < W) or (gr, gc) in _grid_walls:
                                    break
                                living_gaze.add((gr, gc))

                # Temporal Evasion: Determine safe waiting action (explicit non-displacement action)
                bump_wait_action: Any | None = non_disp_wait_action

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

                    cur_cargo_pos: tuple[int, int] | None = None
                    if is_cargo_attached and attached_cargo_offset is not None:
                        cur_cargo_pos = (
                            cur_pos[0] + attached_cargo_offset[0],
                            cur_pos[1] + attached_cargo_offset[1],
                        )

                    # Core Knowledge Physics: Evaluate Spelke Continuity along candidate path step
                    if hasattr(engine.symbolic_theory, "evaluates_spelke_continuity"):
                        active_barriers = static_barriers - cur_open
                        if cur_cargo_pos is not None and cur_cargo_pos in active_barriers:
                            active_barriers = active_barriers - {cur_cargo_pos}
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
                        if (kr, kc) in living_gaze:
                            ray_blocked = True
                            break
                        is_barrier_cell = (
                            (kr, kc) in static_barriers
                            and (kr, kc) not in cur_open
                            and (cur_cargo_pos is None or (kr, kc) != cur_cargo_pos)
                        )
                        if is_barrier_cell:
                            cell_feat = (
                                int(curr_grid[kr, kc]) if 0 <= kr < H and 0 <= kc < W else -1
                            )
                            if engine.working_memory.body_schema.is_barrier_permeable(cell_feat):
                                pass
                            elif (
                                cell_feat in engine.symbolic_theory.goal_features
                                or cell_feat in engine.symbolic_theory.candidate_goal_features
                                or (kr, kc) in all_goal_cells
                            ):
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

                    walkable_feats = engine.symbolic_theory.walkable_features
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

                    if (nr, nc) in living_gaze:
                        continue

                    threat_penalty = 0.0
                    new_threats = cur_threats
                    if (nr, nc) in cur_threats:
                        flanked_t = threat_map.get((nr, nc))
                        if flanked_t is not None and cur_pos != flanked_t.gaze_pos:
                            threat_cells_to_remove = {
                                p
                                for p, ot in threat_map.items()
                                if ot is flanked_t or p == (nr, nc)
                            }
                            new_threats = cur_threats - threat_cells_to_remove
                            threat_penalty = 5.0
                        else:
                            continue

                    new_patrols = cur_patrols
                    patrol_proximity_cost = 0.0
                    if cur_patrols:
                        predicted = advance_patrols(cur_patrols, (nr, nc))
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

                    if (
                        is_cargo_attached
                        and cur_cargo_pos is not None
                        and attached_cargo_offset is not None
                    ):
                        # Cannot walk directly into another unattached obstacle/block
                        if (nr, nc) in cur_blocks and (nr, nc) != cur_cargo_pos:
                            continue

                        next_cargo_pos = (
                            nr + attached_cargo_offset[0],
                            nc + attached_cargo_offset[1],
                        )

                        # Cargo boundary check
                        if not (0 <= next_cargo_pos[0] < H and 0 <= next_cargo_pos[1] < W):
                            continue

                        # Cargo collision against walls and static barriers (can enter cur_pos vacated by avatar)
                        if next_cargo_pos in _grid_walls:
                            continue
                        if (
                            next_cargo_pos in static_barriers
                            and next_cargo_pos not in cur_open
                            and next_cargo_pos != cur_pos
                        ):
                            continue

                        # Continuous ray check for cargo along motion step
                        cargo_ray_blocked = False
                        for pk in range(1, dist + 1):
                            ckr = cur_cargo_pos[0] + step_r * pk
                            ckc = cur_cargo_pos[1] + step_c * pk
                            if (ckr, ckc) in living_gaze:
                                cargo_ray_blocked = True
                                break
                            if (ckr, ckc) in _grid_walls:
                                cargo_ray_blocked = True
                                break
                            if (
                                (ckr, ckc) in static_barriers
                                and (ckr, ckc) not in cur_open
                                and (ckr, ckc) != cur_pos
                            ):
                                cargo_ray_blocked = True
                                break
                            if (
                                bg_is_impassable_void
                                and 0 <= ckr < H
                                and 0 <= ckc < W
                                and int(curr_grid[ckr, ckc]) == bg
                                and (ckr, ckc) not in goals
                                and (ckr, ckc) not in all_goal_cells
                            ):
                                cargo_ray_blocked = True
                                break
                        if cargo_ray_blocked:
                            continue

                        # Hazard check for cargo
                        if engine.hazard_tracker.is_hazard_at(
                            next_cargo_pos[0],
                            next_cargo_pos[1],
                            sim_step,
                            bg,
                            avatar_features=av_feats,
                            walkable_features=walkable_feats,
                        ):
                            continue

                        # Deadlock evaluation for cargo if not delivering to a goal cell
                        if next_cargo_pos not in goals and next_cargo_pos not in all_goal_cells:
                            candidate_block_set = (set(cur_blocks) - {cur_cargo_pos}) | {
                                next_cargo_pos
                            }
                            dl_eval = CounterfactualDeadlockDetector.evaluate_deadlock(
                                next_cargo_pos,
                                set(goals),
                                static_barriers,
                                candidate_block_set,
                                (H, W),
                            )
                            if dl_eval.is_deadlock:
                                continue

                        block_set = set(cur_blocks)
                        if cur_cargo_pos in block_set:
                            block_set.remove(cur_cargo_pos)
                        block_set.add(next_cargo_pos)
                        new_blocks = frozenset(block_set)

                    elif (nr, nc) in cur_blocks:
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
                                engine.symbolic_theory.barrier_features,
                                static_barriers,
                            ):
                                land_b, _, is_b_lethal = engine.physics_engine.project_fall(
                                    pushed_r,
                                    pushed_c,
                                    curr_grid,
                                    engine.symbolic_theory.barrier_features,
                                    static_barriers,
                                )
                                if is_b_lethal:
                                    continue
                                pushed_r, pushed_c = land_b

                        if (
                            bool(cur_blocks)
                            and (
                                pushed_r,
                                pushed_c,
                            )
                            not in goals
                        ):
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
                    new_h = heuristic(new_pos, new_blocks)
                    new_step = MentalSimulationStep(action=real_act, predicted_avatar_pos=new_pos)
                    counter += 1
                    parent_map[counter] = (node_id, new_step)
                    if new_h < best_frontier_h:
                        best_frontier_h = new_h
                        best_frontier_node = counter
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

            if (
                (timed_out or max_expansions <= 0)
                and best_frontier_node is not None
                and best_frontier_h < h0
            ):
                partial_path = _reconstruct_path(best_frontier_node)
                if partial_path:
                    logger.info(
                        "AutonomousEpistemicEngine: Any-time RTA* partial plan found (%d steps, h: %.1f -> %.1f).",
                        len(partial_path),
                        h0,
                        best_frontier_h,
                    )
                    if partial_path[-1].predicted_avatar_pos is not None:
                        engine.current_simulated_goal = partial_path[-1].predicted_avatar_pos
                    return partial_path, had_relaxable_block

            return None, had_relaxable_block

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
        if mutation_triggers or goals:
            subgoal_plan = MentalSimulationPlanner._plan_hierarchical_subgoals(
                engine,
                curr_grid,
                available_actions,
                start_pos,
                set(goals),
                static_barriers,
                mutation_triggers,
                movable_actions,
                oriented_threats=oriented_threats,
            )
            if subgoal_plan:
                return subgoal_plan

        # Teleological Backward Cut-Set Decomposition Fallback (Human BA10 Reasoning)
        if goals and start_pos is not None:
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
                return teleo_plan

        return None

    @staticmethod
    def _plan_affordance_subgoals(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        entities: list[SpatialEntity],
        bg: int,
        av_feats: set[int],
        available_actions: Sequence[Any],
        movable_actions: list[tuple[Any, int, int]],
        static_barriers: set[tuple[int, int]],
        goals: Sequence[tuple[int, int]] | set[tuple[int, int]],
        dynamic_obstacles: set[tuple[int, int]] | None = None,
    ) -> list[MentalSimulationStep] | None:
        """Formulate multi-stage frontopolar affordance subgoals for portable/cargo entities and receptacles.

        Decomposes object manipulation into clean, modular subgoals:
        [Approach Cargo -> Attach via Interaction -> Compound Navigate to Receptacle -> Detach].
        Eliminates entangled space-time search heuristics.
        """
        start_pos = engine.avatar_pos
        if start_pos is None:
            return None

        step_sz = max(1, getattr(engine, "lattice_stride", 1) or 1)
        for _, dr, dc in movable_actions:
            dist = max(abs(dr), abs(dc))
            if dist > step_sz:
                step_sz = dist
        stride = step_sz

        if hasattr(engine, "snap_to_lattice"):
            start_pos = engine.snap_to_lattice(start_pos) or start_pos

        # 1. Identify destination Receptacle / Goal Drop Zone
        receptacle_candidates: list[tuple[int, int]] = []
        if getattr(engine, "learned_receptacle_positions", None):
            for r_pos in engine.learned_receptacle_positions:
                if r_pos not in receptacle_candidates:
                    receptacle_candidates.append(r_pos)
        for g_pos in engine.learned_goal_positions:
            if g_pos not in receptacle_candidates:
                receptacle_candidates.append(g_pos)
        for e in entities:
            if engine.symbolic_theory.has_affordance(e, "receptacle") or getattr(e, "role", "") in (
                EntityRole.RECEPTACLE,
                EntityRole.GOAL,
            ):
                if e.grid_pos not in receptacle_candidates:
                    receptacle_candidates.append(e.grid_pos)
        if not receptacle_candidates and goals:
            for g in goals:
                receptacle_candidates.append(g)

        if not receptacle_candidates:
            return None

        # 2. Check if avatar is currently carrying an incorporated object
        is_holding = engine.working_memory.body_schema.has_manipulandum()
        active_manip = engine.working_memory.body_schema.active_manipulandum if is_holding else None

        # 3. Identify all uncollected pickable entities on the board
        cargos_known = engine.symbolic_theory.cargo_features | engine.learned_cargo_features
        learned_recs = getattr(engine, "learned_receptacle_positions", set())

        def _is_receptacle_component(e: SpatialEntity) -> bool:
            if e.feature_id in engine.symbolic_theory.receptacle_features:
                return True
            if e.feature_id in cargos_known or engine.symbolic_theory.has_affordance(
                e, "can_pickup"
            ):
                return False
            e_p = (
                engine.snap_to_lattice(e.grid_pos)
                if hasattr(engine, "snap_to_lattice")
                else e.grid_pos
            )
            if (
                e.grid_pos in receptacle_candidates
                or e.grid_pos in learned_recs
                or e_p in receptacle_candidates
                or e_p in learned_recs
            ):
                return True
            for r_pos in receptacle_candidates:
                if (
                    e.bounding_box[0] <= r_pos[0] <= e.bounding_box[1]
                    and e.bounding_box[2] <= r_pos[1] <= e.bounding_box[3]
                ):
                    return True
            for r_pos in learned_recs:
                if (
                    e.bounding_box[0] <= r_pos[0] <= e.bounding_box[1]
                    and e.bounding_box[2] <= r_pos[1] <= e.bounding_box[3]
                ):
                    return True
            return False

        filled_receptacle_slots: set[tuple[int, int]] = {
            (
                engine.snap_to_lattice(e.grid_pos)
                if hasattr(engine, "snap_to_lattice")
                else e.grid_pos
            )
            for e in entities
            if (
                e.feature_id in cargos_known
                or engine.symbolic_theory.has_affordance(e, "can_pickup")
            )
            and e.feature_id not in engine.symbolic_theory.receptacle_features
            and (
                e.grid_pos in learned_recs
                or e.grid_pos in receptacle_candidates
                or (
                    engine.snap_to_lattice(e.grid_pos)
                    if hasattr(engine, "snap_to_lattice")
                    else e.grid_pos
                )
                in learned_recs
                or (
                    engine.snap_to_lattice(e.grid_pos)
                    if hasattr(engine, "snap_to_lattice")
                    else e.grid_pos
                )
                in receptacle_candidates
            )
            and (not is_holding or active_manip is None or e.grid_pos != active_manip.pos)
        }

        pickable_entities: list[SpatialEntity] = []
        seen_pickable_pos: set[tuple[int, int]] = set()
        for e in entities:
            if (
                e.feature_id in av_feats
                or e.feature_id == bg
                or e.feature_id == 0
                or getattr(e, "role", "") == EntityRole.OBSTACLE
                or (
                    e.bounding_box[0] == e.bounding_box[1] or e.bounding_box[2] == e.bounding_box[3]
                )
                or engine.symbolic_theory.is_barrier(e.feature_id)
            ):
                continue
            if not (
                engine.symbolic_theory.has_affordance(e, "can_pickup")
                or e.feature_id in cargos_known
                or getattr(e, "role", "") == EntityRole.MANIPULABLE
            ):
                continue
            if _is_receptacle_component(e):
                continue
            e_p = (
                engine.snap_to_lattice(e.grid_pos)
                if hasattr(engine, "snap_to_lattice")
                else e.grid_pos
            )
            if e_p is None or e_p in seen_pickable_pos or e_p in filled_receptacle_slots:
                continue
            seen_pickable_pos.add(e_p)
            pickable_entities.append(e)

        if not is_holding and not pickable_entities:
            return None

        # 4. Resolve interaction action
        interact_act = getattr(engine.symbolic_theory, "primary_interaction_action", None)
        if interact_act is None and getattr(engine.symbolic_theory, "interaction_actions", None):
            interact_act = next(iter(engine.symbolic_theory.interaction_actions))
        if interact_act is None:
            interact_act = getattr(engine, "interaction_action", None)
        if interact_act is None:
            movable_act_ids = {m[0] for m in movable_actions}
            interact_act = next(
                (
                    a
                    for a in available_actions
                    if a not in movable_act_ids
                    and (
                        a == 5
                        or engine.is_spatial_effector(a)
                        or not engine.is_displacement_action(a)
                    )
                ),
                None,
            )
        if interact_act is None:
            return None

        H, W = curr_grid.shape

        # Clean Geodesic BFS router for a single subgoal segment
        def _route_segment(
            from_pos: tuple[int, int],
            to_pos: tuple[int, int],
            held_offset: tuple[int, int] | None = None,
            obstacles: set[tuple[int, int]] = set(),
        ) -> list[tuple[Any, tuple[int, int]]] | None:
            if from_pos == to_pos:
                return []
            q: deque[tuple[int, int]] = deque([from_pos])
            parent: dict[tuple[int, int], tuple[tuple[int, int], Any]] = {}
            visited: set[tuple[int, int]] = {from_pos}
            while q:
                curr = q.popleft()
                if curr == to_pos:
                    path: list[tuple[Any, tuple[int, int]]] = []
                    c = curr
                    while c in parent:
                        p, act = parent[c]
                        path.append((act, c))
                        c = p
                    path.reverse()
                    return path
                for act, dr, dc in movable_actions:
                    nr, nc = curr[0] + dr, curr[1] + dc
                    nxt = (nr, nc)
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if nxt in static_barriers or nxt in obstacles:
                        continue
                    step_r = int(np.sign(dr))
                    step_c = int(np.sign(dc))
                    dist = max(abs(dr), abs(dc))
                    ray_blocked = False
                    for k in range(1, dist + 1):
                        kr, kc = curr[0] + step_r * k, curr[1] + step_c * k
                        if (kr, kc) in static_barriers or (kr, kc) in obstacles:
                            ray_blocked = True
                            break
                    if ray_blocked:
                        continue
                    if held_offset is not None:
                        cr, cc = nr + held_offset[0], nc + held_offset[1]
                        if not (0 <= cr < H and 0 <= cc < W):
                            continue
                        if (cr, cc) in static_barriers or (cr, cc) in obstacles:
                            continue
                        for k in range(1, dist + 1):
                            ckr = (curr[0] + held_offset[0]) + step_r * k
                            ckc = (curr[1] + held_offset[1]) + step_c * k
                            if (ckr, ckc) in static_barriers or (ckr, ckc) in obstacles:
                                ray_blocked = True
                                break
                        if ray_blocked:
                            continue
                    if nxt not in visited:
                        visited.add(nxt)
                        parent[nxt] = (curr, act)
                        q.append(nxt)
            return None

        plan_steps: list[MentalSimulationStep] = []
        curr_av = start_pos
        remaining_entities = list(pickable_entities)

        # Subgoal Stage A: If avatar is already carrying cargo, deliver it first
        if is_holding and active_manip is not None:
            held_offset = (active_manip.pos[0] - curr_av[0], active_manip.pos[1] - curr_av[1])
            other_obs = {
                p
                for e in remaining_entities
                if e.grid_pos != active_manip.pos
                and (
                    p := (
                        engine.snap_to_lattice(e.grid_pos)
                        if hasattr(engine, "snap_to_lattice")
                        else e.grid_pos
                    )
                )
                is not None
            }
            other_obs.update(filled_receptacle_slots)
            if dynamic_obstacles:
                other_obs.update(dynamic_obstacles)
            deliv_seg = None
            dest_rec = None
            for cand_rec in receptacle_candidates:
                if cand_rec in filled_receptacle_slots or cand_rec in other_obs:
                    continue
                target_av = (cand_rec[0] - held_offset[0], cand_rec[1] - held_offset[1])
                seg = _route_segment(
                    curr_av, target_av, held_offset=held_offset, obstacles=other_obs
                )
                if seg is not None and (deliv_seg is None or len(seg) < len(deliv_seg)):
                    deliv_seg = seg
                    dest_rec = cand_rec
            if deliv_seg is not None and dest_rec is not None:
                for act, nxt_p in deliv_seg:
                    plan_steps.append(
                        MentalSimulationStep(
                            action=act,
                            predicted_avatar_pos=nxt_p,
                            expected_mutation="deliver_cargo",
                        )
                    )
                curr_av = (dest_rec[0] - held_offset[0], dest_rec[1] - held_offset[1])
                plan_steps.append(
                    MentalSimulationStep(
                        action=interact_act,
                        predicted_avatar_pos=curr_av,
                        expected_mutation="detach_cargo_at_goal",
                    )
                )
                filled_receptacle_slots.add(dest_rec)
            else:
                return None

        # Subgoal Stage B: Collect and deliver each remaining pickable entity
        uncollected_pickable = [
            e
            for e in remaining_entities
            if not (is_holding and active_manip is not None and e.grid_pos == active_manip.pos)
        ]
        for ent in uncollected_pickable:
            ent_pos = (
                engine.snap_to_lattice(ent.grid_pos)
                if hasattr(engine, "snap_to_lattice")
                else ent.grid_pos
            )
            if ent_pos is None:
                continue
            other_obs = {
                p
                for e in remaining_entities
                if (
                    p := (
                        engine.snap_to_lattice(e.grid_pos)
                        if hasattr(engine, "snap_to_lattice")
                        else e.grid_pos
                    )
                )
                is not None
                and p != ent_pos
                and (not is_holding or active_manip is None or p != active_manip.pos)
            }
            other_obs.update(filled_receptacle_slots)
            if dynamic_obstacles:
                other_obs.update(dynamic_obstacles)

            adj_candidates = [
                (ent_pos[0] + dr, ent_pos[1] + dc)
                for dr, dc in [(-stride, 0), (stride, 0), (0, -stride), (0, stride)]
                if 0 <= ent_pos[0] + dr < H
                and 0 <= ent_pos[1] + dc < W
                and (ent_pos[0] + dr, ent_pos[1] + dc) not in static_barriers
                and (ent_pos[0] + dr, ent_pos[1] + dc) not in other_obs
            ]
            best_pickup_seg: list[tuple[Any, tuple[int, int]]] | None = None
            best_pickup_cell: tuple[int, int] | None = None
            for cand in adj_candidates:
                seg = _route_segment(curr_av, cand, obstacles=other_obs | {ent_pos})
                if seg is not None:
                    if best_pickup_seg is None or len(seg) < len(best_pickup_seg):
                        best_pickup_seg = seg
                        best_pickup_cell = cand
            if best_pickup_seg is None or best_pickup_cell is None:
                continue

            # Orient towards cargo if approaching from a non-aligned heading
            dr_to_box = ent_pos[0] - best_pickup_cell[0]
            dc_to_box = ent_pos[1] - best_pickup_cell[1]
            face_act = next(
                (
                    act
                    for act, dr, dc in movable_actions
                    if (dr * dr_to_box > 0 or dc * dc_to_box > 0)
                ),
                None,
            )
            last_mov = best_pickup_seg[-1][0] if best_pickup_seg else None
            held_offset = (ent_pos[0] - best_pickup_cell[0], ent_pos[1] - best_pickup_cell[1])

            # Compound navigate to deliver cargo into an available receptacle slot
            deliv_seg = None
            dest_rec = None
            for cand_rec in receptacle_candidates:
                if cand_rec in filled_receptacle_slots or cand_rec in other_obs:
                    continue
                target_av = (cand_rec[0] - held_offset[0], cand_rec[1] - held_offset[1])
                seg = _route_segment(
                    best_pickup_cell, target_av, held_offset=held_offset, obstacles=other_obs
                )
                if seg is not None and (deliv_seg is None or len(seg) < len(deliv_seg)):
                    deliv_seg = seg
                    dest_rec = cand_rec
            if deliv_seg is None or dest_rec is None:
                continue

            # BOTH pickup and delivery segments are verified! Commit all steps atomically:
            for act, nxt_p in best_pickup_seg:
                plan_steps.append(
                    MentalSimulationStep(
                        action=act,
                        predicted_avatar_pos=nxt_p,
                        expected_mutation="approach_cargo",
                    )
                )
            curr_av = best_pickup_cell
            if face_act is not None and (last_mov is None or last_mov != face_act):
                plan_steps.append(
                    MentalSimulationStep(
                        action=face_act,
                        predicted_avatar_pos=curr_av,
                        expected_mutation="orient_towards_cargo",
                    )
                )

            plan_steps.append(
                MentalSimulationStep(
                    action=interact_act,
                    predicted_avatar_pos=curr_av,
                    expected_mutation="attach_cargo",
                )
            )

            for act, nxt_p in deliv_seg:
                plan_steps.append(
                    MentalSimulationStep(
                        action=act,
                        predicted_avatar_pos=nxt_p,
                        expected_mutation="deliver_cargo",
                    )
                )
            curr_av = (dest_rec[0] - held_offset[0], dest_rec[1] - held_offset[1])

            # Detach cargo inside receptacle
            plan_steps.append(
                MentalSimulationStep(
                    action=interact_act,
                    predicted_avatar_pos=curr_av,
                    expected_mutation="detach_cargo_at_goal",
                )
            )
            filled_receptacle_slots.add(dest_rec)
            if ent in remaining_entities:
                remaining_entities.remove(ent)
            if dynamic_obstacles:
                break

        if plan_steps:
            logger.info(
                "MentalSimulationPlanner: Formulated Affordance-Driven Subgoal Plan (%d steps for %d entities)",
                len(plan_steps),
                len(pickable_entities),
            )
            return plan_steps

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
        oriented_threats: Sequence[Any] = (),
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

        threat_hazard_cells: set[tuple[int, int]] = set()
        for t in oriented_threats:
            threat_hazard_cells.add(t.gaze_pos)
            if getattr(t, "facing", (0, 0)) != (0, 0):
                d_r, d_c = t.facing
                step_r = int(np.sign(d_r))
                step_c = int(np.sign(d_c))
                dist_gaze = max(abs(t.gaze_pos[0] - t.pos[0]), abs(t.gaze_pos[1] - t.pos[1]))
                for k in range(1, dist_gaze + 1):
                    threat_hazard_cells.add((t.pos[0] + step_r * k, t.pos[1] + step_c * k))

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
                    if (nr, nc) in visited or (nr, nc) in threat_hazard_cells:
                        continue

                    step_r = int(np.sign(dr))
                    step_c = int(np.sign(dc))
                    dist = max(abs(dr), abs(dc))
                    ray_blocked = False
                    for k in range(1, dist + 1):
                        kr = p[0] + step_r * k
                        kc = p[1] + step_c * k
                        if (kr, kc) in threat_hazard_cells:
                            ray_blocked = True
                            break
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
                        for e in entities:
                            if e.feature_id == req_tool and e.grid_pos in reachable_cells:
                                t_path = find_shortest_path(
                                    cur_pos, {e.grid_pos}, effective_barriers
                                )
                                if t_path is not None:
                                    candidate_triggers.append((e.grid_pos, t_path, 80, {b_pos}))

            # 4. Known state mutation triggers (switches and pressure plates)
            for tr_pos, opened_set in mutation_triggers.items():
                if tr_pos in visited_triggers:
                    continue
                unopened = (opened_set & effective_barriers) - open_barriers
                if not unopened:
                    continue
                tr_path = find_shortest_path(cur_pos, {tr_pos}, effective_barriers)
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
                    and e.grid_pos in reachable_cells
                ]
                for t_pos in candidate_tools:
                    t_path = find_shortest_path(cur_pos, {t_pos}, effective_barriers)
                    if t_path is not None and len(t_path) > 0:
                        candidate_triggers.append((t_pos, t_path, 1, set()))

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
                    body_schema=getattr(engine.working_memory, "body_schema", None),
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
            cur_pos = chosen_pos
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

        for e in entities:
            if e.grid_pos == start_pos or e.feature_id in av_feats or e.feature_id == bg:
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
            if e.grid_pos in reachable_cells:
                target_path = reconstruct_path(e.grid_pos)
            else:
                # Entity pos might be tentatively marked as barrier; check if approachable from adjacent manifold
                best_adj_p: list[MentalSimulationStep] | None = None
                for act, dr, dc in movable_actions:
                    adj_p = (e.grid_pos[0] - dr, e.grid_pos[1] - dc)
                    if adj_p in reachable_cells:
                        p_to_adj = reconstruct_path(adj_p)
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
                body_schema=getattr(engine.working_memory, "body_schema", None),
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
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export dynamic replanner capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="dynamic_replanner",
            operator="==",
            value="ACTIVE",
            relation="divergence_rerouting",
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
