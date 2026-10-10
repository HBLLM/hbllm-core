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
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.counterfactual_simulation import CounterfactualDeadlockDetector
from hbllm.hcir.world.frontopolar_subgoal_stack import FrontopolarSubgoal, SubgoalType

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


class MentalSimulationPlanner:
    """Simulates candidate action sequences internally in imagination using learned world theory."""

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
        H: int,
        W: int,
    ) -> bool:
        """Check whether a cargo block is trapped along a boundary wall with no goals on it."""
        return CounterfactualDeadlockDetector.is_wall_deadlock(pos, goals, set(), (H, W))

    @staticmethod
    def simulate_in_mind(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[int],
        target_role: EntityRole = EntityRole.GOAL,
        blocked_cells: set[tuple[int, int]] | None = None,
    ) -> list[MentalSimulationStep] | None:
        """Simulate candidate action sequences internally in memory without taking physical steps."""
        if not engine.is_motor_grounded() or engine.avatar_pos is None:
            return None

        H, W = curr_grid.shape
        bg = engine.estimate_background(curr_grid)
        entities = engine.extract_entities(curr_grid, bg)

        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        # 1. Identify goals using a principled epistemic priority hierarchy.
        #    (a) Confirmed goals — causal evidence from past wins
        confirmed_goals: list[tuple[int, int]] = []
        for e in entities:
            if engine.avatar_pos is not None and e.grid_pos == engine.avatar_pos:
                continue
            if e.feature_id in av_feats:
                if (
                    engine.avatar_pos is not None
                    and (
                        abs(e.grid_pos[0] - engine.avatar_pos[0])
                        + abs(e.grid_pos[1] - engine.avatar_pos[1])
                    )
                    > 4
                    and (e.role == EntityRole.GOAL or engine.symbolic_theory.is_goal(e))
                ):
                    confirmed_goals.append(e.grid_pos)
                continue
            if (
                engine.symbolic_theory.is_goal(e)
                or e.feature_id in engine.symbolic_theory.goal_features
                or (e.feature_id in engine.learned_goal_features and 1 <= e.area <= 64)
            ):
                if e.grid_pos not in confirmed_goals:
                    confirmed_goals.append(e.grid_pos)
                    engine.learned_goal_positions.add(e.grid_pos)

        # Object Permanence: Retain known goal positions if currently occluded by mobile entities
        for gp in engine.learned_goal_positions:
            if 0 <= gp[0] < H and 0 <= gp[1] < W and gp != engine.avatar_pos:
                if gp not in confirmed_goals:
                    confirmed_goals.append(gp)

        goals: list[tuple[int, int]] = []
        if confirmed_goals:
            goals = list(confirmed_goals)

        # Faculty: Brodmann Area 10 (aPFC) Frontopolar Subgoal Prioritization
        # If an executive subgoal is active and unsatisfied, prioritize its target destination ahead of confirmed goals.
        if hasattr(engine, "subgoal_stack") and engine.subgoal_stack.has_pending_goals:
            cur_sg = engine.subgoal_stack.current_subgoal
            if (
                cur_sg is not None
                and cur_sg.target_destination != engine.avatar_pos
                and not engine.subgoal_stack.is_subgoal_satisfied(
                    cur_sg, set(), avatar_pos=engine.avatar_pos
                )
            ):
                sg_dest = cur_sg.target_destination
                if 0 <= sg_dest[0] < H and 0 <= sg_dest[1] < W:
                    if sg_dest in goals:
                        goals.remove(sg_dest)
                    goals.insert(0, sg_dest)
            elif cur_sg is not None and cur_sg.target_destination == engine.avatar_pos:
                engine.subgoal_stack.pop_subgoal()

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

        # (b) Hypothesized goals from previous level knowledge or role inference.
        if not goals:
            hypothesized_goals: list[tuple[int, int]] = []
            for e in entities:
                if engine.avatar_pos is not None and e.grid_pos == engine.avatar_pos:
                    continue
                if engine.avatar_pos is not None and e.feature_id in av_feats:
                    if (
                        abs(e.grid_pos[0] - engine.avatar_pos[0]) <= 3
                        and abs(e.grid_pos[1] - engine.avatar_pos[1]) <= 3
                    ):
                        continue
                elif e.feature_id in av_feats:
                    continue
                if e.grid_pos in panel_coords:
                    continue
                if e.feature_id in engine.symbolic_theory.candidate_goal_features:
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
                )
            ]
            cand_sgs = primary_sgs if primary_sgs else structural_goals
            for sg in cand_sgs:
                feat = sg.get("feature")
                if feat is not None and (
                    (feat in av_feats and not sg.get("is_midpoint"))
                    or feat in engine.hazard_tracker.known_lethal_features
                    or engine.symbolic_theory.is_barrier(feat)
                ):
                    continue
                cells = sg.get("cells")
                if cells:
                    for cp in cells:
                        if (
                            0 <= cp[0] < H
                            and 0 <= cp[1] < W
                            and cp != engine.avatar_pos
                            and cp not in panel_coords
                        ):
                            if cp not in goals:
                                goals.append(cp)
                else:
                    pos = sg.get("position")
                    if (
                        pos
                        and 0 <= pos[0] < H
                        and 0 <= pos[1] < W
                        and pos != engine.avatar_pos
                        and pos not in panel_coords
                    ):
                        if pos not in goals:
                            goals.append(pos)

        # (d) Exploratory guesses: unknown entities
        if not goals:
            exploratory_goals = [
                e.grid_pos
                for e in entities
                if e.grid_pos not in panel_coords
                and e.role in (EntityRole.UNKNOWN, EntityRole.AGENT)
                and 1 <= e.area <= 64
                and e.feature_id != bg
                and e.feature_id not in engine.hazard_tracker.known_lethal_features
                and not engine.symbolic_theory.is_barrier(e.feature_id)
                and (
                    e.feature_id not in av_feats
                    or (
                        engine.avatar_pos is not None
                        and (
                            abs(e.grid_pos[0] - engine.avatar_pos[0])
                            + abs(e.grid_pos[1] - engine.avatar_pos[1])
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
                    ):
                        engine.symbolic_theory.candidate_goal_features.add(e.feature_id)

        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        engine.symbolic_theory.goal_features.difference_update(av_feats)
        if engine.avatar_pos is not None:
            goals = [
                g
                for g in goals
                if g != engine.avatar_pos and (g in confirmed_goals or g not in panel_coords)
            ]

        unexhausted_goals = [g for g in goals if g not in engine.exhausted_candidate_goals]
        if unexhausted_goals:
            goals = unexhausted_goals
        elif goals:
            engine.exhausted_candidate_goals.clear()

        if engine.avatar_pos is not None and len(goals) > 8:
            goals.sort(
                key=lambda g: abs(g[0] - engine.avatar_pos[0]) + abs(g[1] - engine.avatar_pos[1])
            )
            goals = goals[:8]

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

            action_deltas = {
                act: dyn.get_displacement()
                for act, dyn in engine.action_dynamics.items()
                if dyn.is_displacement_action()
            }
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

        if goals and engine.avatar_pos is not None:
            av_pts = (
                np.argwhere(curr_grid == engine.avatar_feature)
                if engine.avatar_feature is not None
                else np.empty((0, 2))
            )
            if len(av_pts) >= 15:
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
                        if (
                            proj != engine.avatar_pos
                            and 0 <= proj[1] < W
                            and proj not in vert_goals
                        ):
                            vert_goals.append(proj)
                    if vert_goals:
                        goals = vert_goals
                elif is_extended_horiz:
                    horiz_goals = []
                    for g in goals:
                        proj = (g[0], engine.avatar_pos[1])
                        if (
                            proj != engine.avatar_pos
                            and 0 <= proj[0] < H
                            and proj not in horiz_goals
                        ):
                            horiz_goals.append(proj)
                    if horiz_goals:
                        goals = horiz_goals

        if not goals:
            return None

        # 2. Identify candidate pushable blocks / cargo
        pushable_blocks: list[tuple[int, int]] = [
            e.grid_pos
            for e in entities
            if (
                engine.symbolic_theory.is_cargo(e)
                or e.feature_id in engine.learned_cargo_features
                or (
                    e.role == EntityRole.MANIPULABLE
                    and 1 <= e.area <= 64
                    and e.feature_id != bg
                    and e.feature_id not in av_feats
                    and e.grid_pos not in goals
                )
            )
            and e.feature_id not in engine.hazard_tracker.known_lethal_features
            and not engine.symbolic_theory.is_barrier(e.feature_id)
        ]

        # 3. Static barriers
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

        # Structural dividing boundaries are physical barriers
        for feat in np.unique(curr_grid):
            feat_int = int(feat)
            if feat_int == bg or feat_int in av_feats:
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

        # Corridor Manifold Continuity
        bg_is_impassable_void = (
            len(engine.symbolic_theory.walkable_features) > 0
            and bg not in engine.symbolic_theory.walkable_features
        )
        if bg_is_impassable_void:
            for r in range(H):
                for c in range(W):
                    if int(curr_grid[r, c]) == bg and (r, c) not in goals:
                        static_barriers.add((r, c))

        # Enforce HCIR Attractor Invariant: Goals must never be static barriers
        static_barriers.difference_update(goals)
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
        if blocked_cells:
            static_barriers.update(blocked_cells)

        # Check known state mutation triggers (switches that open doors)
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

        # 4. Available directional movements in mental model
        movable_actions: list[tuple[Any, int, int]] = []
        for act in available_actions:
            if engine.is_spatial_effector(act):
                continue
            if (
                getattr(engine, "immobile_entity_actions", None)
                and act in engine.immobile_entity_actions
            ):
                continue
            dyn = engine.action_dynamics.get(act)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                movable_actions.append((act, dr, dc))

        if not movable_actions:
            return None

        # Unknown solid entity cells: risk-aware path planning
        unknown_entity_cells: set[tuple[int, int]] = set()
        for e in entities:
            if (
                e.feature_id != bg
                and e.feature_id != 0
                and e.feature_id not in av_feats
                and not engine.symbolic_theory.is_walkable(e.feature_id)
                and not engine.symbolic_theory.is_goal(e)
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
        )

        half_step = max(1, step_size // 2)

        _barrier_feats = engine.symbolic_theory.barrier_features
        _grid_walls: set[tuple[int, int]] = set(static_barriers)
        for rr in range(H):
            for cc in range(W):
                if int(curr_grid[rr, cc]) in _barrier_feats:
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
        init_threats = frozenset(t.pos for t in static_threats)
        threat_map = {t.pos: t for t in static_threats}
        init_patrols: frozenset[tuple[tuple[int, int], tuple[int, int]]] = frozenset(
            (t.pos, t.facing)
            for t in oriented_threats
            if t.feature_id in engine.mobile_threat_features
        )

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
                    new_f = (-f[0], -f[1])
                    nxt.add((p, new_f))
                    continue
                if dest == avatar_new:
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
            if (
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

        is_block_delivery = bool(
            1 <= len(pushable_blocks) <= 5
            and any(
                engine.symbolic_theory.is_cargo(e)
                or e.feature_id in engine.learned_cargo_features
                or e.role == EntityRole.MANIPULABLE
                for e in entities
            )
        )

        acc_monitor = getattr(getattr(engine, "working_memory", None), "acc_conflict", None)
        acc_weight = acc_monitor.compute_heuristic_weight(1.0) if acc_monitor else 1.0
        # Precompute Vectorized 2D Distance Field Matrix for O(1) heuristic lookup
        grid_dist_field = np.full((H, W), 999999.0, dtype=np.float32)
        if goals and not is_block_delivery:
            r_indices = np.arange(H, dtype=np.int32)[:, None]
            c_indices = np.arange(W, dtype=np.int32)[None, :]
            for gr, gc in goals:
                d_mat = np.abs(r_indices - gr) + np.abs(c_indices - gc)
                grid_dist_field = np.minimum(grid_dist_field, d_mat)

        def heuristic(pos: tuple[int, int], blocks: frozenset[tuple[int, int]]) -> float:
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
                list[MentalSimulationStep],
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
                [],
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
        has_periodic = (
            bool(engine.hazard_tracker.periodic_cells)
            and engine.hazard_tracker.environmental_period >= 2
        )
        has_temporal = bool(init_patrols) or bool(init_chasers) or has_periodic
        if getattr(engine, "consecutive_simulation_failures", 0) > 0 and not has_temporal:
            max_expansions = 1200
        else:
            max_expansions = 1500 if not has_temporal else 5000

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

        while open_set and max_expansions > 0:
            max_expansions -= 1
            (
                f_score,
                g_cost,
                _,
                cur_pos,
                cur_blocks,
                cur_open,
                cur_threats,
                cur_patrols,
                cur_chasers,
                path,
            ) = heapq.heappop(open_set)

            time_mod = len(path) % engine.hazard_tracker.environmental_period if has_periodic else 0
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

            if not is_block_delivery and cur_pos in goals and len(path) > 0:
                logger.info(
                    "AutonomousEpistemicEngine: Mental Simulation SUCCEEDED! Synthesized %d-step path to goal %s.",
                    len(path),
                    cur_pos,
                )
                engine.current_simulated_goal = cur_pos
                return path

            if is_block_delivery and all(g in cur_blocks for g in goals) and len(path) > 0:
                logger.info(
                    "AutonomousEpistemicEngine: Compound Mental Simulation SUCCEEDED! Synthesized %d-step block delivery plan.",
                    len(path),
                )
                engine.current_simulated_goal = list(goals)[0] if goals else None
                return path

            candidate_actions: list[tuple[Any, int, int]] = list(movable_actions)
            consecutive_waits = 0
            if path:
                for s in reversed(path):
                    if s.predicted_avatar_pos == cur_pos:
                        consecutive_waits += 1
                    else:
                        break
            max_consecutive_waits = max(4, min(16, engine.hazard_tracker.environmental_period))

            # Temporal Evasion: Determine safe waiting action (explicit wait or wall-bump in alcove)
            bump_wait_action: Any | None = non_disp_wait_action
            if bump_wait_action is None and (cur_patrols or has_periodic):
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
                (cur_patrols or has_periodic)
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
                if not is_wait and engine.inhibited_actions.get(act, 0) > 0:
                    continue
                if not is_wait and (cur_pos, act) in engine.failed_transitions:
                    dest_r, dest_c = cur_pos[0] + dr, cur_pos[1] + dc
                    if not (
                        has_periodic and (dest_r, dest_c) in engine.hazard_tracker.periodic_cells
                    ):
                        continue
                nr, nc = cur_pos[0] + dr, cur_pos[1] + dc
                if not (0 <= nr < H and 0 <= nc < W):
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
                        cell_feat = int(curr_grid[kr, kc]) if 0 <= kr < H and 0 <= kc < W else -1
                        if engine.working_memory.body_schema.is_barrier_permeable(cell_feat):
                            pass
                        else:
                            ray_blocked = True
                            break
                    if bg_is_impassable_void and 0 <= kr < H and 0 <= kc < W:
                        if int(curr_grid[kr, kc]) == bg and (kr, kc) not in goals:
                            ray_blocked = True
                            break
                if ray_blocked:
                    continue

                sim_step = len(path) + 1
                walkable_feats = (
                    engine.symbolic_theory.walkable_features | engine.verified_safe_features
                )
                if engine.hazard_tracker.is_hazard_at(
                    nr, nc, sim_step, bg, avatar_features=av_feats, walkable_features=walkable_feats
                ):
                    continue
                if engine.collision_cones.is_collision_hazard(
                    nr, nc, sim_step, ignored_features=engine.mobile_threat_features
                ):
                    continue
                if (
                    sim_step <= 1
                    and int(curr_grid[nr, nc]) in engine.hazard_tracker.known_lethal_features
                    and int(curr_grid[nr, nc]) not in engine.mobile_threat_features
                ):
                    continue

                living_gaze: set[tuple[int, int]] = set()
                for t_pos in cur_threats:
                    t = threat_map.get(t_pos)
                    if t is not None:
                        living_gaze.add(t.gaze_pos)

                living_gaze_cost = 0.0
                if (nr, nc) in living_gaze:
                    if any(
                        t.feature_id in engine.hazard_tracker.known_lethal_features
                        for t in static_threats
                        if t.gaze_pos == (nr, nc)
                    ):
                        continue
                    living_gaze_cost = 80.0

                new_threats = cur_threats
                if (nr, nc) in cur_threats:
                    t = threat_map.get((nr, nc))
                    if t is not None:
                        if cur_pos == t.gaze_pos:
                            continue
                        new_threats = cur_threats - {(nr, nc)}

                new_patrols = cur_patrols
                patrol_proximity_cost = 0.0
                if cur_patrols:
                    predicted = advance_patrols(cur_patrols, (nr, nc))
                    if predicted is None:
                        continue
                    new_patrols = predicted

                    upcoming = _patrol_next_cells(cur_patrols)
                    if (nr, nc) in upcoming:
                        patrol_proximity_cost += 2.0
                    for pp, _pf in cur_patrols:
                        dist_to_patrol = abs(nr - pp[0]) + abs(nc - pp[1])
                        if 0 < dist_to_patrol <= step_size:
                            moving_towards = False
                            if _pf[0] != 0 and (nr - pp[0]) * _pf[0] > 0:
                                moving_towards = True
                            elif _pf[1] != 0 and (nc - pp[1]) * _pf[1] > 0:
                                moving_towards = True
                            if moving_towards:
                                patrol_proximity_cost += 1.0

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
                            else:
                                is_flanked_threat = (
                                    (kr, kc) in cur_threats
                                    and (kr, kc) == (nr, nc)
                                    and (kr, kc) in threat_map
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

                if engine.physics_engine.has_gravity:
                    if not engine.physics_engine.is_supported(
                        nr, nc, curr_grid, engine.symbolic_theory.barrier_features, static_barriers
                    ):
                        land_pos, fall_traj, is_lethal = engine.physics_engine.project_fall(
                            nr,
                            nc,
                            curr_grid,
                            engine.symbolic_theory.barrier_features,
                            static_barriers,
                            lethal_features=engine.hazard_tracker.known_lethal_features,
                        )
                        if is_lethal:
                            continue
                        new_pos = land_pos

                if (nr, nc) in cur_blocks:
                    pushed_r, pushed_c = nr + dr, nc + dc
                    if not (0 <= pushed_r < H and 0 <= pushed_c < W):
                        continue
                    if (
                        pushed_r,
                        pushed_c,
                    ) in static_barriers and (pushed_r, pushed_c) not in cur_open:
                        continue
                    if (pushed_r, pushed_c) in cur_blocks:
                        continue
                    walkable_feats = (
                        engine.symbolic_theory.walkable_features | engine.verified_safe_features
                    )
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

                    if is_block_delivery and (pushed_r, pushed_c) not in goals:
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

                step_cost = (
                    1
                    + epistemic_cost
                    + patrol_proximity_cost
                    + chaser_proximity_cost
                    + wait_cost
                    + living_gaze_cost
                )
                new_g = g_cost + step_cost
                new_h = heuristic(new_pos, new_blocks)
                new_step = MentalSimulationStep(action=real_act, predicted_avatar_pos=new_pos)
                counter += 1
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
                        path + [new_step],
                    ),
                )

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
            )
            if subgoal_plan:
                return subgoal_plan

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
        bg_is_impassable_void = (
            len(engine.symbolic_theory.walkable_features) > 0
            and bg not in engine.symbolic_theory.walkable_features
        )
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
                        if (kr, kc) in active_barriers:
                            ray_blocked = True
                            break
                        if bg_is_impassable_void and 0 <= kr < H and 0 <= kc < W:
                            if int(curr_grid[kr, kc]) == bg and (kr, kc) not in goals:
                                ray_blocked = True
                                break
                        if 0 <= kr < H and 0 <= kc < W:
                            if engine.is_feature_unverified(int(curr_grid[kr, kc]), bg):
                                ray_blocked = True
                                break
                    if ray_blocked:
                        continue

                    visited.add((nr, nc))
                    step = MentalSimulationStep(action=act, predicted_avatar_pos=(nr, nc))
                    q.append(((nr, nc), path + [step]))
            return None

        buffered_subgoals: list[FrontopolarSubgoal] = []
        for _ in range(max_stages):
            effective_barriers = static_barriers - open_barriers
            goal_path = find_shortest_path(cur_pos, goals, effective_barriers)
            if goal_path is not None:
                accumulated_plan.extend(goal_path)
                logger.info(
                    "AutonomousEpistemicEngine: Hierarchical Subgoal Decomposition SUCCEEDED with %d total steps!",
                    len(accumulated_plan),
                )
                for sg in buffered_subgoals:
                    engine.subgoal_stack.push_subgoal(sg)
                return accumulated_plan

            candidate_triggers: list[tuple[tuple[int, int], list[MentalSimulationStep], int]] = []
            for tr_pos, opened_set in mutation_triggers.items():
                unopened = opened_set - open_barriers
                if not unopened:
                    continue
                tr_path = find_shortest_path(cur_pos, {tr_pos}, effective_barriers)
                if tr_path is not None:
                    candidate_triggers.append((tr_pos, tr_path, len(unopened)))

            if not candidate_triggers:
                if engine.working_memory.held_items:
                    adjacent_barrier_cells: set[tuple[int, int]] = set()
                    for br, bc in effective_barriers:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ar, ac = br + dr, bc + dc
                            if (ar, ac) not in effective_barriers and 0 <= ar < H and 0 <= ac < W:
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

                bg = engine.estimate_background(curr_grid)
                entities = engine.extract_entities(curr_grid, bg)
                av_feats = engine.avatar_features or (
                    {engine.avatar_feature} if engine.avatar_feature is not None else set()
                )
                candidate_tools = [
                    e.grid_pos
                    for e in entities
                    if e.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE, EntityRole.UNKNOWN)
                    and 1 <= e.area <= 64
                    and e.feature_id not in av_feats
                    and e.grid_pos not in goals
                    and e.grid_pos != cur_pos
                    and (engine.avatar_pos is None or e.grid_pos != engine.avatar_pos)
                ]
                for t_pos in candidate_tools:
                    t_path = find_shortest_path(cur_pos, {t_pos}, effective_barriers)
                    if t_path is not None and len(t_path) > 0:
                        candidate_triggers.append((t_pos, t_path, 1))

            if not candidate_triggers:
                break

            candidate_triggers.sort(key=lambda x: (len(x[1]), -x[2]))
            chosen_pos, switch_path, _ = candidate_triggers[0]

            buffered_subgoals.append(
                FrontopolarSubgoal(
                    subgoal_id=f"trigger_{chosen_pos[0]}_{chosen_pos[1]}",
                    subgoal_type=SubgoalType.UNLOCK_REMOTE_MECHANISM,
                    target_entity_pos=chosen_pos,
                    target_destination=chosen_pos,
                    priority=2.0,
                )
            )

            accumulated_plan.extend(switch_path)
            cur_pos = chosen_pos
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


@dataclass
class ReplanResult:
    """W140: Outcome of unexpected state divergence replanning."""

    replanned: bool
    new_plan: list[MentalSimulationStep]
    divergence_reason: str
    replan_count: int


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
