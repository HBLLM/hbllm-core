"""ARC-AGI-3 Archetype Affordance Scorer and Deliberation Policy.

Implements modular archetype heuristic policies decoupled from the core cognitive architecture:
1. ARC3ArchetypeScorer: Evaluates click target candidates using game archetype knowledge
   (memory pair matching, minesweeper constraint safe cells, optical reflection targets,
    pop-out minority buttons in affordance panels, and kinetic repulsion wave beacons).
2. ARC3DeliberationPolicy: Deliberation arbitration rules for interactive control panels
   and optical symmetry targets.
3. install_arc3_solvers(engine): Wires these solvers into an AutonomousEpistemicEngine.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from plugins.arc_agi_adapter.solvers.game_solvers import (
    MemoryCardMatchingSolver,
    MirroredConvergenceSolver,
    VortexAttractorSolver,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


class ARC3ArchetypeScorer:
    """Specialized target candidate scorer for ARC-AGI-3 game archetypes."""

    def __init__(self) -> None:
        pass

    def score_targets(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        entities: Sequence[Any],
        bg: int,
        H: int,
        W: int,
    ) -> list[tuple[int, int, float]]:
        """Score candidate coordinates across all 5 ARC-AGI-3 game archetypes."""
        candidates: list[tuple[int, int, float]] = []
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        # 1. dlPFC Working Memory Pattern Completion (Card/tile matching)
        candidates.extend(self._score_working_memory_pairs(engine, H, W, bg))

        # 2. LOC Ventral Stream Geometric Symmetry Discrepancies
        candidates.extend(self._score_symmetry_discrepancies(curr_grid, bg, H, W, engine))

        # 3. dlPFC Constraint Propagation: Minesweeper deduced safe cells
        candidates.extend(self._score_safe_cells(engine, H, W))

        # 4. Optical alignment & symmetry completion goals
        candidates.extend(self._score_structural_goals(curr_grid, bg, av_feats, engine, H, W))

        # 5. Gestalt Affordance Panels (interactive buttons & minority pop-outs)
        candidates.extend(self._score_affordance_panels(entities, curr_grid, bg, H, W, engine))

        # 5b. Kinetic wave & repulsion targeting (Vortex / attractor games)
        candidates.extend(
            self._score_kinetic_repulsion_targets(entities, curr_grid, bg, H, W, engine)
        )

        return candidates

    def _score_working_memory_pairs(
        self,
        engine: AutonomousEpistemicEngine,
        H: int,
        W: int,
        bg: int,
    ) -> list[tuple[int, int, float]]:
        candidates: list[tuple[int, int, float]] = []
        vm = getattr(engine.working_memory, "visuospatial", None)
        if vm is None:
            return candidates
        last_probed = getattr(vm, "last_probed_coord", None)
        last_feat = getattr(vm, "last_probed_feature", None)
        if last_probed is not None and last_feat is not None and last_feat != bg and last_feat != 0:
            pair_target = vm.find_matching_pair(last_probed, last_feat)
            return MemoryCardMatchingSolver.score_matching_pairs(
                last_probed=last_probed,
                last_feat=last_feat,
                matching_partner_pos=pair_target,
                H=H,
                W=W,
                bg=bg,
                quiescent_targets=engine.quiescent_click_targets,
            )
        return candidates

    def _score_symmetry_discrepancies(
        self,
        curr_grid: np.ndarray,
        bg: int,
        H: int,
        W: int,
        engine: AutonomousEpistemicEngine,
    ) -> list[tuple[int, int, float]]:
        from hbllm.hcir.world.visual_symmetry import VisualSymmetryAnalyzer

        candidates: list[tuple[int, int, float]] = []
        sym_discrepancies = VisualSymmetryAnalyzer.extract_discrepancy_targets(
            curr_grid, background_color=bg, threshold=0.55
        )
        for dr, dc, _, sym_conf in sym_discrepancies:
            if (dr, dc) not in engine.quiescent_click_targets and 0 <= dr < H and 0 <= dc < W:
                v_count = engine.entity_visit_counts.get(f"click_{dr}_{dc}", 0)
                if v_count < 3:
                    is_void = int(curr_grid[dr, dc]) == bg
                    base = 120.0 if is_void else 320.0
                    cand_score = base + (sym_conf * 40.0) - float(v_count) * 15.0
                    candidates.append((dr, dc, cand_score))
        return candidates

    def _score_safe_cells(
        self,
        engine: AutonomousEpistemicEngine,
        H: int,
        W: int,
    ) -> list[tuple[int, int, float]]:
        candidates: list[tuple[int, int, float]] = []
        deduced_safe = engine.working_memory.get_unrevealed_safe_cells()
        for sr, sc in deduced_safe:
            if (sr, sc) not in engine.quiescent_click_targets and 0 <= sr < H and 0 <= sc < W:
                visit_count = engine.entity_visit_counts.get(f"click_{sr}_{sc}", 0)
                if visit_count == 0:
                    candidates.append((sr, sc, 300.0))
        return candidates

    def _score_structural_goals(
        self,
        curr_grid: np.ndarray,
        bg: int,
        av_feats: set[int],
        engine: AutonomousEpistemicEngine,
        H: int,
        W: int,
    ) -> list[tuple[int, int, float]]:
        from hbllm.hcir.world.cortex_perception import PerceptionEngine

        structural_goals = PerceptionEngine.detect_structural_goals(
            curr_grid,
            bg=bg,
            avatar_features=av_feats,
            avatar_feature=engine.avatar_feature,
            barrier_features=engine.symbolic_theory.barrier_features,
            known_lethal_features=engine.hazard_tracker.known_lethal_features,
        )
        return MirroredConvergenceSolver.extract_mirror_targets(
            structural_goals=structural_goals,
            curr_grid=curr_grid,
            bg=bg,
            H=H,
            W=W,
            quiescent_targets=engine.quiescent_click_targets,
            visit_counts=engine.entity_visit_counts,
        )

    def _score_affordance_panels(
        self,
        entities: Sequence[Any],
        curr_grid: np.ndarray,
        bg: int,
        H: int,
        W: int,
        engine: AutonomousEpistemicEngine,
    ) -> list[tuple[int, int, float]]:
        from hbllm.hcir.world.cortex_motor import MotorCortexEffector
        from hbllm.hcir.world.cortex_perception import PerceptionEngine

        candidates: list[tuple[int, int, float]] = []
        panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
        for panel in panels:
            panel_coords = set(panel["item_coords"])
            has_effective_item = any(p in engine.effective_click_targets for p in panel_coords)

            for m_item in panel["minority_items"]:
                mr, mc = m_item.grid_pos
                if (
                    (mr, mc) not in engine.quiescent_click_targets
                    and 0 <= mr < H
                    and 0 <= mc < W
                    and not MotorCortexEffector.is_peripheral_frame_cell(mr, mc, H, W, curr_grid)
                ):
                    visit_count = engine.entity_visit_counts.get(f"click_{mr}_{mc}", 0)
                    last_step = engine.last_effector_target_step.get((mr, mc), -999)
                    delta_t = max(1, engine.step_counter - last_step)
                    refractory = 35.0 if delta_t == 1 else (35.0 / float(delta_t))
                    visit_damping = float(visit_count) * 220.0
                    score = 400.0 - refractory - visit_damping
                    candidates.append((mr, mc, score))

            for item in panel["items"]:
                ir, ic = item.grid_pos
                if (
                    (ir, ic) not in engine.quiescent_click_targets
                    and 0 <= ir < H
                    and 0 <= ic < W
                    and not MotorCortexEffector.is_peripheral_frame_cell(ir, ic, H, W, curr_grid)
                ):
                    visit_count = engine.entity_visit_counts.get(f"click_{ir}_{ic}", 0)
                    bonus = 260.0 if has_effective_item else 200.0
                    feat_bonus = 40.0 if item.feature_id in engine.effective_features else 0.0
                    last_step = engine.last_effector_target_step.get((ir, ic), -999)
                    delta_t = max(1, engine.step_counter - last_step)
                    is_goal_converging = (
                        (ir, ic) == engine.active_goal_converging_coord
                        and engine.consecutive_goal_converging_clicks < 12
                    )
                    is_active_momentum = (
                        (ir, ic) == engine.last_effective_click_coord
                        and engine.consecutive_effective_clicks < 6
                    )
                    if is_goal_converging:
                        momentum_bonus = 120.0
                        ior_penalty = 0.0
                    elif is_active_momentum:
                        momentum_bonus = 60.0
                        ior_penalty = 0.0
                    else:
                        momentum_bonus = 0.0
                        refractory = 35.0 if delta_t == 1 else (35.0 / float(delta_t))
                        visit_damping = min(30.0, float(visit_count) * 4.0)
                        ior_penalty = refractory + visit_damping
                    score = bonus + feat_bonus + momentum_bonus - ior_penalty
                    candidates.append((ir, ic, score))
        return candidates

    def _score_kinetic_repulsion_targets(
        self,
        entities: Sequence[Any],
        curr_grid: np.ndarray,
        bg: int,
        H: int,
        W: int,
        engine: AutonomousEpistemicEngine,
    ) -> list[tuple[int, int, float]]:
        last_action_coord = (
            (engine.last_action_data["x"], engine.last_action_data["y"])
            if engine.last_action_data and "x" in engine.last_action_data
            else None
        )
        return VortexAttractorSolver.score_repulsion_wave_targets(
            entities=entities,
            curr_grid=curr_grid,
            bg=bg,
            H=H,
            W=W,
            quiescent_targets=engine.quiescent_click_targets,
            visit_counts=engine.entity_visit_counts,
            last_action_coord=last_action_coord,
        )


class ARC3DeliberationPolicy:
    """Prefrontal deliberation policy for ARC-AGI-3 affordance panel arbitration."""

    @staticmethod
    def arbitrate_affordances(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        available_actions: Sequence[Any],
        known_av_feats: set[int],
    ) -> tuple[Any, dict[str, Any] | None] | None:
        from hbllm.hcir.world.cortex_assimilator import EpistemicPhase
        from hbllm.hcir.world.cortex_perception import PerceptionEngine

        has_displacement_actions = any(
            engine.is_displacement_action(a) for a in available_actions
        ) or any(
            not engine.is_spatial_effector(a) and a not in engine.action_dynamics
            for a in available_actions
        )
        spatial_effector_actions = [a for a in available_actions if engine.is_spatial_effector(a)]
        has_immobile_failures = bool(
            engine.immobile_entity_actions
            and len(engine.recent_actions) >= 2
            and any(a in engine.immobile_entity_actions for a in list(engine.recent_actions)[-2:])
        )
        is_stuck_or_looping = (
            engine.consecutive_stuck_steps >= 2
            or (
                engine.avatar_pos is not None
                and engine.recent_positions.count(engine.avatar_pos) >= 2
            )
            or (engine.avatar_pos is None and engine.step_counter > 4)
            or has_immobile_failures
        )
        has_aligned_optical_subgoal = False
        if engine.avatar_pos is not None and spatial_effector_actions:
            bg_eval = engine.estimate_background(curr_grid)
            sgs = PerceptionEngine.detect_structural_goals(
                curr_grid,
                bg=bg_eval,
                avatar_features=known_av_feats,
                avatar_feature=engine.avatar_feature,
            )
            for sg in sgs:
                if (
                    sg.get("type")
                    in (
                        "relational_alignment",
                        "optical_mirror_target",
                        "reflection_target",
                    )
                    and sg.get("is_midpoint")
                    and (
                        engine.avatar_feature is None
                        or sg.get("feature") == engine.avatar_feature
                        or sg.get("feature") in engine.avatar_features
                    )
                ):
                    mp = sg["position"]
                    if (
                        abs(mp[1] - engine.avatar_pos[1]) <= 1
                        and abs(mp[0] - engine.avatar_pos[0]) <= 2
                    ):
                        has_aligned_optical_subgoal = True
                        break

        discrete_transform_actions = [
            a
            for a in available_actions
            if not engine.is_displacement_action(a) and not engine.is_spatial_effector(a)
        ]

        if (
            (spatial_effector_actions or discrete_transform_actions)
            and (
                not has_displacement_actions
                or is_stuck_or_looping
                or (has_aligned_optical_subgoal and not engine.mental_plan)
            )
            and (engine.is_motor_grounded() or not has_displacement_actions)
        ):
            bg = engine.estimate_background(curr_grid)
            entities = engine.extract_entities(curr_grid, bg)
            panels = PerceptionEngine.detect_affordance_panels(entities, curr_grid, bg=bg)
            active_panel_target: tuple[int, int] | None = None
            for p in panels:
                unvisited_minority = [
                    m.grid_pos
                    for m in p["minority_items"]
                    if engine.entity_visit_counts.get(f"click_{m.grid_pos[0]}_{m.grid_pos[1]}", 0)
                    == 0
                    and m.grid_pos not in engine.quiescent_click_targets
                ]
                if unvisited_minority:
                    active_panel_target = unvisited_minority[0]
                    break

            if active_panel_target is None:
                optical_goals = [
                    g
                    for g in PerceptionEngine.detect_structural_goals(
                        curr_grid,
                        bg=bg,
                        avatar_features=known_av_feats,
                        avatar_feature=engine.avatar_feature,
                    )
                    if g.get("type")
                    in (
                        "relational_alignment",
                        "optical_mirror_target",
                        "reflection_target",
                    )
                    and g.get("position") is not None
                    and not g.get("is_midpoint")
                ]
                optical_goals.sort(key=lambda g: g.get("confidence", 0.5), reverse=True)
                for og in optical_goals:
                    mp = og["position"]
                    if engine.avatar_pos is not None:
                        dist_l1 = abs(mp[0] - engine.avatar_pos[0]) + abs(
                            mp[1] - engine.avatar_pos[1]
                        )
                        same_col = abs(mp[1] - engine.avatar_pos[1]) <= 3
                        same_row = abs(mp[0] - engine.avatar_pos[0]) <= 3
                        if dist_l1 < 8 or same_col or same_row:
                            continue
                    if engine.entity_visit_counts.get(f"click_{mp[0]}_{mp[1]}", 0) < 4:
                        active_panel_target = mp
                        break

            if active_panel_target is not None and spatial_effector_actions:
                chosen_action = spatial_effector_actions[0]
                tr, tc = active_panel_target
                engine.entity_visit_counts[f"click_{tr}_{tc}"] = (
                    engine.entity_visit_counts.get(f"click_{tr}_{tc}", 0) + 1
                )
                aff = engine.action_affordances.get(chosen_action)
                param_keys = aff.target_param_keys if aff else ("x", "y")
                chosen_data: dict[str, Any] = {}
                for k in param_keys:
                    if k in ("x", "col", "c", "column", "azimuth"):
                        chosen_data[k] = tc
                    elif k in ("y", "row", "r", "elevation", "distance"):
                        chosen_data[k] = tr
                    else:
                        chosen_data[k] = 0
                engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                return chosen_action, chosen_data
            elif (
                (
                    not spatial_effector_actions
                    or any(engine.is_focus_switch_action(a) for a in discrete_transform_actions)
                )
                and (
                    is_stuck_or_looping or (has_aligned_optical_subgoal and not engine.mental_plan)
                )
                and discrete_transform_actions
            ):
                focus_switches = [
                    a for a in discrete_transform_actions if engine.is_focus_switch_action(a)
                ]
                if focus_switches:
                    chosen_action = focus_switches[0]
                    engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    return chosen_action, None
                if not spatial_effector_actions:
                    undo_acts = getattr(engine, "undo_actions", set())
                    uninhibited_discrete = [
                        a
                        for a in discrete_transform_actions
                        if engine.inhibited_actions.get(a, 0) == 0 and a not in undo_acts
                    ]
                    chosen_action = (
                        uninhibited_discrete[0]
                        if uninhibited_discrete
                        else discrete_transform_actions[0]
                    )
                    engine.phase = EpistemicPhase.EPISTEMIC_EXPLORATION
                    return chosen_action, None

        return None


def install_arc3_solvers(engine: AutonomousEpistemicEngine) -> None:
    """Wire ARC-AGI-3 game solvers and archetype policies into an epistemic engine instance."""
    scorer = ARC3ArchetypeScorer()
    if hasattr(engine, "register_target_scorer"):
        engine.register_target_scorer(scorer)
    if hasattr(engine, "register_deliberation_policy"):
        engine.register_deliberation_policy(ARC3DeliberationPolicy())
    from hbllm.hcir.world.macro_actions import MacroActionDriverManager

    MacroActionDriverManager.register_synthesis_hook(
        MirroredConvergenceSolver.synthesize_macro_plan
    )
    logger.info("Installed ARC-AGI-3 modular game solvers and archetype scorers.")
