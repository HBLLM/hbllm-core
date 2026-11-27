"""Motor Cortex Effector Selection & Spatial Targeting."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.spatial_planner import SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.cortex_causal import ActionAffordance
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.engine_state import get_engine_state_attr
from hbllm.hcir.world.inferotemporal_segmentation import InferotemporalSegmentationEngine
from hbllm.hcir.world.motor.curiosity import EpistemicCuriosityExplorer

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


class EffectorTargetScorer(ABC):
    """Abstract interface for external or domain-specific target candidate scoring.

    Allows modular domain-specific or environment-adapter extensions to register
    candidate scoring strategies without coupling the core cognitive architecture.
    """

    @abstractmethod
    def score_targets(
        self,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        entities: Sequence[SpatialEntity],
        bg: int,
        H: int,
        W: int,
    ) -> list[tuple[int, int, float]]:
        """Score candidate coordinates for allocentric effector targeting."""
        ...


class MotorCortexEffector:
    """Allocentric spatial effector grounding and action affordance arbitration."""

    _registered_target_scorers: list[EffectorTargetScorer] = []

    @classmethod
    def register_target_scorer(cls, scorer: EffectorTargetScorer) -> None:
        """Register an external or modular domain target candidate scorer."""
        if scorer not in cls._registered_target_scorers:
            cls._registered_target_scorers.append(scorer)

    @classmethod
    def unregister_target_scorer(cls, scorer: EffectorTargetScorer) -> None:
        """Unregister an external target candidate scorer."""
        if scorer in cls._registered_target_scorers:
            cls._registered_target_scorers.remove(scorer)

    @classmethod
    def clear_target_scorers(cls) -> None:
        """Clear all registered target scorers."""
        cls._registered_target_scorers.clear()

    @staticmethod
    def _probe_looming_threat_evasion(
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        disp_actions: list[tuple[int, tuple[int, int]]],
        bg: int,
        bg_is_impassable_void: bool,
    ) -> int | None:
        """Forwarding alias to EpistemicCuriosityExplorer._probe_looming_threat_evasion."""
        return EpistemicCuriosityExplorer._probe_looming_threat_evasion(
            engine, curr_grid, disp_actions, bg, bg_is_impassable_void
        )

    @staticmethod
    def snap_effector_to_valid_targets(
        engine: AutonomousEpistemicEngine,
        action: Any,
        valid_targets: Sequence[Any],
        target_r: int,
        target_c: int,
    ) -> dict[str, Any] | None:
        """Snap a preferred actuation coordinate onto a declared valid-target lattice.

        When an adapter/affordance enumerates the only reachable actuation
        coordinates (``valid_targets`` payload dicts), probing must stay on
        that discrete manifold: arbitrary lattice coordinates are physical
        no-ops. This enumerates the declared targets and picks the
        un-quiescent, un-inhibited one nearest to ``(target_r, target_c)``,
        discounted by inhibition-of-return visit counts and habenular
        repulsion.

        Args:
            engine: Epistemic engine or journaled faculty view exposing
                ``quiescent_actuation_targets``, ``entity_visit_counts``,
                ``avatar_pos`` and ``working_memory``/``habenular_ior``.
            action: The spatial effector action being grounded.
            valid_targets: Declared target payloads (``dict``s carrying
                ``x``/``y`` or ``c``/``r`` style coordinates).
            target_r: Preferred target row (intent site / field centre).
            target_c: Preferred target column.

        Returns:
            The full declared payload dict for the chosen target (preserving
            any extra payload keys), or ``None`` when every enumerated target
            is already proven quiescent or inhibited.
        """
        best_score = float("inf")
        snapped_target: dict[str, Any] | None = None
        for vt in valid_targets:
            if isinstance(vt, dict):
                raw_vx = vt.get("x") if vt.get("x") is not None else vt.get("c", vt.get("col", 0))
                raw_vy = vt.get("y") if vt.get("y") is not None else vt.get("r", vt.get("row", 0))
                vx = int(raw_vx) if raw_vx is not None else 0
                vy = int(raw_vy) if raw_vy is not None else 0
                if (vy, vx) in engine.quiescent_actuation_targets:
                    continue
                if engine.working_memory.habenular_ior.is_action_inhibited(
                    engine.avatar_pos, action, action_data={"x": vx, "y": vy}
                ):
                    continue
                haben_pen = engine.working_memory.habenular_ior.get_repulsion_penalty(
                    engine.avatar_pos, action, action_data={"x": vx, "y": vy}
                )
                vis = engine.entity_visit_counts.get(f"actuate_{vy}_{vx}", 0)
                # Distance penalization with inhibition-of-return visit penalty and Habenular repulsion
                d = float(abs(vx - target_c) + abs(vy - target_r)) + float(vis) * 25.0 + haben_pen
                if d < best_score:
                    best_score = d
                    snapped_target = vt

        # If all un-quiescent candidates were exhausted, select least-visited un-quiescent valid target
        if snapped_target is None and valid_targets:
            least_visited_vt: dict[str, Any] | None = None
            min_vis = float("inf")
            for vt in valid_targets:
                if isinstance(vt, dict):
                    raw_vx = vt.get("x", vt.get("c", 0))
                    raw_vy = vt.get("y", vt.get("r", 0))
                    vx = int(raw_vx) if raw_vx is not None else 0
                    vy = int(raw_vy) if raw_vy is not None else 0
                    if (vy, vx) in engine.quiescent_actuation_targets:
                        continue
                    vis = engine.entity_visit_counts.get(f"actuate_{vy}_{vx}", 0)
                    if vis < min_vis:
                        min_vis = vis
                        least_visited_vt = vt
            snapped_target = least_visited_vt

        if snapped_target is None:
            return None

        # Update visit count for snapped coordinate
        svx = int(snapped_target.get("x", snapped_target.get("c", 0)))
        svy = int(snapped_target.get("y", snapped_target.get("r", 0)))
        engine.entity_visit_counts[f"actuate_{svy}_{svx}"] = (
            engine.entity_visit_counts.get(f"actuate_{svy}_{svx}", 0) + 1
        )
        return dict(snapped_target)

    @staticmethod
    def parse_action_spec(spec: Any) -> ActionAffordance:
        """Parse external action specification into an ActionAffordance descriptor."""
        if isinstance(spec, ActionAffordance):
            return spec
        if isinstance(spec, dict):
            aid = spec.get("action_id", spec.get("id", spec.get("name", 0)))
            if isinstance(aid, str) and aid.isdigit():
                aid = int(aid)
            name = str(spec.get("name", aid))
            params = spec.get("parameters") or spec.get("params") or {}
            param_keys = tuple(params.keys()) if isinstance(params, dict) else ("x", "y")
            is_point_kind = str(spec.get("kind", "")).lower() in (
                "point",
                "spatial",
                "coord",
                "coordinate",
            )
            spatial_coord_keys = (
                "x",
                "y",
                "col",
                "row",
                "lat",
                "lon",
                "azimuth",
                "elevation",
                "distance",
                "c",
                "r",
            )
            req_spatial = is_point_kind or bool(
                isinstance(params, dict) and any(k in params for k in spatial_coord_keys)
            )
            if req_spatial and not any(k in spatial_coord_keys for k in param_keys):
                param_keys = ("x", "y")
            name_upper = name.upper()
            is_disp = any(
                k in name_upper
                for k in (
                    "MOVE",
                    "UP",
                    "DOWN",
                    "LEFT",
                    "RIGHT",
                    "NORTH",
                    "SOUTH",
                    "EAST",
                    "WEST",
                    "STEP",
                    "WALK",
                )
            )
            meta: dict[str, Any] = {}
            if "metadata" in spec and isinstance(spec["metadata"], dict):
                meta.update(spec["metadata"])
            if "valid_targets" in spec:
                meta["valid_targets"] = spec["valid_targets"]
            return ActionAffordance(
                action_id=aid,
                name=name,
                is_displacement=is_disp,
                requires_spatial_target=req_spatial,
                target_param_keys=param_keys if param_keys else ("x", "y"),
                metadata=meta,
            )
        return ActionAffordance(action_id=spec, name=str(spec))

    @staticmethod
    def infer_motor_step_size(
        available_actions: Sequence[Any], action_dynamics: dict[Any, Any]
    ) -> int:
        """Largest calibrated displacement quantum (the world's 'cell' size)."""
        step = 1
        for act in available_actions:
            dyn = action_dynamics.get(act)
            if dyn is not None and dyn.is_displacement_action():
                dr, dc = dyn.get_displacement()
                step = max(step, abs(dr), abs(dc))
        return step

    @staticmethod
    def _is_exogenous_or_boundary(
        engine: Any, r: int, c: int, lattice: np.ndarray, bg: Any = 0
    ) -> bool:
        """True when ``(r, c)`` lies in an enclosing border, partitioned band, or exogenous locus."""
        return bool(
            engine.working_memory.is_exogenous_locus(
                (r, c), lattice=lattice, bg=bg, include_outer_edge=True
            )
        )

    @staticmethod
    def _role_prior_bias(rp_map: dict[int, dict[str, float]], feat_id: int) -> float:
        """Score a candidate feature using graded actuator/switch vs. hazard/barrier priors."""
        rp = rp_map.get(int(feat_id))
        if rp is None:
            return 0.0
        act_deg = max(
            float(rp.get("actuator", 0.0)),
            float(rp.get("switch", 0.0)),
            float(rp.get("receptacle", 0.0)) * 0.7,
            float(rp.get("goal", 0.0)) * 0.5,
        )
        avoid_deg = max(
            float(rp.get("hazard", 0.0)),
            float(rp.get("barrier", 0.0)),
        )
        return 80.0 * act_deg - 60.0 * avoid_deg

    @classmethod
    def ground_effector_action(
        cls,
        engine: AutonomousEpistemicEngine,
        curr_grid: np.ndarray,
        action: Any = None,
    ) -> dict[str, Any] | None:
        """Spatially ground an allocentric effector command onto salient affordances."""
        curr_grid = engine.normalize_sensory_input(curr_grid)
        H, W = curr_grid.shape
        aff = engine.action_affordances.get(action)
        param_keys = aff.target_param_keys if aff else ("x", "y")

        bg = (
            engine.bg_feature
            if get_engine_state_attr(engine, "bg_feature", None) is not None
            else engine.estimate_background(curr_grid)
        )
        entities = engine.extract_entities(curr_grid, bg)
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )

        actuation_candidates: list[tuple[int, int, float]] = []

        # 1. Registered external or adapter target scorers
        for scorer in cls._registered_target_scorers:
            try:
                actuation_candidates.extend(
                    scorer.score_targets(engine, curr_grid, entities, bg, H, W)
                )
            except Exception as e:
                logger.debug("Registered target scorer %s failed: %s", scorer, e)

        for scorer in get_engine_state_attr(engine, "target_scorers", []):
            try:
                actuation_candidates.extend(
                    scorer.score_targets(engine, curr_grid, entities, bg, H, W)
                )
            except Exception as e:
                logger.debug("Engine-local target scorer %s failed: %s", scorer, e)

        # 2. Relational Structural Affordances (Optical Targets, Receptacles, Alignment Sockets)
        rp_map = engine.working_memory.concept_role_priors
        structural_goals = PerceptionEngine.detect_structural_goals(
            curr_grid,
            bg=bg,
            avatar_features=av_feats,
            avatar_feature=engine.avatar_feature,
        )
        for sg in structural_goals:
            pos = sg.get("position")
            if pos is not None and isinstance(pos, (tuple, list)) and len(pos) >= 2:
                sr, sc = int(pos[0]), int(pos[1])
                feat = int(curr_grid[sr, sc]) if 0 <= sr < H and 0 <= sc < W else 0
                if (
                    (sr, sc) not in engine.quiescent_actuation_targets
                    and feat not in engine.quiescent_features
                    and not cls._is_exogenous_or_boundary(engine, sr, sc, curr_grid, bg)
                ):
                    vis = engine.entity_visit_counts.get(f"actuate_{sr}_{sc}", 0)
                    role_prior_bias = cls._role_prior_bias(rp_map, feat)
                    actuation_candidates.append(
                        (sr, sc, 180.0 + role_prior_bias - float(vis) * 15.0)
                    )

        # 3. Modal Manipulandum Destination Affordances:
        # If an external manipulandum is currently focused, prioritize destination sockets/receptacles
        if engine.working_memory.body_schema.has_manipulandum():
            man = engine.working_memory.body_schema.active_manipulandum
            if man is not None:
                for r_feat in engine.symbolic_theory.receptacle_features:
                    cells = np.argwhere(curr_grid == r_feat)
                    for cr, cc in cells[:8]:
                        if (int(cr), int(cc)) not in engine.quiescent_actuation_targets:
                            vis = engine.entity_visit_counts.get(f"actuate_{cr}_{cc}", 0)
                            actuation_candidates.append(
                                (int(cr), int(cc), 220.0 - float(vis) * 10.0)
                            )

        # 6. Salient entities if no panel candidates exist
        if not actuation_candidates:
            for e in entities:
                if (
                    e.feature_id == bg
                    or e.feature_id in av_feats
                    or e.area > 120
                    or e.feature_id in engine.quiescent_features
                ):
                    continue
                cr, cc = e.grid_pos
                if (
                    (
                        cr,
                        cc,
                    )
                    in engine.quiescent_actuation_targets
                    or MotorCortexEffector._is_exogenous_or_boundary(engine, cr, cc, curr_grid, bg)
                ):
                    continue
                visit_count = engine.entity_visit_counts.get(f"actuate_{cr}_{cc}", 0)
                saliency = 100.0 / math.log2(2 + max(1, e.area))
                affordance_bonus = 60.0 if (cr, cc) in engine.effective_actuation_targets else 0.0
                feat_bias = 40.0 if e.feature_id in engine.effective_features else 0.0
                role_prior_bias = cls._role_prior_bias(rp_map, int(e.feature_id))
                telic_bonus = (
                    150.0
                    if (
                        engine.symbolic_theory.telic_goal_type == "deplete"
                        and e.feature_id == engine.symbolic_theory.telic_target_feature
                    )
                    else 0.0
                )
                ior_penalty = float(visit_count) * 30.0 + (float(visit_count) ** 2) * 15.0
                if engine.working_memory.habenular_ior.is_action_inhibited(
                    engine.avatar_pos, action, action_data={"x": cc, "y": cr}
                ):
                    continue
                haben_pen = engine.working_memory.habenular_ior.get_repulsion_penalty(
                    engine.avatar_pos, action, action_data={"x": cc, "y": cr}
                )
                cand_score = (
                    saliency
                    + affordance_bonus
                    + feat_bias
                    + role_prior_bias
                    + telic_bonus
                    - ior_penalty
                    - haben_pen
                )
                actuation_candidates.append((cr, cc, cand_score))

        # 7. Inferotemporal Affordance Centroid Segmentation
        if not actuation_candidates:
            it_anchors = InferotemporalSegmentationEngine.extract_affordance_anchors(
                grid=curr_grid,
                background_feature=bg,
                avatar_features=av_feats,
                quiescent_coords=engine.quiescent_actuation_targets,
                effective_coords=engine.effective_actuation_targets,
                quiescent_features=engine.quiescent_features,
                effective_features=engine.effective_features,
                visit_counts=engine.entity_visit_counts,
            )
            for ar, ac, score in it_anchors:
                if (
                    (ar, ac) in engine.quiescent_actuation_targets
                    or (ar, ac) == engine.avatar_pos
                    or int(curr_grid[ar, ac]) in av_feats
                    or int(curr_grid[ar, ac]) in engine.quiescent_features
                    or cls._is_exogenous_or_boundary(engine, ar, ac, curr_grid, bg)
                ):
                    continue
                actuation_candidates.append((ar, ac, score))

        # 8. Saccadic Attention Fixations
        if not actuation_candidates:
            fixations = engine.saccadic_attention.extract_fixations(
                grid=curr_grid,
                prev_grid=engine.prev_grid,
                background_feature=bg,
                top_k=24,
            )
            for f in fixations:
                cr, cc = f.r, f.c
                feat = int(curr_grid[cr, cc])
                if (
                    (cr, cc) in engine.quiescent_actuation_targets
                    or feat == bg
                    or feat in av_feats
                    or feat in engine.quiescent_features
                    or (cr, cc) == engine.avatar_pos
                    or cls._is_exogenous_or_boundary(engine, cr, cc, curr_grid, bg)
                ):
                    continue
                visit_count = engine.entity_visit_counts.get(f"actuate_{cr}_{cc}", 0)
                affordance_bonus = 60.0 if (cr, cc) in engine.effective_actuation_targets else 0.0
                feat_bias = 40.0 if feat in engine.effective_features else 0.0
                cand_score = (
                    f.salience
                    + affordance_bonus
                    + feat_bias
                    - float(visit_count) * (0.1 if affordance_bonus > 0 else 0.35)
                )
                actuation_candidates.append((cr, cc, cand_score))

        # 9. Fallback: unvisited non-background cells
        if not actuation_candidates:
            step_sz = (
                getattr(
                    get_engine_state_attr(engine, "spatial_tracker", None),
                    "inferred_step_size",
                    None,
                )
                or cls.infer_motor_step_size(
                    list(engine.action_dynamics.keys()), engine.action_dynamics
                )
                or 1
            )
            seen_cand_cells: set[tuple[int, int]] = set()
            non_bg = np.argwhere(curr_grid != bg)
            for r, c in non_bg:
                cr, cc = int(r), int(c)
                if step_sz > 1:
                    # Quantize to the lattice cell centroid to hit multi-pixel blocks accurately
                    cr = (cr // step_sz) * step_sz + step_sz // 2
                    cc = (cc // step_sz) * step_sz + step_sz // 2
                    cr = min(max(0, cr), H - 1)
                    cc = min(max(0, cc), W - 1)
                if (cr, cc) in seen_cand_cells:
                    continue
                seen_cand_cells.add((cr, cc))
                feat = int(curr_grid[cr, cc])
                if (
                    (cr, cc) in engine.quiescent_actuation_targets
                    or feat in av_feats
                    or feat == bg
                    or feat == engine.bg_feature
                    or feat in engine.quiescent_features
                    or (cr, cc) == engine.avatar_pos
                    or cls._is_exogenous_or_boundary(engine, cr, cc, curr_grid, bg)
                ):
                    continue
                visit_count = engine.entity_visit_counts.get(f"actuate_{cr}_{cc}", 0)
                feat_bias = 40.0 if feat in engine.effective_features else 0.0
                cand_score = 20.0 + feat_bias - float(visit_count) * 2.0
                actuation_candidates.append((cr, cc, cand_score))

        # 10. Fallback: re-probe confirmed effective targets
        if not actuation_candidates and engine.effective_actuation_targets:
            eff_sorted = sorted(
                engine.effective_actuation_targets,
                key=lambda p: engine.entity_visit_counts.get(f"actuate_{p[0]}_{p[1]}", 0),
            )
            actuation_candidates.append((eff_sorted[0][0], eff_sorted[0][1], 50.0))

        # Quiescence Gating: Ensure candidates known to have 0 physical effect are excluded
        actuation_candidates = [
            c
            for c in actuation_candidates
            if (c[0], c[1]) not in engine.quiescent_actuation_targets
        ]

        # Parietal Effector Repertoire Grounding:
        # If the action affordance declares a discrete valid-target lattice, actuation is
        # restricted to those enumerated coordinates — arbitrary lattice positions are
        # physical no-ops. The salient candidate site only biases *which* declared target
        # is attempted first (nearest valid point); when no salient candidate exists the
        # lattice is enumerated systematically from the field centre, ordered by visit
        # counts and habenular repulsion inside the snap routine.
        valid_targets = aff.metadata.get("valid_targets") if aff else None
        has_valid_targets = isinstance(valid_targets, (list, tuple)) and bool(valid_targets)

        if actuation_candidates or has_valid_targets:
            if actuation_candidates:
                actuation_candidates.sort(key=lambda x: x[2], reverse=True)
                best_r, best_c, _ = actuation_candidates[0]
                engine.entity_visit_counts[f"actuate_{best_r}_{best_c}"] = (
                    engine.entity_visit_counts.get(f"actuate_{best_r}_{best_c}", 0) + 1
                )
                target_r, target_c = int(best_r), int(best_c)
                engine.last_effector_target_step[(target_r, target_c)] = engine.step_counter
            else:
                target_r, target_c = H // 2, W // 2

            if has_valid_targets and isinstance(valid_targets, (list, tuple)):
                snapped_target = cls.snap_effector_to_valid_targets(
                    engine, action, valid_targets, target_r, target_c
                )
                if snapped_target is not None:
                    return snapped_target

                # All discrete valid targets are exhausted and proven quiescent!
                if action is not None:
                    engine.inhibited_actions[action] = max(
                        engine.inhibited_actions.get(action, 0), 6
                    )
                return None

            coords: dict[str, Any] = {}
            for k in param_keys:
                if k in ("x", "col", "c", "column", "azimuth"):
                    coords[k] = target_c
                elif k in ("y", "row", "r", "elevation", "distance"):
                    coords[k] = target_r
                else:
                    coords[k] = 0
            return coords

        # Ventral Stream Invariant: When an embodied displacement avatar is present and all
        # targeted actuation candidates are exhausted / quiescent, inhibit targeted actuation
        # and return None so deliberation and exploration switch back to locomotion.
        if engine.avatar_pos is not None and engine.has_displacement_capabilities():
            if action is not None:
                engine.inhibited_actions[action] = max(engine.inhibited_actions.get(action, 0), 6)
            return None

        # Robust Spatial Grounding Invariant for un-embodied targeted actuation environments:
        # Effector actions must not return None on featureless boards when no avatar exists.
        best_r, best_c = H // 2, W // 2
        min_v = float("inf")
        step_stride = max(1, min(H, W) // 16)
        for r in range(1, H - 1, step_stride):
            for c in range(1, W - 1, step_stride):
                if cls._is_exogenous_or_boundary(engine, r, c, curr_grid, bg):
                    continue
                v = engine.entity_visit_counts.get(f"actuate_{r}_{c}", 0)
                feat = int(curr_grid[r, c])
                bonus = -20.0 if feat != bg else 0.0
                score = float(v) + bonus
                if score < min_v:
                    min_v = score
                    best_r, best_c = r, c

        engine.entity_visit_counts[f"actuate_{best_r}_{best_c}"] = (
            engine.entity_visit_counts.get(f"actuate_{best_r}_{best_c}", 0) + 1
        )
        engine.last_effector_target_step[(best_r, best_c)] = engine.step_counter
        coords_fallback: dict[str, Any] = {}
        for k in param_keys:
            if k in ("x", "col", "c", "column", "azimuth"):
                coords_fallback[k] = best_c
            elif k in ("y", "row", "r", "elevation", "distance"):
                coords_fallback[k] = best_r
            else:
                coords_fallback[k] = 0
        return coords_fallback

    @classmethod
    def to_causal_predicate(cls, view: Any = None) -> CausalPredicate:
        """Export motor cortex effector capability and runtime DOFs as a declarative causal predicate."""
        val = "ACTIVE"
        rel = "allocentric_spatial_grounding"
        if view is not None:
            dofs = len(getattr(view, "action_dynamics", {}) or {})
            rel = f"allocentric_spatial_grounding:dofs:{dofs}"
        return CausalPredicate(
            variable="motor_cortex_effector",
            operator="==",
            value=val,
            relation=rel,
            target_entity="cortex_motor",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering motor cortex effector."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="motor_cortex_effector",
                node_data={
                    "faculty": "motor_cortex_effector",
                    "status": "ACTIVE",
                },
            )
        ]
