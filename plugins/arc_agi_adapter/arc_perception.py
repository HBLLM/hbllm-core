from __future__ import annotations

"""ARC-AGI Sensory Perception Adapter.

Implements domain-specific perceptual lifting for ARC-AGI 2D grid environments.
Lifts raw 2D pixel matrices into formal HCIR PerceptionFrames (SpatialEntities,
topological passability maps, candidate exits, and avatar anchors).

All domain-specific visual heuristics (grid background estimation, perimeter
notch detection, ARC palette symmetries, corridor manifold continuity) live here
in the plugin layer, keeping the cognitive core 100% domain-agnostic.
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.macro_actions import ObjectSpec
from hbllm.hcir.world.percept import BasePerceptionAdapter, PerceptionFrame

logger = logging.getLogger(__name__)

# Canonical Semantic Keyword Mapping (Domain-Agnostic Vocabulary)
SEMANTIC_KEYWORD_ROLES: dict[str, str] = {
    # Avatar / Controllable Agent
    "avatar": "avatar",
    "player": "avatar",
    "agent": "avatar",
    "hero": "avatar",
    # Cargo / Movable Items
    "cargo": "cargo",
    "crate": "cargo",
    "box": "cargo",
    "block": "cargo",
    "gem": "cargo",
    "coin": "cargo",
    "key": "cargo",
    "item": "cargo",
    "pickup": "cargo",
    # Receptacles / Silhouette Sockets
    "receptacle": "receptacle",
    "silhouette": "receptacle",
    "socket": "receptacle",
    "slot": "receptacle",
    "basket": "receptacle",
    "bin": "receptacle",
    # Goals / Exits
    "goal": "goal",
    "exit": "goal",
    "target": "goal",
    "portal": "goal",
    "flag": "goal",
    "finish": "goal",
    # Manipulandum / Sliders
    "manipulandum": "manipulandum",
    "slider": "manipulandum",
    "shape": "manipulandum",
    "piece": "manipulandum",
    # Switches / Buttons / Actuators
    "switch": "switch",
    "button": "switch",
    "lever": "switch",
    "valve": "switch",
    "dial": "switch",
    "actuator": "switch",
    "click": "switch",
    "sys_click": "switch",
    # Dynamic Hazards / Enemies
    "hazard": "hazard",
    "enemy": "hazard",
    "threat": "hazard",
    "patrol": "hazard",
    "collider": "hazard",
    "spike": "hazard",
    "lava": "hazard",
    "danger": "hazard",
    # Barriers / Impassable Structures
    "barrier": "barrier",
    "wall": "barrier",
    "obstacle": "barrier",
    "border": "barrier",
    "blocker": "barrier",
}


def classify_tag_semantic_role(tags: Sequence[str] | None) -> str | None:
    """Classify semantic role from string tags via canonical keyword & stem matching.

    Decoupled from hardcoded environment hashes. Resolves any tag containing
    domain-general stems ('portal_gate' -> 'goal', 'btn_actuator' -> 'switch').
    """
    if not tags:
        return None
    for raw_t in tags:
        t = str(raw_t).strip().lower()
        if not t:
            continue
        if t in SEMANTIC_KEYWORD_ROLES:
            return SEMANTIC_KEYWORD_ROLES[t]
        for kw, role in SEMANTIC_KEYWORD_ROLES.items():
            if kw in t:
                return role
    return None


def _is_truthy_flag(val: Any) -> bool:
    """Check boolean flag safely without being fooled by unconfigured MagicMocks."""
    if val is None or val is False:
        return False
    if hasattr(val, "_mock_name"):
        return False
    return bool(val)


class ARCPerceptionAdapter(BasePerceptionAdapter):
    """Domain Sensory Adapter lifting ARC-AGI 2D grids into standardized PerceptionFrames."""

    def __init__(self) -> None:
        pass

    def lift_percept(
        self,
        raw_observation: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> PerceptionFrame:
        """Lift a raw ARC observation into a standardized PerceptionFrame."""
        # 1. Normalize sensory array
        grid: np.ndarray
        if isinstance(raw_observation, np.ndarray):
            grid = raw_observation[-1] if raw_observation.ndim == 3 else raw_observation
        elif hasattr(raw_observation, "frame") and getattr(raw_observation, "frame"):
            f = raw_observation.frame
            grid = f[-1] if isinstance(f, np.ndarray) and f.ndim == 3 else np.asarray(f, dtype=int)
        elif hasattr(raw_observation, "raw_data"):
            rd = raw_observation.raw_data
            grid = (
                rd[-1] if isinstance(rd, np.ndarray) and rd.ndim == 3 else np.asarray(rd, dtype=int)
            )
        else:
            try:
                grid = np.asarray(raw_observation, dtype=int)
                if grid.ndim == 3 and len(grid) > 0:
                    grid = grid[-1]
            except Exception:
                grid = np.zeros((16, 16), dtype=int)

        H, W = grid.shape

        # 2. Extract context attributes safely
        bg = PerceptionEngine.estimate_background(grid)
        av_pos = getattr(context, "avatar_pos", None) or kwargs.get("avatar_pos")
        av_feature = getattr(context, "avatar_feature", None) or kwargs.get("avatar_feature")
        av_feats = set(
            getattr(context, "avatar_features", set()) or kwargs.get("avatar_features", set())
        )
        if av_feature is not None:
            av_feats.add(av_feature)

        symbolic_theory = getattr(context, "symbolic_theory", None)
        barrier_features = set()
        goal_features = set()
        walkable_features = set()
        if symbolic_theory is not None:
            barrier_features.update(symbolic_theory.barrier_features)
            goal_features.update(symbolic_theory.goal_features)
            walkable_features.update(symbolic_theory.walkable_features)

        learned_barriers = set(
            getattr(context, "learned_barriers", set()) or kwargs.get("learned_barriers", set())
        )
        learned_goals = set(
            getattr(context, "learned_goal_positions", set())
            or kwargs.get("learned_goal_positions", set())
        )
        known_lethals = set(
            getattr(context, "current_level_lethal_positions", set())
            or kwargs.get("lethal_cells", set())
        )
        if hasattr(context, "hazard_tracker") and hasattr(
            context.hazard_tracker, "static_lethal_positions"
        ):
            known_lethals.update(context.hazard_tracker.static_lethal_positions)

        known_lethal_features = set()
        if hasattr(context, "hazard_tracker") and hasattr(
            context.hazard_tracker, "known_lethal_features"
        ):
            known_lethal_features.update(context.hazard_tracker.known_lethal_features)

        # 3. Detect structural exit and candidate goals FIRST to seed goal spatial priors
        candidate_goals: set[tuple[int, int]] = set(learned_goals)
        structural_goals = PerceptionEngine.detect_structural_goals(
            grid=grid,
            bg=bg,
            avatar_features=av_feats,
            avatar_feature=av_feature,
            barrier_features=barrier_features,
            known_lethal_features=known_lethal_features,
            avatar_pos=av_pos,
        )
        for sg in structural_goals:
            pos = sg.get("position")
            feat = sg.get("feature")
            if pos is not None and av_pos is not None:
                if pos == av_pos or (abs(pos[0] - av_pos[0]) <= 2 and abs(pos[1] - av_pos[1]) <= 2):
                    continue
            if feat is not None and feat in av_feats:
                continue
            if sg.get("type") in ("exit_marker", "gateway_notch", "target_zone"):
                for cp in sg.get("cells", []):
                    if av_pos is None or (
                        cp != av_pos and (abs(cp[0] - av_pos[0]) > 2 or abs(cp[1] - av_pos[1]) > 2)
                    ):
                        candidate_goals.add(cp)
                if pos and (
                    av_pos is None
                    or (
                        pos != av_pos
                        and (abs(pos[0] - av_pos[0]) > 2 or abs(pos[1] - av_pos[1]) > 2)
                    )
                ):
                    candidate_goals.add(pos)
        if (
            context is not None
            and hasattr(context, "learned_goal_positions")
            and av_pos is not None
        ):
            context.learned_goal_positions.discard(av_pos)

        # 4. Extract segmented spatial entities informed by candidate goals
        entities = PerceptionEngine.extract_entities(
            grid=grid,
            bg=bg,
            symbolic_theory=symbolic_theory,
            avatar_features=av_feats,
            state_mutations=getattr(context, "state_mutations", None),
            learned_barriers=learned_barriers,
            learned_goal_positions=candidate_goals,
            avatar_pos=av_pos,
        )

        # 5. Synthesize Passable Topology Map
        # 1 = traversable/walkable, 0 = impassable barrier/lethal
        passable_topology = np.ones((H, W), dtype=bool)
        static_barriers: set[tuple[int, int]] = set(learned_barriers)

        # Mark static barrier features
        for r in range(H):
            for c in range(W):
                val = int(grid[r, c])
                if val in barrier_features or (symbolic_theory and symbolic_theory.is_barrier(val)):
                    static_barriers.add((r, c))

        # Corridor Manifold Continuity:
        # If foreground features are explicitly walkable and the background is NOT walkable,
        # treat background as impassable void (e.g. tu93 maze walls)
        walkable_and_safe = set(walkable_features) | set(
            getattr(context, "verified_safe_features", set()) or set()
        )
        if len(walkable_and_safe) > 0 and bg not in walkable_and_safe:
            for r in range(H):
                for c in range(W):
                    if int(grid[r, c]) == bg and (r, c) not in candidate_goals:
                        static_barriers.add((r, c))

        for r, c in static_barriers:
            if 0 <= r < H and 0 <= c < W:
                passable_topology[r, c] = False

        for r, c in known_lethals:
            if 0 <= r < H and 0 <= c < W:
                passable_topology[r, c] = False

        # Compute observation difference if previous grid is available
        diff = None
        prev_grid = getattr(context, "prev_grid", None)
        if prev_grid is None:
            prev_grid = kwargs.get("prev_grid")
        if (
            prev_grid is not None
            and isinstance(prev_grid, np.ndarray)
            and prev_grid.shape == grid.shape
        ):
            diff = PerceptionEngine.compute_frame_diff(prev_grid, grid)

        return PerceptionFrame(
            entities=entities,
            passable_topology=passable_topology,
            avatar_pos=av_pos,
            avatar_features=av_feats,
            candidate_goals=sorted(list(candidate_goals)),
            static_barriers=static_barriers,
            lethal_cells=known_lethals,
            shape=(H, W),
            diff=diff,
            metadata={"bg": bg, "grid": grid},
        )


def _attach_object_macro_tasks(
    role: str,
    pos: tuple[int, int],
    feat: int = 0,
    obj_id: str = "",
    props: dict[str, Any] | None = None,
) -> list[Any]:
    """Break down the smallest atomic macro-tasks available for an object."""
    from hbllm.hcir.world.macro_actions import (
        DepositCargo,
        ExploreObject,
        InteractWith,
        NavigateTo,
        PickUpCargo,
        PushObject,
        SwitchFocus,
        TranslateObject,
        UseToolOnBarrier,
    )

    tasks: list[Any] = []
    props = props or {}

    if "cargo" in role or "box" in role or "key" in role or "crate" in role or "item" in role:
        # Box / Cargo: PickUp is the primary atomic macro-task
        tasks.append(
            PickUpCargo(cargo_pos=pos, cargo_feature=feat, description=f"Pick up {role} at {pos}")
        )
        tasks.append(PushObject(target_pos=pos, description=f"Push {role} at {pos}"))
    elif (
        "switch" in role
        or "button" in role
        or "lever" in role
        or "valve" in role
        or "actuator" in role
    ):
        try:
            from plugins.arc_agi_adapter.arc_macro_actions import ArcClickCell, ToggleSwitch

            tasks.append(
                ToggleSwitch(
                    switch_pos=pos,
                    switch_id=obj_id,
                    switch_feature=feat,
                    description=f"Toggle switch at {pos}",
                )
            )
            tasks.append(ArcClickCell(cell_coord=pos, description=f"Click cell at {pos}"))
        except Exception:
            pass
        tasks.append(InteractWith(target_coord=pos, description=f"Interact with {role} at {pos}"))
    elif "goal" in role or "exit" in role:
        tasks.append(NavigateTo(destination=pos, description=f"Navigate to goal at {pos}"))
    elif "receptacle" in role or "door" in role or "silhouette" in role or "container" in role:
        tasks.append(
            DepositCargo(
                receptacle_pos=pos,
                cargo_feature=feat,
                description=f"Deposit into receptacle at {pos}",
            )
        )
        tasks.append(
            UseToolOnBarrier(
                barrier_pos=pos, tool_feature=feat, description=f"Unlock barrier at {pos}"
            )
        )
    elif "manipulandum" in role or "shape" in role or "slider" in role:
        tasks.append(TranslateObject(target_pos=pos, description=f"Translate {role} at {pos}"))
        tasks.append(SwitchFocus(switch_action=5, description="Switch active manipulandum focus"))
    elif "barrier" in role or "obstacle" in role:
        tasks.append(
            UseToolOnBarrier(
                barrier_pos=pos, tool_feature=feat, description=f"Use tool on barrier at {pos}"
            )
        )
    else:
        tasks.append(
            ExploreObject(
                target_pos=pos, target_feature=feat, description=f"Explore entity at {pos}"
            )
        )
        try:
            from plugins.arc_agi_adapter.arc_macro_actions import ArcClickCell

            tasks.append(ArcClickCell(cell_coord=pos, description=f"Click cell at {pos}"))
        except Exception:
            pass

    return tasks


def extract_arc_objects(
    grid: Any,
    engine: Any = None,
    game_obj: Any = None,
    adapter: ARCPerceptionAdapter | None = None,
) -> list[ObjectSpec]:
    """Domain perception lifting: transforms raw ARC-AGI 2D matrices into formal ObjectSpec instances.

    Lifts:
    1. Spatial entities segmented by PerceptionEngine (doors, keys, switches, barriers, hazards).
    2. Candidate exit and gateway goals detected structurally.
    3. Learned goals and receptacle memory anchors from engine.
    4. Sprite tag metadata when running within ARC-AGI arcade environments.

    Conforms strictly to AGENTS.md Rule 1.7 (domain-agnostic core): all ARC-specific visual
    lifting, palette symmetries, and sprite tags remain decoupled in the plugin layer.
    """
    if isinstance(grid, np.ndarray):
        g = grid[-1] if grid.ndim == 3 else grid
    elif hasattr(grid, "frame") and getattr(grid, "frame") is not None:
        f = grid.frame
        g = f[-1] if isinstance(f, np.ndarray) and f.ndim == 3 else np.asarray(f, dtype=int)
    elif hasattr(grid, "grid") and getattr(grid, "grid") is not None:
        g = np.asarray(grid.grid, dtype=int)
    else:
        try:
            g = np.asarray(grid, dtype=int)
            if g.ndim == 3 and len(g) > 0:
                g = g[-1]
        except Exception:
            return []

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return []

    H, W = g.shape
    seen_positions: set[tuple[int, int]] = set()
    objects: list[ObjectSpec] = []

    # Identify avatar features/pos to avoid lifting avatar as a target object
    av_pos = getattr(engine, "avatar_pos", None)
    av_feat = getattr(engine, "avatar_feature", None)
    av_feats = set(getattr(engine, "avatar_features", set()) or set())
    if av_feat is not None:
        av_feats.add(av_feat)

    # Ingest from symbolic theory if available
    symbolic_theory = getattr(engine, "symbolic_theory", None)
    goal_feats: set[int] = set()
    cargo_feats: set[int] = set()
    rec_feats: set[int] = set()
    barrier_feats: set[int] = set()
    actuator_feats: set[int] = set()

    if symbolic_theory is not None:
        goal_feats.update(getattr(symbolic_theory, "goal_features", set()))
        cargo_feats.update(getattr(symbolic_theory, "cargo_features", set()))
        rec_feats.update(getattr(symbolic_theory, "receptacle_features", set()))
        barrier_feats.update(getattr(symbolic_theory, "barrier_features", set()))
        actuator_feats.update(getattr(symbolic_theory, "actuator_features", set()))

    # 1. Ingest arcade sprite tags if available in environment game_obj FIRST
    # (Grounds avatar pos and key landmarks before general matrix lifting)
    cur_lvl = getattr(game_obj, "current_level", None) if game_obj is not None else None
    if cur_lvl is not None:
        try:
            cam = getattr(game_obj, "camera", None)
            cam_w = getattr(cam, "width", None) if cam else None
            cam_h = getattr(cam, "height", None) if cam else None
            if (
                isinstance(cam_w, (int, float))
                and isinstance(cam_h, (int, float))
                and cam_w > 0
                and cam_h > 0
            ):
                scale_x = int(64 / cam_w)
                scale_y = int(64 / cam_h)
                scale = max(1, min(scale_x, scale_y))
                x_pad = max(0, int((64 - (cam_w * scale)) / 2))
                y_pad = max(0, int((64 - (cam_h * scale)) / 2))
            else:
                scale = 1
                x_pad = 0
                y_pad = 0

            # Ingest arcade sprites directly from level representation
            candidate_sprites: list[Any] = []
            if hasattr(cur_lvl, "get_sprites") and callable(cur_lvl.get_sprites):
                res = cur_lvl.get_sprites()
                if isinstance(res, (list, tuple, set)):
                    candidate_sprites.extend(res)
            elif hasattr(cur_lvl, "sprites"):
                spr = getattr(cur_lvl, "sprites", None)
                if isinstance(spr, (list, tuple, set)):
                    candidate_sprites.extend(spr)
                elif isinstance(spr, dict):
                    candidate_sprites.extend(spr.values())

            for sp in candidate_sprites:
                s_name = getattr(sp, "name", "").lower()
                s_tags = [str(t).lower() for t in getattr(sp, "tags", [])]
                w = getattr(sp, "width", 1)
                h = getattr(sp, "height", 1)
                disp_x = int(sp.x * scale + x_pad + (w * scale) // 2)
                disp_y = int(sp.y * scale + y_pad + (h * scale) // 2)
                s_pos = (disp_y, disp_x)

                if (
                    not (0 <= disp_y < H and 0 <= disp_x < W)
                    or s_pos in seen_positions
                    or s_pos == av_pos
                ):
                    continue

                feat_val = int(g[disp_y, disp_x]) if 0 <= disp_y < H and 0 <= disp_x < W else 0
                s_feat = feat_val
                if hasattr(sp, "pixels") and sp.pixels is not None:
                    try:
                        pix_arr = np.asarray(sp.pixels)
                        non_neg = pix_arr[pix_arr >= 0]
                        if len(non_neg) > 0:
                            s_feat = int(non_neg[0])
                    except Exception:
                        pass

                # 1. Semantic Tag/Name Matching (Canonical English Vocabulary)
                role: str = (
                    classify_tag_semantic_role(s_tags)
                    or classify_tag_semantic_role([s_name])
                    or "unknown"
                )

                # 2. Visual Cortex & Topological Affordance Invariants (Tag-Agnostic Ground Truth)
                if role == "unknown":
                    # (a) Physical Avatar Invariant (layer >= 1, compact, non-background)
                    if (
                        engine is not None
                        and getattr(sp, "layer", 0) >= 1
                        and 1 <= w <= 12
                        and 1 <= h <= 12
                        and s_name != "background"
                        and (
                            engine.avatar_pos is None
                            or s_feat in getattr(engine, "avatar_features", set())
                            or s_feat == getattr(engine, "avatar_feature", None)
                            or not getattr(engine, "avatar_features", set())
                        )
                    ):
                        role = "avatar"
                    elif (
                        (
                            engine is not None
                            and s_pos in getattr(engine, "learned_goal_positions", set())
                        )
                        or _is_truthy_flag(getattr(sp, "is_goal", False))
                        or _is_truthy_flag(getattr(sp, "is_target", False))
                    ):
                        role = "goal"
                    elif getattr(sp, "layer", 0) <= 0 and (
                        _is_truthy_flag(getattr(sp, "is_collidable", False))
                        or _is_truthy_flag(getattr(sp, "blocking", False))
                        or _is_truthy_flag(getattr(sp, "solid", False))
                        or (
                            engine is not None
                            and s_pos in getattr(engine, "learned_barriers", set())
                        )
                    ):
                        role = "barrier"
                    elif (
                        _is_truthy_flag(getattr(sp, "clickable", False))
                        or _is_truthy_flag(getattr(sp, "interactive", False))
                        or _is_truthy_flag(getattr(sp, "is_button", False))
                    ):
                        role = "switch"
                    elif (
                        _is_truthy_flag(getattr(sp, "lethal", False))
                        or _is_truthy_flag(getattr(sp, "is_hazard", False))
                        or (
                            engine is not None
                            and hasattr(engine, "hazard_tracker")
                            and s_pos
                            in getattr(engine.hazard_tracker, "static_lethal_positions", set())
                        )
                    ):
                        role = "hazard"
                    elif (
                        _is_truthy_flag(getattr(sp, "movable", False))
                        or _is_truthy_flag(getattr(sp, "draggable", False))
                        or _is_truthy_flag(getattr(sp, "pushable", False))
                        or _is_truthy_flag(getattr(sp, "collectible", False))
                    ):
                        role = "cargo"
                    elif _is_truthy_flag(getattr(sp, "is_receptacle", False)):
                        role = "receptacle"
                    elif symbolic_theory is not None:
                        if s_feat in getattr(symbolic_theory, "goal_features", set()):
                            role = "goal"
                        elif s_feat in getattr(symbolic_theory, "barrier_features", set()):
                            role = "barrier"
                        elif s_feat in getattr(symbolic_theory, "cargo_features", set()):
                            role = "cargo"
                        elif s_feat in getattr(symbolic_theory, "receptacle_features", set()):
                            role = "receptacle"
                        elif s_feat in getattr(symbolic_theory, "actuator_features", set()):
                            role = "switch"

                if role == "avatar":
                    is_active_avatar = True
                    if hasattr(sp, "pixels") and sp.pixels is not None:
                        try:
                            parr = np.asarray(sp.pixels)
                            if parr.ndim == 2 and parr.shape[0] >= 3 and parr.shape[1] >= 3:
                                mid_r, mid_c = parr.shape[0] // 2, parr.shape[1] // 2
                                if (
                                    parr[mid_r, mid_c] == parr[0, 0]
                                    and engine is not None
                                    and engine.avatar_pos is not None
                                    and s_feat not in getattr(engine, "avatar_features", set())
                                ):
                                    is_active_avatar = False
                        except Exception:
                            pass
                    if is_active_avatar:
                        if engine is not None:
                            engine.avatar_pos = s_pos
                            engine.avatar_feature = s_feat
                            if hasattr(engine, "avatar_features"):
                                engine.avatar_features.add(s_feat)
                        av_pos = s_pos
                        av_feat = s_feat
                        av_feats.add(s_feat)
                    seen_positions.add(s_pos)
                    continue
                elif role == "barrier":
                    seen_positions.add(s_pos)
                    if engine is not None:
                        engine.learned_barriers.add(s_pos)
                        if s_feat:
                            engine.symbolic_theory.barrier_features.add(s_feat)
                    continue

                # 1. Interactive clickable switches & buttons & actuators
                if role == "switch":
                    seen_positions.add(s_pos)
                    s_id = f"switch_sprite_{disp_y}_{disp_x}"
                    s_props = {
                        "source": "sprite_actuator",
                        "click_coord": s_pos,
                        "click_data": {"x": disp_x, "y": disp_y},
                        "sprite_name": s_name,
                    }
                    objects.append(
                        ObjectSpec(
                            id=s_id,
                            pos=s_pos,
                            feature_id=s_feat,
                            role="switch",
                            properties=s_props,
                            macro_tasks=_attach_object_macro_tasks(
                                role, s_pos, s_feat, s_id, s_props
                            ),
                        )
                    )
                # 2. Exits, target markers, and goals
                elif role == "goal":
                    seen_positions.add(s_pos)
                    g_id = f"goal_sprite_{disp_y}_{disp_x}"
                    g_props = {"source": "sprite_goal", "sprite_name": s_name}
                    objects.append(
                        ObjectSpec(
                            id=g_id,
                            pos=s_pos,
                            feature_id=s_feat,
                            role="goal",
                            properties=g_props,
                            macro_tasks=_attach_object_macro_tasks(
                                role, s_pos, s_feat, g_id, g_props
                            ),
                        )
                    )
                # 3. Dynamic hazards, enemies, patrollers
                elif role == "hazard":
                    seen_positions.add(s_pos)
                    h_id = f"hazard_sprite_{disp_y}_{disp_x}"
                    h_props = {"source": "sprite_hazard", "sprite_name": s_name}
                    objects.append(
                        ObjectSpec(
                            id=h_id,
                            pos=s_pos,
                            feature_id=s_feat,
                            role="hazard",
                            properties=h_props,
                            macro_tasks=_attach_object_macro_tasks(
                                role, s_pos, s_feat, h_id, h_props
                            ),
                        )
                    )
                    if engine is not None:
                        if hasattr(engine, "hazard_tracker") and s_feat != 0:
                            engine.hazard_tracker.register_lethal_feature(s_feat)
                        if hasattr(engine, "mobile_threat_features") and s_feat != 0:
                            engine.mobile_threat_features.add(s_feat)
                # 4. Movable manipulandums, shapes, sliders
                elif role == "manipulandum":
                    seen_positions.add(s_pos)
                    if engine is not None and engine.avatar_pos is None:
                        engine.avatar_pos = s_pos
                        engine.avatar_feature = s_feat
                        if hasattr(engine, "avatar_features"):
                            engine.avatar_features.add(s_feat)
                    m_id = f"shape_sprite_{disp_y}_{disp_x}"
                    m_props = {
                        "source": "sprite_manipulandum",
                        "width": w,
                        "height": h,
                        "focus_switch_actions": {5},
                    }
                    objects.append(
                        ObjectSpec(
                            id=m_id,
                            pos=s_pos,
                            feature_id=s_feat,
                            role="manipulandum",
                            properties=m_props,
                            macro_tasks=_attach_object_macro_tasks(
                                role, s_pos, s_feat, m_id, m_props
                            ),
                        )
                    )
                # 5. Receptacles, silhouettes, fitting zones, baskets
                elif role == "receptacle":
                    seen_positions.add(s_pos)
                    r_id = f"silhouette_sprite_{disp_y}_{disp_x}"
                    r_props = {"source": "sprite_receptacle", "sprite_name": s_name}
                    objects.append(
                        ObjectSpec(
                            id=r_id,
                            pos=s_pos,
                            feature_id=s_feat,
                            role="receptacle",
                            properties=r_props,
                            macro_tasks=_attach_object_macro_tasks(
                                role, s_pos, s_feat, r_id, r_props
                            ),
                        )
                    )
                # 6. Cargo, crates, gems, keys, pickups
                elif role == "cargo":
                    seen_positions.add(s_pos)
                    c_id = f"cargo_sprite_{disp_y}_{disp_x}"
                    c_props = {"source": "sprite_cargo", "sprite_name": s_name}
                    objects.append(
                        ObjectSpec(
                            id=c_id,
                            pos=s_pos,
                            feature_id=s_feat,
                            role="cargo",
                            properties=c_props,
                            macro_tasks=_attach_object_macro_tasks(
                                role, s_pos, s_feat, c_id, c_props
                            ),
                        )
                    )
                # 7. Stations / tools
                elif role == "tool":
                    seen_positions.add(s_pos)
                    t_id = f"tool_station_{disp_y}_{disp_x}"
                    t_props = {"source": "sprite_tool", "sprite_name": s_name}
                    objects.append(
                        ObjectSpec(
                            id=t_id,
                            pos=s_pos,
                            feature_id=s_feat,
                            role="tool",
                            properties=t_props,
                            macro_tasks=_attach_object_macro_tasks(
                                role, s_pos, s_feat, t_id, t_props
                            ),
                        )
                    )
                else:
                    # Unclassified sprite in unseen game:
                    # Do NOT flood seen_positions with a dummy "entity" object.
                    # Leave locus unmasked so the Ventral Stream visual segmenter
                    # (frame.entities) has primary jurisdiction over spatial representation.
                    pass
        except Exception as exc:
            logger.debug("Failed extracting sprites from cur_lvl: %s", exc)

    # 1B. Lift standardized PerceptionFrame with pre-grounded avatar context
    adapter = adapter or ARCPerceptionAdapter()
    try:
        frame = adapter.lift_percept(g, context=engine)
    except Exception as exc:
        logger.debug("ARCPerceptionAdapter.lift_percept failed in extract_arc_objects: %s", exc)
        frame = None

    if frame is not None:
        if frame.avatar_pos is not None and av_pos is None:
            av_pos = frame.avatar_pos
        if frame.avatar_features:
            av_feats.update(frame.avatar_features)

    # 2. Convert segmented SpatialEntity instances into ObjectSpec
    if frame is not None and frame.entities:
        for ent in frame.entities:
            # Skip avatar
            if ent.role == EntityRole.AGENT:
                continue
            pos = (int(ent.grid_pos[0]), int(ent.grid_pos[1]))
            if (
                pos == av_pos
                or (
                    av_pos is not None
                    and abs(pos[0] - av_pos[0]) <= 2
                    and abs(pos[1] - av_pos[1]) <= 2
                )
                or pos in seen_positions
            ):
                continue

            feat = int(
                ent.properties.get(
                    "feature_id", g[pos[0], pos[1]] if 0 <= pos[0] < H and 0 <= pos[1] < W else 0
                )
            )
            if feat in av_feats:
                continue

            # Determine role: Prioritize learned symbolic theory ontology over heuristic visual defaults
            role = "unknown"
            if pos in getattr(engine, "learned_goal_positions", set()) or feat in goal_feats:
                role = "goal"
            elif feat in cargo_feats:
                role = "cargo"
            elif pos in getattr(engine, "learned_receptacle_positions", set()) or feat in rec_feats:
                role = "receptacle"
            elif feat in actuator_feats:
                role = "switch"
            elif pos in frame.static_barriers or feat in barrier_feats:
                role = "barrier"
            elif pos in frame.lethal_cells:
                role = "hazard"
            elif ent.role == EntityRole.GOAL:
                role = "goal"
            elif ent.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE):
                role = "cargo"
            elif ent.role in (EntityRole.RECEPTACLE, EntityRole.PORTAL):
                role = "receptacle"
            elif ent.role == EntityRole.ACTUATOR:
                role = "switch"
            elif ent.role == EntityRole.OBSTACLE:
                role = "barrier"
            elif ent.role == EntityRole.DYNAMIC_HAZARD:
                role = "hazard"
            else:
                role = str(ent.role.value if hasattr(ent.role, "value") else ent.role).lower()

            if role == "unknown":
                if pos in getattr(engine, "learned_goal_positions", set()):
                    role = "goal"
                elif feat == getattr(frame, "bg", None) or (
                    engine is not None
                    and feat in getattr(engine.symbolic_theory, "walkable_features", set())
                ):
                    role = "floor"
                elif (
                    ent.role in (EntityRole.MANIPULABLE, EntityRole.RESOURCE) or feat in cargo_feats
                ):
                    role = "cargo"
                elif ent.role == EntityRole.ACTUATOR or feat in actuator_feats:
                    role = "switch"
                else:
                    role = "entity"
            seen_positions.add(pos)
            ent_id = ent.id or f"{role}_{pos[0]}_{pos[1]}"
            ent_props = dict(ent.properties)
            objects.append(
                ObjectSpec(
                    id=ent_id,
                    pos=pos,
                    feature_id=feat,
                    role=role,
                    properties=ent_props,
                    macro_tasks=_attach_object_macro_tasks(role, pos, feat, ent_id, ent_props),
                )
            )

    # 3. Ingest Candidate Goals / Exits from PerceptionFrame
    if frame is not None and frame.candidate_goals:
        for gp in frame.candidate_goals:
            pos = (int(gp[0]), int(gp[1]))
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in seen_positions and pos != av_pos:
                seen_positions.add(pos)
                cg_id = f"exit_goal_{pos[0]}_{pos[1]}"
                cg_feat = int(g[pos[0], pos[1]])
                cg_props = {"source": "candidate_goal"}
                objects.append(
                    ObjectSpec(
                        id=cg_id,
                        pos=pos,
                        feature_id=cg_feat,
                        role="goal",
                        properties=cg_props,
                        macro_tasks=_attach_object_macro_tasks(
                            "goal", pos, cg_feat, cg_id, cg_props
                        ),
                    )
                )

    # 4. Ingest Learned Goal Positions from engine
    if engine is not None and hasattr(engine, "learned_goal_positions"):
        for gp in engine.learned_goal_positions:
            pos = (int(gp[0]), int(gp[1]))
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in seen_positions and pos != av_pos:
                seen_positions.add(pos)
                lg_id = f"learned_goal_{pos[0]}_{pos[1]}"
                lg_feat = int(g[pos[0], pos[1]])
                lg_props = {"source": "learned_goal"}
                objects.append(
                    ObjectSpec(
                        id=lg_id,
                        pos=pos,
                        feature_id=lg_feat,
                        role="goal",
                        properties=lg_props,
                        macro_tasks=_attach_object_macro_tasks(
                            "goal", pos, lg_feat, lg_id, lg_props
                        ),
                    )
                )

    # 5. Ingest Learned Receptacle Positions from engine
    if engine is not None and hasattr(engine, "learned_receptacle_positions"):
        for rp in engine.learned_receptacle_positions:
            pos = (int(rp[0]), int(rp[1]))
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in seen_positions and pos != av_pos:
                seen_positions.add(pos)
                lr_id = f"learned_receptacle_{pos[0]}_{pos[1]}"
                lr_feat = int(g[pos[0], pos[1]])
                lr_props = {"source": "learned_receptacle"}
                objects.append(
                    ObjectSpec(
                        id=lr_id,
                        pos=pos,
                        feature_id=lr_feat,
                        role="receptacle",
                        properties=lr_props,
                        macro_tasks=_attach_object_macro_tasks(
                            "receptacle", pos, lr_feat, lr_id, lr_props
                        ),
                    )
                )

    return objects
