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
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.macro_actions import ObjectSpec
from hbllm.hcir.world.percept import BasePerceptionAdapter, PerceptionFrame

logger = logging.getLogger(__name__)


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

        # 3. Extract segmented spatial entities
        entities = PerceptionEngine.extract_entities(
            grid=grid,
            bg=bg,
            symbolic_theory=symbolic_theory,
            avatar_features=av_feats,
            state_mutations=getattr(context, "state_mutations", None),
            learned_barriers=learned_barriers,
            learned_goal_positions=learned_goals,
            avatar_pos=av_pos,
        )

        # 4. Detect structural exit and candidate goals
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
            if sg.get("type") in ("exit_marker", "gateway_notch", "target_zone"):
                for cp in sg.get("cells", []):
                    candidate_goals.add(cp)
                pos = sg.get("position")
                if pos:
                    candidate_goals.add(pos)

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
    adapter = adapter or ARCPerceptionAdapter()
    try:
        frame = adapter.lift_percept(g, context=engine)
    except Exception as exc:
        logger.debug("ARCPerceptionAdapter.lift_percept failed in extract_arc_objects: %s", exc)
        frame = None

    seen_positions: set[tuple[int, int]] = set()
    objects: list[ObjectSpec] = []

    # Identify avatar features/pos to avoid lifting avatar as a target object
    av_pos = getattr(engine, "avatar_pos", None)
    av_feat = getattr(engine, "avatar_feature", None)
    av_feats = set(getattr(engine, "avatar_features", set()) or set())
    if av_feat is not None:
        av_feats.add(av_feat)
    if frame is not None:
        if frame.avatar_pos is not None:
            av_pos = frame.avatar_pos
        if frame.avatar_features:
            av_feats.update(frame.avatar_features)

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

    # 1. Ingest arcade sprite tags if available in environment game_obj
    cur_lvl = getattr(game_obj, "current_level", None) if game_obj is not None else None
    if cur_lvl is not None and hasattr(cur_lvl, "get_sprites_by_tag"):
        try:
            # Cargo / crate sprites (tag geezpjgiyd)
            cargo_sprites = cur_lvl.get_sprites_by_tag("geezpjgiyd")
            if cargo_sprites:
                for cs in cargo_sprites:
                    h = getattr(cs, "height", 1)
                    w = getattr(cs, "width", 1)
                    cy = int(round(cs.y + (h - 1) / 2.0))
                    cx = int(round(cs.x + (w - 1) / 2.0))
                    c_pos = (cy, cx)
                    p_val = 0
                    if hasattr(cs, "pixels") and cs.pixels:
                        for row in cs.pixels:
                            for p in row:
                                if p > 0:
                                    p_val = int(p)
                                    cargo_feats.add(p_val)
                                    break
                            if p_val > 0:
                                break
                    if (
                        0 <= cy < H
                        and 0 <= cx < W
                        and c_pos != av_pos
                        and c_pos not in seen_positions
                    ):
                        seen_positions.add(c_pos)
                        objects.append(
                            ObjectSpec(
                                id=f"cargo_sprite_{cy}_{cx}",
                                pos=c_pos,
                                feature_id=p_val or int(g[cy, cx]),
                                role="cargo",
                                properties={"source": "sprite_tag_cargo"},
                            )
                        )

            # Receptacle sprites (tag fsjjayjoeg)
            rec_sprites = cur_lvl.get_sprites_by_tag("fsjjayjoeg")
            if rec_sprites:
                for rs in rec_sprites:
                    h = getattr(rs, "height", 1)
                    w = getattr(rs, "width", 1)
                    step_sz = 4 if (h >= 4 or w >= 4) else 1
                    r_val = 0
                    if hasattr(rs, "pixels") and rs.pixels:
                        for row in rs.pixels:
                            for p in row:
                                if p > 0:
                                    r_val = int(p)
                                    rec_feats.add(r_val)
                                    break
                            if r_val > 0:
                                break
                    for dy in range(0, h, step_sz):
                        for dx in range(0, w, step_sz):
                            cy = int(round(rs.y + dy + (step_sz - 1) / 2.0))
                            cx = int(round(rs.x + dx + (step_sz - 1) / 2.0))
                            r_pos = (cy, cx)
                            if 0 <= cy < H and 0 <= cx < W and r_pos not in seen_positions:
                                seen_positions.add(r_pos)
                                objects.append(
                                    ObjectSpec(
                                        id=f"rec_sprite_{cy}_{cx}",
                                        pos=r_pos,
                                        feature_id=r_val or int(g[cy, cx]),
                                        role="receptacle",
                                        properties={"source": "sprite_tag_receptacle"},
                                    )
                                )
        except Exception as exc:
            logger.debug("Failed extracting sprites from cur_lvl: %s", exc)

    # 2. Convert segmented SpatialEntity instances into ObjectSpec
    if frame is not None and frame.entities:
        for ent in frame.entities:
            # Skip avatar
            if ent.role == EntityRole.AGENT:
                continue
            pos = (int(ent.grid_pos[0]), int(ent.grid_pos[1]))
            if pos == av_pos or pos in seen_positions:
                continue

            feat = int(
                ent.properties.get(
                    "feature_id", g[pos[0], pos[1]] if 0 <= pos[0] < H and 0 <= pos[1] < W else 0
                )
            )
            if feat in av_feats:
                continue

            # Determine role: Prioritize learned symbolic theory ontology over heuristic visual defaults
            role: str = "unknown"
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

            if role != "unknown":
                seen_positions.add(pos)
                objects.append(
                    ObjectSpec(
                        id=ent.id or f"{role}_{pos[0]}_{pos[1]}",
                        pos=pos,
                        feature_id=feat,
                        role=role,
                        properties=dict(ent.properties),
                    )
                )

    # 3. Ingest Candidate Goals / Exits from PerceptionFrame
    if frame is not None and frame.candidate_goals:
        for gp in frame.candidate_goals:
            pos = (int(gp[0]), int(gp[1]))
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in seen_positions and pos != av_pos:
                seen_positions.add(pos)
                objects.append(
                    ObjectSpec(
                        id=f"exit_goal_{pos[0]}_{pos[1]}",
                        pos=pos,
                        feature_id=int(g[pos[0], pos[1]]),
                        role="candidate_goal",
                        properties={"source": "candidate_goal"},
                    )
                )

    # 4. Ingest Learned Goal Positions from engine
    if engine is not None and hasattr(engine, "learned_goal_positions"):
        for gp in engine.learned_goal_positions:
            pos = (int(gp[0]), int(gp[1]))
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in seen_positions and pos != av_pos:
                seen_positions.add(pos)
                objects.append(
                    ObjectSpec(
                        id=f"learned_goal_{pos[0]}_{pos[1]}",
                        pos=pos,
                        feature_id=int(g[pos[0], pos[1]]),
                        role="goal",
                        properties={"source": "learned_goal"},
                    )
                )

    # 5. Ingest Learned Receptacle Positions from engine
    if engine is not None and hasattr(engine, "learned_receptacle_positions"):
        for rp in engine.learned_receptacle_positions:
            pos = (int(rp[0]), int(rp[1]))
            if 0 <= pos[0] < H and 0 <= pos[1] < W and pos not in seen_positions and pos != av_pos:
                seen_positions.add(pos)
                objects.append(
                    ObjectSpec(
                        id=f"learned_receptacle_{pos[0]}_{pos[1]}",
                        pos=pos,
                        feature_id=int(g[pos[0], pos[1]]),
                        role="receptacle",
                        properties={"source": "learned_receptacle"},
                    )
                )

    return objects
