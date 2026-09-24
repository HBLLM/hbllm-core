"""ARC-AGI Perceptual Lifter — Pixel grid to SpatialEntity conversion.

Lifts raw 2D pixel grids and segmented objects into domain-agnostic SpatialEntity
instances and barrier coordinate sets for HCIRSpatialEntityPlanner.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.world.morphology import MorphologicalConcept, ShapeArchetype

logger = logging.getLogger(__name__)


class ARCPerceptualLifter:
    """Perceptual front-end for ARC-AGI-3 environments.

    Lifts raw pixel grids and segmented objects into domain-agnostic SpatialEntity instances
    and barrier coordinate sets for HCIRSpatialEntityPlanner.
    """

    @staticmethod
    def lift(
        grid: np.ndarray,
        raw_objects: list[Any],
        avatar_color: int | None = None,
        avatar_centroid: tuple[float, float] | None = None,
        learned_item_colors: set[int] | None = None,
        learned_receptacle_colors: set[int] | None = None,
        learned_barrier_colors: set[int] | None = None,
        walkable_colors: set[int] | None = None,
        step_size: int = 1,
        target_zone_bounds: tuple[int, int, int, int] | None = None,
        shape_concepts: dict[tuple[tuple[int, int], ...], MorphologicalConcept] | None = None,
    ) -> tuple[list[SpatialEntity], set[tuple[int, int]]]:
        H, W = grid.shape
        step = step_size
        counts = np.bincount(grid.ravel())
        bg_color = int(np.argmax(counts))
        walkable = set(walkable_colors or set()) | {0, bg_color}
        for col, count in enumerate(counts):
            if count >= int(H * W * 0.20) and col != avatar_color:
                walkable.add(int(col))
        learned_b = (learned_barrier_colors or set()) - walkable
        learned_i = (learned_item_colors or set()) - walkable
        learned_r = (learned_receptacle_colors or set()) - walkable

        raw_barriers: set[tuple[int, int]] = set()
        for b_col in learned_b:
            if b_col not in walkable and b_col != avatar_color:
                raw_barriers.update(set(zip(*np.where(grid == b_col))))

        # Detect collinear segmented wall blocks
        from collections import defaultdict

        col_blocks = defaultdict(list)
        row_blocks = defaultdict(list)
        for o in raw_objects:
            if (
                o.color in walkable
                or o.color == 0
                or (avatar_color is not None and o.color == avatar_color)
            ):
                continue
            span_r = o.max_r - o.min_r
            span_c = o.max_c - o.min_c
            # Single continuous wall spanning >= 40% of grid
            if span_r >= int(H * 0.4) and span_c <= max(4, step * 2):
                for br in range(o.min_r, o.max_r + 1):
                    for bc in range(o.min_c, o.max_c + 1):
                        raw_barriers.add((br, bc))
            elif span_c >= int(W * 0.4) and span_r <= max(4, step * 2):
                for br in range(o.min_r, o.max_r + 1):
                    for bc in range(o.min_c, o.max_c + 1):
                        raw_barriers.add((br, bc))
            else:
                col_key = int(o.min_c // step * step) if step > 1 else int(round(o.centroid[1]))
                row_key = int(o.min_r // step * step) if step > 1 else int(round(o.centroid[0]))
                col_blocks[(o.color, col_key)].append(o)
                row_blocks[(o.color, row_key)].append(o)

        for (b_col, c_pos), b_list in col_blocks.items():
            if len(b_list) >= 3:
                min_r = min(o.min_r for o in b_list)
                max_r = max(o.max_r for o in b_list)
                if (max_r - min_r) >= int(H * 0.4):
                    for o in b_list:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))

        for (b_col, r_pos), b_list in row_blocks.items():
            if len(b_list) >= 3:
                min_c = min(o.min_c for o in b_list)
                max_c = max(o.max_c for o in b_list)
                if (max_c - min_c) >= int(W * 0.4):
                    for o in b_list:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))

        # Detect receptacle bounds to prevent interior cavity colors from being classified as items
        receptacle_bounds = []
        min_receptacle_span = max(3, step * 2)
        min_receptacle_area = min_receptacle_span * min_receptacle_span
        if target_zone_bounds:
            receptacle_bounds.append(target_zone_bounds)
        else:
            for o in raw_objects:
                if o.color in walkable or o.color == 0 or o.area >= int(H * W * 0.35):
                    continue
                if (
                    o.color in learned_r
                    or getattr(o, "is_frame", False)
                    or (
                        o.color not in (avatar_color, 0)
                        and (o.max_r - o.min_r >= min_receptacle_span)
                        and (o.max_c - o.min_c >= min_receptacle_span)
                        and o.area >= min_receptacle_area
                    )
                ):
                    receptacle_bounds.append((o.min_r, o.max_r, o.min_c, o.max_c))

        entities: list[SpatialEntity] = []
        max_item_area = max(16, int((step * 2) ** 2 * 2.5))

        if target_zone_bounds:
            tz_r = (
                int(target_zone_bounds[0] // step * step)
                if step > 1
                else int(round((target_zone_bounds[0] + target_zone_bounds[1]) * 0.5))
            )
            tz_c = (
                int(target_zone_bounds[2] // step * step)
                if step > 1
                else int(round((target_zone_bounds[2] + target_zone_bounds[3]) * 0.5))
            )
            receptacle_ent = SpatialEntity(
                id=f"receptacle_{target_zone_bounds[0]}_{target_zone_bounds[2]}",
                role=EntityRole.RECEPTACLE,
                centroid=(
                    (target_zone_bounds[0] + target_zone_bounds[1]) * 0.5,
                    (target_zone_bounds[2] + target_zone_bounds[3]) * 0.5,
                ),
                grid_pos=(tz_r, tz_c),
                area=(target_zone_bounds[1] - target_zone_bounds[0] + 1)
                * (target_zone_bounds[3] - target_zone_bounds[2] + 1),
                bounding_box=target_zone_bounds,
            )
            entities.append(receptacle_ent)

        candidate_item_counts = Counter(
            int(o.color)
            for o in raw_objects
            if o.color not in walkable
            and o.color != 0
            and (avatar_color is None or o.color != avatar_color)
            and o.color not in learned_b
            and o.color not in learned_r
            and o.area <= max_item_area
        )

        for o in raw_objects:
            if o.color in walkable or o.color == 0 or o.area >= int(H * W * 0.30):
                continue
            if H >= 16 and W >= 16 and (o.min_r <= 4 or o.max_r >= H - 2):
                is_av_cand = (avatar_color is not None and o.color == avatar_color) or (
                    avatar_centroid is not None
                    and math.hypot(
                        o.centroid[0] - avatar_centroid[0], o.centroid[1] - avatar_centroid[1]
                    )
                    < step * 1.5
                )
                if not is_av_cand:
                    continue

            # 1. Avatar / Goal check
            is_avatar = False
            is_goal = False
            if avatar_color is not None and o.color == avatar_color:
                if avatar_centroid is not None:
                    d_av = math.hypot(
                        o.centroid[0] - avatar_centroid[0], o.centroid[1] - avatar_centroid[1]
                    )
                    if d_av <= step * 1.5:
                        is_avatar = True
                    elif d_av > step * 1.5:
                        is_goal = True
                else:
                    is_avatar = True
            elif (
                avatar_centroid
                and math.hypot(
                    o.centroid[0] - avatar_centroid[0], o.centroid[1] - avatar_centroid[1]
                )
                < 2.0
            ):
                is_avatar = True

            if step > 1:
                r = int(o.min_r // step * step)
                c = int(o.min_c // step * step)
            else:
                r = int(round(o.centroid[0]))
                c = int(round(o.centroid[1]))
            e_id = f"ent_{o.color}_{r}_{c}"

            if is_avatar:
                ent = SpatialEntity(
                    id=e_id,
                    role=EntityRole.AGENT,
                    centroid=(float(o.centroid[0]), float(o.centroid[1])),
                    grid_pos=(r, c),
                    area=int(o.area),
                    bounding_box=(int(o.min_r), int(o.max_r), int(o.min_c), int(o.max_c)),
                    color=int(o.color),
                )
                entities.append(ent)
                continue

            if is_goal:
                ent = SpatialEntity(
                    id=e_id,
                    role=EntityRole.GOAL,
                    centroid=(float(o.centroid[0]), float(o.centroid[1])),
                    grid_pos=(r, c),
                    area=int(o.area),
                    bounding_box=(int(o.min_r), int(o.max_r), int(o.min_c), int(o.max_c)),
                    color=int(o.color),
                )
                entities.append(ent)
                continue

            # 2. If inside receptacle bounds, it is part of receptacle or a delivered item
            is_inside_receptacle = any(
                b[0] <= o.min_r and o.max_r <= b[1] and b[2] <= o.min_c and o.max_c <= b[3]
                for b in receptacle_bounds
            )
            if is_inside_receptacle and target_zone_bounds:
                # Any external object inside receptacle (delivered items, NPC bots) is a physical obstacle
                if o.color not in learned_r and o.color not in walkable and o.color != avatar_color:
                    if hasattr(o, "coords"):
                        raw_barriers.update((int(cr), int(cc)) for cr, cc in o.coords)
                    else:
                        for br in range(o.min_r, o.max_r + 1):
                            for bc in range(o.min_c, o.max_c + 1):
                                raw_barriers.add((br, bc))
                continue

            o_coords = (
                set((int(cr), int(cc)) for cr, cc in o.coords)
                if hasattr(o, "coords")
                else {
                    (br, bc)
                    for br in range(o.min_r, o.max_r + 1)
                    for bc in range(o.min_c, o.max_c + 1)
                }
            )
            arch = ShapeArchetype.from_coords(o_coords)

            role = EntityRole.UNKNOWN
            if shape_concepts and arch.canonical_id in shape_concepts:
                concept = shape_concepts[arch.canonical_id]
                concept.observed_colors.add(int(o.color))
                if concept.inferred_role in (EntityRole.PORTAL, EntityRole.PORTAL):
                    if int(o.color) in concept.passable_colors:
                        role = EntityRole.PORTAL
                    else:
                        role = EntityRole.PORTAL
                elif concept.inferred_role == EntityRole.MANIPULABLE:
                    role = EntityRole.MANIPULABLE
                elif concept.inferred_role == EntityRole.ACTUATOR:
                    role = EntityRole.ACTUATOR
                elif concept.inferred_role == EntityRole.RECEPTACLE:
                    role = EntityRole.RECEPTACLE

            if role == EntityRole.UNKNOWN:
                if o.color in learned_r or is_inside_receptacle:
                    role = EntityRole.RECEPTACLE
                elif (
                    o.color in learned_b
                    or (int(round(o.centroid[0])), int(round(o.centroid[1]))) in raw_barriers
                    or (o.min_r, o.min_c) in raw_barriers
                ):
                    role = EntityRole.OBSTACLE
                    raw_barriers.update(o_coords)
                elif o.color in learned_i:
                    role = (
                        EntityRole.MANIPULABLE
                        if (receptacle_bounds or learned_r)
                        else EntityRole.GOAL
                    )
                elif not learned_i and o.color != avatar_color and o.area <= max_item_area:
                    # Multi-instance objects are manipulable cargo items; singleton unique colors are transformation pads/actuators
                    if (receptacle_bounds or learned_r) and candidate_item_counts.get(
                        int(o.color), 0
                    ) > 1:
                        role = EntityRole.MANIPULABLE
                    elif receptacle_bounds or learned_r:
                        role = EntityRole.ACTUATOR
                    else:
                        role = EntityRole.GOAL
                else:
                    role = EntityRole.ACTUATOR

            ent = SpatialEntity(
                id=e_id,
                role=role,
                centroid=(float(o.centroid[0]), float(o.centroid[1])),
                grid_pos=(r, c),
                area=int(o.area),
                bounding_box=(int(o.min_r), int(o.max_r), int(o.min_c), int(o.max_c)),
                color=int(o.color),
                shape_archetype=arch,
            )
            entities.append(ent)

        return entities, raw_barriers


def arc_perception_lifter(
    perception_data: dict[str, Any],
    state: Any,
) -> tuple[list[SpatialEntity], set[tuple[int, int]]]:
    """Domain-specific perception lifter for ARC-AGI-3 grid environments.

    Lifts 2D pixel grids into domain-agnostic SpatialEntities with roles:
    AGENT, MANIPULABLE, RECEPTACLE, PORTAL, ACTUATOR, and OBSTACLE.
    """
    from plugins.arc_agi_adapter.arc_agi_runner import ARCGrid, GridTopologyExtractor

    grid = perception_data.get("grid")
    if grid is None:
        return [], set()

    H, W = grid.shape
    step = getattr(state, "step_size", 1) or perception_data.get("step_size", 1)
    avatar_color = getattr(state, "avatar_feature", None) or perception_data.get("avatar_color")

    raw_objects = GridTopologyExtractor.extract_objects(
        ARCGrid(grid.tolist() if hasattr(grid, "tolist") else grid)
    )

    counts = np.bincount(grid.ravel())
    bg_color = int(np.argmax(counts))
    traversable = getattr(state, "learned_traversable_features", set())
    walkable = set(traversable) | {0, bg_color}
    for col, count in enumerate(counts):
        if count >= int(H * W * 0.20) and col != avatar_color:
            walkable.add(int(col))

    # Detect receptacle candidate bounds
    receptacle_bounds = perception_data.get("target_zone_bounds")
    domain_instr = getattr(state, "domain_instructions", {})
    if not receptacle_bounds and "receptacle_bounds" in domain_instr:
        receptacle_bounds = domain_instr["receptacle_bounds"]

    if not receptacle_bounds:
        for o in raw_objects:
            if (
                o.color not in walkable
                and o.color not in (0, avatar_color)
                and 24 <= o.area <= int(H * W * 0.20)
                and o.width >= 6
                and o.height >= 3
            ):
                receptacle_bounds = (o.min_r, o.max_r, o.min_c, o.max_c)
                break

    learned_r = domain_instr.get("learned_receptacle_colors", set())
    learned_i = getattr(state, "learned_target_features", set())
    learned_b = getattr(state, "learned_obstacle_features", set())

    entities, barriers = ARCPerceptualLifter.lift(
        grid=grid,
        raw_objects=raw_objects,
        avatar_color=avatar_color,
        learned_item_colors=learned_i,
        learned_receptacle_colors=learned_r,
        learned_barrier_colors=learned_b,
        walkable_colors=walkable,
        step_size=step,
        target_zone_bounds=receptacle_bounds,
    )
    return entities, barriers
