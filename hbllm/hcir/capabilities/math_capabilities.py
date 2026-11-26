"""Stateless Layer-0 math, topology, symmetry, and set-cardinality ALUs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from hbllm.hcir.primitives import detect_background_color, segment_entities
from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer

MATH_CAPABILITIES: tuple[str, ...] = (
    "math.connected_components",
    "math.topology_cuts",
    "math.symmetry_score",
    "math.set_cardinality",
)


def _to_2d(grid: Any) -> np.ndarray:
    arr = np.asarray(grid)
    return arr[-1] if arr.ndim == 3 else arr


def compute_connected_components(
    grid: Any, *, connectivity: int = 4, bg_color: int | None = None
) -> dict[str, Any]:
    """Segment discrete 4/8-connected entities from a 2D lattice."""
    arr = _to_2d(grid)
    bg = int(detect_background_color(arr)) if bg_color is None else int(bg_color)
    entities = segment_entities(arr, bg_color=bg, connectivity=connectivity)
    return {
        "background_color": bg,
        "count": len(entities),
        "entities": [
            {
                "color": int(e.color),
                "area": int(e.area),
                "width": int(e.width),
                "height": int(e.height),
                "is_frame": bool(e.is_frame),
                "centroid": [float(e.centroid[0]), float(e.centroid[1])],
            }
            for e in entities
        ],
    }


def compute_topology_cuts(
    grid: Any, *, bg_color: int | None = None, walkable_val: int | None = None
) -> dict[str, Any]:
    """Count disconnected passable partitions separated by cut-set barriers."""
    arr = _to_2d(grid)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if walkable_val is not None:
        bg, is_passable = int(walkable_val), arr == int(walkable_val)
    else:
        bg = int(detect_background_color(arr)) if bg_color is None else int(bg_color)
        is_passable = arr != bg
    barriers = {(r, c) for r in range(h) for c in range(w) if not bool(is_passable[r, c])}
    passable = [(r, c) for r in range(h) for c in range(w) if bool(is_passable[r, c])]
    visited: set[tuple[int, int]] = set()
    partitions = 0
    for cell in passable:
        if cell not in visited:
            visited.update(
                TopologicalCutSetAnalyzer.get_reachable_component(
                    start=cell, barrier_cells=barriers, grid_shape=(h, w)
                )
            )
            partitions += 1
    return {"background_color": bg, "partition_count": partitions, "has_cut_set": partitions >= 2}


def compute_symmetry_score(
    grid: Any,
    *,
    axis: str = "vertical",
    max_deviation: float = 2.0,
    min_entities: int = 2,
    min_area: int = 9,
    max_area: int = 36,
    ignore_top_colors: int = 3,
    bg_color: int | None = None,
) -> dict[str, Any]:
    """Measure bilateral centroid symmetry across bounded segmented entities."""
    arr = _to_2d(grid)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    bg = int(detect_background_color(arr)) if bg_color is None else int(bg_color)
    vals, cnts = np.unique(arr, return_counts=True)
    ignored = {int(v) for _, v in sorted(zip(cnts, vals), reverse=True)[:ignore_top_colors]} | {
        0,
        bg,
    }
    entities = [
        e
        for e in segment_entities(arr, bg_color=bg, connectivity=4)
        if e.color not in ignored and min_area <= e.area <= max_area
    ]
    if len(entities) < min_entities:
        return {"symmetric": False, "axis": axis, "min_deviation": float("inf")}
    mid, min_dev, symmetric = (
        (w - 1) / 2.0 if axis == "vertical" else (h - 1) / 2.0,
        float("inf"),
        False,
    )
    for color in {e.color for e in entities}:
        col_ents = [e for e in entities if e.color == color]
        if len(col_ents) in (2, 4):
            coords = [e.centroid[1] if axis == "vertical" else e.centroid[0] for e in col_ents]
            dev = abs((sum(coords) / len(col_ents)) - mid)
            min_dev = min(min_dev, float(dev))
            symmetric = symmetric or (dev <= max_deviation)
    return {"symmetric": symmetric, "axis": axis, "min_deviation": min_dev}


def compute_set_cardinality(
    items_a: Sequence[Any], items_b: Sequence[Any] | None = None
) -> dict[str, Any]:
    """Discrete numerosity & set-comparison ALU."""
    sa, sb = set(items_a), set(items_b or ())
    return {
        "count_a": len(sa),
        "count_b": len(sb),
        "equals": sa == sb,
        "subset": sa.issubset(sb),
        "disjoint": sa.isdisjoint(sb),
    }


class MathCapabilityExecutor:
    """Stateless kernel executor exposing `math.*` Layer-0 ALUs via `Opcode.EXECUTE`."""

    def __init__(self, default_capability: str = "math.topology_cuts") -> None:
        self._default_capability = default_capability

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        cap = str(params.get("capability", self._default_capability))
        if cap == "math.connected_components":
            return compute_connected_components(
                params["grid"],
                connectivity=int(params.get("connectivity", 4)),
                bg_color=params.get("bg_color"),
            )
        if cap == "math.topology_cuts":
            return compute_topology_cuts(
                params["grid"],
                bg_color=params.get("bg_color"),
                walkable_val=params.get("walkable_val"),
            )
        if cap == "math.symmetry_score":
            return compute_symmetry_score(
                params["grid"],
                axis=str(params.get("axis", "vertical")),
                max_deviation=float(params.get("max_deviation", 2.0)),
                min_entities=int(params.get("min_entities", 2)),
                min_area=int(params.get("min_area", 9)),
                max_area=int(params.get("max_area", 36)),
                ignore_top_colors=int(params.get("ignore_top_colors", 3)),
                bg_color=params.get("bg_color"),
            )
        if cap == "math.set_cardinality":
            return compute_set_cardinality(params.get("items_a", ()), params.get("items_b"))
        return {"error": f"Unknown math capability: {cap}"}
