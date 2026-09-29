"""Declarative Neuro-Symbolic Skill Grammar and Compilation Engine — HCIR §6.

Provides a domain-agnostic declarative domain-specific language (DSL) to specify
and compile hierarchical skills in the core HBLLM Cognitive OS.

Features:
1. Composable Symbolic Predicates (ActionAffordance, EntityCount, PanelConstraint,
   CutSet, Symmetry, and Boolean logic AllOf, AnyOf, Not).
2. Cognitive Lifting (transforms 2D lattice observations into a cached, typed
   SkillEvaluationContext with segmented EntityComponents and spatial metrics).
3. Declarative Subgoals & Intent Hierarchy (SymbolicSubgoal, SubgoalSequence).
4. Direct HCIR Bytecode Compilation (compiles declarative subgoals into
   canonical InstructionStream: QUERY -> ASSERT -> EXECUTE).
5. Seamless execution dispatch conforming to BaseHierarchicalSkill.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.primitives import (
    EntityComponent,
    Grid,
    detect_background_color,
    segment_entities,
)
from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.spatial_planner import SpatialActionIntent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Cognitive Lifting & Evaluation Context
# ═══════════════════════════════════════════════════════════════════════════


class SkillEvaluationContext:
    """Cached, lifted cognitive context for evaluating symbolic skill predicates.

    Provides lazy evaluation and caching of segmented entities, background
    colors, and panel subgrids so predicates do not perform redundant work.
    """

    def __init__(
        self,
        grid: Grid,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if grid.ndim == 3:
            grid = grid[-1]
        self.grid: Grid = grid
        self.available_actions: list[int] = list(available_actions)
        self.available_actions_set: set[int] = set(available_actions)
        self.metadata: dict[str, Any] = metadata or {}

        # Lazy cached values
        self._bg_color: int | None = None
        self._entities_4conn: list[EntityComponent] | None = None
        self._entities_8conn: list[EntityComponent] | None = None
        self._color_hist: dict[int, int] | None = None

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def background_color(self) -> int:
        if self._bg_color is None:
            self._bg_color = detect_background_color(self.grid)
        return self._bg_color

    @property
    def color_histogram(self) -> dict[int, int]:
        if self._color_hist is None:
            counts = Counter(int(x) for x in self.grid.flatten())
            self._color_hist = dict(counts)
        return self._color_hist

    def get_entities(self, connectivity: int = 4) -> list[EntityComponent]:
        """Return segmented entities on the lattice, lazily computed and cached."""
        if connectivity == 8:
            if self._entities_8conn is None:
                self._entities_8conn = segment_entities(
                    self.grid, bg_color=self.background_color, connectivity=8
                )
            return self._entities_8conn
        if self._entities_4conn is None:
            self._entities_4conn = segment_entities(
                self.grid, bg_color=self.background_color, connectivity=4
            )
        return self._entities_4conn

    def create_subcontext(
        self, r_min: int, r_max: int, c_min: int, c_max: int
    ) -> SkillEvaluationContext:
        """Create a subcontext for an extracted spatial subgrid (e.g. console panel)."""
        subgrid = self.grid[r_min:r_max, c_min:c_max].copy()
        return SkillEvaluationContext(
            grid=subgrid,
            available_actions=self.available_actions,
            metadata=dict(self.metadata),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Composable Symbolic Predicates
# ═══════════════════════════════════════════════════════════════════════════


class SymbolicPredicate(ABC):
    """Abstract base class for all declarative neuro-symbolic predicates."""

    @abstractmethod
    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        """Evaluate the predicate against the lifted cognitive context."""
        ...

    def __and__(self, other: SymbolicPredicate) -> AllOf:
        return AllOf(self, other)

    def __or__(self, other: SymbolicPredicate) -> AnyOf:
        return AnyOf(self, other)

    def __invert__(self) -> Not:
        return Not(self)


class AllOf(SymbolicPredicate):
    """Logical conjunction (AND) of multiple symbolic predicates."""

    def __init__(self, *predicates: SymbolicPredicate) -> None:
        self.predicates: list[SymbolicPredicate] = list(predicates)

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        return all(p.evaluate(ctx) for p in self.predicates)

    def __repr__(self) -> str:
        return f"AllOf({', '.join(repr(p) for p in self.predicates)})"


class AnyOf(SymbolicPredicate):
    """Logical disjunction (OR) of multiple symbolic predicates."""

    def __init__(self, *predicates: SymbolicPredicate) -> None:
        self.predicates: list[SymbolicPredicate] = list(predicates)

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        return any(p.evaluate(ctx) for p in self.predicates)

    def __repr__(self) -> str:
        return f"AnyOf({', '.join(repr(p) for p in self.predicates)})"


class Not(SymbolicPredicate):
    """Logical negation (NOT) of a symbolic predicate."""

    def __init__(self, predicate: SymbolicPredicate) -> None:
        self.predicate: SymbolicPredicate = predicate

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        return not self.predicate.evaluate(ctx)

    def __repr__(self) -> str:
        return f"Not({self.predicate!r})"


# ── Structural & Affordance Predicates ─────────────────────────────────────


class ActionAffordancePredicate(SymbolicPredicate):
    """Evaluates whether available actions satisfy specified affordance constraints."""

    def __init__(
        self,
        required: set[int] | Sequence[int] | None = None,
        forbidden: set[int] | Sequence[int] | None = None,
        exact: set[int] | Sequence[int] | None = None,
    ) -> None:
        self.required: set[int] = set(required) if required is not None else set()
        self.forbidden: set[int] = set(forbidden) if forbidden is not None else set()
        self.exact: set[int] | None = set(exact) if exact is not None else None

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        acts = ctx.available_actions_set
        if self.exact is not None and acts != self.exact:
            return False
        if self.required and not self.required.issubset(acts):
            return False
        if self.forbidden and not self.forbidden.isdisjoint(acts):
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"ActionAffordancePredicate(required={self.required}, "
            f"forbidden={self.forbidden}, exact={self.exact})"
        )


class GridDimensionPredicate(SymbolicPredicate):
    """Evaluates whether grid dimensions fall within specified bounds."""

    def __init__(
        self,
        min_height: int | None = None,
        max_height: int | None = None,
        min_width: int | None = None,
        max_width: int | None = None,
        exact_shape: tuple[int, int] | None = None,
    ) -> None:
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width
        self.exact_shape = exact_shape

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        h, w = ctx.height, ctx.width
        if self.exact_shape is not None and (h, w) != self.exact_shape:
            return False
        if self.min_height is not None and h < self.min_height:
            return False
        if self.max_height is not None and h > self.max_height:
            return False
        if self.min_width is not None and w < self.min_width:
            return False
        if self.max_width is not None and w > self.max_width:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"GridDimensionPredicate(h=[{self.min_height}, {self.max_height}], "
            f"w=[{self.min_width}, {self.max_width}], exact={self.exact_shape})"
        )


class EntityBoundingBoxPredicate(SymbolicPredicate):
    """Evaluates whether at least `min_count` entities match bounding-box constraints."""

    def __init__(
        self,
        min_width: int | None = None,
        max_width: int | None = None,
        min_height: int | None = None,
        max_height: int | None = None,
        min_count: int = 1,
        color: int | None = None,
        connectivity: int = 4,
    ) -> None:
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.min_count = min_count
        self.color = color
        self.connectivity = connectivity

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        matching = 0
        for e in ctx.get_entities(connectivity=self.connectivity):
            if self.color is not None and e.color != self.color:
                continue
            if self.min_width is not None and e.width < self.min_width:
                continue
            if self.max_width is not None and e.width > self.max_width:
                continue
            if self.min_height is not None and e.height < self.min_height:
                continue
            if self.max_height is not None and e.height > self.max_height:
                continue
            matching += 1
            if matching >= self.min_count:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"EntityBoundingBoxPredicate(w=[{self.min_width}, {self.max_width}], "
            f"h=[{self.min_height}, {self.max_height}], min_count={self.min_count})"
        )


class EntityCountPredicate(SymbolicPredicate):
    """Evaluates whether the count of segmented entities meeting criteria falls in range."""

    def __init__(
        self,
        min_count: int = 1,
        max_count: int | None = None,
        color: int | None = None,
        is_frame: bool | None = None,
        min_area: int | None = None,
        max_area: int | None = None,
        connectivity: int = 4,
    ) -> None:
        self.min_count = min_count
        self.max_count = max_count
        self.color = color
        self.is_frame = is_frame
        self.min_area = min_area
        self.max_area = max_area
        self.connectivity = connectivity

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        matching = 0
        for e in ctx.get_entities(connectivity=self.connectivity):
            if self.color is not None and e.color != self.color:
                continue
            if self.is_frame is not None and e.is_frame != self.is_frame:
                continue
            if self.min_area is not None and e.area < self.min_area:
                continue
            if self.max_area is not None and e.area > self.max_area:
                continue
            matching += 1

        if matching < self.min_count:
            return False
        if self.max_count is not None and matching > self.max_count:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"EntityCountPredicate(min={self.min_count}, max={self.max_count}, color={self.color})"
        )


class PanelConstraint(SymbolicPredicate):
    """Extracts a relative spatial window (panel) and evaluates sub-predicates inside it."""

    def __init__(
        self,
        min_row_ratio: float = 0.0,
        max_row_ratio: float = 1.0,
        min_col_ratio: float = 0.0,
        max_col_ratio: float = 1.0,
        contains_entities: SymbolicPredicate | None = None,
        min_distinct_colors: int = 1,
    ) -> None:
        self.min_row_ratio = min_row_ratio
        self.max_row_ratio = max_row_ratio
        self.min_col_ratio = min_col_ratio
        self.max_col_ratio = max_col_ratio
        self.contains_entities = contains_entities
        self.min_distinct_colors = min_distinct_colors

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        h, w = ctx.height, ctx.width
        r_min = int(round(self.min_row_ratio * h))
        r_max = int(round(self.max_row_ratio * h))
        c_min = int(round(self.min_col_ratio * w))
        c_max = int(round(self.max_col_ratio * w))

        if r_max <= r_min or c_max <= c_min:
            return False

        sub_ctx = ctx.create_subcontext(r_min, r_max, c_min, c_max)
        distinct = len(set(int(x) for x in sub_ctx.grid.flatten()))
        if distinct < self.min_distinct_colors:
            return False

        if self.contains_entities is not None:
            return self.contains_entities.evaluate(sub_ctx)

        return True

    def __repr__(self) -> str:
        return (
            f"PanelConstraint(r=[{self.min_row_ratio}, {self.max_row_ratio}], "
            f"c=[{self.min_col_ratio}, {self.max_col_ratio}], contains={self.contains_entities!r})"
        )


class ColorDistributionPredicate(SymbolicPredicate):
    """Evaluates the presence, absence, and variety of colors in the lattice."""

    def __init__(
        self,
        present_colors: set[int] | Sequence[int] | None = None,
        absent_colors: set[int] | Sequence[int] | None = None,
        min_distinct_colors: int | None = None,
        max_distinct_colors: int | None = None,
    ) -> None:
        self.present_colors: set[int] = set(present_colors) if present_colors is not None else set()
        self.absent_colors: set[int] = set(absent_colors) if absent_colors is not None else set()
        self.min_distinct_colors = min_distinct_colors
        self.max_distinct_colors = max_distinct_colors

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        hist = ctx.color_histogram
        distinct = len(hist)

        if self.min_distinct_colors is not None and distinct < self.min_distinct_colors:
            return False
        if self.max_distinct_colors is not None and distinct > self.max_distinct_colors:
            return False
        if self.present_colors and not self.present_colors.issubset(hist.keys()):
            return False
        if self.absent_colors and not self.absent_colors.isdisjoint(hist.keys()):
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"ColorDistributionPredicate(present={self.present_colors}, "
            f"absent={self.absent_colors}, distinct=[{self.min_distinct_colors}, {self.max_distinct_colors}])"
        )


class CutSetPredicate(SymbolicPredicate):
    """Evaluates whether the environment contains topological bottleneck cut-sets."""

    def __init__(self, min_partitions: int = 2) -> None:
        self.min_partitions = min_partitions

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        from hbllm.hcir.topological_cut_set import TopologicalCutSetAnalyzer

        barrier_cells = {
            (r, c)
            for r in range(ctx.height)
            for c in range(ctx.width)
            if ctx.grid[r, c] == ctx.background_color
        }
        passable_cells = [
            (r, c)
            for r in range(ctx.height)
            for c in range(ctx.width)
            if ctx.grid[r, c] != ctx.background_color
        ]
        if not passable_cells:
            return False

        # Check if entities/cells are partitioned into multiple disconnected components
        visited: set[tuple[int, int]] = set()
        partitions = 0
        for cell in passable_cells:
            if cell not in visited:
                comp = TopologicalCutSetAnalyzer.get_reachable_component(
                    start=cell,
                    barrier_cells=barrier_cells,
                    grid_shape=(ctx.height, ctx.width),
                )
                visited.update(comp)
                partitions += 1
                if partitions >= self.min_partitions:
                    return True
        return partitions >= self.min_partitions

    def __repr__(self) -> str:
        return f"CutSetPredicate(min_partitions={self.min_partitions})"


class HomogeneousRegionPredicate(SymbolicPredicate):
    """Evaluates whether a specified subgrid region is completely homogeneous (all background)."""

    def __init__(
        self,
        min_row_ratio: float = 0.0,
        max_row_ratio: float = 1.0,
        min_col_ratio: float = 0.0,
        max_col_ratio: float = 1.0,
        require_background: bool = True,
    ) -> None:
        self.min_row_ratio = min_row_ratio
        self.max_row_ratio = max_row_ratio
        self.min_col_ratio = min_col_ratio
        self.max_col_ratio = max_col_ratio
        self.require_background = require_background

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        h, w = ctx.height, ctx.width
        r_min = int(round(self.min_row_ratio * h))
        r_max = int(round(self.max_row_ratio * h))
        c_min = int(round(self.min_col_ratio * w))
        c_max = int(round(self.max_col_ratio * w))

        if r_min >= r_max or c_min >= c_max:
            return False

        sub = ctx.grid[r_min:r_max, c_min:c_max]
        if sub.size == 0:
            return False

        if self.require_background:
            return bool(np.all(sub == ctx.background_color))
        first = sub.flat[0]
        return bool(np.all(sub == first))

    def __repr__(self) -> str:
        return (
            f"HomogeneousRegionPredicate(r=[{self.min_row_ratio:.2f}, {self.max_row_ratio:.2f}], "
            f"c=[{self.min_col_ratio:.2f}, {self.max_col_ratio:.2f}])"
        )


class LatticeConduitPredicate(SymbolicPredicate):
    """Evaluates whether the grid contains a high-density periodic lattice conduit network."""

    def __init__(self, stride: int = 6, min_conduits: int = 15, patch_size: int = 3) -> None:
        self.stride = stride
        self.min_conduits = min_conduits
        self.patch_size = patch_size

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        from hbllm.hcir.skills.common_subskills import LatticeQuantizer

        anchor = LatticeQuantizer.detect_lattice_anchor(
            ctx.grid, stride=self.stride, patch_size=self.patch_size
        )
        off_y, off_x = anchor
        h, w = ctx.height, ctx.width
        bg = ctx.background_color
        bridges = 0
        for y in range(off_y, h - self.stride, self.stride):
            for x in range(off_x, w - self.stride, self.stride):
                if x + self.stride <= w and ctx.grid[y + 1, x + 3] != bg:
                    bridges += 1
                if y + self.stride <= h and ctx.grid[y + 3, x + 1] != bg:
                    bridges += 1
        return bridges >= self.min_conduits

    def __repr__(self) -> str:
        return f"LatticeConduitPredicate(stride={self.stride}, min_conduits={self.min_conduits})"


class SymmetryPredicate(SymbolicPredicate):
    """Evaluates bilateral or rotational symmetry across segmented entities or grid."""

    def __init__(
        self,
        axis: str = "vertical",
        max_deviation: float = 2.0,
        min_entities: int = 2,
        min_area: int = 9,
        max_area: int = 36,
        ignore_top_colors: int = 3,
    ) -> None:
        self.axis = axis
        self.max_deviation = max_deviation
        self.min_entities = min_entities
        self.min_area = min_area
        self.max_area = max_area
        self.ignore_top_colors = ignore_top_colors

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        hist = ctx.color_histogram
        sorted_colors = sorted(hist.keys(), key=lambda c: hist[c], reverse=True)
        ignored = set(sorted_colors[: self.ignore_top_colors]) | {0, ctx.background_color}

        entities = [
            e
            for e in ctx.get_entities()
            if e.color not in ignored and self.min_area <= e.area <= self.max_area
        ]
        if len(entities) < self.min_entities:
            return False

        colors = {e.color for e in entities}
        mid = (ctx.width - 1) / 2.0 if self.axis == "vertical" else (ctx.height - 1) / 2.0
        for c in colors:
            col_ents = [e for e in entities if e.color == c]
            if len(col_ents) in (2, 4):
                if self.axis == "vertical":
                    c_mean = sum(e.centroid[1] for e in col_ents) / len(col_ents)
                else:
                    c_mean = sum(e.centroid[0] for e in col_ents) / len(col_ents)
                if abs(c_mean - mid) <= self.max_deviation:
                    return True
        return False

    def __repr__(self) -> str:
        return f"SymmetryPredicate(axis={self.axis}, max_dev={self.max_deviation})"


class TileGridPredicate(SymbolicPredicate):
    """Evaluates whether the grid contains an array of regular interactive tiles."""

    def __init__(self, min_tiles: int = 4, min_tile_size: int = 3) -> None:
        self.min_tiles = min_tiles
        self.min_tile_size = min_tile_size

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        entities = ctx.get_entities()
        tiles = [
            e
            for e in entities
            if e.width >= self.min_tile_size
            and e.height >= self.min_tile_size
            and abs(e.width - e.height) <= 2
        ]
        return len(tiles) >= self.min_tiles

    def __repr__(self) -> str:
        return f"TileGridPredicate(min_tiles={self.min_tiles})"


class MetadataPredicate(SymbolicPredicate):
    """Evaluates conditions on contextual observation or agent metadata."""

    def __init__(self, key: str, expected_value: Any = True) -> None:
        self.key = key
        self.expected_value = expected_value

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        if not ctx.metadata:
            return False
        return ctx.metadata.get(self.key) == self.expected_value

    def __repr__(self) -> str:
        return f"MetadataPredicate({self.key}={self.expected_value})"


class PixelDensityPredicate(SymbolicPredicate):
    """Evaluates non-background pixel count within a region."""

    def __init__(
        self,
        min_count: int = 1,
        max_count: int | None = None,
        min_row_ratio: float = 0.0,
        max_row_ratio: float = 1.0,
        min_col_ratio: float = 0.0,
        max_col_ratio: float = 1.0,
    ) -> None:
        self.min_count = min_count
        self.max_count = max_count
        self.min_row_ratio = min_row_ratio
        self.max_row_ratio = max_row_ratio
        self.min_col_ratio = min_col_ratio
        self.max_col_ratio = max_col_ratio

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        h, w = ctx.height, ctx.width
        r_min = int(round(self.min_row_ratio * h))
        r_max = int(round(self.max_row_ratio * h))
        c_min = int(round(self.min_col_ratio * w))
        c_max = int(round(self.max_col_ratio * w))
        sub = ctx.grid[r_min:r_max, c_min:c_max]
        non_bg = int(np.sum(sub != ctx.background_color))
        if non_bg < self.min_count:
            return False
        if self.max_count is not None and non_bg > self.max_count:
            return False
        return True

    def __repr__(self) -> str:
        return f"PixelDensityPredicate(min_count={self.min_count})"


class TileFramePredicate(SymbolicPredicate):
    """Evaluates whether a region contains periodic wireframe or square tile frames."""

    def __init__(
        self,
        frame_size: int = 7,
        min_frames: int = 2,
        y_stride: int = 1,
        y_start: int = 0,
        x_stride: int = 1,
        x_start: int = 0,
    ) -> None:
        self.frame_size = frame_size
        self.min_frames = min_frames
        self.y_stride = y_stride
        self.y_start = y_start
        self.x_stride = x_stride
        self.x_start = x_start

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        grid = ctx.grid
        bg = ctx.background_color
        H, W = grid.shape[-2:]
        sz = self.frame_size
        found = 0
        visited_origins: set[tuple[int, int]] = set()
        for y in range(self.y_start, H - sz + 1, self.y_stride):
            for x in range(self.x_start, W - sz + 1, self.x_stride):
                if any(abs(y - vy) < sz and abs(x - vx) < sz for vy, vx in visited_origins):
                    continue
                c = grid[y, x]
                if (
                    c != bg
                    and grid[y + sz - 1, x] == c
                    and grid[y, x + sz - 1] == c
                    and grid[y + sz - 1, x + sz - 1] == c
                ):
                    found += 1
                    visited_origins.add((y, x))
                    if found >= self.min_frames:
                        return True
        return found >= self.min_frames

    def __repr__(self) -> str:
        return (
            f"TileFramePredicate(size={self.frame_size}, min={self.min_frames}, "
            f"y_stride={self.y_stride}, y_start={self.y_start})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Declarative Subgoal Programs & Bytecode Compilation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SymbolicSubgoal:
    """A single declarative subgoal in the hierarchical skill program.

    Specifies the semantic intent, a structural query for the target entity
    or location, and optional preconditions/parameters.
    """

    intent: SpatialActionIntent
    target_query: dict[str, Any] = field(default_factory=dict)
    precondition: SymbolicPredicate | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_steps: int = 50

    def compile_to_instructions(self, author: str = "declarative_skill") -> list[Instruction]:
        """Compile this subgoal into canonical HCIR bytecode instructions."""
        instructions: list[Instruction] = []

        # 1. QUERY the graph for matching target entity
        if self.target_query:
            instructions.append(
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"query": dict(self.target_query)},
                    cost_estimate=10,
                )
            )

        # 2. ASSERT the active subgoal milestone
        instructions.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "intent": self.intent.value,
                    "target_query": self.target_query,
                    "timeout": self.timeout_steps,
                    "author": author,
                },
                cost_estimate=5,
            )
        )

        # 3. EXECUTE the declarative spatial intent
        instructions.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": f"spatial.{self.intent.value.lower()}",
                    "parameters": dict(self.parameters),
                },
                cost_estimate=25,
            )
        )
        return instructions


class SubgoalSequence:
    """An ordered sequence of declarative subgoals representing a skill program."""

    def __init__(self, *subgoals: SymbolicSubgoal) -> None:
        self.subgoals: list[SymbolicSubgoal] = list(subgoals)

    def __iter__(self):
        return iter(self.subgoals)

    def __len__(self) -> int:
        return len(self.subgoals)

    def __getitem__(self, index: int) -> SymbolicSubgoal:
        return self.subgoals[index]

    def compile_to_stream(self, skill_name: str = "declarative_skill") -> InstructionStream:
        """Compile all subgoals in the sequence into an HCIR InstructionStream."""
        stream = InstructionStream(
            author=skill_name,
            description=f"Compiled HCIR program for {skill_name}",
        )
        for subgoal in self.subgoals:
            for ins in subgoal.compile_to_instructions(author=skill_name):
                stream.append(ins)
        return stream


# ═══════════════════════════════════════════════════════════════════════════
# 4. Declarative Neuro-Symbolic Skill Base Class
# ═══════════════════════════════════════════════════════════════════════════


class DeclarativeNeuroSymbolicSkill(BaseHierarchicalSkill):
    """Base class for skills specified via declarative neuro-symbolic grammar.

    Subclasses define:
    - `signature`: A composable `SymbolicPredicate` defining invariant recognition.
    - `program`: An optional `SubgoalSequence` defining the high-level intent plan.
    - Subclasses can optionally implement `solve(ctx)` to synthesize concrete
      actions using core mathematical/spatial solvers.
    """

    signature: SymbolicPredicate = AllOf()
    program: SubgoalSequence | None = None

    def get_metadata(self) -> dict[str, Any]:
        """Optional hook for skill instances to contribute dynamic state to evaluation context."""
        return {}

    def can_handle(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate invariant recognition using the declarative symbolic signature."""
        meta = {**self.get_metadata(), **(metadata or {})}
        ctx = SkillEvaluationContext(
            grid=grid,
            available_actions=available_actions,
            metadata=meta,
        )
        try:
            return bool(self.signature.evaluate(ctx))
        except Exception as e:
            logger.debug(
                "Declarative skill '%s' error evaluating signature: %s",
                self.skill_name,
                e,
            )
            return False

    def compile_to_instruction_stream(
        self,
        grid: np.ndarray,
        available_actions: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> InstructionStream:
        """Compile this declarative skill into an executable HCIR InstructionStream."""
        if self.program is not None:
            return self.program.compile_to_stream(skill_name=self.skill_name)
        return InstructionStream(
            author=self.skill_name,
            description=f"Empty instruction stream for {self.skill_name}",
        )

    def solve(
        self, ctx: SkillEvaluationContext, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]] | list[int]:
        """Synthesize concrete actions from the lifted cognitive context.

        Override in subclass to provide procedural or symbolic solver logic.
        Default implementation returns an empty list.
        """
        return []

    def plan(
        self,
        grid: np.ndarray,
        current_level: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, dict[str, int] | None]] | list[int]:
        """Synthesize action sequence using the cognitive evaluation context."""
        ctx = SkillEvaluationContext(
            grid=grid,
            available_actions=[1, 2, 3, 4, 5, 6, 7],  # default superset
            metadata=metadata,
        )
        return self.solve(ctx, current_level=current_level)
