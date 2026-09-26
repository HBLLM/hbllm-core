"""ARC-AGI Benchmark Runner & Relational Grid Transformation Engine.

Evaluates developmental cognitive reasoning schemas (Stage D2 containment,
Stage D1 causal displacement, Stage D4 tool reach, and Stage D3 attribute mapping)
on the Abstraction and Reasoning Corpus (ARC-AGI) challenge.

Transforms 2D integer matrices into topological CognitiveGraph representations,
induces candidate transformation schemas from training pairs, and predicts test outputs
with exact match verification and calibrated Brier uncertainty scoring.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType, PhysicalEntityNode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARC Grid Representation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ARCGrid:
    """2D discrete integer grid representation for ARC tasks."""

    cells: list[list[int]]

    def __post_init__(self) -> None:
        if not self.cells or not self.cells[0]:
            self.height = 0
            self.width = 0
            return
        self.height = len(self.cells)
        self.width = len(self.cells[0])
        # Ensure rectangular grid
        for row in self.cells:
            if len(row) != self.width:
                raise ValueError(
                    f"Irregular grid dimensions: expected width {self.width}, got {len(row)}"
                )

    @classmethod
    def from_dimensions(cls, height: int, width: int, fill: int = 0) -> ARCGrid:
        """Create a new grid filled with a uniform color."""
        return cls([[fill for _ in range(width)] for _ in range(height)])

    @classmethod
    def from_list(cls, matrix: list[list[int]]) -> ARCGrid:
        """Create an ARCGrid deep-copying a 2D matrix."""
        return cls([[int(c) for c in row] for row in matrix])

    def get(self, r: int, c: int, default: int = 0) -> int:
        """Safe get cell value at (r, c)."""
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.cells[r][c]
        return default

    def set(self, r: int, c: int, val: int) -> None:
        """Set cell value at (r, c)."""
        if 0 <= r < self.height and 0 <= c < self.width:
            self.cells[r][c] = int(val)

    def to_list(self) -> list[list[int]]:
        """Export as 2D list of integers."""
        return [[c for c in row] for row in self.cells]

    def clone(self) -> ARCGrid:
        """Create a deep clone."""
        return ARCGrid([[c for c in row] for row in self.cells])

    def unique_colors(self) -> set[int]:
        """Return set of distinct colors present in the grid."""
        colors = set()
        for row in self.cells:
            colors.update(row)
        return colors

    def count_color(self, color: int) -> int:
        """Count occurrences of a specific color."""
        return sum(row.count(color) for row in self.cells)

    def find_coordinates(self, color: int) -> set[tuple[int, int]]:
        """Return all (r, c) coordinates matching color."""
        coords = set()
        for r in range(self.height):
            for c in range(self.width):
                if self.cells[r][c] == color:
                    coords.add((r, c))
        return coords

    def crop(self, min_r: int, min_c: int, max_r: int, max_c: int) -> ARCGrid:
        """Crop subgrid [min_r..max_r, min_c..max_c] inclusive."""
        min_r = max(0, min_r)
        min_c = max(0, min_c)
        max_r = min(self.height - 1, max_r)
        max_c = min(self.width - 1, max_c)
        if min_r > max_r or min_c > max_c:
            return ARCGrid.from_dimensions(0, 0)
        return ARCGrid(
            [[self.cells[r][c] for c in range(min_c, max_c + 1)] for r in range(min_r, max_r + 1)]
        )

    def pixel_accuracy(self, other: ARCGrid) -> float:
        """Compute exact pixel matching fraction against another grid."""
        if self.height != other.height or self.width != other.width:
            return 0.0
        total = self.height * self.width
        if total == 0:
            return 1.0
        matches = sum(
            1
            for r in range(self.height)
            for c in range(self.width)
            if self.cells[r][c] == other.cells[r][c]
        )
        return matches / total

    def brier_score(self, other: ARCGrid, confidence: float = 1.0) -> float:
        """Compute calibrated Brier error score."""
        is_exact = self == other
        target = 1.0 if is_exact else 0.0
        return (confidence - target) ** 2

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ARCGrid):
            return False
        if self.height != other.height or self.width != other.width:
            return False
        return self.cells == other.cells

    def render_ascii(self) -> str:
        """Render grid as human-readable ASCII."""
        symbols = {
            0: ".",
            1: "B",
            2: "R",
            3: "G",
            4: "Y",
            5: "X",
            6: "M",
            7: "O",
            8: "C",
            9: "K",
        }
        lines = []
        for row in self.cells:
            lines.append("".join(symbols.get(c, str(c)) for c in row))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Discrete Objects & Topological Graph Extraction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GridObject:
    """Discrete contiguous component in an ARC grid."""

    object_id: str
    color: int
    coords: set[tuple[int, int]]
    min_r: int
    max_r: int
    min_c: int
    max_c: int
    area: int
    centroid: tuple[float, float]
    is_frame: bool = False
    is_line: bool = False
    enclosed_coords: set[tuple[int, int]] = field(default_factory=set)

    @property
    def height(self) -> int:
        return self.max_r - self.min_r + 1

    @property
    def width(self) -> int:
        return self.max_c - self.min_c + 1

    def contains_point(self, r: int, c: int) -> bool:
        return (r, c) in self.coords

    def touches(self, other: GridObject) -> bool:
        """Check 4-neighborhood contact with another object."""
        for r, c in self.coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) in other.coords:
                    return True
        return False


class GridTopologyExtractor:
    """Extracts objects, boundaries, voids, and spatial graphs from ARC grids."""

    @staticmethod
    def extract_objects(
        grid: ARCGrid,
        background_color: int = 0,
        connectivity: int = 4,
    ) -> list[GridObject]:
        """Connected component labeling using BFS."""
        visited: set[tuple[int, int]] = set()
        objects: list[GridObject] = []

        deltas = (
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if connectivity == 4
            else [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        )

        for r in range(grid.height):
            for c in range(grid.width):
                color = grid.cells[r][c]
                if color == background_color or (r, c) in visited:
                    continue

                # BFS component search
                coords: set[tuple[int, int]] = set()
                queue = deque([(r, c)])
                visited.add((r, c))

                while queue:
                    curr_r, curr_c = queue.popleft()
                    coords.add((curr_r, curr_c))

                    for dr, dc in deltas:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (
                            0 <= nr < grid.height
                            and 0 <= nc < grid.width
                            and (nr, nc) not in visited
                            and grid.cells[nr][nc] == color
                        ):
                            visited.add((nr, nc))
                            queue.append((nr, nc))

                area = len(coords)
                if area == 1:
                    cr, cc = next(iter(coords))
                    min_r = max_r = cr
                    min_c = max_c = cc
                    centroid = (float(cr), float(cc))
                    h = w = 1
                    is_line = True
                    enclosed: set[tuple[int, int]] = set()
                    is_frame = False
                else:
                    min_r = min_c = 1000000
                    max_r = max_c = -1000000
                    sum_r = sum_c = 0
                    for cr, cc in coords:
                        if cr < min_r:
                            min_r = cr
                        if cr > max_r:
                            max_r = cr
                        if cc < min_c:
                            min_c = cc
                        if cc > max_c:
                            max_c = cc
                        sum_r += cr
                        sum_c += cc
                    centroid = (sum_r / area, sum_c / area)
                    h = max_r - min_r + 1
                    w = max_c - min_c + 1
                    is_line = (h == 1 and area == w) or (w == 1 and area == h)

                    # Check if it forms a closed frame / container
                    # A closed frame requires at least a 3x3 bounding perimeter (>= 8 pixels) and cannot be a straight line
                    if h >= 3 and w >= 3 and area >= 8 and not is_line:
                        enclosed = GridTopologyExtractor.detect_interior_void(
                            grid, coords, min_r=min_r, max_r=max_r, min_c=min_c, max_c=max_c
                        )
                        is_frame = len(enclosed) > 0
                    else:
                        enclosed = set()
                        is_frame = False

                obj = GridObject(
                    object_id=f"obj_{color}_{r}_{c}",
                    color=color,
                    coords=coords,
                    min_r=min_r,
                    max_r=max_r,
                    min_c=min_c,
                    max_c=max_c,
                    area=area,
                    centroid=centroid,
                    is_frame=is_frame,
                    is_line=is_line,
                    enclosed_coords=enclosed,
                )
                objects.append(obj)

        return objects

    @staticmethod
    def detect_interior_void(
        grid: ARCGrid,
        boundary_coords: set[tuple[int, int]],
        min_r: int | None = None,
        max_r: int | None = None,
        min_c: int | None = None,
        max_c: int | None = None,
    ) -> set[tuple[int, int]]:
        """Detect internal void pixels enclosed by a set of boundary coordinates."""
        if not boundary_coords:
            return set()

        if min_r is None or max_r is None or min_c is None or max_c is None:
            min_r = min(cr for cr, _ in boundary_coords)
            max_r = max(cr for cr, _ in boundary_coords)
            min_c = min(cc for _, cc in boundary_coords)
            max_c = max(cc for _, cc in boundary_coords)

        # If boundary doesn't have an area >= 3x3, it cannot enclose anything
        if (max_r - min_r < 2) or (max_c - min_c < 2):
            return set()

        # Flood fill from the outer bounding box edge to find outside void pixels
        # Any pixel inside the bounding box not reachable from the bbox perimeter is interior!
        bbox_pixels = {
            (r, c)
            for r in range(min_r, max_r + 1)
            for c in range(min_c, max_c + 1)
            if (r, c) not in boundary_coords
        }

        # Start flood fill from exterior points around the bounding box
        outside: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque()

        for r in range(min_r, max_r + 1):
            for c in (min_c, max_c):
                if (r, c) in bbox_pixels:
                    outside.add((r, c))
                    queue.append((r, c))
        for c in range(min_c, max_c + 1):
            for r in (min_r, max_r):
                if (r, c) in bbox_pixels and (r, c) not in outside:
                    outside.add((r, c))
                    queue.append((r, c))

        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (
                    min_r <= nr <= max_r
                    and min_c <= nc <= max_c
                    and (nr, nc) in bbox_pixels
                    and (nr, nc) not in outside
                ):
                    outside.add((nr, nc))
                    queue.append((nr, nc))

        # Interior void is bbox pixels that were not reached by outside flood fill
        interior_void = bbox_pixels - outside
        return interior_void

    @staticmethod
    def to_cognitive_graph(grid: ARCGrid, background_color: int = 0) -> CognitiveGraph:
        """Lift ARCGrid objects and spatial relationships into HCIR CognitiveGraph."""
        graph = CognitiveGraph()
        objects = GridTopologyExtractor.extract_objects(grid, background_color=background_color)

        for obj in objects:
            node = PhysicalEntityNode(
                id=obj.object_id,
                entity_name=f"grid_obj_{obj.color}",
                entity_type="frame" if obj.is_frame else ("line" if obj.is_line else "blob"),
                properties={
                    "color": obj.color,
                    "area": obj.area,
                    "centroid": obj.centroid,
                    "is_frame": obj.is_frame,
                    "is_line": obj.is_line,
                    "bounds": (obj.min_r, obj.min_c, obj.max_r, obj.max_c),
                    "enclosed_count": len(obj.enclosed_coords),
                },
            )
            graph.add_node(node)

        # Relational Edges: Adjacency and Containment
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                o1, o2 = objects[i], objects[j]

                # Check contact/adjacency
                if o1.touches(o2):
                    edge = HCIREdge(
                        id=f"edge_touch_{o1.object_id}_{o2.object_id}",
                        edge_type=HCIREdgeType.SUPPORTS,
                        sources=[o1.object_id],
                        targets=[o2.object_id],
                        properties={"relation": "TOUCHING"},
                    )
                    graph.add_edge(edge)

                # Check containment: is o2 inside o1's enclosed void?
                if o1.is_frame and any(pt in o1.enclosed_coords for pt in o2.coords):
                    edge = HCIREdge(
                        id=f"edge_inside_{o2.object_id}_{o1.object_id}",
                        edge_type=HCIREdgeType.PART_OF,
                        sources=[o2.object_id],
                        targets=[o1.object_id],
                        properties={"relation": "INSIDE", "container": o1.object_id},
                    )
                    graph.add_edge(edge)

                elif o2.is_frame and any(pt in o2.enclosed_coords for pt in o1.coords):
                    edge = HCIREdge(
                        id=f"edge_inside_{o1.object_id}_{o2.object_id}",
                        edge_type=HCIREdgeType.PART_OF,
                        sources=[o1.object_id],
                        targets=[o2.object_id],
                        properties={"relation": "INSIDE", "container": o2.object_id},
                    )
                    graph.add_edge(edge)

        return graph


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inductive Relational Schemas & ARC Solver
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ARCTaskResult:
    """Outcome of solving a single ARC-AGI task."""

    task_id: str
    exact_match: bool
    pixel_accuracy: float
    brier_score: float
    confidence: float
    schema_name: str
    predicted_output: list[list[int]]
    ground_truth_output: list[list[int]] | None = None


@dataclass
class ARCBenchmarkReport:
    """Consolidated ARC-AGI benchmark results across multiple tasks."""

    total_tasks: int
    tasks_solved: int
    exact_accuracy: float
    mean_pixel_accuracy: float
    mean_brier_score: float
    task_results: list[ARCTaskResult] = field(default_factory=list)

    def format_markdown(self) -> str:
        lines = [
            "# ARC-AGI Cognitive Relational Benchmark Report",
            f"**Tasks Solved (Exact Match)**: {self.tasks_solved}/{self.total_tasks} ({self.exact_accuracy * 100:.1f}%)",
            f"**Mean Pixel Accuracy**: {self.mean_pixel_accuracy * 100:.2f}%",
            f"**Mean Calibrated Brier Score**: {self.mean_brier_score:.4f}",
            "",
            "| Task ID | Schema | Exact Match | Pixel Acc | Brier |",
            "|---|---|---|---|---|",
        ]
        for r in self.task_results:
            status = "PASSED" if r.exact_match else "FAILED"
            lines.append(
                f"| `{r.task_id}` | {r.schema_name} | **{status}** | {r.pixel_accuracy * 100:.1f}% | {r.brier_score:.4f} |"
            )
        return "\n".join(lines)


class ARCRelationalSolver:
    """Inductive solver applying developmental schemas to ARC tasks."""

    def __init__(self) -> None:
        self.known_schemas = [
            self._try_boundary_containment_fill,
            self._try_tool_reach_ray_extension,
            self._try_obstacle_clearance_displacement,
            self._try_attribute_color_mapping,
            self._try_tiling_inversion_symmetry,
        ]

    def solve_task(self, task: dict[str, Any], task_id: str = "arc_task") -> ARCTaskResult:
        """Induce transformation schema from train pairs and apply to test pair."""
        train_pairs = [
            (ARCGrid.from_list(p["input"]), ARCGrid.from_list(p["output"])) for p in task["train"]
        ]
        test_inputs = [ARCGrid.from_list(p["input"]) for p in task["test"]]
        test_ground_truth = (
            [ARCGrid.from_list(p["output"]) for p in task["test"]]
            if "output" in task["test"][0]
            else None
        )

        best_schema = None
        best_schema_name = "unknown"
        best_train_acc = -1.0

        for schema_fn in self.known_schemas:
            name, fn = schema_fn(train_pairs)
            if fn is not None:
                # Validate across all train pairs
                accs = []
                for in_grid, out_grid in train_pairs:
                    pred = fn(in_grid)
                    accs.append(pred.pixel_accuracy(out_grid))
                avg_acc = sum(accs) / len(accs) if accs else 0.0

                if avg_acc > best_train_acc:
                    best_train_acc = avg_acc
                    best_schema = fn
                    best_schema_name = name

                if avg_acc >= 1.0:
                    # Perfect fit on all training pairs!
                    break

        if best_schema is None or best_train_acc < 0.5:
            # Fallback: identity or unhandled
            def _identity(g: ARCGrid) -> ARCGrid:
                return g.clone()

            best_schema = _identity
            best_schema_name = "identity_fallback"
            confidence = 0.1
        else:
            confidence = min(0.99, max(0.6, best_train_acc))

        # Predict test output
        test_pred = best_schema(test_inputs[0])

        if test_ground_truth:
            gt = test_ground_truth[0]
            exact = test_pred == gt
            pixel_acc = test_pred.pixel_accuracy(gt)
            brier = test_pred.brier_score(gt, confidence=confidence)
            gt_list = gt.to_list()
        else:
            exact = False
            pixel_acc = 0.0
            brier = (confidence - 0.5) ** 2
            gt_list = None

        return ARCTaskResult(
            task_id=task_id,
            exact_match=exact,
            pixel_accuracy=pixel_acc,
            brier_score=brier,
            confidence=confidence,
            schema_name=best_schema_name,
            predicted_output=test_pred.to_list(),
            ground_truth_output=gt_list,
        )

    def run_benchmark(self, tasks: dict[str, dict[str, Any]]) -> ARCBenchmarkReport:
        """Run evaluation over a collection of ARC tasks."""
        results = []
        for task_id, task in tasks.items():
            res = self.solve_task(task, task_id=task_id)
            results.append(res)

        total = len(results)
        solved = sum(1 for r in results if r.exact_match)
        exact_acc = solved / total if total > 0 else 0.0
        mean_pix = sum(r.pixel_accuracy for r in results) / total if total > 0 else 0.0
        mean_brier = sum(r.brier_score for r in results) / total if total > 0 else 0.0

        return ARCBenchmarkReport(
            total_tasks=total,
            tasks_solved=solved,
            exact_accuracy=exact_acc,
            mean_pixel_accuracy=mean_pix,
            mean_brier_score=mean_brier,
            task_results=results,
        )

    # ── Schema 1: Topological Boundary Containment Fill (Stage D2) ───────────

    def _try_boundary_containment_fill(
        self, train_pairs: list[tuple[ARCGrid, ARCGrid]]
    ) -> tuple[str, Any]:
        """Schema: Interior void enclosed by boundary walls of color C is filled with color F."""
        fill_candidates: list[int] = []

        for in_grid, out_grid in train_pairs:
            # Detect objects and frames in in_grid
            objs = GridTopologyExtractor.extract_objects(in_grid)
            frames = [o for o in objs if o.is_frame]
            if not frames:
                return "containment_fill", None

            # Look for what was filled in out_grid inside frame's enclosed void
            for frame in frames:
                fill_colors = {
                    out_grid.get(r, c)
                    for r, c in frame.enclosed_coords
                    if in_grid.get(r, c) != out_grid.get(r, c)
                }
                if fill_colors:
                    fill_candidates.extend(list(fill_colors))

        if not fill_candidates:
            return "containment_fill", None

        # Most common fill color across demonstrations
        target_fill = max(set(fill_candidates), key=fill_candidates.count)

        def transform(grid: ARCGrid) -> ARCGrid:
            out = grid.clone()
            objs = GridTopologyExtractor.extract_objects(grid)
            for obj in objs:
                if obj.is_frame:
                    for r, c in obj.enclosed_coords:
                        if grid.get(r, c) == 0:  # Fill empty interior void
                            out.set(r, c, target_fill)
            return out

        return "boundary_containment_fill", transform

    # ── Schema 2: Tool Reach & Ray Extension (Stage D4) ──────────────────────

    def _try_tool_reach_ray_extension(
        self, train_pairs: list[tuple[ARCGrid, ARCGrid]]
    ) -> tuple[str, Any]:
        """Schema: Ray/beam extends from tool/pointer in cardinal directions until hitting obstacle."""
        detected_directions: list[tuple[int, int]] = []
        ray_colors: list[int] = []
        anchor_colors: list[int] = []

        for in_grid, out_grid in train_pairs:
            in_objs = GridTopologyExtractor.extract_objects(in_grid)
            for obj in in_objs:
                if obj.area <= 4:  # Small anchor or tool marker
                    # Look if out_grid extended rays from this obj
                    r_c, c_c = int(obj.centroid[0]), int(obj.centroid[1])
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        steps = 1
                        while True:
                            nr, nc = r_c + dr * steps, c_c + dc * steps
                            if not (0 <= nr < in_grid.height and 0 <= nc < in_grid.width):
                                break
                            if in_grid.get(nr, nc) == 0 and out_grid.get(nr, nc) != 0:
                                detected_directions.append((dr, dc))
                                ray_colors.append(out_grid.get(nr, nc))
                                anchor_colors.append(obj.color)
                            else:
                                break
                            steps += 1

        if not detected_directions or not ray_colors or not anchor_colors:
            return "tool_reach_ray_extension", None

        best_color = max(set(ray_colors), key=ray_colors.count)
        best_anchor_color = max(set(anchor_colors), key=anchor_colors.count)
        best_dirs = list(set(detected_directions))

        def transform(grid: ARCGrid) -> ARCGrid:
            out = grid.clone()
            objs = GridTopologyExtractor.extract_objects(grid)
            for obj in objs:
                if obj.color == best_anchor_color:
                    r_c, c_c = int(obj.centroid[0]), int(obj.centroid[1])
                    for dr, dc in best_dirs:
                        steps = 1
                        while True:
                            nr, nc = r_c + dr * steps, c_c + dc * steps
                            if not (0 <= nr < grid.height and 0 <= nc < grid.width):
                                break
                            if grid.get(nr, nc) != 0:
                                break  # Blocked by obstacle or wall
                            out.set(nr, nc, best_color)
                            steps += 1
            return out

        return "tool_reach_ray_extension", transform

    # ── Schema 3: Obstacle Clearance & Displacement (Stage D1/D2) ────────────

    def _try_obstacle_clearance_displacement(
        self, train_pairs: list[tuple[ARCGrid, ARCGrid]]
    ) -> tuple[str, Any]:
        """Schema: Object translates along an axis until resting against obstacle or boundary."""
        displacements: list[tuple[int, int]] = []

        for in_grid, out_grid in train_pairs:
            in_objs = GridTopologyExtractor.extract_objects(in_grid)
            for obj in in_objs:
                # Find matching object in out_grid by color and area
                matching_coords = {
                    (r, c)
                    for r in range(out_grid.height)
                    for c in range(out_grid.width)
                    if out_grid.cells[r][c] == obj.color
                }
                if (
                    matching_coords
                    and len(matching_coords) == obj.area
                    and matching_coords != obj.coords
                ):
                    min_r_out = min(r for r, _ in matching_coords)
                    min_c_out = min(c for _, c in matching_coords)
                    dr = min_r_out - obj.min_r
                    dc = min_c_out - obj.min_c
                    if dr != 0 or dc != 0:
                        displacements.append((dr, dc))

        if not displacements:
            return "obstacle_clearance_displacement", None

        # Displace in direction of movement until collision
        common_dr = (
            1
            if any(dr > 0 for dr, _ in displacements)
            else (-1 if any(dr < 0 for dr, _ in displacements) else 0)
        )
        common_dc = (
            1
            if any(dc > 0 for _, dc in displacements)
            else (-1 if any(dc < 0 for _, dc in displacements) else 0)
        )

        def transform(grid: ARCGrid) -> ARCGrid:
            out = grid.clone()
            objs = GridTopologyExtractor.extract_objects(grid)
            # Find mobile objects (e.g. area < max area)
            if not objs:
                return out
            max_area = max(o.area for o in objs)
            for obj in objs:
                if obj.area < max_area:  # Mobile payload
                    # Erase original
                    for r, c in obj.coords:
                        out.set(r, c, 0)
                    # Displace until obstacle hit
                    curr_coords = set(obj.coords)
                    while True:
                        next_coords = {(r + common_dr, c + common_dc) for r, c in curr_coords}
                        # Check boundary
                        if any(
                            not (0 <= nr < grid.height and 0 <= nc < grid.width)
                            for nr, nc in next_coords
                        ):
                            break
                        # Check collision with other non-empty cells
                        if any(
                            grid.get(nr, nc) != 0 and (nr, nc) not in obj.coords
                            for nr, nc in next_coords
                        ):
                            break
                        curr_coords = next_coords
                    # Write at resting position
                    for r, c in curr_coords:
                        out.set(r, c, obj.color)
            return out

        return "obstacle_clearance_displacement", transform

    # ── Schema 4: Attribute / Color Mapping (Stage D3/D5) ─────────────────────

    def _try_attribute_color_mapping(
        self, train_pairs: list[tuple[ARCGrid, ARCGrid]]
    ) -> tuple[str, Any]:
        """Schema: 1-to-1 or condition-based color substitution."""
        color_map: dict[int, int] = {}
        consistent = True

        for in_grid, out_grid in train_pairs:
            if in_grid.height != out_grid.height or in_grid.width != out_grid.width:
                return "attribute_color_mapping", None
            for r in range(in_grid.height):
                for c in range(in_grid.width):
                    cin = in_grid.get(r, c)
                    cout = out_grid.get(r, c)
                    if cin in color_map and color_map[cin] != cout:
                        consistent = False
                        break
                    color_map[cin] = cout
            if not consistent:
                break

        if not consistent or not color_map or all(k == v for k, v in color_map.items()):
            return "attribute_color_mapping", None

        def transform(grid: ARCGrid) -> ARCGrid:
            out = grid.clone()
            for r in range(grid.height):
                for c in range(grid.width):
                    cin = grid.get(r, c)
                    out.set(r, c, color_map.get(cin, cin))
            return out

        return "attribute_color_mapping", transform

    # ── Schema 5: Tiling, Inversion & Symmetry ────────────────────────────────

    def _try_tiling_inversion_symmetry(
        self, train_pairs: list[tuple[ARCGrid, ARCGrid]]
    ) -> tuple[str, Any]:
        """Schema: Horizontal/Vertical flip or 180-degree rotation symmetry."""
        in_grid, out_grid = train_pairs[0]
        if in_grid.height != out_grid.height or in_grid.width != out_grid.width:
            return "symmetry_inversion", None

        # Check horizontal reflection
        is_hflip = all(p[0].clone().cells[::-1] == p[1].cells for p in train_pairs)
        if is_hflip:
            return "horizontal_flip", lambda g: ARCGrid(g.cells[::-1])

        # Check vertical reflection
        is_vflip = all([row[::-1] for row in p[0].cells] == p[1].cells for p in train_pairs)
        if is_vflip:
            return "vertical_flip", lambda g: ARCGrid([row[::-1] for row in g.cells])

        return "symmetry_inversion", None
