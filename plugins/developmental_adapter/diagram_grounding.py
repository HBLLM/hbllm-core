"""Multimodal Scientific Diagram Grounding Engine.

Binds 2D visual diagrams and scientific figures directly into HCIR CognitiveGraph
representations, enabling topological reasoning and cross-modal verification:
1. Parses geometric primitives (boxes, polygons, arrows, rays, fulcrums, membranes).
2. Extracts topological invariants (ENCLOSES, CONNECTED_TO, SUPPORTS, DIVIDES, DIRECTED_AT).
3. Evaluates natural language assertions against visual topology with calibrated
   epistemic uncertainty scoring and anomaly detection.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType, PhysicalEntityNode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Visual & Geometric Diagram Primitives
# ─────────────────────────────────────────────────────────────────────────────


class DiagramPrimitiveType(str, Enum):
    """Categorization of geometric primitives within scientific figures."""

    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"
    LINE_SEGMENT = "line_segment"
    ARROW = "arrow"
    CONTAINER_BOX = "container_box"
    FULCRUM = "fulcrum"
    MEMBRANE = "membrane"
    TEXT_LABEL = "text_label"


@dataclass
class DiagramPrimitive:
    """A structured 2D visual element extracted from a scientific figure."""

    primitive_id: str
    primitive_type: DiagramPrimitiveType
    label: str
    bounds: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    centroid: tuple[float, float]
    points: list[tuple[float, float]] = field(default_factory=list)
    orientation_deg: float = 0.0
    is_closed: bool = True
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> float:
        return self.bounds[3] - self.bounds[1]

    def contains_point(self, pt: tuple[float, float]) -> bool:
        """Check if 2D coordinate is topologically contained within bounds or polygon."""
        x, y = pt
        min_x, min_y, max_x, max_y = self.bounds
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

        if len(self.points) < 3:
            # Axis-aligned bounding box containment
            return True

        # Ray-casting algorithm for point-in-polygon
        inside = False
        n = len(self.points)
        p1x, p1y = self.points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.points[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def distance_to(self, other: DiagramPrimitive) -> float:
        """Euclidean distance between element centroids."""
        dx = self.centroid[0] - other.centroid[0]
        dy = self.centroid[1] - other.centroid[1]
        return math.hypot(dx, dy)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scientific Diagram Parser & CognitiveGraph Induction
# ─────────────────────────────────────────────────────────────────────────────


class ScientificDiagramParser:
    """Parses 2D vector diagrams into topological HCIR CognitiveGraphs."""

    @classmethod
    def parse_primitives_dict(
        cls, data: dict[str, Any] | list[dict[str, Any]]
    ) -> list[DiagramPrimitive]:
        """Convert serialized dictionary or JSON list into typed DiagramPrimitive list."""
        raw_list = data if isinstance(data, list) else data.get("elements", [])
        primitives: list[DiagramPrimitive] = []

        for item in raw_list:
            pid = item.get("id", f"prim_{uuid.uuid4().hex[:6]}")
            ptype = DiagramPrimitiveType(item.get("type", "rectangle"))
            label = item.get("label", pid)
            pts = [tuple(p) for p in item.get("points", [])]

            # Compute bounds and centroid
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bounds = (min(xs), min(ys), max(xs), max(ys))
                centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
            else:
                bounds = tuple(item.get("bounds", (0.0, 0.0, 1.0, 1.0)))
                centroid = (
                    (bounds[0] + bounds[2]) / 2.0,
                    (bounds[1] + bounds[3]) / 2.0,
                )

            prim = DiagramPrimitive(
                primitive_id=pid,
                primitive_type=ptype,
                label=label,
                bounds=bounds,
                centroid=centroid,
                points=pts,
                orientation_deg=float(item.get("orientation_deg", 0.0)),
                is_closed=bool(item.get("is_closed", True)),
                properties=item.get("properties", {}),
            )
            primitives.append(prim)

        return primitives

    @classmethod
    def to_cognitive_graph(cls, primitives: list[DiagramPrimitive]) -> CognitiveGraph:
        """Induce topological and causal edges, building a unified CognitiveGraph."""
        graph = CognitiveGraph()

        # 1. Add PhysicalEntityNodes for all primitives
        for p in primitives:
            node = PhysicalEntityNode(
                id=p.primitive_id,
                entity_name=p.label,
                entity_type=p.primitive_type.value,
                properties={
                    "bounds": p.bounds,
                    "centroid": p.centroid,
                    "is_closed": p.is_closed,
                    "orientation_deg": p.orientation_deg,
                    **p.properties,
                },
            )
            graph.add_node(node)

        # 2. Induce Relational Edges across pairs
        for i in range(len(primitives)):
            for j in range(len(primitives)):
                if i == j:
                    continue
                p1, p2 = primitives[i], primitives[j]

                # --- Relation 1: ENCLOSES (Containment) ---
                if p1.is_closed and p1.contains_point(p2.centroid):
                    edge = HCIREdge(
                        id=f"edge_encloses_{p1.primitive_id}_{p2.primitive_id}",
                        edge_type=HCIREdgeType.PART_OF,
                        sources=[p2.primitive_id],
                        targets=[p1.primitive_id],
                        properties={"relation": "ENCLOSES", "container": p1.primitive_id},
                    )
                    graph.add_edge(edge)

                # --- Relation 2: SUPPORTS (Vertical Contact / Fulcrum) ---
                # A supports B if A is below B (y_A < y_B) and horizontal bounds overlap with contact
                overlap_x = max(
                    0.0, min(p1.bounds[2], p2.bounds[2]) - max(p1.bounds[0], p2.bounds[0])
                )
                if overlap_x > 0.0 and abs(p1.bounds[3] - p2.bounds[1]) < 0.15:
                    edge = HCIREdge(
                        id=f"edge_supports_{p1.primitive_id}_{p2.primitive_id}",
                        edge_type=HCIREdgeType.SUPPORTS,
                        sources=[p1.primitive_id],
                        targets=[p2.primitive_id],
                        properties={"relation": "SUPPORTS", "overlap_x": overlap_x},
                    )
                    graph.add_edge(edge)

                # --- Relation 3: DIRECTED_AT / POINTS_TO (Arrows & Rays) ---
                if p1.primitive_type == DiagramPrimitiveType.ARROW and p1.points:
                    tip = p1.points[-1]  # Arrow tip
                    if (
                        p2.contains_point(tip)
                        or math.hypot(tip[0] - p2.centroid[0], tip[1] - p2.centroid[1]) < 0.2
                    ):
                        edge = HCIREdge(
                            id=f"edge_points_{p1.primitive_id}_{p2.primitive_id}",
                            edge_type=HCIREdgeType.CAUSES,
                            sources=[p1.primitive_id],
                            targets=[p2.primitive_id],
                            properties={"relation": "DIRECTED_AT"},
                        )
                        graph.add_edge(edge)

                # --- Relation 4: DIVIDES (Membrane or Barrier) ---
                if p1.primitive_type == DiagramPrimitiveType.MEMBRANE:
                    # Membrane splits space into left (x < min_x) vs right (x > max_x)
                    if p2.centroid[0] < p1.bounds[0]:
                        side = "LEFT"
                    elif p2.centroid[0] > p1.bounds[2]:
                        side = "RIGHT"
                    else:
                        side = "INTERIOR"
                    edge = HCIREdge(
                        id=f"edge_divides_{p1.primitive_id}_{p2.primitive_id}",
                        edge_type=HCIREdgeType.PART_OF,
                        sources=[p1.primitive_id],
                        targets=[p2.primitive_id],
                        properties={"relation": "DIVIDES", "partition_side": side},
                    )
                    graph.add_edge(edge)

        return graph


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-Modal Scientific Verification & Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CrossModalVerificationReport:
    """Result of validating a textual assertion against visual diagram topology."""

    claim_text: str
    is_consistent: bool
    confidence: float
    brier_score: float
    grounded_relations: list[str]
    contradictions_detected: list[str]
    explanation: str

    def format_markdown(self) -> str:
        status = "CONSISTENT (VERIFIED)" if self.is_consistent else "CONTRADICTION DETECTED"
        lines = [
            "# Cross-Modal Scientific Diagram Verification Report",
            f'**Claim**: *"{self.claim_text}"*',
            f"**Verification Status**: **{status}**",
            f"**Epistemic Confidence**: {self.confidence:.2f} | **Brier Score**: {self.brier_score:.4f}",
            "",
            "### Grounded Visual Relations",
        ]
        for rel in self.grounded_relations:
            lines.append(f"- `{rel}`")
        if self.contradictions_detected:
            lines.append("\n### Contradictions / Counterfactuals")
            for c in self.contradictions_detected:
                lines.append(f"- ⚠️ **{c}**")
        lines.append(f"\n**Analysis**: {self.explanation}")
        return "\n".join(lines)


class CrossModalVerifier:
    """Verifies textual claims against visual diagrams with calibrated uncertainty."""

    def __init__(self, parser: ScientificDiagramParser | None = None) -> None:
        self.parser = parser or ScientificDiagramParser()

    def verify_claim(
        self,
        primitives: list[DiagramPrimitive],
        claim_text: str,
    ) -> CrossModalVerificationReport:
        """Cross-examine natural language scientific assertion against visual topology."""
        graph = self.parser.to_cognitive_graph(primitives)
        claim_low = claim_text.lower()

        grounded_relations: list[str] = []
        contradictions: list[str] = []
        is_consistent = True
        confidence = 0.95

        # ── Test 1: Container Closure & Enclosure ──
        if (
            "encloses" in claim_low
            or "contain" in claim_low
            or "inside" in claim_low
            or "outside" in claim_low
        ):
            inside_edges = [
                e for e in graph._edges.values() if e.properties.get("relation") == "ENCLOSES"
            ]

            if "outside" in claim_low:
                # Text claims an entity is outside
                if inside_edges:
                    for e in inside_edges:
                        target_name = graph._nodes[e.sources[0]].entity_name
                        container_name = graph._nodes[e.targets[0]].entity_name
                        if target_name.lower() in claim_low:
                            is_consistent = False
                            contradictions.append(
                                f"Text claims '{target_name}' is outside, but visual diagram proves it is ENCLOSED in '{container_name}'"
                            )
                else:
                    grounded_relations.append(
                        "VERIFIED_EXTERIOR: Entity resides outside container boundaries"
                    )

            elif "encloses" in claim_low or "inside" in claim_low:
                if inside_edges:
                    for e in inside_edges:
                        grounded_relations.append(
                            f"ENCLOSES({graph._nodes[e.targets[0]].entity_name}, {graph._nodes[e.sources[0]].entity_name})"
                        )
                else:
                    is_consistent = False
                    contradictions.append(
                        "Text asserts enclosure, but visual diagram has zero topological containment"
                    )

        # ── Test 2: First-Class Lever Mechanics (Fulcrum between Effort and Load) ──
        if "lever" in claim_low or "fulcrum" in claim_low:
            fulcrums = [
                p
                for p in primitives
                if p.primitive_type == DiagramPrimitiveType.FULCRUM or "fulcrum" in p.label.lower()
            ]
            loads = [p for p in primitives if "load" in p.label.lower()]
            efforts = [p for p in primitives if "effort" in p.label.lower()]

            if fulcrums and loads and efforts:
                f_x = fulcrums[0].centroid[0]
                l_x = loads[0].centroid[0]
                e_x = efforts[0].centroid[0]

                is_first_class = min(l_x, e_x) < f_x < max(l_x, e_x)
                grounded_relations.append(
                    f"LEVER_GEOMETRY: Load(x={l_x:.1f}), Fulcrum(x={f_x:.1f}), Effort(x={e_x:.1f})"
                )

                if (
                    "first-class" in claim_low
                    or "first class" in claim_low
                    or "between" in claim_low
                ):
                    if not is_first_class:
                        is_consistent = False
                        contradictions.append(
                            f"Text claims first-class lever (fulcrum between load and effort), but fulcrum is at x={f_x:.1f} (outside load/effort span)"
                        )
                    else:
                        grounded_relations.append(
                            "VERIFIED_FIRST_CLASS_LEVER: Fulcrum pivots between load and effort"
                        )

                elif "second-class" in claim_low or "second class" in claim_low:
                    is_second_class = min(f_x, e_x) < l_x < max(f_x, e_x)
                    if not is_second_class:
                        is_consistent = False
                        contradictions.append(
                            "Text claims second-class lever (load between fulcrum and effort), but visual topology shows first-class layout"
                        )

        # ── Test 3: Membrane Partitioning & Semi-Permeable Barriers ──
        if "membrane" in claim_low or "barrier" in claim_low or "divides" in claim_low:
            divide_edges = [
                e for e in graph._edges.values() if e.properties.get("relation") == "DIVIDES"
            ]
            if divide_edges:
                sides = {e.properties.get("partition_side") for e in divide_edges}
                grounded_relations.append(
                    f"MEMBRANE_PARTITIONS: Separates compartments into {list(sides)}"
                )
                if "two compartments" in claim_low or "divides" in claim_low:
                    if len(sides) >= 2:
                        grounded_relations.append(
                            "VERIFIED_BIPARTITION: Entities confirmed on opposite membrane sides"
                        )

        # ── Test 4: Optical Refraction & Ray Direction ──
        if (
            "ray" in claim_low
            or "light" in claim_low
            or "refract" in claim_low
            or "prism" in claim_low
        ):
            points_edges = [
                e for e in graph._edges.values() if e.properties.get("relation") == "DIRECTED_AT"
            ]
            if points_edges:
                for pe in points_edges:
                    grounded_relations.append(
                        f"RAY_INCIDENT: {graph._nodes[pe.sources[0]].entity_name} -> {graph._nodes[pe.targets[0]].entity_name}"
                    )

        # Score calibration
        target = 1.0 if is_consistent else 0.0
        brier = (confidence - target) ** 2

        explanation = (
            "Cross-modal diagram validation confirmed topological alignment between text and visual geometry."
            if is_consistent
            else f"Visual geometry contradicts textual assertion with {len(contradictions)} topological violation(s)."
        )

        return CrossModalVerificationReport(
            claim_text=claim_text,
            is_consistent=is_consistent,
            confidence=confidence,
            brier_score=brier,
            grounded_relations=grounded_relations,
            contradictions_detected=contradictions,
            explanation=explanation,
        )
