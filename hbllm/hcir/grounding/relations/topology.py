from __future__ import annotations

from enum import Enum


class TopologicalInvariant(str, Enum):
    """Topological and spatial invariants emergent between physical entities."""

    DISJOINT = "disjoint"  # A ∩ B = ∅, distance > 0
    TOUCHING = "touching"  # Boundary contact without penetration
    OVERLAPPING = "overlapping"  # A ∩ B ≠ ∅, 0 < Vol(A ∩ B) < min(Vol(A), Vol(B))
    CONTAINED = "contained"  # A ⊆ B, A is inside B
    CONTAINS = "contains"  # A ⊇ B, A encapsulates B
    ABOVE_SUPPORTED = "above_supported"  # A rests on top boundary of B
    BELOW_SUPPORTING = "below_supporting"  # B rests on top boundary of A
    SYNCHRONOUS_MOTION = "synchronous_motion"  # Trajectories move in unison with stable d
    APPROACHING = "approaching"  # Relative distance is strictly decreasing
    RECEDING = "receding"  # Relative distance is strictly increasing


class TopologyClassifier:
    """Calculates spatial and topological invariants from geometry and coordinate bounds.

    Operates deterministically on bounding boxes and spatial hulls without LLMs.
    """

    @staticmethod
    def classify_2d_boxes(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
        tolerance: float = 1e-2,
    ) -> tuple[TopologicalInvariant, float]:
        """Classifies topological relationship between two 2D boxes (x, y, w, h).

        Returns:
            Tuple of (TopologicalInvariant, metric_ratio) where metric_ratio
            represents containment or overlap fraction in [0.0, 1.0].
        """
        x_a, y_a, w_a, h_a = box_a
        x_b, y_b, w_b, h_b = box_b

        area_a = max(1e-6, w_a * h_a)
        area_b = max(1e-6, w_b * h_b)

        x_left = max(x_a, x_b)
        x_right = min(x_a + w_a, x_b + w_b)
        y_bottom = max(y_a, y_b)
        y_top = min(y_a + h_a, y_b + h_b)

        overlap_w = x_right - x_left
        overlap_h = y_top - y_bottom

        if overlap_w > tolerance and overlap_h > tolerance:
            # Overlap or containment
            inter_area = overlap_w * overlap_h
            contain_a = inter_area / area_a
            contain_b = inter_area / area_b

            if contain_a >= 0.95:
                return (TopologicalInvariant.CONTAINED, contain_a)
            elif contain_b >= 0.95:
                return (TopologicalInvariant.CONTAINS, contain_b)
            else:
                return (TopologicalInvariant.OVERLAPPING, inter_area / min(area_a, area_b))

        # Check touching or vertical resting support
        # A resting on top of B: bottom of A touches top of B with horizontal overlap
        if abs(y_a - (y_b + h_b)) <= tolerance and overlap_w > tolerance:
            support_ratio = overlap_w / max(1e-6, w_a)
            return (TopologicalInvariant.ABOVE_SUPPORTED, min(1.0, support_ratio))

        # A supporting B from below: top of A touches bottom of B
        if abs((y_a + h_a) - y_b) <= tolerance and overlap_w > tolerance:
            support_ratio = overlap_w / max(1e-6, w_b)
            return (TopologicalInvariant.BELOW_SUPPORTING, min(1.0, support_ratio))

        # General boundary touching
        touching_horiz = (
            abs(overlap_w) <= tolerance
            and (y_a <= y_b + h_b + tolerance)
            and (y_a + h_a >= y_b - tolerance)
        )
        touching_vert = (
            abs(overlap_h) <= tolerance
            and (x_a <= x_b + w_b + tolerance)
            and (x_a + w_a >= x_b - tolerance)
        )

        if touching_horiz or touching_vert:
            return (TopologicalInvariant.TOUCHING, 1.0)

        # Distance between disjoint boxes
        dx = max(0.0, x_left - x_right)
        dy = max(0.0, y_bottom - y_top)
        dist = (dx**2 + dy**2) ** 0.5
        return (TopologicalInvariant.DISJOINT, dist)
