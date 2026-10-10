"""Spatiotemporal Hazard Tracking & Phase Precession System.

Modeled on mammalian entorhinal grid cell phase precession and hippocampal spatiotemporal
projection:
1. Dynamic Sensory Differencing: Detects oscillating cells and moving entities across time steps.
2. Periodicity & Phase Estimation: Computes cyclic period T and phase offset for periodic hazards
   (blinking obstacles, oscillating lasers, patrolling entities).
3. Predictive Hazard Projection: Predicts future spatial hazards at (x, y, t mod T) to enable
   collision-free navigation and temporal waiting/hesitation impulses.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DynamicCellPhase:
    """Estimated temporal phase model for a dynamic/oscillating spatial cell."""

    r: int
    c: int
    period: int
    cycle_values: list[int]
    hazardous_values: set[int]
    confidence: float = 1.0


class SpatiotemporalHazardTracker:
    """Biologically-inspired tracker for moving hazards and periodic spatial oscillations."""

    def __init__(self, history_len: int = 24, max_period: int = 8) -> None:
        self.history_len = history_len
        self.max_period = max_period
        self.grid_history: deque[np.ndarray] = deque(maxlen=history_len)
        self.step_history: deque[int] = deque(maxlen=history_len)
        self.periodic_cells: dict[tuple[int, int], DynamicCellPhase] = {}
        self.environmental_period: int = 1
        self.known_lethal_features: set[int] = set()
        self.static_lethal_positions: set[tuple[int, int]] = set()

    def reset_episode(self) -> None:
        """Reset temporal observation history while preserving learned lethal feature identities and static lethal positions."""
        self.grid_history.clear()
        self.step_history.clear()
        self.periodic_cells.clear()
        self.environmental_period = 1

    def register_lethal_position(self, pos: tuple[int, int]) -> None:
        """Register a coordinate that resulted in avatar destruction/death upon entry."""
        self.static_lethal_positions.add((int(pos[0]), int(pos[1])))

    def clear_lethal_positions(self) -> None:
        """Clear recorded static lethal positions."""
        self.static_lethal_positions.clear()

    def register_lethal_feature(self, feature_id: int) -> None:
        """Register a feature value that resulted in avatar destruction/death upon contact."""
        self.known_lethal_features.add(int(feature_id))
        # Update existing periodic cells with new lethal feature
        for cell in self.periodic_cells.values():
            if int(feature_id) in cell.cycle_values:
                cell.hazardous_values.add(int(feature_id))

    def record_frame(
        self,
        step: int,
        grid: np.ndarray,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
        walkable_features: set[int] | None = None,
    ) -> None:
        if not isinstance(grid, np.ndarray):
            if isinstance(grid, (list, tuple)) and len(grid) > 0:
                grid = grid[-1]
            grid = np.asarray(grid, dtype=int)
        if grid.ndim == 3 and len(grid) > 0:
            grid = grid[-1]
        self.grid_history.append(grid.copy())
        self.step_history.append(step)

        if len(self.grid_history) < 4:
            return

        H, W = grid.shape
        # Identify dynamic cells that fluctuated across history
        grids_arr = np.stack(list(self.grid_history), axis=0)  # Shape: (N, H, W)
        min_vals = np.min(grids_arr, axis=0)
        max_vals = np.max(grids_arr, axis=0)
        fluctuating_mask = min_vals != max_vals

        fluctuating_indices = np.argwhere(fluctuating_mask)
        if len(fluctuating_indices) == 0:
            self.periodic_cells.clear()
            self.environmental_period = 1
            return

        discovered_periods: list[int] = []
        av_set = avatar_features or set()
        safe_features = {0, background_feature}
        if walkable_features:
            safe_features.update(walkable_features)
        if av_set:
            safe_features.update(av_set)

        for r, c in fluctuating_indices:
            r_idx, c_idx = int(r), int(c)
            series = [int(g[r_idx, c_idx]) for g in self.grid_history]

            best_period = self._estimate_period(series)
            if best_period is not None and best_period >= 2:
                # Cycle values over the period
                cycle = series[-best_period:]
                # Determine which values in this cycle are hazardous:
                # Known lethal features or periodic non-background environmental features (excluding avatar and safe/walkable features)
                haz_vals = {
                    v
                    for v in cycle
                    if v in self.known_lethal_features
                    or (v not in safe_features and not self.known_lethal_features)
                }
                if haz_vals:
                    self.periodic_cells[(r_idx, c_idx)] = DynamicCellPhase(
                        r=r_idx,
                        c=c_idx,
                        period=best_period,
                        cycle_values=cycle,
                        hazardous_values=haz_vals,
                    )
                    discovered_periods.append(best_period)
                else:
                    self.periodic_cells.pop((r_idx, c_idx), None)
            else:
                self.periodic_cells.pop((r_idx, c_idx), None)

        # Compute environmental LCM period
        if discovered_periods:
            self.environmental_period = self._compute_lcm(discovered_periods)
        else:
            self.environmental_period = 1

    def _estimate_period(self, series: list[int]) -> int | None:
        """Estimate repeating periodicity T using backward autocorrelation."""
        n = len(series)
        for T in range(2, min(self.max_period + 1, n // 2 + 1)):
            # Check if series[i] == series[i - T] for recent history
            matches = sum(1 for i in range(T, n) if series[i] == series[i - T])
            total = n - T
            if total > 0 and (matches / total) >= 0.85:
                return T
        return None

    @staticmethod
    def _compute_lcm(numbers: Sequence[int]) -> int:
        """Compute the least common multiple of a sequence of integers (capped at 24)."""
        lcm = 1
        for num in set(numbers):
            if num <= 0:
                continue
            lcm = (lcm * num) // math.gcd(lcm, num)
            if lcm > 24:
                return 24
        return max(1, lcm)

    def is_hazard_at(
        self,
        r: int,
        c: int,
        future_relative_step: int,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
        walkable_features: set[int] | None = None,
    ) -> bool:
        """Predict whether coordinate (r, c) will be lethal or impassable at t_current + future_relative_step."""
        if (r, c) in self.static_lethal_positions:
            return True
        if (r, c) not in self.periodic_cells:
            return False

        cell_phase = self.periodic_cells[(r, c)]
        T = cell_phase.period
        # Predicted index in cycle_values
        idx = (future_relative_step - 1) % T
        predicted_val = cell_phase.cycle_values[idx]
        if predicted_val == 0 or predicted_val == background_feature:
            return False
        if avatar_features and predicted_val in avatar_features:
            return False
        if walkable_features and predicted_val in walkable_features:
            return False
        return predicted_val in cell_phase.hazardous_values

    def get_hazard_schedule(
        self,
        horizon: int = 24,
        background_feature: int = 0,
        walkable_features: set[int] | None = None,
    ) -> dict[int, set[tuple[int, int]]]:
        """Generate a schedule mapping relative future step -> set of hazardous coordinates."""
        schedule: dict[int, set[tuple[int, int]]] = {}
        safe_features = {0, background_feature}
        if walkable_features:
            safe_features.update(walkable_features)
        for dt in range(horizon + 1):
            haz_set: set[tuple[int, int]] = set(self.static_lethal_positions)
            for (r, c), cell_phase in self.periodic_cells.items():
                T = cell_phase.period
                idx = (dt - 1) % T if dt > 0 else -1
                val = cell_phase.cycle_values[idx]
                if val not in safe_features and val in cell_phase.hazardous_values:
                    haz_set.add((r, c))
            if haz_set:
                schedule[dt] = haz_set
        return schedule


# ── W059: Morphological Deformation & Topological Invariant Tracker ──────────


from enum import StrEnum


class MorphologicalDeformationType(StrEnum):
    """Classification of continuous non-rigid entity shape changes."""

    RIGID_TRANSLATION = "RIGID_TRANSLATION"
    EXTRUSION = "EXTRUSION"  # Growth along a directional ray / line
    GRAVITY_SETTLING = "GRAVITY_SETTLING"  # Downward vertical compaction
    CONTOUR_EROSION = "CONTOUR_EROSION"  # Shrinkage preserving core
    TOPOLOGICAL_GROWTH = "TOPOLOGICAL_GROWTH"  # Expansion / flood fill
    UNKNOWN_DEFORMATION = "UNKNOWN_DEFORMATION"


@dataclass
class MorphologicalEntity:
    """Entity representation for non-rigid shape tracking."""

    entity_id: str
    feature_id: int
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]]
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]


@dataclass
class DeformationRecord:
    """Record of continuous entity identity preservation across morphing."""

    entity_id: str
    prior_cells_count: int
    current_cells_count: int
    iou: float
    euler_characteristic: int
    deformation_type: MorphologicalDeformationType


@dataclass
class EntityLineageNode:
    """Directed causal lineage node tracing entity provenance across lifecycle events (W014)."""

    entity_id: str
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    generation: int = 0
    transition_type: str = (
        "IDENTITY"  # IDENTITY, FISSION, FUSION, DEFORMATION, CREATION, DESTRUCTION
    )
    confidence: float = 1.0
    is_mass_conserved: bool = True
    ambiguity_score: float = 0.0
    component_correspondences: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FissionRecord:
    """Record of a single parent entity dividing into multiple child entities (W014)."""

    parent_entity_id: str
    child_entity_ids: list[str]
    parent_cell_count: int
    child_cell_counts: list[int]
    mass_conservation_ratio: float  # sum(child_cells) / parent_cells
    spatial_coverage_ratio: float  # (sum child cells & parent) / parent_cells
    is_mass_conserved: bool = True
    conservation_required: bool = False
    ambiguity_score: float = 0.0
    component_correspondences: dict[str, float] = field(default_factory=dict)


@dataclass
class FusionRecord:
    """Record of multiple parent entities coalescing into a single merged entity (W014)."""

    parent_entity_ids: list[str]
    merged_entity_id: str
    parent_cell_counts: list[int]
    merged_cell_count: int
    mass_conservation_ratio: float  # merged_cells / sum(parent_cells)
    spatial_coverage_ratio: float  # (parent_cells & merged_cells) / merged_cells
    is_mass_conserved: bool = True
    conservation_required: bool = False
    ambiguity_score: float = 0.0
    component_correspondences: dict[str, float] = field(default_factory=dict)


@dataclass
class LifecycleTrackingResult:
    """Complete result of entity lifecycle tracking across temporal frames (W014, W059)."""

    one_to_one_mappings: dict[str, str]  # prior_id -> current_id
    deformation_records: list[DeformationRecord]
    fission_records: list[FissionRecord]
    fusion_records: list[FusionRecord]
    unmatched_prior_ids: list[str]
    unmatched_current_ids: list[str]
    lineage_graph: dict[str, EntityLineageNode] = field(default_factory=dict)


class MorphologicalDeformationTracker:
    """Tracks entity identity across non-rigid spatial changes and topological fission/fusion (W014, W059)."""

    @staticmethod
    def compute_euler_characteristic(
        cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> int:
        """Compute 2D discrete Euler characteristic chi = V - E + F for a cell set.

        For a connected solid object with g holes, chi = 1 - g.
        """
        if not cells:
            return 0

        # Vertices: 4 corner points per cell
        vertices: set[tuple[float, float]] = set()
        # Edges: 4 border segments per cell
        edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()

        for r, c in cells:
            # 4 vertices: (r, c), (r+1, c), (r, c+1), (r+1, c+1)
            v_tl = (float(r), float(c))
            v_tr = (float(r), float(c + 1))
            v_bl = (float(r + 1), float(c))
            v_br = (float(r + 1), float(c + 1))

            vertices.update([v_tl, v_tr, v_bl, v_br])

            # 4 canonical undirected edges
            edges.add((min(v_tl, v_tr), max(v_tl, v_tr)))
            edges.add((min(v_tr, v_br), max(v_tr, v_br)))
            edges.add((min(v_bl, v_br), max(v_bl, v_br)))
            edges.add((min(v_tl, v_bl), max(v_tl, v_bl)))

        V = len(vertices)
        E = len(edges)
        F = len(cells)
        return V - E + F

    @staticmethod
    def compute_iou(
        cells_a: set[tuple[int, int]] | frozenset[tuple[int, int]],
        cells_b: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> float:
        """Compute Intersection over Union (IoU) between two sets of grid cells."""
        intersection = len(cells_a & cells_b)
        union = len(cells_a | cells_b)
        return float(intersection / union) if union > 0 else 0.0

    @classmethod
    def classify_deformation(
        cls,
        prior: set[tuple[int, int]] | frozenset[tuple[int, int]],
        current: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> MorphologicalDeformationType:
        """Classify the geometric deformation type between two sequential states."""
        if prior == current:
            return MorphologicalDeformationType.RIGID_TRANSLATION

        iou = cls.compute_iou(prior, current)
        len_p = len(prior)
        len_c = len(current)

        # Check vertical settling / gravity drop
        p_cols = {c for _, c in prior}
        c_cols = {c for _, c in current}
        if p_cols == c_cols and len_c == len_p:
            p_min_r = min(r for r, _ in prior)
            c_min_r = min(r for r, _ in current)
            if c_min_r > p_min_r:
                return MorphologicalDeformationType.GRAVITY_SETTLING

        # Extrusion: current strictly contains prior and grows along single axis
        if prior.issubset(current) and len_c > len_p:
            diff = current - prior
            diff_rows = {r for r, _ in diff}
            diff_cols = {c for _, c in diff}
            if len(diff_rows) == 1 or len(diff_cols) == 1:
                return MorphologicalDeformationType.EXTRUSION
            return MorphologicalDeformationType.TOPOLOGICAL_GROWTH

        # Erosion: current is subset of prior
        if current.issubset(prior) and len_c < len_p:
            return MorphologicalDeformationType.CONTOUR_EROSION

        if iou >= 0.2:
            return MorphologicalDeformationType.TOPOLOGICAL_GROWTH

        return MorphologicalDeformationType.UNKNOWN_DEFORMATION

    def match_entities(
        self,
        prior_entities: list[MorphologicalEntity],
        current_entities: list[MorphologicalEntity],
        iou_threshold: float = 0.15,
    ) -> tuple[dict[str, str], list[DeformationRecord]]:
        """Preserve entity UUIDs across non-rigid shape deformations.

        Returns:
            (mapping: prior_id -> current_id, deformation_records)
        """
        mapping: dict[str, str] = {}
        records: list[DeformationRecord] = []
        matched_curr_ids: set[str] = set()

        for pe in prior_entities:
            best_curr = None
            best_score = -1.0
            best_iou = 0.0

            for ce in current_entities:
                if ce.entity_id in matched_curr_ids:
                    continue
                if pe.feature_id != ce.feature_id:
                    continue

                iou = self.compute_iou(pe.cells, ce.cells)
                # Centroid distance penalty
                dr = pe.centroid[0] - ce.centroid[0]
                dc = pe.centroid[1] - ce.centroid[1]
                dist = math.hypot(dr, dc)
                score = iou - 0.05 * dist

                if iou >= iou_threshold and score > best_score:
                    best_score = score
                    best_curr = ce
                    best_iou = iou

            if best_curr is not None:
                mapping[pe.entity_id] = best_curr.entity_id
                matched_curr_ids.add(best_curr.entity_id)

                def_type = self.classify_deformation(pe.cells, best_curr.cells)
                chi = self.compute_euler_characteristic(best_curr.cells)
                records.append(
                    DeformationRecord(
                        entity_id=pe.entity_id,
                        prior_cells_count=len(pe.cells),
                        current_cells_count=len(best_curr.cells),
                        iou=best_iou,
                        euler_characteristic=chi,
                        deformation_type=def_type,
                    )
                )

        return mapping, records

    @classmethod
    def detect_fission(
        cls,
        prior_entity: MorphologicalEntity,
        candidate_children: list[MorphologicalEntity],
        min_coverage: float = 0.20,
        conservation_required: bool = False,
    ) -> FissionRecord | None:
        """Detect if a prior entity split into multiple distinct child entities (W014).

        Supports both mass-conserved physical fission and non-conserved morphological splitting
        (occlusion, laser slicing, cutting, non-rigid detachment).
        """
        if len(candidate_children) < 2:
            return None

        # Filter candidate children by matching feature_id
        valid_children = [c for c in candidate_children if c.feature_id == prior_entity.feature_id]
        if len(valid_children) < 2:
            return None

        parent_count = len(prior_entity.cells)
        if parent_count == 0:
            return None

        child_union = set().union(*(c.cells for c in valid_children))
        overlap = len(child_union & prior_entity.cells)
        spatial_cov = overlap / parent_count
        sum_child_cells = sum(len(c.cells) for c in valid_children)
        mass_ratio = sum_child_cells / parent_count
        is_conserved = 0.90 <= mass_ratio <= 1.10

        if conservation_required and not is_conserved:
            return None

        # Validate fission: children overlap parent or are in close spatial proximity
        if spatial_cov < min_coverage and not (is_conserved and spatial_cov >= 0.10):
            return None

        # Individual component correspondences (child overlap with parent)
        correspondences = {
            c.entity_id: round(len(c.cells & prior_entity.cells) / len(c.cells), 4)
            if len(c.cells) > 0
            else 0.0
            for c in valid_children
        }

        # Ambiguity score: normalized entropy over child cell distributions
        fractions = (
            [len(c.cells) / sum_child_cells for c in valid_children] if sum_child_cells > 0 else []
        )
        if len(fractions) >= 2:
            ambiguity = -sum(p * math.log2(p + 1e-12) for p in fractions) / math.log2(
                len(fractions)
            )
        else:
            ambiguity = 0.0

        return FissionRecord(
            parent_entity_id=prior_entity.entity_id,
            child_entity_ids=[c.entity_id for c in valid_children],
            parent_cell_count=parent_count,
            child_cell_counts=[len(c.cells) for c in valid_children],
            mass_conservation_ratio=round(mass_ratio, 4),
            spatial_coverage_ratio=round(spatial_cov, 4),
            is_mass_conserved=is_conserved,
            conservation_required=conservation_required,
            ambiguity_score=round(ambiguity, 4),
            component_correspondences=correspondences,
        )

    @classmethod
    def detect_fusion(
        cls,
        candidate_parents: list[MorphologicalEntity],
        merged_entity: MorphologicalEntity,
        min_coverage: float = 0.20,
        conservation_required: bool = False,
    ) -> FusionRecord | None:
        """Detect if multiple prior entities coalesced into a single merged entity (W014).

        Supports both mass-conserved physical fusion and non-conserved agglomeration.
        """
        if len(candidate_parents) < 2:
            return None

        valid_parents = [p for p in candidate_parents if p.feature_id == merged_entity.feature_id]
        if len(valid_parents) < 2:
            return None

        merged_count = len(merged_entity.cells)
        if merged_count == 0:
            return None

        parent_union = set().union(*(p.cells for p in valid_parents))
        overlap = len(parent_union & merged_entity.cells)
        spatial_cov = overlap / merged_count
        sum_parent_cells = sum(len(p.cells) for p in valid_parents)
        mass_ratio = merged_count / sum_parent_cells if sum_parent_cells > 0 else 0.0
        is_conserved = 0.90 <= mass_ratio <= 1.10

        if conservation_required and not is_conserved:
            return None

        if spatial_cov < min_coverage and not (is_conserved and spatial_cov >= 0.10):
            return None

        correspondences = {
            p.entity_id: round(len(p.cells & merged_entity.cells) / len(p.cells), 4)
            if len(p.cells) > 0
            else 0.0
            for p in valid_parents
        }

        fractions = (
            [len(p.cells) / sum_parent_cells for p in valid_parents] if sum_parent_cells > 0 else []
        )
        if len(fractions) >= 2:
            ambiguity = -sum(fr * math.log2(fr + 1e-12) for fr in fractions) / math.log2(
                len(fractions)
            )
        else:
            ambiguity = 0.0

        return FusionRecord(
            parent_entity_ids=[p.entity_id for p in valid_parents],
            merged_entity_id=merged_entity.entity_id,
            parent_cell_counts=[len(p.cells) for p in valid_parents],
            merged_cell_count=merged_count,
            mass_conservation_ratio=round(mass_ratio, 4),
            spatial_coverage_ratio=round(spatial_cov, 4),
            is_mass_conserved=is_conserved,
            conservation_required=conservation_required,
            ambiguity_score=round(ambiguity, 4),
            component_correspondences=correspondences,
        )

    def track_lifecycle(
        self,
        prior_entities: list[MorphologicalEntity],
        current_entities: list[MorphologicalEntity],
        iou_threshold: float = 0.15,
        high_conf_iou: float = 0.70,
        conservation_required: bool = False,
    ) -> LifecycleTrackingResult:
        """Unified lifecycle tracking: hierarchical resolution prioritizing high-confidence 1-to-1,
        followed by topological fission/fusion (W014), relaxed 1-to-1 deformations, and causal lineage DAG.
        """
        # Phase 1: High-confidence 1-to-1 matches (IoU >= high_conf_iou)
        one_to_one, def_records = self.match_entities(
            prior_entities, current_entities, iou_threshold=high_conf_iou
        )

        matched_prior = set(one_to_one.keys())
        matched_curr = set(one_to_one.values())

        unmatched_priors = [pe for pe in prior_entities if pe.entity_id not in matched_prior]
        unmatched_currs = [ce for ce in current_entities if ce.entity_id not in matched_curr]

        fission_records: list[FissionRecord] = []
        fusion_records: list[FusionRecord] = []

        # Phase 2: Detect Fissions (1 -> K) among candidate entities
        remaining_currs = list(unmatched_currs)
        for pe in list(unmatched_priors):
            if len(remaining_currs) < 2:
                break
            candidate_children = [
                ce
                for ce in remaining_currs
                if ce.feature_id == pe.feature_id and bool(ce.cells & pe.cells)
            ]
            if len(candidate_children) >= 2:
                fission = self.detect_fission(
                    pe, candidate_children, conservation_required=conservation_required
                )
                if fission:
                    fission_records.append(fission)
                    unmatched_priors.remove(pe)
                    for c_id in fission.child_entity_ids:
                        remaining_currs = [ce for ce in remaining_currs if ce.entity_id != c_id]

        # Phase 3: Detect Fusions (K -> 1) among candidate entities
        remaining_priors = list(unmatched_priors)
        for ce in list(remaining_currs):
            if len(remaining_priors) < 2:
                break
            candidate_parents = [
                pe
                for pe in remaining_priors
                if pe.feature_id == ce.feature_id and bool(pe.cells & ce.cells)
            ]
            if len(candidate_parents) >= 2:
                fusion = self.detect_fusion(
                    candidate_parents, ce, conservation_required=conservation_required
                )
                if fusion:
                    fusion_records.append(fusion)
                    remaining_currs.remove(ce)
                    for p_id in fusion.parent_entity_ids:
                        remaining_priors = [pe for pe in remaining_priors if pe.entity_id != p_id]

        # Phase 4: Relaxed 1-to-1 deformation matching for remaining entities
        if remaining_priors and remaining_currs:
            relaxed_map, relaxed_defs = self.match_entities(
                remaining_priors, remaining_currs, iou_threshold=iou_threshold
            )
            one_to_one.update(relaxed_map)
            def_records.extend(relaxed_defs)
            remaining_priors = [pe for pe in remaining_priors if pe.entity_id not in relaxed_map]
            remaining_currs = [
                ce for ce in remaining_currs if ce.entity_id not in relaxed_map.values()
            ]

        # Phase 5: Construct complete causal lineage DAG
        lineage: dict[str, EntityLineageNode] = {}
        for p_id, c_id in one_to_one.items():
            lineage[c_id] = EntityLineageNode(
                entity_id=c_id,
                parent_ids=[p_id],
                child_ids=[],
                transition_type="IDENTITY",
                confidence=1.0,
                is_mass_conserved=True,
                ambiguity_score=0.0,
            )

        for fission in fission_records:
            lineage[fission.parent_entity_id] = EntityLineageNode(
                entity_id=fission.parent_entity_id,
                child_ids=list(fission.child_entity_ids),
                transition_type="FISSION",
                confidence=round(1.0 - fission.ambiguity_score * 0.3, 4),
                is_mass_conserved=fission.is_mass_conserved,
                ambiguity_score=fission.ambiguity_score,
                component_correspondences=fission.component_correspondences,
            )
            for c_id in fission.child_entity_ids:
                lineage[c_id] = EntityLineageNode(
                    entity_id=c_id,
                    parent_ids=[fission.parent_entity_id],
                    transition_type="FISSION",
                    confidence=round(1.0 - fission.ambiguity_score * 0.3, 4),
                    is_mass_conserved=fission.is_mass_conserved,
                    ambiguity_score=fission.ambiguity_score,
                    component_correspondences=fission.component_correspondences,
                )

        for fusion in fusion_records:
            lineage[fusion.merged_entity_id] = EntityLineageNode(
                entity_id=fusion.merged_entity_id,
                parent_ids=list(fusion.parent_entity_ids),
                transition_type="FUSION",
                confidence=round(1.0 - fusion.ambiguity_score * 0.3, 4),
                is_mass_conserved=fusion.is_mass_conserved,
                ambiguity_score=fusion.ambiguity_score,
                component_correspondences=fusion.component_correspondences,
            )
            for p_id in fusion.parent_entity_ids:
                lineage[p_id] = EntityLineageNode(
                    entity_id=p_id,
                    child_ids=[fusion.merged_entity_id],
                    transition_type="FUSION",
                    confidence=round(1.0 - fusion.ambiguity_score * 0.3, 4),
                    is_mass_conserved=fusion.is_mass_conserved,
                    ambiguity_score=fusion.ambiguity_score,
                    component_correspondences=fusion.component_correspondences,
                )

        for pe in remaining_priors:
            lineage[pe.entity_id] = EntityLineageNode(
                entity_id=pe.entity_id,
                transition_type="DESTRUCTION",
                confidence=1.0,
            )

        for ce in remaining_currs:
            lineage[ce.entity_id] = EntityLineageNode(
                entity_id=ce.entity_id,
                transition_type="CREATION",
                confidence=1.0,
            )

        return LifecycleTrackingResult(
            one_to_one_mappings=one_to_one,
            deformation_records=def_records,
            fission_records=fission_records,
            fusion_records=fusion_records,
            unmatched_prior_ids=[pe.entity_id for pe in remaining_priors],
            unmatched_current_ids=[ce.entity_id for ce in remaining_currs],
            lineage_graph=lineage,
        )
