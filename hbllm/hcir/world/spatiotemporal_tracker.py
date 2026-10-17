"""Spatiotemporal Hazard Tracking & Phase Precession System.

Biologically modeled on mammalian entorhinal grid cell phase precession and
hippocampal spatiotemporal projection:
1. Dynamic Sensory Differencing: Detects oscillating cells and moving entities across time steps.
2. Periodicity & Phase Estimation: Computes cyclic period T and phase offset for periodic hazards
   (blinking obstacles, oscillating lasers, patrolling entities) via autocorrelation.
3. Predictive Hazard Projection: Predicts future spatial hazards at (x, y, t mod T) to enable
   collision-free navigation and temporal waiting/hesitation impulses.
4. Morphological Deformation & Euler Invariants (W059): Classifies continuous shape transformations
   using planar Euler characteristic (chi = V - E + F = 1 - g) and IoU tracking.
5. Causal Lineage DAG & Fission/Fusion Resolution (W014): Tracks entity provenance across
   creation, destruction, 1-to-K fission, K-to-1 fusion, and relaxed deformation.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

# ── Domain 1: Entorhinal Phase Precession & Dynamic Hazard Tracking ───────────


@dataclass
class DynamicCellPhase:
    """Estimated temporal phase model for a dynamic/oscillating spatial cell."""

    r: int
    c: int
    period: int
    cycle_values: list[int]
    hazardous_values: set[int]
    confidence: float = 1.0

    def is_hazard_at_step(
        self,
        future_relative_step: int,
        safe_features: set[int] | None = None,
    ) -> bool:
        """Predict whether cell is hazardous at t_current + future_relative_step."""
        if self.period <= 0 or not self.cycle_values:
            return False
        idx = (future_relative_step - 1) % self.period if future_relative_step > 0 else -1
        val = self.cycle_values[idx]
        if safe_features and val in safe_features:
            return False
        return val in self.hazardous_values

    def to_causal_predicate(self) -> CausalPredicate:
        """Export dynamic cell phase model as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"cell_phase:{self.r}_{self.c}",
            operator="==",
            value=self.period,
            relation=f"hazards:{sorted(self.hazardous_values)}",
            target_entity=f"({self.r},{self.c})",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording dynamic cell phase."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"phase_cell_{self.r}_{self.c}",
                node_data={
                    "r": self.r,
                    "c": self.c,
                    "period": self.period,
                    "cycle_values": self.cycle_values,
                    "hazardous_values": sorted(self.hazardous_values),
                    "confidence": self.confidence,
                },
            )
        ]


class PhasePrecessionEstimator:
    """Pure computational faculty for periodicity estimation and phase decomposition."""

    @staticmethod
    def estimate_period(
        series: list[int],
        max_period: int = 8,
        min_match_ratio: float = 0.85,
    ) -> tuple[int | None, float]:
        """Estimate repeating periodicity T using backward autocorrelation.

        Returns:
            (best_period, confidence) or (None, 0.0) if no periodicity detected.
        """
        n = len(series)
        best_t: int | None = None
        best_confidence: float = 0.0

        for t in range(2, min(max_period + 1, n // 2 + 1)):
            matches = sum(1 for i in range(t, n) if series[i] == series[i - t])
            total = n - t
            if total > 0:
                ratio = matches / total
                if ratio >= min_match_ratio and ratio > best_confidence:
                    best_t = t
                    best_confidence = ratio

        if best_t is not None:
            return best_t, best_confidence
        return None, 0.0

    @staticmethod
    def compute_lcm(numbers: Sequence[int], cap: int = 24) -> int:
        """Compute least common multiple of a sequence of integers (capped to prevent explosion)."""
        lcm = 1
        for num in set(numbers):
            if num <= 0:
                continue
            lcm = (lcm * num) // math.gcd(lcm, num)
            if lcm > cap:
                return cap
        return max(1, lcm)

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export phase precession estimator capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="phase_precession_estimator",
            operator="==",
            value="ACTIVE",
            relation="entorhinal_phase_precession",
            target_entity="spatiotemporal_tracker",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering phase precession estimator."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="phase_precession_estimator",
                node_data={
                    "faculty": "entorhinal_phase_precession",
                    "status": "ACTIVE",
                },
            )
        ]


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
        """Record an environmental observation frame and update periodic hazard models."""
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

        grids_arr = np.stack(list(self.grid_history), axis=0)
        min_vals = np.min(grids_arr, axis=0)
        max_vals = np.max(grids_arr, axis=0)
        fluctuating_mask = min_vals != max_vals

        fluctuating_indices = np.argwhere(fluctuating_mask)
        if len(fluctuating_indices) == 0:
            self.periodic_cells.clear()
            self.environmental_period = 1
            return

        discovered_periods: list[int] = []
        safe_features = {0, background_feature}
        if walkable_features:
            safe_features.update(walkable_features)
        if avatar_features:
            safe_features.update(avatar_features)

        for r, c in fluctuating_indices:
            r_idx, c_idx = int(r), int(c)
            series = [int(g[r_idx, c_idx]) for g in self.grid_history]

            best_period, confidence = PhasePrecessionEstimator.estimate_period(
                series, max_period=self.max_period
            )
            if best_period is not None and best_period >= 2:
                cycle = series[-best_period:]
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
                        confidence=confidence,
                    )
                    discovered_periods.append(best_period)
                else:
                    self.periodic_cells.pop((r_idx, c_idx), None)
            else:
                self.periodic_cells.pop((r_idx, c_idx), None)

        if discovered_periods:
            self.environmental_period = PhasePrecessionEstimator.compute_lcm(discovered_periods)
        else:
            self.environmental_period = 1

    def _estimate_period(self, series: list[int]) -> int | None:
        """Estimate repeating periodicity T using backward autocorrelation (legacy hook)."""
        period, _ = PhasePrecessionEstimator.estimate_period(series, max_period=self.max_period)
        return period

    @staticmethod
    def _compute_lcm(numbers: Sequence[int]) -> int:
        """Compute the least common multiple of a sequence of integers (capped at 24)."""
        return PhasePrecessionEstimator.compute_lcm(numbers, cap=24)

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
        t = cell_phase.period
        idx = (future_relative_step - 1) % t if future_relative_step > 0 else -1
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
                t = cell_phase.period
                idx = (dt - 1) % t if dt > 0 else -1
                val = cell_phase.cycle_values[idx]
                if val not in safe_features and val in cell_phase.hazardous_values:
                    haz_set.add((r, c))
            if haz_set:
                schedule[dt] = haz_set
        return schedule

    def to_causal_predicate(self) -> CausalPredicate:
        """Export spatiotemporal hazard tracker state as a declarative causal predicate."""
        return CausalPredicate(
            variable="spatiotemporal_hazard_tracker",
            operator="==",
            value=self.environmental_period,
            relation=f"periodic_cells:{len(self.periodic_cells)}",
            target_entity=f"lethal:{sorted(self.known_lethal_features)}",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all periodic cell phase models as declarative causal predicates."""
        preds = [self.to_causal_predicate()]
        for cell in self.periodic_cells.values():
            preds.append(cell.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting hazard tracker state."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="hazard_tracker_summary",
                node_data={
                    "environmental_period": self.environmental_period,
                    "periodic_cells_count": len(self.periodic_cells),
                    "known_lethal_features": sorted(self.known_lethal_features),
                    "static_lethal_positions": [list(p) for p in self.static_lethal_positions],
                },
            )
        ]
        for cell in self.periodic_cells.values():
            ops.extend(cell.to_transaction_operations())
        return ops


# ── Domain 2: Morphological Deformation, Euler Invariants & Causal Lineage DAG ─


class MorphologicalDeformationType(StrEnum):
    """Classification of continuous non-rigid entity shape changes."""

    RIGID_TRANSLATION = "RIGID_TRANSLATION"
    EXTRUSION = "EXTRUSION"  # Growth along a directional ray / line
    GRAVITY_SETTLING = "GRAVITY_SETTLING"  # Downward vertical compaction
    CONTOUR_EROSION = "CONTOUR_EROSION"  # Shrinkage preserving core
    TOPOLOGICAL_GROWTH = "TOPOLOGICAL_GROWTH"  # Expansion / flood fill
    UNKNOWN_DEFORMATION = "UNKNOWN_DEFORMATION"


class LifecycleTransitionType(StrEnum):
    """Event types for entity lifecycle and causal provenance DAG (W014, W059)."""

    IDENTITY = "IDENTITY"
    FISSION = "FISSION"
    FUSION = "FUSION"
    DEFORMATION = "DEFORMATION"
    CREATION = "CREATION"
    DESTRUCTION = "DESTRUCTION"


@dataclass
class MorphologicalEntity:
    """Entity representation for non-rigid shape tracking."""

    entity_id: str
    feature_id: int
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]]
    centroid: tuple[float, float]
    bounding_box: tuple[int, int, int, int]

    @classmethod
    def from_spatial_entity(
        cls, entity: Any, fallback_id: str | None = None
    ) -> MorphologicalEntity:
        """Construct MorphologicalEntity from a SpatialEntity or duck-typed object."""
        ent_id = getattr(entity, "id", None) or fallback_id or "unnamed_entity"
        feat_id = getattr(entity, "feature_id", 0)

        props = getattr(entity, "properties", {})
        if isinstance(props, dict) and "cells" in props and props["cells"]:
            cells = set(props["cells"])
        else:
            g_pos = getattr(entity, "grid_pos", None)
            cells = {g_pos} if g_pos is not None else set()

        centroid = getattr(entity, "centroid", (0.0, 0.0))
        bbox = getattr(entity, "bounding_box", (0, 0, 0, 0))

        return cls(
            entity_id=str(ent_id),
            feature_id=int(feat_id),
            cells=cells,
            centroid=centroid,
            bounding_box=bbox,
        )


@dataclass
class DeformationRecord:
    """Record of continuous entity identity preservation across morphing."""

    entity_id: str
    prior_cells_count: int
    current_cells_count: int
    iou: float
    euler_characteristic: int
    deformation_type: MorphologicalDeformationType

    def to_causal_predicate(self) -> CausalPredicate:
        """Export morphological deformation record as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"deformation_record:{self.entity_id}",
            operator="==",
            value=self.deformation_type.value,
            relation=f"iou:{self.iou:.2f}",
            target_entity=f"euler:{self.euler_characteristic}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording morphological deformation."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"deformation_{self.entity_id}",
                node_data={
                    "entity_id": self.entity_id,
                    "prior_cells_count": self.prior_cells_count,
                    "current_cells_count": self.current_cells_count,
                    "iou": self.iou,
                    "euler_characteristic": self.euler_characteristic,
                    "deformation_type": self.deformation_type.value,
                },
            )
        ]


@dataclass
class EntityLineageNode:
    """Directed causal lineage node tracing entity provenance across lifecycle events (W014)."""

    entity_id: str
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    generation: int = 0
    transition_type: str = "IDENTITY"
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

    def get_parents(self, entity_id: str) -> list[str]:
        """Get parent entity IDs for a node in the lineage DAG."""
        node = self.lineage_graph.get(entity_id)
        return list(node.parent_ids) if node else []

    def get_children(self, entity_id: str) -> list[str]:
        """Get child entity IDs for a node in the lineage DAG."""
        node = self.lineage_graph.get(entity_id)
        return list(node.child_ids) if node else []

    def is_root(self, entity_id: str) -> bool:
        """Check if entity was newly created (no parent in prior state)."""
        node = self.lineage_graph.get(entity_id)
        return bool(node and not node.parent_ids)

    def is_leaf(self, entity_id: str) -> bool:
        """Check if entity was destroyed (no children in current state)."""
        node = self.lineage_graph.get(entity_id)
        return bool(node and not node.child_ids and node.transition_type == "DESTRUCTION")

    def to_causal_predicate(self) -> CausalPredicate:
        """Export lifecycle tracking summary as a declarative causal predicate."""
        return CausalPredicate(
            variable="lifecycle_tracking_result",
            operator="==",
            value=len(self.one_to_one_mappings),
            relation=f"deformations:{len(self.deformation_records)}",
            target_entity=f"lineage_nodes:{len(self.lineage_graph)}",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all deformation records as declarative causal predicates."""
        preds = [self.to_causal_predicate()]
        for rec in self.deformation_records:
            preds.append(rec.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting lifecycle tracking result."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="lifecycle_tracking_summary",
                node_data={
                    "one_to_one_mappings_count": len(self.one_to_one_mappings),
                    "deformation_records_count": len(self.deformation_records),
                    "fission_records_count": len(self.fission_records),
                    "fusion_records_count": len(self.fusion_records),
                    "lineage_graph_nodes_count": len(self.lineage_graph),
                },
            )
        ]
        for rec in self.deformation_records:
            ops.extend(rec.to_transaction_operations())
        return ops


class MorphologicalDeformationTracker:
    """Tracks entity identity across non-rigid spatial changes and topological fission/fusion (W014, W059)."""

    @staticmethod
    def compute_euler_characteristic(
        cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> int:
        """Compute 2D discrete Euler characteristic chi = V - E + F for a cell complex.

        For a planar connected solid object with g holes, chi = 1 - g.
        """
        if not cells:
            return 0

        vertices: set[tuple[float, float]] = set()
        edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()

        for r, c in cells:
            v_tl = (float(r), float(c))
            v_tr = (float(r), float(c + 1))
            v_bl = (float(r + 1), float(c))
            v_br = (float(r + 1), float(c + 1))

            vertices.update([v_tl, v_tr, v_bl, v_br])

            edges.add((min(v_tl, v_tr), max(v_tl, v_tr)))
            edges.add((min(v_tr, v_br), max(v_tr, v_br)))
            edges.add((min(v_bl, v_br), max(v_bl, v_br)))
            edges.add((min(v_tl, v_bl), max(v_tl, v_bl)))

        v = len(vertices)
        e = len(edges)
        f = len(cells)
        return v - e + f

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
            best_curr: MorphologicalEntity | None = None
            best_score = -1.0
            best_iou = 0.0

            for ce in current_entities:
                if ce.entity_id in matched_curr_ids:
                    continue
                if pe.feature_id != ce.feature_id:
                    continue

                iou = self.compute_iou(pe.cells, ce.cells)
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

        Supports both mass-conserved physical fission and non-conserved morphological splitting.
        """
        if len(candidate_children) < 2:
            return None

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

        if spatial_cov < min_coverage and not (is_conserved and spatial_cov >= 0.10):
            return None

        correspondences = {
            c.entity_id: round(len(c.cells & prior_entity.cells) / len(c.cells), 4)
            if len(c.cells) > 0
            else 0.0
            for c in valid_children
        }

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
                transition_type=LifecycleTransitionType.IDENTITY,
                confidence=1.0,
                is_mass_conserved=True,
                ambiguity_score=0.0,
            )

        for fission in fission_records:
            lineage[fission.parent_entity_id] = EntityLineageNode(
                entity_id=fission.parent_entity_id,
                child_ids=list(fission.child_entity_ids),
                transition_type=LifecycleTransitionType.FISSION,
                confidence=round(1.0 - fission.ambiguity_score * 0.3, 4),
                is_mass_conserved=fission.is_mass_conserved,
                ambiguity_score=fission.ambiguity_score,
                component_correspondences=fission.component_correspondences,
            )
            for c_id in fission.child_entity_ids:
                lineage[c_id] = EntityLineageNode(
                    entity_id=c_id,
                    parent_ids=[fission.parent_entity_id],
                    transition_type=LifecycleTransitionType.FISSION,
                    confidence=round(1.0 - fission.ambiguity_score * 0.3, 4),
                    is_mass_conserved=fission.is_mass_conserved,
                    ambiguity_score=fission.ambiguity_score,
                    component_correspondences=fission.component_correspondences,
                )

        for fusion in fusion_records:
            lineage[fusion.merged_entity_id] = EntityLineageNode(
                entity_id=fusion.merged_entity_id,
                parent_ids=list(fusion.parent_entity_ids),
                transition_type=LifecycleTransitionType.FUSION,
                confidence=round(1.0 - fusion.ambiguity_score * 0.3, 4),
                is_mass_conserved=fusion.is_mass_conserved,
                ambiguity_score=fusion.ambiguity_score,
                component_correspondences=fusion.component_correspondences,
            )
            for p_id in fusion.parent_entity_ids:
                lineage[p_id] = EntityLineageNode(
                    entity_id=p_id,
                    child_ids=[fusion.merged_entity_id],
                    transition_type=LifecycleTransitionType.FUSION,
                    confidence=round(1.0 - fusion.ambiguity_score * 0.3, 4),
                    is_mass_conserved=fusion.is_mass_conserved,
                    ambiguity_score=fusion.ambiguity_score,
                    component_correspondences=fusion.component_correspondences,
                )

        for pe in remaining_priors:
            lineage[pe.entity_id] = EntityLineageNode(
                entity_id=pe.entity_id,
                transition_type=LifecycleTransitionType.DESTRUCTION,
                confidence=1.0,
            )

        for ce in remaining_currs:
            lineage[ce.entity_id] = EntityLineageNode(
                entity_id=ce.entity_id,
                transition_type=LifecycleTransitionType.CREATION,
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

    @classmethod
    def to_causal_predicate(cls) -> CausalPredicate:
        """Export morphological deformation tracker capability as a declarative causal predicate."""
        return CausalPredicate(
            variable="morphological_deformation_tracker",
            operator="==",
            value="ACTIVE",
            relation="euler_invariants_lineage",
            target_entity="spatiotemporal_tracker",
        )

    @classmethod
    def to_transaction_operations(cls) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering morphological deformation tracker."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="deformation_tracker",
                node_data={
                    "faculty": "morphological_deformation_tracker",
                    "status": "ACTIVE",
                },
            )
        ]
