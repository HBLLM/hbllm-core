"""Sensory Observation Diffs and Oriented Threat Representations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from hbllm.hcir.spatial_planner import SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class FrameDiffType(StrEnum):
    """Categorization of visual changes resulting from an action."""

    NO_CHANGE = "NO_CHANGE"
    TRANSLATION = "TRANSLATION"
    IN_PLACE_MUTATION = "IN_PLACE_MUTATION"
    INDEX_CYCLE = "INDEX_CYCLE"
    CANVAS_TRANSFORMATION = "CANVAS_TRANSFORMATION"
    GLOBAL_TRANSITION = "GLOBAL_TRANSITION"


@dataclass
class EpistemicObservationDiff:
    """Detailed structural difference between consecutive observation frames."""

    changed_pixel_count: int = 0
    changed_mask: np.ndarray | None = None
    displaced_entities: list[tuple[SpatialEntity, tuple[int, int]]] = field(default_factory=list)
    mutated_pixels: list[tuple[int, int, int, int]] = field(
        default_factory=list
    )  # (r, c, old_val, new_val)
    disappeared_features: set[int] = field(default_factory=set)
    appeared_features: set[int] = field(default_factory=set)

    diff_type: FrameDiffType = FrameDiffType.NO_CHANGE
    bounding_box: tuple[int, int, int, int] | None = None  # (min_r, max_r, min_c, max_c)
    translation_delta: tuple[int, int] | None = None  # (dr, dc) if rigid translation detected
    moved_object_feature: int | None = None  # feature/color of the translated object
    moved_object_size: int = 0  # pixel count of the translated object

    def to_causal_predicate(self) -> CausalPredicate:
        """Export visual frame difference as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"observation_diff:{self.diff_type.value}",
            operator="==",
            value=self.changed_pixel_count,
            relation=f"trans:{self.translation_delta}" if self.translation_delta else "in_place",
            target_entity=(
                str(self.moved_object_feature) if self.moved_object_feature is not None else "none"
            ),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording observation difference."""
        node_id = f"diff_{self.diff_type.value}_{self.changed_pixel_count}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "id": node_id,
                    "node_type": "observation",
                    "category": "perception",
                    "diff_type": self.diff_type.value,
                    "changed_pixel_count": self.changed_pixel_count,
                    "translation_delta": self.translation_delta,
                    "moved_object_feature": self.moved_object_feature,
                    "moved_object_size": self.moved_object_size,
                    "disappeared_features": sorted(self.disappeared_features),
                    "appeared_features": sorted(self.appeared_features),
                },
            )
        ]


@dataclass(frozen=True)
class OrientedThreat:
    """An embodied creature or entity with directional facing and visual gaze cone."""

    pos: tuple[int, int]
    facing: tuple[int, int]
    gaze_pos: tuple[int, int]
    feature_id: int

    def to_causal_predicate(self) -> CausalPredicate:
        """Export oriented creature threat as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"oriented_threat:{self.feature_id}",
            operator="==",
            value=f"facing:{self.facing}",
            relation=f"gaze:{self.gaze_pos}",
            target_entity=f"({self.pos[0]},{self.pos[1]})",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording oriented threat."""
        node_id = f"threat_{self.feature_id}_{self.pos[0]}_{self.pos[1]}"
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node_id,
                node_data={
                    "id": node_id,
                    "node_type": "physical_entity",
                    "category": "perception",
                    "feature_id": self.feature_id,
                    "pos": self.pos,
                    "facing": self.facing,
                    "gaze_pos": self.gaze_pos,
                },
            )
        ]
