"""Thalamocortical Visual Perception Faculty (Facade).

Biological Modeling:
1. Sensory Thalamus (thalamus.py)
2. Ventral Visual Stream & Gestalt Affordances (gestalt.py)
3. Dorsal Stream & Frame Difference Analysis (diff.py)
4. Composite Perception Processing (engine.py)
5. PerceptionStage (Cortical Stage 1 Pipeline Owner)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.hcir.spatial_planner import SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.perception import (
    EpistemicObservationDiff,
    FigureGroundSegregator,
    FrameDiffType,
    GestaltAffordanceDetector,
    OrientedThreat,
    PerceptionEngine,
    SensoryThalamus,
    SpatialRoleClassifier,
    _fast_unique_counts,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext

logger = logging.getLogger(__name__)


class PerceptionStage:
    """Cortical Stage 1: Sensory Perception & Coordinate Lifting.

    Transforms raw sensory arrays into SpatialEntity objects and coordinate frames.
    """

    name: str = "perception_stage"
    engine: Any | None = None

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        grid = ctx.curr_grid if ctx.curr_grid is not None else ctx.signals.get("_curr_grid")
        if grid is None:
            grid = ctx.prev_grid
        entities: list[SpatialEntity] = []
        if grid is not None:
            try:
                import numpy as _np

                g = _np.asarray(grid)
                entities = ctx.extract_entities(g, ctx.bg_feature)
            except Exception:
                entities = []
        avatar_candidate = ctx.avatar_pos
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="perception_observation",
                node_data={
                    "id": "perception_observation",
                    "node_type": "observation",
                    "category": "perception",
                    "entity_count": len(entities),
                    "step": ctx.step_counter,
                },
            )
        ]
        return StageResult(
            author=self.name,
            operations=ops,
            signals={
                "extracted_entities": entities,
                "entity_count": len(entities),
                "avatar_candidate": avatar_candidate,
                "step": ctx.step_counter,
            },
            proposed_action=None,
        )


__all__ = [
    "EpistemicObservationDiff",
    "FigureGroundSegregator",
    "FrameDiffType",
    "GestaltAffordanceDetector",
    "OrientedThreat",
    "PerceptionEngine",
    "PerceptionStage",
    "SensoryThalamus",
    "SpatialRoleClassifier",
    "_fast_unique_counts",
]
