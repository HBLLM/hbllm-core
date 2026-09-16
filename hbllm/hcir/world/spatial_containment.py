"""Spatial Containment & Transport Engine for HCIR World Kernel.

Implements domain-agnostic spatial relation detection (INSIDE, NEAR, ON),
synchronous containment transport schema induction, and object permanence tracking.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionEvent,
    BeliefTransitionType,
)

logger = logging.getLogger(__name__)


@dataclass
class SpatialRelationFact:
    """Structured spatial fact induced from continuous geometric perception."""

    relation: Any = "ON"
    subject_id: str = ""
    object_id: str = ""
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)


class BaseSpatialContainmentEngine:
    """Domain-agnostic spatial relations and containment transport engine."""

    def __init__(self) -> None:
        self.discovered_spatial_schemas: list[dict[str, Any]] = []
        self.belief_history: list[BeliefTransitionEvent] = []
        self.interventions_count: int = 0

    @staticmethod
    def detect_spatial_relations(
        percept_items: Sequence[dict[str, Any]],
        near_threshold: float = 0.5,
        inside_relation: Any = "INSIDE",
        near_relation: Any = "NEAR",
    ) -> list[SpatialRelationFact]:
        """Induce relational facts directly from geometric and perceptual observations."""
        facts: list[SpatialRelationFact] = []
        percept_map = {p["percept_id"]: p for p in percept_items if "percept_id" in p}

        # Check containment relations
        for p in percept_items:
            cid = p.get("contained_in")
            if cid and cid in percept_map:
                facts.append(
                    SpatialRelationFact(
                        relation=inside_relation,
                        subject_id=p["percept_id"],
                        object_id=cid,
                        confidence=1.0,
                        evidence={"source": "direct_percept"},
                    )
                )

        # Check proximity / NEAR relations between all pairs
        ids = list(percept_map.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                p1, p2 = percept_map[id1], percept_map[id2]
                coords1 = p1.get("spatial_coordinates")
                coords2 = p2.get("spatial_coordinates")
                if coords1 is not None and coords2 is not None:
                    dx = coords1[0] - coords2[0]
                    dy = coords1[1] - coords2[1]
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist <= near_threshold:
                        confidence = (
                            max(0.0, 1.0 - dist / near_threshold) if near_threshold > 0 else 1.0
                        )
                        facts.append(
                            SpatialRelationFact(
                                relation=near_relation,
                                subject_id=id1,
                                object_id=id2,
                                confidence=confidence,
                                evidence={"distance": dist},
                            )
                        )

        return facts

    @staticmethod
    def evaluate_containment_transport_invariance(
        container_moved: bool,
        container_disp: float,
        inside_disp: float,
        outside_disp: float,
        tolerance: float = 0.05,
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate synchronous containment transport invariance."""
        transport_confirmed = (
            container_moved
            and abs(inside_disp - container_disp) < tolerance
            and outside_disp < tolerance
        )
        schema = {
            "schema_id": "schema_containment_transport",
            "relation": "INSIDE",
            "action": "PUSH",
            "invariant": "SYNCHRONOUS_TRANSPORT",
            "confirmed": transport_confirmed,
            "container_displacement": container_disp,
            "contained_displacement": inside_disp,
            "outside_displacement": outside_disp,
        }
        return transport_confirmed, schema

    @staticmethod
    def evaluate_object_permanence(
        predicted_pos: tuple[float, float],
        actual_pos: tuple[float, float],
        tolerance: float = 0.05,
    ) -> dict[str, Any]:
        """Compute Euclidean distance error between predicted and actual position."""
        dx = predicted_pos[0] - actual_pos[0]
        dy = predicted_pos[1] - actual_pos[1]
        prediction_error = (dx**2 + dy**2) ** 0.5
        return {
            "predicted_position": predicted_pos,
            "actual_position": actual_pos,
            "prediction_error": prediction_error,
            "permanence_preserved": prediction_error < tolerance,
        }

    def record_spatial_schema(
        self,
        schema: dict[str, Any],
        container_id: str,
        target_store: list[dict[str, Any]] | None = None,
        step_index: int | None = None,
    ) -> BeliefTransitionEvent:
        """Register confirmed spatial schema and log belief transition event."""
        self.discovered_spatial_schemas.append(schema)
        if target_store is not None:
            target_store.append(schema)

        step = self.interventions_count if step_index is None else step_index
        event = BeliefTransitionEvent(
            event_type=BeliefTransitionType.SPATIAL_SCHEMA_INDUCED,
            step_index=step,
            hypothesis_id="schema_containment",
            variable="spatial_containment",
            condition=f"INSIDE(x, {container_id}) ∧ MOVE({container_id}) => MOVE(x)",
            prior_confidence=0.5,
            posterior_confidence=1.0,
            is_falsified=False,
            evidence={"schema": schema},
        )
        self.belief_history.append(event)
        logger.info(
            "Spatial schema induced: INSIDE(x, %s) ∧ MOVE(%s) => MOVE(x)",
            container_id,
            container_id,
        )
        return event
