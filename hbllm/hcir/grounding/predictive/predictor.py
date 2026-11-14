from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.causal.schema import CausalSchema
from hbllm.hcir.grounding.contracts.references import GroundedRef


@dataclass(frozen=True)
class PredictedEntityState:
    """The forward simulated mental prediction of an entity state under active inference."""

    entity_ref: GroundedRef
    predicted_position: tuple[float, ...]
    predicted_velocity: tuple[float, ...] = (0.0, 0.0)
    predicted_properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    applied_schema_id: str | None = None
    timestamp: float = field(default_factory=time.time)


class ForwardWorldPredictor:
    """Simulates expected world state transitions conditioned on actions or physics.

    Upholds Invariant 2 (Reasoning Invariant) & Invariant 3 (Zero LLM in Core):
    Pure deterministic physical simulation and causal schema application.
    """

    def __init__(self, schemas: list[CausalSchema] | None = None) -> None:
        self._schemas: dict[str, CausalSchema] = {}
        if schemas:
            for s in schemas:
                self.register_schema(s)

    def register_schema(self, schema: CausalSchema) -> None:
        self._schemas[schema.schema_id] = schema

    def predict_next_state(
        self,
        current_entities: dict[str, dict[str, Any]],
        action_type: str | None = None,
        target_entity_ref: GroundedRef | None = None,
        dt: float = 1.0,
        timestamp: float = 0.0,
    ) -> dict[str, PredictedEntityState]:
        """Rolls forward mental predictions for all entities across time delta dt."""
        now = timestamp if timestamp > 0.0 else time.time()
        predictions: dict[str, PredictedEntityState] = {}

        for ent_id, st in current_entities.items():
            ref = st.get("ref")
            if not isinstance(ref, GroundedRef):
                continue

            pos = tuple(st.get("position", (0.0, 0.0)))
            vel = tuple(st.get("velocity", (0.0, 0.0)))
            props = {
                k: v
                for k, v in st.items()
                if k not in ("ref", "position", "velocity", "speed", "bbox")
            }

            # Check if targeted by an active causal action
            schema_to_apply: CausalSchema | None = None
            if action_type and target_entity_ref and ref == target_entity_ref:
                schema_to_apply = self._find_matching_schema(action_type, props)

            if schema_to_apply and schema_to_apply.is_actionable_cause:
                # Apply causal schema effect
                effect = schema_to_apply.expected_effect
                eff_post = effect.get("post_state", {})
                props.update(eff_post)

                # If effect specifies motion or displacement
                if "speed" in eff_post:
                    spd = float(eff_post["speed"])
                    vel = (spd, 0.0)

                pred_pos = (
                    pos[0] + vel[0] * dt,
                    pos[1] + vel[1] * dt,
                )
                predictions[ent_id] = PredictedEntityState(
                    entity_ref=ref,
                    predicted_position=pred_pos,
                    predicted_velocity=vel,
                    predicted_properties=props,
                    confidence=schema_to_apply.p_effect_given_do_action,
                    applied_schema_id=schema_to_apply.schema_id,
                    timestamp=now + dt,
                )
            else:
                # Default kinematic extrapolation
                pred_pos = (
                    pos[0] + vel[0] * dt,
                    pos[1] + vel[1] * dt,
                )
                predictions[ent_id] = PredictedEntityState(
                    entity_ref=ref,
                    predicted_position=pred_pos,
                    predicted_velocity=vel,
                    predicted_properties=props,
                    confidence=0.85,
                    timestamp=now + dt,
                )

        return predictions

    def _find_matching_schema(
        self, action_type: str, entity_props: dict[str, Any]
    ) -> CausalSchema | None:
        schemas = self._schemas.values() if isinstance(self._schemas, dict) else self._schemas
        for s in schemas:
            if s.action_type == action_type and s.is_actionable_cause:
                # Match preconditions
                match = True
                for pk, pv in s.precondition.items():
                    if entity_props.get(pk) != pv:
                        match = False
                        break
                if match:
                    return s
        return None
