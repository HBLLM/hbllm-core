from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.contracts.references import GroundedRef
from hbllm.hcir.grounding.predictive.predictor import PredictedEntityState


@dataclass(frozen=True)
class GroundingSurpriseAlert:
    """Surprise signal emitted when sensory observations deviate significantly from predictions."""

    alert_id: str
    entity_ref: GroundedRef
    surprise_score: float  # [0.0, 1.0]
    position_error: float
    property_mismatches: dict[str, tuple[Any, Any]]  # prop -> (predicted, observed)
    applied_schema_id: str | None = None
    timestamp: float = field(default_factory=time.time)


class GroundingSurpriseAssimilator:
    """Assimilates prediction errors between forward mental simulations and reality.

    Implements active inference surprise computation:
    Computes spatial and discrete property errors, and raises surprise alerts when
    discrepancies exceed variance bounds.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.35,
        spatial_sigma: float = 1.0,
    ) -> None:
        self._surprise_threshold = surprise_threshold
        self._spatial_sigma = spatial_sigma

    def evaluate_observation(
        self,
        predicted_states: dict[str, PredictedEntityState],
        observed_entities: dict[str, dict[str, Any]],
        timestamp: float = 0.0,
    ) -> list[GroundingSurpriseAlert]:
        """Compares predicted states against actual observed entity states."""
        now = timestamp if timestamp > 0.0 else time.time()
        alerts: list[GroundingSurpriseAlert] = []

        for ent_id, pred in predicted_states.items():
            obs = observed_entities.get(ent_id)
            if obs is None:
                # Entity unexpectedly vanished
                alert = GroundingSurpriseAlert(
                    alert_id=f"alert_vanished_{uuid.uuid4().hex[:8]}",
                    entity_ref=pred.entity_ref,
                    surprise_score=0.9,
                    position_error=float("inf"),
                    property_mismatches={"presence": ("visible", "absent")},
                    applied_schema_id=pred.applied_schema_id,
                    timestamp=now,
                )
                alerts.append(alert)
                continue

            # 1. Spatial Euclidean distance error
            obs_pos = obs.get("position", (0.0, 0.0))
            dx = obs_pos[0] - pred.predicted_position[0]
            dy = obs_pos[1] - pred.predicted_position[1]
            dz = (
                obs_pos[2] - pred.predicted_position[2]
                if len(obs_pos) > 2 and len(pred.predicted_position) > 2
                else 0.0
            )
            pos_err = math.sqrt(dx * dx + dy * dy + dz * dz)

            # 2. Property mismatches
            mismatches: dict[str, tuple[Any, Any]] = {}
            for pk, p_pred in pred.predicted_properties.items():
                p_obs = obs.get(pk)
                if p_obs != p_pred:
                    mismatches[pk] = (p_pred, p_obs)

            # 3. Normalized surprise score
            norm_pos = (pos_err / self._spatial_sigma) ** 2
            norm_props = len(mismatches) * 2.0
            total_err = norm_pos + norm_props

            # S = 1 - exp(-total_err / 2)
            surprise = 1.0 - math.exp(-total_err / 2.0)

            if surprise >= self._surprise_threshold:
                alert = GroundingSurpriseAlert(
                    alert_id=f"alert_surprise_{uuid.uuid4().hex[:8]}",
                    entity_ref=pred.entity_ref,
                    surprise_score=surprise,
                    position_error=pos_err,
                    property_mismatches=mismatches,
                    applied_schema_id=pred.applied_schema_id,
                    timestamp=now,
                )
                alerts.append(alert)

        return alerts
