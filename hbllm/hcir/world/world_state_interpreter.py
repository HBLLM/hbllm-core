"""
World State Interpreter — Telemetry to Cognitive Belief Inference Pipeline.

Translates empirical sensor telemetry from DigitalTwinRegistry into structured cognitive belief hypotheses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.world.world_state_snapshot import WorldStateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class InterpretedBeliefHypothesis:
    """Cognitive interpretation of empirical telemetry."""

    subject: str
    predicate: str
    value: Any
    confidence: float
    evidence_source: str = "sensor_telemetry"


class WorldStateInterpreter:
    """Interprets raw empirical DigitalTwin telemetry into cognitive belief hypotheses."""

    def interpret_snapshot(self, snapshot: WorldStateSnapshot) -> list[InterpretedBeliefHypothesis]:
        """Analyze immutable WorldStateSnapshot and emit cognitive belief hypotheses."""
        hypotheses: list[InterpretedBeliefHypothesis] = []

        for var_name, var_value in snapshot.variables.items():
            if isinstance(var_value, (int, float)):
                if "temp" in var_name.lower() and var_value > 80.0:
                    hypotheses.append(
                        InterpretedBeliefHypothesis(
                            subject=var_name,
                            predicate="status",
                            value="overheating_warning",
                            confidence=0.88,
                            evidence_source=f"snapshot_{snapshot.state_hash}",
                        )
                    )
                elif "vibration" in var_name.lower() and var_value > 0.05:
                    hypotheses.append(
                        InterpretedBeliefHypothesis(
                            subject=var_name,
                            predicate="anomaly",
                            value="excessive_vibration",
                            confidence=0.82,
                            evidence_source=f"snapshot_{snapshot.state_hash}",
                        )
                    )
                else:
                    hypotheses.append(
                        InterpretedBeliefHypothesis(
                            subject=var_name,
                            predicate="status",
                            value="nominal",
                            confidence=0.95,
                            evidence_source=f"snapshot_{snapshot.state_hash}",
                        )
                    )
            else:
                hypotheses.append(
                    InterpretedBeliefHypothesis(
                        subject=var_name,
                        predicate="state",
                        value=str(var_value),
                        confidence=0.90,
                        evidence_source=f"snapshot_{snapshot.state_hash}",
                    )
                )

        logger.debug(
            "Interpreted snapshot %s into %d belief hypotheses",
            snapshot.state_hash,
            len(hypotheses),
        )
        return hypotheses
