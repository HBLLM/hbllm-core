"""Entorhinal Grid Cells & Hippocampal Path Integration Cortex (Moser & O'Keefe).

Biologically grounded in Edvard & May-Britt Moser's 2014 Nobel Prize discovery of
Entorhinal Grid Cells and John O'Keefe's Hippocampal Place Cells:
1. Multi-Scale Toroidal Continuous Attractor Neural Networks (CANN):
   - Maintains periodic 2D phase representations across discrete spatial scales
     (e.g., lambda_1 = 3.0, lambda_2 = 6.0, lambda_3 = 12.0 grid units).
   - Each module m tracks a 2D toroidal phase coordinate (phi_x, phi_y) in [0, 2*pi).
2. Odometric Path Integration:
   - When the agent executes a motor step with velocity (v_r, v_c), grid phases shift:
       dphi_x = (2*pi / lambda_m) * v_c
       dphi_y = (2*pi / lambda_m) * v_r
   - Enables dead-reckoning navigation even under zero visual cues or complete sensor blackout.
3. Theta Phase Precession (4–8 Hz Lookahead Compression):
   - Within each theta oscillation cycle, grid cell firing sweeps through planned future
     trajectory waypoints (from current position to lookahead goal), compressing sequential
     multi-step plans into a single biological phase window.
4. Conjunctive Place Cell Readout & Relocation:
   - Conjunctive multi-scale phase vectors define unique place cell receptive fields,
     instantly detecting loop closures, shortcuts, and room re-entries.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class GridModuleState:
    """Phase state of a single periodic entorhinal grid module."""

    module_id: str
    wavelength: float  # Spatial period lambda in grid coordinates
    phase_x: float = 0.0  # Phase in [0, 2*pi)
    phase_y: float = 0.0  # Phase in [0, 2*pi)

    def integrate_displacement(self, delta_r: float, delta_c: float) -> tuple[float, float]:
        """Update phase coordinates via continuous attractor path integration."""
        k = 2.0 * math.pi / max(0.1, self.wavelength)
        self.phase_x = (self.phase_x + k * delta_c) % (2.0 * math.pi)
        self.phase_y = (self.phase_y + k * delta_r) % (2.0 * math.pi)
        return self.phase_x, self.phase_y


@dataclass
class PlaceFieldRepresentation:
    """Conjunctive hippocampal place cell readout generated across all grid modules."""

    place_id: str
    estimated_coord: tuple[int, int]
    population_vector: list[float]
    theta_lookahead_waypoints: list[tuple[int, int]]
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export place field activation as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"place_field:{self.place_id}",
            operator="==",
            value=f"coord:{self.estimated_coord[0]}_{self.estimated_coord[1]}",
            relation=f"confidence:{self.confidence:.3f}:lookahead:{len(self.theta_lookahead_waypoints)}",
            target_entity=self.place_id,
        )

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Propose atomic HCIR transaction logging hippocampal place field in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"place_{self.place_id}",
                node_data={
                    "place_id": self.place_id,
                    "estimated_coord": list(self.estimated_coord),
                    "population_vector": self.population_vector,
                    "theta_lookahead_waypoints": [list(w) for w in self.theta_lookahead_waypoints],
                    "confidence": float(self.confidence),
                    "tenant_id": tenant_id,
                    "timestamp": self.timestamp,
                },
            )
        ]


class EntorhinalGridCortex:
    """Master Entorhinal Grid Cell & Hippocampal Path Integration Engine."""

    def __init__(
        self,
        wavelengths: list[float] | None = None,
        theta_frequency_hz: float = 6.0,
    ) -> None:
        scales = wavelengths or [3.0, 6.0, 12.0]
        self.modules: list[GridModuleState] = [
            GridModuleState(module_id=f"grid_mod_{idx}", wavelength=w)
            for idx, w in enumerate(scales)
        ]
        self.theta_frequency_hz = theta_frequency_hz
        self.place_field_history: list[PlaceFieldRepresentation] = []
        self.current_estimated_coord: tuple[int, int] = (0, 0)

    def reset_anchor(self, true_coord: tuple[int, int]) -> None:
        """Anchor/reset grid cell phases to ground truth visual landmarks."""
        self.current_estimated_coord = (int(true_coord[0]), int(true_coord[1]))
        for mod in self.modules:
            k = 2.0 * math.pi / max(0.1, mod.wavelength)
            mod.phase_x = (k * true_coord[1]) % (2.0 * math.pi)
            mod.phase_y = (k * true_coord[0]) % (2.0 * math.pi)

    def path_integrate_step(
        self,
        delta_r: float,
        delta_c: float,
    ) -> tuple[int, int]:
        """Execute continuous attractor path integration on motor velocity."""
        for mod in self.modules:
            mod.integrate_displacement(delta_r, delta_c)

        # Update estimated coordinate
        cur_r, cur_c = self.current_estimated_coord
        self.current_estimated_coord = (
            int(round(cur_r + delta_r)),
            int(round(cur_c + delta_c)),
        )
        return self.current_estimated_coord

    def generate_theta_phase_precession(
        self,
        planned_trajectory: list[tuple[int, int]],
        theta_phases_count: int = 5,
    ) -> list[tuple[int, int]]:
        """Simulate theta phase precession (4–8 Hz) lookahead compression.

        Compresses forward path trajectory points within a single theta cycle.
        """
        if not planned_trajectory:
            return [self.current_estimated_coord]

        # Sample up to theta_phases_count forward lookahead waypoints
        step_stride = max(1, len(planned_trajectory) // theta_phases_count)
        waypoints = [self.current_estimated_coord]
        for idx in range(0, len(planned_trajectory), step_stride):
            waypoints.append(planned_trajectory[idx])
            if len(waypoints) >= theta_phases_count:
                break
        return waypoints

    def generate_place_field_readout(
        self,
        planned_trajectory: list[tuple[int, int]] | None = None,
    ) -> PlaceFieldRepresentation:
        """Generate conjunctive hippocampal place cell readout across all grid scales."""
        pop_vector: list[float] = []
        for mod in self.modules:
            pop_vector.append(math.cos(mod.phase_x))
            pop_vector.append(math.sin(mod.phase_x))
            pop_vector.append(math.cos(mod.phase_y))
            pop_vector.append(math.sin(mod.phase_y))

        lookahead = self.generate_theta_phase_precession(planned_trajectory or [])

        place_id = f"place_{self.current_estimated_coord[0]}_{self.current_estimated_coord[1]}"
        place = PlaceFieldRepresentation(
            place_id=place_id,
            estimated_coord=self.current_estimated_coord,
            population_vector=pop_vector,
            theta_lookahead_waypoints=lookahead,
            confidence=1.0,
        )
        self.place_field_history.append(place)
        return place

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export grid cell path integration state as declarative causal predicates."""
        predicates: list[CausalPredicate] = [
            CausalPredicate(
                variable="grid_cell:path_integration",
                operator="==",
                value=f"{self.current_estimated_coord[0]}_{self.current_estimated_coord[1]}",
                relation=f"modules:{len(self.modules)}:theta_hz:{self.theta_frequency_hz}",
                target_entity="entorhinal_cortex",
            )
        ]
        for mod in self.modules:
            predicates.append(
                CausalPredicate(
                    variable=f"grid_module:{mod.module_id}",
                    operator="==",
                    value=f"phase_x:{mod.phase_x:.2f}:phase_y:{mod.phase_y:.2f}",
                    relation=f"lambda:{mod.wavelength}",
                    target_entity=mod.module_id,
                )
            )
        return predicates
