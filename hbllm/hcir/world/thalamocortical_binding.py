"""Thalamocortical Gamma-Band Phase Synchronization (40 Hz Feature Binding).

Biologically grounded in Wolf Singer & Andreas Engel's temporal binding hypothesis
(Singer, 1999; Engel et al., 2001) and Francis Crick's Thalamic Searchlight hypothesis:
1. Distributed Feature Ensembles:
   - Disparate visual and tactile features (coordinates, color hues, motion vectors,
     morphological bounding boxes, and functional affordances) are extracted across
     separate cortical streams (ventral "what" vs dorsal "where/how").
2. 40 Hz Gamma-Band Oscillatory Coupling (Kuramoto Model):
   - Features belonging to the same coherent entity synchronize their oscillatory
     phase angles (phi_i) via non-linear Kuramoto coupling:
       d(phi_i)/dt = omega_i + (K / N) * sum_j(sin(phi_j - phi_i))
   - Phase coherence order parameter R = |(1/N) * sum_j(exp(i * phi_j))| -> 1.0.
3. Thalamic Reticular Nucleus (TRN) Attentional Gating:
   - Amplifies resonant phase-locking of task-salient objects while desynchronizing
     background clutter and distractor noise.
4. Deformation-Invariant Object Tracking:
   - Bound feature bundles maintain continuous object identity through non-rigid
     morphological deformations, color changes, and split-screen transitions.
"""

from __future__ import annotations

import cmath
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class BoundFeatureBundle:
    """A cohesive entity representation bound by oscillatory gamma phase-locking."""

    bundle_id: str
    features: dict[str, Any]
    phase_coherence_r: float
    mean_phase_rad: float
    is_bound: bool
    source_components_count: int
    timestamp: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export bound feature bundle as a formal declarative causal predicate."""
        return CausalPredicate(
            variable=f"bound_entity:{self.bundle_id}",
            operator="==",
            value=f"coherence:{self.phase_coherence_r:.3f}",
            relation=f"phase:{self.mean_phase_rad:.2f}:features:{len(self.features)}",
            target_entity=self.bundle_id,
        )

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Propose atomic HCIR transaction registering bound entity in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"bound_bundle_{self.bundle_id}",
                node_data={
                    "bundle_id": self.bundle_id,
                    "features": self.features,
                    "phase_coherence_r": float(self.phase_coherence_r),
                    "mean_phase_rad": float(self.mean_phase_rad),
                    "is_bound": bool(self.is_bound),
                    "tenant_id": tenant_id,
                    "timestamp": self.timestamp,
                },
            )
        ]


class ThalamocorticalBindingEngine:
    """Oscillatory 40 Hz feature binding engine with TRN attentional gating."""

    def __init__(
        self,
        nominal_frequency_hz: float = 40.0,
        coupling_strength_k: float = 2.5,
        coherence_threshold: float = 0.65,
    ) -> None:
        self.nominal_frequency_hz = nominal_frequency_hz
        self.coupling_strength_k = coupling_strength_k
        self.coherence_threshold = coherence_threshold
        self.active_bundles: dict[str, BoundFeatureBundle] = {}

    def compute_kuramoto_order_parameter(self, phases: list[float]) -> tuple[float, float]:
        """Compute the Kuramoto phase coherence order parameter R and mean phase Psi."""
        if not phases:
            return 0.0, 0.0
        sum_phasor = sum(cmath.exp(1j * phi) for phi in phases)
        n = len(phases)
        r = abs(sum_phasor) / n
        psi = cmath.phase(sum_phasor)
        return float(r), float(psi)

    def simulate_phase_locking(
        self,
        features: list[dict[str, Any]],
        proximity_matrix: list[list[float]] | None = None,
        iterations: int = 15,
        dt: float = 0.05,
    ) -> list[float]:
        """Simulate coupled Kuramoto oscillator phase dynamics among feature units."""
        N = len(features)
        if N <= 1:
            return [0.0] * N

        # Initialize phases with slight dispersion around 0 and intrinsic frequency differences
        phases = [(idx * 1.0) % (2.0 * math.pi) for idx in range(N)]
        base_omega = 2.0 * math.pi * self.nominal_frequency_hz * 0.01
        omegas = [base_omega * (1.0 + 0.12 * (idx - (N / 2.0))) for idx in range(N)]

        for _ in range(iterations):
            new_phases = list(phases)
            for i in range(N):
                coupling_sum = 0.0
                for j in range(N):
                    if i != j:
                        weight = proximity_matrix[i][j] if proximity_matrix else 1.0
                        coupling_sum += weight * math.sin(phases[j] - phases[i])
                d_phi = omegas[i] + (self.coupling_strength_k / max(1, N - 1)) * coupling_sum
                new_phases[i] = (phases[i] + d_phi * dt) % (2.0 * math.pi)
            phases = new_phases

        return phases

    def bind_features_oscillatory(
        self,
        feature_entities: list[dict[str, Any]],
    ) -> list[BoundFeatureBundle]:
        """Synchronize multi-modal feature entities into cohesive bound object representations."""
        if not feature_entities:
            return []

        N = len(feature_entities)
        # Compute spatial / relational proximity weight matrix
        prox: list[list[float]] = [[1.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    pos_i = feature_entities[i].get("pos")
                    pos_j = feature_entities[j].get("pos")
                    if (
                        pos_i
                        and pos_j
                        and isinstance(pos_i, (list, tuple))
                        and isinstance(pos_j, (list, tuple))
                    ):
                        dist = math.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
                        prox[i][j] = math.exp(-dist / 3.0)
                    else:
                        feat_i = feature_entities[i].get("feature")
                        feat_j = feature_entities[j].get("feature")
                        prox[i][j] = 1.0 if feat_i == feat_j else 0.1

        phases = self.simulate_phase_locking(feature_entities, proximity_matrix=prox)

        # Cluster synchronized features into phase-locked bundles (delta_phi < pi/4 and proximity >= 0.2)
        clusters: list[list[int]] = []
        assigned = set()
        for i in range(N):
            if i not in assigned:
                cluster = [i]
                assigned.add(i)
                for j in range(i + 1, N):
                    if j not in assigned:
                        phase_diff = abs(cmath.phase(cmath.exp(1j * (phases[i] - phases[j]))))
                        if phase_diff < (math.pi / 4.0) and prox[i][j] >= 0.2:
                            cluster.append(j)
                            assigned.add(j)
                clusters.append(cluster)

        bundles: list[BoundFeatureBundle] = []
        for cluster in clusters:
            cluster_phases = [phases[idx] for idx in cluster]
            r, psi = self.compute_kuramoto_order_parameter(cluster_phases)
            is_bound = (r >= self.coherence_threshold) or (len(cluster) == 1)

            merged_features: dict[str, Any] = {}
            for idx in cluster:
                merged_features.update(feature_entities[idx])

            bundle_id = f"bundle_{uuid.uuid4().hex[:8]}"
            bundle = BoundFeatureBundle(
                bundle_id=bundle_id,
                features=merged_features,
                phase_coherence_r=r,
                mean_phase_rad=psi,
                is_bound=is_bound,
                source_components_count=len(cluster),
            )
            bundles.append(bundle)
            self.active_bundles[bundle_id] = bundle

        return bundles

    def apply_trn_attentional_gating(
        self,
        bundles: list[BoundFeatureBundle],
        target_features: dict[str, Any] | None = None,
    ) -> list[BoundFeatureBundle]:
        """Thalamic Reticular Nucleus (TRN) searchlight selectively amplifying resonant targets."""
        if not target_features:
            return bundles

        gated: list[BoundFeatureBundle] = []
        for b in bundles:
            match_count = sum(1 for k, v in target_features.items() if b.features.get(k) == v)
            if match_count > 0:
                # TRN amplification
                b.phase_coherence_r = min(1.0, b.phase_coherence_r + 0.25)
                b.is_bound = True
                gated.append(b)
            else:
                # Distractor suppression
                b.phase_coherence_r = max(0.1, b.phase_coherence_r - 0.2)
                gated.append(b)

        gated.sort(key=lambda b: b.phase_coherence_r, reverse=True)
        return gated
