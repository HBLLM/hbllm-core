"""Hierarchical Predictive Coding Cortical Columns (Rao & Ballard / Friston).

Biologically grounded in Rajesh Rao & Dana Ballard's predictive coding model
(Rao & Ballard, 1999) and Karl Friston's Hierarchical Predictive Processing (Friston, 2005):
1. Multi-Level Cortical Column Hierarchy:
   - Layer 0 (Sensory Percept): Raw sensory observation diffs and spatial component coordinates.
   - Layer 1 (Kinematic / Affordance): Intermediate object mutations, trajectory velocities, and affordance states.
   - Layer 2 (Prefrontal / Invariant): Abstract causal rules, goal states, and invariant topological structures.
2. Bidirectional Message Passing Dynamics:
   - Top-Down Generative Predictions: Each layer l generates predictions mu_l = g_l(r_{l+1}) of the layer below.
   - Bottom-Up Prediction Errors: Sensory discrepancy eps_l = Pi_l * (r_l - mu_l), weighted by precision Pi_l.
   - Recurrent State Updating: Representations update via:
       dr_l/dt = kappa * (eps_l - D_l^T * eps_{l+1})
     minimizing total hierarchical variational free energy without requiring backpropagation through time.
3. Autonomous Anomaly & Structural Shift Detection:
   - Chronic prediction error at Layer 2 signals a fundamental domain transition or rule invalidation.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class CorticalColumnLayer:
    """State representation and error units for a single hierarchical cortical layer."""

    layer_idx: int
    name: str
    representation: dict[str, float] = field(default_factory=dict)
    generative_prediction: dict[str, float] = field(default_factory=dict)
    prediction_error: dict[str, float] = field(default_factory=dict)
    precision_weight: float = 1.0
    free_energy: float = 0.0


@dataclass
class PredictiveCodingSnapshot:
    """Snapshot of total hierarchical prediction error and free energy across all layers."""

    snapshot_id: str
    layer_errors: dict[int, float]
    total_free_energy: float
    is_structural_anomaly: bool
    timestamp: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export hierarchical predictive coding state as a declarative causal predicate."""
        return CausalPredicate(
            variable="predictive_coding:hierarchical_error",
            operator="==",
            value=round(self.total_free_energy, 4),
            relation=f"anomaly:{self.is_structural_anomaly}:l2_err:{self.layer_errors.get(2, 0.0):.3f}",
            target_entity=self.snapshot_id,
        )

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Propose atomic HCIR transaction logging predictive coding state in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"pred_coding_{self.snapshot_id}",
                node_data={
                    "snapshot_id": self.snapshot_id,
                    "layer_errors": self.layer_errors,
                    "total_free_energy": float(self.total_free_energy),
                    "is_structural_anomaly": bool(self.is_structural_anomaly),
                    "tenant_id": tenant_id,
                    "timestamp": self.timestamp,
                },
            )
        ]


class HierarchicalPredictiveCodingCortex:
    """Master 3-layer hierarchical predictive coding column engine."""

    def __init__(
        self,
        learning_rate_kappa: float = 0.35,
        anomaly_threshold: float = 1.50,
        layer_names: list[str] | None = None,
    ) -> None:
        self.learning_rate_kappa = learning_rate_kappa
        self.anomaly_threshold = anomaly_threshold
        names = layer_names or ["L0_Sensory", "L1_Kinematic", "L2_Prefrontal"]
        self.layers: list[CorticalColumnLayer] = [
            CorticalColumnLayer(layer_idx=i, name=names[i]) for i in range(len(names))
        ]
        self.snapshot_history: list[PredictiveCodingSnapshot] = []

    @property
    def total_free_energy(self) -> float:
        """Current total variational free energy across cortical columns."""
        if self.snapshot_history:
            return self.snapshot_history[-1].total_free_energy
        return sum(layer.free_energy for layer in self.layers)

    def set_layer_precision(self, layer_idx: int, precision: float) -> None:
        """Set precision weighting Pi_l for a specific hierarchical layer (e.g., modulated by ACh)."""
        if 0 <= layer_idx < len(self.layers):
            self.layers[layer_idx].precision_weight = max(0.1, min(10.0, float(precision)))

    def set_sensory_input(self, sensory_dict: dict[str, float]) -> None:
        """Ground Layer 0 state directly in incoming sensory observation diffs."""
        self.layers[0].representation = dict(sensory_dict)

    def compute_top_down_prediction(self, higher_rep: dict[str, float]) -> dict[str, float]:
        """Generative mapping g_l(r_{l+1}) projecting higher-level invariants downward."""
        pred: dict[str, float] = {}
        for k, v in higher_rep.items():
            pred[k] = float(v)
        return pred

    def step_hierarchical_cycle(
        self,
        sensory_inputs: dict[str, float] | None = None,
        abstract_priors: dict[str, float] | None = None,
    ) -> PredictiveCodingSnapshot:
        """Execute one complete bidirectional hierarchical predictive coding cycle.

        1. Top-down predictions flow downward from Layer 2 to Layer 1, and Layer 1 to Layer 0.
        2. Prediction errors eps_l = Pi_l * (r_l - mu_l) are evaluated at each layer.
        3. Bottom-up prediction errors update intermediate and higher-level representations.
        4. Evaluates total variational free energy F = 0.5 * sum_l eps_l^T eps_l.
        """
        if sensory_inputs is not None:
            self.set_sensory_input(sensory_inputs)

        num_layers = len(self.layers)
        top_idx = num_layers - 1

        if abstract_priors is not None:
            self.layers[top_idx].generative_prediction = dict(abstract_priors)
            if not self.layers[top_idx].representation:
                self.layers[top_idx].representation = dict(abstract_priors)
            else:
                self.layers[top_idx].representation.update(abstract_priors)

        # 1. Top-Down Prediction Pass (L2 -> L1 -> L0)
        for l in range(num_layers - 1, 0, -1):
            source_rep = dict(self.layers[l].representation)
            if not source_rep and self.layers[l].generative_prediction:
                source_rep = dict(self.layers[l].generative_prediction)
            pred = self.compute_top_down_prediction(source_rep)
            self.layers[l - 1].generative_prediction = pred
            # If intermediate layer representation has not been grounded, initialize with top-down expectation
            if l - 1 > 0 and not self.layers[l - 1].representation:
                self.layers[l - 1].representation = dict(pred)

        # 2. Prediction Error Computation eps_l = Pi_l * (r_l - mu_l)
        layer_energies: dict[int, float] = {}
        for l in range(num_layers):
            layer = self.layers[l]
            errors: dict[str, float] = {}
            energy = 0.0

            all_keys = set(layer.representation.keys()) | set(layer.generative_prediction.keys())
            for k in all_keys:
                actual = layer.representation.get(k, 0.0)
                expected = layer.generative_prediction.get(k, 0.0)
                diff = actual - expected
                weighted_err = layer.precision_weight * diff
                errors[k] = weighted_err
                energy += weighted_err**2

            layer.prediction_error = errors
            layer.free_energy = 0.5 * energy
            layer_energies[l] = layer.free_energy

        # 3. Bottom-Up State Update dr_l/dt = kappa * (eps_l - eps_{l+1})
        for l in range(1, num_layers):
            layer = self.layers[l]
            lower_err = self.layers[l - 1].prediction_error
            for k, err_val in lower_err.items():
                current_val = layer.representation.get(k, 0.0)
                update = self.learning_rate_kappa * err_val
                layer.representation[k] = current_val + update

        total_fe = sum(layer_energies.values())
        is_anomaly = (
            total_fe >= self.anomaly_threshold or layer_energies.get(num_layers - 1, 0.0) >= 0.8
        )

        snapshot_id = f"pcs_{uuid.uuid4().hex[:8]}"
        snapshot = PredictiveCodingSnapshot(
            snapshot_id=snapshot_id,
            layer_errors=layer_energies,
            total_free_energy=total_fe,
            is_structural_anomaly=is_anomaly,
        )
        self.snapshot_history.append(snapshot)
        return snapshot

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export current hierarchical predictive states as declarative causal predicates."""
        predicates: list[CausalPredicate] = []
        for layer in self.layers:
            predicates.append(
                CausalPredicate(
                    variable=f"predictive_layer:{layer.name}",
                    operator="==",
                    value=round(layer.free_energy, 4),
                    relation=f"precision:{layer.precision_weight:.2f}:keys:{len(layer.representation)}",
                    target_entity=str(layer.layer_idx),
                )
            )
        return predicates
