"""Hierarchical Active Inference with Free Energy Precision Control.

Implements Karl Friston's Variational Free Energy Principle with dynamic precision
weighting across hierarchical sensory and prior representations (Bogacz 2017,
Parr & Friston 2020).

Precision (gamma / Pi) controls the gain on sensory prediction errors:
- High sensory precision (Pi_obs): In stable, predictable environments, sensory errors
  drive fast, one-shot belief updating and active epistemic probing.
- Low sensory precision / high volatility (sigma_v^2): In chaotic or noisy dynamics,
  sensory gain is attenuated to prevent thrashing, relying on robust prior attractors.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class PrecisionState:
    """Snapshot of hierarchical precision weights and epistemic gains."""

    sensory_precision: float  # Pi_obs in [0.05, 10.0]
    prior_precision: float  # Pi_prior in [0.05, 10.0]
    environmental_volatility: float  # sigma_v^2 in [0.0, 1.0]
    adaptive_learning_rate: float  # eta_t in [0.01, 1.0]
    epistemic_exploration_gain: float  # Gain on information seeking vs exploitation
    mean_prediction_error: float
    timestamp_step: int
    is_volatile_regime: bool = False
    learning_progress: float = 0.0  # -d(PE)/dt in [-1.0, 1.0] (Oudeyer / Schmidhuber)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export precision state as a formal causal predicate."""
        regime = "VOLATILE" if self.is_volatile_regime else "STABLE"
        return CausalPredicate(
            variable=f"precision_state:step_{self.timestamp_step}",
            operator="==",
            value=f"lr:{self.adaptive_learning_rate:.3f}:gain:{self.epistemic_exploration_gain:.2f}:lp:{self.learning_progress:.3f}:{regime}",
            relation=f"volatility:{self.environmental_volatility:.3f}",
            target_entity=f"step_{self.timestamp_step}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert precision state into transactional HCIR graph state mutations."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"precision_state_{self.timestamp_step}",
                node_data={
                    "step": self.timestamp_step,
                    "sensory_precision": self.sensory_precision,
                    "prior_precision": self.prior_precision,
                    "environmental_volatility": self.environmental_volatility,
                    "adaptive_learning_rate": self.adaptive_learning_rate,
                    "epistemic_exploration_gain": self.epistemic_exploration_gain,
                    "learning_progress": self.learning_progress,
                    "is_volatile_regime": self.is_volatile_regime,
                },
            )
        ]


class HierarchicalPrecisionController:
    """Manages hierarchical precision weighting and dynamic epistemic learning rates."""

    def __init__(
        self,
        base_learning_rate: float = 0.5,
        volatility_half_life: int = 16,
        prior_rigidity: float = 1.0,
        volatility_threshold: float = 0.35,
    ) -> None:
        self.base_learning_rate = base_learning_rate
        self.volatility_half_life = volatility_half_life
        self.prior_rigidity = prior_rigidity
        self.volatility_threshold = volatility_threshold

        # Error tracking buffers
        self.error_history: deque[float] = deque(maxlen=64)
        self.short_error_window: deque[float] = deque(maxlen=6)
        self.long_error_window: deque[float] = deque(maxlen=24)
        self.running_mean_error: float = 0.0
        self.running_variance_error: float = 0.0

        # State tracking
        self.current_step: int = 0
        self.current_state = PrecisionState(
            sensory_precision=1.0,
            prior_precision=prior_rigidity,
            environmental_volatility=0.0,
            adaptive_learning_rate=base_learning_rate,
            epistemic_exploration_gain=1.0,
            mean_prediction_error=0.0,
            timestamp_step=0,
            is_volatile_regime=False,
            learning_progress=0.0,
        )

    def update_with_prediction_error(
        self,
        step: int,
        error_magnitude: float,
        is_reset: bool = False,
    ) -> PrecisionState:
        """Update precision estimates given scalar prediction error in [0.0, 1.0]."""
        self.current_step = step
        clamped_error = max(0.0, min(1.0, float(error_magnitude)))
        if is_reset:
            # Resets or catastrophes inject high sudden surprise
            clamped_error = max(clamped_error, 0.85)

        self.error_history.append(clamped_error)
        self.short_error_window.append(clamped_error)
        self.long_error_window.append(clamped_error)

        # Exponential moving average update
        alpha = 1.0 - math.exp(-math.log(2.0) / max(1, self.volatility_half_life))
        delta = clamped_error - self.running_mean_error
        self.running_mean_error += alpha * delta
        self.running_variance_error = (1.0 - alpha) * (
            self.running_variance_error + alpha * (delta**2)
        )

        volatility = max(0.0, min(1.0, math.sqrt(self.running_variance_error)))
        is_volatile = volatility >= self.volatility_threshold

        # Compute Oudeyer / Schmidhuber Learning Progress (-dPE / dt)
        learning_progress = 0.0
        if len(self.long_error_window) >= 8 and len(self.short_error_window) >= 4:
            short_mean = float(np.mean(list(self.short_error_window)))
            long_mean = float(np.mean(list(self.long_error_window)))
            learning_progress = max(-1.0, min(1.0, long_mean - short_mean))

        # Variational precision weights:
        # High error & volatility suppresses sensory precision to prevent thrashing
        sensory_precision = 1.0 / max(0.10, volatility + clamped_error * 0.5 + 0.10)
        # Prior precision remains resilient unless volatility is extreme
        prior_precision = max(0.10, self.prior_rigidity * (1.1 - volatility))

        # Precision-weighted adaptive learning rate (Friston precision ratio)
        precision_ratio = sensory_precision / (sensory_precision + prior_precision)
        adaptive_lr = max(0.02, min(1.0, self.base_learning_rate * precision_ratio * 2.0))

        # Epistemic gain:
        # Base active inference probes modulated by learning progress:
        # High positive learning progress boosts exploration in the zone of proximal development.
        base_gain = 1.0 + 2.0 * volatility if is_volatile else max(0.5, 1.0 - volatility)
        if learning_progress > 0.0:
            epistemic_gain = base_gain * (1.0 + 1.5 * learning_progress)
        else:
            epistemic_gain = base_gain

        self.current_state = PrecisionState(
            sensory_precision=float(sensory_precision),
            prior_precision=float(prior_precision),
            environmental_volatility=float(volatility),
            adaptive_learning_rate=float(adaptive_lr),
            epistemic_exploration_gain=float(epistemic_gain),
            mean_prediction_error=float(self.running_mean_error),
            timestamp_step=step,
            is_volatile_regime=is_volatile,
            learning_progress=float(learning_progress),
        )
        return self.current_state

    def compute_sensory_diff_error(
        self,
        predicted_grid: np.ndarray | None,
        observed_grid: np.ndarray,
    ) -> float:
        """Compute normalized sensory prediction error between expected and observed grid."""
        if predicted_grid is None:
            return 0.5  # Neutral uncertainty when prior prediction is missing
        if predicted_grid.shape != observed_grid.shape:
            return 1.0

        diff_count = int(np.sum(predicted_grid != observed_grid))
        total_cells = max(1, observed_grid.size)
        return min(1.0, float(diff_count / (total_cells * 0.25)))

    def record_observation(
        self,
        step: int,
        predicted_grid: np.ndarray | None,
        observed_grid: np.ndarray,
        is_reset: bool = False,
    ) -> PrecisionState:
        """Compute error from grids and update precision state atomically."""
        error = self.compute_sensory_diff_error(predicted_grid, observed_grid)
        return self.update_with_prediction_error(step, error, is_reset=is_reset)
