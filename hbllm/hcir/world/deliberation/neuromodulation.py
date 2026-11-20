"""Dual Neuromodulation: VTA Dopamine & LC Noradrenaline (Adaptive Gain)."""

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
class NeuromodulatoryState:
    """Composite biological neuromodulatory state governing exploration vs. exploitation."""

    dopamine_tonic: float = 1.0
    dopamine_phasic_rpe: float = 0.0
    noradrenaline_tonic: float = 0.2
    noradrenaline_phasic: float = 0.1
    exploration_gain: float = 1.0
    exploitation_gain: float = 1.0
    is_disengaged: bool = False
    regime: str = "EXPLOITATION"  # "EXPLOITATION", "FOCAL_ENGAGEMENT", "GLOBAL_EXPLORATION"

    def to_causal_predicate(self) -> CausalPredicate:
        """Export neuromodulatory state as a declarative causal predicate."""
        return CausalPredicate(
            variable="neuromodulation:state",
            operator="==",
            value=self.regime,
            relation=f"da_tonic:{self.dopamine_tonic:.2f}:na_tonic:{self.noradrenaline_tonic:.2f}:disengaged:{self.is_disengaged}",
            target_entity="adaptive_governor",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording neuromodulatory state."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="neuromodulatory_state",
                node_data={
                    "dopamine_tonic": float(self.dopamine_tonic),
                    "dopamine_phasic_rpe": float(self.dopamine_phasic_rpe),
                    "noradrenaline_tonic": float(self.noradrenaline_tonic),
                    "noradrenaline_phasic": float(self.noradrenaline_phasic),
                    "exploration_gain": float(self.exploration_gain),
                    "exploitation_gain": float(self.exploitation_gain),
                    "is_disengaged": self.is_disengaged,
                    "regime": self.regime,
                },
            )
        ]


class AdaptiveNeuromodulationGovernor:
    """Dual Neuromodulatory Governor: VTA Dopamine (RPE) & LC Noradrenaline (Adaptive Gain).

    Modeled on:
    - Schultz (1998): Ventral Tegmental Area (VTA) dopaminergic Reward Prediction Error (RPE).
    - Aston-Jones & Cohen (2005) Adaptive Gain Theory: Locus Coeruleus (LC) noradrenergic gating.
      * Phasic LC bursts optimize task exploitation and signal-to-noise ratio.
      * High tonic LC promotes behavioral disengagement from unproductive local loops,
        triggering broad global exploration.
    """

    def __init__(
        self,
        gamma: float = 0.9,
        da_decay: float = 0.1,
        na_decay: float = 0.15,
        disengagement_threshold: float = 0.65,
    ) -> None:
        self.gamma = gamma
        self.da_decay = da_decay
        self.na_decay = na_decay
        self.disengagement_threshold = disengagement_threshold

        self.current_state = NeuromodulatoryState()
        self.history: deque[NeuromodulatoryState] = deque(maxlen=64)

    def step_dynamics(
        self,
        reward: float = 0.0,
        predicted_value: float = 0.0,
        actual_value: float = 0.0,
        conflict_level: float = 0.0,
        stuck_steps: int = 0,
    ) -> NeuromodulatoryState:
        """Step VTA-LC coupled dynamics based on reward progress and cognitive conflict."""
        # 1. VTA Dopamine: Reward Prediction Error (RPE)
        rpe = reward + self.gamma * actual_value - predicted_value
        tonic_da = (1.0 - self.da_decay) * self.current_state.dopamine_tonic + self.da_decay * (
            1.0 + rpe
        )
        tonic_da = float(np.clip(tonic_da, 0.1, 2.5))

        # 2. LC Noradrenaline: Phasic burst on acute conflict / salient prediction error
        phasic_na = float(1.0 / (1.0 + math.exp(-5.0 * (abs(rpe) + conflict_level - 0.5))))

        # Tonic NA: Slowly integrates persistent lack of progress / stuck counter
        stuck_pressure = min(2.0, float(stuck_steps) / 3.0)
        tonic_na = (
            1.0 - self.na_decay
        ) * self.current_state.noradrenaline_tonic + self.na_decay * stuck_pressure
        tonic_na = float(np.clip(tonic_na, 0.05, 2.0))

        # 3. Behavioral Regime Determination
        is_disengaged = tonic_na >= self.disengagement_threshold or stuck_steps >= 4

        if is_disengaged:
            regime = "GLOBAL_EXPLORATION"
            exploration_gain = 1.0 + 1.5 * tonic_na
            exploitation_gain = max(0.1, 1.0 - 0.7 * tonic_na)
        elif phasic_na > 0.6 and tonic_da > 0.8:
            regime = "FOCAL_ENGAGEMENT"
            exploration_gain = 0.4
            exploitation_gain = 1.0 + 0.8 * tonic_da
        else:
            regime = "EXPLOITATION"
            exploration_gain = 1.0
            exploitation_gain = 1.0

        new_state = NeuromodulatoryState(
            dopamine_tonic=tonic_da,
            dopamine_phasic_rpe=float(rpe),
            noradrenaline_tonic=tonic_na,
            noradrenaline_phasic=phasic_na,
            exploration_gain=exploration_gain,
            exploitation_gain=exploitation_gain,
            is_disengaged=is_disengaged,
            regime=regime,
        )
        self.current_state = new_state
        self.history.append(new_state)
        return new_state
