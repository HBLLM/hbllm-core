"""The Four Neuromodulatory Systems (Dopamine, Noradrenaline, Acetylcholine, Serotonin).

Biologically grounded in Kenji Doya's computational neuromodulation framework
(Doya, 2002) and Wolfram Schultz's Reward Prediction Error models (Schultz, 1998):
1. Dopamine (DA) — Reward Prediction Error & Exploitation/Exploration Balance:
   - Positive RPE surges DA -> increases synaptic plasticity/learning rate kappa
     and exploration temperature.
   - Negative RPE suppresses DA -> encourages exploitation of established safe affordances.
2. Noradrenaline / Norepinephrine (NE) — Unexpected Uncertainty & Phase Resets:
   - Activated by unpredicted structural shocks, sudden deaths, or high surprise spikes.
   - High NE prompts executive phase resets (e.g., from simulation back to exploratory grounding).
3. Acetylcholine (ACh) — Expected Uncertainty & Sensory Precision Weighting:
   - Tracks reliable environmental features vs background stochasticity.
   - Directly modulates sensory precision Pi_s in Hierarchical Variational Free Energy.
4. Serotonin (5-HT) — Temporal Discounting Horizon & Risk Aversion:
   - High 5-HT extends the effective discount horizon gamma, enabling patient multi-subgoal planning.
   - Low 5-HT / stress contracts the discount horizon toward immediate short-term survival.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class NeuromodulatorLevels:
    """Quantitative concentrations of the primary biological neuromodulators."""

    dopamine: float = 1.0  # RPE, plasticity & exploration
    noradrenaline: float = 1.0  # Unexpected uncertainty & phase resets
    acetylcholine: float = 1.0  # Expected uncertainty & sensory precision
    serotonin: float = 1.0  # Temporal discount horizon & patience


class NeuromodulatorySystem:
    """Master neuromodulatory regulation engine dynamically tuning cognitive meta-parameters."""

    def __init__(
        self,
        baseline_da: float = 1.0,
        baseline_ne: float = 1.0,
        baseline_ach: float = 1.0,
        baseline_5ht: float = 1.0,
        homeostatic_decay: float = 0.08,
    ) -> None:
        self.baseline = NeuromodulatorLevels(
            dopamine=baseline_da,
            noradrenaline=baseline_ne,
            acetylcholine=baseline_ach,
            serotonin=baseline_5ht,
        )
        self.current = NeuromodulatorLevels(
            dopamine=baseline_da,
            noradrenaline=baseline_ne,
            acetylcholine=baseline_ach,
            serotonin=baseline_5ht,
        )
        self.homeostatic_decay = homeostatic_decay
        self.step_history: list[dict[str, float]] = []

    def update_reward_prediction_error(self, rpe: float) -> float:
        """Update Dopamine (DA) based on temporal-difference Reward Prediction Error."""
        # Surge on positive surprises, dip on disappointments
        delta = 0.4 * float(rpe)
        self.current.dopamine = max(0.1, min(3.0, self.current.dopamine + delta))
        return self.current.dopamine

    def update_unexpected_uncertainty(self, surprise: float, is_unpredicted: bool = True) -> float:
        """Update Noradrenaline (NE) driven by surprise spikes and catastrophic environmental shifts."""
        if is_unpredicted and surprise > 0.3:
            delta = 0.5 * float(surprise)
            self.current.noradrenaline = max(0.1, min(3.0, self.current.noradrenaline + delta))
        return self.current.noradrenaline

    def update_expected_uncertainty(
        self, prediction_accuracy: float, noise_level: float = 0.0
    ) -> float:
        """Update Acetylcholine (ACh) based on sensory channel reliability."""
        reliability = max(-1.0, min(1.0, float(prediction_accuracy) - float(noise_level)))
        delta = 0.3 * reliability
        self.current.acetylcholine = max(0.1, min(3.0, self.current.acetylcholine + delta))
        return self.current.acetylcholine

    def update_temporal_stress(self, risk_level: float, time_pressure: float = 0.0) -> float:
        """Update Serotonin (5-HT) based on existential threat and urgent temporal stress."""
        threat = max(0.0, min(1.0, float(risk_level) + float(time_pressure)))
        # Stress decreases 5-HT (forcing urgent, low-gamma focus); safety increases 5-HT
        if threat > 0.5:
            self.current.serotonin = max(0.2, self.current.serotonin - 0.3 * threat)
        else:
            self.current.serotonin = min(2.5, self.current.serotonin + 0.1)
        return self.current.serotonin

    def step_homeostatic_decay(self) -> None:
        """Decay elevated or depressed concentrations toward homeostatic baseline."""
        self.current.dopamine += self.homeostatic_decay * (
            self.baseline.dopamine - self.current.dopamine
        )
        self.current.noradrenaline += self.homeostatic_decay * (
            self.baseline.noradrenaline - self.current.noradrenaline
        )
        self.current.acetylcholine += self.homeostatic_decay * (
            self.baseline.acetylcholine - self.current.acetylcholine
        )
        self.current.serotonin += self.homeostatic_decay * (
            self.baseline.serotonin - self.current.serotonin
        )

        record = {
            "dopamine": self.current.dopamine,
            "noradrenaline": self.current.noradrenaline,
            "acetylcholine": self.current.acetylcholine,
            "serotonin": self.current.serotonin,
        }
        self.step_history.append(record)

    def get_effective_hyperparameters(self) -> dict[str, Any]:
        """Derive dynamic meta-parameters from the current neuromodulatory profile."""
        # 1. Dopamine regulates exploration temperature and learning plasticity
        exploration_temp = max(0.1, min(2.0, 0.7 * self.current.dopamine))
        plasticity_kappa = max(0.1, min(1.0, 0.5 * (0.5 + 0.5 * self.current.dopamine)))

        # 2. Noradrenaline regulates cognitive flexibility and phase reset threshold
        phase_reset_recommended = self.current.noradrenaline >= 2.0

        # 3. Acetylcholine regulates sensory precision Pi_s in active inference
        sensory_precision = max(0.1, min(4.0, 1.0 * self.current.acetylcholine))

        # 4. Serotonin regulates temporal discounting factor gamma and planning horizon
        gamma = max(0.50, min(0.98, 0.85 + 0.08 * (self.current.serotonin - 1.0)))
        planning_horizon = max(3, min(25, int(round(12 * self.current.serotonin))))

        return {
            "exploration_temperature": exploration_temp,
            "plasticity_kappa": plasticity_kappa,
            "sensory_precision": sensory_precision,
            "discount_factor_gamma": gamma,
            "planning_horizon": planning_horizon,
            "phase_reset_recommended": phase_reset_recommended,
        }

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export neuromodulator concentrations as declarative causal predicates."""
        return [
            CausalPredicate(
                variable="neuromodulator:dopamine",
                operator="==",
                value=round(self.current.dopamine, 3),
                relation="rpe_plasticity",
                target_entity="neuromodulatory_system",
            ),
            CausalPredicate(
                variable="neuromodulator:noradrenaline",
                operator="==",
                value=round(self.current.noradrenaline, 3),
                relation="unexpected_uncertainty",
                target_entity="neuromodulatory_system",
            ),
            CausalPredicate(
                variable="neuromodulator:acetylcholine",
                operator="==",
                value=round(self.current.acetylcholine, 3),
                relation="sensory_precision",
                target_entity="neuromodulatory_system",
            ),
            CausalPredicate(
                variable="neuromodulator:serotonin",
                operator="==",
                value=round(self.current.serotonin, 3),
                relation="temporal_discount_horizon",
                target_entity="neuromodulatory_system",
            ),
        ]

    def to_transaction_operations(self, tenant_id: str = "default") -> list[TransactionOperation]:
        """Propose atomic HCIR transaction logging neuromodulator levels in working memory."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="neuromodulatory_state",
                node_data={
                    "dopamine": float(self.current.dopamine),
                    "noradrenaline": float(self.current.noradrenaline),
                    "acetylcholine": float(self.current.acetylcholine),
                    "serotonin": float(self.current.serotonin),
                    "effective_hyperparameters": self.get_effective_hyperparameters(),
                    "tenant_id": tenant_id,
                },
            )
        ]
