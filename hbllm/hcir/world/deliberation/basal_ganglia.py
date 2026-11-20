"""Basal Ganglia Cortico-Striatal Motor Arbitration (Stage 4 Deliberation)."""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


class BasalGangliaPathway(StrEnum):
    """Cortico-striatal motor pathway classification."""

    DIRECT_GO = "DIRECT_GO"  # Striatal D1 MSN facilitation (Go)
    INDIRECT_NOGO = "INDIRECT_NOGO"  # Striatal D2 MSN suppression (NoGo)
    HYPERDIRECT_BRAKE = "HYPERDIRECT_BRAKE"  # Subthalamic Nucleus (STN) global pause/brake


@dataclass
class BasalGangliaDecision:
    """Action arbitration outcome from the Basal Ganglia tri-pathway circuit."""

    action: Any
    pathway: BasalGangliaPathway
    d1_go_drive: float
    d2_nogo_drive: float
    stn_brake_signal: float
    net_drive: float
    is_suppressed: bool
    reason: str = ""

    def to_causal_predicate(self) -> CausalPredicate:
        """Export basal ganglia decision as a declarative causal predicate."""
        return CausalPredicate(
            variable="basal_ganglia:pathway",
            operator="==",
            value=self.pathway.value,
            relation=f"action:{self.action}:drive:{self.net_drive:.3f}:suppressed:{self.is_suppressed}",
            target_entity=str(self.action),
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording basal ganglia decision."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"bg_decision_{self.action}",
                node_data={
                    "action": str(self.action),
                    "pathway": self.pathway.value,
                    "d1_go_drive": float(self.d1_go_drive),
                    "d2_nogo_drive": float(self.d2_nogo_drive),
                    "stn_brake_signal": float(self.stn_brake_signal),
                    "net_drive": float(self.net_drive),
                    "is_suppressed": self.is_suppressed,
                    "reason": self.reason,
                },
            )
        ]


class BasalGangliaGatingCircuit:
    """Biologically-modeled Basal Ganglia Tri-Pathway Action Arbitration Circuit.

    Models the competition between:
    1. Direct Pathway (Striatal D1 MSNs): Facilitates action execution by disinhibiting the thalamus.
       Modulated positively by dopamine: higher tonic DA / positive RPE increases D1 Go gain.
    2. Indirect Pathway (Striatal D2 MSNs): Suppresses action execution by increasing GPi inhibition.
       Modulated inversely by dopamine: lower tonic DA / negative RPE increases D2 NoGo drive.
    3. Hyperdirect Pathway (Cortical ACC/preSMA -> Subthalamic Nucleus / STN -> GPi):
       Fires upon high conflict (ACC), looming threats, or high surprise, exerting a global
       emergency brake that suppresses candidate execution before catastrophic motor commitment.
    """

    def __init__(
        self,
        w_d1: float = 1.0,
        w_d2: float = 1.2,
        brake_threshold: float = 0.65,
        da_sensitivity: float = 0.5,
    ) -> None:
        self.w_d1 = w_d1
        self.w_d2 = w_d2
        self.brake_threshold = brake_threshold
        self.da_sensitivity = da_sensitivity
        self.arbitration_history: deque[BasalGangliaDecision] = deque(maxlen=64)

    def compute_hyperdirect_brake(
        self,
        acc_conflict: float = 0.0,
        threat_looming: float = 0.0,
        surprise_entropy: float = 0.0,
    ) -> float:
        """Compute instantaneous Subthalamic Nucleus (STN) hyperdirect brake signal."""
        combined = 0.5 * acc_conflict + 0.35 * threat_looming + 0.15 * surprise_entropy
        return float(1.0 / (1.0 + math.exp(-6.0 * (combined - 0.45))))

    def arbitrate_candidates(
        self,
        candidate_utilities: dict[Any, float],
        dopamine_level: float = 1.0,
        acc_conflict: float = 0.0,
        threat_looming: float = 0.0,
        surprise_entropy: float = 0.0,
        habenular_inhibitions: dict[Any, float] | None = None,
        risk_penalties: dict[Any, float] | None = None,
    ) -> tuple[BasalGangliaDecision | None, list[BasalGangliaDecision]]:
        """Arbitrate candidate actions through the 3 cortico-striatal pathways.

        Returns:
            (winning_decision_or_none, all_evaluated_decisions)
        """
        if not candidate_utilities:
            return None, []

        stn_brake = self.compute_hyperdirect_brake(
            acc_conflict=acc_conflict,
            threat_looming=threat_looming,
            surprise_entropy=surprise_entropy,
        )
        global_braked = stn_brake >= self.brake_threshold

        hab_dict = habenular_inhibitions or {}
        risk_dict = risk_penalties or {}

        # Dopamine gain modulation (D1 boosted by DA, D2 attenuated by DA)
        da_delta = float(np.clip(dopamine_level - 1.0, -1.0, 2.0))
        d1_mod = max(0.1, 1.0 + self.da_sensitivity * da_delta)
        d2_mod = max(0.1, 1.0 - self.da_sensitivity * da_delta)

        decisions: list[BasalGangliaDecision] = []

        for action, utility in candidate_utilities.items():
            # D1 Go pathway: utility scaled by D1 gain
            d1_go = float(1.0 / (1.0 + math.exp(-3.0 * (self.w_d1 * utility * d1_mod - 0.2))))

            # D2 NoGo pathway: Habenular failure history + risk penalty scaled by D2 gain
            hab_cost = hab_dict.get(action, 0.0)
            risk_cost = risk_dict.get(action, 0.0)
            nogo_input = (hab_cost + risk_cost) * self.w_d2 * d2_mod
            d2_nogo = float(1.0 / (1.0 + math.exp(-3.0 * (nogo_input - 0.3))))

            net_drive = d1_go - d2_nogo - (stn_brake if global_braked else 0.0)

            if global_braked:
                pathway = BasalGangliaPathway.HYPERDIRECT_BRAKE
                suppressed = True
                reason = f"Hyperdirect STN brake triggered (brake={stn_brake:.2f} >= {self.brake_threshold})"
            elif d2_nogo > d1_go:
                pathway = BasalGangliaPathway.INDIRECT_NOGO
                suppressed = True
                reason = f"Indirect D2 suppression (nogo={d2_nogo:.2f} > go={d1_go:.2f})"
            else:
                pathway = BasalGangliaPathway.DIRECT_GO
                suppressed = False
                reason = f"Direct D1 disinhibition (go={d1_go:.2f} > nogo={d2_nogo:.2f})"

            dec = BasalGangliaDecision(
                action=action,
                pathway=pathway,
                d1_go_drive=d1_go,
                d2_nogo_drive=d2_nogo,
                stn_brake_signal=stn_brake,
                net_drive=net_drive,
                is_suppressed=suppressed,
                reason=reason,
            )
            decisions.append(dec)

        # Sort candidate decisions by net drive descending
        decisions.sort(key=lambda d: d.net_drive, reverse=True)

        winning: BasalGangliaDecision | None = None
        if not global_braked:
            for d in decisions:
                if not d.is_suppressed and d.pathway == BasalGangliaPathway.DIRECT_GO:
                    winning = d
                    break
        else:
            winning = decisions[0] if decisions else None

        if winning is not None:
            self.arbitration_history.append(winning)

        return winning, decisions
