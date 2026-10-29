from __future__ import annotations

"""Ventral Striatum & Lateral Habenula Dual Value-Opponency Learning.

Biologically grounded in:
1. M. Matsumoto & O. Hikosaka (2007) - "Lateral habenula as a source of negative
   reward signals in dopamine neurons" (Nature). Lateral habenula (LHb) neurons fire
   when expected rewards are omitted or when aversive threats are encountered,
   actively inhibiting dopamine neurons.
2. P. Dayan & N.D. Daw (2008) - "Decision theory, reinforcement learning, and the
   brain" (Cogn Affect Behav Neurosci). The brain maintains opponent appetitive
   (Ventral Striatum V+) and aversive (Lateral Habenula V-) value channels.
3. R. Cools et al. (2011) - Striatal dopamine and serotonin opponency in reward
   and punishment learning.

In HBLLM, this module prevents perseveration in deceptively tempting corridors by
computing net value opponency B(s) = V+(s) - λ V-(s) with Habenular avoidance braking.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


class OpponencyRegime(StrEnum):
    """Behavioral regime determined by appetitive vs. aversive value balance."""

    APPETITIVE_DOMINANT = "APPETITIVE_DOMINANT"  # Safe reward pursuit (B > 0.2)
    AVERSIVE_DOMINANT = "AVERSIVE_DOMINANT"  # Caution / threat warning (-0.5 <= B < -0.2)
    AVOIDANCE_BRAKE = (
        "AVOIDANCE_BRAKE"  # Emergency Habenular action inhibition (B < -0.5 or high hazard)
    )
    NEUTRAL = "NEUTRAL"  # Balanced or unvisited (-0.2 <= B <= 0.2)


@dataclass
class OpponencyValueState:
    """Snapshot of dual-channel appetitive vs. aversive value at a state or coordinate."""

    state_id: str
    v_plus: float = 0.0
    v_minus: float = 0.0
    balance: float = 0.0
    risk_aversion: float = 1.5
    regime: OpponencyRegime = OpponencyRegime.NEUTRAL
    is_braked: bool = False
    update_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export state opponency balance as an HCIR causal predicate."""
        return CausalPredicate(
            variable=f"opponency:{self.state_id}",
            operator="==",
            value=self.regime.value,
            relation=f"v_plus:{self.v_plus:.2f}:v_minus:{self.v_minus:.2f}:balance:{self.balance:.2f}:braked:{self.is_braked}",
            target_entity="habenula_striatum_opponency",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording state value opponency."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"opponency_{self.state_id}",
                node_data={
                    "state_id": self.state_id,
                    "v_plus": float(self.v_plus),
                    "v_minus": float(self.v_minus),
                    "balance": float(self.balance),
                    "risk_aversion": float(self.risk_aversion),
                    "regime": self.regime.value,
                    "is_braked": self.is_braked,
                    "update_count": int(self.update_count),
                },
            )
        ]


class DualValueOpponencyCortex:
    """Subcortical circuit modeling Ventral Striatal appetitive and Habenular aversive TD learning."""

    def __init__(
        self,
        learning_rate_plus: float = 0.25,
        learning_rate_minus: float = 0.35,
        discount_gamma: float = 0.90,
        risk_aversion_lambda: float = 1.50,
        brake_threshold: float = 0.60,
    ) -> None:
        self.alpha_plus = learning_rate_plus
        self.alpha_minus = learning_rate_minus
        self.gamma = discount_gamma
        self.risk_aversion = risk_aversion_lambda
        self.brake_threshold = brake_threshold
        self.states: dict[str, OpponencyValueState] = {}

    def _state_key_to_str(self, key: tuple[int, int] | str) -> str:
        if isinstance(key, tuple):
            return f"({key[0]},{key[1]})"
        return str(key)

    def get_state(self, state_key: tuple[int, int] | str) -> OpponencyValueState:
        """Retrieve or initialize opponency state."""
        sid = self._state_key_to_str(state_key)
        if sid not in self.states:
            self.states[sid] = OpponencyValueState(
                state_id=sid,
                risk_aversion=self.risk_aversion,
            )
        return self.states[sid]

    def update_transition(
        self,
        state_key: tuple[int, int] | str,
        reward_plus: float,
        punishment_minus: float,
        next_state_key: tuple[int, int] | str | None = None,
    ) -> OpponencyValueState:
        """Update dual appetitive and aversive values using opponent Temporal Difference learning."""
        curr = self.get_state(state_key)

        next_v_plus = 0.0
        next_v_minus = 0.0
        if next_state_key is not None:
            nxt = self.get_state(next_state_key)
            next_v_plus = nxt.v_plus
            next_v_minus = nxt.v_minus

        # 1. Appetitive Ventral Striatum TD error & update
        td_plus = reward_plus + (self.gamma * next_v_plus) - curr.v_plus
        new_v_plus = max(0.0, curr.v_plus + (self.alpha_plus * td_plus))

        # 2. Aversive Lateral Habenula TD error & update
        td_minus = punishment_minus + (self.gamma * next_v_minus) - curr.v_minus
        new_v_minus = max(0.0, curr.v_minus + (self.alpha_minus * td_minus))

        # 3. Opponency balance B(s) = V+(s) - λ V-(s)
        balance = new_v_plus - (self.risk_aversion * new_v_minus)

        # 4. Regime classification & Habenular brake gating
        is_braked = new_v_minus >= self.brake_threshold or balance < -0.50

        if is_braked:
            regime = OpponencyRegime.AVOIDANCE_BRAKE
        elif balance < -0.20:
            regime = OpponencyRegime.AVERSIVE_DOMINANT
        elif balance > 0.20:
            regime = OpponencyRegime.APPETITIVE_DOMINANT
        else:
            regime = OpponencyRegime.NEUTRAL

        curr.v_plus = new_v_plus
        curr.v_minus = new_v_minus
        curr.balance = balance
        curr.regime = regime
        curr.is_braked = is_braked
        curr.update_count += 1

        return curr

    def should_inhibit_action(self, target_state_key: tuple[int, int] | str) -> tuple[bool, str]:
        """Check whether Lateral Habenular avoidance brake actively inhibits moving to target state."""
        st = self.get_state(target_state_key)
        if st.is_braked:
            return (
                True,
                f"Lateral Habenula brake active: punishment V-={st.v_minus:.2f} >= threshold {self.brake_threshold:.2f} (balance {st.balance:.2f}).",
            )
        return False, "Target state permitted by value opponency circuit."

    def arbitrate_actions(
        self, candidate_transitions: dict[Any, tuple[int, int] | str]
    ) -> dict[Any, float]:
        """Score candidate actions by predicted net value opponency balance B(s')."""
        scores: dict[Any, float] = {}
        for action, next_st in candidate_transitions.items():
            st = self.get_state(next_st)
            if st.is_braked:
                # Heavily penalize braked actions
                scores[action] = -10.0 + st.balance
            else:
                scores[action] = st.balance
        return scores

    def to_causal_predicate(self) -> CausalPredicate:
        """Export global opponency summary as an HCIR causal predicate."""
        braked_count = sum(1 for s in self.states.values() if s.is_braked)
        return CausalPredicate(
            variable="opponency:cortex_summary",
            operator="==",
            value=f"total_states:{len(self.states)}:braked:{braked_count}",
            relation=f"risk_aversion:{self.risk_aversion:.2f}:brake_thresh:{self.brake_threshold:.2f}",
            target_entity="habenula_striatum_opponency",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions recording all opponency states."""
        ops: list[TransactionOperation] = []
        for s in self.states.values():
            ops.extend(s.to_transaction_operations())
        return ops
