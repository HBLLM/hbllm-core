"""Ventrolateral Prefrontal Cortex (vlPFC / rIFG) Motor Rule Retrieval & Response Inhibition.

Biological Foundations:
- Aron, Robbins, Poldrack (2004): "Inhibition and the right inferior frontal cortex"
  (Trends Cogn Sci 8: 170-177).
- Aron (2011): "From Reactive to Proactive and Selective Control: Developing a Multitask
  Approach to Executive Function" (Neuron).
- Wiecki & Frank (2013): "A computational model of inhibitory control in the basal ganglia:
  Subthalamic nucleus and hyperdirect pathway".

Key Cognitive Capabilities:
1. Proactive vs Reactive Motor Braking:
   - Proactive Braking: Up-regulates response caution (latency / threshold) when the
     environment is known to carry hazards or volatile dynamics.
   - Reactive Stopping: Rapid hyperdirect-pathway veto aborting an ongoing or pending
     motor command when an unexpected obstacle, lethal hazard, or conflict occurs.
2. Rule-Based Motor Gating:
   - Evaluates intended motor actions against learned symbolic constraints (e.g. "do not
     move into wall", "do not access locked zone without credential").
3. Action Cancellation & Substitution:
   - Proposes safe fallback or evasive actions (e.g. retreat or 'WAIT') when an action
     is inhibited.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hbllm.hcir.world.causal_discovery import CausalPredicate


class InhibitionRegime(str, Enum):
    """Operational regimes of the ventrolateral prefrontal motor inhibitory circuit."""

    UNRESTRICTED = "unrestricted"
    PROACTIVE_BRAKING = "proactive_braking"
    REACTIVE_STOP = "reactive_stop"
    EMERGENCY_ABORT = "emergency_abort"


@dataclass(frozen=True)
class ResponseInhibitionCommand:
    """Motor gating command emitted by the ventrolateral prefrontal cortex."""

    permitted: bool
    regime: InhibitionRegime
    brake_factor: float
    intended_action: str | int
    sanctioned_action: str | int | None
    cancellation_reason: str

    def to_causal_predicate(self) -> CausalPredicate:
        """Serialize gating command to an HCIR CausalPredicate."""
        return CausalPredicate(
            variable="motor_response_inhibition",
            operator="==",
            value=f"permitted:{self.permitted}:regime:{self.regime.value}:brake:{self.brake_factor:.2f}:action:{self.intended_action}",
            relation="vlpfc_motor_inhibition",
            target_entity="right_inferior_frontal_cortex",
        )


class VentrolateralMotorInhibitionEngine:
    """Orchestrator for right IFG / vlPFC motor rule retrieval and response inhibition."""

    def __init__(
        self,
        reactive_decay_rate: float = 2.0,
        baseline_brake: float = 0.0,
    ) -> None:
        """Initialize motor inhibition engine.

        Args:
            reactive_decay_rate: Rate at which transient reactive stopping decays per second.
            baseline_brake: Baseline proactive brake level in [0.0, 1.0].
        """
        self.reactive_decay_rate: float = max(0.1, float(reactive_decay_rate))
        self.proactive_brake: float = max(0.0, min(1.0, float(baseline_brake)))

        # Transient reactive stop strength in [0.0, 1.0]
        self._reactive_stop_strength: float = 0.0
        self._reactive_stop_reason: str = ""
        self._emergency_flag: bool = False

        # Inhibitory rule registry: (condition_key, forbidden_action) -> safe_alternative
        self._inhibitory_rules: dict[tuple[str, str], str | int | None] = {}

    def set_proactive_brake(self, level: float, reason: str = "") -> None:
        """Set proactive cautiousness level in [0.0, 1.0]."""
        self.proactive_brake = max(0.0, min(1.0, float(level)))

    def trigger_reactive_stop(self, source: str, severity: float = 1.0) -> None:
        """Trigger an immediate hyperdirect reactive stop."""
        sev = max(0.0, min(1.0, float(severity)))
        self._reactive_stop_strength = max(self._reactive_stop_strength, sev)
        self._reactive_stop_reason = str(source)
        if sev >= 0.90:
            self._emergency_flag = True

    def decay_inhibition(self, dt: float) -> None:
        """Decay transient reactive inhibition over time dt."""
        step_dt = max(0.0, float(dt))
        self._reactive_stop_strength = max(
            0.0, self._reactive_stop_strength - self.reactive_decay_rate * step_dt
        )
        if self._reactive_stop_strength <= 0.05:
            self._reactive_stop_reason = ""
            self._emergency_flag = False

    def register_inhibitory_rule(
        self,
        condition_key: str,
        forbidden_action: str | int,
        safe_alternative: str | int | None = None,
    ) -> None:
        """Register a symbolic rule forbidding an action under a given condition."""
        key = (str(condition_key), str(forbidden_action))
        self._inhibitory_rules[key] = safe_alternative

    def evaluate_action_gating(
        self,
        intended_action: str | int,
        conflict_level: float = 0.0,
        risk_score: float = 0.0,
        time_to_impact: float = float("inf"),
        active_conditions: set[str] | list[str] | None = None,
    ) -> ResponseInhibitionCommand:
        """Evaluate whether intended action is permitted or must be inhibited.

        Args:
            intended_action: Proposed action identifier (e.g. movement direction or action ID).
            conflict_level: Anterior cingulate conflict level in [0.0, 1.0].
            risk_score: Estimated environmental or somatic risk score in [0.0, 1.0].
            time_to_impact: Time or distance to collision/threat.
            active_conditions: Currently active symbolic context conditions (e.g. {"ice", "wall_ahead"}).

        Returns:
            ResponseInhibitionCommand indicating permission, regime, brake factor, and alternatives.
        """
        # 1. Emergency Abort Check
        if self._emergency_flag or (time_to_impact <= 0.0 and risk_score >= 0.80):
            return ResponseInhibitionCommand(
                permitted=False,
                regime=InhibitionRegime.EMERGENCY_ABORT,
                brake_factor=1.0,
                intended_action=intended_action,
                sanctioned_action=None,
                cancellation_reason=self._reactive_stop_reason or "emergency_collision_abort",
            )

        # 2. Reactive Stop Check
        if self._reactive_stop_strength >= 0.50:
            return ResponseInhibitionCommand(
                permitted=False,
                regime=InhibitionRegime.REACTIVE_STOP,
                brake_factor=self._reactive_stop_strength,
                intended_action=intended_action,
                sanctioned_action=None,
                cancellation_reason=self._reactive_stop_reason or "reactive_hyperdirect_stop",
            )

        # 3. Rule-Based Symbolic Motor Gating
        conds = set(active_conditions or [])
        action_str = str(intended_action)
        for cond in conds:
            rule_key = (cond, action_str)
            if rule_key in self._inhibitory_rules:
                alternative = self._inhibitory_rules[rule_key]
                return ResponseInhibitionCommand(
                    permitted=False,
                    regime=InhibitionRegime.PROACTIVE_BRAKING,
                    brake_factor=0.85,
                    intended_action=intended_action,
                    sanctioned_action=alternative,
                    cancellation_reason=f"symbolic_rule_violation:{cond}",
                )

        # 4. Proactive / High-Conflict Braking Modulation
        effective_brake = max(self.proactive_brake, risk_score * 0.70, conflict_level * 0.50)

        # High risk with short time to impact triggers reactive suppression
        if risk_score > 0.85 and time_to_impact <= 1.0:
            return ResponseInhibitionCommand(
                permitted=False,
                regime=InhibitionRegime.REACTIVE_STOP,
                brake_factor=0.90,
                intended_action=intended_action,
                sanctioned_action=None,
                cancellation_reason="looming_hazard_proximity",
            )

        regime = (
            InhibitionRegime.PROACTIVE_BRAKING
            if effective_brake > 0.30
            else InhibitionRegime.UNRESTRICTED
        )

        return ResponseInhibitionCommand(
            permitted=True,
            regime=regime,
            brake_factor=float(effective_brake),
            intended_action=intended_action,
            sanctioned_action=intended_action,
            cancellation_reason="",
        )
