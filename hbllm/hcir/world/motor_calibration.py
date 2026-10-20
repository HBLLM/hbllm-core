"""Motor Calibration — Empirical action dynamics and state mutation models.

Provides domain-agnostic data models for tracking:
- Action → displacement / outcome mappings (ActionDynamicsModel)
- Environmental state change rules (StateMutationModel)

These are learned through empirical probing and trial-and-error interaction,
mimicking human motor and causal discovery. All state evolutions can be
lifted to formal CausalPredicates, ActionAffordances, and atomic TransactionOperations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.cortex_causal import ActionAffordance, CausalPredicate

logger = logging.getLogger(__name__)


class TriggerType(StrEnum):
    """Environmental event triggers for state mutations."""

    CONTACT = "CONTACT"
    ACTION = "ACTION"
    PROXIMITY = "PROXIMITY"
    TIMER = "TIMER"
    SENSOR = "SENSOR"


class MutationType(StrEnum):
    """Observable environmental mutation categories."""

    PROPERTY_CHANGE = "PROPERTY_CHANGE"
    BARRIER_OPEN = "BARRIER_OPEN"
    ROTATION = "ROTATION"
    STATE_CHANGE = "STATE_CHANGE"
    COLOR_REMAP = "COLOR_REMAP"
    HOLDING_CHANGE = "HOLDING_CHANGE"
    MUTATION = "MUTATION"


@dataclass
class ActionDynamicsModel:
    """Empirical causal model mapping an action to an observed displacement vector.

    Supports condition-specific branches (e.g. carrying an item, control mode,
    environmental state) to prevent conflicting observations from corrupting
    the exponential moving average.
    """

    action_id: int | str
    delta_r: int = 0
    delta_c: int = 0
    confidence: float = 0.5
    probes_tested: int = 0
    displacement: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    conditional_branches: dict[str, ActionDynamicsModel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.displacement:
            self.displacement = (float(self.delta_r), float(self.delta_c))
        elif self.delta_r == 0 and self.delta_c == 0 and len(self.displacement) >= 2:
            self.delta_r = int(round(self.displacement[0]))
            self.delta_c = int(round(self.displacement[1]))

    def get_dynamics(self, condition: str | None = None) -> ActionDynamicsModel:
        """Retrieve the dynamics model for a specific condition or fallback to self."""
        if condition and condition in self.conditional_branches:
            return self.conditional_branches[condition]
        return self

    def get_displacement(self, condition: str | None = None) -> tuple[int, int]:
        """Get (delta_r, delta_c) displacement for a condition."""
        dyn = self.get_dynamics(condition)
        dr, dc = dyn.delta_r, dyn.delta_c
        if abs(dc) >= 2 * abs(dr):
            dr = 0
        elif abs(dr) >= 2 * abs(dc):
            dc = 0
        return dr, dc

    def is_displacement_action(self, condition: str | None = None) -> bool:
        """True if empirical testing demonstrates spatial translation of the body."""
        dr, dc = self.get_displacement(condition)
        return dr != 0 or dc != 0

    def update_from_trial(
        self,
        observed_delta: tuple[int | float, ...],
        success: bool = True,
        learning_rate: float = 0.5,
        condition: str | None = None,
    ) -> None:
        """Update empirical dynamics through trial-and-error observation.

        If a condition is provided, updates or registers the condition-specific branch.
        If unconditioned but observing a sharp contradiction on a calibrated model,
        forks into a conditional branch rather than corrupting the established EMA.
        """
        self.probes_tested += 1

        if condition is not None and condition != "default":
            if condition not in self.conditional_branches:
                dr = int(round(float(observed_delta[0]))) if len(observed_delta) >= 1 else 0
                dc = int(round(float(observed_delta[1]))) if len(observed_delta) >= 2 else 0
                if abs(dc) >= 2 * abs(dr):
                    dr = 0
                elif abs(dr) >= 2 * abs(dc):
                    dc = 0
                self.conditional_branches[condition] = ActionDynamicsModel(
                    action_id=self.action_id,
                    delta_r=dr,
                    delta_c=dc,
                    confidence=0.6 if success else 0.3,
                    probes_tested=1,
                    displacement=(float(dr), float(dc)),
                )
            else:
                self.conditional_branches[condition].update_from_trial(
                    observed_delta, success=success, learning_rate=learning_rate
                )
            return

        if success:
            if len(observed_delta) >= 2:
                dr, dc = float(observed_delta[0]), float(observed_delta[1])
                target_r = int(round(dr))
                target_c = int(round(dc))
                if abs(target_c) >= 2 * abs(target_r):
                    target_r = 0
                elif abs(target_r) >= 2 * abs(target_c):
                    target_c = 0

                # Ensure baseline self.delta_r and self.delta_c are snapped if dominant axis exists
                if abs(self.delta_c) >= 2 * abs(self.delta_r):
                    self.delta_r = 0
                elif abs(self.delta_r) >= 2 * abs(self.delta_c):
                    self.delta_c = 0

                # Single-step quantum refinement & unconstrained stride expansion:
                # If an action previously observed a boundary wrap (e.g. 28) and now demonstrates
                # a smaller non-zero unit step along the same axis (e.g. 7), adopt the unit quantum.
                if 0 < abs(target_r) < abs(self.delta_r) and self.delta_r != 0:
                    self.delta_r = target_r
                if 0 < abs(target_c) < abs(self.delta_c) and self.delta_c != 0:
                    self.delta_c = target_c
                # If an action previously observed a partial/truncated step (e.g. 3) and now demonstrates
                # a larger integer stride along the same axis (e.g. 4 or 6), adopt the unconstrained stride.
                same_dir_r = self.delta_r * target_r > 0
                same_dir_c = self.delta_c * target_c > 0
                if abs(self.delta_r) > 0 and abs(target_r) > abs(self.delta_r) and same_dir_r:
                    self.delta_r = target_r
                if abs(self.delta_c) > 0 and abs(target_c) > abs(self.delta_c) and same_dir_c:
                    self.delta_c = target_c

                # Multimodal contradiction detection:
                # If this model is already calibrated with high confidence, and the new
                # observation consistently deviates on a different axis/non-multiple, do not corrupt the established EMA.
                # Instead, fork into an alternative conditional branch.
                is_quantum_multiple = (
                    same_dir_r
                    and self.delta_r != 0
                    and target_r != 0
                    and (target_r % self.delta_r == 0 or self.delta_r % target_r == 0)
                ) or (
                    same_dir_c
                    and self.delta_c != 0
                    and target_c != 0
                    and (target_c % self.delta_c == 0 or self.delta_c % target_c == 0)
                )
                is_same_axis_direction = (same_dir_r and target_c == 0 and self.delta_c == 0) or (
                    same_dir_c and target_r == 0 and self.delta_r == 0
                )
                if (
                    not (is_quantum_multiple or is_same_axis_direction)
                    and self.confidence >= 0.7
                    and self.probes_tested >= 2
                ):
                    dist_dev = abs(target_r - self.delta_r) + abs(target_c - self.delta_c)
                    if dist_dev >= 1:
                        cond_key = f"cond_{target_r}_{target_c}"
                        if cond_key not in self.conditional_branches:
                            # Preserve current baseline under 'base' if not yet saved
                            if "base" not in self.conditional_branches:
                                self.conditional_branches["base"] = ActionDynamicsModel(
                                    action_id=self.action_id,
                                    delta_r=self.delta_r,
                                    delta_c=self.delta_c,
                                    confidence=self.confidence,
                                    probes_tested=self.probes_tested,
                                )
                            self.conditional_branches[cond_key] = ActionDynamicsModel(
                                action_id=self.action_id,
                                delta_r=target_r,
                                delta_c=target_c,
                                confidence=0.7,
                                probes_tested=1,
                                displacement=(float(target_r), float(target_c)),
                            )
                        else:
                            self.conditional_branches[cond_key].update_from_trial(
                                observed_delta, success=True, learning_rate=learning_rate
                            )
                        return

                if self.probes_tested <= 1:
                    new_dr = float(target_r)
                    new_dc = float(target_c)
                else:
                    new_dr = (1.0 - learning_rate) * self.delta_r + learning_rate * float(target_r)
                    new_dc = (1.0 - learning_rate) * self.delta_c + learning_rate * float(target_c)
                self.delta_r = int(round(new_dr))
                self.delta_c = int(round(new_dc))
                if abs(self.delta_c) >= 2 * abs(self.delta_r):
                    self.delta_r = 0
                elif abs(self.delta_r) >= 2 * abs(self.delta_c):
                    self.delta_c = 0
                self.displacement = (float(self.delta_r), float(self.delta_c))
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.confidence = max(0.0, self.confidence - 0.15)

    def describe(self) -> str:
        base = f"Action({self.action_id}) -> (Δr={self.delta_r:+d}, Δc={self.delta_c:+d}) [conf={self.confidence:.2f}]"
        if self.conditional_branches:
            branches = ", ".join(
                f"{k}: (Δr={m.delta_r:+d}, Δc={m.delta_c:+d})"
                for k, m in self.conditional_branches.items()
            )
            return f"{base} | Branches: {{{branches}}}"
        return base

    def to_causal_predicate(self) -> CausalPredicate:
        """Lift calibrated motor dynamics into Pearl's formal do(a) causal predicate."""
        return CausalPredicate(
            variable="action_displacement",
            operator="==",
            value=(self.delta_r, self.delta_c),
            target_entity=str(self.action_id),
        )

    def to_affordance(self) -> ActionAffordance:
        """Convert calibrated action dynamics into a declarative ActionAffordance."""
        is_disp = self.is_displacement_action()
        return ActionAffordance(
            action_id=self.action_id,
            name=f"action_{self.action_id}",
            is_displacement=is_disp,
            delta=(self.delta_r, self.delta_c) if is_disp else None,
            confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "delta_r": self.delta_r,
            "delta_c": self.delta_c,
            "confidence": self.confidence,
            "probes_tested": self.probes_tested,
            "displacement": list(self.displacement),
            "metadata": dict(self.metadata),
            "conditional_branches": {k: v.to_dict() for k, v in self.conditional_branches.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionDynamicsModel:
        branches = {}
        for k, bd in data.get("conditional_branches", {}).items():
            try:
                branches[k] = cls.from_dict(bd)
            except Exception:
                pass
        return cls(
            action_id=data.get("action_id", 0),
            delta_r=data.get("delta_r", 0),
            delta_c=data.get("delta_c", 0),
            confidence=data.get("confidence", 0.5),
            probes_tested=data.get("probes_tested", 0),
            displacement=tuple(data.get("displacement", ())),
            metadata=data.get("metadata", {}),
            conditional_branches=branches,
        )


class ConditionedActionDynamics:
    """Mode/state-conditioned container for ActionDynamicsModels.

    Maintains standard dictionary interface keyed by action_id under the active condition,
    while storing and retrieving condition-isolated models. Provides fallback to implicit
    default condition for environments without discrete state modes.
    """

    _IMPLICIT_CONDITION = "__implicit__"

    def __init__(self, active_condition: str | None = None) -> None:
        self._models: dict[tuple[int | str, str], ActionDynamicsModel] = {}
        self.active_condition: str | None = active_condition

    def set_active_condition(self, condition: str | None) -> None:
        """Set current active condition for unconditioned dictionary access."""
        self.active_condition = condition

    def set(
        self,
        action: int | str,
        condition: str | None,
        model: ActionDynamicsModel,
    ) -> None:
        """Register an ActionDynamicsModel for a specific action and condition."""
        cond_key = condition or self._IMPLICIT_CONDITION
        self._models[(action, cond_key)] = model

    def get(
        self,
        action: int | str,
        condition: str | None = None,
        default: Any = None,
    ) -> ActionDynamicsModel | None:
        """Retrieve the dynamics model for an action under a given condition or active condition."""
        cond = condition if condition is not None else self.active_condition
        cond_key = cond or self._IMPLICIT_CONDITION

        key = (action, cond_key)
        if key in self._models:
            return self._models[key]
        implicit_key = (action, self._IMPLICIT_CONDITION)
        if implicit_key in self._models:
            return self._models[implicit_key]
        return default

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all registered dynamics models into declarative causal predicates."""
        predicates: list[CausalPredicate] = []
        for model in self._models.values():
            predicates.append(model.to_causal_predicate())
            for branch in model.conditional_branches.values():
                predicates.append(branch.to_causal_predicate())
        return predicates

    def export_affordances(self) -> list[ActionAffordance]:
        """Export all registered dynamics models into declarative ActionAffordances."""
        return [model.to_affordance() for model in self._models.values()]

    def __getitem__(self, key: int | str | tuple[int | str, str | None]) -> ActionDynamicsModel:
        if isinstance(key, tuple):
            action, cond = key
        else:
            action, cond = key, self.active_condition
        m = self.get(action, cond)
        if m is None:
            raise KeyError(f"No dynamics model for action {action} under condition {cond}")
        return m

    def __setitem__(
        self, key: int | str | tuple[int | str, str | None], model: ActionDynamicsModel
    ) -> None:
        if isinstance(key, tuple):
            action, cond = key
        else:
            action, cond = key, self.active_condition
        self.set(action, cond, model)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            action, cond = key
            return (action, cond or self._IMPLICIT_CONDITION) in self._models
        elif isinstance(key, (int, str)):
            return self.get(key, self.active_condition) is not None
        return False

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def items(self) -> Any:
        cond = self.active_condition or self._IMPLICIT_CONDITION
        result: dict[int | str, ActionDynamicsModel] = {}
        for (a, c), model in self._models.items():
            if c == self._IMPLICIT_CONDITION:
                result[a] = model
        if cond != self._IMPLICIT_CONDITION:
            for (a, c), model in self._models.items():
                if c == cond:
                    result[a] = model
        return result.items()

    def values(self) -> list[ActionDynamicsModel]:
        return [v for _, v in self.items()]

    def keys(self) -> list[int | str]:
        return [k for k, _ in self.items()]

    def clear(self) -> None:
        self._models.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for AgentState / snapshot dictionary."""
        serialized = {}
        for (a, c), m in self._models.items():
            k = f"{a}@@{c}"
            serialized[k] = m.to_dict()
        return {
            "active_condition": self.active_condition,
            "models": serialized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionedActionDynamics:
        """Deserialize from dictionary."""
        dyn = cls(active_condition=data.get("active_condition"))
        for k_str, md in data.get("models", {}).items():
            if "@@" in k_str:
                a_str, c = k_str.split("@@", 1)
            else:
                a_str, c = k_str, cls._IMPLICIT_CONDITION
            try:
                act = int(a_str)
            except ValueError:
                act = a_str
            dyn.set(
                act, c if c != cls._IMPLICIT_CONDITION else None, ActionDynamicsModel.from_dict(md)
            )
        return dyn


@dataclass
class StateMutationModel:
    """Discrete causal rule mapping an environmental trigger to an observable state mutation."""

    trigger_type: TriggerType | str
    trigger_pos: tuple[int, ...] | None = None
    trigger_feature: Any = None  # Generic feature identifier (visual, tactile, token, color)
    mutation_type: MutationType | str = MutationType.MUTATION
    prior_value: Any = None
    posterior_value: Any = None
    confidence: float = 0.5
    occurrences: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        trigger_type: TriggerType | str,
        trigger_pos: tuple[int, ...] | None = None,
        trigger_feature: Any = None,
        mutation_type: MutationType | str = MutationType.MUTATION,
        prior_value: Any = None,
        posterior_value: Any = None,
        confidence: float = 0.5,
        occurrences: int = 1,
        # Backward-compatibility kwargs
        trigger_color: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.trigger_type = trigger_type
        self.trigger_pos = tuple(trigger_pos) if trigger_pos is not None else None
        self.trigger_feature = trigger_feature if trigger_feature is not None else trigger_color
        self.mutation_type = mutation_type
        self.prior_value = prior_value
        self.posterior_value = posterior_value
        self.confidence = confidence
        self.occurrences = occurrences
        self.metadata = dict(kwargs)

    @property
    def trigger_color(self) -> Any:
        return self.trigger_feature

    @trigger_color.setter
    def trigger_color(self, val: Any) -> None:
        self.trigger_feature = val

    def record_observation(self, observed_posterior: Any, learning_rate: float = 0.2) -> None:
        """Update causal rule confidence and state through trial-and-error observation."""
        self.occurrences += 1
        if observed_posterior == self.posterior_value:
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.posterior_value = observed_posterior
            self.confidence = max(0.2, self.confidence - learning_rate)

    def to_causal_predicate(self) -> CausalPredicate:
        """Lift state mutation rule into Pearl's formal causal predicate."""
        return CausalPredicate(
            variable="state_mutation",
            operator="==",
            value=self.posterior_value,
            relation=str(self.mutation_type),
            target_entity=str(self.trigger_pos or self.trigger_feature),
        )

    def to_transaction_operations(self, node_id: str | None = None) -> list[TransactionOperation]:
        """Propose HCIR state mutation transactions for atomic execution by TransactionManager."""
        target_node = node_id or (
            f"pos_{self.trigger_pos[0]}_{self.trigger_pos[1]}"
            if self.trigger_pos and len(self.trigger_pos) >= 2
            else "environment"
        )
        return [
            TransactionOperation(
                op=TransactionOp.MODIFY_NODE,
                node_id=target_node,
                changes={
                    "mutation_type": str(self.mutation_type),
                    "posterior_value": self.posterior_value,
                    "confidence": self.confidence,
                },
            )
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_type": str(self.trigger_type),
            "trigger_pos": list(self.trigger_pos) if self.trigger_pos else None,
            "trigger_feature": self.trigger_feature,
            "mutation_type": str(self.mutation_type),
            "prior_value": self.prior_value,
            "posterior_value": self.posterior_value,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateMutationModel:
        return cls(
            trigger_type=data.get("trigger_type", TriggerType.CONTACT),
            trigger_pos=tuple(data["trigger_pos"]) if data.get("trigger_pos") else None,
            trigger_feature=data.get("trigger_feature"),
            mutation_type=data.get("mutation_type", MutationType.MUTATION),
            prior_value=data.get("prior_value"),
            posterior_value=data.get("posterior_value"),
            confidence=data.get("confidence", 0.5),
            occurrences=data.get("occurrences", 1),
            **data.get("metadata", {}),
        )
