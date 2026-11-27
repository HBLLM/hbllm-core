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
from collections.abc import Collection, Mapping, Sequence
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
    continuous_deltas: list[tuple[float, float]] = field(default_factory=list)
    delta_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    conditional_branches: dict[str, ActionDynamicsModel] = field(default_factory=dict)

    @property
    def slip_rate(self) -> float:
        """Empirical fraction of displacement trials deviating from the modal direction."""
        tot = sum(self.delta_counts.values())
        if tot < 2 or not self.delta_counts:
            return 0.0
        best = max(self.delta_counts.values())
        return max(0.0, 1.0 - (best / float(tot)))

    def __post_init__(self) -> None:
        if not self.displacement:
            self.displacement = (float(self.delta_r), float(self.delta_c))
        elif self.delta_r == 0 and self.delta_c == 0 and len(self.displacement) >= 2:
            self.delta_r = int(round(self.displacement[0]))
            self.delta_c = int(round(self.displacement[1]))
        if not self.delta_counts and (self.delta_r != 0 or self.delta_c != 0):
            self.delta_counts[(self.delta_r, self.delta_c)] = max(1, int(self.probes_tested))

    def record_continuous_delta(self, dx: float, dy: float) -> None:
        """Record an empirically observed continuous displacement vector (dx, dy)."""
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            self.continuous_deltas.append((float(dx), float(dy)))
            if len(self.continuous_deltas) > 64:
                self.continuous_deltas = self.continuous_deltas[-64:]

    def get_continuous_velocity(self) -> tuple[float, float]:
        """Compute the mean continuous velocity vector (vx, vy) from observed trials."""
        if self.continuous_deltas:
            vx = sum(d[0] for d in self.continuous_deltas) / len(self.continuous_deltas)
            vy = sum(d[1] for d in self.continuous_deltas) / len(self.continuous_deltas)
            return (float(vx), float(vy))
        if len(self.displacement) >= 2:
            return (float(self.displacement[0]), float(self.displacement[1]))
        return (float(self.delta_r), float(self.delta_c))

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
        if dr != 0 or dc != 0:
            return True
        vx, vy = self.get_continuous_velocity()
        return (vx * vx + vy * vy) > 1e-6

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
                # Only refine downward to an exact divisor of the current stride so a truncated step
                # against an obstacle (e.g. +4 vs established +5) never corrupts an established stride.
                if (
                    0 < abs(target_r) < abs(self.delta_r)
                    and self.delta_r != 0
                    and self.delta_r % target_r == 0
                ):
                    self.delta_r = target_r
                if (
                    0 < abs(target_c) < abs(self.delta_c)
                    and self.delta_c != 0
                    and self.delta_c % target_c == 0
                ):
                    self.delta_c = target_c
                # If an action previously observed a partial/truncated step (e.g. 3) and now demonstrates
                # a larger integer multiple stride along the same axis (e.g. 6), adopt the unconstrained stride.
                if (
                    abs(self.delta_r) > 0
                    and abs(target_r) > abs(self.delta_r)
                    and target_r % self.delta_r == 0
                ):
                    self.delta_r = target_r
                if (
                    abs(self.delta_c) > 0
                    and abs(target_c) > abs(self.delta_c)
                    and target_c % self.delta_c == 0
                ):
                    self.delta_c = target_c

                # Multimodal contradiction detection:
                # If this model is already calibrated with high confidence, and the new
                # observation consistently deviates on a different axis/non-multiple, do not corrupt the established EMA.
                # Instead, fork into an alternative conditional branch.
                is_quantum_multiple = (
                    self.delta_r != 0
                    and target_r != 0
                    and (target_r % self.delta_r == 0 or self.delta_r % target_r == 0)
                ) or (
                    self.delta_c != 0
                    and target_c != 0
                    and (target_c % self.delta_c == 0 or self.delta_c % target_c == 0)
                )
                has_nonzero_obs = target_r != 0 or target_c != 0
                is_unit_step = has_nonzero_obs and max(abs(target_r), abs(target_c)) == 1
                if has_nonzero_obs:
                    if (
                        not self.delta_counts
                        and self.probes_tested > 1
                        and (self.delta_r != 0 or self.delta_c != 0)
                    ):
                        self.delta_counts[(self.delta_r, self.delta_c)] = self.probes_tested - 1
                    self.delta_counts[(target_r, target_c)] = (
                        self.delta_counts.get((target_r, target_c), 0) + 1
                    )

                if not is_quantum_multiple and self.confidence >= 0.7 and self.probes_tested >= 2:
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
                        if self.delta_counts:
                            cur_vec = (self.delta_r, self.delta_c)
                            modal_d, _ = max(
                                self.delta_counts.items(),
                                key=lambda kv: (
                                    kv[1],
                                    1 if kv[0] == cur_vec else 0,
                                    abs(kv[0][0]) + abs(kv[0][1]),
                                ),
                            )
                            self.delta_r, self.delta_c = modal_d[0], modal_d[1]
                            self.displacement = (float(self.delta_r), float(self.delta_c))
                        return

                if self.delta_counts:
                    cur_vec = (self.delta_r, self.delta_c)
                    modal_d, modal_cnt = max(
                        self.delta_counts.items(),
                        key=lambda kv: (
                            kv[1],
                            1 if kv[0] == cur_vec else 0,
                            abs(kv[0][0]) + abs(kv[0][1]),
                        ),
                    )
                    self.delta_r, self.delta_c = modal_d[0], modal_d[1]
                    if is_unit_step:
                        self.displacement = (float(self.delta_r), float(self.delta_c))
                        tot_obs = sum(self.delta_counts.values())
                        if modal_cnt >= 2 and modal_cnt / max(1, tot_obs) >= 0.5:
                            self.confidence = min(1.0, max(self.confidence, 0.75) + 0.05)
                        else:
                            self.confidence = min(1.0, self.confidence + 0.1)
                        return
                elif has_nonzero_obs and self.delta_r == 0 and self.delta_c == 0:
                    # Transition from ungrounded/obstructed prior to verified empirical displacement
                    self.delta_r = target_r
                    self.delta_c = target_c
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
            except Exception as exc:
                logger.debug("Malformed conditional branch %r skipped: %s", k, exc)
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
        meta_kw = kwargs.pop("metadata", None)
        self.metadata: dict[str, Any] = dict(meta_kw) if isinstance(meta_kw, dict) else {}
        self.metadata.update(kwargs)

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


def disambiguate_stochastic_motor_basis(
    action_dynamics: dict[Any, ActionDynamicsModel],
    action_affordances: dict[Any, Any] | None = None,
    primary_action: Any | None = None,
) -> list[Any]:
    """Resolve motor vector collisions caused by early stochastic wind slips.

    When an uncertain/noisy action shares the same displacement vector as a clean,
    low-slip action, reassign the noisy action to an unclaimed unit vector supported by
    its empirical ``delta_counts`` near the mode. When ``action_affordances`` is passed,
    synchronizes ``is_displacement`` and ``delta`` for reassigned actions and ``primary_action``.
    """
    if not action_dynamics:
        return []
    by_vec: dict[tuple[int, int], list[tuple[Any, ActionDynamicsModel]]] = {}
    for aid, dyn in action_dynamics.items():
        if not dyn.is_displacement_action():
            continue
        vec = dyn.get_displacement()
        if vec != (0, 0):
            by_vec.setdefault(vec, []).append((aid, dyn))

    clean_claimed = {
        vec for vec, owners in by_vec.items() if any(d.slip_rate <= 0.05 for _, d in owners)
    }
    changed: list[Any] = []
    for vec, owners in by_vec.items():
        if len(owners) < 2:
            continue
        owners_sorted = sorted(
            owners,
            key=lambda item: (
                -item[1].delta_counts.get(vec, 0),
                item[1].slip_rate,
                -item[1].confidence,
            ),
        )
        for aid, dyn in owners_sorted[1:]:
            if not dyn.delta_counts or dyn.slip_rate <= 0.05:
                continue
            max_cnt = max(dyn.delta_counts.values())
            candidates = [
                (cnt, cand_v)
                for cand_v, cnt in dyn.delta_counts.items()
                if cand_v != vec
                and cand_v not in clean_claimed
                and (abs(cand_v[0]) + abs(cand_v[1])) == 1
                and cnt >= max_cnt - 1
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_v = candidates[0][1]
            dyn.delta_r, dyn.delta_c = int(best_v[0]), int(best_v[1])
            dyn.displacement = (float(dyn.delta_r), float(dyn.delta_c))
            clean_claimed.add(best_v)
            changed.append(aid)
            if action_affordances is not None and aid in action_affordances:
                r_dr, r_dc = dyn.get_displacement()
                action_affordances[aid].is_displacement = dyn.is_displacement_action()
                action_affordances[aid].delta = (r_dr, r_dc)
    if (
        primary_action is not None
        and action_affordances is not None
        and primary_action in action_affordances
        and primary_action in action_dynamics
    ):
        p_dyn = action_dynamics[primary_action]
        p_dr, p_dc = p_dyn.get_displacement()
        action_affordances[primary_action].is_displacement = p_dyn.is_displacement_action()
        action_affordances[primary_action].delta = (p_dr, p_dc)
    return changed


def resolve_calibrated_transition_deltas(
    action_deltas: Mapping[Any, tuple[int, int]],
    *,
    available_actions: Sequence[Any] | None = None,
    has_displacement: bool = True,
    has_declared_affordances: bool = False,
    inhibited_actions: Mapping[Any, int] | Collection[Any] | None = None,
    action_dynamics: Mapping[Any, ActionDynamicsModel] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Derive deduplicated non-zero operator displacement deltas from calibrated dynamics."""
    if not has_displacement and (available_actions is not None or has_declared_affordances):
        return ()

    acts = list(available_actions) if available_actions is not None else list(action_deltas.keys())
    if inhibited_actions:
        if isinstance(inhibited_actions, Mapping):
            uninhibited = [a for a in acts if inhibited_actions.get(a, 0) <= 0]
        else:
            uninhibited = [a for a in acts if a not in inhibited_actions]
        if uninhibited:
            acts = uninhibited

    # Prefer empirically verified dynamics first when present, else declared action_deltas.
    observed_deltas: list[tuple[int, int]] = []
    fallback_deltas: list[tuple[int, int]] = []
    seen_obs: set[tuple[int, int]] = set()
    seen_fb: set[tuple[int, int]] = set()
    for act in acts:
        d = action_deltas.get(act)
        if d is None or len(d) < 2:
            continue
        pair = (int(d[0]), int(d[1]))
        if pair == (0, 0):
            continue
        dyn = action_dynamics.get(act) if action_dynamics is not None else None
        if dyn is not None and dyn.probes_tested > 0 and dyn.is_displacement_action():
            if pair not in seen_obs:
                seen_obs.add(pair)
                observed_deltas.append(pair)
        elif pair not in seen_fb:
            seen_fb.add(pair)
            fallback_deltas.append(pair)

    combined = observed_deltas if observed_deltas else fallback_deltas
    if combined:
        # Preserve any fallback axes only when no observed axes were found, or add unprobed axes
        if observed_deltas and fallback_deltas:
            for pair in fallback_deltas:
                if pair not in seen_obs:
                    seen_obs.add(pair)
                    combined.append(pair)
        return tuple(combined)
    return (
        ((-1, 0), (1, 0), (0, -1), (0, 1))
        if (has_displacement or not has_declared_affordances)
        else ()
    )


def resolve_effector_coupling_offsets(
    action_deltas: Mapping[Any, tuple[int, int]],
    *,
    available_actions: Sequence[Any] | None = None,
    has_displacement: bool = True,
    has_spatial_effector: bool = False,
    has_discrete_local_interact: bool = False,
    has_declared_affordances: bool = False,
    inhibited_actions: Mapping[Any, int] | Collection[Any] | None = None,
    action_dynamics: Mapping[Any, ActionDynamicsModel] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Derive predecessor interaction offsets {target - delta_a} from calibrated operators."""
    if has_spatial_effector and not has_displacement:
        return ()

    disp_deltas = resolve_calibrated_transition_deltas(
        action_deltas,
        available_actions=available_actions,
        has_displacement=has_displacement,
        has_declared_affordances=has_declared_affordances,
        inhibited_actions=inhibited_actions,
        action_dynamics=action_dynamics,
    )
    offsets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for dr, dc in disp_deltas:
        pred = (-int(dr), -int(dc))
        if pred != (0, 0) and pred not in seen:
            seen.add(pred)
            offsets.append(pred)
    if has_discrete_local_interact:
        for card in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if card not in seen:
                seen.add(card)
                offsets.append(card)
    return tuple(offsets)
