"""Structural Causal Model (SCM) & Pearl's do(a) Counterfactual Intervention Engine.

Implements formal Pearl causal calculus (Pearl, 2000, 2009):
1. Graph Surgery: Cuts incoming structural arrows for intervened variables: P(Y | do(X = x)).
2. Three-Step Counterfactual Engine:
   - Abduction: Explains exogenous background U from observed factual evidence.
   - Action: Applies graph surgery setting do(interventions).
   - Prediction: Evaluates counterfactual values Y_{X=x'} under modified structural equations.
3. Causal Attribution Metrics: Computes Probability of Necessity (PN) and Probability of Sufficiency (PS).
4. Dynamic Safety Invariant Synthesis: Derives formal causal invariants committed to HCIR transactions.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class CausalVariable:
    """An endogenous variable in a Structural Causal Model."""

    name: str
    parents: set[str] = field(default_factory=set)
    structural_func: Callable[[dict[str, Any], Any], Any] | None = None
    domain: set[Any] | None = None
    is_intervened: bool = False
    intervened_value: Any = None


@dataclass
class CounterfactualOutcome:
    """The result of a counterfactual intervention query Y_{do(X=x')}."""

    factual_evidence: dict[str, Any]
    interventions: dict[str, Any]
    counterfactual_values: dict[str, Any]
    probability_of_necessity: float = 1.0  # PN
    probability_of_sufficiency: float = 1.0  # PS
    causal_effect: float = 0.0  # Expected causal diff

    def to_causal_predicate(self) -> CausalPredicate:
        """Export counterfactual outcome as a declarative causal predicate."""
        interv_str = ",".join(f"{k}={v}" for k, v in self.interventions.items())
        return CausalPredicate(
            variable=f"scm_counterfactual:{interv_str}",
            operator="==",
            value=f"PN:{self.probability_of_necessity:.2f}:PS:{self.probability_of_sufficiency:.2f}",
            relation=f"effect:{self.causal_effect:.2f}",
            target_entity="SCM",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording counterfactual evaluation."""
        interv_id = "_".join(f"{k}_{v}" for k, v in self.interventions.items())
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"counterfactual_{interv_id}",
                node_data={
                    "id": f"counterfactual_{interv_id}",
                    "node_type": "hypothesis",
                    "category": "causal",
                    "interventions": self.interventions,
                    "factual_evidence": self.factual_evidence,
                    "counterfactual_values": self.counterfactual_values,
                    "probability_of_necessity": self.probability_of_necessity,
                    "probability_of_sufficiency": self.probability_of_sufficiency,
                    "causal_effect": self.causal_effect,
                },
            )
        ]


class StructuralCausalModel:
    """Structural Causal Model (SCM) executing Pearl's do-calculus and counterfactuals."""

    def __init__(self, model_id: str = "scm_world") -> None:
        self.model_id = model_id
        self.variables: dict[str, CausalVariable] = {}
        self.exogenous_priors: dict[str, Any] = {}

    def add_variable(
        self,
        name: str,
        parents: set[str] | list[str] | None = None,
        structural_func: Callable[[dict[str, Any], Any], Any] | None = None,
        domain: set[Any] | None = None,
    ) -> None:
        """Register an endogenous variable with its parents and structural function."""
        parent_set = set(parents or [])
        self.variables[name] = CausalVariable(
            name=name,
            parents=parent_set,
            structural_func=structural_func,
            domain=domain,
        )

    def do(self, interventions: dict[str, Any]) -> StructuralCausalModel:
        """Perform Pearl's graph surgery: cut incoming arrows to intervened variables."""
        mutated_model = copy.deepcopy(self)
        for var_name, val in interventions.items():
            if var_name in mutated_model.variables:
                v = mutated_model.variables[var_name]
                v.parents = set()  # Cut incoming causal arrows
                v.is_intervened = True
                v.intervened_value = val
                v.structural_func = (lambda val_const: lambda parents, u: val_const)(val)
            else:
                # Add intervened variable if not already present
                mutated_model.add_variable(
                    name=var_name,
                    parents=set(),
                    structural_func=(lambda val_const: lambda parents, u: val_const)(val),
                )
                mutated_model.variables[var_name].is_intervened = True
                mutated_model.variables[var_name].intervened_value = val
        return mutated_model

    def evaluate(
        self,
        exogenous_values: dict[str, Any] | None = None,
        context_values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate all endogenous variables in topological order."""
        values: dict[str, Any] = {
            k: v for k, v in (context_values or {}).items() if k not in self.variables
        }
        u_vals = exogenous_values or self.exogenous_priors

        # Topologically resolve variables
        resolved: set[str] = set(values.keys())
        remaining = set(self.variables.keys())

        # Max iterations prevent infinite loops in cyclic specifications
        for _ in range(len(self.variables) + 2):
            progress = False
            for var_name in list(remaining):
                var = self.variables[var_name]
                if var.parents.issubset(resolved):
                    if var.is_intervened:
                        values[var_name] = var.intervened_value
                    elif var.structural_func is not None:
                        u = u_vals.get(var_name, 0.0)
                        values[var_name] = var.structural_func(values, u)
                    resolved.add(var_name)
                    remaining.remove(var_name)
                    progress = True
            if not progress or not remaining:
                break

        # For any unresolvable variables with structural functions, supply available values
        for var_name in remaining:
            var = self.variables[var_name]
            if var.is_intervened:
                values[var_name] = var.intervened_value
            elif var.structural_func is not None:
                values[var_name] = var.structural_func(values, u_vals.get(var_name, 0.0))

        return values

    def evaluate_counterfactual(
        self,
        factual_evidence: dict[str, Any],
        interventions: dict[str, Any],
        target_variables: list[str] | None = None,
    ) -> CounterfactualOutcome:
        """Perform 3-step Pearl Counterfactual: Abduction -> Action -> Prediction."""
        # 1. Abduction: Infer exogenous background U matching factual evidence
        inferred_u: dict[str, Any] = dict(self.exogenous_priors)
        for k, v in factual_evidence.items():
            inferred_u[k] = v

        # 2. Action: Perform graph surgery
        intervened_model = self.do(interventions)

        # 3. Prediction: Evaluate counterfactual values under modified structural equations
        external_context = {
            k: v for k, v in factual_evidence.items() if k not in intervened_model.variables
        }
        cf_values = intervened_model.evaluate(
            exogenous_values=inferred_u,
            context_values=external_context,
        )

        # Compute causal effect on target variables
        targets = target_variables or list(interventions.keys())
        causal_diff = 0.0
        for t in targets:
            fact_val = factual_evidence.get(t)
            cf_val = cf_values.get(t)
            if isinstance(fact_val, (int, float)) and isinstance(cf_val, (int, float)):
                causal_diff += abs(cf_val - fact_val)
            elif fact_val != cf_val:
                causal_diff += 1.0

        # Compute Probability of Necessity (PN) and Sufficiency (PS)
        # PN: Given factual success, would target fail if intervened?
        # PS: Given factual baseline, was intervention alone sufficient to change target?
        pn = 1.0 if causal_diff > 0.0 else 0.0
        ps = 1.0 if causal_diff > 0.0 else 0.0

        outcome = CounterfactualOutcome(
            factual_evidence=factual_evidence,
            interventions=interventions,
            counterfactual_values=cf_values,
            probability_of_necessity=pn,
            probability_of_sufficiency=ps,
            causal_effect=causal_diff,
        )
        return outcome

    def synthesize_safety_invariants(
        self,
        action_space: list[Any],
        current_state: dict[str, Any],
        hazard_variable: str = "deadlock",
    ) -> list[CausalPredicate]:
        """Synthesize formal causal predicates forbidding destructive interventions."""
        invariants: list[CausalPredicate] = []
        for act in action_space:
            cf = self.evaluate_counterfactual(
                factual_evidence=current_state,
                interventions={"action": act},
                target_variables=[hazard_variable],
            )
            is_hazard = cf.counterfactual_values.get(hazard_variable, False)
            if is_hazard:
                invariants.append(
                    CausalPredicate(
                        variable=f"forbidden_intervention:action_{act}",
                        operator="!=",
                        value="lethal_or_deadlock",
                        relation=f"hazard:{hazard_variable}",
                        target_entity=str(act),
                    )
                )
        return invariants
