"""Causal Network Domain for Scientific Evaluation of Epistemic & Counterfactual Discovery.

Implements a non-spatial, complex causal graph domain (Gene Regulatory / Medical Diagnostic Networks)
demonstrating that HBLLM's counterfactual simulation, analogical schema mapping, and active probing
generalize rigorously beyond blocks-world physics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class CausalVariable:
    """A node in the causal graph with structural equations."""

    name: str
    parents: list[str] = field(default_factory=list)
    is_confounded: bool = False
    base_activation_prob: float = 0.20


@dataclass
class CausalInterventionScenario:
    """A specific causal evaluation problem with observational vs interventional truth."""

    scenario_id: str
    target_gene: str
    intervention_var: str
    intervention_val: int  # 0 or 1
    observational_correlation: float  # High spurious correlation
    true_interventional_effect: float  # Real causal effect under do(X = x)
    is_true_causal_cause: bool
    distractor_surface_name: str = ""


class CausalNetworkEnvironment:
    """Generates synthetic causal networks and ground-truth interventional responses."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def generate_gene_network_scenarios(
        self, n_scenarios: int = 5
    ) -> list[CausalInterventionScenario]:
        """Generate diverse causal scenarios with structural confounders."""
        scenarios: list[CausalInterventionScenario] = [
            CausalInterventionScenario(
                scenario_id="GENE_CASCADE_A",
                target_gene="PROTEIN_SYNTHESIS",
                intervention_var="KINASE_AKT",
                intervention_val=1,
                observational_correlation=0.92,
                true_interventional_effect=0.88,
                is_true_causal_cause=True,
                distractor_surface_name="RED_MARKER",
            ),
            CausalInterventionScenario(
                scenario_id="CONFOUNDED_CORRELATION_B",
                target_gene="CELL_GROWTH",
                intervention_var="SURFACE_RECEPTOR_X",
                intervention_val=1,
                observational_correlation=0.89,
                true_interventional_effect=0.05,  # Spurious correlation from hidden confounder
                is_true_causal_cause=False,
                distractor_surface_name="BLUE_MARKER",
            ),
            CausalInterventionScenario(
                scenario_id="INHIBITION_PATHWAY_C",
                target_gene="TUMOR_SUPPRESSION",
                intervention_var="ENZYME_P53",
                intervention_val=1,
                observational_correlation=0.85,
                true_interventional_effect=0.90,
                is_true_causal_cause=True,
                distractor_surface_name="GREEN_MARKER",
            ),
            CausalInterventionScenario(
                scenario_id="COLLIDER_BIAS_D",
                target_gene="METABOLIC_RATE",
                intervention_var="CYTOKINE_IL6",
                intervention_val=1,
                observational_correlation=0.78,
                true_interventional_effect=0.02,  # Collider bias
                is_true_causal_cause=False,
                distractor_surface_name="YELLOW_MARKER",
            ),
            CausalInterventionScenario(
                scenario_id="FEEDFORWARD_LOOP_E",
                target_gene="TRANSCRIPTION_FACTOR_Y",
                intervention_var="SIGNAL_LIGAND_Z",
                intervention_val=1,
                observational_correlation=0.95,
                true_interventional_effect=0.92,
                is_true_causal_cause=True,
                distractor_surface_name="PURPLE_MARKER",
            ),
        ]
        return self.rng.sample(scenarios, min(n_scenarios, len(scenarios)))
