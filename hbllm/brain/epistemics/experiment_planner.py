"""Experiment Planner — designs experiments optimized for info gain / cost.

Wraps the existing ``ExperimentEngine`` to add discovery-economics-aware
experiment design and ranking.

Design principle::

    Priority = Expected Information Gain × KnowledgeValue.impact
               ─────────────────────────────────────────────────
               Expected Cost (CPU + time + risk)

    Sometimes the optimal action is: don't investigate.

Usage::

    planner = ExperimentPlanner(graph=graph, llm=llm)
    design = await planner.design_discriminative_experiment(
        hypothesis_ids=["h1", "h2"],
        budget=InvestigationBudget(max_llm_calls=5),
    )
"""

from __future__ import annotations

import logging
import math
from typing import Any

from hbllm.brain.epistemics.interfaces import (
    ExperimentDesign,
    InvestigationBudget,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    HypothesisNode,
)

logger = logging.getLogger(__name__)

#: Default investigation budget when none is provided.
_DEFAULT_BUDGET = InvestigationBudget()


class ExperimentPlanner:
    """Designs experiments that maximize information gain per unit cost.

    Implements the ``IExperimentDesigner`` protocol.

    The planner integrates discovery economics:
    - Estimates expected information gain (entropy reduction)
    - Weighs against investigation cost
    - Ensures the experiment actually *discriminates* between hypotheses

    The planner is domain-neutral — it designs based on hypothesis
    structure, prediction conflicts, and resource constraints.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        llm: Any | None = None,
        counterfactual: Any | None = None,
    ) -> None:
        self._graph = graph
        self._llm = llm
        self._counterfactual = counterfactual  # Optional CounterfactualReasoner

    async def design_discriminative_experiment(
        self,
        hypothesis_ids: list[str],
        budget: InvestigationBudget | None = None,
        max_reality_level: str = "simulation",
    ) -> ExperimentDesign:
        """Design an experiment to distinguish between hypotheses.

        The key question is not "Can this confirm my hypothesis?"
        but "Which experiment most reduces total uncertainty?"

        Args:
            hypothesis_ids: The rival hypotheses to discriminate.
            budget: Resource constraints.
            max_reality_level: Maximum experiment reality level.

        Returns:
            An ExperimentDesign with estimated info gain and cost.
        """
        budget = budget or _DEFAULT_BUDGET

        hypotheses = self._load_hypotheses(hypothesis_ids)
        if len(hypotheses) < 1:
            return ExperimentDesign(
                hypothesis_ids=hypothesis_ids,
                reasoning="No valid hypotheses found for experiment design",
            )

        if self._llm is not None:
            design = await self._llm_design(hypotheses, budget, max_reality_level)
        else:
            design = self._template_design(hypotheses, budget, max_reality_level)

        # Estimate information gain
        design.expected_information_gain = self._estimate_info_gain(
            hypotheses,
            design.discriminating_power,
        )

        return design

    async def rank_by_information_gain(
        self,
        designs: list[ExperimentDesign],
    ) -> list[ExperimentDesign]:
        """Rank experiments by (information gain × impact) / cost.

        Integrates discovery economics — high-cost experiments with
        marginal info gain are ranked below cheaper alternatives.
        """

        def score(d: ExperimentDesign) -> float:
            info_gain = d.expected_information_gain or d.discriminating_power
            cost = max(0.01, d.estimated_cost)  # Prevent division by zero
            return info_gain / cost

        return sorted(designs, key=score, reverse=True)

    async def estimate_information_gain(
        self,
        design: ExperimentDesign,
    ) -> float:
        """Estimate the expected entropy reduction from an experiment."""
        hypotheses = self._load_hypotheses(design.hypothesis_ids)
        return self._estimate_info_gain(
            hypotheses,
            design.discriminating_power,
        )

    # ── Internal Methods ───────────────────────────────────────────────

    def _load_hypotheses(
        self,
        hypothesis_ids: list[str],
    ) -> list[HypothesisNode]:
        """Load hypothesis nodes from graph."""
        hypotheses: list[HypothesisNode] = []
        for hid in hypothesis_ids:
            node = self._graph.get_node(hid)
            if isinstance(node, HypothesisNode):
                hypotheses.append(node)
        return hypotheses

    def _estimate_info_gain(
        self,
        hypotheses: list[HypothesisNode],
        discriminating_power: float,
    ) -> float:
        """Estimate expected information gain (Shannon entropy reduction).

        When hypotheses have similar confidence, the experiment
        can maximally reduce entropy.  When one hypothesis is
        already dominant, the experiment adds less information.
        """
        if len(hypotheses) < 2:
            return discriminating_power * 0.5

        # Current entropy: how uncertain are we between hypotheses?
        confidences = [h.uncertainty.confidence for h in hypotheses]
        total = sum(confidences) or 1.0
        probs = [c / total for c in confidences]

        # Shannon entropy
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
        max_entropy = math.log2(len(hypotheses))

        if max_entropy == 0:
            return 0.0

        # Info gain = current uncertainty × discriminating power
        normalized_entropy = entropy / max_entropy
        return normalized_entropy * discriminating_power

    async def _llm_design(
        self,
        hypotheses: list[HypothesisNode],
        budget: InvestigationBudget,
        max_reality_level: str,
    ) -> ExperimentDesign:
        """Use LLM to design a discriminative experiment."""
        hyp_descriptions = "\n".join(
            f"H{i + 1} (confidence={h.uncertainty.confidence:.2f}): {h.claim}"
            for i, h in enumerate(hypotheses)
        )

        prompt = (
            f"Design a single experiment that discriminates between these hypotheses:\n"
            f"{hyp_descriptions}\n\n"
            f"Constraints:\n"
            f"- Maximum reality level: {max_reality_level}\n"
            f"- Budget: max {budget.max_experiments} experiments, "
            f"{budget.max_llm_calls} LLM calls\n"
            f"- Time: {budget.max_wall_time_seconds}s maximum\n\n"
            f"Describe:\n"
            f"1. DESIGN: The experiment procedure\n"
            f"2. VARIABLES: Independent and dependent variables\n"
            f"3. EXPECTED_OUTCOMES: What each hypothesis predicts\n"
            f"4. DISCRIMINATING_POWER: 0.0-1.0 (how well this distinguishes)\n"
            f"5. COST: Estimated cost 0.0-1.0\n"
        )

        try:
            response = await self._llm.generate(prompt)
            text = response if isinstance(response, str) else str(response)
            return self._parse_llm_design(text, hypotheses, max_reality_level)
        except Exception as exc:
            logger.warning("LLM experiment design failed: %s", exc)
            return self._template_design(hypotheses, budget, max_reality_level)

    def _parse_llm_design(
        self,
        text: str,
        hypotheses: list[HypothesisNode],
        max_reality_level: str,
    ) -> ExperimentDesign:
        """Parse LLM experiment design response."""
        design = ""
        disc_power = 0.5
        cost = 0.3
        expected_outcomes: dict[str, str] = {}

        for line in text.split("\n"):
            line_upper = line.strip().upper()
            if line_upper.startswith("DESIGN:"):
                design = line.strip()[7:].strip()
            elif line_upper.startswith("DISCRIMINATING_POWER:"):
                try:
                    disc_power = float(line.strip().split(":")[1].strip())
                    disc_power = min(1.0, max(0.0, disc_power))
                except (ValueError, IndexError):
                    pass
            elif line_upper.startswith("COST:"):
                try:
                    cost = float(line.strip().split(":")[1].strip())
                    cost = min(1.0, max(0.0, cost))
                except (ValueError, IndexError):
                    pass
            elif line_upper.startswith("EXPECTED_OUTCOMES:"):
                for h in hypotheses:
                    expected_outcomes[h.id] = f"See LLM design: {h.claim[:30]}"

        return ExperimentDesign(
            hypothesis_ids=[h.id for h in hypotheses],
            design=design or text[:200],
            discriminating_power=disc_power,
            estimated_cost=cost,
            expected_outcomes=expected_outcomes,
            reality_level=max_reality_level,
            reasoning="LLM-designed discriminative experiment",
        )

    def _template_design(
        self,
        hypotheses: list[HypothesisNode],
        budget: InvestigationBudget,
        max_reality_level: str,
    ) -> ExperimentDesign:
        """Design a basic discriminative experiment template."""
        if len(hypotheses) == 1:
            design = f"Test hypothesis by observing predicted outcomes: {hypotheses[0].claim[:100]}"
            disc_power = 0.4
        else:
            claims = " vs ".join(h.claim[:50] for h in hypotheses[:3])
            design = (
                f"Comparative test between rival hypotheses: {claims}. "
                f"Identify observable predictions that differ between hypotheses."
            )
            disc_power = 0.5

        expected_outcomes = {h.id: f"Outcome consistent with: {h.claim[:50]}" for h in hypotheses}

        return ExperimentDesign(
            hypothesis_ids=[h.id for h in hypotheses],
            design=design,
            discriminating_power=disc_power,
            estimated_cost=0.3,
            expected_outcomes=expected_outcomes,
            reality_level=max_reality_level,
            reasoning="Template-designed discriminative experiment",
        )

    # ── Counterfactual-Enhanced Design ───────────────────────────────

    async def design_counterfactual_experiment(
        self,
        belief_id: str,
        budget: InvestigationBudget | None = None,
    ) -> ExperimentDesign:
        """Design an experiment targeting the most impactful evidence gap.

        Uses CounterfactualReasoner.sensitivity_analysis() to find which
        evidence has the highest impact on the belief.  Then designs an
        experiment that targets the weakest link.

        Args:
            belief_id: The belief to strengthen or challenge.
            budget: Resource constraints.

        Returns:
            An ExperimentDesign targeting the highest-impact evidence.
        """
        budget = budget or _DEFAULT_BUDGET

        if self._counterfactual is None:
            return ExperimentDesign(
                hypothesis_ids=[],
                reasoning="No CounterfactualReasoner available",
            )

        # Find the most impactful evidence
        sensitivity = await self._counterfactual.sensitivity_analysis(belief_id)
        if not sensitivity:
            return ExperimentDesign(
                hypothesis_ids=[],
                reasoning="No evidence found for sensitivity analysis",
            )

        # Target the highest-impact evidence
        top_evidence_id = next(iter(sensitivity))
        top_impact = sensitivity[top_evidence_id]

        # Get belief details
        belief = self._graph.get_node(belief_id)
        belief_claim = getattr(belief, "claim", belief_id) if belief else belief_id

        # Design experiment to strengthen or challenge this evidence
        design = ExperimentDesign(
            hypothesis_ids=[belief_id],
            design=(
                f"Target highest-impact evidence ({top_evidence_id[:20]}, "
                f"impact={top_impact:.3f}) for belief: {str(belief_claim)[:60]}. "
                f"Design a replication or extension experiment for this evidence."
            ),
            discriminating_power=min(1.0, top_impact * 1.5),
            estimated_cost=0.4,
            expected_information_gain=top_impact,
            reality_level="experiment",
            reasoning=(
                f"Counterfactual-guided: this evidence has the highest impact "
                f"({top_impact:.3f}) on the belief. Strengthening or challenging "
                f"it will have the largest effect on our confidence."
            ),
        )

        return design
