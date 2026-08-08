"""Tests for ExperimentPlanner — info-gain experiment design."""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.brain.epistemics.counterfactual import CounterfactualReasoner
from hbllm.brain.epistemics.experiment_planner import ExperimentPlanner
from hbllm.brain.epistemics.hypothesis_builder import HypothesisBuilder
from hbllm.brain.epistemics.idea_generator import IdeaGenerator
from hbllm.brain.epistemics.interfaces import ExperimentDesign, InvestigationBudget
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import CognitiveGraph, HypothesisNode


class TestDesignDiscriminativeExperiment:
    """Test discriminative experiment design."""

    @pytest.mark.asyncio
    async def test_design_with_two_hypotheses(
        self, graph: CognitiveGraph, workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Test", "Why X?")
        obj = workspace.add_objective(prog.program_id, "Find")
        q_id = workspace.add_question(prog.program_id, obj, "Why?", importance=0.8)

        gen = IdeaGenerator(graph=graph)
        ideas = await gen.generate_from_unknown(q_id)
        builder = HypothesisBuilder(graph=graph)
        candidates = await builder.validate(ideas)

        hyp_ids: list[str] = []
        for c in candidates[:2]:
            nid = await builder.promote_to_node(c, prog.program_id)
            hyp_ids.append(nid)

        planner = ExperimentPlanner(graph=graph)
        design = await planner.design_discriminative_experiment(hyp_ids)

        assert isinstance(design, ExperimentDesign)
        assert design.discriminating_power > 0
        assert design.expected_information_gain >= 0

    @pytest.mark.asyncio
    async def test_design_empty_hypotheses(self, graph: CognitiveGraph) -> None:
        planner = ExperimentPlanner(graph=graph)
        design = await planner.design_discriminative_experiment([])
        assert "No valid hypotheses" in design.reasoning

    @pytest.mark.asyncio
    async def test_design_single_hypothesis(
        self, graph: CognitiveGraph,
    ) -> None:
        hyp = HypothesisNode(claim="X causes Y")
        graph.upsert_node(hyp)

        planner = ExperimentPlanner(graph=graph)
        design = await planner.design_discriminative_experiment([hyp.id])

        assert isinstance(design, ExperimentDesign)


class TestRankByInformationGain:
    """Test ranking experiments by info gain / cost."""

    @pytest.mark.asyncio
    async def test_ranking_order(self, graph: CognitiveGraph) -> None:
        planner = ExperimentPlanner(graph=graph)
        designs = [
            ExperimentDesign(
                hypothesis_ids=["h1"],
                expected_information_gain=0.8,
                estimated_cost=0.4,
            ),
            ExperimentDesign(
                hypothesis_ids=["h2"],
                expected_information_gain=0.3,
                estimated_cost=0.1,
            ),
            ExperimentDesign(
                hypothesis_ids=["h3"],
                expected_information_gain=0.5,
                estimated_cost=0.9,
            ),
        ]

        ranked = await planner.rank_by_information_gain(designs)

        # h2 (0.3/0.1=3.0) > h1 (0.8/0.4=2.0) > h3 (0.5/0.9=0.56)
        assert ranked[0].hypothesis_ids == ["h2"]
        assert ranked[1].hypothesis_ids == ["h1"]
        assert ranked[2].hypothesis_ids == ["h3"]


class TestCounterfactualExperiment:
    """Test counterfactual-enhanced experiment design."""

    @pytest.mark.asyncio
    async def test_design_without_counterfactual(
        self, graph: CognitiveGraph,
    ) -> None:
        planner = ExperimentPlanner(graph=graph, counterfactual=None)
        design = await planner.design_counterfactual_experiment("b1")
        assert "No CounterfactualReasoner" in design.reasoning

    @pytest.mark.asyncio
    async def test_design_with_counterfactual(
        self, populated_graph: dict[str, Any],
    ) -> None:
        graph = populated_graph["graph"]
        cf = CounterfactualReasoner(graph=graph)
        planner = ExperimentPlanner(graph=graph, counterfactual=cf)

        design = await planner.design_counterfactual_experiment(
            populated_graph["belief_id"],
        )

        assert "Counterfactual-guided" in design.reasoning
        assert design.expected_information_gain > 0
        assert design.discriminating_power > 0
