"""Tests for PredictionTracker — competing prediction tracking."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.prediction_tracker import PredictionTracker
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import CognitiveGraph


class TestRegisterPrediction:
    """Test prediction registration."""

    @pytest.mark.asyncio
    async def test_register_creates_prediction(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Test", "Why X?")
        obj = workspace.add_objective(prog.program_id, "Find")
        q_id = workspace.add_question(prog.program_id, obj, "Why X?", importance=0.8)

        from hbllm.brain.epistemics.hypothesis_builder import HypothesisBuilder
        from hbllm.brain.epistemics.idea_generator import IdeaGenerator

        gen = IdeaGenerator(graph=graph)
        ideas = await gen.generate_from_unknown(q_id)
        builder = HypothesisBuilder(graph=graph)
        candidates = await builder.validate(ideas)
        hyp_id = await builder.promote_to_node(candidates[0], prog.program_id)

        tracker = PredictionTracker(graph=graph)
        pred_id = await tracker.register_prediction(
            hyp_id,
            "X increases by 10%",
            "increase",
        )

        assert pred_id != ""

    @pytest.mark.asyncio
    async def test_register_for_nonexistent_hypothesis(
        self,
        graph: CognitiveGraph,
    ) -> None:
        tracker = PredictionTracker(graph=graph)
        with pytest.raises(ValueError, match="Dangling edge reference"):
            await tracker.register_prediction(
                "nonexistent",
                "X increases",
                "increase",
            )


class TestCheckPrediction:
    """Test prediction outcome checking."""

    @pytest.mark.asyncio
    async def test_check_correct_prediction(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Test", "Why X?")
        obj = workspace.add_objective(prog.program_id, "Find")
        q_id = workspace.add_question(prog.program_id, obj, "Why X?", importance=0.8)

        from hbllm.brain.epistemics.hypothesis_builder import HypothesisBuilder
        from hbllm.brain.epistemics.idea_generator import IdeaGenerator

        gen = IdeaGenerator(graph=graph)
        ideas = await gen.generate_from_unknown(q_id)
        builder = HypothesisBuilder(graph=graph)
        candidates = await builder.validate(ideas)
        hyp_id = await builder.promote_to_node(candidates[0], prog.program_id)

        tracker = PredictionTracker(graph=graph)
        pred_id = await tracker.register_prediction(
            hyp_id,
            "X increases",
            "increase",
        )
        outcome = await tracker.check_prediction(pred_id, "increased by 15%")

        assert outcome.prediction_id == pred_id
        assert hasattr(outcome, "correct")
        assert hasattr(outcome, "confidence_delta")


class TestExpiredPredictions:
    """Test expired prediction detection."""

    @pytest.mark.asyncio
    async def test_check_expired_empty(self, graph: CognitiveGraph) -> None:
        tracker = PredictionTracker(graph=graph)
        expired = await tracker.check_expired_predictions()
        assert expired == []
