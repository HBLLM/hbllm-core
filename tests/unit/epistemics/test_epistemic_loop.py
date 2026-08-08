"""Tests for EpistemicLoop — orchestrator with memory + calibration."""

from __future__ import annotations

import pytest

from hbllm.brain.epistemics.calibration import EpistemicCalibrationEngine
from hbllm.brain.epistemics.counterfactual import CounterfactualReasoner
from hbllm.brain.epistemics.epistemic_loop import EpistemicLoop
from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
)
from hbllm.hcir.types import BeliefConfidence


class TestLoopCreation:
    """Test loop initialization."""

    def test_create_minimal(self, graph: CognitiveGraph) -> None:
        loop = EpistemicLoop(graph=graph)
        assert loop.cycle_count == 0
        assert len(loop.engines) == 9

    def test_create_full(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
        memory: EpistemicMemory,
    ) -> None:
        calibrator = EpistemicCalibrationEngine(memory=memory)
        cf = CounterfactualReasoner(graph=graph)

        loop = EpistemicLoop(
            graph=graph,
            workspace=workspace,
            memory=memory,
            calibration=calibrator,
            counterfactual=cf,
            calibration_interval=3,
        )

        assert loop.engines["idea_generator"]._memory is memory
        assert loop.engines["experiment_planner"]._counterfactual is cf


class TestRunCycle:
    """Test the epistemic cycle."""

    @pytest.mark.asyncio
    async def test_empty_graph_cycle(self, graph: CognitiveGraph) -> None:
        loop = EpistemicLoop(graph=graph)
        result = await loop.run_cycle()
        # Empty graph → nothing to investigate
        assert result is None
        assert loop.cycle_count == 1

    @pytest.mark.asyncio
    async def test_cycle_with_research_program(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Test", "Why X?")
        obj = workspace.add_objective(prog.program_id, "Find")
        workspace.add_question(prog.program_id, obj, "Why X?", importance=0.8)

        loop = EpistemicLoop(graph=graph, workspace=workspace)
        await loop.run_cycle()

        assert loop.cycle_count == 1
        assert loop.last_cycle_time >= 0


class TestMemoryRecording:
    """Test that the loop records to epistemic memory."""

    @pytest.mark.asyncio
    async def test_belief_snapshots_recorded(
        self,
        graph: CognitiveGraph,
        memory: EpistemicMemory,
        workspace: DiscoveryWorkspace,
    ) -> None:
        # Add a belief to the graph
        belief = BeliefNode(
            claim="X is true",
            belief_confidence=BeliefConfidence(
                evidence_quality=0.8,
                reproducibility=0.7,
            ),
        )
        graph.upsert_node(belief)

        # Add a research program so the loop has something to do
        prog = workspace.create_program("Test", "Why X?")
        obj = workspace.add_objective(prog.program_id, "Find")
        workspace.add_question(prog.program_id, obj, "Why X?", importance=0.8)

        loop = EpistemicLoop(graph=graph, workspace=workspace, memory=memory)
        await loop.run_cycle()

        # Memory should have a snapshot
        trajectory = await memory.get_confidence_trajectory(belief.id)
        assert len(trajectory) >= 1
        assert trajectory[0].derived_confidence == belief.belief_confidence.derived_confidence


class TestCalibrationAutoSwitch:
    """Test calibration-driven strategy switching."""

    @pytest.mark.asyncio
    async def test_calibration_runs_at_interval(
        self,
        graph: CognitiveGraph,
        memory: EpistemicMemory,
    ) -> None:
        calibrator = EpistemicCalibrationEngine(memory=memory)

        loop = EpistemicLoop(
            graph=graph,
            memory=memory,
            calibration=calibrator,
            calibration_interval=2,
        )

        # Cycle 1: no calibration
        await loop.run_cycle()
        assert loop.cycle_count == 1

        # Cycle 2: calibration runs (interval=2)
        await loop.run_cycle()
        assert loop.cycle_count == 2

        # Strategy manager should still have a valid strategy
        sm = loop.engines["strategy_manager"]
        assert sm.active_strategy is not None
