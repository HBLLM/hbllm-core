"""Cross-component test: epistemics ↔ AutonomyCore wiring."""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

from hbllm.brain.epistemics.integration import wire_epistemics
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import CognitiveGraph


class MockAutonomyCore:
    """Minimal AutonomyCore mock for wiring tests."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.tick_count = 0

    def add_proactive_handler(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    def remove_proactive_handler(self, name: str) -> None:
        self.handlers.pop(name, None)

    async def simulate_tick(self) -> Any:
        """Simulate what AutonomyCore does on a cognitive tick."""
        self.tick_count += 1
        results = {}
        for name, handler in self.handlers.items():
            results[name] = await handler()
        return results


class TestWireEpistemicsIntegration:
    """Test wire_epistemics with realistic AutonomyCore interactions."""

    @pytest.mark.asyncio
    async def test_proactive_handler_runs_on_tick(self) -> None:
        """Verify the epistemic handler runs on simulated AutonomyCore ticks."""
        with tempfile.TemporaryDirectory() as td:
            graph = CognitiveGraph()
            core = MockAutonomyCore()

            loop = wire_epistemics(
                autonomy_core=core,
                graph=graph,
                data_dir=td,
            )

            # Handler should be registered
            assert "epistemic" in core.handlers

            # Simulate 3 ticks
            for _ in range(3):
                results = await core.simulate_tick()
                assert "epistemic" in results

            assert core.tick_count == 3
            assert loop.cycle_count == 3

    @pytest.mark.asyncio
    async def test_handler_produces_results_with_program(self) -> None:
        """Handler should produce results when there's a research program."""
        with tempfile.TemporaryDirectory() as td:
            graph = CognitiveGraph()
            core = MockAutonomyCore()

            loop = wire_epistemics(
                autonomy_core=core,
                graph=graph,
                data_dir=td,
            )

            # Access workspace through the loop
            ws = DiscoveryWorkspace(data_dir=td, graph=graph)
            prog = ws.create_program("Test", "Why X?")
            obj = ws.add_objective(prog.program_id, "Find X")
            ws.add_question(prog.program_id, obj, "Why X?", importance=0.8)

            await core.simulate_tick()
            assert loop.cycle_count == 1

    @pytest.mark.asyncio
    async def test_handler_is_idempotent(self) -> None:
        """Multiple calls produce consistent, non-crashing behavior."""
        with tempfile.TemporaryDirectory() as td:
            graph = CognitiveGraph()
            core = MockAutonomyCore()

            wire_epistemics(
                autonomy_core=core,
                graph=graph,
                data_dir=td,
            )

            # Call the handler 5 times directly
            handler = core.handlers["epistemic"]
            for _ in range(5):
                await handler()  # Should not raise

    @pytest.mark.asyncio
    async def test_memory_persists_across_rewiring(self) -> None:
        """Memory recorded by one loop survives rewiring to a new loop."""
        with tempfile.TemporaryDirectory() as td:
            graph = CognitiveGraph()

            # Wire first loop
            core1 = MockAutonomyCore()
            loop1 = wire_epistemics(
                autonomy_core=core1,
                graph=graph,
                data_dir=td,
            )

            # Record something to memory
            from hbllm.brain.epistemics.interfaces import PredictionOutcome

            outcome = PredictionOutcome(
                prediction_id="p1",
                hypothesis_id="h1",
                predicted="x",
                observed="x",
                correct=True,
            )
            await loop1._memory.record_prediction_result(outcome)

            # Wire a second loop (same data_dir)
            core2 = MockAutonomyCore()
            loop2 = wire_epistemics(
                autonomy_core=core2,
                graph=graph,
                data_dir=td,
            )

            # Memory should still have the prediction
            accuracy = await loop2._memory.get_prediction_accuracy()
            assert accuracy == 1.0

    @pytest.mark.asyncio
    async def test_all_engines_accessible(self) -> None:
        """All 9 engines should be accessible through the loop."""
        with tempfile.TemporaryDirectory() as td:
            graph = CognitiveGraph()
            core = MockAutonomyCore()

            loop = wire_epistemics(
                autonomy_core=core,
                graph=graph,
                data_dir=td,
            )

            engines = loop.engines
            expected = {
                "curiosity",
                "idea_generator",
                "hypothesis_builder",
                "prediction_tracker",
                "experiment_planner",
                "evidence_evaluator",
                "contradiction_engine",
                "explanation_engine",
                "strategy_manager",
            }
            assert set(engines.keys()) == expected
