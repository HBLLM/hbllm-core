"""Tests for wire_epistemics() integration helper."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from hbllm.brain.epistemics.epistemic_loop import EpistemicLoop
from hbllm.brain.epistemics.integration import wire_epistemics
from hbllm.hcir.graph import CognitiveGraph


class MockAutonomyCore:
    """Minimal mock of AutonomyCore for testing registration."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def add_proactive_handler(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    def remove_proactive_handler(self, name: str) -> None:
        self.handlers.pop(name, None)


class TestWireEpistemics:
    """Test the wire_epistemics() helper."""

    def test_returns_loop(
        self, graph: CognitiveGraph, tmp_dir: str,
    ) -> None:
        core = MockAutonomyCore()
        loop = wire_epistemics(
            autonomy_core=core,
            graph=graph,
            data_dir=tmp_dir,
        )
        assert isinstance(loop, EpistemicLoop)

    def test_registers_handler(
        self, graph: CognitiveGraph, tmp_dir: str,
    ) -> None:
        core = MockAutonomyCore()
        wire_epistemics(
            autonomy_core=core,
            graph=graph,
            data_dir=tmp_dir,
        )
        assert "epistemic" in core.handlers
        assert callable(core.handlers["epistemic"])

    def test_engines_wired(
        self, graph: CognitiveGraph, tmp_dir: str,
    ) -> None:
        core = MockAutonomyCore()
        loop = wire_epistemics(
            autonomy_core=core,
            graph=graph,
            data_dir=tmp_dir,
        )

        # Memory should be wired into idea generator
        assert loop.engines["idea_generator"]._memory is not None

        # Counterfactual should be wired into experiment planner
        assert loop.engines["experiment_planner"]._counterfactual is not None

    def test_custom_params(
        self, graph: CognitiveGraph, tmp_dir: str,
    ) -> None:
        core = MockAutonomyCore()
        loop = wire_epistemics(
            autonomy_core=core,
            graph=graph,
            data_dir=tmp_dir,
            calibration_interval=10,
            max_investigations_per_cycle=5,
            max_ideas_per_investigation=20,
        )
        assert loop._calibration_interval == 10
        assert loop._max_investigations == 5

    @pytest.mark.asyncio
    async def test_handler_is_callable(
        self, graph: CognitiveGraph, tmp_dir: str,
    ) -> None:
        core = MockAutonomyCore()
        wire_epistemics(
            autonomy_core=core,
            graph=graph,
            data_dir=tmp_dir,
        )

        # Should be able to call the handler without error
        result = await core.handlers["epistemic"]()
        # Empty graph → nothing to investigate
        assert result is None
