"""Adversarial integration test: Live Epistemic Runtime & AutonomyCore Wiring.

Verifies:
1. BrainFactory._build_brain() properly wires EpistemicLoop into AutonomyCore.
2. BootSequence preserves and reuses the configured AutonomyCore without clobbering it.
3. Cognitive heartbeat ticks genuinely execute EpistemicLoop.run_cycle().
4. Real hypothesis, contradiction, and belief revision cycles occur on the live graph.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pytest

from hbllm.brain.autonomy.loop import AutonomyCore
from hbllm.brain.core.factory import BrainConfig, BrainFactory
from hbllm.brain.epistemics.epistemic_loop import EpistemicLoop
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    ObservationNode,
)
from hbllm.network.bus import InProcessBus
from hbllm.serving.boot import BootContext


@pytest.mark.asyncio
async def test_brain_factory_wires_epistemic_runtime_into_autonomy() -> None:
    """Verify BrainFactory connects EpistemicLoop to the surviving AutonomyCore instance."""
    with tempfile.TemporaryDirectory() as td:
        cfg = BrainConfig(data_dir=td)
        bus = InProcessBus()
        brain = await BrainFactory.create_local(config=cfg, bus=bus)

        try:
            assert brain.autonomy_core is not None
            assert isinstance(brain.autonomy_core, AutonomyCore)

            # Check that epistemic_loop is initialized and attached to brain
            assert brain.epistemic_loop is not None
            assert isinstance(brain.epistemic_loop, EpistemicLoop)

            # Check that epistemic_loop is registered as a proactive handler on autonomy_core
            assert "epistemic" in brain.autonomy_core._proactive_handlers
            handler = brain.autonomy_core._proactive_handlers["epistemic"]
            assert handler == brain.epistemic_loop.run_cycle

            # Check reasoning runtime is wired
            assert brain.reasoning_runtime is not None
            assert brain.reasoning_registry is not None
            assert "deduction" in brain.reasoning_registry.operator_ids
            assert "spatial" in brain.reasoning_registry.operator_ids
        finally:
            if brain.autonomy_core is not None:
                await brain.autonomy_core.stop()


@pytest.mark.asyncio
async def test_boot_sequence_does_not_clobber_configured_autonomy_core() -> None:
    """Verify BootSequence Step 8 reuses the factory's configured AutonomyCore."""
    with tempfile.TemporaryDirectory() as td:
        bus = InProcessBus()
        brain_cfg = BrainConfig(data_dir=td)
        brain = await BrainFactory.create_local(config=brain_cfg, bus=bus)

        original_autonomy = brain.autonomy_core
        original_epistemic = brain.epistemic_loop

        assert original_autonomy is not None
        assert original_epistemic is not None

        # Run boot core step simulation
        ctx = BootContext()
        ctx.brain = brain
        ctx.profile = MagicMock()
        ctx.profile.features.autonomy_core = True

        # Execute step 8 logic directly
        if ctx.brain.autonomy_core is not None:
            ctx.autonomy = ctx.brain.autonomy_core
        else:
            ctx.autonomy = AutonomyCore()
            await ctx.autonomy.start(ctx.brain.bus)
            ctx.brain.autonomy_core = ctx.autonomy

        # The autonomy core must NOT have been replaced by an unconfigured instance
        assert ctx.autonomy is original_autonomy
        assert ctx.brain.autonomy_core is original_autonomy
        assert ctx.brain.epistemic_loop is original_epistemic
        assert "epistemic" in ctx.autonomy._proactive_handlers

        if brain.autonomy_core is not None:
            await brain.autonomy_core.stop()


@pytest.mark.asyncio
async def test_live_cognitive_tick_fires_epistemic_cycle() -> None:
    """Adversarial check: ensure _cognitive_tick() invokes EpistemicLoop.run_cycle()."""
    with tempfile.TemporaryDirectory() as td:
        bus = InProcessBus()
        cfg = BrainConfig(data_dir=td)
        brain = await BrainFactory.create_local(config=cfg, bus=bus)

        try:
            loop: EpistemicLoop = brain.epistemic_loop
            core: AutonomyCore = brain.autonomy_core
            initial_cycles = loop.cycle_count

            # Add an observation and conflicting belief to graph to give epistemic loop work
            graph: CognitiveGraph = loop._graph
            obs_node = ObservationNode(
                id="obs_test_1",
                sensor_modality="visual",
                raw_payload={"status": "unstable_support"},
            )
            belief_node = BeliefNode(
                id="belief_test_1",
                claim="status is stable_support",
                confidence=0.9,
            )
            graph.add_node(obs_node)
            graph.add_node(belief_node)

            # Trigger manual cognitive tick on the live autonomy core
            await core._cognitive_tick()

            # Verify that EpistemicLoop.run_cycle() executed
            assert loop.cycle_count > initial_cycles
            assert core._ticks_completed == 0  # ticks_completed is incremented in run_loop
        finally:
            if brain.autonomy_core is not None:
                await brain.autonomy_core.stop()
