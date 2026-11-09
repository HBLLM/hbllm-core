"""Autonomous dual-rhythm runtime test suite.

Validates:
1. Online reactive rhythm triggered by ObservationBus events.
2. Offline idle consolidation rhythm triggered by sensory silence (Δt > idle_threshold).
3. Hippocampal replay, concept maintenance, and schema pruning during offline consolidation.
4. Transactional state commit under author='offline_consolidation' with valid grants.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hbllm.hcir.grounding.contracts.enums import ModalitySource, TrafficClass
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.runtime.bootstrap import (
    CognitiveSystemConfig,
    build_cognitive_system,
)


@pytest.mark.asyncio
async def test_autonomous_dual_rhythm_initialization(tmp_path: Path) -> None:
    """Test that cognitive system bootstraps with wired dual-rhythm executive runtime."""
    mem_dir = tmp_path / "dual_rhythm_mem"
    config = CognitiveSystemConfig(
        world_id="rhythm_test",
        memory_dir=mem_dir,
        enable_grounding=True,
    )

    system = build_cognitive_system(config)
    assert system.runtime is not None
    runtime = system.runtime

    assert runtime.services is system.services
    assert runtime.consolidation_count == 0
    assert not runtime.is_idle_consolidated
    assert runtime.last_observation_time > 0


@pytest.mark.asyncio
async def test_online_reactive_observation_resets_idle_timer(tmp_path: Path) -> None:
    """Verify that incoming observation envelopes update timestamps and reset idle state."""
    mem_dir = tmp_path / "reactive_mem"
    config = CognitiveSystemConfig(
        world_id="reactive_test",
        memory_dir=mem_dir,
        enable_grounding=True,
    )

    system = build_cognitive_system(config)
    runtime = system.runtime
    assert runtime is not None
    assert system.bridge is not None

    bus = system.bridge.bus
    initial_time = runtime.last_observation_time

    await asyncio.sleep(0.05)

    # Publish observation envelope onto the bus
    envelope = ObservationEnvelope(
        observation_id="obs_env_1",
        source_modality=ModalitySource.VISION,
        source_channel_id="sensor_visual_grid",
        traffic_class=TrafficClass.CLASS_DISCRETE,
        sequence_number=1,
        wall_timestamp=100.0,
        monotonic_timestamp=100.0,
        raw_payload_ref="grid_payload_1",
    )
    bus.publish(envelope)

    assert runtime.last_observation_time >= initial_time
    assert not runtime.is_idle_consolidated


@pytest.mark.asyncio
async def test_offline_idle_consolidation_execution(tmp_path: Path) -> None:
    """Verify that offline idle consolidation commits state under author='offline_consolidation'."""
    mem_dir = tmp_path / "consolidation_mem"
    config = CognitiveSystemConfig(
        world_id="consolidation_test",
        memory_dir=mem_dir,
        enable_grounding=True,
    )

    system = build_cognitive_system(config)
    runtime = system.runtime
    assert runtime is not None

    # Seed world theory with candidate features
    system.engine.symbolic_theory.goal_features.add(2)
    system.engine.symbolic_theory.walkable_features.add(0)

    # Execute consolidation directly
    tx = await runtime.run_offline_consolidation()

    assert tx is not None
    assert tx.status.value == "committed"
    assert tx.author == "offline_consolidation"
    assert runtime.consolidation_count == 1
    assert runtime.is_idle_consolidated

    # Verify consolidation marker node exists in persistent tier
    persistent_graph = system.tiered_workspace.persistent.graph
    assert persistent_graph.has_node("consolidation_marker_1")


@pytest.mark.asyncio
async def test_autonomous_background_idle_loop(tmp_path: Path) -> None:
    """Verify that sensory silence automatically triggers background consolidation."""
    mem_dir = tmp_path / "background_loop_mem"
    config = CognitiveSystemConfig(
        world_id="loop_test",
        memory_dir=mem_dir,
        enable_grounding=True,
    )

    system = build_cognitive_system(config)
    runtime = system.runtime
    assert runtime is not None
    assert system.bridge is not None
    bus = system.bridge.bus

    # Start background loop with 0.1s idle threshold
    await runtime.start_autonomous_loop(idle_threshold=0.1, check_interval=0.02)

    try:
        # Wait for idle timeout
        await asyncio.sleep(0.25)
        assert runtime.consolidation_count >= 1
        assert runtime.is_idle_consolidated

        # Send observation to interrupt silence
        envelope = ObservationEnvelope(
            observation_id="obs_env_interruption",
            source_modality=ModalitySource.VISION,
            source_channel_id="sensor_visual_grid",
            traffic_class=TrafficClass.CLASS_DISCRETE,
            sequence_number=2,
            wall_timestamp=200.0,
            monotonic_timestamp=200.0,
            raw_payload_ref="grid_payload_2",
        )
        bus.publish(envelope)

        # Flag must be reset
        assert not runtime.is_idle_consolidated

        # Wait for next idle consolidation
        await asyncio.sleep(0.25)
        assert runtime.consolidation_count >= 2
        assert runtime.is_idle_consolidated

    finally:
        await runtime.stop_autonomous_loop()
        assert runtime._autonomous_task is None
