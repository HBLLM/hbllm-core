"""
Integration Tests for Composite Brain Architecture & HCIR Cognitive Authority.

Covers:
- Brain creation via BrainFactory with v4 composite architecture (use_composites=True)
- Verification of 8 composite subsystems
- HCIR Cognitive OS wiring, ExecutiveRuntime execution, and 100% Cognitive Authority Metric
- Deprecation warning enforcement when use_composites=False is requested
- HCIR-native MemorySystem operations without legacy fallback
"""

from __future__ import annotations

import tempfile
import warnings

import pytest

from hbllm.brain.composites import (
    GovernanceGuard,
    LearningLoop,
    MemorySystem,
    MetaCognition,
    ReasoningCore,
    ResourceManager,
    SkillEngine,
    SocialLayer,
)
from hbllm.brain.core.factory import BrainConfig, BrainFactory
from hbllm.hcir.kernel.governance.policies.migration_policy import MigrationMode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.testing import MockProvider


@pytest.mark.asyncio
async def test_brain_factory_composite_and_hcir_authority():
    """Verify BrainFactory creates a composite brain with 100% HCIR authority."""
    with tempfile.TemporaryDirectory() as tmpdir:
        provider = MockProvider(default_response="Quantum computing harnesses superposition.")
        cfg = BrainConfig(
            data_dir=tmpdir,
            use_composites=True,
            inject_perception=False,
            inject_iot=False,
        )

        brain = await BrainFactory.create(provider=provider, config=cfg)
        try:
            # 1. Verify composite nodes are wired and running
            assert brain.nodes is not None
            assert len(brain.nodes) > 0

            # Check for top-level composite instances
            composite_types = {type(n) for n in brain.nodes}
            assert ReasoningCore in composite_types
            assert MemorySystem in composite_types
            assert GovernanceGuard in composite_types
            assert MetaCognition in composite_types
            assert SkillEngine in composite_types
            assert ResourceManager in composite_types
            assert SocialLayer in composite_types
            assert LearningLoop in composite_types

            # 2. Verify HCIR Cognitive OS Services
            assert brain.hcir_services is not None
            assert brain.hcir_services.workspace is not None
            assert brain.hcir_services.transaction_manager is not None
            assert brain.hcir_services.capability_resolver is not None
            assert brain.hcir_services.constitutional_verifier is not None
            assert brain.hcir_services.bus_bridge is not None
            assert brain.hcir_services.migration_policy is not None
            assert brain.hcir_services.migration_metrics is not None

            # Verify HCIR migration mode is promoted to HCIR
            assert brain.hcir_services.migration_policy.mode == MigrationMode.HCIR

            # Verify initial Cognitive Authority Metric is 100%
            authority = brain.get_cognitive_authority()
            assert authority == 100.0

            # 3. Verify ExecutiveRuntime cycle execution
            assert brain.hcir_runtime is not None
            cycle_result = await brain.hcir_runtime.run_cycle()
            assert cycle_result.cycle_index >= 1

            # Verify authority metric remains 100% after cycle
            assert brain.get_cognitive_authority() == 100.0

            # 4. Verify process query execution
            result = await brain.process("What is quantum computing?", session_id="sess_test")
            assert result is not None
            assert not result.error
            assert brain.get_cognitive_authority() == 100.0

        finally:
            await brain.shutdown()


@pytest.mark.asyncio
async def test_use_composites_false_deprecation_warning():
    """Verify that use_composites=False emits a DeprecationWarning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        provider = MockProvider(default_response="Legacy test response.")
        cfg = BrainConfig(
            data_dir=tmpdir,
            use_composites=False,
            inject_perception=False,
            inject_iot=False,
        )

        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            brain = await BrainFactory.create(provider=provider, config=cfg)
            try:
                # Check for DeprecationWarning
                dep_warnings = [
                    w for w in recorded_warnings if issubclass(w.category, DeprecationWarning)
                ]
                assert len(dep_warnings) > 0
                messages = [str(w.message) for w in dep_warnings]
                assert any("use_composites=False" in m or "create_legacy_nodes" in m for m in messages)
            finally:
                await brain.shutdown()


@pytest.mark.asyncio
async def test_memory_system_hcir_native():
    """Verify MemorySystem executes memory operations via HCIR natively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bus = InProcessBus()
        await bus.start()

        mem_sys = MemorySystem(
            node_id="test_memory_system",
            db_path=f"{tmpdir}/working_memory.db",
        )
        await mem_sys.start(bus)

        try:
            # Test memory.stats
            msg_stats = Message(
                type=MessageType.QUERY,
                source_node_id="tester",
                topic="memory.stats",
                payload={"tenant_id": "default"},
            )
            reply_stats = await bus.request("memory.stats", msg_stats, timeout=3.0)
            assert reply_stats.type != MessageType.ERROR
            assert reply_stats.payload.get("phase") == "legacy_removed"
            assert reply_stats.payload.get("hcir_active") is True

            # Test memory.browse
            msg_browse = Message(
                type=MessageType.QUERY,
                source_node_id="tester",
                topic="memory.browse",
                payload={"offset": 0, "limit": 10},
            )
            reply_browse = await bus.request("memory.browse", msg_browse, timeout=3.0)
            assert reply_browse.type != MessageType.ERROR
            assert "entries" in reply_browse.payload

            # Test memory.search
            msg_search = Message(
                type=MessageType.QUERY,
                source_node_id="tester",
                topic="memory.search",
                payload={"query_text": "quantum"},
            )
            reply_search = await bus.request("memory.search", msg_search, timeout=3.0)
            assert reply_search.type != MessageType.ERROR
            assert "results" in reply_search.payload

        finally:
            await mem_sys.stop()
            await bus.stop()
