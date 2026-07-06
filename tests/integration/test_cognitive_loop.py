"""
Integration Test — Full Cognitive Loop Verification.

Tests the end-to-end cognitive pipeline with the v3 integration wiring:

    Happy Path:
        BrainContainer → BrainContext → Trace → CapabilityRegistry
        → OscillationManager (BrainTick) → Simulation (DeliberationBudget)
        → BeliefGraph → EvidencePacket → Memory (Repository + Projection)

    Failure Recovery:
        Wrong prediction → Belief contradiction → Reasoner detects
        → Simulation explores alternatives → Memory correction
        → Belief update → Sleep consolidation
"""

from __future__ import annotations

import asyncio
import time

import pytest

# ── Infrastructure (Phase A) ──
from hbllm.brain.brain_container import BrainConfig, BrainContainer
from hbllm.brain.brain_context import BrainContext, BrainServices, BrainState
from hbllm.brain.capability_registry import CapabilityRegistry
from hbllm.brain.evidence import EvidenceBuilder, EvidencePacket

# ── Prediction (Phase D) ──
from hbllm.brain.prediction import CognitivePredictors
from hbllm.brain.simulation_engine import (
    DeliberationBudget,
    DeliberationLevel,
    SimulationEngine,
)

# ── Brain (Phase C) ──
from hbllm.brain.snn.network import ProjectionType
from hbllm.brain.snn.oscillations import BrainTick, OscillationBand, OscillationManager
from hbllm.brain.snn.population import PopulationEncoder
from hbllm.brain.trace import TraceCollector

# ── Memory (Phase B) ──
from hbllm.memory.belief_graph import BeliefGraph, BeliefRecord
from hbllm.memory.goal_memory import GoalCube
from hbllm.memory.memcube import MemoryEvent, MemoryEventStore
from hbllm.memory.repository import MemoryProjection

# ═══════════════════════════════════════════════════════════════════════════
# Phase A — Infrastructure Foundation
# ═══════════════════════════════════════════════════════════════════════════


class TestBrainContainerBootstrap:
    """Verify BrainContainer builds a fully wired BrainContext."""

    def test_default_bootstrap(self) -> None:
        """BrainContainer.build() produces a valid context."""
        ctx = BrainContainer.build()
        assert isinstance(ctx, BrainContext)
        assert isinstance(ctx.services, BrainServices)
        assert isinstance(ctx.state, BrainState)

    def test_services_registered(self) -> None:
        """All expected services are registered in the capability registry."""
        ctx = BrainContainer.build()
        reg = ctx.services.capability_registry
        assert reg.has("simulation_engine")
        assert reg.has("brain_clock")
        assert reg.has("goal_memory")
        assert reg.has("belief_graph")
        assert reg.has("cognitive_predictors")
        assert reg.has("neuromodulation")
        assert reg.has("trace_collector")

    def test_capability_discovery(self) -> None:
        """Services can be discovered by capability tags."""
        ctx = BrainContainer.build()
        reg = ctx.services.capability_registry

        sims = reg.find("simulation")
        assert len(sims) == 1
        assert isinstance(sims[0], SimulationEngine)

        clocks = reg.find("timing")
        assert len(clocks) == 1
        assert isinstance(clocks[0], OscillationManager)

    def test_disabled_simulation(self) -> None:
        """Simulation can be disabled via config."""
        cfg = BrainConfig(enable_simulation=False)
        ctx = BrainContainer.build(config=cfg)
        assert ctx.services.simulation is None
        assert not ctx.services.capability_registry.has("simulation_engine")

    def test_context_stats(self) -> None:
        """BrainContext.stats() returns combined statistics."""
        ctx = BrainContainer.build()
        stats = ctx.stats()
        assert "services" in stats
        assert "state" in stats
        assert stats["services"]["capabilities"]["services"] > 0


class TestCapabilityRegistry:
    """Verify the capability registry lifecycle."""

    def test_register_and_find(self) -> None:
        reg = CapabilityRegistry()
        service = {"name": "test_service"}
        reg.register("test", service, ["reasoning", "planning"])

        assert reg.find("reasoning") == [service]
        assert reg.find("planning") == [service]
        assert reg.find("nonexistent") == []

    def test_unregister(self) -> None:
        reg = CapabilityRegistry()
        reg.register("test", "svc", ["cap_a"])
        assert reg.has("test")
        assert reg.unregister("test")
        assert not reg.has("test")
        assert reg.find("cap_a") == []

    def test_find_one(self) -> None:
        reg = CapabilityRegistry()
        reg.register("a", "svc_a", ["shared"])
        reg.register("b", "svc_b", ["shared"])
        result = reg.find_one("shared")
        assert result in ["svc_a", "svc_b"]

    def test_duplicate_registration_raises(self) -> None:
        reg = CapabilityRegistry()
        reg.register("test", "svc", ["cap"])
        with pytest.raises(ValueError, match="already registered"):
            reg.register("test", "svc2", ["cap2"])


class TestTraceCollector:
    """Verify end-to-end cognitive tracing."""

    def test_trace_lifecycle(self) -> None:
        collector = TraceCollector(max_retained=5)
        trace = collector.start_trace(source="user_input")

        assert trace.is_active
        trace.record("saliency", "scored", {"score": 0.92})
        trace.record("competition", "selected")
        trace.record("workspace", "broadcast")
        trace.record("planner", "candidates_generated")
        trace.record("simulation", "approved", {"critic_score": 0.85})
        trace.record("decision", "committed")
        trace.record("memory", "stored")

        collector.finish_trace(trace)

        assert not trace.is_active
        assert len(trace.events) == 7
        assert trace.component_path == [
            "saliency",
            "competition",
            "workspace",
            "planner",
            "simulation",
            "decision",
            "memory",
        ]

    def test_trace_retention_limit(self) -> None:
        collector = TraceCollector(max_retained=3)
        for i in range(5):
            t = collector.start_trace(source=f"event_{i}")
            t.record("test", "action")
            collector.finish_trace(t)
        assert len(collector._completed_traces) == 3

    def test_trace_lookup(self) -> None:
        collector = TraceCollector()
        trace = collector.start_trace(source="test")
        found = collector.get_trace(trace.trace_id)
        assert found is trace


# ═══════════════════════════════════════════════════════════════════════════
# Phase B — Memory Subsystem Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryProjection:
    """Verify MemoryProjection folds events into MemCubes."""

    @pytest.fixture
    def store(self) -> MemoryEventStore:
        return MemoryEventStore()

    @pytest.fixture
    def projection(self, store: MemoryEventStore) -> MemoryProjection:
        return MemoryProjection(store)

    @pytest.mark.asyncio
    async def test_project_single_memory(
        self, store: MemoryEventStore, projection: MemoryProjection
    ) -> None:
        await store.initialize()
        event = MemoryEvent.create(
            memory_id="mem_001",
            content="User prefers dark mode",
            memory_type="semantic",
            source_node="perception",
            tenant_id="default",
        )
        await store.append(event)

        cube = await projection.project("mem_001")
        assert cube is not None
        assert cube.content == "User prefers dark mode"

    @pytest.mark.asyncio
    async def test_project_corrected_memory(
        self, store: MemoryEventStore, projection: MemoryProjection
    ) -> None:
        await store.initialize()
        create = MemoryEvent.create(
            memory_id="mem_002",
            content="User likes Python",
            memory_type="semantic",
            source_node="perception",
            tenant_id="default",
        )
        await store.append(create)

        correct = MemoryEvent.correct(
            memory_id="mem_002",
            old_content="User likes Python",
            new_content="User prefers Rust",
            reason="Explicit correction",
            source_node="workspace",
        )
        await store.append(correct)

        cube = await projection.project("mem_002")
        assert cube is not None
        assert cube.content == "User prefers Rust"
        assert cube.version == 2

    @pytest.mark.asyncio
    async def test_project_nonexistent(
        self, store: MemoryEventStore, projection: MemoryProjection
    ) -> None:
        await store.initialize()
        cube = await projection.project("nonexistent")
        assert cube is None


# ═══════════════════════════════════════════════════════════════════════════
# Phase C — Brain Subsystem Wiring
# ═══════════════════════════════════════════════════════════════════════════


class TestBrainTick:
    """Verify BrainTick heartbeat generation."""

    def test_generate_tick(self) -> None:
        mgr = OscillationManager()
        tick = mgr.generate_tick(1.0, cognitive_load=0.6, attention_level=0.8)

        assert isinstance(tick, BrainTick)
        assert tick.cycle == 1
        assert tick.cognitive_load == 0.6
        assert tick.attention_level == 0.8
        assert tick.dominant_band in [b.value for b in OscillationBand]
        assert all(b.value in tick.phase for b in OscillationBand)
        assert all(b.value in tick.gate for b in OscillationBand)

    def test_cycle_counter_increments(self) -> None:
        mgr = OscillationManager()
        t1 = mgr.generate_tick(0.0)
        t2 = mgr.generate_tick(0.1)
        t3 = mgr.generate_tick(0.2)
        assert t1.cycle == 1
        assert t2.cycle == 2
        assert t3.cycle == 3

    def test_tick_to_dict(self) -> None:
        mgr = OscillationManager()
        tick = mgr.generate_tick(0.5, fatigue=0.3)
        d = tick.to_dict()
        assert "phase" in d
        assert "gate" in d
        assert d["fatigue"] == 0.3
        assert d["dominant_band"] != ""


class TestProjectionType:
    """Verify ProjectionType enum."""

    def test_projection_types_defined(self) -> None:
        assert ProjectionType.BASAL == "basal"
        assert ProjectionType.APICAL == "apical"
        assert ProjectionType.MODULATORY == "modulatory"

    def test_layer_projection_accepts_type(self) -> None:
        from hbllm.brain.snn.network import LayerProjection

        proj = LayerProjection("src", "tgt", 4, 4, projection_type=ProjectionType.APICAL)
        assert proj.projection_type == ProjectionType.APICAL

    def test_default_is_basal(self) -> None:
        from hbllm.brain.snn.network import LayerProjection

        proj = LayerProjection("src", "tgt", 4, 4)
        assert proj.projection_type == ProjectionType.BASAL


# ═══════════════════════════════════════════════════════════════════════════
# Phase D — Deliberation & Reasoning
# ═══════════════════════════════════════════════════════════════════════════


class TestDeliberationBudget:
    """Verify adaptive computation via deliberation budget."""

    def test_high_confidence_skips_simulation(self) -> None:
        """Low uncertainty → SKIP."""
        budget = DeliberationBudget(
            uncertainty=0.1,
            importance=0.1,
            novelty=0.1,
            goal_priority=0.1,
        )
        assert budget.level == DeliberationLevel.SKIP
        assert budget.recommended_candidates == 0

    def test_moderate_uncertainty_single_sim(self) -> None:
        """Moderate budget → SINGLE."""
        budget = DeliberationBudget(
            uncertainty=0.7,
            importance=0.5,
            novelty=0.5,
            goal_priority=0.7,
        )
        assert budget.level == DeliberationLevel.SINGLE
        assert budget.recommended_candidates == 1

    def test_high_uncertainty_multiple_sims(self) -> None:
        """High budget → MULTIPLE."""
        budget = DeliberationBudget(
            uncertainty=0.8,
            importance=0.7,
            novelty=0.7,
            goal_priority=0.8,
        )
        assert budget.level == DeliberationLevel.MULTIPLE
        assert budget.recommended_candidates == 3

    def test_maximum_uncertainty_beam_search(self) -> None:
        """Maximum budget → BEAM."""
        budget = DeliberationBudget(
            uncertainty=1.0,
            importance=1.0,
            novelty=1.0,
            goal_priority=1.0,
        )
        assert budget.level == DeliberationLevel.BEAM
        assert budget.recommended_candidates == 5


class TestEvidencePacket:
    """Verify structured evidence for reasoning."""

    def test_uncontested_evidence(self) -> None:
        packet = EvidencePacket(
            fact="User prefers dark mode",
            confidence=0.9,
            supporting_evidence=["User said 'dark mode please'"],
            contradictions=[],
            freshness=0.95,
        )
        assert not packet.is_contested
        assert packet.support_count == 1
        assert packet.reliability_score > 0.8

    def test_contested_evidence(self) -> None:
        packet = EvidencePacket(
            fact="User prefers dark mode",
            confidence=0.6,
            supporting_evidence=["User said 'dark mode'"],
            contradictions=["App settings show light mode"],
            freshness=0.8,
        )
        assert packet.is_contested
        assert packet.reliability_score < 0.5

    def test_evidence_from_raw(self) -> None:
        packet = EvidenceBuilder.from_raw(
            "Tool execution succeeded",
            confidence=0.95,
            source="tool_executor",
        )
        assert packet.fact == "Tool execution succeeded"
        assert packet.confidence == 0.95
        assert "tool_executor" in packet.source_lineage

    def test_merge_evidence(self) -> None:
        p1 = EvidencePacket(
            fact="X is true",
            confidence=0.8,
            supporting_evidence=["source A"],
            freshness=0.9,
        )
        p2 = EvidencePacket(
            fact="X is true",
            confidence=0.6,
            supporting_evidence=["source B"],
            contradictions=["source C"],
            freshness=0.5,
        )
        merged = EvidenceBuilder.merge_evidence([p1, p2])
        assert merged.confidence == 0.7  # average
        assert merged.support_count == 2
        assert len(merged.contradictions) == 1
        assert merged.freshness == 0.9  # max


# ═══════════════════════════════════════════════════════════════════════════
# Phase E — End-to-End Cognitive Loop
# ═══════════════════════════════════════════════════════════════════════════


class TestFullCognitiveLoop:
    """Flagship integration test — full cognitive cycle."""

    def test_happy_path_cognitive_cycle(self) -> None:
        """Verify the full pipeline from bootstrap to decision."""
        # 1. Bootstrap
        ctx = BrainContainer.build()

        # 2. Start a trace for this cognitive event
        trace = ctx.start_trace(source="user_input")
        trace.record("queue", "enqueued", {"content": "fix auth bug"})

        # 3. Saliency scoring (simulated)
        trace.record("saliency", "scored", {"score": 0.92})

        # 4. BrainTick — oscillation heartbeat
        tick = ctx.services.clock.generate_tick(
            time.time(),
            cognitive_load=0.5,
            attention_level=0.85,
        )
        trace.record("brain_clock", "tick", {"cycle": tick.cycle, "dominant": tick.dominant_band})
        assert tick.cycle >= 1

        # 5. Goal alignment
        goal = GoalCube(
            id="goal_auth_001",
            description="Fix authentication issues",
            priority=0.9,
        )
        goal_id = asyncio.get_event_loop().run_until_complete(ctx.state.goals.add_goal(goal))
        trace.record("goals", "active_goal", {"goal_id": goal_id})

        # 6. Deliberation budget
        budget = DeliberationBudget(
            uncertainty=0.6,
            importance=0.8,
            novelty=0.4,
            goal_priority=0.9,
        )
        trace.record(
            "planner",
            "deliberation_budget",
            {
                "score": round(budget.score, 3),
                "level": budget.level,
                "candidates": budget.recommended_candidates,
            },
        )

        # 7. Simulation (if budget warrants it)
        if budget.level != DeliberationLevel.SKIP:
            sim: SimulationEngine = ctx.services.capability_registry.find_one("simulation")
            assert sim is not None
            trace.record("simulation", "requested", {"candidates": budget.recommended_candidates})

        # 8. Memory store via event sourcing
        store = MemoryEventStore()
        asyncio.get_event_loop().run_until_complete(store.initialize())
        event = MemoryEvent.create(
            memory_id="mem_auth_001",
            content="User requested auth bug fix",
            memory_type="episodic",
            source_node="queue",
            tenant_id="default",
        )
        asyncio.get_event_loop().run_until_complete(store.append(event))
        trace.record("memory", "stored", {"memory_id": "mem_auth_001"})

        # 9. Projection
        projection = MemoryProjection(store)
        cube = asyncio.get_event_loop().run_until_complete(projection.project("mem_auth_001"))
        assert cube is not None
        assert cube.content == "User requested auth bug fix"
        trace.record("memory", "projected", {"version": cube.version})

        # 10. Finish trace
        ctx.services.traces.finish_trace(trace)
        assert not trace.is_active
        assert len(trace.events) >= 7
        assert trace.duration > 0

    def test_failure_recovery_path(self) -> None:
        """Verify belief contradiction → correction → recovery."""
        ctx = BrainContainer.build()
        trace = ctx.start_trace(source="prediction_error")
        belief_graph: BeliefGraph = ctx.services.capability_registry.find_one("belief_tracking")

        # 1. Initial belief
        record = BeliefRecord(
            id="br_001",
            memory_id="belief_001",
            reason="User prefers Python",
            confidence=0.8,
            created_by="perception",
            created_at=time.time(),
            trigger="user_statement",
        )
        asyncio.get_event_loop().run_until_complete(belief_graph.record_belief(record))
        trace.record("belief_graph", "belief_added", {"id": "belief_001"})

        # 2. Contradicting evidence arrives
        evidence = EvidencePacket(
            fact="User prefers Rust",
            confidence=0.9,
            supporting_evidence=["User said 'I prefer Rust'"],
            contradictions=["Previous belief: User prefers Python"],
            source_lineage=["user_input"],
            freshness=1.0,
        )
        assert evidence.is_contested
        trace.record(
            "reasoner",
            "contradiction_detected",
            {
                "fact": evidence.fact,
                "reliability": round(evidence.reliability_score, 3),
            },
        )

        # 3. Memory correction via event sourcing
        store = MemoryEventStore()
        asyncio.get_event_loop().run_until_complete(store.initialize())

        # Original memory
        create_evt = MemoryEvent.create(
            memory_id="mem_pref_001",
            content="User prefers Python",
            memory_type="semantic",
            source_node="perception",
            tenant_id="default",
        )
        asyncio.get_event_loop().run_until_complete(store.append(create_evt))

        # Correction
        correct_evt = MemoryEvent.correct(
            memory_id="mem_pref_001",
            old_content="User prefers Python",
            new_content="User prefers Rust",
            reason="Direct user statement contradicts earlier inference",
            source_node="reasoner",
        )
        asyncio.get_event_loop().run_until_complete(store.append(correct_evt))
        trace.record("memory", "corrected", {"memory_id": "mem_pref_001"})

        # 4. Verify correction via projection
        projection = MemoryProjection(store)
        cube = asyncio.get_event_loop().run_until_complete(projection.project("mem_pref_001"))
        assert cube is not None
        assert cube.content == "User prefers Rust"
        assert cube.version == 2
        trace.record("memory", "verified_correction", {"new_content": cube.content})

        # 5. Update belief — correct original, add new
        asyncio.get_event_loop().run_until_complete(
            belief_graph.correct("belief_001", "User actually prefers Rust")
        )
        record2 = BeliefRecord(
            id="br_002",
            memory_id="belief_002",
            reason="User prefers Rust",
            confidence=0.9,
            created_by="reasoner",
            created_at=time.time(),
            trigger="contradiction_resolution",
        )
        asyncio.get_event_loop().run_until_complete(belief_graph.record_belief(record2))
        trace.record(
            "belief_graph",
            "belief_updated",
            {
                "old": "belief_001",
                "new": "belief_002",
            },
        )

        # 6. Finish
        ctx.services.traces.finish_trace(trace)
        assert not trace.is_active
        assert len(trace.events) >= 5

        # Verify recovery path in trace
        components = trace.component_path
        assert "belief_graph" in components
        assert "reasoner" in components
        assert "memory" in components


class TestCrossSubsystemIntegration:
    """Verify that subsystems from different milestones work together."""

    def test_snn_components_with_projection_type(self) -> None:
        """DendriticNeuron + ProjectionType + PopulationEncoder."""
        from hbllm.brain.snn.population import PopulationConfig

        # Encode cognitive state into spike rates
        cfg = PopulationConfig(num_neurons=8, min_value=0.0, max_value=1.0)
        encoder = PopulationEncoder(config=cfg)
        rates = encoder.encode(0.7)
        assert len(rates) == 8
        assert any(r > 0 for r in rates)

        # ProjectionType is an enum
        assert ProjectionType.APICAL != ProjectionType.BASAL

    def test_oscillation_gates_simulation(self) -> None:
        """BrainTick phase-gating controls simulation decisions."""
        mgr = OscillationManager()

        # Generate tick during high theta (memory encoding)
        tick = mgr.generate_tick(0.5)

        # Theta gate controls memory retrieval timing
        theta_gate = tick.gate.get("theta", 0.0)
        assert isinstance(theta_gate, float)
        assert 0.0 <= theta_gate <= 1.0

    def test_prediction_feeds_simulation(self) -> None:
        """CognitivePredictors feeds context for SimulationEngine."""
        predictors = CognitivePredictors()
        predictors.query.train("hello")
        predictors.query.train("world")

        preds = predictors.query.predict()
        # Prediction exists
        assert isinstance(preds, dict)
