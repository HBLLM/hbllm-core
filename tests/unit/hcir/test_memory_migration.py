"""Tests for the 5-phase memory migration proxy.

Validates that each migration phase routes store/recall operations
correctly between legacy and HCIR backends.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import TestCase

from hbllm.hcir.adapters.hcir_memory_backend import HCIRMemoryBackend, MigrationPhase
from hbllm.hcir.adapters.memory_migration_proxy import MemoryMigrationProxy
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.memory.interface import MemoryType

# ═══════════════════════════════════════════════════════════════════════════
# Stub Legacy MemoryNode
# ═══════════════════════════════════════════════════════════════════════════


class StubLegacyMemory:
    """Minimal stub replicating MemoryNode's UnifiedMemoryInterface."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
            "value": [],
            "knowledge_graph": [],
        }
        self.store_calls: int = 0
        self.retrieve_calls: int = 0

    async def store(self, memory_type: MemoryType, data: Any, **kwargs: Any) -> str:
        self.store_calls += 1
        key = memory_type.value
        entry = {"content": str(data), **kwargs}
        self._store[key].append(entry)
        return f"legacy_{key}_{len(self._store[key])}"

    async def retrieve(self, memory_type: MemoryType, query: Any, **kwargs: Any) -> list[Any]:
        self.retrieve_calls += 1
        key = memory_type.value
        if not query:
            return self._store.get(key, [])
        return [
            e
            for e in self._store.get(key, [])
            if query.lower() in str(e.get("content", "")).lower()
        ]


def _run(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: READ_THROUGH
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase1ReadThrough(TestCase):
    """Phase 1: Legacy authoritative, HCIR warmed from recall results."""

    def setUp(self) -> None:
        self.legacy = StubLegacyMemory()
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.READ_THROUGH)
        self.proxy = MemoryMigrationProxy(
            legacy=self.legacy,
            hcir=self.hcir,
            phase=MigrationPhase.READ_THROUGH,
        )

    def test_store_goes_to_legacy_only(self) -> None:
        result = _run(self.proxy.store(MemoryType.EPISODIC, "Hello world"))
        assert result.startswith("legacy_")
        assert self.legacy.store_calls == 1

    def test_recall_comes_from_legacy(self) -> None:
        _run(self.legacy.store(MemoryType.EPISODIC, "weather forecast"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "weather"))
        assert len(results) == 1
        assert self.legacy.retrieve_calls == 1

    def test_hcir_not_written_on_store(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "test data"))
        hcir_results = _run(self.hcir.recall_episodes())
        assert len(hcir_results) == 0

    def test_stats_show_phase(self) -> None:
        assert self.proxy.stats["phase"] == "read_through"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: DUAL_WRITE
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase2DualWrite(TestCase):
    """Phase 2: Both backends written. Legacy is authoritative for reads."""

    def setUp(self) -> None:
        self.legacy = StubLegacyMemory()
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.DUAL_WRITE)
        self.proxy = MemoryMigrationProxy(
            legacy=self.legacy,
            hcir=self.hcir,
            phase=MigrationPhase.DUAL_WRITE,
        )

    def test_store_writes_to_both(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "dual write test"))
        assert self.legacy.store_calls == 1
        # HCIR should also have the episode
        hcir_results = _run(self.hcir.recall_episodes())
        assert len(hcir_results) == 1

    def test_store_semantic_writes_to_both(self) -> None:
        _run(self.proxy.store(MemoryType.SEMANTIC, "quantum computing"))
        assert self.legacy.store_calls == 1
        hcir_results = _run(self.hcir.recall_concepts())
        assert len(hcir_results) == 1

    def test_store_procedural_writes_to_both(self) -> None:
        _run(self.proxy.store(MemoryType.PROCEDURAL, "search_web", name="web_search"))
        assert self.legacy.store_calls == 1
        hcir_results = _run(self.hcir.recall_skills())
        assert len(hcir_results) == 1

    def test_recall_from_legacy(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "weather data"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "weather"))
        assert len(results) == 1  # From legacy

    def test_stats_count_stores(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "a"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "b"))
        assert self.proxy.stats["stores"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: SHADOW_READ
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase3ShadowRead(TestCase):
    """Phase 3: Both read in parallel. Legacy returned. Divergence tracked."""

    def setUp(self) -> None:
        self.legacy = StubLegacyMemory()
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.SHADOW_READ)
        self.proxy = MemoryMigrationProxy(
            legacy=self.legacy,
            hcir=self.hcir,
            phase=MigrationPhase.SHADOW_READ,
        )

    def test_store_writes_both(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "shadow test"))
        assert self.legacy.store_calls == 1
        hcir_results = _run(self.hcir.recall_episodes())
        assert len(hcir_results) == 1

    def test_recall_returns_legacy_results(self) -> None:
        _run(self.legacy.store(MemoryType.EPISODIC, "shadow data"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "shadow"))
        assert len(results) == 1

    def test_divergence_detected(self) -> None:
        # Only legacy has data — HCIR is empty → divergence
        _run(self.legacy.store(MemoryType.EPISODIC, "only in legacy"))
        _run(self.proxy.recall(MemoryType.EPISODIC))
        assert self.proxy.divergence_count >= 1

    def test_no_divergence_when_aligned(self) -> None:
        # Both have same number of entries → no divergence
        _run(self.proxy.store(MemoryType.EPISODIC, "aligned data"))
        # Now both have 1 entry — query with no filter
        _results = _run(self.proxy.recall(MemoryType.EPISODIC))
        # Legacy has 1 from store, HCIR has 1 from store → aligned
        assert self.proxy.divergence_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: HCIR_PRIMARY
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase4HCIRPrimary(TestCase):
    """Phase 4: HCIR authoritative. Legacy receives writes for rollback."""

    def setUp(self) -> None:
        self.legacy = StubLegacyMemory()
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.HCIR_PRIMARY)
        self.proxy = MemoryMigrationProxy(
            legacy=self.legacy,
            hcir=self.hcir,
            phase=MigrationPhase.HCIR_PRIMARY,
        )

    def test_store_hcir_authoritative(self) -> None:
        result = _run(self.proxy.store(MemoryType.EPISODIC, "hcir primary test"))
        # HCIR result is returned (starts with ep_)
        assert result.startswith("ep_")
        # Legacy also gets a rollback write
        assert self.legacy.store_calls == 1

    def test_recall_from_hcir(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "only hcir reads"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC))
        assert len(results) >= 1
        # These should be HCIR dicts with 'id', 'summary' keys
        assert "id" in results[0]

    def test_is_hcir_authoritative(self) -> None:
        assert self.hcir.is_hcir_authoritative is True

    def test_cross_memory_search(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "weather data"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "weather concepts"))
        results = _run(self.proxy.search_all("weather"))
        assert len(results) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: LEGACY_REMOVED
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase5LegacyRemoved(TestCase):
    """Phase 5: HCIR only. Legacy is not called."""

    def setUp(self) -> None:
        self.legacy = StubLegacyMemory()
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=self.legacy,
            hcir=self.hcir,
            phase=MigrationPhase.LEGACY_REMOVED,
        )

    def test_store_hcir_only(self) -> None:
        result = _run(self.proxy.store(MemoryType.EPISODIC, "hcir only"))
        assert result.startswith("ep_")
        # Legacy is NOT called
        assert self.legacy.store_calls == 0

    def test_recall_hcir_only(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "final phase"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC))
        assert len(results) == 1
        assert self.legacy.retrieve_calls == 0

    def test_legacy_completely_unused(self) -> None:
        _run(self.proxy.store(MemoryType.SEMANTIC, "concept"))
        _run(self.proxy.store(MemoryType.PROCEDURAL, "skill", name="test_skill"))
        _run(self.proxy.recall(MemoryType.SEMANTIC))
        _run(self.proxy.recall(MemoryType.PROCEDURAL))
        assert self.legacy.store_calls == 0
        assert self.legacy.retrieve_calls == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase Transitions
# ═══════════════════════════════════════════════════════════════════════════


class TestPhaseTransitions(TestCase):
    """Test advancing through all 5 phases."""

    def setUp(self) -> None:
        self.legacy = StubLegacyMemory()
        self.hcir = HCIRMemoryBackend(TieredWorkspace())
        self.proxy = MemoryMigrationProxy(
            legacy=self.legacy,
            hcir=self.hcir,
        )

    def test_starts_at_read_through(self) -> None:
        assert self.proxy.phase == MigrationPhase.READ_THROUGH

    def test_advance_through_all_phases(self) -> None:
        phases = [
            MigrationPhase.READ_THROUGH,
            MigrationPhase.DUAL_WRITE,
            MigrationPhase.SHADOW_READ,
            MigrationPhase.HCIR_PRIMARY,
            MigrationPhase.LEGACY_REMOVED,
        ]
        for expected in phases:
            assert self.proxy.phase == expected
            self.proxy.advance_phase()

    def test_advance_past_last_stays(self) -> None:
        for _ in range(10):
            self.proxy.advance_phase()
        assert self.proxy.phase == MigrationPhase.LEGACY_REMOVED

    def test_set_phase_directly(self) -> None:
        self.proxy.phase = MigrationPhase.HCIR_PRIMARY
        assert self.proxy.phase == MigrationPhase.HCIR_PRIMARY
        assert self.hcir.migration_phase == MigrationPhase.HCIR_PRIMARY

    def test_data_survives_phase_transition(self) -> None:
        # Store in dual-write
        self.proxy.phase = MigrationPhase.DUAL_WRITE
        _run(self.proxy.store(MemoryType.EPISODIC, "transition test"))

        # Advance to HCIR primary
        self.proxy.phase = MigrationPhase.HCIR_PRIMARY
        results = _run(self.proxy.recall(MemoryType.EPISODIC))
        assert len(results) >= 1

    def test_stats_track_across_phases(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "a"))
        self.proxy.advance_phase()
        _run(self.proxy.store(MemoryType.EPISODIC, "b"))
        assert self.proxy.stats["stores"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# All Memory Types
# ═══════════════════════════════════════════════════════════════════════════


class TestAllMemoryTypes(TestCase):
    """Verify all memory types work through the proxy at HCIR_PRIMARY."""

    def setUp(self) -> None:
        self.legacy = StubLegacyMemory()
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.HCIR_PRIMARY)
        self.proxy = MemoryMigrationProxy(
            legacy=self.legacy,
            hcir=self.hcir,
            phase=MigrationPhase.HCIR_PRIMARY,
        )

    def test_episodic_roundtrip(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "episode test"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC))
        assert len(results) == 1
        assert "episode test" in results[0].get("summary", "")

    def test_semantic_roundtrip(self) -> None:
        _run(self.proxy.store(MemoryType.SEMANTIC, "quantum physics"))
        results = _run(self.proxy.recall(MemoryType.SEMANTIC))
        assert len(results) == 1

    def test_procedural_roundtrip(self) -> None:
        _run(self.proxy.store(MemoryType.PROCEDURAL, "web search", name="search_web"))
        results = _run(self.proxy.recall(MemoryType.PROCEDURAL))
        assert len(results) == 1

    def test_value_store(self) -> None:
        result = _run(self.proxy.store(MemoryType.VALUE, 0.8, topic="helpfulness"))
        assert result.startswith("val_")

    def test_knowledge_graph_store(self) -> None:
        result = _run(self.proxy.store(MemoryType.KNOWLEDGE_GRAPH, "The sky is blue"))
        assert result.startswith("belief_")
