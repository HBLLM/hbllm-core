"""Phase 5 (LEGACY_REMOVED) integration tests.

Validates that HCIR is the sole memory backend. Legacy MemoryNode
is NOT instantiated, NOT called, and NOT required. All memory
operations go exclusively through HCIRMemoryBackend.
"""

from __future__ import annotations

import asyncio
from unittest import TestCase

from hbllm.hcir.adapters.hcir_memory_backend import HCIRMemoryBackend, MigrationPhase
from hbllm.hcir.adapters.memory_migration_proxy import MemoryMigrationProxy
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.memory.interface import MemoryType


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: No Legacy — HCIR Only
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase5LegacyNone(TestCase):
    """Proxy works with legacy=None (Phase 5 default)."""

    def setUp(self) -> None:
        self.tiered = TieredWorkspace()
        self.hcir = HCIRMemoryBackend(self.tiered, MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None,
            hcir=self.hcir,
            phase=MigrationPhase.LEGACY_REMOVED,
        )

    def test_legacy_is_none(self) -> None:
        assert self.proxy._legacy is None

    def test_phase_is_legacy_removed(self) -> None:
        assert self.proxy.phase == MigrationPhase.LEGACY_REMOVED

    def test_hcir_is_authoritative(self) -> None:
        assert self.hcir.is_hcir_authoritative is True


class TestPhase5EpisodicMemory(TestCase):
    """Episodic memory works without legacy."""

    def setUp(self) -> None:
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None, hcir=self.hcir, phase=MigrationPhase.LEGACY_REMOVED
        )

    def test_store(self) -> None:
        result = _run(self.proxy.store(MemoryType.EPISODIC, "User asked about AI"))
        assert result.startswith("ep_")

    def test_recall(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "weather forecast"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "weather"))
        assert len(results) == 1
        assert "weather" in results[0]["summary"].lower()

    def test_recall_empty(self) -> None:
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "nonexistent"))
        assert results == []

    def test_recall_all(self) -> None:
        for i in range(5):
            _run(self.proxy.store(MemoryType.EPISODIC, f"episode {i}"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC))
        assert len(results) == 5

    def test_recall_with_limit(self) -> None:
        for i in range(20):
            _run(self.proxy.store(MemoryType.EPISODIC, f"episode {i}"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC, limit=5))
        assert len(results) == 5

    def test_content_preserved(self) -> None:
        _run(
            self.proxy.store(
                MemoryType.EPISODIC,
                "User asked: What is the capital of France? → Paris",
                session_id="sess_1",
            )
        )
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "France"))
        assert "France" in results[0]["summary"]


class TestPhase5SemanticMemory(TestCase):
    """Semantic memory works without legacy."""

    def setUp(self) -> None:
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None, hcir=self.hcir, phase=MigrationPhase.LEGACY_REMOVED
        )

    def test_store(self) -> None:
        result = _run(self.proxy.store(MemoryType.SEMANTIC, "quantum computing"))
        assert result.startswith("sem_")

    def test_recall(self) -> None:
        _run(self.proxy.store(MemoryType.SEMANTIC, "machine learning"))
        results = _run(self.proxy.recall(MemoryType.SEMANTIC, "machine"))
        assert len(results) == 1

    def test_multiple_concepts(self) -> None:
        _run(self.proxy.store(MemoryType.SEMANTIC, "neural networks"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "deep learning"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "reinforcement learning"))
        results = _run(self.proxy.recall(MemoryType.SEMANTIC, "learning"))
        assert len(results) == 2  # deep learning + reinforcement learning


class TestPhase5ProceduralMemory(TestCase):
    """Procedural memory works without legacy."""

    def setUp(self) -> None:
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None, hcir=self.hcir, phase=MigrationPhase.LEGACY_REMOVED
        )

    def test_store(self) -> None:
        result = _run(self.proxy.store(MemoryType.PROCEDURAL, "search the web", name="web_search"))
        assert result.startswith("skill_")

    def test_recall(self) -> None:
        _run(self.proxy.store(MemoryType.PROCEDURAL, "API call", name="call_api"))
        results = _run(self.proxy.recall(MemoryType.PROCEDURAL, "API"))
        assert len(results) == 1
        assert "API" in results[0].get("description", "") or "API" in results[0].get(
            "skill_name", ""
        )


class TestPhase5ValueMemory(TestCase):
    """Value memory works without legacy."""

    def setUp(self) -> None:
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None, hcir=self.hcir, phase=MigrationPhase.LEGACY_REMOVED
        )

    def test_store(self) -> None:
        result = _run(self.proxy.store(MemoryType.VALUE, 0.95, topic="helpfulness"))
        assert result.startswith("val_")


class TestPhase5KnowledgeGraph(TestCase):
    """Knowledge graph works without legacy."""

    def setUp(self) -> None:
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None, hcir=self.hcir, phase=MigrationPhase.LEGACY_REMOVED
        )

    def test_store_belief(self) -> None:
        result = _run(self.proxy.store(MemoryType.KNOWLEDGE_GRAPH, "The sky is blue"))
        assert result.startswith("belief_")

    def test_link_beliefs(self) -> None:
        b1 = _run(self.hcir.store_belief("Rain causes wet ground"))
        b2 = _run(self.hcir.store_belief("Wet ground makes driving dangerous"))
        edge = _run(self.hcir.link_beliefs(b1, b2))
        assert edge.startswith("e_")


class TestPhase5CrossMemorySearch(TestCase):
    """Cross-memory-type search works without legacy."""

    def setUp(self) -> None:
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None, hcir=self.hcir, phase=MigrationPhase.LEGACY_REMOVED
        )

    def test_search_across_types(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "weather data collected"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "weather patterns"))
        _run(self.proxy.store(MemoryType.PROCEDURAL, "weather API", name="get_weather"))
        _run(self.proxy.store(MemoryType.KNOWLEDGE_GRAPH, "weather affects mood"))

        results = _run(self.proxy.search_all("weather"))
        assert len(results) >= 4

        types = {r["memory_type"] for r in results}
        assert "episodic" in types
        assert "semantic" in types
        assert "procedural" in types
        assert "knowledge_graph" in types

    def test_search_with_limit(self) -> None:
        for i in range(20):
            _run(self.proxy.store(MemoryType.EPISODIC, f"episode {i}"))
        results = _run(self.proxy.search_all("episode", limit=5))
        assert len(results) == 5


class TestPhase5TenantIsolation(TestCase):
    """Multi-tenant isolation works without legacy."""

    def setUp(self) -> None:
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None, hcir=self.hcir, phase=MigrationPhase.LEGACY_REMOVED
        )

    def test_tenant_isolation_episodic(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "alice secret", tenant_id="alice"))
        _run(self.proxy.store(MemoryType.EPISODIC, "bob secret", tenant_id="bob"))

        alice = _run(self.proxy.recall(MemoryType.EPISODIC, "secret", tenant_id="alice"))
        bob = _run(self.proxy.recall(MemoryType.EPISODIC, "secret", tenant_id="bob"))

        assert len(alice) == 1
        assert len(bob) == 1
        assert "alice" in alice[0]["summary"]
        assert "bob" in bob[0]["summary"]

    def test_tenant_isolation_cross_search(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "t1 data", tenant_id="t1"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "t2 concept", tenant_id="t2"))

        t1 = _run(self.proxy.search_all("data", tenant_id="t1"))
        t2 = _run(self.proxy.search_all("concept", tenant_id="t2"))
        assert len(t1) >= 1
        assert len(t2) >= 1


class TestPhase5Stats(TestCase):
    """Stats and monitoring work without legacy."""

    def setUp(self) -> None:
        self.hcir = HCIRMemoryBackend(TieredWorkspace(), MigrationPhase.LEGACY_REMOVED)
        self.proxy = MemoryMigrationProxy(
            legacy=None, hcir=self.hcir, phase=MigrationPhase.LEGACY_REMOVED
        )

    def test_proxy_stats(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "a"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "b"))
        _run(self.proxy.recall(MemoryType.EPISODIC))

        stats = self.proxy.stats
        assert stats["phase"] == "legacy_removed"
        assert stats["stores"] == 2
        assert stats["recalls"] == 1
        assert stats["divergences"] == 0

    def test_hcir_stats(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "ep"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "sem"))
        _run(self.proxy.store(MemoryType.PROCEDURAL, "proc", name="p"))
        _run(self.proxy.store(MemoryType.VALUE, 0.5, topic="test"))
        _run(self.proxy.store(MemoryType.KNOWLEDGE_GRAPH, "belief"))

        stats = _run(self.hcir.stats())
        assert stats["backend"] == "hcir"
        assert stats["migration_phase"] == "legacy_removed"
        assert stats["episodic_count"] == 1
        assert stats["concept_count"] == 1
        assert stats["skill_count"] == 1
        assert stats["value_count"] == 1
        assert stats["belief_count"] == 1
        assert stats["divergence_count"] == 0
