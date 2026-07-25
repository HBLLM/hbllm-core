"""Phase 4 (HCIR_PRIMARY) integration tests.

Validates that HCIR is the authoritative memory backend with legacy
receiving fire-and-forget writes for rollback safety. Tests the full
store → recall → cross-search → stats lifecycle.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import TestCase

from hbllm.hcir.adapters.hcir_memory_backend import HCIRMemoryBackend, MigrationPhase
from hbllm.hcir.adapters.memory_migration_proxy import MemoryMigrationProxy
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.memory.interface import MemoryType


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class StubLegacy:
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
        self._store[key].append({"content": str(data), **kwargs})
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


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Full Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase4Integration(TestCase):
    """Full Phase 4 integration — HCIR is authoritative."""

    def setUp(self) -> None:
        self.legacy = StubLegacy()
        self.tiered = TieredWorkspace()
        self.hcir = HCIRMemoryBackend(self.tiered, MigrationPhase.HCIR_PRIMARY)
        self.proxy = MemoryMigrationProxy(
            legacy=self.legacy,
            hcir=self.hcir,
            phase=MigrationPhase.HCIR_PRIMARY,
        )

    # ── Store All Types ───────────────────────────────────────────────

    def test_store_episodic(self) -> None:
        result = _run(self.proxy.store(MemoryType.EPISODIC, "User asked about weather"))
        assert result.startswith("ep_")
        # Legacy also receives rollback write
        assert self.legacy.store_calls == 1

    def test_store_semantic(self) -> None:
        result = _run(self.proxy.store(MemoryType.SEMANTIC, "quantum computing"))
        assert result.startswith("sem_")
        assert self.legacy.store_calls == 1

    def test_store_procedural(self) -> None:
        result = _run(
            self.proxy.store(MemoryType.PROCEDURAL, "web search steps", name="search_web")
        )
        assert result.startswith("skill_")
        assert self.legacy.store_calls == 1

    def test_store_value(self) -> None:
        result = _run(self.proxy.store(MemoryType.VALUE, 0.9, topic="helpfulness"))
        assert result.startswith("val_")
        assert self.legacy.store_calls == 1

    def test_store_knowledge_graph(self) -> None:
        result = _run(self.proxy.store(MemoryType.KNOWLEDGE_GRAPH, "The sky is blue"))
        assert result.startswith("belief_")
        assert self.legacy.store_calls == 1

    # ── Recall ────────────────────────────────────────────────────────

    def test_recall_episodic_from_hcir(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "weather forecast for today"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "weather"))
        assert len(results) == 1
        assert "weather" in results[0].get("summary", "").lower()
        # Legacy is NOT read
        assert self.legacy.retrieve_calls == 0

    def test_recall_semantic_from_hcir(self) -> None:
        _run(self.proxy.store(MemoryType.SEMANTIC, "machine learning"))
        results = _run(self.proxy.recall(MemoryType.SEMANTIC, "machine"))
        assert len(results) == 1
        # Legacy is NOT read
        assert self.legacy.retrieve_calls == 0

    def test_recall_procedural_from_hcir(self) -> None:
        _run(self.proxy.store(MemoryType.PROCEDURAL, "API call", name="call_api"))
        results = _run(self.proxy.recall(MemoryType.PROCEDURAL, "API"))
        assert len(results) == 1
        assert self.legacy.retrieve_calls == 0

    def test_recall_empty(self) -> None:
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "nonexistent"))
        assert len(results) == 0

    # ── Cross-Memory Search ───────────────────────────────────────────

    def test_cross_memory_search(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "weather data collected"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "weather patterns"))
        _run(self.proxy.store(MemoryType.PROCEDURAL, "weather API", name="get_weather"))
        _run(self.proxy.store(MemoryType.KNOWLEDGE_GRAPH, "weather affects mood"))

        results = _run(self.proxy.search_all("weather"))
        assert len(results) >= 4
        memory_types_found = {r.get("memory_type") for r in results}
        assert "episodic" in memory_types_found
        assert "semantic" in memory_types_found
        assert "procedural" in memory_types_found
        assert "knowledge_graph" in memory_types_found

    def test_cross_memory_search_with_tenant(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "tenant A data", tenant_id="t_a"))
        _run(self.proxy.store(MemoryType.EPISODIC, "tenant B data", tenant_id="t_b"))

        results_a = _run(self.proxy.search_all("data", tenant_id="t_a"))
        results_b = _run(self.proxy.search_all("data", tenant_id="t_b"))
        assert len(results_a) >= 1
        assert len(results_b) >= 1

    # ── HCIR Authoritative Properties ─────────────────────────────────

    def test_hcir_is_authoritative(self) -> None:
        assert self.hcir.is_hcir_authoritative is True

    def test_legacy_not_read(self) -> None:
        """Legacy should never be called for reads in Phase 4."""
        _run(self.proxy.store(MemoryType.EPISODIC, "test"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "test"))
        _run(self.proxy.recall(MemoryType.EPISODIC))
        _run(self.proxy.recall(MemoryType.SEMANTIC))
        assert self.legacy.retrieve_calls == 0

    def test_legacy_receives_rollback_writes(self) -> None:
        """Legacy receives all writes for rollback safety."""
        _run(self.proxy.store(MemoryType.EPISODIC, "a"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "b"))
        _run(self.proxy.store(MemoryType.PROCEDURAL, "c", name="d"))
        assert self.legacy.store_calls == 3

    # ── Bulk Operations ───────────────────────────────────────────────

    def test_bulk_store_and_recall(self) -> None:
        """Store 50 episodes, recall should return up to limit."""
        for i in range(50):
            _run(self.proxy.store(MemoryType.EPISODIC, f"episode {i}"))
        results = _run(self.proxy.recall(MemoryType.EPISODIC, limit=10))
        assert len(results) == 10

    def test_bulk_mixed_types(self) -> None:
        """Store across all types, verify counts."""
        for i in range(10):
            _run(self.proxy.store(MemoryType.EPISODIC, f"ep {i}"))
            _run(self.proxy.store(MemoryType.SEMANTIC, f"sem {i}"))
            _run(self.proxy.store(MemoryType.PROCEDURAL, f"skill {i}", name=f"s{i}"))

        ep = _run(self.proxy.recall(MemoryType.EPISODIC, limit=100))
        sem = _run(self.proxy.recall(MemoryType.SEMANTIC, limit=100))
        proc = _run(self.proxy.recall(MemoryType.PROCEDURAL, limit=100))
        assert len(ep) == 10
        assert len(sem) == 10
        assert len(proc) == 10

    # ── Stats ─────────────────────────────────────────────────────────

    def test_proxy_stats(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "a"))
        _run(self.proxy.recall(MemoryType.EPISODIC))
        stats = self.proxy.stats
        assert stats["phase"] == "hcir_primary"
        assert stats["stores"] == 1
        assert stats["recalls"] == 1
        assert stats["divergences"] == 0

    def test_hcir_backend_stats(self) -> None:
        _run(self.proxy.store(MemoryType.EPISODIC, "ep"))
        _run(self.proxy.store(MemoryType.SEMANTIC, "sem"))
        _run(self.proxy.store(MemoryType.PROCEDURAL, "proc", name="p"))
        _run(self.proxy.store(MemoryType.KNOWLEDGE_GRAPH, "belief"))
        _run(self.proxy.store(MemoryType.VALUE, 0.5, topic="test"))

        stats = _run(self.hcir.stats())
        assert stats["backend"] == "hcir"
        assert stats["migration_phase"] == "hcir_primary"
        assert stats["episodic_count"] == 1
        assert stats["concept_count"] == 1
        assert stats["skill_count"] == 1
        assert stats["belief_count"] == 1
        assert stats["value_count"] == 1

    # ── Data Integrity ────────────────────────────────────────────────

    def test_episodic_content_preserved(self) -> None:
        """Verify exact content roundtrip."""
        _run(
            self.proxy.store(
                MemoryType.EPISODIC,
                "User asked: What is the capital of France? → Paris",
                session_id="sess_42",
            )
        )
        results = _run(self.proxy.recall(MemoryType.EPISODIC, "France"))
        assert len(results) == 1
        assert "France" in results[0]["summary"]

    def test_semantic_content_preserved(self) -> None:
        _run(self.proxy.store(MemoryType.SEMANTIC, "Photosynthesis"))
        results = _run(self.proxy.recall(MemoryType.SEMANTIC, "Photosynthesis"))
        assert len(results) == 1
        assert "Photosynthesis" in results[0].get("label", "")

    def test_tenant_isolation(self) -> None:
        """Data from tenant A not visible to tenant B."""
        _run(self.proxy.store(MemoryType.EPISODIC, "secret A", tenant_id="alice"))
        _run(self.proxy.store(MemoryType.EPISODIC, "secret B", tenant_id="bob"))

        alice_results = _run(self.proxy.recall(MemoryType.EPISODIC, "secret", tenant_id="alice"))
        bob_results = _run(self.proxy.recall(MemoryType.EPISODIC, "secret", tenant_id="bob"))

        assert len(alice_results) == 1
        assert len(bob_results) == 1
        assert "A" in alice_results[0]["summary"]
        assert "B" in bob_results[0]["summary"]
