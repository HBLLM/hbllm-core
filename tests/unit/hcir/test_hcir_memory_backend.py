"""Tests for Phase 3: HCIR Memory Backend with 5-phase migration."""

from __future__ import annotations

import pytest

from hbllm.hcir.adapters.hcir_memory_backend import HCIRMemoryBackend, MigrationPhase
from hbllm.hcir.graph import (
    BeliefNode,
    ConceptNode,
    EpisodeNode,
    HCIREdgeType,
    SkillNode,
)
from hbllm.hcir.workspace_tiers import TieredWorkspace

# ═══════════════════════════════════════════════════════════════════════════
# Migration Phase Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMigrationPhase:
    """Verify migration phase semantics."""

    def test_phase_values(self) -> None:
        assert MigrationPhase.READ_THROUGH == "read_through"
        assert MigrationPhase.DUAL_WRITE == "dual_write"
        assert MigrationPhase.SHADOW_READ == "shadow_read"
        assert MigrationPhase.HCIR_PRIMARY == "hcir_primary"
        assert MigrationPhase.LEGACY_REMOVED == "legacy_removed"

    def test_is_hcir_authoritative(self) -> None:
        ws = TieredWorkspace()
        backend = HCIRMemoryBackend(ws, MigrationPhase.READ_THROUGH)
        assert not backend.is_hcir_authoritative

        backend.migration_phase = MigrationPhase.DUAL_WRITE
        assert not backend.is_hcir_authoritative

        backend.migration_phase = MigrationPhase.HCIR_PRIMARY
        assert backend.is_hcir_authoritative

        backend.migration_phase = MigrationPhase.LEGACY_REMOVED
        assert backend.is_hcir_authoritative


# ═══════════════════════════════════════════════════════════════════════════
# Episodic Memory Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEpisodicMemory:
    """Verify episodic memory CRUD through HCIR."""

    def setup_method(self) -> None:
        self.workspace = TieredWorkspace()
        self.backend = HCIRMemoryBackend(self.workspace)

    @pytest.mark.asyncio
    async def test_store_episode(self) -> None:
        entry_id = await self.backend.store_episode(
            summary="User asked about weather",
            outcome="Provided forecast",
            reward=0.9,
            tenant_id="tenant_1",
        )
        assert entry_id.startswith("ep_")

        # Verify node exists in persistent workspace
        node = self.workspace.persistent.get_node(entry_id)
        assert node is not None
        assert isinstance(node, EpisodeNode)
        assert node.summary == "User asked about weather"
        assert node.outcome == "Provided forecast"
        assert node.reward == 0.9

    @pytest.mark.asyncio
    async def test_recall_episodes(self) -> None:
        await self.backend.store_episode(summary="Weather discussion", tenant_id="t1")
        await self.backend.store_episode(summary="Music recommendation", tenant_id="t1")
        await self.backend.store_episode(summary="Weather forecast", tenant_id="t1")

        results = await self.backend.recall_episodes(tenant_id="t1")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_recall_episodes_tenant_isolation(self) -> None:
        await self.backend.store_episode(summary="Tenant 1 memory", tenant_id="t1")
        await self.backend.store_episode(summary="Tenant 2 memory", tenant_id="t2")

        results = await self.backend.recall_episodes(tenant_id="t1")
        assert len(results) == 1
        assert results[0]["summary"] == "Tenant 1 memory"


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Memory Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSemanticMemory:
    """Verify semantic memory through HCIR."""

    def setup_method(self) -> None:
        self.workspace = TieredWorkspace()
        self.backend = HCIRMemoryBackend(self.workspace)

    @pytest.mark.asyncio
    async def test_store_concept(self) -> None:
        entry_id = await self.backend.store_concept(
            label="Python",
            definition="A programming language",
            domain="technology",
        )
        assert entry_id.startswith("sem_")

        node = self.workspace.persistent.get_node(entry_id)
        assert isinstance(node, ConceptNode)
        assert node.label == "Python"
        assert node.definition == "A programming language"
        assert node.domain == "technology"

    @pytest.mark.asyncio
    async def test_recall_concepts(self) -> None:
        await self.backend.store_concept(label="Python", definition="A language")
        await self.backend.store_concept(label="Rust", definition="A systems language")

        results = await self.backend.recall_concepts()
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Procedural Memory Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProceduralMemory:
    """Verify procedural memory through HCIR."""

    def setup_method(self) -> None:
        self.workspace = TieredWorkspace()
        self.backend = HCIRMemoryBackend(self.workspace)

    @pytest.mark.asyncio
    async def test_store_skill(self) -> None:
        entry_id = await self.backend.store_skill(
            skill_name="weather_lookup",
            description="Query weather API",
            success_rate=0.95,
        )
        assert entry_id.startswith("skill_")

        node = self.workspace.persistent.get_node(entry_id)
        assert isinstance(node, SkillNode)
        assert node.skill_name == "weather_lookup"
        assert node.success_rate == 0.95

    @pytest.mark.asyncio
    async def test_recall_skills(self) -> None:
        await self.backend.store_skill(skill_name="weather_lookup")
        await self.backend.store_skill(skill_name="calendar_check")

        results = await self.backend.recall_skills()
        assert len(results) == 2
        skill_names = {r["skill_name"] for r in results}
        assert "weather_lookup" in skill_names
        assert "calendar_check" in skill_names


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Graph (Belief) Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKnowledgeGraph:
    """Verify knowledge graph through HCIR beliefs + edges."""

    def setup_method(self) -> None:
        self.workspace = TieredWorkspace()
        self.backend = HCIRMemoryBackend(self.workspace)

    @pytest.mark.asyncio
    async def test_store_belief(self) -> None:
        entry_id = await self.backend.store_belief(
            claim="The Earth is round",
            belief_type="factual",
            evidence_sources=["direct_observation", "scientific_consensus"],
        )
        assert entry_id.startswith("belief_")

        node = self.workspace.persistent.get_node(entry_id)
        assert isinstance(node, BeliefNode)
        assert node.claim == "The Earth is round"
        assert len(node.evidence_sources) == 2

    @pytest.mark.asyncio
    async def test_link_beliefs(self) -> None:
        b1 = await self.backend.store_belief(claim="Climate change is real")
        b2 = await self.backend.store_belief(claim="Sea levels are rising")

        edge_id = await self.backend.link_beliefs(b1, b2, HCIREdgeType.SUPPORTS)
        assert edge_id.startswith("e_")

        edge = self.workspace.persistent.get_edge(edge_id)
        assert edge is not None
        assert b1 in edge.sources
        assert b2 in edge.targets


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Memory Query Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossMemoryQueries:
    """Verify cross-memory-type queries — the primary HCIR advantage."""

    def setup_method(self) -> None:
        self.workspace = TieredWorkspace()
        self.backend = HCIRMemoryBackend(self.workspace)

    @pytest.mark.asyncio
    async def test_search_across_memory_types(self) -> None:
        await self.backend.store_episode(summary="Weather discussion", tenant_id="t1")
        await self.backend.store_concept(
            label="Weather", definition="Atmospheric state", tenant_id="t1"
        )
        await self.backend.store_skill(skill_name="weather_lookup", tenant_id="t1")
        await self.backend.store_belief(claim="Weather affects mood", tenant_id="t1")

        results = await self.backend.search_across_memory_types(query="weather", tenant_id="t1")
        # Should find entries across multiple memory types
        assert len(results) >= 1  # At least some should match text filter
        memory_types = {r.get("memory_type") for r in results}
        # At least one type should be found
        assert len(memory_types) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Stats Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStats:
    """Verify memory statistics."""

    def setup_method(self) -> None:
        self.workspace = TieredWorkspace()
        self.backend = HCIRMemoryBackend(self.workspace)

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        await self.backend.store_episode(summary="Test episode")
        await self.backend.store_concept(label="Test concept")
        await self.backend.store_skill(skill_name="test_skill")

        stats = await self.backend.stats()
        assert stats["backend"] == "hcir"
        assert stats["migration_phase"] == "read_through"
        assert stats["episodic_count"] == 1
        assert stats["concept_count"] == 1
        assert stats["skill_count"] == 1

    @pytest.mark.asyncio
    async def test_divergence_tracking(self) -> None:
        self.backend.record_divergence("episodic", [1, 2], [1, 2, 3])
        assert self.backend.divergence_count == 1

        self.backend.record_divergence("semantic", [], [1])
        assert self.backend.divergence_count == 2
