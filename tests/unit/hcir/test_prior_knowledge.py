"""Tests for prior-knowledge query answers and memory command capabilities.

Verifies:
  1. "What is X?" → answers from LexicalOntologyStore definitions
  2. "Remember that X" → stores into persistent memory via memory.store
  3. "Recall X" → retrieves from persistent memory via memory.recall
  4. Definition verbalization in EnglishRealizer
  5. Memory verb → capability mappings in VERB_CAPABILITY_MAP
"""

from __future__ import annotations

import asyncio

import pytest

from hbllm.hcir.graph import CognitiveGraph
from hbllm.hcir.kernel.command_dispatcher import VERB_CAPABILITY_MAP
from hbllm.hcir.language.core.epistemic_policy import (
    CognitiveEpistemicState,
    EpistemicRealizationPolicy,
)
from hbllm.hcir.language.core.gateway import HCIRGateway
from hbllm.hcir.language.english.realizer import EnglishRealizer

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


class FakeOntologyEntry:
    """Mimics StoredLexicalEntry for testing without real SQLite."""

    def __init__(self, lemma: str, definition: str, pos: str = "noun", source: str = "test"):
        self.lemma = lemma
        self.definition = definition
        self.pos = pos
        self.source = source
        self.language = "en"
        self.hypernyms = ()
        self.synonyms = ()
        self.thematic_roles = ()
        self.domain = "general"


class FakeOntologyStore:
    """In-memory ontology store for unit tests."""

    def __init__(self, entries: dict[str, str] | None = None):
        self._entries = entries or {}

    def lookup(self, lemma: str, *, language: str = "en") -> list[FakeOntologyEntry]:
        defn = self._entries.get(lemma.strip().lower())
        if defn:
            return [FakeOntologyEntry(lemma=lemma.strip().lower(), definition=defn)]
        return []


@pytest.fixture
def ontology_store() -> FakeOntologyStore:
    return FakeOntologyStore(
        {
            "microscope": "an optical instrument used for viewing very small objects",
            "server": "a computer or computer program that manages access to a centralized resource or service in a network",
            "algorithm": "a precise rule (or set of rules) specifying how to solve some problem",
            "elephant": "five-toed pachyderm",
        }
    )


@pytest.fixture
def gateway(ontology_store: FakeOntologyStore) -> HCIRGateway:
    graph = CognitiveGraph()
    return HCIRGateway(
        graph,
        ontology_store=ontology_store,
        tiered_workspace=None,
    )


@pytest.fixture
def realizer() -> EnglishRealizer:
    return EnglishRealizer(EpistemicRealizationPolicy())


# ═══════════════════════════════════════════════════════════════════════════
# A. Prior Knowledge — Gateway epistemic prior lookup
# ═══════════════════════════════════════════════════════════════════════════


class TestGatewayEpistemicPrior:
    """Tests that _lookup_epistemic_prior returns definitions from ontology store."""

    def test_lookup_known_word(self, gateway: HCIRGateway):
        """Ontology lookup for 'microscope' should return a definition."""
        result = gateway._lookup_epistemic_prior("microscope")
        assert result is not None
        assert result.is_known is True
        assert result.target_predicate == "definition"
        assert result.target_subject == "microscope"
        assert "optical instrument" in result.target_object
        assert result.confidence == 0.85

    def test_lookup_unknown_word(self, gateway: HCIRGateway):
        """Unknown word should return None."""
        result = gateway._lookup_epistemic_prior("zyxwvut")
        assert result is None

    def test_lookup_server(self, gateway: HCIRGateway):
        result = gateway._lookup_epistemic_prior("server")
        assert result is not None
        assert "computer" in result.target_object

    def test_lookup_provenance(self, gateway: HCIRGateway):
        """Provenance should indicate ontology source."""
        result = gateway._lookup_epistemic_prior("algorithm")
        assert result is not None
        assert "ontology:" in result.provenance


# ═══════════════════════════════════════════════════════════════════════════
# B. Definition Verbalization — EnglishRealizer
# ═══════════════════════════════════════════════════════════════════════════


class TestDefinitionRealization:
    """Tests that the realizer generates natural definition sentences."""

    def test_certain_definition(self, realizer: EnglishRealizer):
        """High-confidence definition should produce 'A X is Y.'"""
        state = CognitiveEpistemicState(
            target_predicate="definition",
            target_subject="microscope",
            target_object="an optical instrument used for viewing very small objects",
            confidence=0.95,
            is_known=True,
            support_count=1,
        )
        text = realizer.realize(state)
        assert text.startswith("A microscope is")
        assert "optical instrument" in text

    def test_probable_definition(self, realizer: EnglishRealizer):
        """Medium-confidence definition should produce 'generally described as'."""
        state = CognitiveEpistemicState(
            target_predicate="definition",
            target_subject="server",
            target_object="a computer that manages network resources",
            confidence=0.75,
            is_known=True,
            support_count=1,
        )
        text = realizer.realize(state)
        assert "generally described as" in text

    def test_vowel_article(self, realizer: EnglishRealizer):
        """Words starting with vowels should use 'An'."""
        state = CognitiveEpistemicState(
            target_predicate="definition",
            target_subject="elephant",
            target_object="a large mammal",
            confidence=0.95,
            is_known=True,
            support_count=1,
        )
        text = realizer.realize(state)
        assert text.startswith("An elephant is")

    def test_indefinite_article_helper(self, realizer: EnglishRealizer):
        assert realizer._indefinite_article("apple") == "An"
        assert realizer._indefinite_article("banana") == "A"
        assert realizer._indefinite_article("umbrella") == "An"
        assert realizer._indefinite_article("cat") == "A"


# ═══════════════════════════════════════════════════════════════════════════
# C. Memory Verb Mappings
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryVerbMappings:
    """Tests that memory verbs are correctly mapped to capabilities."""

    def test_remember_maps_to_memory_store(self):
        assert VERB_CAPABILITY_MAP["remember"] == "memory.store"

    def test_store_maps_to_memory_store(self):
        assert VERB_CAPABILITY_MAP["store"] == "memory.store"

    def test_memorize_maps_to_memory_store(self):
        assert VERB_CAPABILITY_MAP["memorize"] == "memory.store"

    def test_recall_maps_to_memory_recall(self):
        assert VERB_CAPABILITY_MAP["recall"] == "memory.recall"

    def test_retrieve_maps_to_memory_recall(self):
        assert VERB_CAPABILITY_MAP["retrieve"] == "memory.recall"

    def test_forget_maps_to_memory_forget(self):
        assert VERB_CAPABILITY_MAP["forget"] == "memory.forget"

    def test_ingest_maps_to_knowledge_ingest(self):
        assert VERB_CAPABILITY_MAP["ingest"] == "knowledge.ingest"

    def test_index_maps_to_knowledge_ingest(self):
        assert VERB_CAPABILITY_MAP["index"] == "knowledge.ingest"


# ═══════════════════════════════════════════════════════════════════════════
# D. Memory Capability Executor
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryCapabilityExecutor:
    """Tests the MemoryCapabilityExecutor store/recall/forget operations."""

    def test_memory_store(self):
        from hbllm.hcir.capabilities.memory_capabilities import MemoryCapabilityExecutor
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        tw = TieredWorkspace()
        executor = MemoryCapabilityExecutor(
            default_capability="memory.store",
            tiered_workspace=tw,
        )
        result = asyncio.run(
            executor.execute(
                {"capability": "memory.store", "content": "that the staging IP is 10.0.0.5"}
            )
        )
        assert result["ok"] is True
        assert result["action"] == "stored"
        assert "staging IP is 10.0.0.5" in result["content"]

    def test_memory_recall_empty(self):
        from hbllm.hcir.capabilities.memory_capabilities import MemoryCapabilityExecutor
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        tw = TieredWorkspace()
        executor = MemoryCapabilityExecutor(
            default_capability="memory.recall",
            tiered_workspace=tw,
        )
        result = asyncio.run(
            executor.execute({"capability": "memory.recall", "content": "staging IP"})
        )
        assert result["ok"] is True
        assert result["matches"] == 0

    def test_memory_store_then_recall(self):
        from hbllm.hcir.capabilities.memory_capabilities import MemoryCapabilityExecutor
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        tw = TieredWorkspace()
        store_exec = MemoryCapabilityExecutor(
            default_capability="memory.store",
            tiered_workspace=tw,
        )
        recall_exec = MemoryCapabilityExecutor(
            default_capability="memory.recall",
            tiered_workspace=tw,
        )

        # Store
        store_result = asyncio.run(
            store_exec.execute(
                {"capability": "memory.store", "content": "the staging IP is 10.0.0.5"}
            )
        )
        assert store_result["ok"] is True

        # Recall
        recall_result = asyncio.run(
            recall_exec.execute({"capability": "memory.recall", "content": "staging"})
        )
        assert recall_result["ok"] is True
        assert recall_result["matches"] > 0
        assert (
            "staging" in recall_result["content"].lower() or "10.0.0.5" in recall_result["content"]
        )

    def test_memory_recall_with_ontology_fallback(self):
        from hbllm.hcir.capabilities.memory_capabilities import MemoryCapabilityExecutor
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        tw = TieredWorkspace()
        store = FakeOntologyStore({"python": "a large nonvenomous snake"})
        executor = MemoryCapabilityExecutor(
            default_capability="memory.recall",
            tiered_workspace=tw,
            ontology_store=store,
        )
        result = asyncio.run(executor.execute({"capability": "memory.recall", "content": "python"}))
        assert result["ok"] is True
        assert result["matches"] == 1
        assert "snake" in result["content"].lower()

    def test_memory_store_requires_content(self):
        from hbllm.hcir.capabilities.memory_capabilities import MemoryCapabilityExecutor
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        tw = TieredWorkspace()
        executor = MemoryCapabilityExecutor(
            default_capability="memory.store",
            tiered_workspace=tw,
        )
        result = asyncio.run(executor.execute({"capability": "memory.store", "content": ""}))
        assert result["ok"] is False

    def test_memory_forget(self):
        from hbllm.hcir.capabilities.memory_capabilities import MemoryCapabilityExecutor
        from hbllm.hcir.workspace_tiers import TieredWorkspace

        tw = TieredWorkspace()
        executor = MemoryCapabilityExecutor(
            default_capability="memory.forget",
            tiered_workspace=tw,
        )
        result = asyncio.run(
            executor.execute({"capability": "memory.forget", "content": "nonexistent thing"})
        )
        assert result["ok"] is True
        assert result["action"] == "not_found"


# ═══════════════════════════════════════════════════════════════════════════
# E. MEMORY_CAPABILITIES tuple
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryCapabilitiesTuple:
    def test_memory_capabilities_registered(self):
        from hbllm.hcir.capabilities.memory_capabilities import MEMORY_CAPABILITIES

        assert "memory.store" in MEMORY_CAPABILITIES
        assert "memory.recall" in MEMORY_CAPABILITIES
        assert "memory.forget" in MEMORY_CAPABILITIES
        assert "knowledge.ingest" in MEMORY_CAPABILITIES


# ═══════════════════════════════════════════════════════════════════════════
# F. End-to-End Bootstrap Dialogue Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBootstrapDialoguePriorKnowledge:
    """Verifies end-to-end dialogue processing for definition queries and memory commands."""

    @pytest.fixture(scope="class")
    @classmethod
    def system(cls):
        from hbllm.runtime.bootstrap import build_cognitive_system

        return build_cognitive_system()

    def test_what_is_microscope(self, system):
        res = system.process_dialogue("What is a microscope?")
        assert res.is_success is True
        assert "microscope" in res.response_text.lower()
        assert (
            "magnifier" in res.response_text.lower()
            or "optical" in res.response_text.lower()
            or "instrument" in res.response_text.lower()
        )

    def test_what_is_server(self, system):
        res = system.process_dialogue("What is a server?")
        assert res.is_success is True
        assert "server" in res.response_text.lower()
        assert "computer" in res.response_text.lower()

    def test_what_is_algorithm(self, system):
        res = system.process_dialogue("What is an algorithm?")
        assert res.is_success is True
        assert "algorithm" in res.response_text.lower()
        assert "rule" in res.response_text.lower() or "problem" in res.response_text.lower()

    def test_remember_and_recall_dialogue(self, system):
        # Store
        res_store = system.process_dialogue("Remember that the primary database is PostgreSQL")
        assert res_store.is_success is True
        assert "PostgreSQL" in res_store.response_text

        # Recall
        res_recall = system.process_dialogue("Recall the primary database")
        assert res_recall.is_success is True
        assert "postgresql" in res_recall.response_text.lower()
