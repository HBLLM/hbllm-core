"""Unit tests for Option A: Autonomous Curiosity, Multi-Turn Coreference, and Belief Revision.

Verifies:
  1. Multi-turn coreference & discourse topic resolution ("Explain focus" -> "Why does it happen?").
  2. Autonomous Epistemic Curiosity loop (uncertainty-gated web.search -> ingest -> re-explain).
  3. Structural contradiction detection & confidence-weighted belief revision (CONTRADICTS edge).
  4. End-to-end integration through CognitiveSystem.process_dialogue with mocked web capabilities.
"""

from __future__ import annotations

import asyncio
from typing import Any

from hbllm.hcir.capabilities.belief_revision import BeliefConflictResolver
from hbllm.hcir.capabilities.epistemic_curiosity import EpistemicCuriosityController
from hbllm.hcir.capabilities.memory_capabilities import MemoryCapabilityExecutor
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    ConceptNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
)
from hbllm.hcir.kernel.capability_resolver import CapabilityImplementation, CapabilityResolver
from hbllm.hcir.language.core.reference import ReferenceResolver, is_anaphoric_term
from hbllm.hcir.language.core.semantic_frame import FrameType, ThematicRole
from hbllm.hcir.language.english.parser import EnglishParser
from hbllm.hcir.language.runtime import MultilingualLanguageRuntime
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


class _MockWebSearchExecutor:
    """Deterministic mock web.search executor for offline curiosity tests."""

    def __init__(self, tiered_workspace: TieredWorkspace | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._tiered = tiered_workspace

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(params))
        query = str(params.get("query", ""))
        # Simulate ingesting a causal & taxonomic fact into persistent memory
        if self._tiered is not None and "photosynthesis" in query.lower():
            p_graph = self._tiered.persistent.graph
            p_graph.add_node(
                ConceptNode(
                    id="web_concept_photosynthesis",
                    label="photosynthesis",
                    definition="the biological process by which plants convert sunlight into chemical energy",
                )
            )
            p_graph.add_node(ConceptNode(id="web_concept_sunlight", label="sunlight"))
            p_graph.add_edge(
                HCIREdge(
                    id="web_causes_sunlight__photosynthesis",
                    edge_type=HCIREdgeType.CAUSES,
                    sources=["web_concept_sunlight"],
                    targets=["web_concept_photosynthesis"],
                )
            )
            return {
                "ok": True,
                "propositions_ingested": 1,
                "results": [
                    {
                        "title": "Photosynthesis",
                        "url": "https://en.wikipedia.org/wiki/Photosynthesis",
                        "snippet": "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                    }
                ],
            }
        return {
            "ok": True,
            "propositions_ingested": 0,
            "results": [
                {
                    "title": "Quantum Tunneling",
                    "url": "https://en.wikipedia.org/wiki/Quantum_tunnelling",
                    "snippet": "Quantum tunneling is a quantum mechanical phenomenon where a wave function propagates through a potential barrier.",
                }
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Multi-Turn Coreference & Discourse Topic Resolution Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiTurnCoreference:
    """Verifies conceptual topic tracking and pronoun resolution across turns."""

    def test_reference_resolver_topic_tracking(self) -> None:
        ref = ReferenceResolver()
        assert is_anaphoric_term("it") is True
        assert is_anaphoric_term("they") is True
        assert is_anaphoric_term("photosynthesis") is False

        ref.register_topic("photosynthesis")
        topic = ref.active_topic()
        assert topic is not None
        assert topic.concept_name == "photosynthesis"
        assert ref.resolve_concept("it") == "photosynthesis"
        assert ref.resolve_concept("gravity") == "gravity"

    def test_parser_marks_anaphoric_pronouns_in_why_how(self) -> None:
        parser = EnglishParser()
        frame_why = parser.parse("Why does it happen?")
        assert frame_why.frame_type == FrameType.COMMAND
        assert frame_why.predicate == "why"
        patient = frame_why.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.specifier == "anaphoric"

        frame_how = parser.parse("How do they work?")
        assert frame_how.frame_type == FrameType.COMMAND
        assert frame_how.predicate == "how"
        patient_h = frame_how.get_role(ThematicRole.PATIENT)
        assert patient_h is not None
        assert patient_h.specifier == "anaphoric"

    def test_multiturn_runtime_resolves_conceptual_anaphor(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)

        # Turn 1: establish conceptual topic "focus"
        t1 = rt.process_utterance("Explain focus")
        assert t1.is_success is True

        # Turn 2: follow-up using pronoun "it"
        t2 = rt.process_utterance("Why does it happen?")
        assert t2.is_success is True
        patient = t2.semantic_frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "focus"

    def test_e2e_multiturn_dialogue_coreference(self) -> None:
        sys = build_cognitive_system()
        r1 = sys.process_dialogue("Explain focus")
        assert r1.is_success is True

        r2 = sys.process_dialogue("Why does it happen?")
        assert r2.is_success is True
        assert "knowledge.explain_why" in r2.response_text
        assert "focus" in r2.response_text.lower() or "spiking" in r2.response_text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Autonomous Epistemic Curiosity Loop Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicCuriosityLoop:
    """Verifies uncertainty-gated autonomous web research and memory grounding."""

    def test_curiosity_deduplication_and_uncertainty_gate(self) -> None:
        resolver = CapabilityResolver()
        mock_web = _MockWebSearchExecutor()
        resolver.register(
            CapabilityImplementation(
                capability_name="web.search",
                implementation_id="mock_web_search",
                executor=mock_web,
                priority=10,
            )
        )
        ctrl = EpistemicCuriosityController(resolver, uncertainty_threshold=0.75)

        # Low uncertainty should NOT trigger web research
        assert ctrl.should_research("quantum tunneling", uncertainty=0.2) is False
        assert ctrl.should_research("quantum tunneling", uncertainty=0.9) is True

        res1 = asyncio.run(ctrl.research_concept("quantum tunneling", uncertainty=0.95))
        assert res1.researched is True
        assert len(mock_web.calls) == 1

        # Second call for same concept in same session is deduplicated (loop protection)
        res2 = asyncio.run(ctrl.research_concept("quantum tunneling", uncertainty=0.95))
        assert res2.researched is False
        assert len(mock_web.calls) == 1

    def test_memory_executor_curiosity_researches_and_ingests(self) -> None:
        tiered = TieredWorkspace()
        resolver = CapabilityResolver()
        mock_web = _MockWebSearchExecutor(tiered_workspace=tiered)
        resolver.register(
            CapabilityImplementation(
                capability_name="web.search",
                implementation_id="mock_web_search",
                executor=mock_web,
                priority=10,
            )
        )
        ctrl = EpistemicCuriosityController(
            resolver, tiered_workspace=tiered, fetch_top_result=False
        )
        mem_exec = MemoryCapabilityExecutor(
            default_capability="knowledge.explain_why",
            tiered_workspace=tiered,
            curiosity_controller=ctrl,
        )

        # Initially unknown -> triggers curiosity -> ingests sunlight->photosynthesis -> explains
        res = asyncio.run(
            mem_exec.execute({"capability": "knowledge.explain_why", "content": "photosynthesis"})
        )
        assert res["ok"] is True
        assert res.get("curiosity_researched") is True
        assert "sunlight" in res["content"].lower()
        assert "wikipedia.org/wiki/Photosynthesis" in res["content"]

    def test_curiosity_commits_summary_belief_when_no_triples_extracted(self) -> None:
        tiered = TieredWorkspace()
        resolver = CapabilityResolver()
        mock_web = _MockWebSearchExecutor(tiered_workspace=tiered)
        resolver.register(
            CapabilityImplementation(
                capability_name="web.search",
                implementation_id="mock_web_search",
                executor=mock_web,
                priority=10,
            )
        )
        ctrl = EpistemicCuriosityController(
            resolver, tiered_workspace=tiered, fetch_top_result=False
        )

        res = asyncio.run(ctrl.research_concept("quantum tunneling", uncertainty=1.0))
        assert res.researched is True
        assert res.propositions_ingested == 1
        # Verify persisted in TieredWorkspace.persistent.graph
        p_graph = tiered.persistent.graph
        assert p_graph.get_node("web_concept_quantum_tunneling") is not None
        assert p_graph.get_node("web_curiosity_belief_quantum_tunneling") is not None

    def test_bootstrap_wires_curiosity_when_enabled(self) -> None:
        cfg = CognitiveSystemConfig(memory_dir=None, enable_autonomous_curiosity=True)
        sys = build_cognitive_system(cfg)
        assert "epistemic_curiosity" in sys.services.extensions
        # Swap web.search executor with mock to verify end-to-end dialogue without live network
        mock_web = _MockWebSearchExecutor(tiered_workspace=sys.tiered_workspace)
        for impl in sys.services.capability_resolver.list_implementations("web.search"):
            impl.executor = mock_web
        turn = sys.process_dialogue("Why photosynthesis")
        assert turn.is_success is True
        assert "sunlight" in turn.response_text.lower()
        assert len(mock_web.calls) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. Structural Contradiction Detection & Belief Revision Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBeliefRevisionAndContradictions:
    """Verifies transactional CONTRADICTS edges and confidence-weighted belief revision."""

    def test_supersedes_lower_confidence_contradicted_belief(self) -> None:
        graph = CognitiveGraph()
        old_belief = BeliefNode(
            id="b_old",
            statement="Pluto is a planet",
            confidence=0.60,
        )
        graph.add_node(old_belief)

        new_belief = BeliefNode(
            id="b_new",
            statement="Pluto is not a planet",
            confidence=0.90,
        )
        resolver = BeliefConflictResolver()
        report = resolver.evaluate_belief(new_belief, graph)

        assert report.has_conflicts is True
        assert len(report.conflicts) == 1
        assert report.conflicts[0].resolution == "superseded_existing"
        assert report.adjusted_confidence == 0.90
        # Should emit UPSERT_NODE for revised old belief + ADD_EDGE for CONTRADICTS
        op_types = [op.op.value for op in report.operations]
        assert "upsert_node" in op_types
        assert "add_edge" in op_types

    def test_contested_when_confidences_are_close(self) -> None:
        graph = CognitiveGraph()
        old_belief = BeliefNode(
            id="b_old",
            statement="Coffee is healthy",
            confidence=0.85,
        )
        graph.add_node(old_belief)

        new_belief = BeliefNode(
            id="b_new",
            statement="Coffee is not healthy",
            confidence=0.80,
        )
        resolver = BeliefConflictResolver()
        report = resolver.evaluate_belief(new_belief, graph)

        assert report.has_conflicts is True
        assert report.contested is True
        assert report.conflicts[0].resolution == "contested"
        assert report.adjusted_confidence < 0.80

    def test_memory_store_detects_and_commits_contradiction_edge(self) -> None:
        tiered = TieredWorkspace()
        mem_exec = MemoryCapabilityExecutor(
            default_capability="memory.store",
            tiered_workspace=tiered,
        )

        r1 = asyncio.run(
            mem_exec.execute({"capability": "memory.store", "content": "Pluto is a planet"})
        )
        assert r1["ok"] is True
        assert r1["contradictions_detected"] == 0

        r2 = asyncio.run(
            mem_exec.execute({"capability": "memory.store", "content": "Pluto is not a planet"})
        )
        assert r2["ok"] is True
        assert r2["contradictions_detected"] >= 1
        assert "contradiction" in r2["content"].lower()

        # Verify CONTRADICTS edge was transactionally committed to persistent graph
        p_graph = tiered.persistent.graph
        contra_edges = [
            e for e in p_graph.edges.values() if e.edge_type == HCIREdgeType.CONTRADICTS
        ]
        assert len(contra_edges) >= 1
        beliefs = list(p_graph.nodes_by_type(HCIRNodeType.BELIEF))
        assert len(beliefs) >= 2
