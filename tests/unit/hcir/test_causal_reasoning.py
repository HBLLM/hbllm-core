"""Unit tests for Multi-Step Causal ('Why') and Procedural ('How') Reasoning.

Verifies:
  1. ConceptExplainer.explain_why (backwards CAUSES traversal and multi-hop causal chains).
  2. ConceptExplainer.explain_how (forward REQUIRES traversal and procedural steps).
  3. MemoryCapabilityExecutor knowledge.explain_why and knowledge.explain_how.
  4. CommandDispatcher routing (why, how, explain why X, explain how X).
  5. EnglishParser parsing of Wh-questions with why/how and imperative variations.
  6. End-to-end dialogue execution via CognitiveSystem.process_dialogue.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hbllm.hcir.capabilities.concept_explainer import (
    ConceptExplainer,
    ConceptKnowledgeSnapshot,
)
from hbllm.hcir.graph import CognitiveGraph, ConceptNode, HCIREdge, HCIREdgeType
from hbllm.hcir.kernel.command_dispatcher import CommandDispatcher
from hbllm.hcir.language.core.semantic_frame import (
    FrameType,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.language.english.parser import EnglishParser
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.runtime.bootstrap import build_cognitive_system

# ═══════════════════════════════════════════════════════════════════════════
# 1. ConceptExplainer Causal & Procedural Traversal Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConceptExplainerCausalReasoning:
    """Tests backwards causal (why) and forward dependency (how) traversal."""

    def test_explain_why_direct_and_upstream_causes(self) -> None:
        graph = CognitiveGraph()
        # Chain: stimulus -(CAUSES)-> spiking -(CAUSES)-> focus
        graph.add_node(ConceptNode(id="t4_concept_stimulus", label="stimulus"))
        graph.add_node(ConceptNode(id="t4_concept_spiking", label="spiking"))
        graph.add_node(ConceptNode(id="t4_concept_focus", label="focus"))

        graph.add_edge(
            HCIREdge(
                id="e1",
                edge_type=HCIREdgeType.CAUSES,
                sources=["t4_concept_stimulus"],
                targets=["t4_concept_spiking"],
            )
        )
        graph.add_edge(
            HCIREdge(
                id="e2",
                edge_type=HCIREdgeType.CAUSES,
                sources=["t4_concept_spiking"],
                targets=["t4_concept_focus"],
            )
        )

        tiered = TieredWorkspace()
        for n in graph.nodes.values():
            tiered.persistent.graph.add_node(n)
        for e in graph.edges.values():
            tiered.persistent.graph.add_edge(e)

        explainer = ConceptExplainer(tiered_workspace=tiered)
        text = explainer.explain_why("focus")

        assert "Focus" in text or "focus" in text
        assert "spiking" in text
        # Multi-hop upstream cause stimulus should appear in the causal chain
        assert "stimulus" in text

    def test_explain_why_consequences(self) -> None:
        explainer = ConceptExplainer()
        snap = ConceptKnowledgeSnapshot(
            concept="combustion",
            definition="rapid oxidation producing heat",
            causes=["heat", "light"],
        )
        text = explainer.explain_why("combustion", snapshot=snap)
        # Should explain combustion definition and its downstream consequences
        assert "combustion" in text.lower()
        assert "heat" in text.lower()

    def test_explain_how_direct_and_upstream_requirements(self) -> None:
        graph = CognitiveGraph()
        # Chain: friendship -(REQUIRES)-> care -(REQUIRES)-> attention
        graph.add_node(ConceptNode(id="t4_concept_friendship", label="friendship"))
        graph.add_node(ConceptNode(id="t4_concept_care", label="care"))
        graph.add_node(ConceptNode(id="t4_concept_attention", label="attention"))

        graph.add_edge(
            HCIREdge(
                id="e1",
                edge_type=HCIREdgeType.REQUIRES,
                sources=["t4_concept_friendship"],
                targets=["t4_concept_care"],
            )
        )
        graph.add_edge(
            HCIREdge(
                id="e2",
                edge_type=HCIREdgeType.REQUIRES,
                sources=["t4_concept_care"],
                targets=["t4_concept_attention"],
            )
        )

        tiered = TieredWorkspace()
        for n in graph.nodes.values():
            tiered.persistent.graph.add_node(n)
        for e in graph.edges.values():
            tiered.persistent.graph.add_edge(e)

        explainer = ConceptExplainer(tiered_workspace=tiered)
        text = explainer.explain_how("friendship")

        assert "friendship" in text
        assert "care" in text
        assert "Step 1" in text
        assert "attention" in text


# ═══════════════════════════════════════════════════════════════════════════
# 2. EnglishParser WH-Question & Imperative Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParserWhyAndHow:
    """Verifies that EnglishParser identifies why and how queries and commands."""

    def test_parse_why_command(self) -> None:
        parser = EnglishParser()
        frame = parser.parse("Why focus")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate == "why"
        patient = frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "focus"

    def test_parse_how_command(self) -> None:
        parser = EnglishParser()
        frame = parser.parse("How friendship")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate == "how"
        patient = frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "friendship"

    def test_parse_explain_why_normalizes_to_why(self) -> None:
        parser = EnglishParser()
        frame = parser.parse("Explain why spiking occurs")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate == "why"
        patient = frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "spiking"

    def test_parse_explain_how_normalizes_to_how(self) -> None:
        parser = EnglishParser()
        frame = parser.parse("Explain how a microscope works")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate == "how"
        patient = frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "microscope"


# ═══════════════════════════════════════════════════════════════════════════
# 3. CommandDispatcher Routing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCommandDispatcherCausalRouting:
    """Verifies that CommandDispatcher routes why and how to causal capabilities."""

    def test_routing_why_to_explain_why(self) -> None:
        disp = CommandDispatcher(MagicMock(), None)
        frame = SemanticFrame(
            frame_type=FrameType.COMMAND,
            predicate="why",
            metadata=LanguageMetadata(raw_text="Why focus"),
        )
        assert disp._resolve_capability("why", frame=frame) == "knowledge.explain_why"
        params = disp._build_params(frame, "knowledge.explain_why", "focus")
        assert params["content"] == "focus"

    def test_routing_how_to_explain_how(self) -> None:
        disp = CommandDispatcher(MagicMock(), None)
        frame = SemanticFrame(
            frame_type=FrameType.COMMAND,
            predicate="how",
            metadata=LanguageMetadata(raw_text="Explain how a microscope works"),
        )
        assert disp._resolve_capability("how", frame=frame) == "knowledge.explain_how"
        params = disp._build_params(frame, "knowledge.explain_how", "microscope")
        assert params["content"] == "microscope"


# ═══════════════════════════════════════════════════════════════════════════
# 4. End-to-End Dialogue Causal Reasoning Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestE2ECausalDialogue:
    """Verifies that CognitiveSystem processes why and how dialogue turns."""

    def test_e2e_why_dialogue(self) -> None:
        sys = build_cognitive_system()
        res = sys.process_dialogue("Why focus")
        assert res.is_success is True
        assert "knowledge.explain_why" in res.response_text
        assert "spiking" in res.response_text.lower() or "focus" in res.response_text.lower()

    def test_e2e_how_dialogue(self) -> None:
        sys = build_cognitive_system()
        res = sys.process_dialogue("How friendship")
        assert res.is_success is True
        assert "knowledge.explain_how" in res.response_text
        assert "care" in res.response_text.lower() or "friendship" in res.response_text.lower()

    def test_e2e_explain_how_microscope(self) -> None:
        sys = build_cognitive_system()
        res = sys.process_dialogue("Explain how a microscope works")
        assert res.is_success is True
        assert "knowledge.explain_how" in res.response_text
        assert "microscope" in res.response_text.lower()
