"""A16 Benchmark — Multilingual Non-LLM Language Runtime.

12 end-to-end benchmark scenarios:
1.  Morphological analysis (inflections, irregulars, lemmatization)
2.  Phrase structure & syntax parsing (declarative, imperative, copular, yes/no, wh)
3.  Reference & pronoun anaphora resolution ("Move it")
4.  Concept & entity grounding in A13/A15 HCIR
5.  Language as evidence boundary (creates EvidenceNode with provenance)
6.  Command to GoalNode proposal
7.  Calibrated epistemic surface realization
8.  Cross-lingual semantic equivalence (English vs. Sinhala)
9.  Deterministic interlingual semantic transfer (English <-> Sinhala via HCIR)
10. Deterministic replay invariant
11. Graceful error handling (UNRESOLVED_LANGUAGE, AMBIGUOUS_REFERENCE)
12. Zero-LLM invariant (100% deterministic rule-guided code)
"""

from __future__ import annotations

from hbllm.brain.language.core.epistemic_policy import (
    CognitiveEpistemicState,
)
from hbllm.brain.language.core.semantic_frame import (
    FrameType,
    LanguageErrorType,
    ThematicRole,
)
from hbllm.brain.language.english.morphology import EnglishMorphology
from hbllm.brain.language.english.parser import EnglishParser
from hbllm.brain.language.english.realizer import EnglishRealizer
from hbllm.brain.language.english.syntax import ConstructionType, EnglishSyntaxParser
from hbllm.brain.language.runtime import MultilingualLanguageRuntime
from hbllm.brain.language.sinhala.parser import SinhalaParser
from hbllm.hcir.graph import (
    CognitiveGraph,
    GroundedConceptNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    PhysicalEntityNode,
)


def _setup_world(graph: CognitiveGraph) -> tuple[PhysicalEntityNode, PhysicalEntityNode]:
    """Helper to populate HCIR with a red ball and a wooden table."""
    table = PhysicalEntityNode(
        entity_type="table",
        properties={"material": "wood"},
    )
    ball = PhysicalEntityNode(
        entity_type="ball",
        properties={"color": "red", "shape": "round"},
    )
    graph.add_node(table)
    graph.add_node(ball)

    # Ball LOCATED_ON Table edge
    edge = HCIREdge(
        edge_type=HCIREdgeType.LOCATED_IN,
        sources=[ball.id],
        targets=[table.id],
    )
    graph.add_edge(edge)

    # Add A15 GroundedConceptNodes
    concept_ball = GroundedConceptNode(
        concept_name="ball",
        feature_prototype={"shape": "round"},
        domain="physical_object",
    )
    concept_table = GroundedConceptNode(
        concept_name="table",
        feature_prototype={"surface": "flat"},
        domain="furniture",
    )
    graph.add_node(concept_ball)
    graph.add_node(concept_table)

    return ball, table


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: Morphological Analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestMorphology:
    """Inflections, irregular forms, and lemmatization."""

    def test_irregular_verbs_and_plurals(self) -> None:
        morph = EnglishMorphology()

        # Irregular verbs
        tok_fell = morph.analyze("fell")
        assert tok_fell.lemma == "fall"
        assert tok_fell.features["tense"] == "past"

        tok_gave = morph.analyze("gave")
        assert tok_gave.lemma == "give"
        assert tok_gave.features["tense"] == "past"

        # Plurals
        tok_boxes = morph.analyze("boxes")
        assert tok_boxes.lemma == "box"
        assert tok_boxes.features["number"] == "plural"

        # Regular inflection generation
        assert morph.inflect_verb("push", tense="past") == "pushed"
        assert morph.inflect_verb("roll", tense="present", person="3s") == "rolls"
        assert morph.pluralize_noun("ball") == "balls"
        assert morph.pluralize_noun("box") == "boxes"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: Phrase Structure & Syntax Parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestSyntaxParsing:
    """Grammatical constructions parse into typed AST nodes."""

    def test_parse_major_constructions(self) -> None:
        parser = EnglishSyntaxParser()
        morph = EnglishMorphology()

        # Declarative
        toks = morph.lemmatize_sequence("The red ball is on the table.")
        ast = parser.parse(toks)
        assert ast is not None
        assert ast.construction == ConstructionType.DECLARATIVE
        assert ast.subject is not None
        assert ast.subject.head_noun.lemma == "ball"
        assert len(ast.subject.adjectives) == 1
        assert ast.subject.adjectives[0].lemma == "red"

        # Imperative
        toks = morph.lemmatize_sequence("Move the red ball to the table.")
        ast = parser.parse(toks)
        assert ast is not None
        assert ast.construction == ConstructionType.IMPERATIVE
        assert ast.verb_phrase is not None
        assert ast.verb_phrase.head_verb.lemma == "move"

        # Wh-Question
        toks = morph.lemmatize_sequence("Where is the ball?")
        ast = parser.parse(toks)
        assert ast is not None
        assert ast.construction == ConstructionType.WH_QUESTION
        assert ast.wh_word.lemma == "where"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: Reference & Pronoun Resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestReferenceResolution:
    """Anaphoric pronoun ('it') resolves to the most salient entity in discourse."""

    def test_resolve_anaphoric_it(self) -> None:
        graph = CognitiveGraph()
        ball, table = _setup_world(graph)

        runtime = MultilingualLanguageRuntime(graph)

        # Turn 1: User mentions "The red ball is on the table."
        res1 = runtime.process_utterance("The red ball is on the table.")
        assert res1.is_success

        # Turn 2: User says "Move it." -> 'it' should ground to the red ball
        res2 = runtime.process_utterance("Move it.")
        assert res2.is_success
        assert res2.grounded_frame is not None
        assert res2.grounded_frame.grounded_entities.get(ThematicRole.PATIENT) == ball.id


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: Concept & Entity Grounding
# ═══════════════════════════════════════════════════════════════════════════


class TestGrounding:
    """'the red ball' maps to the specific PhysicalEntityNode in A13."""

    def test_ground_red_ball(self) -> None:
        graph = CognitiveGraph()
        ball, table = _setup_world(graph)

        runtime = MultilingualLanguageRuntime(graph)
        res = runtime.process_utterance("Where is the red ball?")

        assert res.is_success
        assert res.grounded_frame is not None
        assert res.grounded_frame.grounded_entities.get(ThematicRole.THEME) == ball.id


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: Language as Evidence Boundary
# ═══════════════════════════════════════════════════════════════════════════


class TestLanguageEvidenceBoundary:
    """Assertions produce EvidenceNodes with linguistic provenance, not direct truth mutations."""

    def test_assertion_creates_evidence_node(self) -> None:
        graph = CognitiveGraph()
        ball, table = _setup_world(graph)

        runtime = MultilingualLanguageRuntime(graph)
        res = runtime.process_utterance("The red ball is on the table.", speaker="user_alice")

        assert res.is_success
        assert res.hcir_node_id is not None

        # Verify EvidenceNode exists in HCIR
        evidence = graph.get_node(res.hcir_node_id)
        assert evidence is not None
        assert evidence.node_type == HCIRNodeType.EVIDENCE
        assert "language_evidence" in evidence.tags
        assert "user_alice" in evidence.source_uri


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: Command to GoalNode Proposal
# ═══════════════════════════════════════════════════════════════════════════


class TestCommandToGoal:
    """Imperative commands create GoalNodes for A12 planning."""

    def test_command_creates_goal_node(self) -> None:
        graph = CognitiveGraph()
        ball, table = _setup_world(graph)

        runtime = MultilingualLanguageRuntime(graph)
        res = runtime.process_utterance("Move the red ball to the table.")

        assert res.is_success
        assert res.hcir_node_id is not None

        goal = graph.get_node(res.hcir_node_id)
        assert goal is not None
        assert goal.node_type == HCIRNodeType.GOAL
        assert "move" in goal.description.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: Calibrated Epistemic Surface Realization
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicRealization:
    """Realizer chooses appropriate hedges based on rich EpistemicState."""

    def test_epistemic_hedges_by_confidence(self) -> None:
        realizer = EnglishRealizer()

        # 1. High certainty (>= 0.92)
        state_certain = CognitiveEpistemicState(
            target_predicate="located_on",
            target_subject="cup",
            target_object="table",
            confidence=0.96,
            support_count=4,
            is_known=True,
        )
        assert realizer.realize(state_certain) == "The cup is on the table."

        # 2. Probable (0.70 - 0.92)
        state_probable = CognitiveEpistemicState(
            target_predicate="located_on",
            target_subject="cup",
            target_object="table",
            confidence=0.82,
            support_count=2,
            is_known=True,
        )
        assert realizer.realize(state_probable) == "The cup is probably on the table."

        # 3. Plausible / Weak belief (0.40 - 0.70)
        state_plausible = CognitiveEpistemicState(
            target_predicate="located_on",
            target_subject="cup",
            target_object="table",
            confidence=0.55,
            support_count=1,
            is_known=True,
        )
        assert realizer.realize(state_plausible) == "I think the cup may be on the table."

        # 4. Insufficient evidence / Knowledge gap
        state_unknown = CognitiveEpistemicState(
            target_predicate="located_on",
            target_subject="cup",
            confidence=0.10,
            support_count=0,
            is_known=False,
        )
        assert "not have enough evidence" in realizer.realize(state_unknown)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: Cross-Lingual Semantic Equivalence
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossLingualEquivalence:
    """English (SVO) and Sinhala (SOV) map to identical language-neutral SemanticFrames."""

    def test_english_and_sinhala_identical_frames(self) -> None:
        en_parser = EnglishParser()
        si_parser = SinhalaParser()

        # English: "The red ball is on the table."
        en_frame = en_parser.parse("The red ball is on the table.")

        # Sinhala: "රතු බෝලය මේසය මත තියෙනවා."
        si_frame = si_parser.parse("රතු බෝලය මේසය මත තියෙනවා.")

        assert en_frame.frame_type == si_frame.frame_type == FrameType.ASSERTION
        assert en_frame.predicate == si_frame.predicate == "located_on"

        # Check theme entity
        en_theme = en_frame.get_role(ThematicRole.THEME)
        si_theme = si_frame.get_role(ThematicRole.THEME)
        assert en_theme is not None and si_theme is not None
        assert en_theme.concept_name == si_theme.concept_name == "ball"
        assert en_theme.properties.get("color") == si_theme.properties.get("color") == "red"

        # Check location entity
        en_loc = en_frame.get_role(ThematicRole.LOCATION)
        si_loc = si_frame.get_role(ThematicRole.LOCATION)
        assert en_loc is not None and si_loc is not None
        assert en_loc.concept_name == si_loc.concept_name == "table"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: Deterministic Interlingual Semantic Transfer
# ═══════════════════════════════════════════════════════════════════════════


class TestInterlingualTransfer:
    """Translate English <-> Sinhala via language-neutral SemanticFrames in HCIR."""

    def test_english_to_sinhala_and_back(self) -> None:
        graph = CognitiveGraph()
        runtime = MultilingualLanguageRuntime(graph)

        # English -> Sinhala
        si_text = runtime.translate(
            "The red ball is on the table.", source_lang="en", target_lang="si"
        )
        assert "රතු" in si_text
        assert "බෝලය" in si_text
        assert "මේසය" in si_text
        assert "මත" in si_text

        # Sinhala -> English
        en_text = runtime.translate("රතු බෝලය මේසය මත තියෙනවා.", source_lang="si", target_lang="en")
        assert en_text == "The red ball is on the table."


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: Deterministic Replay Invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterministicReplay:
    """Same text + same HCIR state = byte-for-byte identical results."""

    def test_replay_determinism(self) -> None:
        graph = CognitiveGraph()
        ball, table = _setup_world(graph)

        runtime = MultilingualLanguageRuntime(graph)
        prompt = "Where is the red ball?"

        res1 = runtime.process_utterance(prompt)
        res2 = runtime.process_utterance(prompt)

        assert res1.response_text == res2.response_text == "The ball is on the table."
        assert res1.semantic_frame.predicate == res2.semantic_frame.predicate
        assert res1.grounded_frame.grounded_entities == res2.grounded_frame.grounded_entities


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11: Graceful Epistemic Failure Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicErrorHandling:
    """Emits UNRESOLVED_LANGUAGE or AMBIGUOUS_REFERENCE rather than guessing."""

    def test_unresolved_language_on_syntax_failure(self) -> None:
        graph = CognitiveGraph()
        runtime = MultilingualLanguageRuntime(graph)

        res = runtime.process_utterance("Table ball on the is quickly blue.")
        assert not res.is_success
        assert res.error_type == LanguageErrorType.UNRESOLVED_LANGUAGE

    def test_ambiguous_reference_on_multiple_matches(self) -> None:
        graph = CognitiveGraph()
        # Add TWO red balls
        b1 = PhysicalEntityNode(entity_type="ball", properties={"color": "red"})
        b2 = PhysicalEntityNode(entity_type="ball", properties={"color": "red"})
        graph.add_node(b1)
        graph.add_node(b2)

        runtime = MultilingualLanguageRuntime(graph)
        res = runtime.process_utterance("Where is the red ball?")

        assert not res.is_success
        assert res.error_type == LanguageErrorType.AMBIGUOUS_REFERENCE


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12: Zero-LLM Invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Entire A16 runtime operates without any external neural or LLM calls."""

    def test_no_llm_imports(self) -> None:
        import subprocess
        import sys

        check_code = """
import sys
import hbllm.brain.language.core.epistemic_policy
import hbllm.brain.language.core.gateway
import hbllm.brain.language.core.grounding
import hbllm.brain.language.core.protocol
import hbllm.brain.language.core.reference
import hbllm.brain.language.core.semantic_frame
import hbllm.brain.language.english.lexicon
import hbllm.brain.language.english.morphology
import hbllm.brain.language.english.parser
import hbllm.brain.language.english.realizer
import hbllm.brain.language.english.syntax
import hbllm.brain.language.runtime
import hbllm.brain.language.sinhala.lexicon
import hbllm.brain.language.sinhala.parser
import hbllm.brain.language.sinhala.realizer

llm_markers = [
    "openai",
    "anthropic",
    "litellm",
    "langchain",
    "transformers",
]

loaded = set(sys.modules.keys())
for marker in llm_markers:
    assert marker not in loaded, f"LLM module loaded: {marker}"
"""
        import os

        env = dict(os.environ, PYTHONPATH=":".join(sys.path))
        res = subprocess.run(
            [sys.executable, "-c", check_code], capture_output=True, text=True, env=env
        )
        assert res.returncode == 0, f"Zero-LLM verification failed:\n{res.stderr}"
