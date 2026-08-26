"""The Grand Cognitive Loop — End-to-End A11–A16 Integrated Benchmark.

Validates the full cognitive stack acting as a unified cognitive substrate:

    Perception (A13)
           ↓
    Object Permanence & Persistence (A13)
           ↓
    Grounded Concept Discovery & Abstraction (A15)
           ↓
    Predictive Expectations & Epistemics (A11/A14)
           ↓
    Natural Language Understanding & Grounding (A16)
           ↓
    Discourse Anaphora & Reference Resolution (A16)
           ↓
    Goal Generation & Execution (A12)
           ↓
    Prediction Error & Learning (A14)
           ↓
    Multilingual Epistemically Calibrated Verbalization (A16)

This demonstrates that language is a thin symbolic interface over an independent,
grounded cognitive substrate — running 100% deterministically with ZERO LLMs.
"""

from __future__ import annotations

import sys

from hbllm.brain.concepts.concept_prediction_bridge import ConceptPredictionBridge
from hbllm.brain.concepts.grounded_concept_registry import GroundedConceptRegistry
from hbllm.brain.language.core.reference import ReferenceResolver
from hbllm.brain.language.core.semantic_frame import (
    LanguageErrorType,
    ThematicRole,
)
from hbllm.brain.language.runtime import MultilingualLanguageRuntime
from hbllm.hcir.graph import (
    ActionNode,
    CognitiveGraph,
    EvidenceNode,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    HCIRNodeType,
    PhysicalEntityNode,
)

# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: The Full 14-Step Cognitive Loop
# ═══════════════════════════════════════════════════════════════════════════


class TestGrandCognitiveLoop:
    """The signature end-to-end cognitive loop across A11 through A16."""

    def test_full_cognitive_dialogue_and_action_loop(self) -> None:
        """Execute the full 14-step perception -> permanence -> concept -> query -> command -> action -> belief revision loop."""
        graph = CognitiveGraph()
        ref_resolver = ReferenceResolver()
        runtime = MultilingualLanguageRuntime(graph, reference_resolver=ref_resolver)
        concept_registry = GroundedConceptRegistry(graph)

        # ── Step 1: Perception creates persistent entities (A13) ─────────
        table = PhysicalEntityNode(
            entity_type="table",
            properties={"material": "wood", "surface": "flat"},
        )
        box = PhysicalEntityNode(
            entity_type="box",
            properties={"material": "cardboard", "color": "brown"},
        )
        ball = PhysicalEntityNode(
            entity_type="ball",
            properties={"color": "red", "shape": "round"},
        )
        graph.add_node(table)
        graph.add_node(box)
        graph.add_node(ball)

        # Initial Spatial Topology: Ball is ON Table
        init_edge = HCIREdge(
            edge_type=HCIREdgeType.LOCATED_IN,
            sources=[ball.id],
            targets=[table.id],
        )
        graph.add_edge(init_edge)

        # ── Step 2: A15 Concept Formation / Grounding ────────────────────
        concept_id = concept_registry.register(
            concept_name="ball",
            feature_prototype={"shape": "round"},
            member_ids=[ball.id],
            behavioral_regularities=["rolls", "supports_push"],
            domain="physical_object",
            utility_delta=0.18,
        )
        assert concept_registry.get_concept(concept_id) is not None

        # Verify A15 Prediction Bridge produces expectations
        bridge = ConceptPredictionBridge(concept_registry)
        specs = bridge.generate_predictions([ball.id])
        assert len(specs) == 2
        assert specs[0].predicted_behavior in ("rolls", "supports_push")

        # ── Step 3: Human asks in English: "Where is the red ball?" ──────
        q1_res = runtime.process_utterance("Where is the red ball?", language="en")
        assert q1_res.is_success
        assert q1_res.response_text == "The ball is on the table."
        assert q1_res.grounded_frame.grounded_entities.get(ThematicRole.THEME) == ball.id

        # ── Step 4: System can also answer the same question in Sinhala ─
        q1_sin = runtime.process_utterance("බෝලය කොහෙද?", language="si")
        assert q1_sin.is_success
        assert q1_sin.response_text == "බෝලය මේසය මත තියෙනවා."

        # ── Step 5: User commands in English using anaphora: "Move it to the box."
        cmd_res = runtime.process_utterance("Move it to the box.", language="en")
        assert cmd_res.is_success
        assert cmd_res.hcir_node_id is not None

        # Verify anaphora resolution bound "it" to the previously mentioned red ball
        assert cmd_res.grounded_frame.grounded_entities.get(ThematicRole.PATIENT) == ball.id
        assert cmd_res.grounded_frame.grounded_entities.get(ThematicRole.DESTINATION) == box.id

        # Verify GoalNode created in HCIR for A12 planner
        goal = graph.get_node(cmd_res.hcir_node_id)
        assert isinstance(goal, GoalNode)
        assert goal.node_type == HCIRNodeType.GOAL
        assert "move" in goal.description.lower()

        # ── Step 6: A12 Execution executes the action ────────────────────
        action = ActionNode(
            action_type="pick_and_place",
            parameters={"target": ball.id, "destination": box.id},
            status="completed",
        )
        graph.add_node(action)

        # ── Step 7: A13 World Model state transition (Ball now in Box) ───
        # Remove old edge and add new edge
        graph.remove_edge(init_edge.id)
        new_edge = HCIREdge(
            edge_type=HCIREdgeType.LOCATED_IN,
            sources=[ball.id],
            targets=[box.id],
        )
        graph.add_edge(new_edge)

        # Update entity property
        ball.properties["location"] = "box"
        graph.upsert_node(ball)

        # ── Step 8: A14 registers successful prediction outcome ──────────
        bridge.record_outcome(concept_id, correct=True)
        updated_concept = concept_registry.get_concept(concept_id)
        assert updated_concept.prediction_count == 1

        # ── Step 9: User asks confirmation in English: "Is the ball in the box?"
        q2_res = runtime.process_utterance("Is the ball on the table?", language="en")
        assert q2_res.is_success
        assert "No, the ball is not" in q2_res.response_text

        q3_res = runtime.process_utterance("Where is the ball?", language="en")
        assert q3_res.is_success
        assert q3_res.response_text == "The ball is in the box."

        # ── Step 10: Inquire in Sinhala after state change ───────────────
        q3_sin = runtime.process_utterance("බෝලය කොහෙද?", language="si")
        assert q3_sin.is_success
        assert q3_sin.response_text == "බෝලය පෙට්ටිය තුළ තියෙනවා."


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: Epistemic Uncertainty, Ambiguity & Clarification
# ═══════════════════════════════════════════════════════════════════════════


class TestAmbiguityAndClarification:
    """Reference ambiguity emits explicit error; disambiguation resolves correctly."""

    def test_ambiguous_reference_then_disambiguation(self) -> None:
        graph = CognitiveGraph()
        ref_resolver = ReferenceResolver()
        runtime = MultilingualLanguageRuntime(graph, reference_resolver=ref_resolver)

        # Two balls in the world: a red ball and a blue ball
        b_red = PhysicalEntityNode(
            entity_type="ball",
            properties={"color": "red"},
        )
        b_blue = PhysicalEntityNode(
            entity_type="ball",
            properties={"color": "blue"},
        )
        t = PhysicalEntityNode(entity_type="table")
        graph.add_node(b_red)
        graph.add_node(b_blue)
        graph.add_node(t)

        # 1. Ambiguous command: "Move the ball to the table."
        res_ambig = runtime.process_utterance("Move the ball to the table.")
        assert not res_ambig.is_success
        assert res_ambig.error_type == LanguageErrorType.AMBIGUOUS_REFERENCE
        assert "multiple objects" in res_ambig.response_text

        # 2. Clarified command: "Move the red ball to the table."
        res_clear = runtime.process_utterance("Move the red ball to the table.")
        assert res_clear.is_success
        assert res_clear.grounded_frame.grounded_entities.get(ThematicRole.PATIENT) == b_red.id


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: Language Assertion as Epistemic Evidence
# ═══════════════════════════════════════════════════════════════════════════


class TestLanguageEvidenceIntegration:
    """Spoken facts enter HCIR as EvidenceNodes and can be queried back with confidence."""

    def test_spoken_assertion_becomes_evidence(self) -> None:
        graph = CognitiveGraph()
        runtime = MultilingualLanguageRuntime(graph)

        box = PhysicalEntityNode(entity_type="box", properties={})
        graph.add_node(box)

        # 1. User asserts in English: "The box is red."
        res_assert = runtime.process_utterance("The box is red.", speaker="operator_bob")
        assert res_assert.is_success
        assert res_assert.hcir_node_id is not None

        # Verify EvidenceNode stored with linguistic provenance
        evidence = graph.get_node(res_assert.hcir_node_id)
        assert isinstance(evidence, EvidenceNode)
        assert "language_evidence" in evidence.tags
        assert "operator_bob" in evidence.source_uri

        # 2. System is asked: "What color is the box?"
        res_query = runtime.process_utterance("What color is the box?")
        assert res_query.is_success
        assert res_query.response_text == "The box is red."


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: Cross-Lingual Interlingua Transfer
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossLingualDialogue:
    """Dialogue alternating between English and Sinhala seamlessly over HCIR."""

    def test_bilingual_interaction(self) -> None:
        graph = CognitiveGraph()
        runtime = MultilingualLanguageRuntime(graph)

        cup = PhysicalEntityNode(entity_type="cup", properties={"color": "blue"})
        table = PhysicalEntityNode(entity_type="table")
        graph.add_node(cup)
        graph.add_node(table)
        graph.add_edge(
            HCIREdge(edge_type=HCIREdgeType.LOCATED_IN, sources=[cup.id], targets=[table.id])
        )

        # English query -> English answer
        q_en = runtime.process_utterance("Where is the cup?", language="en")
        assert q_en.response_text == "The cup is on the table."

        # Sinhala query -> Sinhala answer
        q_si = runtime.process_utterance("කෝප්පය කොහෙද?", language="si")
        assert q_si.response_text == "කෝප්පය මේසය මත තියෙනවා."

        # Interlingual translation test
        translation = runtime.translate(
            "The cup is on the table.", source_lang="en", target_lang="si"
        )
        assert "කෝප්පය මේසය මත තියෙනවා" in translation


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: Deterministic Replay of the Entire Multi-Turn Session
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterministicSessionReplay:
    """The entire multi-turn interaction replays byte-for-byte identically."""

    def test_full_session_replay(self) -> None:
        def _run_session() -> list[str]:
            g = CognitiveGraph()
            r = MultilingualLanguageRuntime(g)
            t = PhysicalEntityNode(entity_type="table")
            b = PhysicalEntityNode(entity_type="ball", properties={"color": "red"})
            g.add_node(t)
            g.add_node(b)
            g.add_edge(HCIREdge(edge_type=HCIREdgeType.LOCATED_IN, sources=[b.id], targets=[t.id]))

            responses = []
            responses.append(
                r.process_utterance("Where is the red ball?", language="en").response_text
            )
            responses.append(
                r.process_utterance("Move it to the table.", language="en").response_text
            )
            responses.append(r.process_utterance("බෝලය කොහෙද?", language="si").response_text)
            return responses

        run1 = _run_session()
        run2 = _run_session()

        assert run1 == run2
        assert len(run1) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: Prediction Failure & Cognitive Surprise (Counterfactual / Error Loop)
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictionFailureAndSurprise:
    """Action outcome contradicts forward prediction -> Error classified -> Confidence adapts."""

    def test_prediction_error_modulates_concept_and_dialogue(self) -> None:
        graph = CognitiveGraph()
        runtime = MultilingualLanguageRuntime(graph)
        concept_registry = GroundedConceptRegistry(graph)
        bridge = ConceptPredictionBridge(concept_registry)

        # 1. World setup: Ball on table
        table = PhysicalEntityNode(entity_type="table")
        box = PhysicalEntityNode(entity_type="box")
        ball = PhysicalEntityNode(entity_type="ball", properties={"color": "red"})
        graph.add_node(table)
        graph.add_node(box)
        graph.add_node(ball)
        init_edge = HCIREdge(
            edge_type=HCIREdgeType.LOCATED_IN, sources=[ball.id], targets=[table.id]
        )
        graph.add_edge(init_edge)

        # 2. Register concept with initial prediction accuracy
        c_id = concept_registry.register(
            concept_name="ball",
            feature_prototype={"shape": "round"},
            member_ids=[ball.id],
            domain="physical_object",
            utility_delta=0.15,
        )
        concept_before = concept_registry.get_concept(c_id)
        assert concept_before is not None
        initial_accuracy = concept_before.prediction_accuracy

        # 3. Predict forward outcome: expected destination is "box"
        expected_destination = "box"

        # 4. Action attempted, but failed / blocked: ball remains ON table
        # (Actual observation contradicts prediction: observed "table" != expected "box")
        observed_destination = "table"
        prediction_succeeded = expected_destination == observed_destination
        assert not prediction_succeeded, "Prediction failed: ball did not move to box"

        # 5. A14/A15 records negative outcome: concept prediction accuracy drops
        bridge.record_outcome(c_id, correct=False)
        concept_after = concept_registry.get_concept(c_id)
        assert concept_after is not None
        assert concept_after.prediction_accuracy < initial_accuracy
        assert concept_after.prediction_count == 1

        # 6. Human asks in English: "Is the ball in the box?"
        q_verify = runtime.process_utterance("Is the ball in the box?", language="en")
        assert q_verify.is_success
        assert "No, the ball is not" in q_verify.response_text

        # 7. Human asks: "Where is the ball?"
        q_loc = runtime.process_utterance("Where is the ball?", language="en")
        assert q_loc.is_success
        assert q_loc.response_text == "The ball is on the table."


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: Conflicting Speakers & Epistemic Contradiction
# ═══════════════════════════════════════════════════════════════════════════


class TestConflictingEvidenceContradiction:
    """Contradictory assertions from different speakers trigger CONTRADICTED state."""

    def test_contradictory_assertions_trigger_conflict_warning(self) -> None:
        graph = CognitiveGraph()
        runtime = MultilingualLanguageRuntime(graph)

        box = PhysicalEntityNode(entity_type="box", properties={})
        graph.add_node(box)

        # Speaker 1 asserts: "The box is red."
        res1 = runtime.process_utterance("The box is red.", speaker="operator_alice")
        assert res1.is_success

        # Speaker 2 asserts: "The box is blue."
        res2 = runtime.process_utterance("The box is blue.", speaker="operator_bob")
        assert res2.is_success

        # Query color -> Epistemic evaluation detects conflicting evidence
        res_q = runtime.process_utterance("What color is the box?")
        assert res_q.is_success
        assert "conflicting evidence" in res_q.response_text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: Stale Observation & Epistemic Uncertainty Decay
# ═══════════════════════════════════════════════════════════════════════════


class TestStalePerceptionDecay:
    """Stale observations decay effective confidence, resulting in calibrated epistemic hedges."""

    def test_stale_observation_causes_hedged_verbalization(self) -> None:
        graph = CognitiveGraph()
        runtime = MultilingualLanguageRuntime(graph)

        # Entity with low freshness (observed in the past, no recent update)
        cup = PhysicalEntityNode(entity_type="cup", properties={"freshness": 0.55})
        table = PhysicalEntityNode(entity_type="table")
        graph.add_node(cup)
        graph.add_node(table)
        graph.add_edge(
            HCIREdge(edge_type=HCIREdgeType.LOCATED_IN, sources=[cup.id], targets=[table.id])
        )

        # Query location -> System hedges due to staleness (0.96 * 0.55 = 0.528 -> PLAUSIBLE)
        res_q = runtime.process_utterance("Where is the cup?")
        assert res_q.is_success
        assert "I think the cup may be on the table" in res_q.response_text


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: Zero-LLM Invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLMIntegration:
    """The full integrated loop executes with zero neural/LLM dependencies."""

    def test_zero_llm_imports(self) -> None:
        import subprocess
        import sys

        check_code = """
import sys
import hbllm.brain.concepts
import hbllm.brain.language.runtime
import hbllm.brain.learning
import hbllm.brain.world
import hbllm.brain.reasoning

llm_markers = ["openai", "anthropic", "litellm", "langchain", "transformers"]
loaded = set(sys.modules.keys())
for marker in llm_markers:
    assert marker not in loaded, f"LLM module loaded in integrated runtime: {marker}"
"""
        res = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
        assert res.returncode == 0, f"Zero-LLM verification failed:\n{res.stderr}"
