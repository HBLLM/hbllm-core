from __future__ import annotations

from hbllm.hcir.bytecode import Opcode
from hbllm.hcir.compiler import HCIRCompiler, IntentType
from hbllm.hcir.grounding.bus.observation_bus import ObservationBus
from hbllm.hcir.grounding.concepts.concept_learner import ConceptLearner
from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.dialogue import DialogueInterpretationResult
from hbllm.hcir.grounding.contracts.enums import (
    ConceptLifecycle,
    DecisionOutcome,
    DialogueAct,
    ModalitySource,
    TrafficClass,
)
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.contracts.references import (
    GroundedConceptRef,
    GroundedEntityRef,
    UnresolvedRef,
)
from hbllm.hcir.grounding.features.composite_observation_adapter import (
    CompositeObservationAdapter,
    SemanticLLMObservationAdapter,
)
from hbllm.hcir.grounding.features.feature_extractor import (
    PerceptualFeatureVector,
)


def _make_visual_feature(
    embedding: tuple[float, ...],
    feature_id: str = "feat_vis_1",
    entity_id: str | None = None,
) -> PerceptualFeatureVector:
    meta = {"entity_id": entity_id} if entity_id else {}
    return PerceptualFeatureVector(
        feature_id=feature_id,
        observation_id="obs_vis_01",
        modality=ModalitySource.VISION,
        channel_id="cam_front",
        embedding=embedding,
        feature_metadata=meta,
    )


class TestGroundedLanguageBinding:
    """Validation suite for natural language observation ingestion, multimodal

    concept binding, and compiler entity resolution with active inquiry gating.
    """

    def test_dialogue_observation_envelope_publishing(self) -> None:
        """Verify feeding dialogue onto ObservationBus via SEMANTIC_LLM_ADAPTER with discrete QoS."""
        bus = ObservationBus()
        received_envelopes: list[ObservationEnvelope] = []

        bus.subscribe(
            callback=lambda env: received_envelopes.append(env),
            modality=ModalitySource.SEMANTIC_LLM_ADAPTER,
        )

        env = bus.publish_dialogue(
            text="push the red block",
            speaker_id="operator_1",
            channel_id="dialogue_inbound",
            dialogue_act="command",
        )

        assert env is not None
        assert env.source_modality == ModalitySource.SEMANTIC_LLM_ADAPTER
        assert env.traffic_class == TrafficClass.CLASS_DISCRETE
        assert env.metadata["text"] == "push the red block"
        assert env.metadata["speaker_id"] == "operator_1"
        assert "push" in env.metadata["tokens"]
        assert len(received_envelopes) == 1
        assert received_envelopes[0].observation_id == env.observation_id

    def test_semantic_text_feature_extraction_from_dialogue(self) -> None:
        """Verify feature extractor extracts semantic feature vectors from dialogue envelope."""
        adapter = SemanticLLMObservationAdapter(channel_id="dialogue_test")
        envelope, features = adapter.adapt(
            text="push the red block",
            named_entities={"OBJECT": ["red block"]},
        )

        assert envelope.source_modality == ModalitySource.SEMANTIC_LLM_ADAPTER
        assert len(features) >= 1
        ne_feat = next(f for f in features if f.feature_metadata.get("text") == "red block")
        assert ne_feat.modality == ModalitySource.SEMANTIC_LLM_ADAPTER
        assert ne_feat.semantic_labels.get("OBJECT") == 1.0

    def test_composite_observation_adapter_dialogue_dispatch(self) -> None:
        """Verify CompositeObservationAdapter dispatches dialogue and driver inputs."""
        comp = CompositeObservationAdapter()
        env, feats = comp.adapt_dialogue("grab the yellow cylinder")
        assert env.source_modality == ModalitySource.SEMANTIC_LLM_ADAPTER
        assert len(feats) >= 1

        # Driver input dispatch
        class MockDriverInput:
            modality = "dialogue"
            raw_data = "open the safety latch"
            metadata = {"speaker_id": "supervisor"}

        env_drv, feats_drv = comp.adapt_driver_input(MockDriverInput())
        assert env_drv.source_modality == ModalitySource.SEMANTIC_LLM_ADAPTER
        assert env_drv.metadata["speaker_id"] == "supervisor"

    def test_concept_learner_bind_linguistic_alias(self) -> None:
        """Verify binding linguistic alias to concept raises cross-modal support and updates store."""
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store)

        # Learn initial perceptual representation
        feat = _make_visual_feature((1.0, 0.0, 0.0), feature_id="f_red_block")
        assign = learner.learn(feat, exemplar_entity_id="e_block_1")
        concept_id = assign.concept_ref.id

        concept_before = store.get_concept(concept_id)
        assert concept_before is not None
        assert "red block" not in concept_before.linguistic_aliases

        # Bind linguistic alias
        updated_concept = learner.bind_linguistic_alias(
            concept_ref_or_id=concept_id,
            alias="red block",
            confidence=1.0,
        )

        assert updated_concept.linguistic_aliases.get("red block") == 1.0
        assert updated_concept.grounding_quality.cross_modal_support >= 0.8
        assert updated_concept.grounding_quality.get_measurement("cross_modal_support").is_tested

        # Lookup by alias in WorkingConceptStore
        found = store.find_concept_by_alias("red block")
        assert found is not None
        assert found.concept_ref.id == concept_id

        # Lookup with leading article
        found_with_article = store.find_concept_by_alias("the red block")
        assert found_with_article is not None
        assert found_with_article.concept_ref.id == concept_id

    def test_concept_learner_bind_text_feature_via_exemplar_entity(self) -> None:
        """Verify binding text feature vector to concept matched by active exemplar entity ID."""
        store = WorkingConceptStore()
        learner = ConceptLearner(concept_store=store)

        # 1. Vision learns concept for e_cube_42
        feat_vis = _make_visual_feature((0.5, 0.5, 0.2), feature_id="f_cube_vis")
        assign = learner.learn(feat_vis, exemplar_entity_id="e_cube_42")

        # 2. Text feature arrives for "cube"
        text_feat = PerceptualFeatureVector(
            feature_id="feat_txt_cube",
            observation_id="obs_txt_1",
            modality=ModalitySource.SEMANTIC_LLM_ADAPTER,
            channel_id="user_dialogue",
            embedding=(0.1, 0.2, 0.3),
            feature_metadata={"text": "cube", "entity_id": "e_cube_42"},
            semantic_labels={"OBJECT": 1.0},
        )

        bound_concept = learner.bind_text_feature(text_feat, target_entity_id="e_cube_42")
        assert bound_concept is not None
        assert bound_concept.concept_ref.id == assign.concept_ref.id
        assert bound_concept.linguistic_aliases.get("cube") == 1.0

    def test_concept_split_preserves_linguistic_aliases(self) -> None:
        """Verify concept splitting into sub-clusters preserves inherited linguistic aliases."""
        store = WorkingConceptStore()
        learner = ConceptLearner(
            concept_store=store,
            split_dispersion_threshold=2.0,
            min_prototypes_for_split=4,
            min_split_gap=1.5,
        )

        # Create emerging concept
        cid = "C-TESTSPLIT"
        c = GroundedConcept(
            concept_ref=GroundedConceptRef(id=cid),
            lifecycle_state=ConceptLifecycle.EMERGING,
        )
        c.add_linguistic_alias("polymorphic block", confidence=0.9)
        # Add bimodal prototypes
        c.add_prototype({"dim_0": 0.0, "dim_1": 0.0})
        c.add_prototype({"dim_0": 0.1, "dim_1": 0.1})
        c.add_prototype({"dim_0": 10.0, "dim_1": 10.0})
        c.add_prototype({"dim_0": 10.1, "dim_1": 10.1})
        store.register_concept(c)

        splits, _ = learner.maintain()
        assert len(splits) == 1
        child_ids = splits[0].child_concept_ids
        assert len(child_ids) == 2

        child_a = store.get_concept(child_ids[0])
        child_b = store.get_concept(child_ids[1])
        assert child_a is not None and child_b is not None
        assert child_a.linguistic_aliases.get("polymorphic block") == 0.9
        assert child_b.linguistic_aliases.get("polymorphic block") == 0.9

    def test_compiler_ast_resolver_resolves_known_grounded_entity(self) -> None:
        """Verify compiler resolves 'push the red block' to GroundedEntityRef and emits motor bytecode."""
        store = WorkingConceptStore()
        c = GroundedConcept(
            concept_ref=GroundedConceptRef(id="C-REDBLOCK"),
            lifecycle_state=ConceptLifecycle.GROUNDED,
            exemplar_entity_ids={"e_block_1"},
        )
        c.add_linguistic_alias("red block", confidence=1.0)
        store.register_concept(c)

        compiler = HCIRCompiler(concept_store=store)
        ast = compiler.parse_utterance("push the red block")
        assert ast.intent == IntentType.ACT
        assert ast.action == "push"
        assert ast.subject == "red block"

        resolved = compiler.resolve_grounded_entities(ast)
        assert "red block" in resolved
        assert isinstance(resolved["red block"], GroundedEntityRef)
        assert resolved["red block"].id == "e_block_1"

        # Compile instruction stream
        stream = compiler.compile(ast)
        assert stream.length >= 2
        # Verify ASSERT instruction contains resolved target
        assert_inst = next(i for i in stream.instructions if i.opcode == Opcode.ASSERT)
        assert assert_inst.params["node_data"]["target_ref"] == "e_block_1"
        assert assert_inst.params["node_data"]["action"] == "push"

        # Verify EXECUTE instruction targets e_block_1
        exec_inst = next(i for i in stream.instructions if i.opcode == Opcode.EXECUTE)
        assert exec_inst.params["target_entity_id"] == "e_block_1"
        assert exec_inst.params["capability"] == "motor_push"

    def test_compiler_ast_resolver_unseen_noun_triggers_active_inquiry(self) -> None:
        """Verify compiler generates UnresolvedRef and emits REQUEST_EVIDENCE for unseen nouns."""
        store = WorkingConceptStore()
        # Store has "red block" but NOT "mystery cylinder"
        c = GroundedConcept(
            concept_ref=GroundedConceptRef(id="C-REDBLOCK"),
            lifecycle_state=ConceptLifecycle.GROUNDED,
            exemplar_entity_ids={"e_block_1"},
        )
        c.add_linguistic_alias("red block", confidence=1.0)
        store.register_concept(c)

        compiler = HCIRCompiler(concept_store=store)
        ast = compiler.parse_utterance("push the mystery cylinder")
        assert ast.intent == IntentType.ACT
        assert ast.action == "push"
        assert ast.subject == "mystery cylinder"

        # Compile should detect unresolved reference and emit active inquiry
        stream = compiler.compile(ast)

        # Must not emit motor execution for an ungrounded hallucination
        assert not any(i.opcode == Opcode.EXECUTE for i in stream.instructions)

        # Must emit Opcode.QUERY with query_type='request_evidence' and Opcode.WAIT
        query_inst = stream.instructions[0]
        assert query_inst.opcode == Opcode.QUERY
        assert query_inst.params["query_type"] == "request_evidence"
        assert query_inst.params["outcome"] == DecisionOutcome.REQUEST_EVIDENCE.value
        assert "mystery cylinder" in query_inst.params["target_noun"]

        wait_inst = stream.instructions[1]
        assert wait_inst.opcode == Opcode.WAIT
        assert "evidence_resolved:" in wait_inst.params["condition"]

    def test_interpret_dialogue_structured_output(self) -> None:
        """Verify interpret_dialogue returns structured DialogueInterpretationResult."""
        store = WorkingConceptStore()
        c = GroundedConcept(
            concept_ref=GroundedConceptRef(id="C-GOLDENKEY"),
            lifecycle_state=ConceptLifecycle.GROUNDED,
            exemplar_entity_ids={"e_key_9"},
        )
        c.add_linguistic_alias("golden key", confidence=1.0)
        store.register_concept(c)

        compiler = HCIRCompiler(concept_store=store)

        # 1. Known target
        res_known: DialogueInterpretationResult = compiler.interpret_dialogue("grab the golden key")
        assert res_known.dialogue_act == DialogueAct.COMMAND
        assert res_known.confidence == 1.0
        assert len(res_known.action_proposals) == 1
        prop = res_known.action_proposals[0]
        assert prop.action_type == "grab"
        assert prop.target_ref == GroundedEntityRef("e_key_9")

        # 2. Unknown target
        res_unknown: DialogueInterpretationResult = compiler.interpret_dialogue(
            "grab the ancient relic"
        )
        assert res_unknown.confidence == 0.5
        assert len(res_unknown.action_proposals) == 1
        prop_unknown = res_unknown.action_proposals[0]
        assert prop_unknown.action_type == "grab"
        assert isinstance(prop_unknown.target_ref, UnresolvedRef)
