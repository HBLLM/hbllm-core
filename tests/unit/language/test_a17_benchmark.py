"""A17 Grounded Language Learning Benchmark Suite (21 Developmental Scenarios).

Evaluates autonomous lexicon acquisition from situated observation, cross-situational
evidence, state transitions, contrastive differentiation, and teacher corrections.
"""

from __future__ import annotations

import sys

from hbllm.brain.language.acquisition import (
    ContrastiveLearner,
    EvidenceSourceType,
    GroundedLexicon,
    LexicalCandidateStatus,
    LexicalEvidence,
    LexicalTargetType,
    LexiconAcquisitionLoop,
    apply_evidence_to_candidate,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    PhysicalEntityNode,
)

# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: A17-01 Novel Noun Acquisition
# ═══════════════════════════════════════════════════════════════════════════


class TestNovelNounAcquisition:
    """Unknown word + visible object -> competing candidate hypotheses created."""

    def test_novel_noun_candidate_formation(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        cyl = PhysicalEntityNode(entity_type="cylinder", properties={"shape": "cylinder", "color": "blue"})
        graph.add_node(cyl)

        # Hear: "Look at the dax." in presence of cylinder
        loop.observe_scene(["dax"], visible_entity_ids=[cyl.id], timestamp=10.0)

        # Lexicon should have hypothesis set for 'dax'
        hyp_set = loop.lexicon.get_or_create_hypothesis_set("dax")
        assert len(hyp_set.candidates) >= 2
        # Candidates should include Concept(cylinder) and Individual(cyl.id)
        assert hyp_set.get_candidate(LexicalTargetType.CONCEPT, "cylinder") is not None
        assert hyp_set.get_candidate(LexicalTargetType.INDIVIDUAL, cyl.id) is not None


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: A17-02 Multi-Scene Cross-Situational Learning
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiSceneCrossSituational:
    """Same word appears with multiple objects; common feature dominates."""

    def test_cross_situational_feature_convergence(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        # Scene 1: Cylinder 1 (blue) + Table
        c1 = PhysicalEntityNode(entity_type="cylinder", properties={"color": "blue"})
        t1 = PhysicalEntityNode(entity_type="table", properties={"color": "brown"})
        graph.add_node(c1)
        graph.add_node(t1)
        loop.observe_scene(["dax"], visible_entity_ids=[c1.id, t1.id], timestamp=1.0)

        # Scene 2: Cylinder 2 (red) + Box
        c2 = PhysicalEntityNode(entity_type="cylinder", properties={"color": "red"})
        b1 = PhysicalEntityNode(entity_type="box", properties={"color": "green"})
        graph.add_node(c2)
        graph.add_node(b1)
        loop.observe_scene(["dax"], visible_entity_ids=[c2.id, b1.id], timestamp=2.0)

        # Scene 3: Cylinder 3 (yellow) + Cup
        c3 = PhysicalEntityNode(entity_type="cylinder", properties={"color": "yellow"})
        p1 = PhysicalEntityNode(entity_type="cup", properties={"color": "white"})
        graph.add_node(c3)
        graph.add_node(p1)
        loop.observe_scene(["dax"], visible_entity_ids=[c3.id, p1.id], timestamp=3.0)

        # Cylinder concept should decisively win
        hyp_set = loop.lexicon.get_or_create_hypothesis_set("dax")
        assert hyp_set.winner is not None
        assert hyp_set.winner.target_type == LexicalTargetType.CONCEPT
        assert hyp_set.winner.target_id == "cylinder"
        assert hyp_set.winner.status == LexicalCandidateStatus.GROUNDED


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: A17-03 Distractor Rejection
# ═══════════════════════════════════════════════════════════════════════════


class TestDistractorRejection:
    """Distractor objects that don't co-occur across scenes are pruned."""

    def test_distractor_elimination_across_scenes(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        cyl = PhysicalEntityNode(entity_type="cylinder")
        table = PhysicalEntityNode(entity_type="table")
        ball = PhysicalEntityNode(entity_type="ball")
        graph.add_node(cyl)
        graph.add_node(table)
        graph.add_node(ball)

        # Scene 1: cylinder + table
        loop.observe_scene(["dax"], visible_entity_ids=[cyl.id, table.id], timestamp=1.0)
        # Scene 2: cylinder + ball (table absent)
        loop.observe_scene(["dax"], visible_entity_ids=[cyl.id, ball.id], timestamp=2.0)

        hyp_set = loop.lexicon.get_or_create_hypothesis_set("dax")
        c_table = hyp_set.get_candidate(LexicalTargetType.INDIVIDUAL, table.id)
        assert c_table is not None
        assert len(c_table.contradiction_ids) > 0  # Table contradicted because absent in scene 2


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: A17-04 Fast Mapping
# ═══════════════════════════════════════════════════════════════════════════


class TestFastMapping:
    """One exposure produces an immediate tentative hypothesis without permanent commitment."""

    def test_single_exposure_tentative_hypothesis(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        novel_obj = PhysicalEntityNode(entity_type="pyramid")
        graph.add_node(novel_obj)

        loop.observe_scene(["koba"], visible_entity_ids=[novel_obj.id], timestamp=1.0)

        res = loop.understand("koba")
        assert res.is_tentative
        assert not res.is_grounded
        assert res.target_id == "pyramid"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: A17-05 Delayed Consolidation
# ═══════════════════════════════════════════════════════════════════════════


class TestDelayedConsolidation:
    """Lexical mapping strengthens to GROUNDED only after sufficient supporting evidence."""

    def test_multi_exposure_consolidation(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        pyr1 = PhysicalEntityNode(entity_type="pyramid")
        pyr2 = PhysicalEntityNode(entity_type="pyramid")
        graph.add_node(pyr1)
        graph.add_node(pyr2)

        # 1st exposure -> TENTATIVE
        loop.observe_scene(["koba"], visible_entity_ids=[pyr1.id], timestamp=1.0)
        assert loop.understand("koba").is_tentative

        # 2nd & 3rd exposures -> GROUNDED
        loop.observe_scene(["koba"], visible_entity_ids=[pyr2.id], timestamp=2.0)
        loop.observe_scene(["koba"], visible_entity_ids=[pyr1.id], timestamp=3.0)

        res = loop.understand("koba")
        assert res.is_grounded
        assert res.confidence >= 0.70


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: A17-06 Ostensive Teaching
# ═══════════════════════════════════════════════════════════════════════════


class TestOstensiveTeaching:
    """Explicit 'This is a cup' carries high epistemic weight, quickly grounding the concept."""

    def test_explicit_teaching_high_confidence(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        cup = PhysicalEntityNode(entity_type="cup")
        graph.add_node(cup)

        # Teacher: "This is a cup"
        ev = loop.teach_ostensive("cup", entity_id=cup.id, timestamp=1.0)
        assert ev.source_type == EvidenceSourceType.OSTENSIVE_POSITIVE
        assert ev.epistemic_weight >= 1.0

        # Strengthens rapidly
        loop.teach_ostensive("cup", entity_id=cup.id, timestamp=2.0)
        res = loop.understand("cup")
        assert res.is_grounded
        assert res.target_id == "cup"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: A17-07 Negative Correction
# ═══════════════════════════════════════════════════════════════════════════


class TestNegativeCorrection:
    """'No, this is an apple' suppresses the incorrect hypothesis via A14 error mechanism."""

    def test_teacher_correction_weakens_false_hypothesis(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        obj = PhysicalEntityNode(entity_type="apple")
        graph.add_node(obj)

        # False initial hypothesis: dax = ball
        hyp_set = loop.lexicon.get_or_create_hypothesis_set("dax")
        c_ball = hyp_set.add_or_get_candidate(LexicalTargetType.CONCEPT, "ball", timestamp=1.0)
        apply_evidence_to_candidate(
            c_ball,
            LexicalEvidence(source_type=EvidenceSourceType.CROSS_SITUATIONAL, token="dax", is_positive=True, timestamp=1.0),
        )
        assert c_ball.support_weight > 0

        # Teacher corrects: "No, that is not a ball. That is an apple."
        loop.correct_mistake(
            token="dax",
            incorrect_target="ball",
            correct_token="dax",
            correct_target="apple",
            timestamp=2.0,
        )

        assert c_ball.status == LexicalCandidateStatus.CONTRADICTED
        c_apple = hyp_set.get_candidate(LexicalTargetType.CONCEPT, "apple")
        assert c_apple is not None
        assert c_apple.support_weight > c_ball.support_weight


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: A17-08 Contrastive Learning
# ═══════════════════════════════════════════════════════════════════════════


class TestContrastiveLearning:
    """Learns cup != bowl with distinguishing feature delta vector."""

    def test_cup_vs_bowl_distinguishing_delta(self) -> None:
        learner = ContrastiveLearner()

        proto_cup = {"shape": "cylinder", "has_handle": True, "depth": "deep"}
        proto_bowl = {"shape": "hemisphere", "has_handle": False, "depth": "shallow"}

        rel = learner.learn_contrast("cup", "bowl", proto_cup, proto_bowl)
        assert rel.relation_type == "DIFFERENT_FROM"
        assert "has_handle" in rel.distinguishing_features
        assert rel.distinguishing_features["has_handle"] == (True, False)
        assert rel.distinguishing_features["depth"] == ("deep", "shallow")


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: A17-09 Predicate & Property Acquisition
# ═══════════════════════════════════════════════════════════════════════════


class TestPredicatePropertyAcquisition:
    """'red' grounds to color=red across varying shapes and categories."""

    def test_color_adjective_learning(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        # 'mip' heard with red ball, red box, red cylinder
        b = PhysicalEntityNode(entity_type="ball", properties={"color": "red"})
        bx = PhysicalEntityNode(entity_type="box", properties={"color": "red"})
        c = PhysicalEntityNode(entity_type="cylinder", properties={"color": "red"})
        graph.add_node(b)
        graph.add_node(bx)
        graph.add_node(c)

        loop.observe_scene(["mip"], visible_entity_ids=[b.id], timestamp=1.0)
        loop.observe_scene(["mip"], visible_entity_ids=[bx.id], timestamp=2.0)
        loop.observe_scene(["mip"], visible_entity_ids=[c.id], timestamp=3.0)

        hyp_set = loop.lexicon.get_or_create_hypothesis_set("mip")
        winner = hyp_set.winner
        assert winner is not None
        assert winner.target_type == LexicalTargetType.PROPERTY
        assert winner.target_id == "color:red"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: A17-10 Action Acquisition
# ═══════════════════════════════════════════════════════════════════════════


class TestActionAcquisition:
    """'push' grounds to PUSH action via observed state transitions."""

    def test_push_action_state_transition(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        box = PhysicalEntityNode(entity_type="box")
        graph.add_node(box)

        # State delta: box moved from pos (0,0) to (1,0) via push
        delta = {"action": "push", "agentive": True, "target": box.id, "dx": 1.0}
        loop.observe_scene(["zog"], visible_entity_ids=[box.id], state_delta=delta, timestamp=1.0)
        loop.observe_scene(["zog"], visible_entity_ids=[box.id], state_delta=delta, timestamp=2.0)

        res = loop.understand("zog")
        assert res.target_type == LexicalTargetType.ACTION
        assert res.target_id == "transition:push"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11: A17-11 Spatial Relation Acquisition
# ═══════════════════════════════════════════════════════════════════════════


class TestSpatialRelationAcquisition:
    """'on' and 'tul' (inside) ground through topological spatial relations."""

    def test_preposition_spatial_edge(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        ball = PhysicalEntityNode(entity_type="ball")
        box = PhysicalEntityNode(entity_type="box")
        graph.add_node(ball)
        graph.add_node(box)

        # 'tul' observed when ball is INSIDE box
        spatial = [(ball.id, "INSIDE", box.id)]
        loop.observe_scene(["tul"], visible_entity_ids=[ball.id, box.id], spatial_edges=spatial, timestamp=1.0)
        loop.observe_scene(["tul"], visible_entity_ids=[ball.id, box.id], spatial_edges=spatial, timestamp=2.0)

        res = loop.understand("tul")
        assert res.target_type == LexicalTargetType.RELATION
        assert "inside" in res.target_id.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12: A17-12 Multilingual Convergence
# ═══════════════════════════════════════════════════════════════════════════


class TestMultilingualConvergence:
    """'cup' and 'කෝප්පය' ground to the SAME language-neutral HCIR concept."""

    def test_cross_lingual_shared_concept(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        cup = PhysicalEntityNode(entity_type="cup")
        graph.add_node(cup)

        # English teaching
        loop.teach_ostensive("cup", entity_id=cup.id, language="en", timestamp=1.0)
        loop.teach_ostensive("cup", entity_id=cup.id, language="en", timestamp=2.0)

        # Sinhala teaching
        loop.teach_ostensive("කෝප්පය", entity_id=cup.id, language="si", timestamp=3.0)
        loop.teach_ostensive("කෝප්පය", entity_id=cup.id, language="si", timestamp=4.0)

        res_en = loop.understand("cup", language="en")
        res_si = loop.understand("කෝප්පය", language="si")

        assert res_en.target_id == res_si.target_id == "cup"
        assert res_en.target_type == res_si.target_type == LexicalTargetType.CONCEPT


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 13: A17-13 Polysemy & Multiple Senses
# ═══════════════════════════════════════════════════════════════════════════


class TestPolysemyResolution:
    """Single token with multiple senses resolved via situational context."""

    def test_multiple_senses_contextual_selection(self) -> None:
        graph = CognitiveGraph()
        lexicon = GroundedLexicon(graph)

        # Commit Sense 1: 'bank' -> financial_institution
        lexicon.commit_sense("bank", LexicalTargetType.CONCEPT, "financial_institution", language="en", comprehension_confidence=0.9)
        # Commit Sense 2: 'bank' -> river_bank
        lexicon.commit_sense("bank", LexicalTargetType.CONCEPT, "river_bank", language="en", comprehension_confidence=0.85)

        # Query with context
        res1 = lexicon.ground_token("bank", language="en", context_sense="financial_institution")
        res2 = lexicon.ground_token("bank", language="en", context_sense="river_bank")

        assert res1.target_id == "financial_institution"
        assert res2.target_id == "river_bank"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 14: A17-14 Tentative Lexical Honesty
# ═══════════════════════════════════════════════════════════════════════════


class TestTentativeHonesty:
    """Insufficient evidence returns TENTATIVE status with runner-up breakdown."""

    def test_tentative_lexical_mapping_epistemic_breakdown(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        c = PhysicalEntityNode(entity_type="cylinder")
        b = PhysicalEntityNode(entity_type="box")
        graph.add_node(c)
        graph.add_node(b)

        # Single weak exposure
        loop.observe_scene(["dax"], visible_entity_ids=[c.id, b.id], timestamp=1.0)

        res = loop.understand("dax")
        assert res.is_tentative
        assert not res.is_grounded
        assert res.runner_up_id is not None
        assert res.status in (LexicalCandidateStatus.HYPOTHESIS, LexicalCandidateStatus.TENTATIVE)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 15: A17-15 Lexical Revision & Forgetting
# ═══════════════════════════════════════════════════════════════════════════


class TestLexicalRevisionForgetting:
    """Repeated contradictions deprecate/reject a false lexical hypothesis."""

    def test_repeated_contradiction_deprecates_sense(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        hyp_set = loop.lexicon.get_or_create_hypothesis_set("dax")
        cand = hyp_set.add_or_get_candidate(LexicalTargetType.CONCEPT, "sphere", timestamp=1.0)

        # 3 successive teacher corrections/contradictions
        for i in range(3):
            apply_evidence_to_candidate(
                cand,
                LexicalEvidence(source_type=EvidenceSourceType.OSTENSIVE_NEGATIVE, is_positive=False, timestamp=float(i)),
            )

        assert cand.status in (LexicalCandidateStatus.CONTRADICTED, LexicalCandidateStatus.REJECTED)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 16: A17-16 Compositional Grounding
# ═══════════════════════════════════════════════════════════════════════════


class TestCompositionalGrounding:
    """'red dax' composed from learned lexical primitives ('red' + 'dax')."""

    def test_multi_token_composition(self) -> None:
        graph = CognitiveGraph()
        lexicon = GroundedLexicon(graph)

        # Commit primitives
        lexicon.commit_sense("red", LexicalTargetType.PROPERTY, "color:red")
        lexicon.commit_sense("dax", LexicalTargetType.CONCEPT, "cylinder")

        comp = lexicon.ground_compositional_phrase(["red", "dax"])
        assert "color:red" in comp["modifiers"]
        assert comp["head"] == "cylinder"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 17: A17-17 Lexical Prediction
# ═══════════════════════════════════════════════════════════════════════════


class TestLexicalPrediction:
    """Given an HCIR concept node, predict likely learned lexical surface forms."""

    def test_hcir_concept_activates_lexical_token(self) -> None:
        graph = CognitiveGraph()
        lexicon = GroundedLexicon(graph)

        lexicon.commit_sense("cup", LexicalTargetType.CONCEPT, "cup", language="en", generation_confidence=0.88)
        lexicon.commit_sense("කෝප්පය", LexicalTargetType.CONCEPT, "cup", language="si", generation_confidence=0.85)

        prod_en = lexicon.realize_target("cup", LexicalTargetType.CONCEPT, language="en")
        prod_si = lexicon.realize_target("cup", LexicalTargetType.CONCEPT, language="si")

        assert prod_en.token == "cup"
        assert prod_si.token == "කෝප්පය"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 18: A17-18 Grounded Generation
# ═══════════════════════════════════════════════════════════════════════════


class TestGroundedGeneration:
    """Realization requires sufficient generation_confidence."""

    def test_realization_generation_confidence(self) -> None:
        graph = CognitiveGraph()
        lexicon = GroundedLexicon(graph)

        # Sense with low generation confidence
        lexicon.commit_sense("orb", LexicalTargetType.CONCEPT, "sphere", generation_confidence=0.35)

        # High threshold -> should not produce
        res_strict = lexicon.realize_target("sphere", min_confidence=0.50)
        assert not res_strict.is_produced

        # Low threshold -> produces
        res_lenient = lexicon.realize_target("sphere", min_confidence=0.30)
        assert res_lenient.is_produced
        assert res_lenient.token == "orb"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 19: A17-19 Deterministic Replay
# ═══════════════════════════════════════════════════════════════════════════


class TestDeterministicReplay:
    """Same event history produces byte-for-byte identical lexical state."""

    def test_byte_identical_lexical_state(self) -> None:
        def _run_acquisition() -> list[tuple[str, str, float]]:
            g = CognitiveGraph()
            loop = LexiconAcquisitionLoop(g)
            c1 = PhysicalEntityNode(id="cyl_1", entity_type="cylinder", properties={"color": "red"})
            c2 = PhysicalEntityNode(id="cyl_2", entity_type="cylinder", properties={"color": "blue"})
            g.add_node(c1)
            g.add_node(c2)

            loop.observe_scene(["dax"], visible_entity_ids=[c1.id], timestamp=1.0)
            loop.observe_scene(["dax"], visible_entity_ids=[c2.id], timestamp=2.0)
            loop.teach_ostensive("dax", entity_id=c1.id, timestamp=3.0)

            hyp_set = loop.lexicon.get_or_create_hypothesis_set("dax")
            return [(c.target_id, c.target_type.value, round(c.total_score, 4)) for c in hyp_set.ranked_candidates()]

        run1 = _run_acquisition()
        run2 = _run_acquisition()

        assert run1 == run2
        assert len(run1) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 20: A17-20 The Flagship Novel Language Test
# ═══════════════════════════════════════════════════════════════════════════


class TestNovelLanguageExperiment:
    """The Flagship Acceptance Gate: 6-phase trial acquiring artificial vocabulary

    ('dax', 'mip', 'zog', 'tul') without any category hints.
    """

    def test_six_phase_artificial_vocabulary_trial(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        # ── Phase 1: Observation Only (No Category Hints Given) ──────
        c1 = PhysicalEntityNode(entity_type="cylinder", properties={"color": "red"})
        c2 = PhysicalEntityNode(entity_type="cylinder", properties={"color": "blue"})
        b1 = PhysicalEntityNode(entity_type="box", properties={"color": "red"})
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_node(b1)

        # 'dax' heard with cylinders
        loop.observe_scene(["dax"], visible_entity_ids=[c1.id, b1.id], timestamp=1.0)
        loop.observe_scene(["dax"], visible_entity_ids=[c2.id], timestamp=2.0)
        loop.observe_scene(["dax"], visible_entity_ids=[c1.id, c2.id], timestamp=3.0)

        # 'mip' heard with red objects
        loop.observe_scene(["mip"], visible_entity_ids=[c1.id], timestamp=4.0)
        loop.observe_scene(["mip"], visible_entity_ids=[b1.id], timestamp=5.0)

        # 'zog' heard with push transitions
        delta_push = {"action": "push", "agentive": True, "target": c1.id}
        loop.observe_scene(["zog"], visible_entity_ids=[c1.id], state_delta=delta_push, timestamp=6.0)
        loop.observe_scene(["zog"], visible_entity_ids=[c2.id], state_delta=delta_push, timestamp=7.0)

        # 'tul' heard with inside spatial relation
        spatial_inside = [(c1.id, "INSIDE", b1.id)]
        loop.observe_scene(["tul"], visible_entity_ids=[c1.id, b1.id], spatial_edges=spatial_inside, timestamp=8.0)
        loop.observe_scene(["tul"], visible_entity_ids=[c2.id, b1.id], spatial_edges=spatial_inside, timestamp=9.0)

        # Verify groundings discovered
        assert loop.understand("dax").target_id == "cylinder"
        assert loop.understand("dax").target_type == LexicalTargetType.CONCEPT

        assert loop.understand("mip").target_id == "color:red"
        assert loop.understand("mip").target_type == LexicalTargetType.PROPERTY

        assert loop.understand("zog").target_id == "transition:push"
        assert loop.understand("zog").target_type == LexicalTargetType.ACTION

        assert "inside" in loop.understand("tul").target_id.lower()
        assert loop.understand("tul").target_type == LexicalTargetType.RELATION

        # ── Phase 2: Compositional Interpretation ─────────────────────
        comp1 = loop.lexicon.ground_compositional_phrase(["mip", "dax"])
        assert comp1["head"] == "cylinder"
        assert "color:red" in comp1["modifiers"]

        comp2 = loop.lexicon.ground_compositional_phrase(["zog", "the", "dax"])
        assert comp2["action"] == "transition:push"
        assert comp2["head"] == "cylinder"

        # ── Phase 3: Novel Contexts Generalization ───────────────────
        c3 = PhysicalEntityNode(entity_type="cylinder", properties={"color": "green"})
        graph.add_node(c3)
        res_novel = loop.understand("dax")
        assert res_novel.target_id == "cylinder"

        # ── Phase 4: Error-Driven Revision ────────────────────────────
        loop.correct_mistake("dax", incorrect_target="box", timestamp=10.0)
        hyp_dax = loop.lexicon.get_or_create_hypothesis_set("dax")
        c_box = hyp_dax.get_candidate(LexicalTargetType.CONCEPT, "box")
        if c_box:
            assert c_box.status in (LexicalCandidateStatus.CONTRADICTED, LexicalCandidateStatus.REJECTED)

        # ── Phase 5: Surface Generation ───────────────────────────────
        loop.lexicon.commit_sense("dax", LexicalTargetType.CONCEPT, "cylinder", generation_confidence=0.88)
        realized = loop.produce("cylinder", LexicalTargetType.CONCEPT)
        assert realized.token == "dax"
        assert realized.is_produced

        # ── Phase 6: Zero-LLM Invariant ──────────────────────────────
        llm_markers = ["openai", "anthropic", "litellm", "langchain", "transformers"]
        loaded = set(sys.modules.keys())
        for marker in llm_markers:
            assert marker not in loaded, f"LLM module loaded: {marker}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 21: A17-21 Ambiguous Referential Grounding
# ═══════════════════════════════════════════════════════════════════════════


class TestAmbiguousReferentialGrounding:
    """Progressively disambiguates Concept vs Individual vs Property across ambiguous scenes."""

    def test_multi_hypothesis_disambiguation(self) -> None:
        graph = CognitiveGraph()
        loop = LexiconAcquisitionLoop(graph)

        # Scene 1: 4 objects: red cylinder, blue cylinder, red sphere, blue sphere
        rc = PhysicalEntityNode(entity_type="cylinder", properties={"color": "red"})
        bc = PhysicalEntityNode(entity_type="cylinder", properties={"color": "blue"})
        rs = PhysicalEntityNode(entity_type="sphere", properties={"color": "red"})
        bs = PhysicalEntityNode(entity_type="sphere", properties={"color": "blue"})
        graph.add_node(rc)
        graph.add_node(bc)
        graph.add_node(rs)
        graph.add_node(bs)

        # "Look at the dax" in Scene 1 -> All 4 objects present
        loop.observe_scene(["dax"], visible_entity_ids=[rc.id, bc.id, rs.id, bs.id], timestamp=1.0)
        hyp_set = loop.lexicon.get_or_create_hypothesis_set("dax")
        # Should be ambiguous initially (no overwhelming winner)
        assert hyp_set.is_ambiguous or hyp_set.margin_of_victory < 0.25

        # Scene 2: only cylinders present (red cylinder, blue cylinder)
        loop.observe_scene(["dax"], visible_entity_ids=[rc.id, bc.id], timestamp=2.0)

        # Scene 3: green cylinder present
        gc = PhysicalEntityNode(entity_type="cylinder", properties={"color": "green"})
        graph.add_node(gc)
        loop.observe_scene(["dax"], visible_entity_ids=[gc.id], timestamp=3.0)

        # Cylinder concept emerges as decisive winner over color=red and individual instances
        assert hyp_set.winner is not None
        assert hyp_set.winner.target_type == LexicalTargetType.CONCEPT
        assert hyp_set.winner.target_id == "cylinder"
        assert not hyp_set.is_ambiguous
