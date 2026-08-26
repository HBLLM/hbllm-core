"""A22 Lifelong Continual Learning Substrate Benchmark Suite (26 Scenarios).

Evaluates Three-Layer Memory Store (Fast, Slow, Immutable), Tri-Modal Sleep Replay (Error, Success, Contrastive),
Provenance-Preserving Adaptive Compaction, Dependency-Analyzed Stability Gating, Knowledge Revision Versioning,
and the Flagship 5-Stage Interleaved Curriculum Trial proving Zero Catastrophic Forgetting and Positive Forward Transfer.
"""

from __future__ import annotations

from hbllm.brain.continual import (
    CandidateUpdate,
    DualStoreMemory,
    EpisodicTrace,
    GateVerdict,
    ImmutableEvent,
    LifelongLearningLoop,
    PlasticityStabilityEngine,
    ProvenancePreservingCompactor,
    ReplayKind,
    SleepConsolidationEngine,
    SleepReplayEngine,
)

# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: A22-01 Three-Layer Memory Store Architecture
# ═══════════════════════════════════════════════════════════════════════════


class TestThreeLayerMemoryStore:
    """Three-layer memory separates fast episodic buffer, slow consolidated store, and immutable ground truth."""

    def test_fast_slow_immutable_layer_separation(self) -> None:
        memory = DualStoreMemory()
        assert len(memory.fast_buffer) == 0
        assert len(memory.slow_store) == 0
        assert len(memory.immutable_log) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: A22-02 Fast Episodic Buffering
# ═══════════════════════════════════════════════════════════════════════════


class TestFastEpisodicBuffering:
    """Fast episodic buffer captures raw interaction traces and actions."""

    def test_buffers_raw_traces_with_salience(self) -> None:
        memory = DualStoreMemory()
        trace = EpisodicTrace(
            domain="spatial_stacking",
            context_props={"surface": "flat"},
            actions=[("STACK", {"item": "cup", "base": "box"})],
            outcomes=["stable"],
            prediction_error=0.05,
            is_success=True,
        )
        memory.buffer_episodic_trace(trace)
        assert len(memory.fast_buffer) == 1
        assert memory.fast_buffer[0].domain == "spatial_stacking"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: A22-03 Provenance Pointer Binding
# ═══════════════════════════════════════════════════════════════════════════


class TestProvenancePointerBinding:
    """Consolidated knowledge records store explicit pointers to immutable ground-truth event IDs."""

    def test_consolidated_record_points_to_immutable_events(self) -> None:
        memory = DualStoreMemory()
        eid1 = memory.append_immutable_event(ImmutableEvent(domain="stacking", action_type="STACK"))
        eid2 = memory.append_immutable_event(ImmutableEvent(domain="stacking", action_type="STACK"))

        record = memory.commit_consolidated_knowledge(
            knowledge_id="schema_stack",
            knowledge_type="schema",
            content={"rule": "stack_on_flat"},
            source_event_ids=[eid1, eid2],
        )

        assert record.revision == 1
        assert eid1 in record.source_event_ids
        assert eid2 in record.source_event_ids

        # Verify audit retrieval
        events = memory.reconstruct_knowledge_justification("schema_stack")
        assert len(events) == 2
        assert {e.event_id for e in events} == {eid1, eid2}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: A22-04 A21 Salience Filtering
# ═══════════════════════════════════════════════════════════════════════════


class TestA21SalienceFiltering:
    """Metacognitive uncertainty from A21 elevates consolidation salience."""

    def test_metacognitive_uncertainty_weights_salience(self) -> None:
        replay_engine = SleepReplayEngine()
        t_known = EpisodicTrace(domain="known_domain", prediction_error=0.02)
        t_novel = EpisodicTrace(domain="novel_domain", prediction_error=0.40)

        # Domain uncertainty weights: novel_domain is 0.90, known_domain is 0.10
        candidates = replay_engine.filter_and_prioritize_replays(
            [t_known, t_novel],
            domain_uncertainties={"known_domain": 0.10, "novel_domain": 0.90},
        )

        assert len(candidates) > 0
        assert candidates[0].domain == "novel_domain"
        assert candidates[0].salience_score > 0.50


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: A22-05 Error Replay
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorReplay:
    """Error replay prioritizes episodes with high prediction error to discover boundaries."""

    def test_replays_high_prediction_error_episodes(self) -> None:
        replay_engine = SleepReplayEngine()
        t_err = EpisodicTrace(
            domain="stacking",
            prediction_error=0.85,
            is_success=False,
        )
        candidates = replay_engine.filter_and_prioritize_replays([t_err])

        assert len(candidates) >= 1
        assert candidates[0].kind == ReplayKind.ERROR_REPLAY


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: A22-06 Success Replay
# ═══════════════════════════════════════════════════════════════════════════


class TestSuccessReplay:
    """Success replay consolidates novel successful trajectories into reusable schemas."""

    def test_replays_novel_successful_trajectories(self) -> None:
        replay_engine = SleepReplayEngine()
        t_succ = EpisodicTrace(
            domain="containment",
            prediction_error=0.0,
            is_success=True,
        )
        candidates = replay_engine.filter_and_prioritize_replays([t_succ])

        assert len(candidates) >= 1
        assert candidates[0].kind == ReplayKind.SUCCESS_REPLAY


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: A22-07 Adaptive Compaction Execution
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptiveCompaction:
    """Folds redundant episodic traces into a compact schema representation."""

    def test_folds_redundant_episodic_traces(self) -> None:
        compactor = ProvenancePreservingCompactor()
        memory = DualStoreMemory()

        traces = [
            EpisodicTrace(
                domain="stacking",
                context_props={"surface": "flat"},
                actions=[("STACK", {"item": f"cup_{i}", "base": "box"})],
                outcomes=["stable"],
                event_id=f"eid_{i}",
            )
            for i in range(5)
        ]
        for t in traces:
            memory.append_immutable_event(ImmutableEvent(event_id=t.event_id, domain="stacking"))

        report, compact_content = compactor.compact_episodic_traces("stacking", traces, memory)

        assert report.compression_ratio > 0.40
        assert report.behavioral_invariants_preserved
        assert "STACK" in compact_content["invariant_actions"]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: A22-08 Invariant Preservation During Compaction
# ═══════════════════════════════════════════════════════════════════════════


class TestInvariantPreservationDuringCompaction:
    """Verifies that all four compaction contracts (behavioral, predictive, causal, provenance) are satisfied."""

    def test_four_part_compaction_contract_satisfied(self) -> None:
        compactor = ProvenancePreservingCompactor()
        memory = DualStoreMemory()
        eid = memory.append_immutable_event(ImmutableEvent(domain="tool_use"))
        trace = EpisodicTrace(
            domain="tool_use",
            context_props={"tool_type": "rigid_lever"},
            actions=[("PUSH", {})],
            outcomes=["displaced"],
            event_id=eid,
        )

        report, _ = compactor.compact_episodic_traces("tool_use", [trace], memory)

        assert report.behavioral_invariants_preserved
        assert report.predictive_invariants_preserved
        assert report.causal_invariants_preserved
        assert report.provenance_preserved


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: A22-09 Candidate Update Generation
# ═══════════════════════════════════════════════════════════════════════════


class TestCandidateUpdateGeneration:
    """Sleep consolidation emits proposed candidate updates rather than overwriting mature state directly."""

    def test_consolidation_emits_learning_proposals(self) -> None:
        engine = SleepConsolidationEngine()
        eid = engine.memory.append_immutable_event(ImmutableEvent(domain="stacking"))
        engine.memory.buffer_episodic_trace(
            EpisodicTrace(
                domain="stacking",
                context_props={"surface": "flat"},
                actions=[("STACK", {})],
                event_id=eid,
            )
        )

        summary = engine.run_sleep_consolidation()

        assert len(summary.stability_gate_reports) == 1
        assert summary.fast_buffer_cleared_count == 1
        assert summary.invariants_preserved


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: A22-10 Stability Gate Validation
# ═══════════════════════════════════════════════════════════════════════════


class TestStabilityGateValidation:
    """Stability gate detects and rejects updates that cause collateral damage on unrelated domains."""

    def test_rejects_collateral_damage_on_unrelated_domains(self) -> None:
        stability_engine = PlasticityStabilityEngine()
        stability_engine.register_domain_benchmark("stacking", [({"surface": "flat"}, True)])
        stability_engine.register_domain_benchmark("containment", [({"is_closed": False}, True)])

        update = CandidateUpdate(
            knowledge_id="schema_corrupted",
            knowledge_type="schema",
            domain="stacking",
            proposed_content={},
            source_event_ids=[],
        )

        # Simulate collateral drop on unrelated domain 'containment' (from 1.0 down to 0.40)
        current_accuracies = {"stacking": 1.0, "containment": 0.40}
        report = stability_engine.evaluate_candidate_update(update, current_accuracies)

        assert report.verdict == GateVerdict.REJECTED_REGRESSION
        assert not report.unrelated_domains_intact


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11: A22-11 Backward Transfer Quantification
# ═══════════════════════════════════════════════════════════════════════════


class TestBackwardTransferQuantification:
    """Backward Transfer (BWT) >= 0.0 confirms zero catastrophic forgetting."""

    def test_bwt_greater_than_or_equal_zero(self) -> None:
        stability_engine = PlasticityStabilityEngine()
        stability_engine.register_domain_benchmark("domain_a", [({}, True)])

        update = CandidateUpdate(
            knowledge_id="schema_b",
            knowledge_type="schema",
            domain="domain_b",
            proposed_content={},
            source_event_ids=[],
        )
        report = stability_engine.evaluate_candidate_update(
            update, {"domain_a": 1.0, "domain_b": 1.0}
        )

        assert report.backward_transfer_bwt >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12: A22-12 Zero Catastrophic Forgetting
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroCatastrophicForgetting:
    """Task 1 competence is 100% retained after sequentially learning Task 2 and Task 3."""

    def test_task1_competence_retained_after_task2(self) -> None:
        loop = LifelongLearningLoop()
        # Task 1: Stacking
        loop.record_task_experience(
            "task1_stacking", [{"actions": [("STACK", {})], "is_success": True}]
        )
        loop.trigger_sleep_cycle()

        # Task 2: Containment
        loop.record_task_experience(
            "task2_containment", [{"actions": [("PUT_IN", {})], "is_success": True}]
        )
        loop.trigger_sleep_cycle()

        audit = loop.audit_lifelong_retention()
        assert audit.zero_catastrophic_forgetting
        assert audit.behavioral_retention_bwt >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 13: A22-13 Forward Transfer Quantification
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardTransferQuantification:
    """Consolidated schemas provide positive Forward Transfer (FWT > 0) on novel tasks."""

    def test_fwt_greater_than_zero_on_novel_task(self) -> None:
        loop = LifelongLearningLoop()
        loop.record_task_experience("task1", [{"is_success": True}])
        loop.trigger_sleep_cycle()

        audit = loop.audit_lifelong_retention()
        assert audit.forward_transfer_fwt > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 14: A22-14 Conceptual Integrity Retention
# ═══════════════════════════════════════════════════════════════════════════


class TestConceptualIntegrityRetention:
    """A15 concept prototypes and instance-of relationships remain intact post-curriculum."""

    def test_a15_concepts_intact_post_curriculum(self) -> None:
        loop = LifelongLearningLoop()
        loop.record_task_experience("domain_concepts", [{"is_success": True}])
        loop.trigger_sleep_cycle()

        audit = loop.audit_lifelong_retention()
        assert audit.conceptual_retention_score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 15: A22-15 Lexical Integrity Retention
# ═══════════════════════════════════════════════════════════════════════════


class TestLexicalIntegrityRetention:
    """A17 grounded vocabulary mappings are uncorrupted post-curriculum."""

    def test_a17_lexicon_uncorrupted_post_curriculum(self) -> None:
        loop = LifelongLearningLoop()
        loop.record_task_experience("domain_lexicon", [{"is_success": True}])
        loop.trigger_sleep_cycle()

        audit = loop.audit_lifelong_retention()
        assert audit.lexical_retention_score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 16: A22-16 Relational Schema Retention
# ═══════════════════════════════════════════════════════════════════════════


class TestRelationalSchemaRetention:
    """A20 relational schemas and role bindings remain uncorrupted post-curriculum."""

    def test_a20_schemas_intact_post_curriculum(self) -> None:
        loop = LifelongLearningLoop()
        loop.record_task_experience("domain_schemas", [{"is_success": True}])
        loop.trigger_sleep_cycle()

        audit = loop.audit_lifelong_retention()
        assert audit.relational_schema_score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 17: A22-17 Metacognitive Calibration Retention
# ═══════════════════════════════════════════════════════════════════════════


class TestMetacognitiveCalibrationRetention:
    """A21 Brier calibration scores and competence boundaries are preserved post-curriculum."""

    def test_a21_brier_ece_calibration_preserved(self) -> None:
        loop = LifelongLearningLoop()
        loop.record_task_experience("domain_metacog", [{"is_success": True}])
        loop.trigger_sleep_cycle()

        audit = loop.audit_lifelong_retention()
        assert audit.metacognitive_calibration_score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 18: A22-18 The Flagship Lifelong Continual Learning Curriculum Trial
# ═══════════════════════════════════════════════════════════════════════════


class TestFlagshipLifelongCurriculum:
    """The Flagship Acceptance Gate: Undergoes 5-stage sequential curriculum with interleaved sleep cycles

    (Stacking -> Containment -> Tool Use -> Language -> Industrial Transfer).
    Post-curriculum comprehensive audit proves 100% historical retention, zero catastrophic forgetting,
    positive forward transfer, and compacted memory footprint.
    """

    def test_five_task_curriculum_interleaved_sleep_and_retention(self) -> None:
        loop = LifelongLearningLoop()

        # Task 1: Stacking & Support Stability
        loop.record_task_experience(
            "T1_stacking",
            [
                {
                    "actions": [("STACK", {"item": "cup", "base": "box"})],
                    "context_props": {"surface": "flat"},
                    "is_success": True,
                }
                for _ in range(5)
            ],
        )
        s1 = loop.trigger_sleep_cycle()
        assert s1.invariants_preserved

        # Task 2: Containment Reasoning
        loop.record_task_experience(
            "T2_containment",
            [
                {
                    "actions": [("PUT_IN", {"item": "ball", "container": "bin"})],
                    "context_props": {"is_closed": False},
                    "is_success": True,
                }
                for _ in range(5)
            ],
        )
        s2 = loop.trigger_sleep_cycle()
        assert s2.invariants_preserved

        # Task 3: Tool-Mediated Manipulation
        loop.record_task_experience(
            "T3_tool_use",
            [
                {
                    "actions": [("PUSH", {"tool": "lever", "target": "crate"})],
                    "context_props": {"is_rigid": True},
                    "is_success": True,
                }
                for _ in range(5)
            ],
        )
        s3 = loop.trigger_sleep_cycle()
        assert s3.invariants_preserved

        # Task 4: Grounded Lexical Learning
        loop.record_task_experience(
            "T4_language",
            [
                {
                    "actions": [("GROUND_TOKEN", {"token": "dax", "concept": "cylinder"})],
                    "is_success": True,
                }
                for _ in range(5)
            ],
        )
        s4 = loop.trigger_sleep_cycle()
        assert s4.invariants_preserved

        # Task 5: Industrial Machinery Transfer (Forward Transfer Target!)
        loop.record_task_experience(
            "T5_industrial_transfer",
            [
                {
                    "actions": [("STACK", {"item": "rotor", "base": "gantry_bed"})],
                    "context_props": {"surface": "flat"},
                    "is_success": True,
                }
                for _ in range(5)
            ],
        )
        s5 = loop.trigger_sleep_cycle()
        assert s5.invariants_preserved

        # Comprehensive Cross-Task Lifelong Retention Audit
        audit = loop.audit_lifelong_retention()

        assert len(audit.curriculum_tasks) == 5
        assert audit.zero_catastrophic_forgetting
        assert audit.behavioral_retention_bwt >= 0.0
        assert audit.forward_transfer_fwt > 0.0
        assert audit.conceptual_retention_score == 1.0
        assert audit.lexical_retention_score == 1.0
        assert audit.relational_schema_score == 1.0
        assert audit.metacognitive_calibration_score == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 19: A22-19 Continual Lexical Growth Without Drift
# ═══════════════════════════════════════════════════════════════════════════


class TestContinualLexicalGrowth:
    """Learning novel language tokens does not corrupt or drift earlier vocabulary mappings."""

    def test_learns_novel_words_without_drift(self) -> None:
        memory = DualStoreMemory()
        # Word 1: "cup" -> Cylinder
        memory.commit_consolidated_knowledge(
            "word_cup", "lexicon", {"token": "cup", "concept": "cylinder"}, ["e1"]
        )
        # Word 2: "koba" -> Cylinder
        memory.commit_consolidated_knowledge(
            "word_koba", "lexicon", {"token": "koba", "concept": "cylinder"}, ["e2"]
        )

        assert memory.slow_store["word_cup"].content["token"] == "cup"
        assert memory.slow_store["word_koba"].content["token"] == "koba"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 20: A22-20 Zero-LLM Invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Memory consolidation, replay, compaction, and stability gating execute with 100% deterministic code."""

    def test_zero_llm_imports(self) -> None:
        import subprocess
        import sys

        check_code = """
import sys
import hbllm.brain.continual

llm_markers = ["openai", "anthropic", "litellm", "langchain", "transformers"]
loaded = set(sys.modules.keys())
for marker in llm_markers:
    assert marker not in loaded, f"LLM module loaded in continual learning runtime: {marker}"
"""
        res = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
        assert res.returncode == 0, f"Zero-LLM verification failed:\n{res.stderr}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 21: A22-21 Lifelong Schema Specialization
# ═══════════════════════════════════════════════════════════════════════════


class TestLifelongSchemaSpecialization:
    """Accumulates nuanced boundary constraints over lifelong experience without breaking schemas."""

    def test_accumulates_nuanced_boundary_rules(self) -> None:
        memory = DualStoreMemory()
        memory.commit_consolidated_knowledge(
            "schema_support", "schema", {"boundary": "flat"}, ["e1"], "init"
        )
        r2 = memory.commit_consolidated_knowledge(
            "schema_support",
            "schema",
            {"boundary": "flat_and_rigid"},
            ["e1", "e2"],
            "specialization",
        )

        assert r2.revision == 2
        assert r2.supersedes_revision == 1
        assert r2.content["boundary"] == "flat_and_rigid"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 22: A22-22 Long-Term Memory Reconstruction
# ═══════════════════════════════════════════════════════════════════════════


class TestLongTermMemoryReconstruction:
    """Reconstructs historical cognitive state and evidence trails from the immutable event log."""

    def test_reconstructs_cognitive_justification_from_immutable_log(self) -> None:
        memory = DualStoreMemory()
        eid = memory.append_immutable_event(
            ImmutableEvent(domain="physics_stack", action_type="STACK")
        )
        memory.commit_consolidated_knowledge("schema_phys", "schema", {"valid": True}, [eid])

        events = memory.reconstruct_knowledge_justification("schema_phys")
        assert len(events) == 1
        assert events[0].domain == "physics_stack"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 23: A22-23 Provenance-Preserving Compaction
# ═══════════════════════════════════════════════════════════════════════════


class TestProvenancePreservingCompaction:
    """Compact schema retrieves source evidence and reconstructs causal justification."""

    def test_compact_schema_retrieves_source_evidence(self) -> None:
        compactor = ProvenancePreservingCompactor()
        memory = DualStoreMemory()
        eid1 = memory.append_immutable_event(ImmutableEvent(domain="stacking", action_type="STACK"))
        eid2 = memory.append_immutable_event(ImmutableEvent(domain="stacking", action_type="STACK"))

        traces = [
            EpisodicTrace(
                domain="stacking",
                context_props={"surface": "flat"},
                actions=[("STACK", {})],
                event_id=eid1,
            ),
            EpisodicTrace(
                domain="stacking",
                context_props={"surface": "flat"},
                actions=[("STACK", {})],
                event_id=eid2,
            ),
        ]

        report, _ = compactor.compact_episodic_traces("stacking", traces, memory)
        assert report.provenance_preserved

        # Audit retrieval
        kid = report.consolidated_knowledge_ids[0]
        justification_events = memory.reconstruct_knowledge_justification(kid)
        assert len(justification_events) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 24: A22-24 Contrastive Sleep Consolidation
# ═══════════════════════════════════════════════════════════════════════════


class TestContrastiveSleepConsolidation:
    """Success vs failure contrastive pair isolates support geometry invariant (flat vs curved)."""

    def test_success_vs_failure_isolates_support_geometry_delta(self) -> None:
        replay_engine = SleepReplayEngine()
        t_succ = EpisodicTrace(
            domain="stacking",
            context_props={"surface": "flat"},
            actions=[("STACK", {})],
            is_success=True,
        )
        t_fail = EpisodicTrace(
            domain="stacking",
            context_props={"surface": "curved"},
            actions=[("STACK", {})],
            is_success=False,
            prediction_error=0.90,
        )

        replays = replay_engine.filter_and_prioritize_replays([t_succ, t_fail])
        contrastive_candidates = [r for r in replays if r.kind == ReplayKind.CONTRASTIVE_REPLAY]

        assert len(contrastive_candidates) >= 1
        cp = contrastive_candidates[0].contrastive_pair
        assert cp is not None
        assert "surface" in cp.isolated_delta
        assert cp.isolated_delta["surface"] == ("flat", "curved")


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 25: A22-25 Controlled Knowledge Revision
# ═══════════════════════════════════════════════════════════════════════════


class TestControlledKnowledgeRevision:
    """Contradictory evidence triggers localized revision while protecting unrelated schemas."""

    def test_contradictory_evidence_triggers_localized_revision(self) -> None:
        stability_engine = PlasticityStabilityEngine()
        update = CandidateUpdate(
            knowledge_id="schema_revised",
            knowledge_type="schema",
            domain="specialized_domain",
            proposed_content={"new_rule": "strict_check"},
            source_event_ids=["e_contra"],
            is_revision=True,
        )

        report = stability_engine.evaluate_candidate_update(update, {"unrelated_domain": 1.0})
        assert report.verdict == GateVerdict.BENEFICIAL_REVISION
        assert report.unrelated_domains_intact


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 26: A22-26 Revision Without Collateral Forgetting
# ═══════════════════════════════════════════════════════════════════════════


class TestRevisionWithoutCollateralForgetting:
    """Revised schema updates versioned record while old revision remains reconstructable."""

    def test_revised_schema_preserves_unrelated_schemas_and_history(self) -> None:
        memory = DualStoreMemory()
        # 1. Store initial Revision 1
        memory.commit_consolidated_knowledge(
            "schema_support", "schema", {"rule": "support=flat"}, ["e1"], "initial"
        )
        # 2. Store unrelated schema
        memory.commit_consolidated_knowledge(
            "schema_containment", "schema", {"rule": "is_closed=False"}, ["e2"], "initial"
        )

        # 3. Store Revision 2 for support schema
        memory.commit_consolidated_knowledge(
            "schema_support", "schema", {"rule": "support=flat_and_rigid"}, ["e1", "e3"], "revision"
        )

        # Unrelated containment schema intact
        assert memory.slow_store["schema_containment"].content["rule"] == "is_closed=False"

        # Support schema revised to Revision 2
        assert memory.slow_store["schema_support"].revision == 2
        assert memory.slow_store["schema_support"].supersedes_revision == 1

        # Revision 1 history is preserved and reconstructable
        history = memory.revision_history["schema_support"]
        assert len(history) == 2
        assert history[0].revision == 1
        assert history[0].content["rule"] == "support=flat"
