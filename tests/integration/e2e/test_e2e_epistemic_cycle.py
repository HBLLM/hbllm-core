"""E2E test: full epistemic discovery cycle.

Runs the complete epistemic pipeline end-to-end:
    Curiosity → Ideas → Hypotheses → Predictions → Evidence →
    Belief Revision → Memory Recording → Calibration Report
"""

from __future__ import annotations

import tempfile

import pytest

from hbllm.brain.epistemics.belief_manager import DiscoveryBeliefManager
from hbllm.brain.epistemics.calibration import EpistemicCalibrationEngine
from hbllm.brain.epistemics.counterfactual import CounterfactualReasoner
from hbllm.brain.epistemics.epistemic_loop import EpistemicLoop
from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory
from hbllm.brain.epistemics.interfaces import PredictionOutcome
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.types import BeliefConfidence, EvidenceStrength, FalsificationStatus


class TestFullEpistemicCycle:
    """End-to-end: question → discovery → belief → revision → calibration."""

    @pytest.mark.asyncio
    async def test_complete_discovery_cycle(self) -> None:
        """Run a full discovery cycle and verify every stage produces output."""
        with tempfile.TemporaryDirectory() as td:
            graph = CognitiveGraph()
            workspace = DiscoveryWorkspace(data_dir=td, graph=graph)
            memory = EpistemicMemory(data_dir=td)
            calibrator = EpistemicCalibrationEngine(memory=memory)
            counterfactual = CounterfactualReasoner(graph=graph)
            belief_manager = DiscoveryBeliefManager(graph=graph)

            loop = EpistemicLoop(
                graph=graph,
                workspace=workspace,
                memory=memory,
                calibration=calibrator,
                counterfactual=counterfactual,
                calibration_interval=2,
            )

            # ── Phase 1: Seed a research program ──────────────────────
            prog = workspace.create_program(
                "E2E Test Program",
                "Why does X happen under conditions C?",
            )
            obj = workspace.add_objective(prog.program_id, "Determine mechanism of X")
            workspace.add_question(
                prog.program_id,
                obj,
                "Why does X happen?",
                importance=0.9,
            )
            workspace.add_question(
                prog.program_id,
                obj,
                "Is Z involved in X?",
                importance=0.7,
            )

            # ── Phase 2: First cycle — curiosity + ideas + hypotheses ─
            await loop.run_cycle()
            assert loop.cycle_count == 1

            # Loop should have generated at least one hypothesis
            from hbllm.hcir.graph import HypothesisNode

            hypotheses = [n for n in graph.all_nodes() if isinstance(n, HypothesisNode)]
            assert len(hypotheses) >= 1, "Cycle 1 should produce at least 1 hypothesis"

            # ── Phase 3: Inject evidence + build belief ───────────────
            ev_support = EvidenceNode(
                evidence_type=EvidenceStrength.EXPERIMENTAL,
                methodology="RCT n=200",
                sample_size=200,
                reproducible=True,
            )
            graph.upsert_node(ev_support)

            belief = BeliefNode(
                claim="X is caused by mechanism Z under conditions C",
                belief_confidence=BeliefConfidence(
                    evidence_quality=0.7,
                    evidence_quantity=0.4,
                    reproducibility=0.6,
                    prediction_accuracy=0.5,
                ),
            )
            graph.upsert_node(belief)
            graph.add_edge(
                HCIREdge(
                    sources=[ev_support.id],
                    targets=[belief.id],
                    edge_type=HCIREdgeType.SUPPORTS,
                )
            )

            # ── Phase 4: Revise belief with supporting evidence ───────
            revision1 = await belief_manager.revise_belief(
                belief.id,
                ev_support.id,
                direction="supporting",
                evidence_strength="experimental",
            )
            assert revision1.new_confidence > revision1.old_confidence
            assert "supporting" in revision1.reason

            # ── Phase 5: Second cycle — memory recording triggers ─────
            await loop.run_cycle()
            assert loop.cycle_count == 2

            # Memory should have recorded confidence snapshots
            trajectory = await memory.get_confidence_trajectory(belief.id)
            assert len(trajectory) >= 1, "Memory should record belief snapshots"

            # ── Phase 6: Inject contradicting evidence ────────────────
            ev_contra = EvidenceNode(
                evidence_type=EvidenceStrength.EXPERIMENTAL,
                methodology="Large-scale replication n=500",
                sample_size=500,
                reproducible=True,
            )
            graph.upsert_node(ev_contra)
            graph.add_edge(
                HCIREdge(
                    sources=[ev_contra.id],
                    targets=[belief.id],
                    edge_type=HCIREdgeType.WEAKENS,
                )
            )

            revision2 = await belief_manager.revise_belief(
                belief.id,
                ev_contra.id,
                direction="contradicting",
                evidence_strength="experimental",
            )
            assert revision2.new_confidence < revision2.old_confidence

            # ── Phase 7: Prediction-based revision ────────────────────
            outcome = PredictionOutcome(
                prediction_id="pred_e2e_1",
                hypothesis_id=hypotheses[0].id,
                predicted="increase",
                observed="decrease",
                correct=False,
                confidence_delta=-0.1,
            )
            await memory.record_prediction_result(
                outcome,
                predicted_confidence=0.8,
            )

            revision3 = await belief_manager.revise_from_prediction(
                belief.id,
                outcome,
            )
            assert revision3.new_confidence < revision3.old_confidence

            # ── Phase 8: Third cycle — calibration runs ───────────────
            # Record a correct prediction too
            outcome_correct = PredictionOutcome(
                prediction_id="pred_e2e_2",
                hypothesis_id=hypotheses[0].id,
                predicted="stable",
                observed="stable",
                correct=True,
            )
            await memory.record_prediction_result(
                outcome_correct,
                predicted_confidence=0.6,
            )

            await loop.run_cycle()
            assert loop.cycle_count == 3

            # ── Phase 9: Verify calibration report ────────────────────
            report = await calibrator.calibrate()
            assert report.total_predictions == 2
            assert report.prediction_accuracy == 0.5  # 1 correct, 1 wrong

            # ── Phase 10: Counterfactual analysis ─────────────────────
            sensitivity = await counterfactual.sensitivity_analysis(belief.id)
            assert len(sensitivity) >= 1, "Sensitivity analysis should find evidence"

            # ── Phase 11: Falsification candidates ────────────────────
            candidates = await belief_manager.get_falsification_candidates()
            assert isinstance(candidates, list)

            # ── Phase 12: Belief summary ──────────────────────────────
            summary = await belief_manager.get_belief_summary(belief.id)
            assert summary["revision_count"] >= 2
            assert summary["confidence"] < 0.7  # Should have dropped

            memory.close()

    @pytest.mark.asyncio
    async def test_multi_cycle_memory_trajectory(self) -> None:
        """Verify that confidence trajectory tracks across multiple cycles."""
        with tempfile.TemporaryDirectory() as td:
            graph = CognitiveGraph()
            workspace = DiscoveryWorkspace(data_dir=td, graph=graph)
            memory = EpistemicMemory(data_dir=td)

            # Create a belief
            belief = BeliefNode(
                claim="Test trajectory belief",
                belief_confidence=BeliefConfidence(evidence_quality=0.5),
            )
            graph.upsert_node(belief)

            # Create program so loop doesn't skip
            prog = workspace.create_program("Trajectory", "Test")
            obj = workspace.add_objective(prog.program_id, "Track")
            workspace.add_question(prog.program_id, obj, "Test?", importance=0.5)

            loop = EpistemicLoop(
                graph=graph,
                workspace=workspace,
                memory=memory,
            )

            # Run 3 cycles
            for _ in range(3):
                await loop.run_cycle()

            trajectory = await memory.get_confidence_trajectory(belief.id)
            assert len(trajectory) == 3

            memory.close()

    @pytest.mark.asyncio
    async def test_hypothesis_to_falsification(self) -> None:
        """A belief with repeated failed predictions should become falsified."""
        with tempfile.TemporaryDirectory() as td:
            graph = CognitiveGraph()
            memory = EpistemicMemory(data_dir=td)
            belief_manager = DiscoveryBeliefManager(graph=graph)

            # Create a belief with moderate confidence
            belief = BeliefNode(
                claim="X causes Y",
                belief_confidence=BeliefConfidence(evidence_quality=0.5),
            )
            belief.uncertainty.confidence = 0.5
            graph.upsert_node(belief)

            # Repeatedly fail predictions
            for i in range(10):
                outcome = PredictionOutcome(
                    prediction_id=f"p_fail_{i}",
                    hypothesis_id="h1",
                    predicted="x",
                    observed="y",
                    correct=False,
                )
                await belief_manager.revise_from_prediction(belief.id, outcome)

            # Should now be falsified
            updated = graph.get_node(belief.id)
            assert isinstance(updated, BeliefNode)
            assert updated.falsification_status == FalsificationStatus.FALSIFIED
            assert updated.uncertainty.confidence < 0.1

            memory.close()
