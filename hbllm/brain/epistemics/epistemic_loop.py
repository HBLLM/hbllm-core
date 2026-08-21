"""Epistemic Loop — orchestrator for the recursive discovery cycle.

The EpistemicLoop is a ``ProactiveHandler`` compatible with
``AutonomyCore.add_proactive_handler()``.  When the system is idle,
it runs one round of the epistemic cycle:

1. **Scan** — CuriosityEngine identifies investigation targets
2. **Generate** — IdeaGenerator produces raw ideas for top targets
3. **Filter** — HypothesisBuilder validates and deduplicates
4. **Predict** — PredictionTracker registers predictions
5. **Experiment** — ExperimentPlanner designs discriminative tests
6. **Evaluate** — EvidenceEvaluator scores results
7. **Revise** — BeliefManager updates belief confidence
8. **Explain** — ExplanationEngine traces provenance
9. **Recurse** — New unknowns feed back into step 1

Architecture::

    AutonomyCore
        └── proactive_handlers
                └── epistemic_loop.run_cycle()  → list[Message] | None

    EpistemicLoop
        ├── CuriosityEngine       → what to investigate
        ├── IdeaGenerator         → raw ideas
        ├── HypothesisBuilder     → validated hypotheses
        ├── PredictionTracker     → competing predictions
        ├── ExperimentPlanner     → info-gain-optimized experiments
        ├── EvidenceEvaluator     → evidence quality scoring
        ├── ContradictionEngine   → proactive contradiction hunting
        ├── ResearchStrategyManager → pluggable strategy patterns
        ├── ExplanationEngine     → provenance graph traversal
        └── DiscoveryBeliefManager → Bayesian belief revision

Design principle: everything can create new unknowns.  Discovery
is recursive, not linear.

Usage::

    loop = EpistemicLoop(
        graph=graph,
        workspace=workspace,
        llm=llm,
        reputation=tracker,
    )
    # Register with AutonomyCore
    autonomy.add_proactive_handler("epistemic_loop", loop.run_cycle)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.brain.epistemics.belief_manager import DiscoveryBeliefManager
from hbllm.brain.epistemics.contradiction_engine import ContradictionEngine
from hbllm.brain.epistemics.curiosity_engine import CuriosityEngine
from hbllm.brain.epistemics.evidence_evaluator import EvidenceEvaluator
from hbllm.brain.epistemics.experiment_planner import ExperimentPlanner
from hbllm.brain.epistemics.explanation import ExplanationEngine
from hbllm.brain.epistemics.hypothesis_builder import HypothesisBuilder
from hbllm.brain.epistemics.idea_generator import IdeaGenerator
from hbllm.brain.epistemics.interfaces import (
    ConfidenceSnapshot,
    CuriositySignal,
    InvestigationBudget,
)
from hbllm.brain.epistemics.likelihood_evaluator import EpistemicLikelihoodEvaluator
from hbllm.brain.epistemics.perceptual_evaluator import PerceptualEvidenceEvaluator
from hbllm.brain.epistemics.prediction_tracker import PredictionTracker
from hbllm.brain.epistemics.research_strategy import (
    ResearchStrategyManager,
)
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import BeliefNode, CognitiveGraph, EvidenceNode
from hbllm.hcir.types import DiscoveryTrigger

logger = logging.getLogger(__name__)

# Try to import Message type for AutonomyCore compatibility.
# Falls back gracefully if the network package isn't available.
try:
    from hbllm.network.messages import Message, MessageType
except ImportError:
    Message = None  # type: ignore[assignment, misc]
    MessageType = None  # type: ignore[assignment, misc]


class EpistemicLoop:
    """Orchestrator for the recursive epistemic discovery cycle.

    Compatible with ``AutonomyCore.add_proactive_handler()``::

        autonomy.add_proactive_handler("epistemic", loop.run_cycle)

    Each ``run_cycle()`` call performs one round of the epistemic
    cycle for all active research programs.  The loop is deliberately
    bounded to prevent runaway computation.

    The loop is domain-neutral — it orchestrates domain-neutral
    engines.  Domain knowledge arrives through the graph.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        workspace: DiscoveryWorkspace | None = None,
        llm: Any | None = None,
        reputation_tracker: Any | None = None,
        budget: InvestigationBudget | None = None,
        max_investigations_per_cycle: int = 3,
        max_ideas_per_investigation: int = 15,
        memory: Any | None = None,
        calibration: Any | None = None,
        counterfactual: Any | None = None,
        calibration_interval: int = 5,
    ) -> None:
        """Initialize the epistemic loop.

        All engines are instantiated internally.  External code only
        needs to provide the shared graph, optional workspace, and
        optional LLM.

        Args:
            graph: The shared HCIR cognitive graph.
            workspace: Optional DiscoveryWorkspace for program context.
            llm: Optional LLM instance for creative reasoning.
            reputation_tracker: Optional SourceReputationTracker.
            budget: Default investigation budget per cycle.
            max_investigations_per_cycle: Max targets per cycle.
            max_ideas_per_investigation: Max ideas per target.
            memory: Optional EpistemicMemory for outcome recording.
            calibration: Optional EpistemicCalibrationEngine for auto-switching.
            counterfactual: Optional CounterfactualReasoner for experiment design.
            calibration_interval: Run calibration every N cycles.
        """
        self._graph = graph
        self._workspace = workspace
        self._llm = llm
        self._budget = budget or InvestigationBudget()
        self._max_investigations = max_investigations_per_cycle
        self._max_ideas = max_ideas_per_investigation

        # Wave 3 meta-cognition (optional)
        self._memory = memory
        self._calibration = calibration
        self._counterfactual = counterfactual
        self._calibration_interval = calibration_interval

        # Instantiate all engines
        self._curiosity = CuriosityEngine(
            graph=graph,
            reputation_tracker=reputation_tracker,
        )
        self._idea_generator = IdeaGenerator(
            graph=graph,
            llm=llm,
            max_ideas_per_generation=max_ideas_per_investigation,
            memory=memory,
        )
        self._hypothesis_builder = HypothesisBuilder(
            graph=graph,
            llm=llm,
        )
        self._prediction_tracker = PredictionTracker(
            graph=graph,
            llm=llm,
        )
        self._experiment_planner = ExperimentPlanner(
            graph=graph,
            llm=llm,
            counterfactual=counterfactual,
        )
        self._evidence_evaluator = EvidenceEvaluator(
            graph=graph,
            reputation_tracker=reputation_tracker,
            llm=llm,
        )
        self._contradiction_engine = ContradictionEngine(
            graph=graph,
            llm=llm,
        )
        self._explanation_engine = ExplanationEngine(
            graph=graph,
        )
        self._strategy_manager = ResearchStrategyManager(graph=graph)

        # Perceptual epistemics & belief management
        self._perceptual_evaluator = PerceptualEvidenceEvaluator(
            graph=graph,
            reputation_tracker=reputation_tracker,
        )
        self._likelihood_evaluator = EpistemicLikelihoodEvaluator(
            graph=graph,
            llm=llm,
        )
        self._belief_manager = DiscoveryBeliefManager(graph=graph)

        # Wire the contradiction engine into the curiosity engine
        self._curiosity._contradiction_engine = self._contradiction_engine

        # Cycle counter
        self._cycle_count = 0
        self._last_cycle_time = 0.0

    # ── AutonomyCore-Compatible Entry Point ────────────────────────────

    async def run_cycle(self) -> list[Any] | None:
        """Run one epistemic cycle.  AutonomyCore-compatible.

        Returns:
            A list of Message objects describing cycle results,
            or None if nothing worth investigating was found.
        """
        self._cycle_count += 1
        cycle_start = time.time()
        logger.info("Epistemic cycle #%d starting", self._cycle_count)

        results: list[str] = []

        try:
            # Step 0: Process and evaluate unassessed sensory evidence
            revisions_made = await self._process_perceptual_evidence()
            if revisions_made:
                results.append(f"Evaluated sensory evidence: {revisions_made} belief revisions")

            # Step 1: Check expired predictions
            expired = await self._prediction_tracker.check_expired_predictions()
            if expired:
                results.append(f"Checked {len(expired)} expired predictions")

            # Step 2: Scan for contradictions (including perceptual contradictions)
            contradictions = await self._contradiction_engine.scan_for_contradictions()
            if contradictions:
                results.append(f"Found {len(contradictions)} potential contradictions")

            # Step 3: Scan for curiosity signals (incorporating new unknowns and contradictions)
            signals = await self._curiosity.prioritize_investigations(
                self._budget,
            )

            if not signals:
                logger.info("Epistemic cycle #%d: nothing to investigate", self._cycle_count)
                self._last_cycle_time = time.time() - cycle_start
                return results or None

            # Step 4: Investigate top signals
            investigated = 0
            for signal in signals[: self._max_investigations]:
                try:
                    result = await self._investigate_signal(signal)
                    if result:
                        results.append(result)
                    investigated += 1
                except Exception as exc:
                    logger.warning(
                        "Investigation failed for %s: %s",
                        signal.source_id,
                        exc,
                    )

            # Step 5: Generate spontaneous unknowns
            new_unknowns = await self._curiosity.generate_spontaneous_unknowns()
            if new_unknowns:
                results.append(
                    f"Generated {len(new_unknowns)} spontaneous unknowns from uncertainty hotspots"
                )

            # Step 6: Record to epistemic memory
            await self._record_cycle_to_memory(results)

            # Step 7: Periodic calibration + auto strategy switching
            if (
                self._calibration is not None
                and self._cycle_count % self._calibration_interval == 0
            ):
                try:
                    rec = await self._calibration.recommend_strategy_adjustment()
                    if rec:
                        self._strategy_manager.set_active_strategy(rec)
                        results.append(f"Calibration: switching strategy → {rec}")
                except Exception as exc:
                    logger.warning("Calibration failed: %s", exc)

        except Exception as exc:
            logger.error("Epistemic cycle #%d failed: %s", self._cycle_count, exc)
            results.append(f"Cycle error: {exc}")

        self._last_cycle_time = time.time() - cycle_start
        logger.info(
            "Epistemic cycle #%d completed in %.2fs (%d results)",
            self._cycle_count,
            self._last_cycle_time,
            len(results),
        )

        if not results:
            return None

        # Convert to Messages if available
        if Message is not None and MessageType is not None:
            messages = []
            for result in results:
                msg = Message(
                    type=MessageType.EVENT,
                    source_node_id="epistemic_loop",
                    topic="epistemic_cycle",
                    content=result,
                    metadata={"source": "epistemic_loop", "cycle": self._cycle_count},
                )
                messages.append(msg)
            return messages

        return results  # type: ignore[return-value]

    async def _process_perceptual_evidence(self) -> int:
        """Process unassessed sensory evidence and evaluate candidate belief revisions."""
        revisions_count = 0

        # Find all perceptual evidence nodes
        perceptual_evidence: list[EvidenceNode] = []
        beliefs: list[BeliefNode] = []

        for _node in self._graph.all_nodes():
            node = self._graph.get_node(_node.id)
            if isinstance(node, EvidenceNode) and (
                node.modality or node.epistemic_profile is not None
            ):
                perceptual_evidence.append(node)
            elif isinstance(node, BeliefNode):
                beliefs.append(node)

        if not perceptual_evidence or not beliefs:
            return 0

        # Evaluate evidence and candidate revisions
        for evidence in perceptual_evidence:
            assessment = self._perceptual_evaluator.evaluate(evidence)
            if assessment.reliability < 0.4:
                continue

            for belief in beliefs:
                # Check if this evidence was already incorporated
                if evidence.id in belief.evidence_sources or evidence.id in belief.counter_evidence:
                    continue

                prop_lik = self._likelihood_evaluator.evaluate_likelihood(
                    belief=belief,
                    evidence=evidence,
                    assessment=assessment,
                )

                if prop_lik.status in ("informative", "contradictory"):
                    try:
                        transition = await self._belief_manager.revise(
                            belief_id=belief.id,
                            proposition_likelihood=prop_lik,
                            evidence_assessment=assessment,
                            rationale=f"Perceptual revision from {evidence.modality} evidence ({evidence.id}) [{prop_lik.status}]",
                        )
                        if transition.transition_id:
                            revisions_count += 1
                            logger.info(
                                "Revised belief %s from perceptual evidence %s: prior=%.2f -> post=%.2f (LR=%.2f)",
                                belief.id,
                                evidence.id,
                                transition.prior_confidence,
                                transition.posterior_confidence,
                                transition.likelihood_ratio,
                            )
                    except Exception as exc:
                        logger.warning("Failed to revise belief %s: %s", belief.id, exc)

        return revisions_count

    # ── Investigation Pipeline ─────────────────────────────────────────

    async def _investigate_signal(
        self,
        signal: CuriositySignal,
    ) -> str | None:
        """Run the full investigation pipeline for a single signal.

        Pipeline::

            Signal → Ideas → Hypotheses → Predictions → Experiment Design
        """
        logger.debug(
            "Investigating: %s (trigger=%s)",
            signal.description[:60],
            signal.trigger,
        )

        # Step 1: Generate ideas
        ideas = await self._generate_ideas(signal)
        if not ideas:
            return None

        # Step 2: Build hypotheses
        candidates = await self._hypothesis_builder.validate(ideas)
        novel = await self._hypothesis_builder.deduplicate(candidates)

        if not novel:
            return f"Investigated {signal.description[:60]}: {len(ideas)} ideas, none novel"

        # Step 3: Promote to graph nodes
        promoted_ids: list[str] = []
        program_id = self._get_program_id(signal)

        for candidate in novel[:3]:  # Max 3 hypotheses per signal
            try:
                node_id = await self._hypothesis_builder.promote_to_node(
                    candidate,
                    program_id,
                )
                promoted_ids.append(node_id)
            except Exception as exc:
                logger.warning("Hypothesis promotion failed: %s", exc)

        if not promoted_ids:
            return None

        # Step 4: Design discriminative experiment (if multiple hypotheses)
        experiment_result = ""
        if len(promoted_ids) >= 2:
            try:
                design = await self._experiment_planner.design_discriminative_experiment(
                    hypothesis_ids=promoted_ids,
                    budget=self._budget,
                )
                experiment_result = (
                    f", experiment designed (info_gain={design.expected_information_gain:.2f}, "
                    f"cost={design.estimated_cost:.2f})"
                )
            except Exception as exc:
                logger.warning("Experiment design failed: %s", exc)

        return (
            f"Investigated '{signal.description[:50]}': "
            f"{len(ideas)} ideas → {len(candidates)} valid → "
            f"{len(novel)} novel → {len(promoted_ids)} hypotheses"
            f"{experiment_result}"
        )

    async def _generate_ideas(
        self,
        signal: CuriositySignal,
    ) -> list[Any]:
        """Generate ideas based on the signal trigger type."""
        trigger = signal.trigger
        source_id = signal.source_id

        if trigger in (
            "knowledge_gap",
            str(DiscoveryTrigger.KNOWLEDGE_GAP),
            str(DiscoveryTrigger.CURIOSITY),
        ):
            if signal.unknown_id:
                return await self._idea_generator.generate_from_unknown(
                    signal.unknown_id,
                )
            # If source is a hypothesis, generate from its uncertainty
            return (
                await self._idea_generator.generate_from_unknown(
                    source_id,
                )
                if source_id
                else []
            )

        elif trigger in (
            "contradiction",
            str(DiscoveryTrigger.CONTRADICTION),
            "perceptual_anomaly",
            str(DiscoveryTrigger.PERCEPTUAL_ANOMALY),
            "perceptual_ambiguity",
            str(DiscoveryTrigger.PERCEPTUAL_AMBIGUITY),
        ):
            return (
                await self._idea_generator.generate_from_contradiction(
                    source_id,
                )
                if source_id
                else []
            )

        elif trigger in (
            "anomaly",
            str(DiscoveryTrigger.ANOMALY),
            "unexpected_failure",
            str(DiscoveryTrigger.UNEXPECTED_FAILURE),
            "unexpected_success",
            str(DiscoveryTrigger.UNEXPECTED_SUCCESS),
        ):
            return (
                await self._idea_generator.generate_from_anomaly(
                    source_id,
                )
                if source_id
                else []
            )

        else:
            # Default: try to generate from unknown
            target = signal.unknown_id or source_id
            if target:
                return await self._idea_generator.generate_from_unknown(target)
            return []

    def _get_program_id(self, signal: CuriositySignal) -> str:
        """Extract research program ID from signal context."""
        if signal.unknown_id:
            node = self._graph.get_node(signal.unknown_id)
            if node is not None:
                return getattr(node, "research_program_id", "")
        if signal.source_id:
            node = self._graph.get_node(signal.source_id)
            if node is not None:
                return getattr(node, "research_program_id", "")
        return ""

    # ── Status & Diagnostics ───────────────────────────────────────────

    @property
    def cycle_count(self) -> int:
        """Number of completed epistemic cycles."""
        return self._cycle_count

    @property
    def last_cycle_time(self) -> float:
        """Duration of the last cycle in seconds."""
        return self._last_cycle_time

    @property
    def engines(self) -> dict[str, Any]:
        """Access to individual engines for direct use."""
        return {
            "curiosity": self._curiosity,
            "idea_generator": self._idea_generator,
            "hypothesis_builder": self._hypothesis_builder,
            "prediction_tracker": self._prediction_tracker,
            "experiment_planner": self._experiment_planner,
            "evidence_evaluator": self._evidence_evaluator,
            "contradiction_engine": self._contradiction_engine,
            "explanation_engine": self._explanation_engine,
            "strategy_manager": self._strategy_manager,
        }

    # ── Memory Recording ─────────────────────────────────────────────────

    async def _record_cycle_to_memory(self, results: list[str]) -> None:
        """Record cycle outcomes to epistemic memory.

        Snapshots all beliefs' confidence after each cycle for
        trajectory tracking and calibration.
        """
        if self._memory is None:
            return

        try:
            # Snapshot all belief confidences
            for node in self._graph.all_nodes():
                if isinstance(node, BeliefNode):
                    bc = node.belief_confidence
                    snap = ConfidenceSnapshot(
                        belief_id=node.id,
                        derived_confidence=bc.derived_confidence,
                        evidence_quality=bc.evidence_quality,
                        evidence_quantity=bc.evidence_quantity,
                        reproducibility=bc.reproducibility,
                        prediction_accuracy=bc.prediction_accuracy,
                        model_agreement=bc.model_agreement,
                        source_trust=bc.source_trust,
                    )
                    await self._memory.snapshot_belief_confidence(
                        node.id,
                        snap,
                    )
        except Exception as exc:
            logger.debug("Memory recording failed: %s", exc)
