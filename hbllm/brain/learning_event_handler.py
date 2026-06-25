"""
Learning Event Handler — event-driven bridge between experience and learning.

Learning is not a separate activity.  It emerges from:
    acting → failing → observing → sleeping

This Node subscribes to experience events on the MessageBus and dispatches
them to the appropriate learning components (CausalModelBuilder,
FailureAnalyzer, BeliefRevisionEngine, MechanismStore, MetaLearner).

No LLM calls happen here during query time — only lightweight
routing.  Heavy operations (model building, concept formation) are
queued for background processing or sleep consolidation.

Bus Topics:
    learning.experience.success  → reinforce mechanisms, queue model building
    learning.experience.failure  → root cause analysis → maybe belief revision
    curiosity.research.complete  → extend models with research findings
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.failure_analyzer import FailureAnalyzer
    from hbllm.brain.mechanism_store import MechanismStore

logger = logging.getLogger(__name__)


class LearningEventHandler(Node):
    """Bridges experience events to the causal learning engine.

    Lightweight Node that turns HBLLM's experience stream into
    learning signals — reinforcing mechanisms on success, analyzing
    root causes on failure, and queueing heavy operations for sleep.

    Design principle: learning should emerge from the existing
    cognitive loop, not be a separate parallel system.
    """

    def __init__(
        self,
        node_id: str = "learning_event_handler",
        mechanism_store: MechanismStore | None = None,
        failure_analyzer: FailureAnalyzer | None = None,
        belief_engine: Any | None = None,
        contradiction_detector: Any | None = None,
        meta_learner: Any | None = None,
        causal_model_builder: Any | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["experience_learning", "failure_analysis", "mechanism_reinforcement"],
        )
        self.mechanism_store = mechanism_store
        self.failure_analyzer = failure_analyzer
        self.belief_engine = belief_engine
        self.contradiction_detector = contradiction_detector
        self.meta_learner = meta_learner
        self.causal_model_builder = causal_model_builder

        # Queue for background model building (processed during idle/sleep)
        self._model_build_queue: list[dict[str, Any]] = []
        self._stats = {
            "successes_processed": 0,
            "failures_processed": 0,
            "mechanisms_reinforced": 0,
            "belief_revisions_triggered": 0,
            "models_queued": 0,
        }

    async def on_start(self) -> None:
        logger.info("Starting LearningEventHandler")
        await self.bus.subscribe(
            "learning.experience.success", self._handle_success
        )
        await self.bus.subscribe(
            "learning.experience.failure", self._handle_failure
        )
        await self.bus.subscribe(
            "curiosity.research.complete", self._handle_research_complete
        )

    async def on_stop(self) -> None:
        logger.info(
            "Stopping LearningEventHandler (stats: %s)", self._stats
        )

    async def handle_message(self, message: Message) -> Message | None:
        """Direct message handling — returns stats."""
        if message.payload.get("action") == "stats":
            return message.create_response({
                "stats": self._stats,
                "model_build_queue_size": len(self._model_build_queue),
            })
        return None

    # ─── Success Handler ─────────────────────────────────────────────

    async def _handle_success(self, message: Message) -> Message | None:
        """Handle successful experience — reinforce mechanisms and queue model building.

        Expected payload:
            - domain: str (e.g. "security", "devops")
            - execution_trace: list[dict] (steps taken)
            - mechanism_ids: list[str] (mechanisms used, if known)
            - skill_id: str (skill that was executed, if any)
            - query: str (original query)
        """
        self._stats["successes_processed"] += 1
        payload = message.payload

        # 1. Reinforce mechanisms that were involved
        mechanism_ids = payload.get("mechanism_ids", [])
        if self.mechanism_store and mechanism_ids:
            for mech_id in mechanism_ids:
                self.mechanism_store.record_usage(mech_id, success=True)
                self._stats["mechanisms_reinforced"] += 1

        # 2. Record session with meta-learner (lightweight)
        if self.meta_learner:
            try:
                self.meta_learner.record_session(
                    domain=payload.get("domain", "general"),
                    strategy="execution",
                    success=True,
                    notes=f"Skill: {payload.get('skill_id', 'direct')}",
                )
            except Exception:
                logger.debug("MetaLearner recording failed", exc_info=True)

        # 3. Reinforce beliefs for involved mechanisms
        if self.belief_engine and mechanism_ids:
            try:
                for mech_id in mechanism_ids:
                    self.belief_engine.integrate_evidence(
                        entity_id=mech_id,
                        evidence_type="success",
                        evidence=f"Mechanism successfully applied: {payload.get('query', '')[:100]}",
                    )
            except Exception:
                logger.debug("BeliefEngine reinforcement failed", exc_info=True)

        # 4. Queue causal model building for background/sleep (NO LLM during queries)
        trace = payload.get("execution_trace", [])
        if trace:
            self._model_build_queue.append({
                "domain": payload.get("domain", "general"),
                "trace": trace,
                "query": payload.get("query", ""),
                "skill_id": payload.get("skill_id"),
                "mechanism_ids": mechanism_ids,
            })
            self._stats["models_queued"] += 1
            logger.info(
                "Queued causal model building for domain '%s' (queue=%d)",
                payload.get("domain", "general"),
                len(self._model_build_queue),
            )

        return None

    # ─── Failure Handler ─────────────────────────────────────────────

    async def _handle_failure(self, message: Message) -> Message | None:
        """Handle failed experience — analyze root cause, maybe revise beliefs.

        Expected payload:
            - expected: str (what was expected)
            - actual: str (what happened)
            - error_message: str (raw error/traceback)
            - domain: str
            - mechanism_ids: list[str]
            - skill_id: str
            - query: str
            - context: dict (additional context)
        """
        self._stats["failures_processed"] += 1
        payload = message.payload

        expected = payload.get("expected", "success")
        actual = payload.get("actual", "failure")
        error_msg = payload.get("error_message", "")
        mechanism_ids = payload.get("mechanism_ids", [])

        # 1. Root Cause Analysis (heuristic, no LLM)
        if self.failure_analyzer:
            root_cause = self.failure_analyzer.analyze(
                expected=expected,
                actual=actual,
                error_message=error_msg,
                context=payload.get("context", {}),
                mechanism_ids=mechanism_ids,
            )

            logger.info(
                "Failure root cause: %s (belief_error=%s)",
                root_cause.category.value,
                root_cause.is_belief_error,
            )

            # 2. Only flow true belief errors to ContradictionDetector
            if root_cause.is_belief_error:
                if self.contradiction_detector:
                    try:
                        self.contradiction_detector.record(
                            expected=root_cause.expected,
                            actual=root_cause.actual,
                            category=root_cause.category.value,
                            affected_belief=root_cause.affected_belief,
                        )
                    except Exception:
                        logger.debug("ContradictionDetector recording failed", exc_info=True)

                # 3. Revise beliefs for affected mechanisms
                if self.belief_engine and root_cause.requires_belief_revision:
                    try:
                        self.belief_engine.integrate_evidence(
                            entity_id=root_cause.affected_belief or "unknown",
                            evidence_type="contradiction",
                            evidence=f"Expected: {expected}, Got: {actual}. {error_msg[:200]}",
                        )
                        self._stats["belief_revisions_triggered"] += 1
                    except Exception:
                        logger.debug("BeliefEngine revision failed", exc_info=True)

                # 4. Emit curiosity signal for investigation
                if self.bus.has_subscribers("curiosity.investigate"):
                    try:
                        curiosity_msg = Message(
                            type=MessageType.EVENT,
                            source_node_id=self.node_id,
                            topic="curiosity.investigate",
                            payload={
                                "question": (
                                    f"Why did '{expected}' fail with '{actual}'? "
                                    f"Root cause category: {root_cause.category.value}"
                                ),
                                "domain": payload.get("domain", "general"),
                                "priority": "high" if root_cause.confidence > 0.7 else "low",
                            },
                        )
                        await self.bus.publish("curiosity.investigate", curiosity_msg)
                    except Exception:
                        logger.debug("Curiosity signal failed", exc_info=True)
        else:
            logger.debug("No FailureAnalyzer available, skipping root cause analysis")

        # 5. Record mechanism failures
        if self.mechanism_store and mechanism_ids:
            for mech_id in mechanism_ids:
                self.mechanism_store.record_usage(mech_id, success=False)

        # 6. Record session with meta-learner
        if self.meta_learner:
            try:
                self.meta_learner.record_session(
                    domain=payload.get("domain", "general"),
                    strategy="execution",
                    success=False,
                    notes=f"Error: {error_msg[:200]}",
                )
            except Exception:
                logger.debug("MetaLearner recording failed", exc_info=True)

        return None

    # ─── Research Complete Handler ───────────────────────────────────

    async def _handle_research_complete(self, message: Message) -> Message | None:
        """Handle completed autonomous research — extend causal models.

        Expected payload:
            - domain: str
            - findings: list[dict]
            - new_mechanisms: list[dict] (discovered mechanisms)
            - source_goal: str
        """
        payload = message.payload

        # 1. Store any discovered mechanisms
        new_mechanisms = payload.get("new_mechanisms", [])
        if self.mechanism_store and new_mechanisms:
            for mech_data in new_mechanisms:
                try:
                    self.mechanism_store.create(
                        description=mech_data.get("description", ""),
                        preconditions=mech_data.get("preconditions", []),
                        process_steps=mech_data.get("process_steps", []),
                        expected_outcomes=mech_data.get("expected_outcomes", []),
                        domain=payload.get("domain", "general"),
                        abstraction_level=mech_data.get("abstraction_level", 0),
                    )
                except Exception:
                    logger.debug("Failed to store discovered mechanism", exc_info=True)

        # 2. Queue model building with research findings
        findings = payload.get("findings", [])
        if findings:
            self._model_build_queue.append({
                "domain": payload.get("domain", "general"),
                "trace": findings,
                "query": payload.get("source_goal", ""),
                "source": "research",
            })
            self._stats["models_queued"] += 1

        return None

    # ─── Background Processing ───────────────────────────────────────

    async def process_build_queue(self) -> int:
        """Process queued causal model building tasks.

        Called during sleep consolidation or idle periods.
        This is where LLM-heavy operations happen.

        Returns:
            Number of models built.
        """
        if not self.causal_model_builder or not self._model_build_queue:
            return 0

        built = 0
        while self._model_build_queue:
            item = self._model_build_queue.pop(0)
            try:
                model = await asyncio.wait_for(
                    self.causal_model_builder.build_model(
                        domain=item["domain"],
                        observations=item["trace"],
                    ),
                    timeout=30.0,
                )
                if model:
                    built += 1
                    logger.info(
                        "Built causal model for domain '%s' from queued experience",
                        item["domain"],
                    )
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning("Causal model building timed out, re-queueing")
                self._model_build_queue.append(item)
                break
            except Exception:
                logger.debug("Causal model building failed", exc_info=True)

        return built

    def get_queue_size(self) -> int:
        """Get the number of pending model build tasks."""
        return len(self._model_build_queue)
