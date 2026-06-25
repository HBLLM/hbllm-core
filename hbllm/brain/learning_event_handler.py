"""
Learning Event Handler — event-driven bridge between experience and learning.

Learning is not a separate activity.  It emerges from:
    acting → failing → observing → sleeping

This Node subscribes to experience events on the MessageBus and dispatches
them to the appropriate learning components via the shared LearningSubsystem.

No LLM calls happen here during query time — only lightweight
routing.  Heavy operations (model building, concept formation) are
queued for background processing or sleep consolidation.

Event Taxonomy:
    Experience events (from WorkspaceNode / SIL — real execution):
        learning.experience.success  → reinforce mechanisms, update beliefs, record meta
        learning.experience.failure  → root cause analysis → maybe belief revision

    Research events (from AutonomousLearner — NO belief/meta update to avoid echo):
        learning.session.complete    → store discovered mechanisms only
        learning.contradiction.discovered → log for sleep analysis only

    Curiosity events:
        curiosity.research.complete  → extend models with research findings

Critical invariant:
    Each evidence source updates beliefs exactly once.
    AutonomousLearner already calls BeliefRevisionEngine and MetaLearner
    directly during research.  LearningEventHandler must NOT call them
    again when handling research events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.learning_subsystem import LearningSubsystem

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
        learning_subsystem: LearningSubsystem | None = None,
        # Legacy direct injection (used when LearningSubsystem not yet available)
        mechanism_store: Any | None = None,
        failure_analyzer: Any | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["experience_learning", "failure_analysis", "mechanism_reinforcement"],
        )

        # Prefer LearningSubsystem, fall back to direct injection
        if learning_subsystem is not None:
            self._subsystem = learning_subsystem
        else:
            # Build a minimal subsystem from direct args (backward compat)
            from hbllm.brain.learning_subsystem import LearningSubsystem as _LearningSubsystem

            self._subsystem = _LearningSubsystem(
                mechanism_store=mechanism_store,
                failure_analyzer=failure_analyzer,
            )

        # Queue for background model building (processed during idle/sleep)
        self._model_build_queue: list[dict[str, Any]] = []
        self._stats = {
            "successes_processed": 0,
            "failures_processed": 0,
            "mechanisms_reinforced": 0,
            "belief_revisions_triggered": 0,
            "models_queued": 0,
            "sessions_received": 0,
            "contradictions_received": 0,
        }

    # ─── Properties for clean access ─────────────────────────────────

    @property
    def mechanism_store(self) -> Any | None:
        return self._subsystem.mechanism_store

    @property
    def failure_analyzer(self) -> Any | None:
        return self._subsystem.failure_analyzer

    @property
    def belief_engine(self) -> Any | None:
        return self._subsystem.belief_engine

    @property
    def contradiction_detector(self) -> Any | None:
        return self._subsystem.contradiction_detector

    @property
    def meta_learner(self) -> Any | None:
        return self._subsystem.meta_learner

    @property
    def causal_model_builder(self) -> Any | None:
        return self._subsystem.causal_model_builder

    def inject_subsystem(self, subsystem: LearningSubsystem) -> None:
        """Replace the current subsystem (used for late factory wiring)."""
        self._subsystem = subsystem

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def on_start(self) -> None:
        logger.info("Starting LearningEventHandler")
        # Experience events (from real execution — full belief pipeline)
        await self.bus.subscribe(
            "learning.experience.success", self._handle_success
        )
        await self.bus.subscribe(
            "learning.experience.failure", self._handle_failure
        )
        # Research events (from AutonomousLearner — NO belief/meta to avoid echo)
        await self.bus.subscribe(
            "learning.session.complete", self._handle_session_complete
        )
        await self.bus.subscribe(
            "learning.contradiction.discovered",
            self._handle_contradiction_discovered,
        )
        # Curiosity events
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
                "subsystem": self._subsystem.summary(),
            })
        return None

    # ─── Experience: Success (full belief pipeline) ──────────────────

    async def _handle_success(self, message: Message) -> Message | None:
        """Handle successful execution experience.

        This is a REAL execution outcome from WorkspaceNode/SIL.
        Full pipeline: reinforce mechanisms, update beliefs, record meta.
        """
        self._stats["successes_processed"] += 1
        payload = message.payload
        mechanism_ids = payload.get("mechanism_ids", [])

        # 1. Reinforce mechanisms that were involved
        if self.mechanism_store and mechanism_ids:
            for mech_id in mechanism_ids:
                self.mechanism_store.record_usage(mech_id, success=True)
                self._stats["mechanisms_reinforced"] += 1

        # 2. Update beliefs for involved mechanisms (execution evidence)
        if self.belief_engine and mechanism_ids:
            for mech_id in mechanism_ids:
                try:
                    await self.belief_engine.integrate_evidence(
                        concept=mech_id,
                        claim=f"Mechanism {mech_id} is reliable",
                        confidence=0.7,
                        evidence=(
                            f"Successfully applied in execution: "
                            f"{payload.get('query', '')[:100]}"
                        ),
                        source="execution_experience",
                    )
                except Exception:
                    logger.debug("BeliefEngine reinforcement failed", exc_info=True)

        # 3. Record session with meta-learner
        if self.meta_learner:
            try:
                await self.meta_learner.record_session(
                    domain=payload.get("domain", "general"),
                    method="execution",
                    confidence_before=0.5,
                    confidence_after=0.7,
                    resource_cost=1.0,
                )
            except Exception:
                logger.debug("MetaLearner recording failed", exc_info=True)

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

    # ─── Experience: Failure (full belief pipeline) ──────────────────

    async def _handle_failure(self, message: Message) -> Message | None:
        """Handle failed execution experience.

        This is a REAL execution failure from WorkspaceNode/SIL.
        Full pipeline: root cause analysis, maybe belief revision.
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

            # 2. Only flow true belief errors to belief revision
            if root_cause.is_belief_error and self.belief_engine:
                try:
                    from hbllm.brain.contradiction_detector import Contradiction

                    contradiction = Contradiction(
                        existing_claim=root_cause.expected,
                        new_claim=root_cause.actual,
                        concept=payload.get("domain", "unknown"),
                        severity=root_cause.confidence,
                    )
                    await self.belief_engine.handle_contradiction(contradiction)
                    self._stats["belief_revisions_triggered"] += 1
                except Exception:
                    logger.debug("Belief revision failed", exc_info=True)

                # 3. Emit curiosity signal for investigation
                if self.bus.has_subscribers("curiosity.investigate"):
                    try:
                        curiosity_msg = Message(
                            type=MessageType.EVENT,
                            source_node_id=self.node_id,
                            topic="curiosity.investigate",
                            payload={
                                "question": (
                                    f"Why did '{expected}' fail with '{actual}'? "
                                    f"Root cause: {root_cause.category.value}"
                                ),
                                "domain": payload.get("domain", "general"),
                                "priority": (
                                    "high" if root_cause.confidence > 0.7 else "low"
                                ),
                            },
                        )
                        await self.bus.publish("curiosity.investigate", curiosity_msg)
                    except Exception:
                        logger.debug("Curiosity signal failed", exc_info=True)
        else:
            logger.debug("No FailureAnalyzer available, skipping root cause analysis")

        # 4. Record mechanism failures
        if self.mechanism_store and mechanism_ids:
            for mech_id in mechanism_ids:
                self.mechanism_store.record_usage(mech_id, success=False)

        # 5. Record session with meta-learner
        if self.meta_learner:
            try:
                await self.meta_learner.record_session(
                    domain=payload.get("domain", "general"),
                    method="execution",
                    confidence_before=0.5,
                    confidence_after=0.3,
                    resource_cost=1.0,
                )
            except Exception:
                logger.debug("MetaLearner recording failed", exc_info=True)

        return None

    # ─── Research: Session Complete (mechanisms only — NO belief/meta) ─

    async def _handle_session_complete(self, message: Message) -> Message | None:
        """Handle completed autonomous learning session.

        This comes from AutonomousLearner AFTER it has already called
        BeliefRevisionEngine and MetaLearner directly.

        We ONLY log and store mechanisms here.  NO belief updates,
        NO meta-learner recording — that would double-count evidence.
        """
        self._stats["sessions_received"] += 1
        payload = message.payload
        topic = payload.get("topic", "unknown")

        logger.info(
            "Learning session complete: topic='%s', confidence_gain=%.2f, "
            "models=%d",
            topic,
            payload.get("confidence_after", 0) - payload.get("confidence_before", 0),
            payload.get("causal_models_built", 0),
        )

        return None

    # ─── Research: Contradiction Discovered (log only — NO belief update) ─

    async def _handle_contradiction_discovered(
        self, message: Message,
    ) -> Message | None:
        """Handle contradiction discovered during autonomous research.

        AutonomousLearner already called BeliefRevisionEngine directly.
        We ONLY log this for observability — NO belief updates.
        """
        self._stats["contradictions_received"] += 1
        payload = message.payload

        logger.info(
            "Research contradiction: concept='%s', severity=%.2f, "
            "claims: '%s' vs '%s'",
            payload.get("concept", "unknown"),
            payload.get("severity", 0.0),
            payload.get("claim_a", "?")[:50],
            payload.get("claim_b", "?")[:50],
        )

        return None

    # ─── Curiosity: Research Complete ────────────────────────────────

    async def _handle_research_complete(self, message: Message) -> Message | None:
        """Handle completed curiosity-driven research."""
        payload = message.payload

        # Store discovered mechanisms
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

        # Queue model building with research findings
        findings = payload.get("findings", [])
        if findings:
            self._model_build_queue.append({
                "domain": payload.get("domain", "general"),
                "trace": findings,
                "query": payload.get("source_goal", ""),
                "source": "curiosity_research",
            })
            self._stats["models_queued"] += 1

        return None

    # ─── Background Processing ───────────────────────────────────────

    async def process_build_queue(self) -> int:
        """Process queued causal model building tasks.

        Called during sleep consolidation or idle periods.
        This is where LLM-heavy operations happen.
        """
        if not self.causal_model_builder or not self._model_build_queue:
            return 0

        built = 0
        while self._model_build_queue:
            item = self._model_build_queue.pop(0)
            try:
                model = await asyncio.wait_for(
                    self.causal_model_builder.build_model(
                        concept=item.get("query", "unknown"),
                        domain=item["domain"],
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
