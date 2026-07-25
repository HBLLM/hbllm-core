"""
HCIR Bus Bridge — primary integration: MessageBus → HCIR Projection.

The bus bridge is the ONLY component that connects the existing HBLLM
node architecture to the HCIR Cognitive OS.  No node ever imports HCIR.

Architecture::

    Nodes
      ↓
    MessageBus
      ↓
    HCIRBusBridge
      ├── SemanticNormalizer   (raw event → canonical kind)
      ├── CognitiveJournal     (record every event)
      ├── CognitiveEventLog    (record significant events)
      └── TransactionManager   (build + commit HCIR transactions)
           ↓
         TieredWorkspace       (graph projection)

Core invariant: **Nodes emit intent. HCIR owns state.**

Usage::

    bridge = HCIRBusBridge(bus, normalizer, journal, event_log,
                           tiered_workspace, tx_manager)
    await bridge.start()
    # ... nodes publish events normally ...
    await bridge.stop()
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from hbllm.hcir.cognitive_event_log import CognitiveEventLog
from hbllm.hcir.cognitive_journal import CognitiveEvent, CognitiveJournal
from hbllm.hcir.graph import (
    ActionNode,
    BeliefNode,
    EventNode,
    GoalNode,
    ObservationNode,
    SkillNode,
)
from hbllm.hcir.semantic_normalizer import CognitiveEventKind, SemanticNormalizer
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope
from hbllm.hcir.workspace_tiers import TieredWorkspace, WorkspaceTier

if TYPE_CHECKING:
    from hbllm.hcir.kernel.transaction_manager import TransactionManager
    from hbllm.network.bus import MessageBus, Subscription
    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Topic Patterns
# ═══════════════════════════════════════════════════════════════════════════

# Bus topics that the bridge subscribes to.
# Each entry is a topic prefix — the bridge intercepts all matching messages.
_SUBSCRIBED_TOPICS: list[str] = [
    "memory.store",
    "memory.search",
    "memory.recall",
    "memory.reflection",
    "memory.feedback",
    "memory.browse",
    "memory.forget",
    "perception.vision",
    "perception.audio",
    "perception.video",
    "perception.multimodal",
    "decision.made",
    "decision.result",
    "planning.goal_created",
    "planning.goal_completed",
    "planning.plan_created",
    "action.executed",
    "action.result",
    "governance.evaluation",
    "governance.violation",
    "router.query",
    "router.decision",
    "learning.skill",
    "learning.update",
    "cognitive_state.updated",
    "cognitive_state.snapshot",
    "emotion.state",
    "emotion.update",
    "world.state",
    "world.prediction",
]


# ═══════════════════════════════════════════════════════════════════════════
# HCIR Bus Bridge
# ═══════════════════════════════════════════════════════════════════════════


class HCIRBusBridge:
    """Primary integration mechanism: MessageBus → HCIR projection.

    Subscribes to all relevant bus topics and projects domain events
    into the HCIR workspace via the Cognitive Event Log.  Nodes never
    know HCIR exists — they emit domain events normally.

    Pipeline per event:
        1. Normalize raw event to canonical ``CognitiveEventKind``
        2. Record to ``CognitiveJournal`` (always, every event)
        3. Record to ``CognitiveEventLog`` (only state transitions)
        4. Build ``HCIRTransaction`` from canonical event
        5. Submit through ``TransactionManager`` (constitutional checks)
        6. Notify ``TieredWorkspace`` of commit (periodic snapshot trigger)

    Usage::

        bridge = HCIRBusBridge(bus, normalizer, journal, event_log,
                               tiered_workspace, tx_manager)
        await bridge.start()
    """

    def __init__(
        self,
        bus: MessageBus,
        normalizer: SemanticNormalizer,
        journal: CognitiveJournal,
        event_log: CognitiveEventLog,
        tiered_workspace: TieredWorkspace,
        tx_manager: TransactionManager | None = None,
    ) -> None:
        self._bus = bus
        self._normalizer = normalizer
        self._journal = journal
        self._event_log = event_log
        self._workspace = tiered_workspace
        self._tx_manager = tx_manager
        self._subscriptions: list[Subscription] = []
        self._running = False
        self._events_processed: int = 0
        self._events_projected: int = 0

    @property
    def events_processed(self) -> int:
        """Total number of events received from the bus."""
        return self._events_processed

    @property
    def events_projected(self) -> int:
        """Number of events that resulted in HCIR transactions."""
        return self._events_projected

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to all relevant bus topics."""
        if self._running:
            return

        for topic in _SUBSCRIBED_TOPICS:
            try:
                sub = await self._bus.subscribe(topic, self._on_event)
                self._subscriptions.append(sub)
            except Exception as exc:
                logger.warning("BusBridge: failed to subscribe to %s: %s", topic, exc)

        self._running = True
        logger.info(
            "HCIRBusBridge started: subscribed to %d topics",
            len(self._subscriptions),
        )

    async def stop(self) -> None:
        """Unsubscribe from all bus topics."""
        for sub in self._subscriptions:
            try:
                await self._bus.unsubscribe(sub)
            except Exception:
                pass
        self._subscriptions.clear()
        self._running = False
        logger.info(
            "HCIRBusBridge stopped: processed=%d, projected=%d",
            self._events_processed,
            self._events_projected,
        )

    # ── Event Handler ────────────────────────────────────────────────

    async def _on_event(self, message: Message) -> Message | None:
        """Handle a bus event: normalize → journal → log → project.

        This is the core integration pipeline.  Every bus event
        flows through this method.
        """
        self._events_processed += 1

        # Extract topic from message metadata
        topic = self._extract_topic(message)

        # 1. Normalize raw event to canonical vocabulary
        kind = self._normalizer.normalize(topic, message)
        if kind is None:
            # Not a recognized cognitive event — still journal it
            self._journal.record(
                CognitiveEvent(
                    kind=CognitiveEventKind.OBSERVATION_RECEIVED,
                    author=getattr(message, "source_node_id", "unknown"),
                    tenant_id=getattr(message, "tenant_id", "default"),
                    data={"raw_topic": topic, "unrecognized": True},
                    raw_topic=topic,
                )
            )
            return None

        # 2. Build CognitiveEvent with full provenance
        cognitive_event = self._build_cognitive_event(kind, message, topic)

        # 3. Record to journal (always, every event)
        self._journal.record(cognitive_event)

        # 4. Record to event log (only state transitions)
        self._event_log.record_if_significant(cognitive_event)

        # 5. Project to HCIR workspace via transaction
        self._project_to_workspace(kind, cognitive_event, message)

        return None

    # ── Projection Logic ─────────────────────────────────────────────

    def _project_to_workspace(
        self,
        kind: CognitiveEventKind,
        event: CognitiveEvent,
        message: Message,
    ) -> None:
        """Build and commit an HCIR transaction from a cognitive event.

        Different event kinds produce different graph node types:
            GOAL_CREATED → GoalNode in working workspace
            DECISION_MADE → EventNode in brain workspace
            MEMORY_STORED → persistent workspace
            COGNITIVE_STATE_CHANGED → ObservationNode in brain workspace
            etc.
        """
        operations = self._build_operations(kind, event, message)
        if not operations:
            return

        provenance = Provenance(
            created_by=event.author,
            session_id=event.session_id,
            goal_id=event.goal_id,
            trace_id=event.trace_id,
            model_used=event.model_used,
            reason=f"Bus projection: {kind.value}",
            source_node=event.source_node,
            logical_time=event.logical_time,
        )

        tx = HCIRTransaction(
            author=f"bus_bridge:{event.author}",
            operations=operations,
            provenance=provenance,
        )

        # Commit through transaction manager if available
        if self._tx_manager is not None:
            result = self._tx_manager.commit(tx)
            if result.is_committed:
                self._events_projected += 1
                self._workspace.notify_commit()
        else:
            # Direct write (no governance) — used in testing
            self._apply_direct(kind, event, message)
            self._events_projected += 1

    def _build_operations(
        self,
        kind: CognitiveEventKind,
        event: CognitiveEvent,
        message: Message,
    ) -> list[TransactionOperation]:
        """Build transaction operations for a cognitive event."""
        node = self._build_node(kind, event, message)
        if node is None:
            return []

        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node.id,
                node_data=node.model_dump(),
            )
        ]

    def _build_node(
        self,
        kind: CognitiveEventKind,
        event: CognitiveEvent,
        message: Message,
    ) -> Any | None:
        """Build the appropriate HCIR graph node for a cognitive event kind."""
        node_id = f"proj_{event.id}"
        scope = Scope(
            tenant_id=event.tenant_id,
        )
        provenance = Provenance(
            created_by=event.author,
            session_id=event.session_id,
            goal_id=event.goal_id,
            trace_id=event.trace_id,
            logical_time=event.logical_time,
            reason=event.reason,
        )

        msg_data = getattr(message, "data", {}) or {}

        if kind == CognitiveEventKind.GOAL_CREATED:
            return GoalNode(
                id=node_id,
                description=msg_data.get("description", event.reason or "Projected goal"),
                priority=msg_data.get("priority", 0.5),
                scope=scope,
                provenance=provenance,
                tags=["bus_projected"],
            )

        elif kind in (CognitiveEventKind.DECISION_MADE, CognitiveEventKind.ROUTING_DECIDED):
            return EventNode(
                id=node_id,
                event_kind=kind.value,
                event_data=dict(event.data),
                scope=scope,
                provenance=provenance,
                tags=["bus_projected", "decision"],
            )

        elif kind in (
            CognitiveEventKind.ACTION_PLANNED,
            CognitiveEventKind.ACTION_EXECUTED,
        ):
            return ActionNode(
                id=node_id,
                intent=msg_data.get("intent", kind.value),
                scope=scope,
                provenance=provenance,
                tags=["bus_projected", "action"],
            )

        elif kind == CognitiveEventKind.MEMORY_STORED:
            return EventNode(
                id=node_id,
                event_kind="memory_stored",
                event_data=dict(event.data),
                scope=scope,
                provenance=provenance,
                tags=["bus_projected", "memory"],
            )

        elif kind == CognitiveEventKind.SKILL_LEARNED:
            return SkillNode(
                id=node_id,
                skill_name=msg_data.get("skill_name", "unknown_skill"),
                description=msg_data.get("description", "Bus-projected skill"),
                success_rate=msg_data.get("success_rate", 0.5),
                scope=scope,
                provenance=provenance,
                tags=["bus_projected", "skill"],
            )

        elif kind in (
            CognitiveEventKind.BELIEF_UPDATED,
            CognitiveEventKind.BELIEF_REVISED,
        ):
            return BeliefNode(
                id=node_id,
                claim=msg_data.get("claim", event.reason or "Projected belief"),
                evidence_sources=[event.author],
                scope=scope,
                provenance=provenance,
                tags=["bus_projected", "belief"],
            )

        elif kind in (
            CognitiveEventKind.OBSERVATION_RECEIVED,
            CognitiveEventKind.PERCEPTION_RECEIVED,
            CognitiveEventKind.COGNITIVE_STATE_CHANGED,
            CognitiveEventKind.ATTENTION_SHIFTED,
            CognitiveEventKind.EMOTION_CHANGED,
            CognitiveEventKind.WORLD_STATE_UPDATED,
        ):
            return ObservationNode(
                id=node_id,
                sensor_source=event.author,
                payload=dict(event.data),
                scope=scope,
                provenance=provenance,
                tags=["bus_projected", kind.value.split(".")[0]],
            )

        elif kind in (
            CognitiveEventKind.GOVERNANCE_EVALUATED,
            CognitiveEventKind.GOVERNANCE_BLOCKED,
        ):
            return EventNode(
                id=node_id,
                event_kind=kind.value,
                event_data=dict(event.data),
                scope=scope,
                provenance=provenance,
                tags=["bus_projected", "governance"],
            )

        else:
            # Generic event node for any unhandled kind
            return EventNode(
                id=node_id,
                event_kind=kind.value,
                event_data=dict(event.data),
                scope=scope,
                provenance=provenance,
                tags=["bus_projected"],
            )

    def _apply_direct(
        self,
        kind: CognitiveEventKind,
        event: CognitiveEvent,
        message: Message,
    ) -> None:
        """Direct write to workspace without transaction manager.

        Used when no TransactionManager is configured (testing/bootstrap).
        Routes nodes to the appropriate workspace tier.
        """
        node = self._build_node(kind, event, message)
        if node is None:
            return

        tier = self._route_to_tier(kind)
        if tier == WorkspaceTier.WORKING:
            # Find or create a task frame for this goal
            goal_id = event.goal_id or "default_goal"
            frame = self._workspace.working.get_frame_by_goal(goal_id)
            if frame is None:
                frame = self._workspace.create_task_frame(goal_id)
            frame.workspace.upsert_node(node, author=f"bus_bridge:{event.author}")
        elif tier == WorkspaceTier.META:
            self._workspace.meta.upsert_node(node, author=f"bus_bridge:{event.author}")
        elif tier == WorkspaceTier.PERSISTENT:
            self._workspace.persistent.upsert_node(node, author=f"bus_bridge:{event.author}")
        else:
            self._workspace.brain.upsert_node(node, author=f"bus_bridge:{event.author}")

    def _route_to_tier(self, kind: CognitiveEventKind) -> WorkspaceTier:
        """Determine which workspace tier a cognitive event belongs to."""
        if kind in (
            CognitiveEventKind.GOAL_CREATED,
            CognitiveEventKind.GOAL_COMPLETED,
            CognitiveEventKind.GOAL_ABANDONED,
            CognitiveEventKind.ACTION_PLANNED,
            CognitiveEventKind.ACTION_EXECUTED,
            CognitiveEventKind.PREDICTION_MADE,
        ):
            return WorkspaceTier.WORKING

        if kind in (
            CognitiveEventKind.MEMORY_STORED,
            CognitiveEventKind.MEMORY_CONSOLIDATED,
            CognitiveEventKind.SKILL_LEARNED,
            CognitiveEventKind.BELIEF_UPDATED,
            CognitiveEventKind.BELIEF_REVISED,
        ):
            return WorkspaceTier.PERSISTENT

        if kind in (
            CognitiveEventKind.COGNITIVE_STATE_CHANGED,
            CognitiveEventKind.ATTENTION_SHIFTED,
            CognitiveEventKind.EMOTION_CHANGED,
            CognitiveEventKind.DECISION_MADE,
            CognitiveEventKind.ROUTING_DECIDED,
        ):
            return WorkspaceTier.BRAIN

        # Default: brain workspace
        return WorkspaceTier.BRAIN

    # ── Helpers ──────────────────────────────────────────────────────

    def _build_cognitive_event(
        self,
        kind: CognitiveEventKind,
        message: Message,
        topic: str,
    ) -> CognitiveEvent:
        """Build a CognitiveEvent from a bus message."""
        msg_data = getattr(message, "data", {}) or {}

        return CognitiveEvent(
            kind=kind,
            timestamp=time.time(),
            author=getattr(message, "source_node_id", "unknown"),
            tenant_id=getattr(message, "tenant_id", "default"),
            session_id=msg_data.get("session_id", ""),
            goal_id=msg_data.get("goal_id", ""),
            trace_id=getattr(message, "id", ""),
            model_used=msg_data.get("model_used", ""),
            confidence=msg_data.get("confidence", 1.0),
            reason=msg_data.get("reason", ""),
            source_node=getattr(message, "source_node_id", ""),
            logical_time=self._journal.logical_clock,
            data=msg_data if isinstance(msg_data, dict) else {},
            raw_topic=topic,
        )

    @staticmethod
    def _extract_topic(message: Message) -> str:
        """Extract the bus topic from a message.

        Messages don't always carry their topic directly.
        We check multiple locations.
        """
        # Check for explicit topic field
        topic = getattr(message, "topic", None)
        if topic:
            return str(topic)

        # Infer from message type
        msg_type = getattr(message, "type", None)
        if msg_type:
            return str(msg_type)

        return "unknown"
