"""
UserModelNode — bus-connected adapter for the UserModelEngine.

Listens for interaction events and continuously updates the user model.
Publishes model change notifications for downstream consumers.

Bus Topics:
    Subscribes:
        system.experience    → Extract expertise, interest, preference signals
        system.feedback      → Learn from explicit user feedback
        system.evaluation    → Track trust (did user accept output?)
        emotion.state        → Update stress/engagement proxy
        habit.detected       → Incorporate temporal patterns
        sensory.transcription → Link voice identity to user model

    Publishes:
        user.model.updated   → When model changes significantly
        user.model.focus_changed → When user's focus topic shifts
"""

from __future__ import annotations

import logging

from hbllm.brain.user_model import UserModelEngine
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class UserModelNode(Node):
    """Bus-connected wrapper for UserModelEngine.

    Subscribes to cognitive events and continuously learns about the user.
    Thin adapter — all logic lives in UserModelEngine.
    """

    def __init__(
        self,
        node_id: str,
        user_model_engine: UserModelEngine | None = None,
        data_dir: str = "data",
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["user_modeling", "preference_learning", "expertise_inference"],
        )
        self.engine = user_model_engine or UserModelEngine(data_dir=data_dir)
        self._updates_emitted = 0

    async def on_start(self) -> None:
        logger.info("Starting UserModelNode")
        await self.bus.subscribe("system.experience", self._handle_experience)
        await self.bus.subscribe("system.feedback", self._handle_feedback)
        await self.bus.subscribe("system.evaluation", self._handle_evaluation)
        await self.bus.subscribe("emotion.state", self._handle_emotion)
        await self.bus.subscribe("habit.detected", self._handle_habit)
        await self.bus.subscribe("user_model.query", self._handle_query)
        await self.bus.subscribe("sensory.transcription", self._handle_voice_identity)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping UserModelNode — %d updates emitted",
            self._updates_emitted,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Event Handlers ───────────────────────────────────────────────

    async def _handle_experience(self, message: Message) -> None:
        """Extract user signals from every interaction."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        query = payload.get("text", payload.get("query", ""))
        response = payload.get("response", "")

        if not query:
            return

        changed = self.engine.update_from_interaction(
            tenant_id=tenant_id,
            query=query,
            response=response,
            metadata=payload,
        )

        if changed:
            self._updates_emitted += 1
            model = self.engine.get_model(tenant_id)
            await self.publish(
                "user.model.updated",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="user.model.updated",
                    tenant_id=tenant_id,
                    payload={
                        "tenant_id": tenant_id,
                        "focus": model.current_focus.value,
                        "expertise_count": len(model.expertise),
                        "stats": self.engine.stats(tenant_id),
                    },
                ),
            )

    async def _handle_feedback(self, message: Message) -> None:
        """Learn from explicit user feedback."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        rating = payload.get("rating", 0)

        # Negative feedback = potential preference signal
        if rating < 0:
            reason = payload.get("reason", "")
            if "verbose" in reason.lower():
                self.engine.learn_preference(tenant_id, "verbosity", "concise", source="explicit")
            elif "brief" in reason.lower() or "short" in reason.lower():
                self.engine.learn_preference(tenant_id, "verbosity", "detailed", source="explicit")
            elif "technical" in reason.lower():
                self.engine.learn_preference(
                    tenant_id, "technical_depth", "less_technical", source="explicit"
                )

    async def _handle_evaluation(self, message: Message) -> None:
        """Track trust from evaluation results."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        domain = str(payload.get("domain", payload.get("category", "general")))
        accepted = payload.get("accepted", payload.get("overall_score", 0.5) > 0.6)
        overridden = payload.get("overridden", False)

        if accepted:
            self.engine.update_trust(tenant_id, domain, delegated=True)
        if overridden:
            self.engine.update_trust(tenant_id, domain, overridden=True)

    async def _handle_emotion(self, message: Message) -> None:
        """Update stress/engagement from emotion signals."""
        payload = message.payload
        tenant_id = message.tenant_id or "default"
        valence = payload.get("valence", 0.0)
        arousal = payload.get("arousal", 0.5)

        # Map emotion dimensions to stress/engagement
        stress = max(0.0, -valence * 0.5 + arousal * 0.5)
        self.engine.update_stress(tenant_id, stress)

        engagement = max(0.0, arousal * 0.7 + abs(valence) * 0.3)
        self.engine.update_engagement(tenant_id, engagement)

    async def _handle_habit(self, message: Message) -> None:
        """Incorporate detected habits as preferences."""
        payload = message.payload
        tenant_id = message.tenant_id or payload.get("tenant_id", "default")
        habit_desc = payload.get("description", "")
        action_type = payload.get("action_type", "")

        if action_type and habit_desc:
            self.engine.learn_preference(
                tenant_id,
                f"habit_{action_type}",
                habit_desc,
                source="inferred",
            )

    async def _handle_query(self, message: Message) -> Message | None:
        """Return user model stats."""
        tenant_id = message.tenant_id or "default"
        return message.create_response(self.engine.stats(tenant_id))

    async def _handle_voice_identity(self, message: Message) -> None:
        """Link voice-identified speaker to the user model.

        When SpeakerIdNode identifies a speaker via voice biometrics,
        this handler bridges that identity into the cognitive model:
        - Maps speaker_id → tenant_id for user model lookups
        - Updates engagement level (speaking = engaged)
        - Records the voice interaction channel preference
        - Feeds the transcribed text as an interaction for learning
        """
        payload = message.payload
        speaker_id = payload.get("speaker_id", "unknown")
        speaker_name = payload.get("speaker_name", "")
        confidence = payload.get("speaker_confidence", 0.0)
        text = payload.get("text", "")

        if speaker_id == "unknown" or confidence < 0.7:
            return

        # Use speaker_id as tenant_id when available — this bridges
        # biometric identity to the cognitive model
        tenant_id = message.tenant_id or speaker_id

        # Record that this user prefers voice interaction
        self.engine.learn_preference(tenant_id, "interaction_channel", "voice", source="inferred")

        # Voice interaction signals engagement
        self.engine.update_engagement(tenant_id, 0.7)

        # Feed the transcription into the model for expertise/focus learning
        if text:
            self.engine.update_from_interaction(
                tenant_id=tenant_id,
                query=text,
                response="",
                metadata={
                    "source": "voice",
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "speaker_confidence": confidence,
                },
            )
