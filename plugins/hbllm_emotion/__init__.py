"""
Emotion Modeling Plugin for HBLLM.

Subscribes to sensory input, evaluates sentiment/emotion heuristically,
and publishes the updated emotional state to influence the IdentityNode.
"""

import logging
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger("hbllm_emotion")

__plugin__ = {
    "name": "hbllm_emotion",
    "version": "0.1.0",
    "description": "Analyzes conversation to maintain an emotional state.",
}


class EmotionNode(HBLLMPlugin):
    """
    Subscribes to sensory input to maintain and update the active
    emotional state of the AI.
    """

    def __init__(self, node_id: str = "emotion_analyzer") -> None:
        super().__init__(node_id=node_id, capabilities=["sentiment", "emotion_tracking"])
        self.current_valence = 0.0  # -1.0 to 1.0
        self.current_arousal = 0.0  # 0.0 to 1.0

    @subscribe("sensory.input")
    async def on_sensory_input(self, message: Message) -> None:
        """
        Intercepts sensory input (e.g. from user) and extracts emotion.
        """
        text = message.payload.get("text", "")
        if not text:
            return

        # TODO: Replace heuristic with a tiny Sentiment/Value LoRA
        # Mocking sentiment extraction based on keywords
        sentiment_score = 0.0
        text_lower = text.lower()

        if any(w in text_lower for w in ["happy", "great", "awesome", "love"]):
            sentiment_score = 0.5
        elif any(w in text_lower for w in ["sad", "angry", "hate", "terrible", "bad"]):
            sentiment_score = -0.5

        # Decay current emotion slightly towards neutral
        self.current_valence *= 0.8
        self.current_valence += sentiment_score

        # Clamp valence
        self.current_valence = max(-1.0, min(1.0, self.current_valence))

        # Classify basic emotion
        current_emotion = "neutral"
        if self.current_valence > 0.3:
            current_emotion = "happy"
        elif self.current_valence < -0.3:
            current_emotion = "sad"

        logger.info(
            "[%s] Emotion updated: valence=%.2f, emotion=%s",
            self.node_id,
            self.current_valence,
            current_emotion,
        )

        # Publish emotional state to the bus
        emotion_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            topic="identity.emotion",
            payload={
                "valence": self.current_valence,
                "arousal": self.current_arousal,
                "emotion_label": current_emotion,
                "triggered_by": message.id,
            },
        )
        if self.bus:
            await self.bus.publish("identity.emotion", emotion_msg)
