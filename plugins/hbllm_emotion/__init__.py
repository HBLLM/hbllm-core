"""
Emotion Modeling Plugin for HBLLM.

Subscribes to sensory input, evaluates sentiment/emotion using ML models,
and publishes the updated emotional state to influence the IdentityNode.

Supports multiple sentiment analysis backends:
- Transformers (pre-trained models like DistilBERT)
- TextBlob (lightweight rule-based)
- Keyword heuristic (fallback)
"""

import logging
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger("hbllm_emotion")

__plugin__ = {
    "name": "hbllm_emotion",
    "version": "0.2.0",
    "description": "Analyzes conversation to maintain an emotional state using ML models.",
}


class EmotionNode(HBLLMPlugin):
    """
    Subscribes to sensory input to maintain and update the active
    emotional state of the AI.
    """

    def __init__(self, node_id: str = "emotion_analyzer", backend: str = "auto") -> None:
        super().__init__(node_id=node_id, capabilities=["sentiment", "emotion_tracking"])
        self.current_valence = 0.0  # -1.0 to 1.0
        self.current_arousal = 0.0  # 0.0 to 1.0
        self._backend = backend
        self._sentiment_model: Any = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the sentiment analysis backend."""
        if self._backend == "auto":
            # Try transformers first, then textblob, then fallback
            self._backend = self._try_transformers() or self._try_textblob() or "heuristic"
            logger.info("EmotionNode initialized with backend: %s", self._backend)
        elif self._backend == "transformers":
            if not self._try_transformers():
                logger.warning(
                    "Transformers backend requested but not available, falling back to heuristic"
                )
                self._backend = "heuristic"
        elif self._backend == "textblob":
            if not self._try_textblob():
                logger.warning(
                    "TextBlob backend requested but not available, falling back to heuristic"
                )
                self._backend = "heuristic"

    def _try_transformers(self) -> str | None:
        """Try to initialize transformers sentiment model."""
        try:
            from transformers import pipeline  # type: ignore[import-not-found]

            # Use a lightweight model for sentiment analysis
            self._sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,  # CPU
            )
            logger.info("Transformers sentiment model loaded")
            return "transformers"
        except ImportError:
            logger.debug("transformers not installed. Install with: pip install transformers")
            return None
        except Exception as e:
            logger.warning("Failed to load transformers model: %s", e)
            return None

    def _try_textblob(self) -> str | None:
        """Try to initialize TextBlob sentiment."""
        try:
            from textblob import TextBlob  # type: ignore[import-not-found]

            self._sentiment_model = TextBlob
            logger.info("TextBlob sentiment loaded")
            return "textblob"
        except ImportError:
            logger.debug("textblob not installed. Install with: pip install textblob")
            return None

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text and return valence score (-1.0 to 1.0)."""
        if self._backend == "transformers" and self._sentiment_model:
            try:
                result = self._sentiment_model(text)[0]
                # Convert label and score to valence
                if result["label"] == "POSITIVE":
                    return result["score"]
                else:
                    return -result["score"]
            except Exception as e:
                logger.debug("Transformers sentiment analysis failed: %s", e)

        elif self._backend == "textblob" and self._sentiment_model:
            try:
                blob = self._sentiment_model(text)
                # TextBlob returns polarity from -1.0 to 1.0
                return blob.sentiment.polarity
            except Exception as e:
                logger.debug("TextBlob sentiment analysis failed: %s", e)

        # Fallback to heuristic
        return self._heuristic_sentiment(text)

    def _heuristic_sentiment(self, text: str) -> float:
        """Heuristic sentiment analysis using keyword matching."""
        sentiment_score = 0.0
        text_lower = text.lower()

        positive_words = [
            "happy",
            "great",
            "awesome",
            "love",
            "excellent",
            "wonderful",
            "fantastic",
            "amazing",
            "good",
            "nice",
        ]
        negative_words = [
            "sad",
            "angry",
            "hate",
            "terrible",
            "bad",
            "awful",
            "horrible",
            "worst",
            "disappointed",
            "upset",
        ]

        if any(w in text_lower for w in positive_words):
            sentiment_score = 0.5
        elif any(w in text_lower for w in negative_words):
            sentiment_score = -0.5

        return sentiment_score

    @subscribe("sensory.input")
    async def on_sensory_input(self, message: Message) -> None:
        """
        Intercepts sensory input (e.g. from user) and extracts emotion.
        """
        text = message.payload.get("text", "")
        if not text:
            return

        # Use ML-based sentiment analysis
        sentiment_score = self._analyze_sentiment(text)

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
        elif self.current_valence < -0.6:
            current_emotion = "angry"

        logger.info(
            "[%s] Emotion updated: valence=%.2f, emotion=%s (backend=%s)",
            self.node_id,
            self.current_valence,
            current_emotion,
            self._backend,
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
                "backend": self._backend,
            },
        )
        if self.bus:
            await self.bus.publish("identity.emotion", emotion_msg)
