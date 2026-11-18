"""
Voice Streaming Bridge — connects ExpressionStream to AudioOutNode.

Provides real-time voice output by intercepting ExpressionStream fragment
callbacks and forwarding completed sentences to AudioOutNode for
sentence-level TTS streaming.

This is the missing link between text generation and voice output:
    ExpressionStream.on_fragment → VoiceStreamBridge → AudioOutNode

Bus Topics:
    sensory.audio.out.stream  → Published (sentence-level TTS requests)
    sensory.stream.end        → Subscribed (to detect response completion)

Usage::

    bridge = VoiceStreamBridge(node_id="voice_bridge")
    await bridge.start(bus)

    # Connect to ExpressionStream:
    expression_stream = ExpressionStream(
        ...,
        on_fragment=bridge.on_fragment,
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class VoiceStreamConfig:
    """Configuration for the voice streaming bridge."""

    # Sentence buffering
    min_sentence_length: int = 10  # Skip very short fragments
    max_sentence_length: int = 500  # Split long sentences
    sentence_delay_ms: float = 50.0  # Delay between sentence dispatches

    # Barge-in support
    allow_barge_in: bool = True  # Allow user to interrupt
    barge_in_topic: str = "perception.wake_word.detected"

    # Voice settings
    default_voice_id: str = "af_heart"
    speed: float = 1.0

    # Filtering
    skip_code_blocks: bool = True  # Don't vocalize code blocks
    skip_urls: bool = True  # Don't vocalize raw URLs


# ── Sentence Buffer ──────────────────────────────────────────────────────────


class _SentenceBuffer:
    """Accumulates text fragments and yields complete sentences."""

    def __init__(self, config: VoiceStreamConfig) -> None:
        self._config = config
        self._buffer = ""
        self._sentence_pattern = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

    def add(self, text: str) -> list[str]:
        """Add text and return any complete sentences."""
        self._buffer += text
        sentences: list[str] = []

        # Split on sentence boundaries
        parts = self._sentence_pattern.split(self._buffer)

        if len(parts) > 1:
            # All parts except last are complete sentences
            for part in parts[:-1]:
                cleaned = self._clean(part.strip())
                if cleaned and len(cleaned) >= self._config.min_sentence_length:
                    sentences.append(cleaned)
            self._buffer = parts[-1]

        # Handle buffer overflow
        if len(self._buffer) > self._config.max_sentence_length:
            cleaned = self._clean(self._buffer.strip())
            if cleaned:
                sentences.append(cleaned)
            self._buffer = ""

        return sentences

    def flush(self) -> list[str]:
        """Flush remaining buffer as final sentence."""
        if self._buffer.strip():
            cleaned = self._clean(self._buffer.strip())
            self._buffer = ""
            if cleaned and len(cleaned) >= self._config.min_sentence_length:
                return [cleaned]
        self._buffer = ""
        return []

    def _clean(self, text: str) -> str:
        """Clean text for vocalization."""
        # Remove MCTS Planner headers
        if "[MCTS Planner]" in text:
            if "\n\n" in text:
                parts = text.split("\n\n")
                text = " ".join(parts[1:])
            else:
                return ""

        text = re.sub(
            r"Best path:\s*(?:\[D\d+:\s*Q=\d*(?:\.\d+)?\s*,\s*N=\d+\](?:\s*(?:→|->)\s*)?)+",
            "",
            text,
        )

        if self._config.skip_code_blocks:
            text = re.sub(r"```[\s\S]*?```", " code block omitted ", text)
            text = re.sub(r"`[^`]+`", "", text)

        if self._config.skip_urls:
            text = re.sub(r"https?://\S+", "", text)

        # Remove list numbers/bullets at start of sentences
        text = re.sub(r"(?:^|\s)\d+\.\s+", " ", text)
        text = re.sub(r"(?:^|\s)[-*•]\s+", " ", text)

        # Remove markdown formatting
        text = re.sub(r"[*_~]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text


# ── Voice Streaming Bridge Node ──────────────────────────────────────────────


class VoiceStreamBridge(Node):
    """Bridges ExpressionStream text output to AudioOutNode voice output.

    Accumulates text fragments from ExpressionStream, extracts complete
    sentences, and dispatches them to AudioOutNode for real-time
    sentence-level TTS synthesis.

    This enables "start speaking as soon as the first sentence is ready"
    behavior, rather than waiting for the full response.
    """

    def __init__(
        self,
        node_id: str = "voice_stream_bridge",
        config: VoiceStreamConfig | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["voice_streaming", "sentence_tts", "barge_in"],
        )
        self.config = config or VoiceStreamConfig()
        self._buffers: dict[str, _SentenceBuffer] = {}
        self._active_sessions: set[str] = set()
        self._interrupted_sessions: set[str] = set()

        # Stats
        self._total_sentences: int = 0
        self._total_interruptions: int = 0

    async def on_start(self) -> None:
        """Subscribe to stream events and barge-in signals."""
        await self.bus.subscribe("sensory.stream.end", self._on_stream_end)

        if self.config.allow_barge_in:
            await self.bus.subscribe(self.config.barge_in_topic, self._on_barge_in)

        logger.info(
            "VoiceStreamBridge started (barge_in=%s)",
            self.config.allow_barge_in,
        )

    async def on_stop(self) -> None:
        """Clean up buffers."""
        self._buffers.clear()
        self._active_sessions.clear()
        logger.info("VoiceStreamBridge stopped")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Fragment Callback (for ExpressionStream) ─────────────────────────

    async def on_fragment(self, fragment: Any) -> None:
        """Callback for ExpressionStream.on_fragment.

        This method is passed as the ``on_fragment`` callback to
        ExpressionStream, receiving each ThoughtFragment as it's generated.

        Args:
            fragment: A ThoughtFragment with ``.text`` and ``.goal_id``
        """
        text = getattr(fragment, "text", "")
        if not text:
            return

        session_id = getattr(fragment, "session_id", "default")

        # Check if session was interrupted
        if session_id in self._interrupted_sessions:
            return

        # Get or create buffer
        if session_id not in self._buffers:
            self._buffers[session_id] = _SentenceBuffer(self.config)
            self._active_sessions.add(session_id)

        buf = self._buffers[session_id]
        sentences = buf.add(text)

        for sentence in sentences:
            await self._dispatch_sentence(session_id, sentence, is_final=False)

    # ── Bus Event Handlers ───────────────────────────────────────────────

    async def _on_stream_end(self, message: Message) -> None:
        """Handle end of a response stream — flush remaining buffer."""
        session_id = message.session_id or "default"

        if session_id in self._buffers:
            remaining = self._buffers[session_id].flush()
            for sentence in remaining:
                await self._dispatch_sentence(session_id, sentence, is_final=True)
            del self._buffers[session_id]

        self._active_sessions.discard(session_id)
        self._interrupted_sessions.discard(session_id)

    async def _on_barge_in(self, message: Message) -> None:
        """Handle barge-in (user speaking while system is talking)."""
        session_id = message.session_id or "default"

        if session_id in self._active_sessions:
            self._interrupted_sessions.add(session_id)
            self._total_interruptions += 1

            logger.info(
                "🔇 Barge-in: interrupted voice stream for session %s",
                session_id,
            )

            # Publish cancellation
            await self.bus.publish(
                "sensory.audio.out.cancel",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    session_id=session_id,
                    topic="sensory.audio.out.cancel",
                    payload={"reason": "barge_in"},
                ),
            )

    # ── Sentence Dispatch ────────────────────────────────────────────────

    async def _dispatch_sentence(self, session_id: str, sentence: str, is_final: bool) -> None:
        """Send a sentence to AudioOutNode for TTS synthesis."""
        self._total_sentences += 1

        await self.bus.publish(
            "sensory.audio.out.stream",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                session_id=session_id,
                topic="sensory.audio.out.stream",
                payload={
                    "text": sentence,
                    "is_final": is_final,
                    "voice_id": self.config.default_voice_id,
                    "speed": self.config.speed,
                    "sentence_index": self._total_sentences,
                },
            ),
        )

        logger.debug(
            "📢 Voice dispatch [%s]: '%s' (final=%s)",
            session_id,
            sentence[:60],
            is_final,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def start_session(self, session_id: str) -> None:
        """Explicitly start voice streaming for a session."""
        self._buffers[session_id] = _SentenceBuffer(self.config)
        self._active_sessions.add(session_id)
        self._interrupted_sessions.discard(session_id)

    def stop_session(self, session_id: str) -> None:
        """Stop voice streaming for a session."""
        self._buffers.pop(session_id, None)
        self._active_sessions.discard(session_id)
        self._interrupted_sessions.discard(session_id)

    def is_active(self, session_id: str) -> bool:
        """Check if a session has active voice streaming."""
        return session_id in self._active_sessions

    def stats(self) -> dict[str, Any]:
        """Return bridge statistics."""
        return {
            "active_sessions": len(self._active_sessions),
            "total_sentences_dispatched": self._total_sentences,
            "total_interruptions": self._total_interruptions,
            "barge_in_enabled": self.config.allow_barge_in,
        }
