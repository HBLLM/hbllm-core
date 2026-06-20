"""
Conversation Turn Manager — full-duplex audio state machine.

Manages the flow of a voice conversation:
    IDLE → LISTENING → PROCESSING → SPEAKING → LISTENING → ...

Handles:
    - Wake word activation (IDLE → LISTENING)
    - End-of-utterance detection (LISTENING → PROCESSING)
    - Barge-in interruption (SPEAKING → LISTENING, cancels TTS)
    - Silence timeout (LISTENING → IDLE after no speech)
    - Concurrent audio I/O (listen while speaking for barge-in)

Integrates with:
    - AudioInputNode (STT)
    - AudioOutputNode (TTS)
    - WakeWordDetector
    - AutonomyCore (proactive speech)
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class TurnState(StrEnum):
    """Conversation turn states."""

    IDLE = "idle"  # Not in conversation — waiting for wake word
    LISTENING = "listening"  # Capturing user speech
    PROCESSING = "processing"  # Brain is thinking
    SPEAKING = "speaking"  # TTS output active
    INTERRUPTED = "interrupted"  # User barged in during speech


class ConversationTurnManager(Node):
    """
    State machine that orchestrates full-duplex voice conversation.

    Subscribes to:
        - sensory.wake_word      → IDLE → LISTENING
        - sensory.transcription  → LISTENING → PROCESSING
        - sensory.output         → PROCESSING → SPEAKING
        - sensory.audio.chunk    → tracks TTS progress
        - sensory.audio.stream   → detects barge-in during SPEAKING

    Publishes:
        - conversation.state_change   → state transition notifications
        - sensory.audio.out           → TTS requests
        - sensory.audio.interrupt     → cancel TTS on barge-in
    """

    # Configuration
    IDLE_TIMEOUT_S = 30.0  # Return to IDLE after this much silence in LISTENING
    PROCESSING_TIMEOUT_S = 30.0  # Abort if brain doesn't respond
    SPEAKING_BARGE_IN_THRESHOLD = 0.3  # VAD probability to trigger barge-in
    CONTINUOUS_LISTEN = True  # Stay in LISTENING after SPEAKING (no wake word needed)

    def __init__(self, node_id: str = "conversation_turn") -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["turn_management", "barge_in", "conversation_state"],
        )
        self._state = TurnState.IDLE
        self._state_since = time.monotonic()
        self._current_session: str | None = None
        self._speaking_task: asyncio.Task[Any] | None = None
        self._timeout_task: asyncio.Task[Any] | None = None
        self._turn_count = 0

    @property
    def state(self) -> TurnState:
        return self._state

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def on_start(self) -> None:
        """Subscribe to all conversation-relevant topics."""
        logger.info("ConversationTurnManager starting (state=IDLE)")
        await self.bus.subscribe("sensory.wake_word", self._on_wake_word)
        await self.bus.subscribe("sensory.transcription", self._on_transcription)
        await self.bus.subscribe("sensory.output", self._on_brain_output)
        await self.bus.subscribe("sensory.audio.chunk", self._on_tts_chunk)
        await self.bus.subscribe("sensory.tts.done", self._on_tts_done)
        await self.bus.subscribe("conversation.force_idle", self._on_force_idle)

    async def on_stop(self) -> None:
        logger.info("ConversationTurnManager stopping")
        if self._timeout_task:
            self._timeout_task.cancel()
        if self._speaking_task:
            self._speaking_task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── State Transitions ────────────────────────────────────────────────

    async def _transition(self, new_state: TurnState, reason: str = "") -> None:
        """Perform a state transition with logging and notification."""
        old = self._state
        if old == new_state:
            return

        self._state = new_state
        self._state_since = time.monotonic()

        logger.info(
            "Turn state: %s → %s (reason=%s, session=%s)",
            old.value, new_state.value, reason, self._current_session,
        )

        # Cancel any pending timeout
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            self._timeout_task = None

        # Publish state change
        await self.bus.publish(
            "conversation.state_change",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                session_id=self._current_session,
                topic="conversation.state_change",
                payload={
                    "old_state": old.value,
                    "new_state": new_state.value,
                    "reason": reason,
                    "turn_count": self._turn_count,
                },
            ),
        )

        # Start appropriate timeout
        if new_state == TurnState.LISTENING:
            self._timeout_task = asyncio.create_task(
                self._idle_timeout()
            )
        elif new_state == TurnState.PROCESSING:
            self._timeout_task = asyncio.create_task(
                self._processing_timeout()
            )

    # ── Event Handlers ───────────────────────────────────────────────────

    async def _on_wake_word(self, msg: Message) -> None:
        """Wake word detected — start listening."""
        if self._state == TurnState.IDLE:
            self._current_session = msg.session_id or f"voice_{int(time.time())}"
            self._turn_count = 0
            await self._transition(TurnState.LISTENING, "wake_word")
        elif self._state == TurnState.SPEAKING:
            # Barge-in: user said wake word while we're speaking
            await self._handle_barge_in("wake_word")

    async def _on_transcription(self, msg: Message) -> None:
        """Speech transcribed — send to brain for processing."""
        if self._state != TurnState.LISTENING:
            return

        text = msg.payload.get("text", "").strip()
        if not text:
            return

        self._turn_count += 1
        await self._transition(TurnState.PROCESSING, "transcription_received")

        # Forward to brain pipeline
        await self.bus.publish(
            "router.query",
            Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                session_id=self._current_session,
                topic="router.query",
                payload={
                    "text": text,
                    "source": "voice",
                    "turn_number": self._turn_count,
                    "speaker_id": msg.payload.get("speaker_id", "unknown"),
                    "speaker_name": msg.payload.get("speaker_name", ""),
                },
                correlation_id=msg.correlation_id,
            ),
        )

    async def _on_brain_output(self, msg: Message) -> None:
        """Brain produced a response — speak it."""
        if self._state != TurnState.PROCESSING:
            return

        response_text = msg.payload.get("response", "") or msg.payload.get("text", "")
        if not response_text:
            # Empty response — go back to listening
            if self.CONTINUOUS_LISTEN:
                await self._transition(TurnState.LISTENING, "empty_response")
            else:
                await self._transition(TurnState.IDLE, "empty_response")
            return

        await self._transition(TurnState.SPEAKING, "brain_responded")

        # Request TTS
        await self.bus.publish(
            "sensory.audio.out",
            Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                session_id=self._current_session,
                topic="sensory.audio.out",
                payload={
                    "text": response_text,
                    "stream": True,  # Sentence-level streaming for low latency
                },
                correlation_id=msg.correlation_id,
            ),
        )

    async def _on_tts_chunk(self, msg: Message) -> None:
        """TTS audio chunk generated — track progress."""
        if self._state != TurnState.SPEAKING:
            return

        is_final = msg.payload.get("is_final", False)
        if is_final:
            await self._on_tts_done(msg)

    async def _on_tts_done(self, msg: Message) -> None:
        """TTS finished speaking — return to listening or idle."""
        if self._state != TurnState.SPEAKING:
            return

        if self.CONTINUOUS_LISTEN:
            await self._transition(TurnState.LISTENING, "tts_complete")
        else:
            await self._transition(TurnState.IDLE, "tts_complete")

    async def _on_force_idle(self, msg: Message) -> None:
        """Force return to idle (e.g., user said 'stop' or timeout)."""
        await self._transition(TurnState.IDLE, "forced")

    # ── Barge-in ─────────────────────────────────────────────────────────

    async def _handle_barge_in(self, reason: str) -> None:
        """User interrupted during speech — cancel TTS and listen."""
        logger.info("Barge-in detected (reason=%s)", reason)

        # Cancel TTS playback
        await self.bus.publish(
            "sensory.audio.interrupt",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                session_id=self._current_session,
                topic="sensory.audio.interrupt",
                payload={"reason": reason},
            ),
        )

        await self._transition(TurnState.LISTENING, f"barge_in:{reason}")

    # ── Timeouts ─────────────────────────────────────────────────────────

    async def _idle_timeout(self) -> None:
        """Return to IDLE if no speech detected within timeout."""
        try:
            await asyncio.sleep(self.IDLE_TIMEOUT_S)
            if self._state == TurnState.LISTENING:
                await self._transition(TurnState.IDLE, "silence_timeout")
        except asyncio.CancelledError:
            pass

    async def _processing_timeout(self) -> None:
        """Abort processing if brain doesn't respond."""
        try:
            await asyncio.sleep(self.PROCESSING_TIMEOUT_S)
            if self._state == TurnState.PROCESSING:
                logger.warning("Processing timeout — brain didn't respond in %.0fs", self.PROCESSING_TIMEOUT_S)
                if self.CONTINUOUS_LISTEN:
                    await self._transition(TurnState.LISTENING, "processing_timeout")
                else:
                    await self._transition(TurnState.IDLE, "processing_timeout")
        except asyncio.CancelledError:
            pass

    # ── Status ───────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Current conversation state snapshot."""
        return {
            "state": self._state.value,
            "state_duration_s": round(time.monotonic() - self._state_since, 1),
            "session_id": self._current_session,
            "turn_count": self._turn_count,
            "continuous_listen": self.CONTINUOUS_LISTEN,
        }
