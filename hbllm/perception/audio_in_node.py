"""
Audio Input Node (Speech-to-Text).

Listens for `sensory.audio.in` payloads (file paths or raw bytes),
transcribes the audio using a native Whisper transformer, and immediately
dispatches a router query so the system can respond to spoken word.

Also supports streaming audio via `sensory.audio.stream` — chunks are
buffered per session and auto-transcribed when silence is detected or
the buffer reaches a configurable max duration.
"""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any, cast

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

# Streaming settings
STREAM_MAX_BUFFER_SECONDS = 15.0
STREAM_SILENCE_TIMEOUT = 2.0  # Flush after 2s of no new chunks


class AudioInputNode(Node):
    """
    Service node that acts as the model's "ears".
    Supports both file-based and streaming audio input.
    """

    def __init__(self, node_id: str, model_size: str = "tiny.en") -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["speech_to_text", "audio_streaming"],
        )
        self.model_size = model_size
        self.model: Any | None = None
        # Streaming buffers: session_id -> {"chunks": [bytes], "start": float, "last_chunk": float}
        self._stream_buffers: dict[str, dict[str, Any]] = {}

    def _load_model(self) -> None:
        if self.model is None:
            try:
                import whisper # type: ignore
                logger.info("Loading Whisper %s model for AudioInput...", self.model_size)
                self.model = whisper.load_model(self.model_size)
            except ImportError:
                logger.error("Whisper library not found. Audio transcription will fail.")

    async def on_start(self) -> None:
        """Subscribe to sensory audio streams, streaming input, and workspace evaluation."""
        logger.info("Starting AudioInputNode (streaming enabled)")
        await self.bus.subscribe("sensory.audio.in", self.handle_transcribe)
        await self.bus.subscribe("sensory.audio.stream", self.handle_stream)
        # Multi-modal workspace: participate as a competing thought source
        await self.bus.subscribe("module.evaluate", self.handle_workspace_query)

    async def on_stop(self) -> None:
        logger.info("Stopping AudioInputNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_transcribe(self, message: Message) -> Message | None:
        """
        Handles `sensory.audio.in` messages.
        Payload expects:
            file_path: Optional[str] -> path to wav/mp3 file
        """
        payload = message.payload
        file_path = payload.get("file_path")

        if not file_path:
            return message.create_error("Missing 'file_path'")

        try:
            import asyncio

            # Offload heavy whisper STT inference to thread
            def _transcribe() -> str:
                self._load_model()
                if self.model is None:
                    raise RuntimeError("Whisper model not loaded")
                logger.info("Transcribing audio file: %s", file_path)
                result = self.model.transcribe(file_path, fp16=False)
                return str(result["text"]).strip()

            transcription = await asyncio.to_thread(_transcribe)
            logger.info("Transcribed text: '%s'", transcription)

            # Formulate the response so the Caller knows we succeeded
            resp = message.create_response({"text": transcription})

            # Automatically forward what the user said to the brain
            query_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="router.query",
                payload={"text": transcription},
                correlation_id=message.id,  # Maintain chain
            )
            # Fire and forget the internal brain query
            asyncio.create_task(self.bus.publish("router.query", query_msg))

            return resp

        except Exception as e:
            logger.error("Audio transcription failed: %s", e)
            return message.create_error(str(e))

    async def handle_stream(self, message: Message) -> Message | None:
        """
        Handle streaming audio chunks. Buffers chunks per session and
        auto-transcribes when silence is detected or buffer is full.

        Payload expects:
            chunk: str -> hex-encoded audio bytes
            sample_rate: int -> sample rate (default 16000)
            is_final: bool -> if True, flush buffer immediately
        """
        payload = message.payload
        session_id = message.session_id or "default"
        chunk_hex = str(payload.get("chunk", ""))
        is_final = bool(payload.get("is_final", False))

        if not chunk_hex and not is_final:
            return None

        now = time.monotonic()

        # Initialize buffer for this session
        if session_id not in self._stream_buffers:
            self._stream_buffers[session_id] = {
                "chunks": [],
                "start": now,
                "last_chunk": now,
                "sample_rate": payload.get("sample_rate", 16000),
            }

        buf = self._stream_buffers[session_id]

        # Append chunk
        if chunk_hex:
            try:
                buf["chunks"].append(bytes.fromhex(chunk_hex))
            except ValueError:
                return message.create_error("Invalid hex chunk data")
            buf["last_chunk"] = now

        # Decide whether to flush
        elapsed = now - buf["start"]
        silence = now - buf["last_chunk"]
        should_flush = (
            is_final
            or elapsed >= STREAM_MAX_BUFFER_SECONDS
            or (silence >= STREAM_SILENCE_TIMEOUT and buf["chunks"])
        )

        if should_flush and buf["chunks"]:
            await self._flush_stream_buffer(session_id, message)

        return None

    async def _flush_stream_buffer(self, session_id: str, message: Message) -> None:
        """Concatenate buffered chunks, write to temp file, transcribe, and forward."""
        buf = self._stream_buffers.pop(session_id, None)
        if not buf or not buf["chunks"]:
            return

        try:
            import asyncio
            import os

            combined = b"".join(buf["chunks"])
            sample_rate = int(buf.get("sample_rate", 16000))

            def _transcribe_bytes() -> str:
                # Write raw PCM to a temp wav file
                import wave

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                try:
                    with wave.open(tmp.name, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sample_rate)
                        wf.writeframes(combined)

                    self._load_model()
                    if self.model is None:
                        raise RuntimeError("Whisper model not loaded")
                    result = self.model.transcribe(tmp.name, fp16=False)
                    return str(result["text"]).strip()
                finally:
                    os.unlink(tmp.name)

            transcription = await asyncio.to_thread(_transcribe_bytes)

            if transcription:
                logger.info("Stream transcribed (%s): '%s'", session_id, transcription[:60])

                # Forward to router
                query_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    tenant_id=message.tenant_id,
                    session_id=session_id,
                    topic="router.query",
                    payload={"text": transcription, "source": "audio_stream"},
                )
                asyncio.create_task(self.bus.publish("router.query", query_msg))

        except Exception as e:
            logger.warning("Stream transcription failed for %s: %s", session_id, e)

    async def handle_workspace_query(self, message: Message) -> Message | None:
        """
        Multi-modal workspace participation: if the query references audio,
        transcribe it and post a speech_perception thought to the blackboard.
        """
        payload = message.payload
        audio_path = str(payload.get("audio_path") or payload.get("file_path") or "")

        if not audio_path:
            return None  # Not an audio-relevant query

        try:
            import asyncio

            def _transcribe() -> str:
                self._load_model()
                if self.model is None:
                    raise RuntimeError("Whisper model not loaded")
                result = self.model.transcribe(audio_path, fp16=False)
                return str(result["text"]).strip()

            transcription = await asyncio.to_thread(_transcribe)

            # Post as a competing thought on the Workspace blackboard
            thought_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": "speech_perception",
                    "confidence": 0.80,
                    "content": f"[Audio Transcription] {transcription}",
                    "modality": "audio",
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", thought_msg)
        except Exception as e:
            logger.warning("AudioInputNode workspace thought failed: %s", e)

        return None
