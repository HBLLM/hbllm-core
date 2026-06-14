"""
Audio Input Node (Speech-to-Text).

Real-time speech recognition using Moonshine ASR with Silero VAD for
voice activity detection. Supports both file-based and streaming audio.

Primary backend: Moonshine (ONNX, ~50MB, <100ms latency)
Fallback backends: Whisper (local), NVIDIA Cloud ASR

.. deprecated::
    Whisper backend is kept for backward compatibility but will be removed
    in a future release. Prefer Moonshine for real-time applications.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import tempfile
import time
from collections import deque
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType
from hbllm.perception.voice_config import ASRBackend, AudioPipelineConfig

logger = logging.getLogger(__name__)


class AudioInputNode(Node):
    """
    Service node that acts as the model's "ears".

    Supports real-time streaming ASR with voice activity detection (VAD).
    Audio chunks arrive via ``sensory.audio.stream``, are buffered until
    the VAD detects an utterance boundary, then transcribed and forwarded
    to the router.
    """

    def __init__(
        self,
        node_id: str,
        config: AudioPipelineConfig | None = None,
        model_size: str = "base",
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["speech_to_text", "audio_streaming", "vad"],
        )
        self.config = config or AudioPipelineConfig()
        self.model_size = model_size or self.config.asr_model_size

        # ASR model (lazy-loaded)
        self._moonshine_model: Any | None = None
        self._whisper_model: Any | None = None

        # VAD model (lazy-loaded)
        self._vad_model: Any | None = None

        # Streaming state: session_id → buffer
        self._stream_buffers: dict[str, _StreamBuffer] = {}

    # ── Model loading ────────────────────────────────────────────────────

    def _load_moonshine(self) -> None:
        """Load the Moonshine ONNX ASR model."""
        if self._moonshine_model is not None:
            return
        try:
            from moonshine_onnx import MoonshineOnnxModel  # type: ignore[import-untyped]

            logger.info("Loading Moonshine %s ASR model...", self.model_size)
            self._moonshine_model = MoonshineOnnxModel(model_name=self.model_size)
            logger.info("Moonshine ASR loaded successfully")
        except ImportError:
            logger.warning("moonshine-onnx not installed. Install with: pip install moonshine-onnx")
        except Exception as e:
            logger.error("Failed to load Moonshine model: %s", e)

    def _load_whisper(self) -> None:
        """Load the Whisper model (deprecated fallback)."""
        if self._whisper_model is not None:
            return
        try:
            import whisper  # type: ignore[import-untyped]

            logger.info("Loading Whisper %s model (deprecated fallback)...", self.model_size)
            self._whisper_model = whisper.load_model(self.model_size)
        except ImportError:
            logger.error("Whisper library not found. Audio transcription will fail.")

    def _load_vad(self) -> None:
        """Load the Silero VAD model for voice activity detection."""
        if self._vad_model is not None:
            return
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            self._vad_model = model
            logger.info("Silero VAD loaded successfully")
        except ImportError:
            logger.warning("Silero VAD requires torch. VAD will use fallback silence detection.")
        except Exception as e:
            logger.warning("Failed to load Silero VAD: %s. Using fallback.", e)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def on_start(self) -> None:
        """Subscribe to sensory audio streams."""
        logger.info(
            "Starting AudioInputNode (backend=%s, streaming=enabled)",
            self.config.asr_backend.value,
        )
        await self.bus.subscribe("sensory.audio.in", self.handle_transcribe)
        await self.bus.subscribe("sensory.audio.stream", self.handle_stream)
        await self.bus.subscribe("module.evaluate", self.handle_workspace_query)

    async def on_stop(self) -> None:
        logger.info("Stopping AudioInputNode")
        self._stream_buffers.clear()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── File-based transcription ─────────────────────────────────────────

    async def handle_transcribe(self, message: Message) -> Message | None:
        """
        Handle ``sensory.audio.in`` messages.

        Payload expects:
            file_path: str → path to wav/mp3 file
        """
        payload = message.payload
        file_path = payload.get("file_path")

        if not file_path:
            return message.create_error("Missing 'file_path'")

        try:
            transcription = await self._transcribe_file(file_path)
            logger.info("Transcribed text: '%s'", transcription)

            resp = message.create_response({"text": transcription})

            # Forward to the brain
            query_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="router.query",
                payload={"text": transcription, "source": "audio"},
                correlation_id=message.id,
            )
            asyncio.create_task(self.bus.publish("router.query", query_msg))

            return resp

        except Exception as e:
            logger.error("Audio transcription failed: %s", e)
            return message.create_error(str(e))

    # ── Streaming transcription with VAD ─────────────────────────────────

    async def handle_stream(self, message: Message) -> Message | None:
        """
        Handle streaming audio chunks with VAD-based utterance detection.

        Payload expects:
            chunk: str → hex-encoded 16-bit PCM audio bytes
            sample_rate: int → sample rate (default 16000)
            is_final: bool → if True, flush buffer immediately
        """
        payload = message.payload
        session_id = message.session_id or "default"
        chunk_hex = str(payload.get("chunk", ""))
        is_final = bool(payload.get("is_final", False))

        if not chunk_hex and not is_final:
            return None

        # Initialize buffer for this session
        if session_id not in self._stream_buffers:
            self._stream_buffers[session_id] = _StreamBuffer(
                sample_rate=payload.get("sample_rate", self.config.stream_sample_rate),
                vad_threshold=self.config.vad_threshold,
                max_silence_ms=self.config.vad_max_silence_ms,
                max_buffer_seconds=self.config.stream_max_buffer_seconds,
            )

        buf = self._stream_buffers[session_id]

        # Append chunk
        if chunk_hex:
            try:
                buf.append(bytes.fromhex(chunk_hex))
            except ValueError:
                return message.create_error("Invalid hex chunk data")

        # Check VAD for speech boundaries
        should_flush = is_final or buf.should_flush(self._vad_model)

        if should_flush and buf.has_audio:
            await self._flush_stream(session_id, message)

        return None

    async def _flush_stream(self, session_id: str, message: Message) -> None:
        """Transcribe buffered audio and forward to router."""
        buf = self._stream_buffers.pop(session_id, None)
        if not buf or not buf.has_audio:
            return

        try:
            pcm_bytes = buf.get_audio_bytes()
            sample_rate = buf.sample_rate

            transcription = await self._transcribe_pcm(pcm_bytes, sample_rate)

            if transcription:
                logger.info("Stream transcribed (%s): '%s'", session_id, transcription[:80])

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

    # ── Workspace query (multi-modal) ────────────────────────────────────

    async def handle_workspace_query(self, message: Message) -> Message | None:
        """Multi-modal workspace participation for audio queries."""
        payload = message.payload
        audio_path = str(payload.get("audio_path") or payload.get("file_path") or "")

        if not audio_path:
            return None

        try:
            transcription = await self._transcribe_file(audio_path)

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

    # ── Transcription backends ───────────────────────────────────────────

    async def _transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio file using the configured backend."""
        import os

        # Try NVIDIA Cloud first if configured
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_api_key:
            result = await self._transcribe_nvidia(file_path, nvidia_api_key)
            if result:
                return result

        # Use configured backend
        backend = self.config.asr_backend

        if backend == ASRBackend.MOONSHINE:
            return await self._transcribe_file_moonshine(file_path)
        elif backend == ASRBackend.WHISPER:
            return await self._transcribe_file_whisper(file_path)
        else:
            # Auto-detect: try Moonshine first, fall back to Whisper
            try:
                return await self._transcribe_file_moonshine(file_path)
            except (ImportError, RuntimeError):
                return await self._transcribe_file_whisper(file_path)

    async def _transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM audio bytes (16-bit, mono)."""
        backend = self.config.asr_backend

        if backend == ASRBackend.MOONSHINE:
            return await self._transcribe_pcm_moonshine(pcm_bytes, sample_rate)

        # Whisper requires a file — write to temp
        return await self._transcribe_pcm_via_file(pcm_bytes, sample_rate)

    async def _transcribe_pcm_moonshine(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM directly with Moonshine (no temp file)."""
        import numpy as np

        def _transcribe() -> str:
            self._load_moonshine()
            if self._moonshine_model is None:
                raise RuntimeError("Moonshine model not loaded")

            # Convert 16-bit PCM to float32 numpy array
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Moonshine expects (samples,) at 16kHz
            if sample_rate != 16000:
                # Simple resampling via linear interpolation
                import scipy.signal  # type: ignore[import-untyped]

                samples = scipy.signal.resample(samples, int(len(samples) * 16000 / sample_rate))

            tokens = self._moonshine_model.generate(samples)
            return str(tokens[0]).strip() if tokens else ""

        return await asyncio.to_thread(_transcribe)

    async def _transcribe_file_moonshine(self, file_path: str) -> str:
        """Transcribe an audio file with Moonshine."""
        import numpy as np

        def _transcribe() -> str:
            self._load_moonshine()
            if self._moonshine_model is None:
                raise RuntimeError("Moonshine model not loaded")

            import soundfile as sf  # type: ignore[import-not-found]

            audio, sr = sf.read(file_path, dtype="float32")

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample to 16kHz if needed
            if sr != 16000:
                import scipy.signal  # type: ignore[import-untyped]

                audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))

            tokens = self._moonshine_model.generate(audio)
            return str(tokens[0]).strip() if tokens else ""

        return await asyncio.to_thread(_transcribe)

    async def _transcribe_file_whisper(self, file_path: str) -> str:
        """Transcribe with local Whisper (deprecated fallback).

        .. deprecated::
            Will be removed in a future release. Use Moonshine instead.
        """

        def _transcribe() -> str:
            self._load_whisper()
            if self._whisper_model is None:
                raise RuntimeError("Whisper model not loaded")
            logger.info("Transcribing audio file locally with Whisper: %s", file_path)
            result = self._whisper_model.transcribe(file_path, fp16=False)
            return str(result["text"]).strip()

        return await asyncio.to_thread(_transcribe)

    async def _transcribe_pcm_via_file(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """Write PCM to temp WAV file and transcribe (for Whisper fallback)."""
        import os
        import wave

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)

            return await self._transcribe_file(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    async def _transcribe_nvidia(self, file_path: str, api_key: str) -> str | None:
        """Transcribe via NVIDIA Cloud Whisper API."""
        import os

        import httpx

        nvidia_asr_url = os.getenv(
            "NVIDIA_ASR_URL", "https://integrate.api.nvidia.com/v1/audio/transcriptions"
        )
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                with open(file_path, "rb") as f:
                    files = {"file": (os.path.basename(file_path), f, "audio/wav")}
                    data = {"model": "openai/whisper-large-v3"}
                    resp = await client.post(
                        nvidia_asr_url, headers=headers, files=files, data=data
                    )
                    resp.raise_for_status()
                    return str(resp.json().get("text", "")).strip()
        except Exception as e:
            logger.warning("NVIDIA Cloud ASR failed: %s", e)
            return None


class _StreamBuffer:
    """Audio stream buffer with VAD-aware flushing."""

    def __init__(
        self,
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        max_silence_ms: int = 700,
        max_buffer_seconds: float = 15.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.max_silence_ms = max_silence_ms
        self.max_buffer_seconds = max_buffer_seconds

        self.chunks: list[bytes] = []
        self.start_time = time.monotonic()
        self.last_chunk_time = time.monotonic()
        self._speech_detected = False
        self._silence_start: float | None = None

    @property
    def has_audio(self) -> bool:
        return len(self.chunks) > 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def append(self, chunk: bytes) -> None:
        self.chunks.append(chunk)
        self.last_chunk_time = time.monotonic()

    def get_audio_bytes(self) -> bytes:
        return b"".join(self.chunks)

    def should_flush(self, vad_model: Any | None) -> bool:
        """Determine if the buffer should be flushed based on VAD or timeouts."""
        if not self.has_audio:
            return False

        # Hard limit: max buffer duration
        if self.elapsed >= self.max_buffer_seconds:
            return True

        # If we have VAD, use it
        if vad_model is not None and self.chunks:
            return self._check_vad(vad_model)

        # Fallback: simple silence timeout
        silence = time.monotonic() - self.last_chunk_time
        return silence >= (self.max_silence_ms / 1000.0) and self.has_audio

    def _check_vad(self, vad_model: Any) -> bool:
        """Use Silero VAD to detect speech boundaries."""
        try:
            import torch

            # Get the last chunk for VAD analysis
            last_chunk = self.chunks[-1]
            # Convert 16-bit PCM to float tensor
            samples = torch.frombuffer(last_chunk, dtype=torch.int16).float() / 32768.0

            # Silero VAD expects 16kHz mono
            speech_prob = vad_model(samples, self.sample_rate).item()

            if speech_prob >= self.vad_threshold:
                self._speech_detected = True
                self._silence_start = None
                return False

            # Speech was detected before, now silence
            if self._speech_detected:
                if self._silence_start is None:
                    self._silence_start = time.monotonic()
                silence_ms = (time.monotonic() - self._silence_start) * 1000
                if silence_ms >= self.max_silence_ms:
                    return True  # End of utterance

        except Exception as e:
            logger.debug("VAD check failed: %s", e)

        return False
