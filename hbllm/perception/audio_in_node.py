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
import tempfile
import threading
import time
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
        model_size: str | None = None,
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
        self._model_lock = threading.Lock()  # Prevent concurrent model loads

        # VAD model (lazy-loaded)
        self._vad_model: Any | None = None

        # Streaming state: session_id → buffer
        self._stream_buffers: dict[str, _StreamBuffer] = {}

    # ── Model loading ────────────────────────────────────────────────────

    def _load_moonshine(self) -> None:
        """Load the Moonshine ONNX ASR model (thread-safe)."""
        if self._moonshine_model is not None:
            return
        with self._model_lock:
            # Double-check after acquiring lock
            if self._moonshine_model is not None:
                return
            try:
                from moonshine_onnx import MoonshineOnnxModel  # type: ignore[import-untyped]

                logger.info("Loading Moonshine %s ASR model...", self.model_size)
                self._moonshine_model = MoonshineOnnxModel(model_name=self.model_size)
                logger.info("Moonshine ASR loaded successfully")
            except ImportError:
                logger.warning(
                    "moonshine-onnx not installed. Install with: pip install useful-moonshine-onnx"
                )
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
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]

            model = load_silero_vad(onnx=True)
            self._vad_model = model
            logger.info("Silero VAD loaded successfully (via silero-vad package)")
        except ImportError:
            logger.warning("silero-vad package not installed. VAD will use energy-based fallback.")
        except Exception as e:
            logger.warning("Failed to load Silero VAD: %s. Using energy-based fallback.", e)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def on_start(self) -> None:
        """Subscribe to sensory audio streams."""
        logger.info(
            "Starting AudioInputNode (backend=%s, streaming=enabled)",
            self.config.asr_backend.value,
        )
        # Pre-load ASR model on startup so it's cached before first audio
        if self.config.asr_backend == ASRBackend.MOONSHINE:
            await asyncio.to_thread(self._load_moonshine)

        # Pre-load VAD model
        await asyncio.to_thread(self._load_vad)

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
            # If the caller explicitly marked is_final, treat as speech
            # (the client confirmed end-of-utterance)
            if is_final:
                buf._speech_detected = True
            await self._flush_stream(session_id, message, is_final=is_final)

        return None

    async def _flush_stream(
        self, session_id: str, message: Message, *, is_final: bool = False
    ) -> None:
        """Transcribe buffered audio and forward to router."""
        buf = self._stream_buffers.pop(session_id, None)
        if not buf or not buf.has_audio:
            return

        try:
            pcm_bytes = buf.get_audio_bytes()
            sample_rate = buf.sample_rate

            # Debug: log audio buffer stats
            duration_s = len(pcm_bytes) / (2 * sample_rate)
            logger.info(
                "Flushing audio buffer (%s): %d bytes, %.2fs, %d chunks, sr=%d, speech=%s",
                session_id[:8],
                len(pcm_bytes),
                duration_s,
                len(buf.chunks),
                sample_rate,
                buf._speech_detected,
            )

            # Skip if no speech was detected (silence-only buffer from always-on mic)
            if not buf._speech_detected:
                logger.info("Skipping silence-only buffer (no speech detected)")
                return

            # Skip very short audio (< 0.5s) unless the client explicitly
            # marked is_final (they confirmed the utterance is complete)
            if duration_s < 0.5 and not is_final:
                logger.info("Skipping too-short audio (%.2fs < 0.5s)", duration_s)
                return

            import time as _time

            t0 = _time.monotonic()
            transcription = await self._transcribe_pcm(pcm_bytes, sample_rate)
            elapsed = _time.monotonic() - t0
            logger.info(
                "Transcription result (%s): '%s' (took %.1fs)",
                session_id[:8],
                transcription[:80] if transcription else "<empty>",
                elapsed,
            )

            if transcription:
                logger.info("Stream transcribed (%s): '%s'", session_id, transcription[:80])

                # Identify the speaker from the same audio
                speaker_id = "unknown"
                speaker_name = ""
                speaker_confidence = 0.0
                try:
                    speaker_msg = Message(
                        type=MessageType.QUERY,
                        source_node_id=self.node_id,
                        tenant_id=message.tenant_id,
                        session_id=session_id,
                        topic="speaker.identify",
                        payload={
                            "pcm_bytes": pcm_bytes.hex(),
                            "sample_rate": sample_rate,
                        },
                    )
                    speaker_result = await asyncio.wait_for(
                        self.bus.request("speaker.identify", speaker_msg, timeout=3.0),
                        timeout=3.0,
                    )
                    speaker_id = speaker_result.payload.get("speaker_id", "unknown")
                    speaker_name = speaker_result.payload.get("speaker_name", "")
                    speaker_confidence = speaker_result.payload.get("confidence", 0.0)
                    if speaker_id != "unknown":
                        logger.info(
                            "Speaker identified: %s (%s) confidence=%.2f",
                            speaker_id,
                            speaker_name,
                            speaker_confidence,
                        )
                except (TimeoutError, asyncio.TimeoutError):
                    logger.debug("Speaker identification timed out, continuing as unknown")
                except Exception as e:
                    logger.debug("Speaker identification unavailable: %s", e)

                # Publish transcription event for the WebSocket to forward to browser
                transcription_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=message.tenant_id,
                    session_id=session_id,
                    topic="sensory.transcription",
                    payload={
                        "text": transcription,
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "speaker_confidence": speaker_confidence,
                    },
                )
                await self.bus.publish("sensory.transcription", transcription_msg)

                # Forward to the brain for response
                query_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    tenant_id=message.tenant_id,
                    session_id=session_id,
                    topic="router.query",
                    payload={
                        "text": transcription,
                        "source": "audio_stream",
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "speaker_confidence": speaker_confidence,
                    },
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
        """Transcribe raw PCM audio bytes (16-bit, mono). Moonshine primary."""
        # Try Moonshine first (local, fast)
        result = await self._transcribe_pcm_moonshine(pcm_bytes, sample_rate)
        if result:
            return result

        # Fall back to NVIDIA Cloud ASR if Moonshine returns empty
        import os

        nvidia_api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
        if nvidia_api_key:
            try:
                logger.info("Moonshine returned empty, trying NVIDIA Cloud ASR...")
                cloud_result = await self._transcribe_pcm_via_file(pcm_bytes, sample_rate)
                if cloud_result:
                    return cloud_result
            except Exception as e:
                logger.warning("NVIDIA Cloud ASR fallback failed: %s", e)

        return result  # Return empty string

    async def _transcribe_pcm_moonshine(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM directly with Moonshine (no temp file)."""
        import numpy as np

        def _transcribe() -> str:
            self._load_moonshine()
            if self._moonshine_model is None:
                raise RuntimeError("Moonshine model not loaded")

            # Convert 16-bit PCM to float32 numpy array
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                import scipy.signal  # type: ignore[import-untyped]

                samples = scipy.signal.resample(samples, int(len(samples) * 16000 / sample_rate))

            # Skip very short audio (< 0.1s at 16kHz)
            if len(samples) < 1600:
                return ""

            # Log audio stats for debugging
            rms = float(np.sqrt(np.mean(samples**2)))
            peak = float(np.max(np.abs(samples)))
            logger.info(
                "Audio stats (raw): rms=%.4f, peak=%.4f, len=%d (%.2fs)",
                rms,
                peak,
                len(samples),
                len(samples) / 16000,
            )

            # Trim leading/trailing silence (below 2% amplitude)
            threshold = 0.02
            above = np.where(np.abs(samples) > threshold)[0]
            if len(above) > 0:
                # Keep 0.1s padding on each side
                pad = int(0.1 * 16000)
                start = max(0, above[0] - pad)
                end = min(len(samples), above[-1] + pad)
                samples = samples[start:end]
                logger.info(
                    "Trimmed silence: %d→%d samples (%.2fs→%.2fs)",
                    len(above),
                    len(samples),
                    len(above) / 16000,
                    len(samples) / 16000,
                )

            # Skip if trimmed too short
            if len(samples) < 1600:
                logger.info("Skipping: too short after trim (%d samples)", len(samples))
                return ""

            # Normalize audio to peak=0.95 for consistent Moonshine input levels
            peak = float(np.max(np.abs(samples)))
            if peak > 0.001:
                samples = samples * (0.95 / peak)
                logger.info(
                    "Normalized: peak %.4f → 0.95, new rms=%.4f",
                    peak,
                    float(np.sqrt(np.mean(samples**2))),
                )

            # Moonshine expects (1, samples) float32
            samples_2d = samples.reshape(1, -1)

            token_ids = self._moonshine_model.generate(samples_2d)
            logger.info("Moonshine token_ids: %s", token_ids)

            from moonshine_onnx import load_tokenizer  # type: ignore[import-untyped]

            tokenizer = load_tokenizer()
            texts = tokenizer.decode_batch(token_ids)
            text = texts[0].strip() if texts else ""
            logger.info("Moonshine transcribed: '%s'", text if text else "<empty>")
            return text

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

            # Moonshine ONNX expects (1, samples) — reshape from 1D
            audio_2d = audio.reshape(1, -1).astype(np.float32)
            token_ids = self._moonshine_model.generate(audio_2d)

            # Decode with tokenizer (not str()!)
            from moonshine_onnx import load_tokenizer  # type: ignore[import-untyped]

            tokenizer = load_tokenizer()
            texts = tokenizer.decode_batch(token_ids)
            return texts[0].strip() if texts else ""

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
            except Exception as e:
                logger.debug("[AudioInNode] non-critical error: %s", e)

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

        # Absolute hard limit — never allow buffer to exceed 30s no matter what
        if self.elapsed >= 30.0:
            logger.warning(
                "Buffer exceeded absolute 30s limit (%.1fs). Force flushing.", self.elapsed
            )
            return True

        # Normal max buffer duration
        if self.elapsed >= self.max_buffer_seconds:
            if self._speech_detected:
                return True
            # max_buffer_seconds == 0 means "always flush immediately"
            if self.max_buffer_seconds == 0:
                return True
            # No speech in max window — silently discard
            self.chunks.clear()
            self.start_time = time.monotonic()
            self._speech_detected = False
            self._silence_start = None
            return False

        # If we have VAD, use it
        if vad_model is not None and self.chunks:
            return self._check_vad(vad_model)

        # Fallback: energy-based speech detection (no Silero VAD)
        # Check if the latest chunk has enough energy to be speech
        if self.chunks:
            import numpy as np

            last_chunk = self.chunks[-1]
            if len(last_chunk) >= 2:
                samples = np.frombuffer(last_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(samples**2)))
                if rms > 0.01:  # Above noise floor
                    if not self._speech_detected and len(self.chunks) > 2:
                        # Discard pre-speech silence (same as Silero VAD path)
                        self.chunks = self.chunks[-2:]
                        self.start_time = time.monotonic()
                    self._speech_detected = True
                    self._silence_start = None
                elif self._speech_detected:
                    # Speech was detected before, now silence
                    if self._silence_start is None:
                        self._silence_start = time.monotonic()
                    silence_ms = (time.monotonic() - self._silence_start) * 1000
                    if silence_ms >= self.max_silence_ms:
                        return True  # End of utterance

        # Max silence timeout (0 = flush immediately after any audio)
        if self.max_silence_ms == 0 and self.has_audio:
            return True

        # Max buffer hit but no speech → don't flush
        if not self._speech_detected:
            return False

        silence = time.monotonic() - self.last_chunk_time
        return silence >= (self.max_silence_ms / 1000.0) and self.has_audio

    def _check_vad(self, vad_model: Any) -> bool:
        """Use Silero VAD to detect speech boundaries."""
        try:
            import numpy as np
            import torch

            # Get the last chunk for VAD analysis
            last_chunk = self.chunks[-1]
            if len(last_chunk) < 2:
                return False

            # Convert 16-bit PCM to float32 numpy first, then to torch tensor
            np_samples = np.frombuffer(last_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            samples = torch.from_numpy(np_samples)

            # Silero VAD expects specific chunk sizes: 512 samples for 16kHz
            # If our chunk is larger, take the last 512 samples
            expected_size = 512
            if len(samples) > expected_size:
                samples = samples[-expected_size:]
            elif len(samples) < expected_size:
                # Pad with zeros if too short
                samples = torch.nn.functional.pad(samples, (0, expected_size - len(samples)))

            # Silero VAD expects 16kHz mono
            speech_prob = vad_model(samples, self.sample_rate).item()

            # Log every Nth check to avoid spam (log every 4th = ~1 per second)
            if len(self.chunks) % 4 == 0:
                logger.info(
                    "VAD prob=%.3f, threshold=%.2f, speech_detected=%s, chunks=%d",
                    speech_prob,
                    self.vad_threshold,
                    self._speech_detected,
                    len(self.chunks),
                )

            if speech_prob >= self.vad_threshold:
                if not self._speech_detected:
                    logger.info(
                        "VAD: Speech onset detected (prob=%.3f), discarding %d pre-speech chunks",
                        speech_prob,
                        max(0, len(self.chunks) - 2),
                    )
                    # Discard all pre-speech silence chunks, keeping only
                    # the last 2 chunks as lead-in context (~0.5s).
                    # Without this, Moonshine receives 10+ seconds of noise
                    # before the actual speech, causing empty transcription.
                    if len(self.chunks) > 2:
                        self.chunks = self.chunks[-2:]
                        self.start_time = time.monotonic()
                self._speech_detected = True
                self._silence_start = None
                return False

            # Speech was detected before, now silence
            if self._speech_detected:
                if self._silence_start is None:
                    self._silence_start = time.monotonic()
                    logger.debug("VAD: Silence started after speech (prob=%.3f)", speech_prob)
                silence_ms = (time.monotonic() - self._silence_start) * 1000
                if silence_ms >= self.max_silence_ms:
                    logger.debug("VAD: End of utterance (%.0fms silence)", silence_ms)
                    return True  # End of utterance

        except Exception as e:
            logger.debug("VAD check failed: %s", e)

        return False
