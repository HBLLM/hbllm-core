"""
Audio Output Node (Text-to-Speech).

Real-time speech synthesis using Kokoro TTS with sentence-level streaming.
Supports per-tenant voice customization via the VoiceRegistry.

Primary backend: Kokoro TTS (~300MB, <100ms first-chunk latency)
Premium backend: Orpheus TTS (Llama-based, emotional expressiveness)
Fallback backends: SpeechT5 (deprecated), NVIDIA Riva

.. deprecated::
    SpeechT5 backend is kept for backward compatibility but will be
    removed in a future release. Prefer Kokoro for real-time applications.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType
from hbllm.perception.voice_config import (
    AudioPipelineConfig,
    TTSBackend,
    VoiceConfig,
    VoiceRegistry,
)

logger = logging.getLogger(__name__)


class AudioOutputNode(Node):
    """
    Service node that acts as the model's "voice".

    Supports sentence-level streaming: begins audio output as soon as the
    first sentence of a response is generated, rather than waiting for
    the full text.
    """

    def __init__(
        self,
        node_id: str,
        output_dir: str = "workspace/audio",
        config: AudioPipelineConfig | None = None,
        data_dir: str | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["text_to_speech", "voice_customization", "sentence_streaming"],
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or AudioPipelineConfig()

        # Voice registry for per-tenant preferences
        db_dir = Path(data_dir) if data_dir else self.output_dir
        self._voice_registry = VoiceRegistry(db_dir / "voice_preferences.db")

        # TTS engines (lazy-loaded)
        self._kokoro_pipeline: Any | None = None
        self._orpheus_engine: Any | None = None
        self._speecht5_model: Any | None = None
        self._speecht5_processor: Any | None = None
        self._speecht5_vocoder: Any | None = None
        self._default_speaker_embeddings: Any | None = None

    # ── Model loading ────────────────────────────────────────────────────

    def _load_kokoro(self, voice_config: VoiceConfig | None = None) -> None:
        """Load the Kokoro TTS pipeline."""
        if self._kokoro_pipeline is not None:
            return
        try:
            from kokoro import KPipeline  # type: ignore[import-untyped]

            lang = (voice_config or self.config.default_voice).language[:2]
            lang_code = "a" if lang == "en" else lang  # Kokoro uses 'a' for American English
            self._kokoro_pipeline = KPipeline(lang_code=lang_code)
            logger.info("Kokoro TTS pipeline loaded (lang=%s)", lang_code)
        except ImportError:
            logger.warning("kokoro not installed. Install with: pip install kokoro")
        except Exception as e:
            logger.error("Failed to load Kokoro TTS: %s", e)

    def _load_orpheus(self) -> None:
        """Load the Orpheus TTS engine (premium, Llama-based)."""
        if self._orpheus_engine is not None:
            return
        try:
            from orpheus_speech import OrpheusModel  # type: ignore[import-untyped]

            self._orpheus_engine = OrpheusModel()
            logger.info("Orpheus TTS engine loaded (premium)")
        except ImportError:
            logger.warning("orpheus-speech not installed. Install with: pip install orpheus-speech")
        except Exception as e:
            logger.error("Failed to load Orpheus TTS: %s", e)

    def _load_speecht5(self) -> None:
        """Load the SpeechT5 model (deprecated fallback).

        .. deprecated::
            Will be removed in a future release. Use Kokoro instead.
        """
        if self._speecht5_model is not None:
            return
        logger.info("Loading SpeechT5 TTS models (deprecated fallback)...")
        import torch
        from datasets import load_dataset  # type: ignore[import-untyped]
        from transformers import (  # type: ignore[attr-defined]
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        self._speecht5_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self._speecht5_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self._speecht5_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self._default_speaker_embeddings = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0)

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def on_start(self) -> None:
        """Subscribe to TTS and voice config topics."""
        logger.info("Starting AudioOutputNode (backend=%s)", self.config.tts_backend.value)
        await self.bus.subscribe("sensory.audio.out", self.handle_synthesize)
        await self.bus.subscribe("sensory.audio.out.stream", self.handle_stream_synthesize)
        await self.bus.subscribe("voice.config", self._handle_voice_config)
        await self.bus.subscribe("voice.list", self._handle_list_voices)

    async def on_stop(self) -> None:
        logger.info("Stopping AudioOutputNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Full-text synthesis ──────────────────────────────────────────────

    async def handle_synthesize(self, message: Message) -> Message | None:
        """
        Handle ``sensory.audio.out`` messages.

        Payload expects:
            text: str → The response text to vocalize
            voice_id: str → Optional voice override
            stream: bool → If True, emit sentence-level audio chunks
        """
        logger.info(
            "[AudioOut] handle_synthesize called with text=%s", message.payload.get("text", "")[:50]
        )
        payload = message.payload
        text = payload.get("text")
        tenant_id = message.tenant_id or "default"

        if not text:
            return message.create_error("Missing 'text' in payload")

        text = self._clean_text(text)
        voice = self._resolve_voice(tenant_id, payload)
        should_stream = bool(payload.get("stream", False))

        if should_stream:
            # Sentence-level streaming
            await self._synthesize_streaming(text, voice, message)
            return message.create_response({"status": "streamed", "voice": voice.voice_id})

        # Full synthesis
        filename = f"response_{message.id[:8]}.wav"
        audio_path = await self._synthesize_full(text, voice, filename)

        if audio_path:
            return message.create_response({"audio_path": audio_path, "voice": voice.voice_id})
        return message.create_error("All TTS backends failed")

    async def handle_stream_synthesize(self, message: Message) -> Message | None:
        """Handle incremental text streaming for real-time TTS."""
        payload = message.payload
        text_chunk = payload.get("text", "")
        is_final = bool(payload.get("is_final", False))
        tenant_id = message.tenant_id or "default"

        if not text_chunk and not is_final:
            return None

        if text_chunk:
            text_chunk = self._clean_text(text_chunk)

        voice = self._resolve_voice(tenant_id, payload)

        if text_chunk:
            await self._synthesize_streaming(text_chunk, voice, message)

        return None

    # ── Synthesis backends ───────────────────────────────────────────────

    async def _synthesize_full(self, text: str, voice: VoiceConfig, filename: str) -> str | None:
        """Synthesize full text to a WAV file."""
        # Try NVIDIA first
        result = await self._synthesize_nvidia(text, filename)
        if result:
            return result

        backend = voice.backend

        if backend == TTSBackend.KOKORO:
            return await self._synthesize_kokoro_full(text, voice, filename)
        elif backend == TTSBackend.ORPHEUS:
            return await self._synthesize_orpheus_full(text, voice, filename)
        elif backend == TTSBackend.SPEECHT5:
            return await self._synthesize_speecht5_full(text, voice, filename)

        # Auto-detect: try Kokoro → Orpheus → SpeechT5
        for method in [
            self._synthesize_kokoro_full,
            self._synthesize_orpheus_full,
            self._synthesize_speecht5_full,
        ]:
            try:
                result = await method(text, voice, filename)
                if result:
                    return result
            except (ImportError, RuntimeError) as e:
                logger.debug("Backend failed: %s", e)
                continue

        return None

    async def _synthesize_streaming(self, text: str, voice: VoiceConfig, message: Message) -> None:
        """Synthesize text sentence-by-sentence, emitting audio chunks."""
        sentences = self._split_sentences(text)

        for i, sentence in enumerate(sentences):
            is_final = i == len(sentences) - 1
            try:
                audio_bytes, sample_rate = await self._synthesize_sentence(sentence, voice)
                if audio_bytes:
                    chunk_msg = Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        tenant_id=message.tenant_id,
                        device_id=message.device_id,
                        session_id=message.session_id,
                        topic="sensory.audio.chunk",
                        payload={
                            "audio": audio_bytes.hex(),
                            "sample_rate": sample_rate,
                            "is_final": is_final,
                            "sentence_index": i,
                            "text": sentence,
                        },
                        correlation_id=message.correlation_id,
                    )
                    await self.bus.publish("sensory.audio.chunk", chunk_msg)
            except Exception as e:
                logger.warning("Sentence synthesis failed: %s", e)

    async def _synthesize_sentence(self, sentence: str, voice: VoiceConfig) -> tuple[bytes, int]:
        """Synthesize a single sentence to PCM bytes."""
        if voice.backend == TTSBackend.KOKORO:
            return await self._kokoro_sentence(sentence, voice)
        elif voice.backend == TTSBackend.ORPHEUS:
            return await self._orpheus_sentence(sentence, voice)
        else:
            # Try Kokoro first
            try:
                return await self._kokoro_sentence(sentence, voice)
            except (ImportError, RuntimeError):
                return await self._orpheus_sentence(sentence, voice)

    # ── Kokoro TTS ───────────────────────────────────────────────────────

    async def _kokoro_sentence(self, text: str, voice: VoiceConfig) -> tuple[bytes, int]:
        """Synthesize a single sentence with Kokoro."""
        import numpy as np

        def _synth() -> tuple[bytes, int]:
            self._load_kokoro(voice)
            if self._kokoro_pipeline is None:
                raise RuntimeError("Kokoro TTS not loaded")

            # Generate audio for this sentence
            audio_segments = []
            for _, _, audio in self._kokoro_pipeline(text, voice=voice.voice_id, speed=voice.speed):
                if audio is not None:
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu()
                    if hasattr(audio, "numpy"):
                        audio = audio.numpy()
                    audio_segments.append(audio)

            if not audio_segments:
                return b"", 24000

            combined = (
                np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
            )
            # Convert float32 to 16-bit PCM
            pcm = (combined * 32767).astype(np.int16).tobytes()
            return pcm, 24000

        return await asyncio.to_thread(_synth)

    async def _synthesize_kokoro_full(
        self, text: str, voice: VoiceConfig, filename: str
    ) -> str | None:
        """Synthesize full text with Kokoro and save to file."""
        import numpy as np

        def _synth() -> str | None:
            self._load_kokoro(voice)
            if self._kokoro_pipeline is None:
                return None

            import soundfile as sf  # type: ignore[import-not-found]

            audio_segments = []
            for _, _, audio in self._kokoro_pipeline(text, voice=voice.voice_id, speed=voice.speed):
                if audio is not None:
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu()
                    if hasattr(audio, "numpy"):
                        audio = audio.numpy()
                    audio_segments.append(audio)

            if not audio_segments:
                return None

            combined = (
                np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
            )
            out_path = self.output_dir / filename
            sf.write(str(out_path), combined, samplerate=24000)
            logger.info("Kokoro TTS: saved %s (%.1fs audio)", out_path, len(combined) / 24000)
            return str(out_path)

        return await asyncio.to_thread(_synth)

    # ── Orpheus TTS ──────────────────────────────────────────────────────

    async def _orpheus_sentence(self, text: str, voice: VoiceConfig) -> tuple[bytes, int]:
        """Synthesize a single sentence with Orpheus TTS."""
        import numpy as np

        def _synth() -> tuple[bytes, int]:
            self._load_orpheus()
            if self._orpheus_engine is None:
                raise RuntimeError("Orpheus TTS not loaded")

            # Apply emotional tags if configured
            tagged_text = text
            if voice.orpheus_emotion:
                tagged_text = f"<{voice.orpheus_emotion}>{text}</{voice.orpheus_emotion}>"

            audio = self._orpheus_engine.synthesize(tagged_text)
            if hasattr(audio, "cpu"):
                audio = audio.cpu()
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            pcm = (audio * 32767).astype(np.int16).tobytes()
            return pcm, 24000

        return await asyncio.to_thread(_synth)

    async def _synthesize_orpheus_full(
        self, text: str, voice: VoiceConfig, filename: str
    ) -> str | None:
        """Synthesize full text with Orpheus and save to file."""

        def _synth() -> str | None:
            self._load_orpheus()
            if self._orpheus_engine is None:
                return None

            import soundfile as sf  # type: ignore[import-not-found]

            tagged_text = text
            if voice.orpheus_emotion:
                tagged_text = f"<{voice.orpheus_emotion}>{text}</{voice.orpheus_emotion}>"

            audio = self._orpheus_engine.synthesize(tagged_text)
            if hasattr(audio, "cpu"):
                audio = audio.cpu()
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            out_path = self.output_dir / filename
            sf.write(str(out_path), audio, samplerate=24000)
            logger.info("Orpheus TTS: saved %s", out_path)
            return str(out_path)

        return await asyncio.to_thread(_synth)

    # ── SpeechT5 fallback (deprecated) ───────────────────────────────────

    async def _synthesize_speecht5_full(
        self, text: str, voice: VoiceConfig, filename: str
    ) -> str | None:
        """Synthesize with SpeechT5 (deprecated).

        .. deprecated::
            Will be removed in a future release. Use Kokoro instead.
        """
        import numpy as np

        clean_text = re.sub(r"[^A-Za-z0-9 .,?!'-]", "", text)
        chunks = self._chunk_text(clean_text, max_len=450)

        def _synth() -> str | None:
            import torch

            self._load_speecht5()
            if self._speecht5_model is None or self._speecht5_processor is None:
                return None

            import soundfile as sf  # type: ignore[import-not-found]

            speaker_emb = self._default_speaker_embeddings
            if voice.custom_embedding:
                speaker_emb = torch.tensor(voice.custom_embedding).unsqueeze(0)

            all_speech = []
            for chunk in chunks:
                inputs = self._speecht5_processor(text=chunk, return_tensors="pt")
                with torch.no_grad():
                    speech = self._speecht5_model.generate_speech(
                        inputs["input_ids"], speaker_emb, vocoder=self._speecht5_vocoder
                    )
                all_speech.append(speech.numpy())

            combined = np.concatenate(all_speech) if len(all_speech) > 1 else all_speech[0]
            out_path = self.output_dir / filename
            sf.write(str(out_path), combined, samplerate=16000)
            logger.info("SpeechT5 TTS (deprecated): saved %s", out_path)
            return str(out_path)

        return await asyncio.to_thread(_synth)

    # ── NVIDIA Riva ──────────────────────────────────────────────────────

    async def _synthesize_nvidia(self, text: str, filename: str) -> str | None:
        """Synthesize speech using NVIDIA Riva gRPC client."""
        import os

        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        nvidia_tts_uri = os.getenv("NVIDIA_TTS_URI", "grpc.nvcf.nvidia.com:443")
        nvidia_tts_function_id = os.getenv("NVIDIA_TTS_FUNCTION_ID")
        nvidia_tts_voice_name = os.getenv("NVIDIA_TTS_VOICE_NAME", "Magpie-Multilingual.EN-US.Aria")

        is_cloud = "grpc.nvcf.nvidia.com" in nvidia_tts_uri
        if is_cloud and (not nvidia_api_key or not nvidia_tts_function_id):
            return None

        try:
            import riva.client  # type: ignore
        except ImportError:
            return None

        logger.info("Synthesizing via NVIDIA Riva at %s", nvidia_tts_uri)
        try:
            if is_cloud:
                auth = riva.client.Auth(
                    use_ssl=True,
                    uri=nvidia_tts_uri,
                    metadata_args=[
                        ["function-id", nvidia_tts_function_id],
                        ["authorization", f"Bearer {nvidia_api_key}"],
                    ],
                )
            else:
                auth = riva.client.Auth(use_ssl=False, uri=nvidia_tts_uri)

            tts_client = riva.client.SpeechSynthesisService(auth)

            def _grpc_call():
                return tts_client.synthesize(
                    text=text, voice_name=nvidia_tts_voice_name, language_code="en-US"
                )

            response = await asyncio.to_thread(_grpc_call)

            out_path = self.output_dir / filename
            with open(out_path, "wb") as f:
                f.write(response.audio)

            logger.info("NVIDIA Riva TTS: saved %s", out_path)
            return str(out_path)

        except Exception as e:
            logger.warning("NVIDIA Riva TTS failed: %s", e)
            return None

    # ── Voice management ─────────────────────────────────────────────────

    def _resolve_voice(self, tenant_id: str, payload: dict[str, Any]) -> VoiceConfig:
        """Resolve voice config from payload overrides or tenant preferences."""
        voice = self._voice_registry.get(tenant_id)

        # Allow per-request overrides
        if "voice_id" in payload:
            voice.voice_id = payload["voice_id"]
        if "speed" in payload:
            voice.speed = float(payload["speed"])
        if "backend" in payload:
            try:
                voice.backend = TTSBackend(payload["backend"])
            except ValueError:
                pass
        if "emotion" in payload:
            voice.orpheus_emotion = payload["emotion"]

        return voice

    async def _handle_voice_config(self, message: Message) -> Message | None:
        """Configure a per-tenant voice preference."""
        payload = message.payload
        tenant_id = message.tenant_id or payload.get("tenant_id", "")

        if not tenant_id:
            return message.create_error("Missing tenant_id")

        voice = VoiceConfig(
            voice_id=payload.get("voice_id", "af_heart"),
            speed=float(payload.get("speed", 1.0)),
            backend=TTSBackend(payload.get("backend", "kokoro")),
            language=payload.get("language", "en-us"),
            orpheus_emotion=payload.get("emotion"),
        )

        # Handle custom embeddings
        embedding = payload.get("voice_embedding")
        if embedding and isinstance(embedding, list):
            voice.custom_embedding = embedding

        await asyncio.to_thread(self._voice_registry.set, tenant_id, voice)
        logger.info("Updated voice config for tenant %s: %s", tenant_id, voice.voice_id)
        return message.create_response({"status": "updated", "voice_id": voice.voice_id})

    async def _handle_list_voices(self, message: Message) -> Message | None:
        """List available voices for a backend."""
        backend = TTSBackend(message.payload.get("backend", "kokoro"))
        voices = self._voice_registry.list_voices(backend)
        return message.create_response({"voices": voices, "backend": backend.value})

    # ── Text utilities ───────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for TTS output (remove markdown, URLs, MCTS logs, list bullets)."""
        # Remove MCTS Planner headers
        text = re.sub(
            r"\s*\[MCTS Planner\][\s\S]*?Best path:\s*(?:\[D\d+:\s*Q=\d*(?:\.\d+)?\s*,\s*N=\d+\](?:\s*(?:→|->)\s*)?)+",
            "",
            text,
        )

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", " code block omitted ", text)
        text = re.sub(r"`[^`]+`", "", text)

        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Remove list numbers/bullets at start of sentences
        text = re.sub(r"(?:^|\s)\d+\.\s+", " ", text)
        text = re.sub(r"(?:^|\s)[-*•]\s+", " ", text)

        # Remove markdown formatting
        text = re.sub(r"[*_~]", "", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for streaming TTS."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _chunk_text(text: str, max_len: int = 450) -> list[str]:
        """Split text into sentence-aligned chunks for TTS quality."""
        if len(text) <= max_len:
            return [text] if text.strip() else ["..."]

        chunks = []
        current = ""
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_len:
                current = f"{current} {sentence}".strip() if current else sentence
            else:
                if current:
                    chunks.append(current)
                current = sentence[:max_len]

        if current:
            chunks.append(current)

        return chunks if chunks else [text[:max_len]]
