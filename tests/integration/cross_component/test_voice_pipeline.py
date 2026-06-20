"""
Tests for the real-time voice pipeline.

Covers AudioInputNode (Moonshine/Whisper), AudioOutputNode (Kokoro/Orpheus),
VoiceConfig, VoiceRegistry, and the streaming pipeline.

All tests use mocks — no real ML models needed.
"""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock

import pytest_asyncio

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.perception.voice_config import (
    ASRBackend,
    AudioPipelineConfig,
    TTSBackend,
    VoiceConfig,
    VoiceRegistry,
)

# ── VoiceConfig Tests ──────────────────────────────────────────────────


class TestVoiceConfig:
    def test_default_voice_config(self):
        config = VoiceConfig()
        assert config.voice_id == "af_heart"
        assert config.speed == 1.0
        assert config.backend == TTSBackend.KOKORO
        assert config.language == "en-us"

    def test_custom_voice_config(self):
        config = VoiceConfig(
            voice_id="am_adam",
            speed=1.2,
            backend=TTSBackend.ORPHEUS,
            orpheus_emotion="happy",
        )
        assert config.voice_id == "am_adam"
        assert config.speed == 1.2
        assert config.backend == TTSBackend.ORPHEUS
        assert config.orpheus_emotion == "happy"

    def test_audio_pipeline_config_defaults(self):
        config = AudioPipelineConfig()
        assert config.asr_backend == ASRBackend.MOONSHINE
        assert config.tts_backend == TTSBackend.KOKORO
        assert config.vad_threshold == 0.5
        assert config.stream_sample_rate == 16000


class TestVoiceRegistry:
    def test_registry_create_and_get(self, tmp_path):
        db_path = tmp_path / "voice_test.db"
        registry = VoiceRegistry(db_path)

        # Default should return defaults
        voice = registry.get("tenant-1")
        assert voice.voice_id == "af_heart"
        assert voice.backend == TTSBackend.KOKORO

    def test_registry_set_and_get(self, tmp_path):
        db_path = tmp_path / "voice_test.db"
        registry = VoiceRegistry(db_path)

        custom = VoiceConfig(
            voice_id="am_michael",
            speed=0.9,
            backend=TTSBackend.KOKORO,
            language="en-us",
        )
        registry.set("tenant-2", custom)

        result = registry.get("tenant-2")
        assert result.voice_id == "am_michael"
        assert result.speed == 0.9

    def test_registry_cache(self, tmp_path):
        db_path = tmp_path / "voice_test.db"
        registry = VoiceRegistry(db_path)

        custom = VoiceConfig(voice_id="bf_emma")
        registry.set("tenant-3", custom)

        # Second get should come from cache
        result = registry.get("tenant-3")
        assert result.voice_id == "bf_emma"
        assert "tenant-3" in registry._cache

    def test_registry_list_voices(self, tmp_path):
        db_path = tmp_path / "voice_test.db"
        registry = VoiceRegistry(db_path)

        voices = registry.list_voices(TTSBackend.KOKORO)
        assert len(voices) > 0
        assert any(v["voice_id"] == "af_heart" for v in voices)
        assert all("gender" in v and "language" in v for v in voices)

    def test_registry_persistence(self, tmp_path):
        db_path = tmp_path / "voice_persist.db"

        # Write with one instance
        r1 = VoiceRegistry(db_path)
        r1.set("t1", VoiceConfig(voice_id="am_adam"))

        # Read with a fresh instance (no cache)
        r2 = VoiceRegistry(db_path)
        result = r2.get("t1")
        assert result.voice_id == "am_adam"


# ── AudioInputNode Tests ───────────────────────────────────────────────


class TestAudioInputNode:
    @pytest_asyncio.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest_asyncio.fixture
    async def audio_in(self, bus):
        from hbllm.perception.audio_in_node import AudioInputNode

        config = AudioPipelineConfig(asr_backend=ASRBackend.MOONSHINE)
        node = AudioInputNode(node_id="test_audio_in", config=config)
        await node.start(bus)
        yield node
        await node.stop()

    async def test_transcribe_message_missing_path(self, audio_in):
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="sensory.audio.in",
            payload={},
        )
        resp = await audio_in.handle_transcribe(msg)
        assert resp is not None
        assert resp.type == MessageType.ERROR

    async def test_transcribe_forwards_to_router(self, audio_in, bus):
        """Verify that a successful transcription publishes to router.query."""
        received = []

        async def capture(msg):
            received.append(msg)

        await bus.subscribe("router.query", capture)

        # Mock the transcription
        async def _mock_transcribe(path):
            return "Hello world"

        audio_in._transcribe_file = _mock_transcribe

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="sensory.audio.in",
            payload={"file_path": "/tmp/test.wav"},
        )
        resp = await audio_in.handle_transcribe(msg)
        assert resp.payload["text"] == "Hello world"

        # Wait for the fire-and-forget task
        await asyncio.sleep(0.1)
        assert len(received) == 1
        assert received[0].payload["text"] == "Hello world"
        assert received[0].payload["source"] == "audio"

    async def test_stream_buffer_append(self, audio_in):
        """Test that streaming chunks are properly buffered."""
        # Create a small PCM chunk (100 samples, 16-bit)
        pcm = struct.pack("<" + "h" * 100, *([1000] * 100))

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            session_id="s1",
            topic="sensory.audio.stream",
            payload={"chunk": pcm.hex(), "sample_rate": 16000},
        )
        await audio_in.handle_stream(msg)

        assert "s1" in audio_in._stream_buffers
        assert audio_in._stream_buffers["s1"].has_audio

    async def test_stream_flush_on_final(self, audio_in, bus):
        """Test that is_final forces a flush and transcription."""

        async def _mock_transcribe_pcm(pcm_bytes, sr):
            return "stream test"

        audio_in._transcribe_pcm = _mock_transcribe_pcm

        received = []

        async def capture(msg):
            received.append(msg)

        await bus.subscribe("router.query", capture)

        pcm = struct.pack("<" + "h" * 100, *([500] * 100))
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            session_id="s2",
            topic="sensory.audio.stream",
            payload={"chunk": pcm.hex(), "is_final": True},
        )
        await audio_in.handle_stream(msg)

        await asyncio.sleep(0.2)
        assert len(received) == 1
        assert received[0].payload["text"] == "stream test"


# ── AudioOutputNode Tests ──────────────────────────────────────────────


class TestAudioOutputNode:
    @pytest_asyncio.fixture
    async def bus(self):
        bus = InProcessBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest_asyncio.fixture
    async def audio_out(self, bus, tmp_path):
        from hbllm.perception.audio_out_node import AudioOutputNode

        config = AudioPipelineConfig(tts_backend=TTSBackend.KOKORO)
        node = AudioOutputNode(
            node_id="test_audio_out",
            output_dir=str(tmp_path / "audio"),
            config=config,
            data_dir=str(tmp_path),
        )
        await node.start(bus)
        yield node
        await node.stop()

    async def test_synthesize_missing_text(self, audio_out):
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="sensory.audio.out",
            payload={},
        )
        resp = await audio_out.handle_synthesize(msg)
        assert resp is not None
        assert resp.type == MessageType.ERROR

    async def test_synthesize_full_with_mock(self, audio_out, tmp_path):
        """Test full synthesis with a mocked Kokoro pipeline."""

        async def _mock_synth(text, voice, filename):
            out = tmp_path / "audio" / filename
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"fake_wav_data")
            return str(out)

        audio_out._synthesize_full = AsyncMock(side_effect=_mock_synth)

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="sensory.audio.out",
            payload={"text": "Hello, how are you?"},
        )
        audio_out.handle_synthesize = audio_out.__class__.handle_synthesize.__get__(audio_out)
        resp = await audio_out.handle_synthesize(msg)
        # Since we mocked _synthesize_full, it should succeed
        assert resp is not None

    async def test_sentence_splitting(self, audio_out):
        """Test sentence splitting for streaming TTS."""
        sentences = audio_out._split_sentences("Hello there. How are you? I'm doing great!")
        assert len(sentences) == 3
        assert sentences[0] == "Hello there."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I'm doing great!"

    async def test_chunk_text(self, audio_out):
        """Test text chunking for long input."""
        short = "Hello."
        assert audio_out._chunk_text(short) == ["Hello."]

        # Long text
        long_text = ". ".join([f"Sentence {i}" for i in range(100)])
        chunks = audio_out._chunk_text(long_text, max_len=100)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    async def test_voice_config_handler(self, audio_out, bus):
        """Test per-tenant voice configuration via bus."""
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            topic="voice.config",
            payload={
                "voice_id": "am_adam",
                "speed": 1.5,
                "backend": "kokoro",
            },
        )
        resp = await audio_out._handle_voice_config(msg)
        assert resp is not None
        assert resp.payload["status"] == "updated"

        # Verify it's stored
        voice = audio_out._voice_registry.get("t1")
        assert voice.voice_id == "am_adam"
        assert voice.speed == 1.5

    async def test_list_voices(self, audio_out):
        """Test listing available voices."""
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="voice.list",
            payload={"backend": "kokoro"},
        )
        resp = await audio_out._handle_list_voices(msg)
        assert resp is not None
        voices = resp.payload["voices"]
        assert len(voices) > 0

    async def test_streaming_synthesis(self, audio_out, bus):
        """Test sentence-level streaming synthesis."""
        import numpy as np

        chunks_received = []

        async def capture(msg):
            chunks_received.append(msg)

        await bus.subscribe("sensory.audio.chunk", capture)

        # Mock the sentence synthesizer
        async def _mock_sentence(sentence, voice):
            pcm = np.zeros(4800, dtype=np.int16).tobytes()  # 0.2s silence
            return pcm, 24000

        audio_out._synthesize_sentence = _mock_sentence

        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            tenant_id="default",
            session_id="s1",
            topic="sensory.audio.out",
            payload={"text": "First sentence. Second sentence.", "stream": True},
        )
        await audio_out.handle_synthesize(msg)

        await asyncio.sleep(0.2)
        assert len(chunks_received) == 2
        assert chunks_received[-1].payload["is_final"] is True


# ── End-to-end pipeline test ───────────────────────────────────────────


class TestVoicePipelineE2E:
    async def test_audio_in_to_audio_out(self):
        """Full bus round-trip: audio.in → router.query → ... → audio.out."""
        from hbllm.perception.audio_in_node import AudioInputNode
        from hbllm.perception.audio_out_node import AudioOutputNode

        bus = InProcessBus()
        await bus.start()

        config = AudioPipelineConfig()

        audio_in = AudioInputNode(node_id="e2e_in", config=config)
        audio_out = AudioOutputNode(node_id="e2e_out", config=config, output_dir="/tmp/e2e_audio")

        await audio_in.start(bus)
        await audio_out.start(bus)

        # Track what gets published to router.query
        router_received = []

        async def capture_router(msg):
            router_received.append(msg)

        await bus.subscribe("router.query", capture_router)

        # Mock the transcription
        async def _mock_transcribe(path):
            return "What is the weather?"

        audio_in._transcribe_file = _mock_transcribe

        # Send audio transcription request
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="sensory.audio.in",
            payload={"file_path": "/tmp/test_e2e.wav"},
        )
        resp = await audio_in.handle_transcribe(msg)
        assert resp.payload["text"] == "What is the weather?"

        await asyncio.sleep(0.2)
        assert len(router_received) == 1
        assert router_received[0].payload["text"] == "What is the weather?"

        await audio_out.stop()
        await audio_in.stop()
        await bus.stop()


# ── StreamBuffer Tests ─────────────────────────────────────────────────


class TestStreamBuffer:
    def test_buffer_creation(self):
        from hbllm.perception.audio_in_node import _StreamBuffer

        buf = _StreamBuffer(sample_rate=16000)
        assert not buf.has_audio
        assert buf.sample_rate == 16000

    def test_buffer_append(self):
        from hbllm.perception.audio_in_node import _StreamBuffer

        buf = _StreamBuffer()
        buf.append(b"\x00\x01" * 100)
        assert buf.has_audio
        assert len(buf.get_audio_bytes()) == 200

    def test_buffer_flush_on_max_duration(self):
        from hbllm.perception.audio_in_node import _StreamBuffer

        buf = _StreamBuffer(max_buffer_seconds=0.0)  # Immediate flush
        buf.append(b"\x00" * 100)
        assert buf.should_flush(None)  # No VAD, but max time exceeded

    def test_buffer_silence_timeout(self):
        from hbllm.perception.audio_in_node import _StreamBuffer

        buf = _StreamBuffer(max_silence_ms=0)
        buf.append(b"\x00" * 100)
        # last_chunk_time is now, silence is 0ms which == max_silence_ms
        assert buf.should_flush(None)
