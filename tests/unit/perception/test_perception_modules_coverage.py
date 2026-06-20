"""Unit tests for perception modules — speaker_id_node, voice_stream_bridge, perception_fuser."""

import time

import numpy as np

from hbllm.network.node import NodeType
from hbllm.perception.speaker_id_node import SpeakerIdNode, UnknownEmbedding


class TestUnknownEmbedding:
    def test_creation(self):
        emb = UnknownEmbedding(
            embedding=np.zeros(256), session_id="sess-1", timestamp=time.time()
        )
        assert emb.session_id == "sess-1"
        assert emb.embedding.shape == (256,)


class _ConcreteSpeakerIdNode(SpeakerIdNode):
    """Concrete subclass for testing (SpeakerIdNode is abstract)."""

    async def on_start(self):
        pass

    async def on_stop(self):
        pass


class TestSpeakerIdNode:
    def test_init(self):
        node = _ConcreteSpeakerIdNode(node_id="speaker_id", node_type=NodeType.PERCEPTION)
        assert node.node_id == "speaker_id"

    def test_max_unknown_cache(self):
        assert SpeakerIdNode.MAX_UNKNOWN_CACHE == 200

    def test_unknown_ttl(self):
        assert SpeakerIdNode.UNKNOWN_TTL == 3600.0

    def test_payload_to_audio_missing(self):
        node = _ConcreteSpeakerIdNode(node_id="speaker_id", node_type=NodeType.PERCEPTION)
        result = node._payload_to_audio({})
        assert result is None

    def test_cache_unknown(self):
        node = _ConcreteSpeakerIdNode(node_id="speaker_id", node_type=NodeType.PERCEPTION)
        emb = np.random.randn(256).astype(np.float32)
        node._cache_unknown("tenant1", emb, session_id="s1", text="test")
        assert len(node._unknown_cache.get("tenant1", [])) == 1

    def test_cache_unknown_eviction(self):
        node = _ConcreteSpeakerIdNode(node_id="speaker_id", node_type=NodeType.PERCEPTION)
        for i in range(SpeakerIdNode.MAX_UNKNOWN_CACHE + 10):
            emb = np.random.randn(256).astype(np.float32)
            node._cache_unknown("t1", emb, session_id=f"s{i}")
        assert len(node._unknown_cache["t1"]) <= SpeakerIdNode.MAX_UNKNOWN_CACHE


# ── Voice Stream Bridge ──────────────────────────────────────────────────────

from hbllm.perception.voice_stream_bridge import (
    VoiceStreamBridge,
    VoiceStreamConfig,
    _SentenceBuffer,
)


class TestVoiceStreamConfig:
    def test_default_config(self):
        config = VoiceStreamConfig()
        assert config.min_sentence_length == 10
        assert config.max_sentence_length == 500


class TestSentenceBuffer:
    def test_init(self):
        config = VoiceStreamConfig()
        buf = _SentenceBuffer(config)
        assert buf is not None

    def test_add_text(self):
        config = VoiceStreamConfig()
        buf = _SentenceBuffer(config)
        sentences = buf.add("Hello world.")
        assert isinstance(sentences, list)

    def test_flush(self):
        config = VoiceStreamConfig()
        buf = _SentenceBuffer(config)
        buf.add("Hello world")
        flushed = buf.flush()
        assert isinstance(flushed, list)

    def test_clean(self):
        config = VoiceStreamConfig()
        buf = _SentenceBuffer(config)
        cleaned = buf._clean("  Hello   World  ")
        assert isinstance(cleaned, str)


class TestVoiceStreamBridge:
    def test_init(self):
        bridge = VoiceStreamBridge(node_id="voice_bridge")
        assert bridge.node_id == "voice_bridge"


# ── Perception Fuser ─────────────────────────────────────────────────────────

from hbllm.perception.perception_fuser import FusedContext, PerceptionEvent, PerceptionFuser


class TestPerceptionEvent:
    def test_creation(self):
        event = PerceptionEvent(modality="audio", content="Hello", confidence=0.9)
        assert event.modality == "audio"

    def test_age_seconds(self):
        event = PerceptionEvent(
            modality="text", content="test", confidence=1.0, timestamp=time.time() - 5
        )
        # age_seconds is a method, not a property
        age = event.age_seconds() if callable(event.age_seconds) else event.age_seconds
        assert age >= 4.0


class TestFusedContext:
    def test_is_multimodal_single(self):
        ctx = FusedContext(
            events=[PerceptionEvent("text", "hello", 1.0)],
            modalities={"text"},
        )
        assert ctx.is_multimodal is False

    def test_is_multimodal_multiple(self):
        ctx = FusedContext(
            events=[
                PerceptionEvent("text", "hello", 1.0),
                PerceptionEvent("audio", "sound", 0.8),
            ],
            modalities={"text", "audio"},
        )
        assert ctx.is_multimodal is True

    def test_to_dict(self):
        ctx = FusedContext(
            events=[PerceptionEvent("text", "hello", 1.0)],
            modalities={"text"},
        )
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert "modalities" in d or "is_multimodal" in d

    def test_summary_text(self):
        ctx = FusedContext(
            events=[PerceptionEvent("text", "hello world", 1.0)],
            modalities={"text"},
        )
        summary = ctx.summary_text
        if callable(summary):
            summary = summary()
        assert isinstance(summary, str)


class TestPerceptionFuser:
    def test_init(self):
        fuser = PerceptionFuser()
        assert fuser is not None
