"""Tests for VoiceStreamBridge — ExpressionStream to AudioOutNode bridge."""

from hbllm.perception.voice_stream_bridge import (
    VoiceStreamBridge,
    VoiceStreamConfig,
    _SentenceBuffer,
)


class TestSentenceBuffer:
    def _make_buffer(self, **kwargs) -> _SentenceBuffer:
        return _SentenceBuffer(VoiceStreamConfig(**kwargs))

    def test_single_sentence(self):
        buf = self._make_buffer(min_sentence_length=3)
        sentences = buf.add("Hello world. ")
        assert len(sentences) == 1
        assert "Hello world" in sentences[0]

    def test_multiple_sentences(self):
        buf = self._make_buffer(min_sentence_length=3)
        sentences = buf.add("First sentence. Second sentence. ")
        assert len(sentences) == 2

    def test_incomplete_sentence_buffered(self):
        buf = self._make_buffer(min_sentence_length=3)
        sentences = buf.add("This is an incomplete")
        assert sentences == []

    def test_flush_remaining(self):
        buf = self._make_buffer(min_sentence_length=3)
        buf.add("Remaining text")
        flushed = buf.flush()
        assert len(flushed) == 1
        assert "Remaining text" in flushed[0]

    def test_flush_empty(self):
        buf = self._make_buffer()
        flushed = buf.flush()
        assert flushed == []

    def test_skip_code_blocks(self):
        buf = self._make_buffer(min_sentence_length=3, skip_code_blocks=True)
        text = "Here is code: ```python\nprint('hello')\n``` and more text. "
        sentences = buf.add(text)
        for s in sentences:
            assert "print" not in s

    def test_skip_urls(self):
        buf = self._make_buffer(min_sentence_length=3, skip_urls=True)
        sentences = buf.add("Visit https://example.com for details. ")
        for s in sentences:
            assert "https://" not in s

    def test_max_sentence_overflow(self):
        buf = self._make_buffer(min_sentence_length=3, max_sentence_length=20)
        long_text = "a" * 30  # No sentence boundaries
        sentences = buf.add(long_text)
        assert len(sentences) >= 1

    def test_incremental_build(self):
        buf = self._make_buffer(min_sentence_length=3)
        assert buf.add("Hello ") == []
        assert buf.add("world") == []
        sentences = buf.add(". Next. ")
        assert len(sentences) >= 1


class TestVoiceStreamConfig:
    def test_defaults(self):
        cfg = VoiceStreamConfig()
        assert cfg.min_sentence_length == 10
        assert cfg.allow_barge_in
        assert cfg.default_voice_id == "af_heart"


class TestVoiceStreamBridge:
    def test_instantiation(self):
        bridge = VoiceStreamBridge()
        assert "voice_streaming" in bridge.capabilities

    def test_stats(self):
        bridge = VoiceStreamBridge()
        stats = bridge.stats()
        assert stats["total_sentences_dispatched"] == 0
        assert stats["total_interruptions"] == 0
        assert stats["barge_in_enabled"]

    def test_session_management(self):
        bridge = VoiceStreamBridge()
        bridge.start_session("s1")
        assert bridge.is_active("s1")
        bridge.stop_session("s1")
        assert not bridge.is_active("s1")

    def test_custom_config(self):
        cfg = VoiceStreamConfig(
            allow_barge_in=False,
            default_voice_id="am_adam",
            speed=1.2,
        )
        bridge = VoiceStreamBridge(config=cfg)
        assert not bridge.config.allow_barge_in
        assert bridge.config.default_voice_id == "am_adam"
