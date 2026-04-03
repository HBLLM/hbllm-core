"""
Tests for AudioOutputNode per-tenant voice customization.

Tests text chunking, voice config, and tenant voice caching
WITHOUT loading the SpeechT5 model.
"""


from hbllm.perception.audio_out_node import AudioOutputNode


class TestTextChunking:
    """Test sentence-aligned text chunking for TTS."""

    def test_short_text_no_split(self):
        chunks = AudioOutputNode._chunk_text("Hello world.", max_len=450)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_empty_text(self):
        chunks = AudioOutputNode._chunk_text("", max_len=450)
        assert len(chunks) == 1
        assert chunks[0] == "..."

    def test_long_text_splits_on_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth very long sentence that adds more words."
        chunks = AudioOutputNode._chunk_text(text, max_len=50)
        assert len(chunks) >= 2
        # Each chunk should be under the limit
        for chunk in chunks:
            assert len(chunk) <= 50 or len(chunk.split('.')[0]) <= 50

    def test_single_very_long_sentence_truncated(self):
        text = "A " * 300  # 600 chars
        chunks = AudioOutputNode._chunk_text(text, max_len=100)
        assert len(chunks) >= 1
        assert len(chunks[0]) <= 100

    def test_exact_boundary(self):
        text = "A" * 450
        chunks = AudioOutputNode._chunk_text(text, max_len=450)
        assert len(chunks) == 1

    def test_multiple_sentences_stay_within_limit(self):
        sentences = [f"Sentence number {i}." for i in range(20)]
        text = " ".join(sentences)
        chunks = AudioOutputNode._chunk_text(text, max_len=100)
        assert all(len(c) <= 100 for c in chunks)


class TestVoiceConfig:
    """Test per-tenant voice configuration."""

    def test_default_tenant_voices_empty(self):
        node = AudioOutputNode(node_id="test_audio_out")
        assert len(node._tenant_voices) == 0

    def test_capabilities_include_voice_customization(self):
        node = AudioOutputNode(node_id="test_audio_out")
        assert "voice_customization" in node.capabilities
        assert "text_to_speech" in node.capabilities
