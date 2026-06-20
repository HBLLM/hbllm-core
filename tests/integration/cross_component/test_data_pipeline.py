"""Integration tests for Data Pipeline — PurePythonPipeline, QualityScorer, ShardWriter."""

from pathlib import Path

from hbllm.data.pipeline import _clean_text
from hbllm.data.scorer import QualityScorer
from hbllm.data.sharder import ShardWriter

# ── Text Cleaning Tests ──────────────────────────────────────────────────────


class TestTextCleaning:
    """Test the pure-Python text cleaning function."""

    def test_normalizes_whitespace(self):
        result = _clean_text("Hello   world\n\nfoo\tbar")
        assert result == "Hello world foo bar"

    def test_strips_control_characters(self):
        result = _clean_text("Hello\x00\x01\x02World")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Hello" in result
        assert "World" in result

    def test_empty_string(self):
        assert _clean_text("") == ""

    def test_none_returns_empty(self):
        assert _clean_text(None) == ""

    def test_non_string_returns_empty(self):
        assert _clean_text(123) == ""

    def test_normal_text_unchanged(self):
        text = "This is a normal sentence."
        assert _clean_text(text) == "This is a normal sentence."

    def test_unicode_preserved(self):
        text = "日本語テスト with unicode"
        result = _clean_text(text)
        assert "日本語テスト" in result


# ── QualityScorer Tests ──────────────────────────────────────────────────────


class TestQualityScorerIntegration:
    """Test quality scoring for document filtering."""

    def test_high_quality_document(self):
        scorer = QualityScorer()
        doc = (
            "Python is a high-level programming language known for its readability. "
            "It supports multiple programming paradigms including procedural, "
            "object-oriented, and functional programming. Python was created by "
            "Guido van Rossum and first released in 1991. Its design philosophy "
            "emphasizes code readability with its use of significant indentation."
        )
        assert scorer.is_high_quality(doc) is True

    def test_low_quality_short_document(self):
        scorer = QualityScorer()
        doc = "Hi"
        assert scorer.is_high_quality(doc) is False

    def test_low_quality_symbols(self):
        """Documents with excessive symbols should be rejected."""
        scorer = QualityScorer()
        doc = "!@#$%^&*()!@#$%^&*()!@#$%^&*() " * 20
        assert scorer.is_high_quality(doc) is False

    def test_empty_document(self):
        scorer = QualityScorer()
        assert scorer.is_high_quality("") is False

    def test_is_high_quality_returns_bool(self):
        scorer = QualityScorer()
        doc = (
            "Machine learning is a subset of artificial intelligence that allows "
            "systems to learn from data. It has applications in natural language "
            "processing, computer vision, and many other fields."
        )
        result = scorer.is_high_quality(doc)
        assert isinstance(result, bool)
        assert result is True


# ── ShardWriter Tests ────────────────────────────────────────────────────────


class TestShardWriterIntegration:
    """Test binary shard creation and format."""

    def test_writes_shards(self, tmp_path):
        writer = ShardWriter(
            output_dir=tmp_path,
            shard_size_mb=1,
            sequence_length=8,
        )

        # Write enough tokens to trigger shard creation
        for _ in range(100):
            writer.add_tokens(list(range(10)))

        writer.flush()

        # Should have created shard files
        assert len(writer.created_shards) >= 1
        assert writer.total_tokens > 0

        # Verify shard files exist
        for shard in writer.created_shards:
            assert Path(shard).exists()

    def test_flush_remaining_buffer(self, tmp_path):
        writer = ShardWriter(
            output_dir=tmp_path,
            shard_size_mb=100,
            sequence_length=8,
        )

        writer.add_tokens([1, 2, 3, 4, 5, 6, 7, 8, 0])
        writer.flush()

        # Should have at least one shard
        assert len(writer.created_shards) >= 1

    def test_empty_write(self, tmp_path):
        writer = ShardWriter(
            output_dir=tmp_path,
            shard_size_mb=100,
            sequence_length=8,
        )
        writer.flush()  # Nothing to flush
        # No shards created (or empty shard)

    def test_respects_sequence_length(self, tmp_path):
        seq_len = 16
        writer = ShardWriter(
            output_dir=tmp_path,
            shard_size_mb=1,
            sequence_length=seq_len,
        )

        for _ in range(50):
            writer.add_tokens(list(range(20)))
        writer.flush()

        # Verify shard file is readable and contains correct token format
        if writer.created_shards:
            shard_path = Path(writer.created_shards[0])
            data = shard_path.read_bytes()
            assert len(data) > 0

    def test_uint32_dtype(self, tmp_path):
        writer = ShardWriter(
            output_dir=tmp_path,
            shard_size_mb=1,
            sequence_length=8,
            dtype="uint32",
        )

        # Token IDs > 65535 (requires uint32)
        big_tokens = [100277, 99999, 80000, 70000, 60000, 50000, 40000, 30000, 0]
        for _ in range(20):
            writer.add_tokens(big_tokens)
        writer.flush()

        assert len(writer.created_shards) >= 1


# ── End-to-End Pipeline Tests (without actual datasets) ──────────────────────


class TestPipelineHelpers:
    """Test pipeline utility functions used in data preparation."""

    def test_clean_and_filter_workflow(self):
        """Simulate the clean → score → filter pipeline."""
        scorer = QualityScorer()

        raw_docs = [
            # Good quality
            "Machine learning is a powerful subset of artificial intelligence that enables "
            "computers to learn and improve from experience without being explicitly "
            "programmed. It uses algorithms and statistical models to analyze and draw "
            "inferences from patterns in data.",
            # Too short
            "Hi.",
            # Symbol-heavy (low quality)
            "!@#$%^& !@#$%^& !@#$%^& " * 30,
            # Good quality
            "Natural language processing is a field of computer science and artificial "
            "intelligence concerned with the interactions between computers and human "
            "language. Modern NLP uses deep learning approaches.",
        ]

        cleaned = [_clean_text(doc) for doc in raw_docs]
        high_quality = [doc for doc in cleaned if scorer.is_high_quality(doc)]

        assert len(high_quality) == 2
        assert "Machine learning" in high_quality[0]
        assert "Natural language" in high_quality[1]

    def test_clean_tokenize_shard_workflow(self, tmp_path):
        """Simulate clean → tokenize → shard without real tokenizer."""
        docs = [
            "Python is a great programming language for data science.",
            "JavaScript powers the modern web with frameworks like React.",
        ]

        cleaned = [_clean_text(doc) for doc in docs]

        # Fake tokenization (character-level)
        writer = ShardWriter(
            output_dir=tmp_path,
            shard_size_mb=1,
            sequence_length=32,
        )

        for doc in cleaned:
            tokens = [ord(c) for c in doc]
            tokens.append(0)  # EOS
            writer.add_tokens(tokens)

        writer.flush()

        assert writer.total_tokens > 0
        assert len(writer.created_shards) >= 1
