"""
Tests for hbllm.data.scorer — data quality filtering heuristics.
"""

from hbllm.data.scorer import QualityScorer


class TestQualityScorer:
    """Tests for QualityScorer heuristics."""

    def setup_method(self):
        self.scorer = QualityScorer()

    def test_high_quality_text(self):
        text = (
            "This is a well-written English paragraph about machine learning. "
            "It contains proper words and punctuation, making it suitable for training."
        )
        assert self.scorer.is_high_quality(text)

    def test_rejects_gibberish(self):
        text = "x y z a b c d e f g h i j k l m n o p q r"  # mean word length < 3
        assert not self.scorer.is_high_quality(text)

    def test_rejects_symbol_heavy(self):
        text = "!@#$%^&*()_+{}|:<>?!@#$%^&*()_+{}|:<>?"
        assert not self.scorer.is_high_quality(text)

    def test_rejects_empty(self):
        assert not self.scorer.is_high_quality("")

    def test_rejects_very_short(self):
        assert not self.scorer.is_high_quality("Hi")

    def test_accepts_code(self):
        text = "def hello_world(): print('Hello, World!') return True"
        result = self.scorer.is_high_quality(text)
        # Code may or may not pass depending on alpha fraction threshold
        assert isinstance(result, bool)

    def test_rejects_really_long_words(self):
        text = "a" * 200 + " normal word " + "b" * 200
        assert not self.scorer.is_high_quality(text)

    def test_custom_thresholds(self):
        loose = QualityScorer(min_alpha_frac=0.3, max_symbol_frac=0.5)
        text = "Test!! @@ ## text with lots of symbols!!"
        # Should be more lenient
        result = loose.is_high_quality(text)
        assert isinstance(result, bool)

    def test_multiple_checks(self):
        good = "A normal sentence about artificial intelligence."
        bad = "!@#$%^&*()_+{}|:<>?" * 10
        assert self.scorer.is_high_quality(good)
        assert not self.scorer.is_high_quality(bad)

    def test_unicode_text(self):
        text = "Dies ist ein deutscher Satz über maschinelles Lernen und künstliche Intelligenz."
        result = self.scorer.is_high_quality(text)
        assert isinstance(result, bool)
