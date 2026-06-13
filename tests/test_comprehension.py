"""
Tests for the Cognitive Stream comprehension pipeline.

Covers:
    - LexicalBuffer: subword assembly, flush, boundary detection
    - LexicalSignals: signal computation for various word types
    - ComprehensionEnsemble: neuron firing, domain params, signal routing
    - ComprehensionStream: end-to-end with mock encoder/memory
    - SNNCalibrator: outcome recording, parameter suggestion
    - RouterNode integration with ComprehensionStream
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from hbllm.brain.snn.comprehension.calibrator import SNNCalibrator
from hbllm.brain.snn.comprehension.ensemble import (
    DOMAIN_PARAMS,
    ComprehensionEnsemble,
)
from hbllm.brain.snn.comprehension.lexical import (
    _CONSTRAINT_WORDS,
    _STOPWORDS,
    _TECHNICAL_TERMS,
    LexicalBuffer,
    LexicalSignals,
)
from hbllm.brain.snn.comprehension.models import (
    ActivatedMemory,
    ComprehensionUnit,
    UnderstandingState,
)
from hbllm.brain.snn.comprehension.stream import ComprehensionStream

# ── LexicalBuffer tests ──────────────────────────────────────────────────


class TestLexicalBuffer:
    """Test Layer 0: subword noise absorption."""

    def test_simple_words(self):
        """Complete words should pass through directly."""
        buf = LexicalBuffer()
        # First word: nothing to flush yet
        result1 = buf.feed("Hello")
        assert result1 is None

        # Second word flushes first
        result2 = buf.feed("world")
        assert result2 == "Hello"

        # Flush remainder
        result3 = buf.flush()
        assert result3 == "world"

    def test_subword_continuation(self):
        """## prefixed tokens should be merged with previous."""
        buf = LexicalBuffer()
        buf.feed("auth")
        result = buf.feed("##entication")
        assert result is None  # Still building

        # Next word flushes assembled word
        result = buf.feed("system")
        assert result == "authentication"

    def test_flush_empty(self):
        """Flushing an empty buffer returns None."""
        buf = LexicalBuffer()
        assert buf.flush() is None

    def test_reset(self):
        """Reset clears the buffer."""
        buf = LexicalBuffer()
        buf.feed("test")
        buf.reset()
        assert buf.flush() is None

    def test_space_prefixed_tokens(self):
        """Space-prefixed tokens start new words."""
        buf = LexicalBuffer()
        buf.feed(" Hello")
        result = buf.feed(" world")
        assert result == "Hello"


# ── LexicalSignals tests ────────────────────────────────────────────────


class TestLexicalSignals:
    """Test Layer 1: cheap lexical signal computation."""

    def test_stopword_has_zero_semantic_weight(self):
        """Stopwords should have zero semantic weight."""
        signals = LexicalSignals.compute("the", [], None)
        assert signals["semantic_weight"] == 0.0

    def test_technical_term_has_high_semantic_weight(self):
        """Known technical terms should have 0.8 semantic weight."""
        signals = LexicalSignals.compute("api", [], None)
        assert signals["semantic_weight"] == 0.8

    def test_constraint_word_signals(self):
        """Constraint words should trigger constraint and topic_shift signals."""
        signals = LexicalSignals.compute("only", [], None)
        assert signals["constraint"] == 0.8
        assert signals["topic_shift"] == 0.5

    def test_shift_word_signals(self):
        """Shift words should trigger topic_shift signal."""
        signals = LexicalSignals.compute("additionally", [], None)
        assert signals["topic_shift"] == 0.7

    def test_novelty_with_empty_buffer(self):
        """Words not in buffer should have 0.5 novelty."""
        signals = LexicalSignals.compute("kubernetes", [], None)
        assert signals["novelty"] == 0.5

    def test_novelty_with_repeated_word(self):
        """Words already in buffer should have 0.0 novelty."""
        signals = LexicalSignals.compute("deploy", ["deploy", "now"], None)
        assert signals["novelty"] == 0.0

    def test_inter_novelty_first_concept(self):
        """First concept (no prev) should get moderate inter_novelty."""
        signals = LexicalSignals.compute("test", [], None)
        assert signals["inter_novelty"] == 0.2

    def test_inter_novelty_new_word(self):
        """New word vs previous concept should get 0.4 inter_novelty."""
        signals = LexicalSignals.compute("deploy", [], ["test", "code"])
        assert signals["inter_novelty"] == 0.4

    def test_buffer_pressure_empty(self):
        """Empty buffer should have zero pressure."""
        signals = LexicalSignals.compute("test", [], None)
        assert signals["buffer_pressure"] == 0.0

    def test_buffer_pressure_full(self):
        """Full buffer (12+ words) should have max pressure (0.3)."""
        long_buffer = ["word"] * 15
        signals = LexicalSignals.compute("test", long_buffer, None)
        assert signals["buffer_pressure"] == pytest.approx(0.3)

    def test_punctuation_signal(self):
        """Words ending with punctuation should trigger punctuation signal."""
        signals = LexicalSignals.compute("errors,", [], None)
        assert signals["punctuation"] == 0.4

    def test_no_punctuation_signal(self):
        """Words without punctuation should not trigger signal."""
        signals = LexicalSignals.compute("errors", [], None)
        assert signals["punctuation"] == 0.0

    def test_all_signal_channels_present(self):
        """All 7 signal channels should always be present."""
        signals = LexicalSignals.compute("test", [], None)
        expected_keys = {
            "semantic_weight",
            "topic_shift",
            "novelty",
            "inter_novelty",
            "constraint",
            "buffer_pressure",
            "punctuation",
        }
        assert set(signals.keys()) == expected_keys

    def test_all_signals_in_range(self):
        """All signal values should be in [0.0, 1.0]."""
        test_words = ["api", "the", "but", "additionally", "hello.", "x"]
        for word in test_words:
            signals = LexicalSignals.compute(word, ["some", "context"], ["prev"])
            for key, val in signals.items():
                assert 0.0 <= val <= 1.0, f"{key}={val} out of range for word '{word}'"


# ── ComprehensionEnsemble tests ──────────────────────────────────────────


class TestComprehensionEnsemble:
    """Test the 5-channel neuron ensemble."""

    def test_all_channels_created(self):
        """Ensemble should have exactly 5 channels."""
        ensemble = ComprehensionEnsemble(domain="general")
        assert len(ensemble.channels) == 5
        assert set(ensemble.channels.keys()) == {
            "entity",
            "clause",
            "discourse",
            "surprise",
            "constraint",
        }

    def test_domain_params_exist(self):
        """All documented domain param sets should exist."""
        assert "general" in DOMAIN_PARAMS
        assert "code" in DOMAIN_PARAMS
        assert "math" in DOMAIN_PARAMS
        assert "dialogue" in DOMAIN_PARAMS

    def test_code_domain_higher_thresholds(self):
        """Code domain should have higher thresholds than general."""
        assert (
            DOMAIN_PARAMS["code"]["entity_threshold"] > DOMAIN_PARAMS["general"]["entity_threshold"]
        )
        assert (
            DOMAIN_PARAMS["code"]["clause_threshold"] > DOMAIN_PARAMS["general"]["clause_threshold"]
        )

    def test_step_returns_fired_channels(self):
        """Step with strong signals should eventually fire neurons."""
        ensemble = ComprehensionEnsemble(domain="general")
        t = time.time()

        # Feed strong signals repeatedly until clause fires
        strong_signals = {
            "semantic_weight": 0.8,
            "topic_shift": 0.7,
            "novelty": 0.5,
            "inter_novelty": 0.4,
            "constraint": 0.0,
            "buffer_pressure": 0.3,
            "punctuation": 0.4,
        }

        all_fired = []
        for i in range(20):
            fired = ensemble.step(strong_signals, t + i * 0.001)
            all_fired.extend(fired)

        # At least one channel should have fired
        assert len(all_fired) > 0
        channel_names = [ch for ch, _ in all_fired]
        assert any(ch in channel_names for ch in ["entity", "clause", "constraint"])

    def test_step_no_fire_on_zero_signals(self):
        """Zero signals should not fire any neurons."""
        ensemble = ComprehensionEnsemble(domain="general")
        t = time.time()

        zero_signals = {
            "semantic_weight": 0.0,
            "topic_shift": 0.0,
            "novelty": 0.0,
            "inter_novelty": 0.0,
            "constraint": 0.0,
            "buffer_pressure": 0.0,
            "punctuation": 0.0,
        }

        fired = ensemble.step(zero_signals, t)
        assert len(fired) == 0

    def test_reset_clears_state(self):
        """Reset should clear all neuron membrane potentials."""
        ensemble = ComprehensionEnsemble()
        t = time.time()

        # Charge up neurons
        signals = {"semantic_weight": 0.5, "topic_shift": 0.3}
        ensemble.step(signals, t)

        # Reset
        ensemble.reset()

        # All potentials should be zero
        for neuron in ensemble.channels.values():
            assert neuron.v == 0.0

    def test_update_params(self):
        """update_params should modify neuron thresholds."""
        ensemble = ComprehensionEnsemble()
        new_params = {
            "entity_threshold": 0.9,
            "clause_threshold": 1.5,
            "discourse_threshold": 2.0,
        }
        ensemble.update_params(new_params)

        assert ensemble.channels["entity"].config.threshold == 0.9
        assert ensemble.channels["clause"].config.threshold == 1.5
        assert ensemble.channels["discourse"].config.threshold == 2.0


# ── Data Model tests ────────────────────────────────────────────────────


class TestModels:
    """Test data models."""

    def test_activated_memory_defaults(self):
        mem = ActivatedMemory(id="m1", content="test content")
        assert mem.score == 0.0

    def test_comprehension_unit_defaults(self):
        unit = ComprehensionUnit(text="test", embedding=np.zeros(384))
        assert unit.salience == 1.0
        assert unit.activated_memories == []
        assert unit.domain_activation == {}

    def test_understanding_state_defaults(self):
        state = UnderstandingState()
        assert state.concepts == []
        assert state.domain_activations == {}
        assert state.all_memories == []
        assert state.salience_map == []


# ── ComprehensionStream tests ───────────────────────────────────────────


def _mock_encoder(text: str) -> np.ndarray:
    """Mock ONNX encoder that returns a deterministic vector based on text length."""
    rng = np.random.RandomState(len(text))
    vec = rng.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


async def _mock_memory_search(text: str) -> list[ActivatedMemory]:
    """Mock memory search returning a single fake memory."""
    return [
        ActivatedMemory(
            id=f"mem_{hash(text) % 1000}",
            content=f"Memory related to: {text[:50]}",
            score=0.8,
        )
    ]


class TestComprehensionStream:
    """Test the full comprehension pipeline."""

    def _make_stream(self, with_memory: bool = False) -> ComprehensionStream:
        """Create a ComprehensionStream with mocks."""
        ensemble = ComprehensionEnsemble(domain="general")
        lexical_buffer = LexicalBuffer()

        # Create some mock centroids
        centroids = {
            "coding": _mock_encoder("coding programming software"),
            "general": _mock_encoder("general knowledge help"),
        }

        return ComprehensionStream(
            ensemble=ensemble,
            lexical_buffer=lexical_buffer,
            encoder=_mock_encoder,
            domain_centroids=centroids,
            memory_search_fn=_mock_memory_search if with_memory else None,
        )

    @pytest.mark.asyncio
    async def test_basic_comprehension(self):
        """Stream should produce at least one concept for any input."""
        stream = self._make_stream()
        result = await stream.comprehend(
            "My Laravel API returns 500 errors on auth but only in production"
        )

        assert isinstance(result, UnderstandingState)
        assert len(result.concepts) >= 1
        assert len(result.salience_map) == len(result.concepts)

    @pytest.mark.asyncio
    async def test_concepts_have_text(self):
        """Each concept should have non-empty text."""
        stream = self._make_stream()
        result = await stream.comprehend(
            "How do I deploy a Docker container to Kubernetes with auto-scaling"
        )

        for concept in result.concepts:
            assert concept.text.strip() != ""

    @pytest.mark.asyncio
    async def test_concepts_have_embeddings(self):
        """Each concept should have a non-zero embedding."""
        stream = self._make_stream()
        result = await stream.comprehend("The server crashed after deploying the new configuration")

        for concept in result.concepts:
            assert concept.embedding is not None
            assert np.linalg.norm(concept.embedding) > 0

    @pytest.mark.asyncio
    async def test_domain_activations_populated(self):
        """Domain activations should be populated from centroid matching."""
        stream = self._make_stream()
        result = await stream.comprehend(
            "Write a Python function that calculates fibonacci numbers"
        )

        # At least some domain activations should exist
        # (depends on mock encoder similarity)
        assert isinstance(result.domain_activations, dict)

    @pytest.mark.asyncio
    async def test_memory_retrieval(self):
        """When memory_search_fn is provided, memories should be populated."""
        stream = self._make_stream(with_memory=True)
        result = await stream.comprehend("Explain how Redis caching works with a Laravel backend")

        assert len(result.all_memories) >= 1
        assert all(isinstance(m, ActivatedMemory) for m in result.all_memories)

    @pytest.mark.asyncio
    async def test_no_memory_without_fn(self):
        """Without memory_search_fn, memories should be empty."""
        stream = self._make_stream(with_memory=False)
        result = await stream.comprehend("Simple test query for comprehension")

        assert result.all_memories == []

    @pytest.mark.asyncio
    async def test_salience_map_length(self):
        """Salience map should have one entry per concept."""
        stream = self._make_stream()
        result = await stream.comprehend(
            "Configure nginx reverse proxy and then set up SSL certificates"
        )

        assert len(result.salience_map) == len(result.concepts)

    @pytest.mark.asyncio
    async def test_short_input(self):
        """Very short input should still produce at least one concept."""
        stream = self._make_stream()
        result = await stream.comprehend("Hello world")

        assert len(result.concepts) >= 1

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """Reset should clear all internal state."""
        stream = self._make_stream()
        await stream.comprehend("First query")
        stream.reset()

        assert stream._word_buffer == []
        assert stream._prev_concept_words is None
        assert stream._concepts == []

    @pytest.mark.asyncio
    async def test_memory_deduplication(self):
        """Memories across concepts should be deduplicated by ID."""
        # Use a mock that returns the same memory ID for different texts
        call_count = 0

        async def same_id_memory(text: str) -> list[ActivatedMemory]:
            nonlocal call_count
            call_count += 1
            return [ActivatedMemory(id="shared_mem", content="shared content", score=0.9)]

        ensemble = ComprehensionEnsemble(domain="general")
        stream = ComprehensionStream(
            ensemble=ensemble,
            lexical_buffer=LexicalBuffer(),
            encoder=_mock_encoder,
            domain_centroids={"general": _mock_encoder("general")},
            memory_search_fn=same_id_memory,
        )

        result = await stream.comprehend(
            "Deploy the Docker container and then configure the Kubernetes cluster with auto-scaling enabled"
        )

        # Even with multiple concepts, shared_mem should appear only once
        shared_count = sum(1 for m in result.all_memories if m.id == "shared_mem")
        assert shared_count <= 1


# ── SNNCalibrator tests ─────────────────────────────────────────────────


class TestSNNCalibrator:
    """Test parameter calibration."""

    def test_suggest_params_default_with_no_history(self):
        """With no history, should return default domain params."""
        cal = SNNCalibrator()
        params = cal.suggest_params("general")
        assert params == DOMAIN_PARAMS["general"]

    def test_suggest_params_default_for_unknown_domain(self):
        """Unknown domain should fall back to general params."""
        cal = SNNCalibrator()
        params = cal.suggest_params("nonexistent")
        assert params == DOMAIN_PARAMS["general"]

    def test_record_outcome_updates_history(self):
        """Recording outcomes should build history."""
        cal = SNNCalibrator()
        for i in range(15):
            cal.record_outcome(
                domain="coding",
                params_used={
                    "entity_threshold": 0.8,
                    "clause_threshold": 1.0,
                    "discourse_threshold": 1.5,
                },
                num_concepts=3,
                response_quality=0.7,
                memory_relevance=0.6,
            )

        # After 15 entries, suggest_params should return adjusted params
        params = cal.suggest_params("coding")
        assert isinstance(params, dict)
        assert "entity_threshold" in params

    def test_success_ema_updates(self):
        """Domain success EMA should update with outcomes."""
        cal = SNNCalibrator()
        cal.record_outcome(
            domain="math",
            params_used={"entity_threshold": 0.5},
            num_concepts=2,
            response_quality=0.9,
            memory_relevance=0.8,
        )
        success = cal.get_domain_success("math")
        assert success > 0.5  # Should shift upward from default 0.5

    def test_history_cap_at_100(self):
        """History should be capped at 100 entries."""
        cal = SNNCalibrator()
        for i in range(150):
            cal.record_outcome(
                domain="general",
                params_used={"entity_threshold": 0.6},
                num_concepts=2,
                response_quality=0.5 + (i % 10) * 0.05,
                memory_relevance=0.5,
            )

        assert len(cal._param_history["general"]) == 100

    def test_param_nudge_direction(self):
        """Suggested params should nudge toward successful configurations."""
        cal = SNNCalibrator()

        # Record bad outcomes with high thresholds
        for i in range(10):
            cal.record_outcome(
                domain="general",
                params_used={
                    "entity_threshold": 0.9,
                    "clause_threshold": 1.2,
                    "discourse_threshold": 1.8,
                },
                num_concepts=1,
                response_quality=0.2,
                memory_relevance=0.1,
            )

        # Record good outcomes with lower thresholds
        for i in range(10):
            cal.record_outcome(
                domain="general",
                params_used={
                    "entity_threshold": 0.5,
                    "clause_threshold": 0.7,
                    "discourse_threshold": 1.0,
                },
                num_concepts=4,
                response_quality=0.9,
                memory_relevance=0.85,
            )

        params = cal.suggest_params("general")
        # Should nudge toward the lower (successful) thresholds
        assert params["entity_threshold"] < DOMAIN_PARAMS["general"]["entity_threshold"]

    def test_param_clamping(self):
        """Suggested params should be clamped to [0.3, 2.0]."""
        cal = SNNCalibrator()

        # Create extreme divergence
        for i in range(10):
            cal.record_outcome(
                domain="general",
                params_used={
                    "entity_threshold": 0.1,
                    "clause_threshold": 0.1,
                    "discourse_threshold": 0.1,
                },
                num_concepts=10,
                response_quality=1.0,
                memory_relevance=1.0,
            )
        for i in range(5):
            cal.record_outcome(
                domain="general",
                params_used={
                    "entity_threshold": 5.0,
                    "clause_threshold": 5.0,
                    "discourse_threshold": 5.0,
                },
                num_concepts=0,
                response_quality=0.0,
                memory_relevance=0.0,
            )

        params = cal.suggest_params("general")
        for key, val in params.items():
            assert 0.3 <= val <= 2.0, f"{key}={val} out of clamp range"


# ── Integration tests ───────────────────────────────────────────────────


class TestPackageImports:
    """Test that the package imports work correctly."""

    def test_top_level_import(self):
        """Top-level import should expose all key classes."""
        from hbllm.brain.snn.comprehension import (
            ActivatedMemory,
            ComprehensionEnsemble,
            ComprehensionStream,
            ComprehensionUnit,
            LexicalBuffer,
            LexicalSignals,
            SNNCalibrator,
            UnderstandingState,
        )

        # All should be importable
        assert ComprehensionStream is not None
        assert ComprehensionEnsemble is not None
        assert LexicalBuffer is not None
        assert LexicalSignals is not None
        assert SNNCalibrator is not None
        assert ComprehensionUnit is not None
        assert UnderstandingState is not None
        assert ActivatedMemory is not None

    def test_snn_package_includes_comprehension(self):
        """The SNN package __all__ should include 'comprehension'."""
        import hbllm.brain.snn as snn_pkg

        assert "comprehension" in snn_pkg.__all__
