"""
Milestone 3: Predictive Brain — Unit Tests.

Validates the Markov prediction engine, dendritic predictive coding
neurons, population encoding/decoding, and predictive memory loading.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# Markov Predictor Tests
# ═══════════════════════════════════════════════════════════════════════════
from hbllm.brain.prediction import (
    CognitivePredictors,
    MarkovPredictor,
)


class TestMarkovPredictor:
    """Validate order-N Markov prediction with online learning."""

    def test_learn_and_predict_simple_pattern(self) -> None:
        """Repeated A→B→C pattern should be predicted."""
        pred = MarkovPredictor(order=2)
        pattern = ["A", "B", "C"] * 20
        for state in pattern:
            pred.train(state)

        # After observing the pattern, it should predict well
        # History ends with C, so next should be A
        dist = pred.predict()
        assert "A" in dist
        assert dist["A"] > dist.get("B", 0)

    def test_accuracy_above_threshold(self) -> None:
        """After 50+ observations of a pattern, accuracy should be >70%."""
        pred = MarkovPredictor(order=2)
        pattern = ["coding", "testing", "debugging"] * 30

        # Train on 60 observations
        for state in pattern[:60]:
            pred.train(state)

        # Evaluate on next 30
        for i, state in enumerate(pattern[60:]):
            pred.evaluate_prediction(state)
            pred.train(state)

        assert pred.accuracy >= 0.7, f"Accuracy {pred.accuracy:.2f} below 0.70"

    def test_entropy_decreases_with_data(self) -> None:
        """Entropy should decrease as the predictor learns a pattern."""
        pred = MarkovPredictor(order=2)
        pred.train("X")
        pred.train("Y")
        initial_entropy = pred.entropy()

        # Train on a deterministic pattern
        for _ in range(50):
            pred.train("X")
            pred.train("Y")

        final_entropy = pred.entropy()
        assert final_entropy <= initial_entropy + 0.1

    def test_fallback_to_lower_order(self) -> None:
        """If order-3 context has no match, fall back gracefully."""
        pred = MarkovPredictor(order=3)
        pred.train("A")
        pred.train("B")
        dist = pred.predict()
        # Should return a valid distribution even with little data
        assert len(dist) > 0
        assert abs(sum(dist.values()) - 1.0) < 0.01

    def test_predict_top_k(self) -> None:
        """top_k returns sorted predictions."""
        pred = MarkovPredictor(order=1)
        for s in ["high"] * 10 + ["medium"] * 5 + ["low"] * 2:
            pred.train(s)

        top = pred.predict_top_k(2)
        assert len(top) == 2
        assert top[0][1] >= top[1][1]

    def test_decay_reduces_old_counts(self) -> None:
        """With decay, old patterns lose influence."""
        pred = MarkovPredictor(order=1, decay_rate=0.05)
        for _ in range(20):
            pred.train("old")

        for _ in range(20):
            pred.train("new")

        dist = pred.predict()
        # "new" should have higher probability than "old" due to decay
        assert dist.get("new", 0) > dist.get("old", 0)

    def test_reset_clears_state(self) -> None:
        """Reset should clear all learned data."""
        pred = MarkovPredictor(order=2)
        pred.train("A")
        pred.train("B")
        pred.reset()
        assert pred.stats()["observations"] == 0
        assert len(pred.predict()) == 0


class TestCognitivePredictors:
    """Validate the multi-domain predictor bundle."""

    def test_all_seven_predictors_exist(self) -> None:
        preds = CognitivePredictors()
        assert isinstance(preds.query, MarkovPredictor)
        assert isinstance(preds.goal, MarkovPredictor)
        assert isinstance(preds.memory, MarkovPredictor)
        assert isinstance(preds.tool, MarkovPredictor)
        assert isinstance(preds.emotion, MarkovPredictor)
        assert isinstance(preds.attention, MarkovPredictor)
        assert isinstance(preds.action, MarkovPredictor)

    def test_predictors_are_independent(self) -> None:
        preds = CognitivePredictors()
        preds.query.train("coding")
        preds.tool.train("git")

        assert preds.query.stats()["observations"] == 1
        assert preds.tool.stats()["observations"] == 1
        assert preds.memory.stats()["observations"] == 0

    @pytest.mark.asyncio
    async def test_ipredictor_interface(self) -> None:
        """MarkovPredictor should implement IPredictor."""
        pred = MarkovPredictor(order=2)
        await pred.observe("coding")
        await pred.observe("testing")
        dist = await pred.predict_next("testing")
        assert isinstance(dist, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Dendritic Neuron Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.brain.snn.dendrite import DendriticConfig, DendriticNeuron
from hbllm.brain.snn.neurons import BaseNeuron, SpikeEvent


class TestDendriticNeuron:
    """Validate predictive coding via match/mismatch detection."""

    def test_is_base_neuron(self) -> None:
        """DendriticNeuron inherits BaseNeuron."""
        neuron = DendriticNeuron(DendriticConfig(), "dn_0")
        assert isinstance(neuron, BaseNeuron)

    def test_match_suppresses_firing(self) -> None:
        """When prediction matches evidence, neuron should rarely fire."""
        config = DendriticConfig(threshold=0.5, match_suppression=0.9)
        neuron = DendriticNeuron(config, "match_test")
        ts = time.time()

        fire_count = 0
        for i in range(50):
            # Basal ≈ apical → match → suppressed
            spike = neuron.step_dual(basal=3.0, apical=3.0, timestamp=ts)
            if spike.fired:
                fire_count += 1
            ts += 0.01

        assert fire_count <= 25, f"Match suppression failed: {fire_count}/50 fired"

    def test_mismatch_produces_prediction_error(self) -> None:
        """When prediction mismatches evidence, neuron should fire strongly."""
        config = DendriticConfig(threshold=0.3, mismatch_gain=3.0)
        neuron = DendriticNeuron(config, "mismatch_test")
        ts = time.time()

        fire_count = 0
        for i in range(20):
            # Basal ≠ apical → mismatch → prediction error
            spike = neuron.step_dual(basal=8.0, apical=1.0, timestamp=ts)
            if spike.fired:
                fire_count += 1
            ts += 0.01

        assert fire_count > 0, "Mismatch should produce at least some spikes"

    def test_prediction_error_strength_encodes_mismatch(self) -> None:
        """Spike strength should encode the magnitude of prediction error."""
        config = DendriticConfig(threshold=0.1, mismatch_gain=5.0)
        neuron = DendriticNeuron(config, "strength_test")
        ts = time.time()

        # Large mismatch
        spike = neuron.step_dual(basal=10.0, apical=0.0, timestamp=ts)
        if spike.fired:
            large_strength = spike.strength
        else:
            large_strength = 0.0

        neuron.reset()
        ts += 1.0

        # Small mismatch
        spike = neuron.step_dual(basal=5.0, apical=4.0, timestamp=ts)
        small_strength = spike.strength if spike.fired else 0.0

        # Large mismatch should produce stronger signal
        assert large_strength >= small_strength

    def test_backward_compatible_step(self) -> None:
        """step() should work as single-input (basal only)."""
        neuron = DendriticNeuron(DendriticConfig(), "compat_test")
        ts = time.time()
        spike = neuron.step(current=5.0, timestamp=ts)
        assert isinstance(spike, SpikeEvent)

    def test_average_match_score(self) -> None:
        """Diagnostic: average match score should be tracked."""
        neuron = DendriticNeuron(DendriticConfig(), "diag_test")
        ts = time.time()
        for _ in range(10):
            neuron.step_dual(basal=5.0, apical=5.0, timestamp=ts)
            ts += 0.01
        assert neuron.average_match_score > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Population Coding Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.brain.snn.population import (
    CognitiveStateEncoder,
    PopulationConfig,
    PopulationEncoder,
)


class TestPopulationEncoder:
    """Validate Gaussian tuning curve encoding/decoding."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode 0.73, decode → within ±0.05."""
        encoder = PopulationEncoder(PopulationConfig(num_neurons=32, tuning_width=0.05))
        target = 0.73
        encoded = encoder.encode(target)
        decoded = encoder.decode(encoded)
        assert abs(decoded - target) < 0.05, f"Decoded {decoded:.3f}, expected ~{target}"

    def test_encode_produces_correct_size(self) -> None:
        config = PopulationConfig(num_neurons=16)
        encoder = PopulationEncoder(config)
        activations = encoder.encode(0.5)
        assert len(activations) == 16

    def test_peak_near_preferred(self) -> None:
        """Activation peak should be near neurons with matching preferred value."""
        encoder = PopulationEncoder(PopulationConfig(num_neurons=10, tuning_width=0.05))
        activations = encoder.encode(0.5)
        peak_idx = activations.index(max(activations))
        # Preferred value of peak neuron should be close to 0.5
        # With 10 neurons spanning [0,1], preferred[5] ≈ 0.556
        assert 3 <= peak_idx <= 6

    def test_uncertainty_decreases_with_sharpness(self) -> None:
        """Narrower tuning curves → less uncertainty in decoding."""
        wide = PopulationEncoder(PopulationConfig(num_neurons=16, tuning_width=0.2))
        narrow = PopulationEncoder(PopulationConfig(num_neurons=16, tuning_width=0.05))

        target = 0.5
        _, wide_unc = wide.decode_with_uncertainty(wide.encode(target))
        _, narrow_unc = narrow.decode_with_uncertainty(narrow.encode(target))

        assert narrow_unc < wide_unc

    def test_boundary_values(self) -> None:
        """Encoding at boundaries should decode correctly."""
        encoder = PopulationEncoder(PopulationConfig(num_neurons=16, tuning_width=0.05))

        decoded_low = encoder.decode(encoder.encode(0.0))
        decoded_high = encoder.decode(encoder.encode(1.0))

        assert abs(decoded_low - 0.0) < 0.1
        assert abs(decoded_high - 1.0) < 0.1

    def test_decode_from_spikes(self) -> None:
        """Decode from SpikeEvent list."""
        encoder = PopulationEncoder(PopulationConfig(num_neurons=8))
        activations = encoder.encode(0.6)
        spikes = [SpikeEvent(fired=a > 0.1, strength=a, timestamp=0.0) for a in activations]
        decoded = encoder.decode_from_spikes(spikes)
        assert abs(decoded - 0.6) < 0.15


class TestCognitiveStateEncoder:
    """Validate multi-field state encoding."""

    def test_encode_decode_state(self) -> None:
        """Roundtrip encode/decode of a cognitive state dict."""
        encoder = CognitiveStateEncoder(neurons_per_field=16, tuning_width=0.08)
        state = {
            "urgency": 0.8,
            "curiosity": 0.3,
            "confidence": 0.9,
            "emotional_valence": 0.5,
            "cognitive_load": 0.4,
        }
        patterns = encoder.encode_state(state)
        decoded = encoder.decode_state(patterns)

        for field_name, expected in state.items():
            if field_name in decoded:
                assert abs(decoded[field_name] - expected) < 0.15, (
                    f"{field_name}: decoded {decoded[field_name]:.2f}, expected ~{expected}"
                )

    def test_total_neurons(self) -> None:
        encoder = CognitiveStateEncoder(neurons_per_field=16)
        assert encoder.total_neurons == 16 * len(CognitiveStateEncoder.FIELD_RANGES)

    def test_uncertainty_output(self) -> None:
        encoder = CognitiveStateEncoder(neurons_per_field=16)
        state = {"urgency": 0.5, "confidence": 0.7}
        patterns = encoder.encode_state(state)
        result = encoder.decode_with_uncertainty(patterns)
        assert "urgency" in result
        val, unc = result["urgency"]
        assert unc >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Predictive Loader Tests
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.memory.predictive_loader import PredictiveLoader


class TestPredictiveLoader:
    """Validate anticipatory memory pre-fetching."""

    @pytest.mark.asyncio
    async def test_prefetch_and_retrieve(self) -> None:
        """Prefetched domains should be retrievable."""

        async def mock_fetch(domain: str) -> list[Any]:
            return [f"result_for_{domain}_1", f"result_for_{domain}_2"]

        loader = PredictiveLoader(fetch_fn=mock_fetch, confidence_threshold=0.1)
        predictions = {"coding": 0.6, "testing": 0.3, "docs": 0.05}

        fetched = await loader.prefetch(predictions)
        assert fetched == 2  # coding + testing (docs below threshold)

        results = loader.get("coding")
        assert results is not None
        assert len(results) == 2

        miss = loader.get("docs")
        assert miss is None

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self) -> None:
        """Hit rate should increase with repeated access."""

        async def mock_fetch(domain: str) -> list[Any]:
            return [f"{domain}_data"]

        loader = PredictiveLoader(fetch_fn=mock_fetch, confidence_threshold=0.1)
        await loader.prefetch({"coding": 0.8})

        # 3 hits
        for _ in range(3):
            loader.get("coding")

        # 1 miss
        loader.get("unknown")

        assert loader.hit_rate == 0.75  # 3/4

    @pytest.mark.asyncio
    async def test_max_cache_size_eviction(self) -> None:
        """Exceeding max_cache_size should evict entries."""

        async def mock_fetch(domain: str) -> list[Any]:
            return [domain]

        loader = PredictiveLoader(
            fetch_fn=mock_fetch,
            confidence_threshold=0.1,
            max_cache_size=2,
        )
        await loader.prefetch({"a": 0.8, "b": 0.7, "c": 0.6})

        stats = loader.stats()
        assert stats["cache_size"] <= 2

    @pytest.mark.asyncio
    async def test_no_fetch_without_function(self) -> None:
        """Without a fetch function, prefetch should be a no-op."""
        loader = PredictiveLoader(fetch_fn=None)
        fetched = await loader.prefetch({"coding": 0.9})
        assert fetched == 0
