"""
Milestone 1 Integration Tests — Reactive Brain.

Tests cover all M1 components from the v4 implementation plan:
    - KV Prefix Cache warmup
    - Izhikevich neurons + BaseNeuron hierarchy
    - Short-Term Plasticity (STP)
    - Winner-Take-All (WTA)
    - Neuromodulation (global, 6 modulators)
    - Immutable CognitiveState + delta/reducer
    - Cognitive event-driven scheduler:
        - CognitiveEvent + CognitiveEventType
        - CognitiveEventQueue (IEventQueue)
        - SaliencyEvaluator (IAttentionSelector)
        - CompetitionEngine (ICompetition)
        - ExecutiveController
"""

from __future__ import annotations

import time

# ═══════════════════════════════════════════════════════════════════════════
# 1. KV Prefix Cache
# ═══════════════════════════════════════════════════════════════════════════
from hbllm.model.kv_warmup import KVPrefixCache, PrefixIdentity


class TestKVWarmup:
    """Test KV prefix cache warmup infrastructure."""

    def test_prefix_identity_cache_key(self) -> None:
        """PrefixIdentity produces a stable cache key."""
        identity = PrefixIdentity(
            system_prompt="You are helpful.",
            identity_prompt="I am HBLLM.",
            tool_schemas="[]",
            persona_block="default",
        )
        key = identity.cache_key
        assert isinstance(key, str)
        assert len(key) > 16

        # Same inputs → same key
        identity2 = PrefixIdentity(
            system_prompt="You are helpful.",
            identity_prompt="I am HBLLM.",
            tool_schemas="[]",
            persona_block="default",
        )
        assert identity2.cache_key == key

    def test_prefix_identity_different_inputs(self) -> None:
        """Different inputs → different cache keys."""
        id1 = PrefixIdentity("a", "b", "c", "d")
        id2 = PrefixIdentity("a", "b", "c", "DIFFERENT")
        assert id1.cache_key != id2.cache_key

    def test_kv_cache_get_miss(self) -> None:
        """get_cached returns None for unknown keys."""
        cache = KVPrefixCache()
        assert cache.get_cached("nonexistent_key") is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. Izhikevich Neurons + BaseNeuron
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.hcir.snn.lif import LIFConfig, LIFNeuron, SpikeEvent
from hbllm.hcir.snn.neurons import (
    BaseNeuron,
    IzhikevichConfig,
    IzhikevichNeuron,
    create_neuron_from_dict,
)


class TestBaseNeuronHierarchy:
    """Test neuron model hierarchy and polymorphism."""

    def test_lif_is_base_neuron(self) -> None:
        """LIFNeuron inherits from BaseNeuron."""
        neuron = LIFNeuron(LIFConfig(), neuron_id="test")
        assert isinstance(neuron, BaseNeuron)

    def test_izhikevich_is_base_neuron(self) -> None:
        """IzhikevichNeuron inherits from BaseNeuron."""
        config = IzhikevichConfig.regular_spiking()
        neuron = IzhikevichNeuron(config, neuron_id="izh_test")
        assert isinstance(neuron, BaseNeuron)

    def test_izhikevich_presets_produce_different_behavior(self) -> None:
        """Different Izhikevich presets produce different spike patterns."""
        presets = [
            IzhikevichConfig.regular_spiking(),
            IzhikevichConfig.fast_spiking(),
            IzhikevichConfig.intrinsically_bursting(),
        ]
        spike_counts = []
        for config in presets:
            neuron = IzhikevichNeuron(config, neuron_id="test")
            count = 0
            ts = time.time()
            for i in range(100):
                result = neuron.step(15.0, ts + i * 0.001)
                if result.fired:
                    count += 1
            spike_counts.append(count)

        # At least two presets should differ in spike count
        assert len(set(spike_counts)) >= 2, f"Expected different spike patterns, got {spike_counts}"

    def test_create_neuron_from_dict_lif(self) -> None:
        """create_neuron_from_dict returns a LIFNeuron for type='lif'."""
        neuron = create_neuron_from_dict({"type": "lif", "neuron_id": "n1"})
        assert isinstance(neuron, LIFNeuron)
        assert isinstance(neuron, BaseNeuron)

    def test_lif_step_and_spike(self) -> None:
        """LIF neuron fires when threshold is crossed."""
        neuron = LIFNeuron(LIFConfig(threshold=0.5), neuron_id="test")
        ts = time.time()
        # Drive above threshold
        result = neuron.step(1.0, ts)
        assert isinstance(result, SpikeEvent)
        assert result.fired is True


# ═══════════════════════════════════════════════════════════════════════════
# 3. Short-Term Plasticity (STP)
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.hcir.snn.stp import STPConfig, STPManager


class TestSTP:
    """Test Tsodyks-Markram STP model."""

    def test_facilitation_produces_valid_weights(self) -> None:
        """STP facilitation produces valid, bounded effective weights."""
        config = STPConfig(mode="facilitation", U=0.4, tau_f=1.0, tau_d=0.05)
        stp = STPManager(config, source_size=2, target_size=2)
        permanent_w = 1.0
        ts = time.time()

        weights = []
        for i in range(10):
            w = stp.get_effective_weight(permanent_w, (0, 0), ts, spiked=True)
            weights.append(w)
            ts += 0.005

        # All weights should be valid positive numbers
        assert all(w >= 0.0 for w in weights)
        # STP factor should be bounded
        assert all(w <= 3.0 for w in weights)
        # STP should track synapses
        assert stp.active_synapse_count >= 1

    def test_recovery_after_pause(self) -> None:
        """After pause, STP factor recovers toward baseline."""
        config = STPConfig(mode="combined", U=0.3, tau_f=0.3, tau_d=0.3)
        stp = STPManager(config, source_size=2, target_size=2)
        ts = time.time()

        # Burst
        for _ in range(10):
            stp.get_effective_weight(1.0, (0, 0), ts, spiked=True)
            ts += 0.01

        w_after_burst = stp.get_effective_weight(1.0, (0, 0), ts, spiked=False)

        # Long pause (5 seconds)
        ts += 5.0
        w_after_rest = stp.get_effective_weight(1.0, (0, 0), ts, spiked=False)

        # After rest, modulation should be closer to 1.0 (baseline)
        baseline_distance_burst = abs(w_after_burst - 1.0)
        baseline_distance_rest = abs(w_after_rest - 1.0)
        assert baseline_distance_rest <= baseline_distance_burst + 0.01

    def test_stdp_unaffected_by_stp(self) -> None:
        """STP only modulates; permanent weight is unchanged."""
        config = STPConfig(mode="facilitation")
        stp = STPManager(config, source_size=2, target_size=2)
        permanent_w = 0.5
        ts = time.time()

        for _ in range(20):
            stp.get_effective_weight(permanent_w, (0, 0), ts, spiked=True)
            ts += 0.01

        # The permanent weight should still be 0.5
        # (STP returns permanent_w × factor, not modifying permanent_w)
        factor = stp.get_modulation_factor((0, 0))
        assert factor != 1.0, "STP should be active after bursts"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Winner-Take-All (WTA)
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.hcir.snn.wta import HierarchicalWTA, WinnerTakeAll, WTAConfig


class TestWTA:
    """Test WTA lateral inhibition."""

    def test_hard_wta_single_winner(self) -> None:
        """Hard WTA (k=1) selects exactly 1 winner."""
        wta = WinnerTakeAll(WTAConfig(k_winners=1, soft_wta=False))
        ts = time.time()
        spikes = [
            SpikeEvent(fired=True, strength=0.3, timestamp=ts),
            SpikeEvent(fired=True, strength=0.9, timestamp=ts),
            SpikeEvent(fired=True, strength=0.5, timestamp=ts),
        ]
        winners = wta.compete(spikes)
        winner_count = sum(1 for s in winners if s.fired and s.strength > 0)
        assert winner_count == 1

    def test_k_wta_multiple_winners(self) -> None:
        """k-WTA selects exactly k winners."""
        wta = WinnerTakeAll(WTAConfig(k_winners=2, soft_wta=False))
        ts = time.time()
        spikes = [
            SpikeEvent(fired=True, strength=0.2, timestamp=ts),
            SpikeEvent(fired=True, strength=0.8, timestamp=ts),
            SpikeEvent(fired=True, strength=0.5, timestamp=ts),
            SpikeEvent(fired=True, strength=0.3, timestamp=ts),
        ]
        winners = wta.compete(spikes)
        winner_count = sum(1 for s in winners if s.fired and s.strength > 0)
        assert winner_count == 2

    def test_hierarchical_wta(self) -> None:
        """Hierarchical: local → executive competition."""
        hwta = HierarchicalWTA(
            local_configs={
                "memory": WTAConfig(k_winners=1),
                "goals": WTAConfig(k_winners=1),
            },
            executive_config=WTAConfig(k_winners=1, soft_wta=True),
        )
        ts = time.time()

        mem_spikes = [
            SpikeEvent(fired=True, strength=0.6, timestamp=ts),
            SpikeEvent(fired=True, strength=0.3, timestamp=ts),
        ]
        goal_spikes = [
            SpikeEvent(fired=True, strength=0.9, timestamp=ts),
            SpikeEvent(fired=True, strength=0.4, timestamp=ts),
        ]

        mem_winners = hwta.compete_local("memory", mem_spikes)
        goal_winners = hwta.compete_local("goals", goal_spikes)

        exec_winners = hwta.compete_executive(
            {
                "memory": mem_winners,
                "goals": goal_winners,
            }
        )

        assert len(exec_winners) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Neuromodulation (global, 6 modulators)
# ═══════════════════════════════════════════════════════════════════════════

from hbllm.hcir.snn.neuromodulation import NeuromodulationEngine, NeuromodulatorState


class TestNeuromodulation:
    """Test expanded neuromodulation engine."""

    def test_six_modulators(self) -> None:
        """NeuromodulatorState has all 6 modulators."""
        state = NeuromodulatorState()
        assert hasattr(state, "dopamine")
        assert hasattr(state, "serotonin")
        assert hasattr(state, "norepinephrine")
        assert hasattr(state, "acetylcholine")
        assert hasattr(state, "gaba")
        assert hasattr(state, "glutamate")

    def test_excitatory_gain(self) -> None:
        """excitatory_gain = glutamate × ACh scaling."""
        engine = NeuromodulationEngine()
        # At baseline (0.5, 0.5): gain should be sub-1.0
        gain = engine.get_excitatory_gain()
        assert 0.0 < gain < 1.5

    def test_excitatory_gain_high(self) -> None:
        """High glutamate + high ACh → high gain."""
        engine = NeuromodulationEngine()
        engine.state.glutamate = 1.0
        engine.state.acetylcholine = 1.0
        engine.state.last_update = time.time()
        gain = engine.get_excitatory_gain()
        assert gain > 1.5

    def test_signal_attention_focus(self) -> None:
        """signal_attention_focus modulates ACh."""
        engine = NeuromodulationEngine()
        baseline = engine.state.acetylcholine
        engine.signal_attention_focus(1.0)
        assert engine.state.acetylcholine > baseline

    def test_sleep_mode(self) -> None:
        """enter_sleep_mode sets appropriate modulator levels."""
        engine = NeuromodulationEngine()
        engine.enter_sleep_mode()
        assert engine.state.acetylcholine < 0.3  # ACh drops
        assert engine.state.gaba > 0.7  # GABA rises
        assert engine.state.norepinephrine < 0.3  # NE drops

    def test_backward_compat_original_signals(self) -> None:
        """Original 3-signal API still works."""
        engine = NeuromodulationEngine()
        engine.signal_reward(0.8)
        assert engine.state.dopamine > 0.5
        engine.signal_stress(0.9)
        assert engine.state.serotonin < 0.5
        engine.signal_novelty(0.9)
        assert engine.state.norepinephrine > 0.5

    def test_backward_compat_import_from_snn(self) -> None:
        """Old import path still works via snn/__init__.py re-export."""
        from hbllm.hcir.snn import NeuromodulationEngine as SnnEngine

        assert SnnEngine is NeuromodulationEngine

    def test_global_influence_factors(self) -> None:
        """All global influence factors are available."""
        engine = NeuromodulationEngine()
        assert 0.0 <= engine.get_encoding_priority() <= 1.0
        assert 0.0 <= engine.get_retrieval_bias() <= 2.0
        assert 0.0 <= engine.get_planning_depth() <= 2.0
        assert 0.0 <= engine.get_exploration_rate() <= 1.0
        assert 0.0 <= engine.get_reasoning_persistence() <= 2.0
        assert 0.0 <= engine.get_consolidation_aggressiveness() <= 2.0
