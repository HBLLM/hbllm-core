"""Tests for SNN improvements — neuromodulation, R-STDP, inhibition, homeostasis.

Covers:
    Phase A: NeuromodulationEngine
    Phase B: Reward-modulated STDP (R-STDP)
    Phase C: Inhibitory neurons + homeostatic plasticity
    Phase D: PlasticWeightMatrix consolidation methods
    Phase E: Wiring integration (import smoke tests)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from hbllm.brain.snn.lif import LIFConfig, LIFNeuron
from hbllm.brain.snn.network import LayerProjection, NeuronLayer, SpikingNetwork
from hbllm.brain.snn.neuromodulation import NeuromodulationEngine, NeuromodulatorState
from hbllm.brain.snn.plasticity import PlasticWeightMatrix, STDPRule, SynapticConnection

# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Neuromodulation
# ═══════════════════════════════════════════════════════════════════════════


class TestNeuromodulatorState:
    def test_default_baseline(self):
        state = NeuromodulatorState()
        assert state.dopamine == 0.5
        assert state.serotonin == 0.5
        assert state.norepinephrine == 0.5

    def test_serialization_roundtrip(self):
        state = NeuromodulatorState(dopamine=0.8, serotonin=0.3, norepinephrine=0.9)
        d = state.to_dict()
        restored = NeuromodulatorState.from_dict(d)
        assert restored.dopamine == pytest.approx(0.8, abs=0.001)
        assert restored.serotonin == pytest.approx(0.3, abs=0.001)
        assert restored.norepinephrine == pytest.approx(0.9, abs=0.001)


class TestNeuromodulationEngine:
    def test_reward_raises_dopamine(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        engine.signal_reward(1.0)
        assert engine.state.dopamine > 0.5

    def test_punishment_lowers_dopamine(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        engine.signal_reward(-0.5)
        assert engine.state.dopamine < 0.5

    def test_stress_lowers_serotonin(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        engine.signal_stress(0.9)
        assert engine.state.serotonin < 0.5

    def test_novelty_raises_norepinephrine(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        engine.signal_novelty(0.9)
        assert engine.state.norepinephrine > 0.5

    def test_learning_rate_factor_range(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        engine.state.dopamine = 0.0
        engine.state.last_update = time.time()
        assert engine.get_learning_rate_factor() == pytest.approx(0.5, abs=0.05)

        engine.state.dopamine = 1.0
        engine.state.last_update = time.time()
        assert engine.get_learning_rate_factor() == pytest.approx(2.0, abs=0.05)

    def test_threshold_factor_range(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        engine.state.serotonin = 0.0
        engine.state.last_update = time.time()
        assert engine.get_threshold_factor() == pytest.approx(1.2, abs=0.05)

        engine.state.serotonin = 1.0
        engine.state.last_update = time.time()
        assert engine.get_threshold_factor() == pytest.approx(0.8, abs=0.05)

    def test_attention_factor_range(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        engine.state.norepinephrine = 0.0
        engine.state.last_update = time.time()
        assert engine.get_attention_factor() == pytest.approx(0.7, abs=0.05)

        engine.state.norepinephrine = 1.0
        engine.state.last_update = time.time()
        assert engine.get_attention_factor() == pytest.approx(1.5, abs=0.05)

    def test_reset(self):
        engine = NeuromodulationEngine()
        engine.signal_reward(1.0)
        engine.reset()
        assert engine.state.dopamine == 0.5

    def test_stats(self):
        engine = NeuromodulationEngine()
        engine.signal_reward(0.5)
        s = engine.stats()
        assert "state" in s
        assert "factors" in s
        assert s["signal_count"] == 1

    def test_signal_from_user_model(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        mock_um = MagicMock()
        mock_model = MagicMock()
        mock_model.stress_level = 0.9
        mock_model.engagement_level = 0.8
        mock_um.get_model.return_value = mock_model
        engine.signal_from_user_model(mock_um, "t1")
        # High stress → low serotonin
        assert engine.state.serotonin < 0.5

    def test_clamping(self):
        engine = NeuromodulationEngine(decay_half_life=1000)
        for _ in range(20):
            engine.signal_reward(1.0)
        assert engine.state.dopamine <= 1.0

        for _ in range(40):
            engine.signal_reward(-1.0)
        assert engine.state.dopamine >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Reward-Modulated STDP
# ═══════════════════════════════════════════════════════════════════════════


class TestRewardModulatedSTDP:
    def test_reward_amplifies_potentiation(self):
        """High reward should amplify weight increase."""
        rule = STDPRule(learning_rate=0.1, time_constant=1.0)
        conn = SynapticConnection(weight=0.5, base_weight=0.5)

        # Causal: pre fires at t=0, post fires at t=0.1
        conn.last_pre_time = 1.0
        conn.last_post_time = 1.1

        delta_normal = rule.update(
            conn, pre_active=True, post_fired=True, timestamp=1.1, reward_signal=1.0
        )

        # Reset
        conn.weight = 0.5
        conn.last_pre_time = 1.0
        conn.last_post_time = 1.1

        delta_rewarded = rule.update(
            conn, pre_active=True, post_fired=True, timestamp=1.1, reward_signal=2.0
        )

        assert abs(delta_rewarded) > abs(delta_normal)

    def test_zero_reward_blocks_learning(self):
        """Zero reward should prevent any weight change."""
        rule = STDPRule(learning_rate=0.1, time_constant=1.0)
        conn = SynapticConnection(weight=0.5, base_weight=0.5)

        conn.last_pre_time = 1.0
        conn.last_post_time = 1.1

        delta = rule.update(
            conn, pre_active=True, post_fired=True, timestamp=1.1, reward_signal=0.0
        )
        assert delta == 0.0

    def test_negative_reward_inverts_learning(self):
        """Negative reward should cause depression instead of potentiation."""
        rule = STDPRule(learning_rate=0.1, time_constant=1.0, w_min=-1.0)
        conn = SynapticConnection(weight=0.5, base_weight=0.5)

        # Causal timing → normally potentiation
        conn.last_pre_time = 1.0
        conn.last_post_time = 1.1

        delta = rule.update(
            conn, pre_active=True, post_fired=True, timestamp=1.1, reward_signal=-1.0
        )
        # With negative reward, causal should become depression
        assert delta < 0.0

    def test_backward_compatible_default(self):
        """Default reward_signal=1.0 should behave like old STDP."""
        rule = STDPRule(learning_rate=0.1)
        conn = SynapticConnection(weight=0.5, base_weight=0.5)

        conn.last_pre_time = 1.0
        conn.last_post_time = 1.1

        # Default (no reward_signal) should work
        delta = rule.update(conn, pre_active=True, post_fired=True, timestamp=1.1)
        assert delta != 0.0


class TestPlasticWeightMatrixNeuromodulation:
    def test_with_neuromodulator(self):
        """PlasticWeightMatrix should use neuromodulator for reward signal."""
        neuromod = NeuromodulationEngine(decay_half_life=1000)
        neuromod.signal_reward(1.0)  # Boost dopamine

        weights = {"channel_a": {"signal_1": 0.5}}
        rule = STDPRule(learning_rate=0.1)
        matrix = PlasticWeightMatrix(weights, rule, neuromodulator=neuromod)

        # _get_reward_signal should return elevated factor
        assert matrix._get_reward_signal() > 1.0

    def test_without_neuromodulator(self):
        """Without neuromodulator, reward signal defaults to 1.0."""
        weights = {"channel_a": {"signal_1": 0.5}}
        rule = STDPRule(learning_rate=0.1)
        matrix = PlasticWeightMatrix(weights, rule)

        assert matrix._get_reward_signal() == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Phase C: Inhibitory Neurons + Homeostasis
# ═══════════════════════════════════════════════════════════════════════════


class TestInhibitoryNeurons:
    def test_inhibitory_flag_on_config(self):
        config = LIFConfig(is_inhibitory=True)
        assert config.is_inhibitory is True

    def test_excitatory_default(self):
        config = LIFConfig()
        assert config.is_inhibitory is False

    def test_inhibitory_layer(self):
        config = LIFConfig(threshold=0.5, is_inhibitory=True)
        layer = NeuronLayer("inhib", 3, config)
        assert layer.is_inhibitory is True

    def test_inhibitory_projection_negates_current(self):
        """Inhibitory source should produce negative currents."""
        proj = LayerProjection("src", "tgt", 2, 2, initial_weights=[[1.0, 0.0], [0.0, 1.0]])

        from hbllm.brain.snn.lif import SpikeEvent

        spikes = [
            SpikeEvent(fired=True, strength=1.0, timestamp=1.0),
            SpikeEvent(fired=False, strength=0.0, timestamp=1.0),
        ]

        # Excitatory: positive current
        excit = proj.project(spikes, 1.0, source_is_inhibitory=False)
        assert excit[0] > 0

        # Inhibitory: negative current
        inhib = proj.project(spikes, 1.0, source_is_inhibitory=True)
        assert inhib[0] < 0

    def test_network_with_inhibitory_layer(self):
        """Network should correctly handle inhibitory layers."""
        net = SpikingNetwork("test_inhib")
        net.add_layer(NeuronLayer("input", 2, LIFConfig(threshold=0.3)))
        net.add_layer(NeuronLayer("inhib", 2, LIFConfig(threshold=0.3, is_inhibitory=True)))
        net.add_layer(NeuronLayer("output", 2, LIFConfig(threshold=0.5)))

        net.connect("input", "inhib")
        net.connect("inhib", "output")

        # Strong input should fire input layer, which fires inhibitory layer
        t = time.time()
        results = net.step({"input": [1.0, 1.0]}, t, learn=False)
        assert "input" in results
        assert "inhib" in results
        assert "output" in results


class TestHomeostaticPlasticity:
    def test_homeostasis_disabled_by_default(self):
        config = LIFConfig()
        assert config.target_firing_rate == 0.0

    def test_homeostasis_raises_threshold_when_firing_too_much(self):
        """If neuron fires too often, threshold should increase."""
        config = LIFConfig(
            threshold=0.5,
            target_firing_rate=0.1,  # Want 10% firing rate
            adaptation_rate=0.5,  # Aggressive for test speed
        )
        neuron = LIFNeuron(config, "test_homeo")

        # Make it fire every step (way above 10%)
        t = time.time()
        for i in range(60):
            neuron.step(1.0, t + i * 0.01)

        # Threshold should have increased
        assert neuron._effective_threshold > config.threshold

    def test_homeostasis_lowers_threshold_when_not_firing(self):
        """If neuron never fires, threshold should decrease."""
        config = LIFConfig(
            threshold=1.0,
            target_firing_rate=0.5,  # Want 50% firing rate
            adaptation_rate=0.5,  # Aggressive for test speed
        )
        neuron = LIFNeuron(config, "test_homeo")

        # Give sub-threshold input (never fires)
        t = time.time()
        for i in range(60):
            neuron.step(0.01, t + i * 0.01)

        # Threshold should have decreased
        assert neuron._effective_threshold < config.threshold

    def test_homeostasis_clamped(self):
        """Threshold should not exceed 2× or fall below 0.5× of base."""
        config = LIFConfig(
            threshold=1.0,
            target_firing_rate=0.1,
            adaptation_rate=1.0,  # Very aggressive
        )
        neuron = LIFNeuron(config, "test_clamp")

        t = time.time()
        for i in range(200):
            neuron.step(10.0, t + i * 0.01)

        assert neuron._effective_threshold <= config.threshold * 2.0
        assert neuron._effective_threshold >= config.threshold * 0.5

    def test_reset_restores_homeostatic_state(self):
        config = LIFConfig(threshold=1.0, target_firing_rate=0.1, adaptation_rate=0.5)
        neuron = LIFNeuron(config, "test_reset")
        t = time.time()
        for i in range(60):
            neuron.step(10.0, t + i * 0.01)
        neuron.reset_state()
        assert neuron._effective_threshold == config.threshold
        assert neuron._firing_history == []


# ═══════════════════════════════════════════════════════════════════════════
# Phase D: Sleep Consolidation Methods
# ═══════════════════════════════════════════════════════════════════════════


class TestPlasticWeightMatrixConsolidation:
    def _make_matrix(self):
        weights = {"ch_a": {"sig_1": 0.5, "sig_2": 0.3}, "ch_b": {"sig_1": 0.4}}
        rule = STDPRule(learning_rate=0.1)
        return PlasticWeightMatrix(weights, rule)

    def test_prune_weak_connections(self):
        matrix = self._make_matrix()
        # Manually weaken a connection
        matrix._connections["ch_a"]["sig_2"].weight = 0.01
        pruned = matrix.prune_weak_connections(threshold=0.05)
        assert pruned >= 1
        # Pruned connection should be reset to base
        assert matrix._connections["ch_a"]["sig_2"].weight == 0.3

    def test_prune_no_weak(self):
        matrix = self._make_matrix()
        pruned = matrix.prune_weak_connections(threshold=0.01)
        assert pruned == 0

    def test_consolidate_strong_connections(self):
        matrix = self._make_matrix()
        # Simulate learning: change weight and bump update_count
        conn = matrix._connections["ch_a"]["sig_1"]
        conn.weight = 0.9  # Learned away from base 0.5
        conn.update_count = 10
        original_base = conn.base_weight

        consolidated = matrix.consolidate(strengthen_factor=0.5)
        assert consolidated >= 1
        # Base weight should move toward learned weight
        assert conn.base_weight > original_base

    def test_consolidate_ignores_unlearned(self):
        matrix = self._make_matrix()
        consolidated = matrix.consolidate(strengthen_factor=0.5)
        assert consolidated == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase E: Import Smoke Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSNNImports:
    def test_import_neuromodulation_from_package(self):
        from hbllm.brain.snn import NeuromodulationEngine, NeuromodulatorState

        assert NeuromodulationEngine is not None
        assert NeuromodulatorState is not None

    def test_import_inhibitory_config(self):
        from hbllm.brain.snn import LIFConfig

        cfg = LIFConfig(is_inhibitory=True, target_firing_rate=0.1)
        assert cfg.is_inhibitory is True
        assert cfg.target_firing_rate == 0.1

    def test_plastic_weight_matrix_accepts_neuromodulator(self):
        nm = NeuromodulationEngine()
        weights = {"a": {"b": 0.5}}
        rule = STDPRule()
        matrix = PlasticWeightMatrix(weights, rule, neuromodulator=nm)
        assert matrix._neuromodulator is nm
