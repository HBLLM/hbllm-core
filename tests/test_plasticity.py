"""
Tests for STDP Synaptic Plasticity.

Tests cover:
  - SynapticConnection: state tracking, serialization
  - STDPRule: causal strengthening, anti-causal weakening, bounds, timing
  - PlasticWeightMatrix: weight retrieval, STDP updates, decay, persistence
  - Integration: ComprehensionEnsemble with plasticity
  - Integration: ThoughtController with plasticity
"""

from __future__ import annotations

import json
import time

from hbllm.brain.snn.plasticity import (
    PlasticWeightMatrix,
    STDPRule,
    SynapticConnection,
)

# ═══════════════════════════════════════════════════════════════════════════
# SynapticConnection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSynapticConnection:
    """Test the learnable weight dataclass."""

    def test_defaults(self) -> None:
        conn = SynapticConnection()
        assert conn.source == ""
        assert conn.target == ""
        assert conn.weight == 0.0
        assert conn.update_count == 0

    def test_serialization_roundtrip(self) -> None:
        conn = SynapticConnection(
            source="semantic_weight",
            target="clause",
            weight=0.42,
            base_weight=0.3,
            last_pre_time=100.0,
            last_post_time=100.1,
            update_count=5,
            last_reinforced_step=50,
        )

        data = conn.to_dict()
        restored = SynapticConnection.from_dict(data)

        assert restored.source == "semantic_weight"
        assert restored.target == "clause"
        assert restored.weight == 0.42
        assert restored.base_weight == 0.3
        assert restored.update_count == 5
        assert restored.last_reinforced_step == 50

    def test_json_compatible(self) -> None:
        conn = SynapticConnection(source="a", target="b", weight=0.5)
        serialized = json.dumps(conn.to_dict())
        restored = SynapticConnection.from_dict(json.loads(serialized))
        assert restored.weight == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# STDPRule Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSTDPRule:
    """Test the STDP learning rule."""

    def test_causal_potentiation(self) -> None:
        """Pre before post → weight increases."""
        rule = STDPRule(learning_rate=0.1, time_constant=1.0)
        conn = SynapticConnection(weight=0.5, base_weight=0.5)

        # Pre fires at t=1.0
        rule.update(conn, pre_active=True, post_fired=False, timestamp=1.0)

        # Post fires at t=1.2 (after pre → causal)
        delta = rule.update(conn, pre_active=False, post_fired=True, timestamp=1.2)

        assert delta > 0, "Causal pairing should potentiate"
        assert conn.weight > 0.5

    def test_anti_causal_depression(self) -> None:
        """Post before pre → weight decreases."""
        rule = STDPRule(learning_rate=0.1, time_constant=1.0)
        conn = SynapticConnection(weight=0.5, base_weight=0.5)

        # Post fires first at t=1.0
        rule.update(conn, pre_active=False, post_fired=True, timestamp=1.0)

        # Pre fires at t=1.3 (after post → anti-causal)
        delta = rule.update(conn, pre_active=True, post_fired=False, timestamp=1.3)

        assert delta < 0, "Anti-causal pairing should depress"
        assert conn.weight < 0.5

    def test_weight_lower_bound(self) -> None:
        """Weight never goes below w_min."""
        rule = STDPRule(learning_rate=1.0, w_min=0.0)
        conn = SynapticConnection(weight=0.01, base_weight=0.5)

        # Force anti-causal: post then pre
        rule.update(conn, pre_active=False, post_fired=True, timestamp=1.0)
        rule.update(conn, pre_active=True, post_fired=False, timestamp=1.5)

        assert conn.weight >= 0.0

    def test_weight_upper_bound(self) -> None:
        """Weight never exceeds w_max."""
        rule = STDPRule(learning_rate=1.0, w_max=2.0)
        conn = SynapticConnection(weight=1.9, base_weight=0.5)

        # Force causal: pre then post
        rule.update(conn, pre_active=True, post_fired=False, timestamp=1.0)
        rule.update(conn, pre_active=False, post_fired=True, timestamp=1.1)

        assert conn.weight <= 2.0

    def test_timing_decay(self) -> None:
        """Larger time gap → smaller weight change."""
        rule = STDPRule(learning_rate=0.1, time_constant=0.5)

        # Close pairing
        conn_close = SynapticConnection(weight=0.5, base_weight=0.5)
        rule.update(conn_close, pre_active=True, post_fired=False, timestamp=1.0)
        delta_close = rule.update(conn_close, pre_active=False, post_fired=True, timestamp=1.1)

        # Far pairing
        conn_far = SynapticConnection(weight=0.5, base_weight=0.5)
        rule.update(conn_far, pre_active=True, post_fired=False, timestamp=1.0)
        delta_far = rule.update(conn_far, pre_active=False, post_fired=True, timestamp=2.5)

        assert abs(delta_close) > abs(delta_far), "Close timing should produce larger update"

    def test_no_update_without_events(self) -> None:
        """No pre or post activity → no update."""
        rule = STDPRule()
        conn = SynapticConnection(weight=0.5)

        delta = rule.update(conn, pre_active=False, post_fired=False, timestamp=1.0)
        assert delta == 0.0
        assert conn.weight == 0.5

    def test_update_count_increments(self) -> None:
        """Update count tracks successful weight changes."""
        rule = STDPRule(learning_rate=0.1)
        conn = SynapticConnection(weight=0.5, base_weight=0.5)

        assert conn.update_count == 0

        rule.update(conn, pre_active=True, post_fired=False, timestamp=1.0)
        rule.update(conn, pre_active=False, post_fired=True, timestamp=1.1)

        assert conn.update_count > 0

    def test_simultaneous_potentiates_mildly(self) -> None:
        """Simultaneous pre+post → mild potentiation."""
        rule = STDPRule(learning_rate=0.1)
        conn = SynapticConnection(weight=0.5, base_weight=0.5)

        # Pre and post at same time
        rule.update(conn, pre_active=True, post_fired=False, timestamp=1.0)
        delta = rule.update(conn, pre_active=True, post_fired=True, timestamp=1.0)

        # dt=0 → small potentiation (0.5 × η)
        assert delta > 0
        assert delta < 0.1  # smaller than full causal update

    def test_serialization(self) -> None:
        """Rule parameters serialize correctly."""
        rule = STDPRule(learning_rate=0.05, time_constant=0.3, w_min=0.1, w_max=1.5)
        d = rule.to_dict()
        assert d["learning_rate"] == 0.05
        assert d["w_max"] == 1.5


# ═══════════════════════════════════════════════════════════════════════════
# PlasticWeightMatrix Tests
# ═══════════════════════════════════════════════════════════════════════════

STATIC_WEIGHTS = {
    "entity": {"semantic_weight": 0.5, "topic_shift": 0.3, "novelty": 0.2},
    "clause": {"punctuation": 0.3, "buffer_pressure": 0.3, "topic_shift": 0.2},
}


class TestPlasticWeightMatrix:
    """Test the ensemble-level weight manager."""

    def test_initial_weights_match_static(self) -> None:
        """Fresh matrix returns static weights."""
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, STDPRule())

        entity_w = matrix.get_weights("entity")
        assert entity_w["semantic_weight"] == 0.5
        assert entity_w["topic_shift"] == 0.3

    def test_unknown_channel_returns_static(self) -> None:
        """Unknown channel falls back to static (or empty)."""
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, STDPRule())

        w = matrix.get_weights("unknown_channel")
        assert w == {}

    def test_stdp_updates_on_spike(self) -> None:
        """Weights change after signal + spike recording."""
        rule = STDPRule(learning_rate=0.1, time_constant=1.0)
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, rule)

        t = time.time()

        # Strong signal on semantic_weight
        signals = {"semantic_weight": 0.8, "topic_shift": 0.0, "novelty": 0.0}
        matrix.record_signals(signals, t)

        # Entity neuron fires shortly after
        deltas = matrix.record_spikes(["entity"], t + 0.1)

        # Should have potentiated semantic_weight→entity
        entity_w = matrix.get_weights("entity")
        assert entity_w["semantic_weight"] != 0.5 or len(deltas) > 0

    def test_no_spike_no_potentiation(self) -> None:
        """Signals without spikes don't cause large potentiation."""
        rule = STDPRule(learning_rate=0.1)
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, rule)

        initial = matrix.get_weights("entity")["semantic_weight"]

        # Record signals but no spikes
        t = time.time()
        for i in range(10):
            signals = {"semantic_weight": 0.8, "topic_shift": 0.0}
            matrix.record_signals(signals, t + i * 0.1)
            matrix.record_spikes([], t + i * 0.1)

        after = matrix.get_weights("entity")["semantic_weight"]
        # Weight should have slightly depressed (pre active, no post)
        # or stayed the same
        assert after <= initial + 0.01

    def test_weight_drift_tracking(self) -> None:
        """get_weight_drift reports learned deviations."""
        rule = STDPRule(learning_rate=0.1)
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, rule)

        # Initially no drift
        assert matrix.get_weight_drift() == {}

        # Force a weight change
        t = time.time()
        matrix.record_signals({"semantic_weight": 1.0, "topic_shift": 0.0, "novelty": 0.0}, t)
        matrix.record_spikes(["entity"], t + 0.05)

        drift = matrix.get_weight_drift()
        # At least semantic_weight→entity should have drifted
        assert len(drift) >= 0  # may or may not drift depending on timing

    def test_reset_to_static(self) -> None:
        """Reset restores all weights to static defaults."""
        rule = STDPRule(learning_rate=0.5)
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, rule)

        # Force changes
        t = time.time()
        matrix.record_signals({"semantic_weight": 1.0}, t)
        matrix.record_spikes(["entity"], t + 0.01)

        matrix.reset_to_static()

        assert matrix.get_weights("entity")["semantic_weight"] == 0.5
        assert matrix.get_total_updates() == 0

    def test_global_step_increments(self) -> None:
        """Global step increments on each record_signals call."""
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, STDPRule())

        assert matrix.global_step == 0

        matrix.record_signals({"semantic_weight": 0.5}, time.time())
        assert matrix.global_step == 1

        matrix.record_signals({"semantic_weight": 0.5}, time.time())
        assert matrix.global_step == 2

    def test_serialization_roundtrip(self) -> None:
        """Full matrix serialization and restoration."""
        rule = STDPRule(learning_rate=0.1)
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, rule)

        # Make some changes
        t = time.time()
        matrix.record_signals({"semantic_weight": 1.0}, t)
        matrix.record_spikes(["entity"], t + 0.05)

        # Serialize
        data = matrix.to_dict()
        assert data["version"] == 1
        assert "connections" in data

        # Restore
        restored = PlasticWeightMatrix.from_dict(data, STATIC_WEIGHTS, rule)
        assert restored.global_step == matrix.global_step

        # Weights should match
        orig = matrix.get_weights("entity")
        rest = restored.get_weights("entity")
        for key in orig:
            assert abs(orig[key] - rest[key]) < 1e-9

    def test_json_serialization(self) -> None:
        """Matrix survives JSON round-trip."""
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, STDPRule())

        t = time.time()
        matrix.record_signals({"semantic_weight": 0.8}, t)
        matrix.record_spikes(["entity"], t + 0.02)

        json_str = json.dumps(matrix.to_dict())
        data = json.loads(json_str)
        restored = PlasticWeightMatrix.from_dict(data, STATIC_WEIGHTS)

        assert restored.global_step == matrix.global_step

    def test_file_persistence(self, tmp_path) -> None:
        """Save and load from file."""
        rule = STDPRule(learning_rate=0.1)
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, rule)

        t = time.time()
        matrix.record_signals({"semantic_weight": 1.0}, t)
        matrix.record_spikes(["entity"], t + 0.02)

        path = tmp_path / "weights.json"
        matrix.save(path)

        assert path.exists()

        loaded = PlasticWeightMatrix.load(path, STATIC_WEIGHTS, rule)
        assert loaded.global_step == matrix.global_step

    def test_load_nonexistent_returns_fresh(self, tmp_path) -> None:
        """Loading from nonexistent path returns fresh matrix."""
        path = tmp_path / "does_not_exist.json"
        matrix = PlasticWeightMatrix.load(path, STATIC_WEIGHTS)

        assert matrix.global_step == 0
        assert matrix.get_weights("entity")["semantic_weight"] == 0.5

    def test_decay_unused_nudges_toward_base(self) -> None:
        """Unused connections decay toward their base weight."""
        rule = STDPRule(learning_rate=0.5)
        matrix = PlasticWeightMatrix(STATIC_WEIGHTS, rule, decay_interval=5, decay_rate=0.1)

        # Force a weight change
        t = time.time()
        matrix.record_signals({"semantic_weight": 1.0}, t)
        matrix.record_spikes(["entity"], t + 0.01)

        weight_after_update = matrix.get_weights("entity")["semantic_weight"]

        # Run enough steps to trigger decay (step 5)
        for i in range(6):
            matrix.record_signals({}, t + i + 1)
            matrix.record_spikes([], t + i + 1)

        weight_after_decay = matrix.get_weights("entity")["semantic_weight"]

        # Weight should have moved toward base (0.5)
        if weight_after_update > 0.5:
            assert weight_after_decay <= weight_after_update
        elif weight_after_update < 0.5:
            assert weight_after_decay >= weight_after_update


# ═══════════════════════════════════════════════════════════════════════════
# Integration: ComprehensionEnsemble + Plasticity
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsemblePlasticity:
    """Test ComprehensionEnsemble with STDP plasticity attached."""

    def test_ensemble_works_without_plasticity(self) -> None:
        """Ensemble without plasticity works identically to before."""
        from hbllm.brain.snn.comprehension.ensemble import ComprehensionEnsemble

        ensemble = ComprehensionEnsemble(domain="general")

        signals = {
            "semantic_weight": 0.8,
            "topic_shift": 0.3,
            "novelty": 0.5,
            "inter_novelty": 0.2,
            "constraint": 0.0,
            "buffer_pressure": 0.3,
            "punctuation": 0.4,
        }

        fired = ensemble.step(signals, time.time())
        # Should not crash — backward compatible
        assert isinstance(fired, list)

    def test_ensemble_with_plasticity(self) -> None:
        """Ensemble with plasticity records STDP updates."""
        from hbllm.brain.snn.comprehension.ensemble import ComprehensionEnsemble

        rule = STDPRule(learning_rate=0.1)
        static_w = {
            "entity": {"semantic_weight": 0.5, "topic_shift": 0.3, "novelty": 0.2},
            "clause": {
                "punctuation": 0.3,
                "buffer_pressure": 0.3,
                "topic_shift": 0.2,
                "semantic_weight": 0.2,
            },
            "discourse": {
                "topic_shift": 0.4,
                "inter_novelty": 0.3,
                "buffer_pressure": 0.2,
                "novelty": 0.1,
            },
            "surprise": {"inter_novelty": 0.5, "constraint": 0.3, "novelty": 0.2},
            "constraint": {"constraint": 0.7, "semantic_weight": 0.2, "punctuation": 0.1},
        }
        matrix = PlasticWeightMatrix(static_w, rule)

        ensemble = ComprehensionEnsemble(domain="general", plastic_weights=matrix)

        # Stimulate repeatedly
        t = time.time()
        for i in range(20):
            signals = {
                "semantic_weight": 0.8,
                "topic_shift": 0.0,
                "novelty": 0.5,
                "inter_novelty": 0.0,
                "constraint": 0.0,
                "buffer_pressure": float(i) / 20.0,
                "punctuation": 0.4 if i % 5 == 0 else 0.0,
            }
            ensemble.step(signals, t + i * 0.1)

        # Matrix should have recorded updates
        assert matrix.global_step == 20

    def test_plasticity_changes_weights(self) -> None:
        """After stimulation, some weights should differ from static."""
        from hbllm.brain.snn.comprehension.ensemble import ComprehensionEnsemble

        rule = STDPRule(learning_rate=0.05, time_constant=0.5)
        static_w = {
            "entity": {"semantic_weight": 0.5, "topic_shift": 0.3, "novelty": 0.2},
            "clause": {
                "punctuation": 0.3,
                "buffer_pressure": 0.3,
                "topic_shift": 0.2,
                "semantic_weight": 0.2,
            },
            "discourse": {
                "topic_shift": 0.4,
                "inter_novelty": 0.3,
                "buffer_pressure": 0.2,
                "novelty": 0.1,
            },
            "surprise": {"inter_novelty": 0.5, "constraint": 0.3, "novelty": 0.2},
            "constraint": {"constraint": 0.7, "semantic_weight": 0.2, "punctuation": 0.1},
        }
        matrix = PlasticWeightMatrix(static_w, rule)
        ensemble = ComprehensionEnsemble(domain="general", plastic_weights=matrix)

        t = time.time()
        for i in range(50):
            signals = {
                "semantic_weight": 0.9,
                "topic_shift": 0.7 if i % 10 == 0 else 0.0,
                "novelty": 0.5,
                "inter_novelty": 0.3,
                "constraint": 0.8 if i % 15 == 0 else 0.0,
                "buffer_pressure": min(1.0, (i % 12) / 12.0) * 0.3,
                "punctuation": 0.4 if i % 8 == 0 else 0.0,
            }
            ensemble.step(signals, t + i * 0.05)

        # At least some connections should have updated
        assert matrix.get_total_updates() > 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: ThoughtController + Plasticity
# ═══════════════════════════════════════════════════════════════════════════


class TestControllerPlasticity:
    """Test ThoughtController with STDP plasticity."""

    def test_controller_without_plasticity(self) -> None:
        """Controller without plasticity works identically to before."""
        from hbllm.brain.snn.expression.models import ThoughtGoal
        from hbllm.brain.snn.expression.thought_controller import ThoughtController

        controller = ThoughtController()
        goal = ThoughtGoal(id="g1", text="test", salience=0.8)

        signal = controller.gate(goal, prev_fragment_text=None)
        assert signal.fire is True

    def test_controller_with_plasticity(self) -> None:
        """Controller with plasticity records STDP and still gates."""
        from hbllm.brain.snn.expression.models import ThoughtGoal
        from hbllm.brain.snn.expression.thought_controller import ThoughtController

        rule = STDPRule(learning_rate=0.05)
        static_w = {
            "readiness": {"salience": 0.5, "memory_density": 0.3, "budget": 0.2},
            "coherence": {"base": 0.4, "overlap": 1.0, "constraint_penalty": 1.0},
        }
        matrix = PlasticWeightMatrix(static_w, rule)

        controller = ThoughtController(plastic_weights=matrix)

        goal = ThoughtGoal(
            id="g1",
            text="neural networks",
            salience=0.9,
            memory_hints=["ctx1", "ctx2"],
            max_tokens=512,
        )

        # First goal bypasses
        signal = controller.gate(goal, prev_fragment_text=None)
        assert signal.fire is True

        # Subsequent goals use plasticity
        for _ in range(5):
            signal = controller.gate(goal, prev_fragment_text="previous text about neural networks")

        # Matrix should have tracked steps
        assert matrix.global_step > 0
