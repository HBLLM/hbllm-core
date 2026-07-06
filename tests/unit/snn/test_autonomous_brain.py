"""
Milestone 5: Autonomous Brain — Unit Tests.

Validates neural oscillation bands, phase gating, phase-amplitude
coupling, and frequency modulation.
"""

from __future__ import annotations

import math

from hbllm.brain.snn.oscillations import (
    BandConfig,
    OscillationBand,
    OscillationManager,
)

# ═══════════════════════════════════════════════════════════════════════════
# Oscillation Band Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOscillationBands:
    """Validate oscillation band definitions."""

    def test_four_bands_defined(self) -> None:
        bands = list(OscillationBand)
        assert len(bands) == 4
        assert OscillationBand.GAMMA in bands
        assert OscillationBand.BETA in bands
        assert OscillationBand.THETA in bands
        assert OscillationBand.DELTA in bands

    def test_band_values(self) -> None:
        assert OscillationBand.GAMMA.value == "gamma"
        assert OscillationBand.DELTA.value == "delta"


# ═══════════════════════════════════════════════════════════════════════════
# OscillationManager Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOscillationManager:
    """Validate phase computation, gating, and modulation."""

    def test_phase_within_range(self) -> None:
        """Phase should be within [0, 2π)."""
        osc = OscillationManager()
        for t in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
            phase = osc.get_phase(OscillationBand.THETA, t)
            assert 0.0 <= phase < 2.0 * math.pi

    def test_gate_within_range(self) -> None:
        """Gate value should be within [0.0, 1.0]."""
        osc = OscillationManager()
        for t in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
            gate = osc.get_gate(OscillationBand.GAMMA, t)
            assert 0.0 <= gate <= 1.0, f"Gate {gate} out of range at t={t}"

    def test_gate_oscillates(self) -> None:
        """Gate should have both high and low values over a full period."""
        osc = OscillationManager(reference_time=0.0)
        band = OscillationBand.THETA  # 6 Hz → period = 1/6 ≈ 0.167s

        gates = [osc.get_gate(band, t * 0.01) for t in range(20)]  # 0 to 0.19s
        assert max(gates) > 0.8, "Gate should reach high values"
        assert min(gates) < 0.2, "Gate should reach low values"

    def test_phase_amplitude_coupling(self) -> None:
        """Theta phase should modulate gamma amplitude."""
        osc = OscillationManager(reference_time=0.0)

        amplitudes = []
        for i in range(100):
            t = i * 0.005
            amp = osc.get_amplitude(
                OscillationBand.GAMMA,
                timestamp=t,
                modulating_band=OscillationBand.THETA,
            )
            amplitudes.append(amp)

        # Amplitude should vary (not constant)
        assert max(amplitudes) > min(amplitudes) + 0.1

    def test_frequency_modulation(self) -> None:
        """Modulating frequency should change the oscillation rate."""
        osc = OscillationManager(reference_time=0.0)

        # Baseline: measure phase change over fixed time
        t = 0.01
        baseline_phase = osc.get_phase(OscillationBand.GAMMA, t)

        # Speed up gamma 2x
        osc.modulate_frequency(OscillationBand.GAMMA, 2.0)
        fast_phase = osc.get_phase(OscillationBand.GAMMA, t)

        # The fast phase should be different (approximately double)
        assert fast_phase != baseline_phase

    def test_frequency_modulation_clamped(self) -> None:
        """Extreme frequency modulation should be clamped."""
        osc = OscillationManager()
        osc.modulate_frequency(OscillationBand.GAMMA, 100.0)  # Should clamp to 10.0
        osc.modulate_frequency(OscillationBand.BETA, 0.001)  # Should clamp to 0.1

        stats = osc.stats()
        assert stats["frequency_modifiers"]["gamma"] == 10.0
        assert stats["frequency_modifiers"]["beta"] == 0.1

    def test_reset_modulation(self) -> None:
        """Reset should restore all modifiers to 1.0."""
        osc = OscillationManager()
        osc.modulate_frequency(OscillationBand.GAMMA, 3.0)
        osc.modulate_frequency(OscillationBand.THETA, 0.5)

        osc.reset_modulation()

        stats = osc.stats()
        for modifier in stats["frequency_modifiers"].values():
            assert modifier == 1.0

    def test_dominant_band(self) -> None:
        """get_dominant_band should return a valid band."""
        osc = OscillationManager()
        dominant = osc.get_dominant_band(timestamp=0.0)
        assert dominant in OscillationBand

    def test_snapshot_contains_all_bands(self) -> None:
        """Snapshot should contain data for all 4 bands."""
        osc = OscillationManager()
        snap = osc.snapshot(timestamp=0.0)
        assert len(snap) == 4
        for band in OscillationBand:
            assert band.value in snap
            assert "phase" in snap[band.value]
            assert "gate" in snap[band.value]
            assert "amplitude" in snap[band.value]
            assert "frequency" in snap[band.value]

    def test_custom_band_config(self) -> None:
        """Custom band configurations should override defaults."""
        custom = {
            OscillationBand.GAMMA: BandConfig(frequency=80.0, amplitude=0.5),
        }
        osc = OscillationManager(configs=custom)
        snap = osc.snapshot(timestamp=0.0)
        assert snap["gamma"]["frequency"] == 80.0
