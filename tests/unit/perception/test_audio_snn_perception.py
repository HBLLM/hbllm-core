"""Tests for Audio SNN Perception — Wave A5.

Tests cheap signal extraction, 4-channel LIF gating ensemble,
and SNN-gated audio perception stream.
"""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.brain.snn.perception.audio_ensemble import (
    AudioPerceptionEnsemble,
)
from hbllm.brain.snn.perception.audio_signals import AudioSignals, extract_audio_signals
from hbllm.brain.snn.perception.gate import (
    PerceptionEventType,
    PerceptionProcessingLevel,
)
from hbllm.perception.audio_perception_runtime import AudioPerceptionRuntime
from hbllm.perception.audio_perception_stream import AudioPerceptionStream
from hbllm.perception.providers.mock_audio_provider import MockAudioProvider

# ═══════════════════════════════════════════════════════════════════════════
# Signal Extraction Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioSignals:
    """Tests for cheap audio feature extraction."""

    def test_silence_signals(self) -> None:
        silence = np.zeros(1600, dtype=np.float32)
        signals = extract_audio_signals(silence, sample_rate=16000)
        assert signals.energy == pytest.approx(0.0)
        assert signals.spectral_centroid == pytest.approx(0.0)
        assert signals.zero_crossing_rate == pytest.approx(0.0)
        assert signals.speech_likelihood == pytest.approx(0.0)

    def test_empty_audio(self) -> None:
        empty = np.array([], dtype=np.float32)
        signals = extract_audio_signals(empty)
        assert signals.energy == 0.0

    def test_pure_tone_1khz(self) -> None:
        """1 kHz sine wave has energy in speech band."""
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        signals = extract_audio_signals(tone, sample_rate=16000)
        assert signals.energy > 0.5
        # 1kHz is well within 300-3400 Hz speech band
        assert signals.speech_likelihood > 0.8

    def test_high_frequency_tone(self) -> None:
        """6 kHz tone has high spectral centroid and low speech likelihood."""
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
        signals = extract_audio_signals(tone, sample_rate=16000)
        assert signals.energy > 0.5
        assert signals.spectral_centroid > 0.5
        assert signals.speech_likelihood < 0.2

    def test_spectral_flux(self) -> None:
        """Sudden change between frames yields high flux."""
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        frame1 = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        frame2 = (0.5 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)

        spec1 = np.abs(np.fft.rfft(frame1 * np.hanning(len(frame1))))
        signals = extract_audio_signals(frame2, sample_rate=16000, prev_spectrum=spec1)
        assert signals.spectral_flux > 0.1

    def test_to_array(self) -> None:
        signals = AudioSignals(
            energy=0.5,
            spectral_centroid=0.3,
            spectral_flux=0.2,
            zero_crossing_rate=0.1,
            speech_likelihood=0.8,
        )
        arr = signals.to_array()
        assert len(arr) == 5
        assert arr[0] == pytest.approx(0.5)
        assert arr[4] == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════════════════
# SNN Ensemble Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioPerceptionEnsemble:
    """Tests for 4-channel audio SNN gating."""

    def test_ensemble_initialization(self) -> None:
        ensemble = AudioPerceptionEnsemble()
        assert "speech" in ensemble.neurons
        assert "event" in ensemble.neurons
        assert "change" in ensemble.neurons
        assert "transient" in ensemble.neurons

    def test_silence_produces_no_action(self) -> None:
        ensemble = AudioPerceptionEnsemble()
        silence_signals = AudioSignals()
        decision = ensemble.step(silence_signals, sample_index=1)
        assert not decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.NONE
        assert decision.modality == "audio"

    def test_speech_signals_trigger_speech_onset(self) -> None:
        ensemble = AudioPerceptionEnsemble()
        speech_signals = AudioSignals(
            energy=0.8,
            speech_likelihood=0.9,
            spectral_centroid=0.3,
            zero_crossing_rate=0.2,
        )
        # Step a few frames to integrate and spike
        decisions = [ensemble.step(speech_signals, sample_index=i) for i in range(1, 5)]
        firing_decisions = [d for d in decisions if d.should_process]
        assert len(firing_decisions) >= 1
        assert any(d.event_type == PerceptionEventType.SPEECH_ONSET for d in firing_decisions)

    def test_sharp_transient_event(self) -> None:
        ensemble = AudioPerceptionEnsemble()
        transient_signals = AudioSignals(
            energy=1.0,
            spectral_flux=1.0,
            spectral_centroid=0.6,
        )
        # Immediate sharp burst
        decisions = [ensemble.step(transient_signals, sample_index=i) for i in range(1, 4)]
        firing_decisions = [d for d in decisions if d.should_process]
        assert len(firing_decisions) >= 1
        assert any(
            d.processing_level in (PerceptionProcessingLevel.HIGH, PerceptionProcessingLevel.URGENT)
            for d in firing_decisions
        )

    def test_ensemble_reset(self) -> None:
        ensemble = AudioPerceptionEnsemble()
        strong_signals = AudioSignals(energy=1.0, speech_likelihood=1.0)
        ensemble.step(strong_signals)
        ensemble.reset()
        assert ensemble._sample_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Stream Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAudioPerceptionStream:
    """Tests for AudioPerceptionStream end-to-end processing."""

    @pytest.fixture
    def mock_provider(self) -> MockAudioProvider:
        return MockAudioProvider()

    @pytest.fixture
    def runtime(self, mock_provider: MockAudioProvider) -> AudioPerceptionRuntime:
        return AudioPerceptionRuntime(
            speech=mock_provider,
            events=mock_provider,
            scene=mock_provider,
            speaker=mock_provider,
        )

    @pytest.mark.asyncio
    async def test_stream_skips_silence(self, runtime: AudioPerceptionRuntime) -> None:
        stream = AudioPerceptionStream(runtime=runtime)
        silence_chunk = np.zeros(1600, dtype=np.float32)

        for _ in range(5):
            decision, assessment = await stream.process_chunk(silence_chunk)
            assert not decision.should_process
            assert assessment is None

        stats = stream.stats
        assert stats["chunks_processed"] == 5
        assert stats["chunks_analyzed"] == 0
        assert stats["skip_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_stream_processes_active_audio(self, runtime: AudioPerceptionRuntime) -> None:
        stream = AudioPerceptionStream(runtime=runtime)
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        active_chunk = (0.8 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        assessments = []
        for _ in range(5):
            _, assessment = await stream.process_chunk(active_chunk)
            if assessment is not None:
                assessments.append(assessment)

        assert len(assessments) >= 1
        assert assessments[0].speech is not None
        assert stream.stats["chunks_analyzed"] >= 1

    @pytest.mark.asyncio
    async def test_stream_callback(self, runtime: AudioPerceptionRuntime) -> None:
        received = []

        def callback(assessment, decision):
            received.append((assessment, decision))

        stream = AudioPerceptionStream(runtime=runtime, on_assessment=callback)
        t = np.linspace(0, 0.1, 1600, endpoint=False)
        active_chunk = (0.9 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)

        for _ in range(5):
            await stream.process_chunk(active_chunk)

        assert len(received) >= 1
