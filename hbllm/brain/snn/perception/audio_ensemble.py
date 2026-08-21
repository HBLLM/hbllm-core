"""Audio Perception Ensemble — SNN-gated audio stream.

A 4-channel LIF neuron ensemble specialized for audio perception.
Decides HOW MUCH compute to spend on analyzing audio.

Channels:
    speech    — detects speech onset/activity (speech band + vocal cues)
    event     — detects transient acoustic events (sharp energy/flux changes)
    change    — detects scene/ambient spectral shifts
    transient — detects rapid burst sounds (clicks, knocks, snaps)

Architecture:
    AudioSignals → AudioPerceptionEnsemble → SpikeEvent[] → PerceptionGateDecision

Invariant:
    The SNN gate answers: "Should we spend compute analyzing this audio?"
    Never: "This is a doorbell."
"""

from __future__ import annotations

import logging

from hbllm.brain.snn.lif import LIFConfig, LIFNeuron
from hbllm.brain.snn.neurons import SpikeEvent
from hbllm.brain.snn.perception.audio_signals import AudioSignals
from hbllm.brain.snn.perception.gate import (
    PerceptionEventType,
    PerceptionGateDecision,
    PerceptionProcessingLevel,
)

logger = logging.getLogger(__name__)


# ── Channel configurations ────────────────────────────────────────────
# LIFConfig: threshold, decay_half_life, reset_potential, refractory_period

AUDIO_CHANNEL_CONFIGS: dict[str, LIFConfig] = {
    "speech": LIFConfig(
        threshold=0.5,
        decay_half_life=1.0,  # Integrates speech presence
        reset_potential=0.0,
        refractory_period=0.3,
    ),
    "event": LIFConfig(
        threshold=0.6,
        decay_half_life=0.4,  # Fast transient response
        reset_potential=0.0,
        refractory_period=0.2,
    ),
    "change": LIFConfig(
        threshold=0.7,
        decay_half_life=3.0,  # Slower integration for ambient shifts
        reset_potential=0.0,
        refractory_period=1.0,
    ),
    "transient": LIFConfig(
        threshold=0.5,
        decay_half_life=0.2,  # Very fast burst detector
        reset_potential=0.0,
        refractory_period=0.1,
    ),
}


# ── Signal-to-channel weight matrix ──────────────────────────────────
# Input vector: [energy, spectral_centroid, spectral_flux, zero_crossing_rate, speech_likelihood]

AUDIO_SIGNAL_WEIGHTS: dict[str, list[float]] = {
    "speech": [0.2, 0.1, 0.1, 0.2, 0.8],  # Speech likelihood dominant
    "event": [0.5, 0.3, 0.6, 0.2, 0.1],  # Flux + energy dominant
    "change": [0.2, 0.6, 0.7, 0.1, 0.0],  # Centroid + flux dominant
    "transient": [0.8, 0.2, 0.8, 0.3, 0.0],  # Energy + flux dominant
}


class AudioPerceptionEnsemble:
    """4-channel SNN ensemble for audio perception gating.

    Converts cheap audio features into spike events that drive
    the PerceptionGateDecision for each audio chunk/frame.

    Usage::

        ensemble = AudioPerceptionEnsemble()
        signals = extract_audio_signals(audio_chunk)
        decision = ensemble.step(signals, sample_index=42)

        if decision.should_process:
            # Run expensive audio providers
            assessment = await runtime.perceive(audio_chunk)
    """

    def __init__(self, configs: dict[str, LIFConfig] | None = None) -> None:
        self.configs = configs or AUDIO_CHANNEL_CONFIGS
        self.neurons: dict[str, LIFNeuron] = {
            name: LIFNeuron(config=cfg, neuron_id=f"aud_{name}")
            for name, cfg in self.configs.items()
        }
        self._sample_count = 0
        self._current_time = 0.0

    def step(
        self,
        signals: AudioSignals,
        dt_s: float = 0.05,
        sample_index: int | None = None,
    ) -> PerceptionGateDecision:
        """Step the SNN ensemble with new audio signals.

        Args:
            signals: Cheap extracted audio features.
            dt_s: Time step in seconds.
            sample_index: Optional index of the audio chunk.

        Returns:
            PerceptionGateDecision with processing level and urgency.
        """
        self._sample_count += 1
        self._current_time += dt_s
        idx = sample_index if sample_index is not None else self._sample_count
        sig_array = signals.to_array()

        fired: list[tuple[str, SpikeEvent]] = []

        for name, neuron in self.neurons.items():
            weights = AUDIO_SIGNAL_WEIGHTS.get(name, [0.2] * 5)
            # Weighted dot product of signals
            current = float(sum(w * s for w, s in zip(weights, sig_array, strict=False)))
            current = max(0.0, current)  # Rectify

            spike = neuron.step(current, self._current_time)
            if spike.fired:
                fired.append((name, spike))

        return self._make_decision(fired, idx)

    def _make_decision(
        self,
        fired: list[tuple[str, SpikeEvent]],
        sample_index: int,
    ) -> PerceptionGateDecision:
        """Convert fired spikes into a PerceptionGateDecision."""
        channels = {ch: spike.strength for ch, spike in fired}

        if not channels:
            return PerceptionGateDecision(
                modality="audio",
                should_process=False,
                processing_level=PerceptionProcessingLevel.NONE,
                frame_index=sample_index,
            )

        # ── Determine event type from priority ──
        if "speech" in channels:
            event_type = PerceptionEventType.SPEECH_ONSET
        elif "event" in channels:
            event_type = PerceptionEventType.ACOUSTIC_EVENT
        elif "transient" in channels:
            event_type = PerceptionEventType.TRANSIENT_BURST
        else:
            event_type = PerceptionEventType.AMBIENT_CHANGE

        urgency = max(channels.values())
        num_channels = len(channels)

        # ── Determine processing level ──
        if num_channels >= 3 or ("event" in channels and urgency > 0.8):
            level = PerceptionProcessingLevel.URGENT
        elif "speech" in channels or ("event" in channels and urgency > 0.5):
            level = PerceptionProcessingLevel.HIGH
        elif "transient" in channels or "change" in channels:
            level = PerceptionProcessingLevel.STANDARD
        else:
            level = PerceptionProcessingLevel.LOW

        return PerceptionGateDecision(
            modality="audio",
            should_process=True,
            processing_level=level,
            urgency=urgency,
            event_type=event_type,
            channels_fired=channels,
            novelty=channels.get("change", 0.0),
            temporal_significance=sum(channels.values()) / num_channels,
            frame_index=sample_index,
        )

    def reset(self) -> None:
        """Reset all neurons in the ensemble."""
        for neuron in self.neurons.values():
            neuron.reset()
        self._sample_count = 0
        self._current_time = 0.0
