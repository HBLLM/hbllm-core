"""Audio Signal Features — cheap SNN gating signals.

Extracts lightweight audio features (~0.1ms) using only numpy.
These features feed the SNN gate, which decides processing depth.

The SNN answers: "Should we spend compute analyzing this audio?"
NOT: "What is this sound?"

Features:
    energy       — RMS energy level
    spectral_centroid — brightness
    spectral_flux    — change rate between frames
    zero_crossing_rate — noisiness/periodicity
    speech_likelihood  — energy-in-speech-band proxy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioSignals:
    """Cheap audio features for SNN gating.

    All fields are normalized to approximately [0.0, 1.0].

    Attributes:
        energy: Normalized RMS energy.
        spectral_centroid: Normalized spectral centroid (brightness).
        spectral_flux: Spectral change rate.
        zero_crossing_rate: Noisiness indicator.
        speech_likelihood: Energy-in-speech-band proxy.

    """

    energy: float = 0.0
    spectral_centroid: float = 0.0
    spectral_flux: float = 0.0
    zero_crossing_rate: float = 0.0
    speech_likelihood: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for SNN input."""
        return np.array(
            [
                self.energy,
                self.spectral_centroid,
                self.spectral_flux,
                self.zero_crossing_rate,
                self.speech_likelihood,
            ],
            dtype=np.float32,
        )


def extract_audio_signals(
    audio: np.ndarray,
    sample_rate: int = 16000,
    prev_spectrum: np.ndarray | None = None,
) -> AudioSignals:
    """Extract cheap audio features from a raw audio buffer.

    Args:
        audio: Mono float32 audio, shape (N,).
        sample_rate: Sample rate in Hz.
        prev_spectrum: Previous magnitude spectrum for flux computation.

    Returns:
        AudioSignals with all features normalized to ~[0, 1].

    """
    if audio.size == 0:
        return AudioSignals()

    audio = audio.astype(np.float32)

    # ── Energy (RMS) ──
    rms = float(np.sqrt(np.mean(audio**2)))
    # Normalize: -60 dB → 0.0, 0 dB → 1.0
    energy_db = 20 * np.log10(max(rms, 1e-10))
    energy = float(np.clip((energy_db + 60) / 60, 0.0, 1.0))

    # ── FFT-based features ──
    n_fft = min(2048, audio.size)
    windowed = audio[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # ── Spectral Centroid ──
    total_magnitude = spectrum.sum()
    if total_magnitude > 0:
        centroid_hz = float(np.sum(freqs * spectrum) / total_magnitude)
        # Normalize: 0 Hz → 0, Nyquist → 1
        spectral_centroid = float(np.clip(centroid_hz / (sample_rate / 2), 0.0, 1.0))
    else:
        spectral_centroid = 0.0

    # ── Spectral Flux ──
    if prev_spectrum is not None and prev_spectrum.size == spectrum.size:
        flux = float(np.sum((spectrum - prev_spectrum) ** 2))
        # Normalize by frame energy
        max_flux = float(np.sum(spectrum**2)) + 1e-10
        spectral_flux = float(np.clip(flux / max_flux, 0.0, 1.0))
    else:
        spectral_flux = 0.0

    # ── Zero Crossing Rate ──
    if audio.size > 1:
        crossings = np.sum(np.abs(np.diff(np.sign(audio))) > 0)
        zcr = float(crossings) / (audio.size - 1)
    else:
        zcr = 0.0

    # ── Speech Likelihood ──
    # Simple proxy: energy in 300-3400 Hz band relative to total
    speech_low = 300
    speech_high = 3400
    speech_mask = (freqs >= speech_low) & (freqs <= speech_high)
    if total_magnitude > 0:
        speech_energy = float(spectrum[speech_mask].sum())
        speech_likelihood = float(np.clip(speech_energy / total_magnitude, 0.0, 1.0))
    else:
        speech_likelihood = 0.0

    return AudioSignals(
        energy=energy,
        spectral_centroid=spectral_centroid,
        spectral_flux=spectral_flux,
        zero_crossing_rate=zcr,
        speech_likelihood=speech_likelihood,
    )
