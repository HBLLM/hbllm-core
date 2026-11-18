"""Visual Signal Extractor — cheap frame features for SNN gating.

Extracts fast (~0.1ms) numerical signals from video frames for the
SNN ensemble to decide whether expensive perception is warranted.

No ML models — just pixel-level statistics.  The SNN learns what
patterns of these cheap signals correlate with cognitively important
visual events.

Signals:
    motion:    Pixel-level frame difference (MAD)
    intensity: Mean brightness change
    edge:      Edge density change (Sobel approximation)
    color:     Color histogram shift
    texture:   Variance change (texture complexity)

All signals are normalized to [0.0, 1.0].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VisualSignals:
    """Cheap frame features — all in [0.0, 1.0]."""

    motion: float = 0.0  # Frame difference magnitude
    intensity: float = 0.0  # Brightness change
    edge: float = 0.0  # Edge density change
    color: float = 0.0  # Color histogram shift
    texture: float = 0.0  # Texture complexity change

    def to_dict(self) -> dict[str, float]:
        return {
            "motion": self.motion,
            "intensity": self.intensity,
            "edge": self.edge,
            "color": self.color,
            "texture": self.texture,
        }


class VisualSignalExtractor:
    """Extracts cheap visual signals from frames.

    Maintains state (previous frame) for frame-difference signals.
    All signals are normalized to [0.0, 1.0].

    Usage::

        extractor = VisualSignalExtractor()
        signals = extractor.extract(frame)
        # signals.motion ∈ [0.0, 1.0]
    """

    def __init__(self, downsample: int = 4) -> None:
        """Initialize.

        Args:
            downsample: Downsample factor for faster computation.
                        4 means process at 1/4 resolution.

        """
        self._prev_gray: np.ndarray | None = None
        self._prev_edges: float = 0.0
        self._prev_variance: float = 0.0
        self._prev_hist: np.ndarray | None = None
        self._downsample = max(1, downsample)

    def extract(self, frame: np.ndarray) -> VisualSignals:
        """Extract cheap visual signals from a video frame.

        Args:
            frame: BGR or grayscale numpy array (H, W, C) or (H, W).

        Returns:
            VisualSignals with all values in [0.0, 1.0].

        """
        # Downsample for speed
        if self._downsample > 1:
            frame = frame[:: self._downsample, :: self._downsample]

        # Convert to grayscale
        if frame.ndim == 3:
            # Simple luminance (avoid OpenCV dependency)
            gray = (
                0.299 * frame[:, :, 2].astype(np.float32)
                + 0.587 * frame[:, :, 1].astype(np.float32)
                + 0.114 * frame[:, :, 0].astype(np.float32)
            )
        else:
            gray = frame.astype(np.float32)

        signals = VisualSignals()

        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            # ── Motion: Mean Absolute Difference ──
            diff = np.abs(gray - self._prev_gray)
            mad = float(np.mean(diff)) / 255.0
            signals.motion = min(1.0, mad * 5.0)  # Scale for sensitivity

            # ── Intensity: Mean brightness change ──
            mean_curr = float(np.mean(gray))
            mean_prev = float(np.mean(self._prev_gray))
            intensity_change = abs(mean_curr - mean_prev) / 255.0
            signals.intensity = min(1.0, intensity_change * 10.0)

            # ── Edge: Sobel-approximation edge density change ──
            edges = self._edge_density(gray)
            edge_change = abs(edges - self._prev_edges)
            signals.edge = min(1.0, edge_change * 20.0)
            self._prev_edges = edges

            # ── Texture: Variance change ──
            variance = float(np.var(gray)) / (255.0 * 255.0)
            variance_change = abs(variance - self._prev_variance)
            signals.texture = min(1.0, variance_change * 50.0)
            self._prev_variance = variance

            # ── Color: Histogram shift (if color frame available) ──
            if frame.ndim == 3:
                hist = self._color_histogram(frame)
                if self._prev_hist is not None:
                    hist_diff = float(np.sum(np.abs(hist - self._prev_hist)))
                    signals.color = min(1.0, hist_diff * 2.0)
                self._prev_hist = hist
        else:
            # First frame — compute baselines
            self._prev_edges = self._edge_density(gray)
            self._prev_variance = float(np.var(gray)) / (255.0 * 255.0)
            if frame.ndim == 3:
                self._prev_hist = self._color_histogram(frame)

        self._prev_gray = gray.copy()
        return signals

    def reset(self) -> None:
        """Reset state (new video stream)."""
        self._prev_gray = None
        self._prev_edges = 0.0
        self._prev_variance = 0.0
        self._prev_hist = None

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        """Simple edge density using Sobel-like gradient magnitude."""
        # Horizontal gradient
        gx = np.diff(gray, axis=1)
        # Vertical gradient
        gy = np.diff(gray, axis=0)
        # Edge density = mean gradient magnitude (approximate)
        edge_x = float(np.mean(np.abs(gx)))
        edge_y = float(np.mean(np.abs(gy)))
        return (edge_x + edge_y) / (2.0 * 255.0)

    @staticmethod
    def _color_histogram(frame: np.ndarray, bins: int = 16) -> np.ndarray:
        """Compute a normalized color histogram."""
        hist = np.zeros(bins * 3, dtype=np.float32)
        for c in range(3):
            channel = frame[:, :, c].ravel()
            h, _ = np.histogram(channel, bins=bins, range=(0, 256))
            total = h.sum()
            if total > 0:
                h = h.astype(np.float32) / total
            hist[c * bins : (c + 1) * bins] = h
        return hist
