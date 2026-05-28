"""
Vector Projector Module.

Projects modality-specific sensory embeddings (vision, audio, sensors)
into the shared high-dimensional cognitive / LLM latent space.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class MultimodalProjector:
    """
    Projects modality-specific sensory embeddings (vision, audio, sensors)
    into the shared high-dimensional cognitive / LLM latent space.

    Supports weight loading from `.safetensors` files, with robust,
    zero-dependency fallbacks (identity mapping + zero-padding/truncation).
    """

    def __init__(self, llm_dim: int = 4096, weights_path: str | None = None) -> None:
        self.llm_dim = llm_dim
        self.weights: dict[str, Any] = {}
        self.weights_path = weights_path

        if weights_path and os.path.exists(weights_path):
            try:
                from safetensors.numpy import load_file  # type: ignore

                self.weights = load_file(weights_path)
                logger.info("Loaded projection weights from %s", weights_path)
            except Exception as e:
                logger.warning(
                    "Failed to load safetensors from %s: %s. Using fallback projection.",
                    weights_path,
                    e,
                )

    def project_vision(self, embedding: list[float]) -> list[float]:
        """Project a vision embedding (e.g. 768-dim) into the LLM latent space (e.g. 4096-dim)."""
        if "vision.weight" in self.weights:
            try:
                import numpy as np

                w = self.weights["vision.weight"]
                x = np.array(embedding, dtype=np.float32)
                if w.shape[0] == self.llm_dim:
                    res = np.dot(w, x)
                else:
                    res = np.dot(x, w)
                if "vision.bias" in self.weights:
                    res += self.weights["vision.bias"]
                return [float(v) for v in res.tolist()]
            except Exception as e:
                logger.error("Vision projection error: %s. Falling back to padding.", e)

        return self._fallback_projection(embedding)

    def project_audio(self, embedding: list[float]) -> list[float]:
        """Project an audio embedding into the LLM latent space."""
        if "audio.weight" in self.weights:
            try:
                import numpy as np

                w = self.weights["audio.weight"]
                x = np.array(embedding, dtype=np.float32)
                if w.shape[0] == self.llm_dim:
                    res = np.dot(w, x)
                else:
                    res = np.dot(x, w)
                if "audio.bias" in self.weights:
                    res += self.weights["audio.bias"]
                return [float(v) for v in res.tolist()]
            except Exception as e:
                logger.error("Audio projection error: %s. Falling back to padding.", e)

        return self._fallback_projection(embedding)

    def project_sensor(self, readings: dict[str, float]) -> list[float]:
        """Normalize and project raw sensor readings into the LLM latent space."""
        sorted_keys = sorted(readings.keys())
        values = [readings[k] for k in sorted_keys]

        if "sensor.weight" in self.weights and len(values) > 0:
            try:
                import numpy as np

                w = self.weights["sensor.weight"]
                x = np.array(values, dtype=np.float32)
                if w.shape[1] == len(values):
                    res = np.dot(w, x)
                elif w.shape[0] == len(values):
                    res = np.dot(x, w)
                else:
                    raise ValueError(
                        f"Sensor weight shape mismatch. Expected dimension compatible with {len(values)}, got {w.shape}"
                    )
                if "sensor.bias" in self.weights:
                    res += self.weights["sensor.bias"]
                return [float(v) for v in res.tolist()]
            except Exception as e:
                logger.error("Sensor projection error: %s. Falling back to padding.", e)

        return self._fallback_projection(values)

    def _fallback_projection(self, x: list[float]) -> list[float]:
        """Helper to pad or truncate a list to target llm_dim."""
        L = len(x)
        if L < self.llm_dim:
            return x + [0.0] * (self.llm_dim - L)
        elif L > self.llm_dim:
            return x[: self.llm_dim]
        return x
