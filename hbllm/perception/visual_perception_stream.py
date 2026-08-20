"""Visual Perception Stream — SNN-gated continuous perception.

Integrates the full perception pipeline:
    Frame → VisualSignalExtractor → PerceptionEnsemble → PerceptionGateDecision
         → (if should_process) → VisualPerceptionRuntime → VisualAssessment
         → VisualPerceptionTransaction → HCIR commit

This is the runtime loop for continuous visual perception. Each frame
is first evaluated cheaply by the SNN ensemble to decide IF expensive
perception should run, and at what processing level.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from hbllm.brain.snn.perception.gate import (
    PerceptionGateDecision,
    PerceptionProcessingLevel,
)
from hbllm.brain.snn.perception.visual_ensemble import PerceptionEnsemble
from hbllm.brain.snn.perception.visual_signals import (
    VisualSignalExtractor,
    VisualSignals,
)

if TYPE_CHECKING:
    from hbllm.perception.visual_perception import VisualPerception

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result of processing one frame."""

    frame_index: int
    signals: VisualSignals
    decision: PerceptionGateDecision
    processed: bool = False
    recognition_label: str | None = None
    is_novel: bool = False
    elapsed_ms: float = 0.0


@dataclass
class StreamStats:
    """Accumulated statistics for a perception stream."""

    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    recognitions: int = 0
    novel_detections: int = 0
    total_process_ms: float = 0.0
    level_histogram: dict[str, int] = field(
        default_factory=lambda: {level.value: 0 for level in PerceptionProcessingLevel}
    )

    @property
    def process_rate(self) -> float:
        """Fraction of frames that triggered expensive perception."""
        return self.processed_frames / max(1, self.total_frames)

    @property
    def avg_process_ms(self) -> float:
        return self.total_process_ms / max(1, self.processed_frames)


class VisualPerceptionStream:
    """SNN-gated continuous visual perception stream.

    Usage::

        stream = VisualPerceptionStream(perception)

        # Process frames from a video feed
        for frame in video_frames:
            result = await stream.process_frame(frame)
            if result.processed:
                print(f"Recognized: {result.recognition_label}")

        print(stream.stats.process_rate)  # e.g., 0.12 = 12% of frames processed
    """

    def __init__(
        self,
        perception: VisualPerception | None = None,
        ensemble: PerceptionEnsemble | None = None,
        extractor: VisualSignalExtractor | None = None,
    ) -> None:
        self.perception = perception
        self.ensemble = ensemble or PerceptionEnsemble()
        self.extractor = extractor or VisualSignalExtractor()
        self.stats = StreamStats()
        self._frame_index = 0

    async def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process a single video frame through the SNN-gated pipeline.

        1. Extract cheap visual signals (~0.1ms)
        2. Run SNN ensemble → PerceptionGateDecision (~0.01ms)
        3. IF decision.should_process → run expensive perception

        Returns:
            FrameResult with signals, decision, and recognition result.

        """
        self._frame_index += 1
        self.stats.total_frames += 1

        # 1. Cheap signals
        signals = self.extractor.extract(frame)

        # 2. SNN gating decision
        decision = self.ensemble.step(signals, frame_index=self._frame_index)
        self.stats.level_histogram[decision.processing_level.value] = (
            self.stats.level_histogram.get(decision.processing_level.value, 0) + 1
        )

        result = FrameResult(
            frame_index=self._frame_index,
            signals=signals,
            decision=decision,
        )

        # 3. Expensive perception (only if gated)
        if decision.should_process and self.perception is not None:
            start = time.perf_counter()
            try:
                recognition = await self.perception.recognize(frame)
                result.processed = True
                self.stats.processed_frames += 1

                if recognition.matched:
                    result.recognition_label = recognition.label
                    self.stats.recognitions += 1
                elif recognition.is_novel:
                    result.is_novel = True
                    self.stats.novel_detections += 1
            except Exception as e:
                logger.error("Perception failed on frame %d: %s", self._frame_index, e)
            finally:
                result.elapsed_ms = (time.perf_counter() - start) * 1000.0
                self.stats.total_process_ms += result.elapsed_ms
        else:
            self.stats.skipped_frames += 1

        return result

    def reset(self) -> None:
        """Reset stream state for a new video."""
        self.ensemble.reset()
        self.extractor.reset()
        self.stats = StreamStats()
        self._frame_index = 0
