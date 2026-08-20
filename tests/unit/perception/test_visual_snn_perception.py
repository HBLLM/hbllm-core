"""Tests for SNN Visual Perception — V3."""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.brain.snn.perception.gate import PerceptionProcessingLevel
from hbllm.brain.snn.perception.visual_ensemble import PerceptionEnsemble
from hbllm.brain.snn.perception.visual_signals import VisualSignalExtractor, VisualSignals
from hbllm.perception.visual_perception_stream import (
    VisualPerceptionStream,
)

# ═══════════════════════════════════════════════════════════════════════════
# Visual Signal Extractor
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualSignalExtractor:
    @pytest.fixture
    def extractor(self) -> VisualSignalExtractor:
        return VisualSignalExtractor(downsample=1)

    def test_first_frame_all_zeros(self, extractor: VisualSignalExtractor) -> None:
        """First frame has no history → all signals are 0."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        signals = extractor.extract(frame)
        assert signals.motion == 0.0
        assert signals.intensity == 0.0
        assert signals.edge == 0.0
        assert signals.color == 0.0
        assert signals.texture == 0.0

    def test_static_scene_no_motion(self, extractor: VisualSignalExtractor) -> None:
        """Two identical frames → no motion signal."""
        frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        extractor.extract(frame)  # First frame (baseline)
        signals = extractor.extract(frame)  # Same frame
        assert signals.motion == 0.0

    def test_scene_change_motion(self, extractor: VisualSignalExtractor) -> None:
        """Drastically different frames → high motion signal."""
        frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame2 = np.full((50, 50, 3), 255, dtype=np.uint8)
        extractor.extract(frame1)
        signals = extractor.extract(frame2)
        assert signals.motion > 0.5
        assert signals.intensity > 0.5

    def test_signals_normalized(self, extractor: VisualSignalExtractor) -> None:
        """All signals must be in [0.0, 1.0]."""
        frame1 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        extractor.extract(frame1)
        signals = extractor.extract(frame2)
        for name, val in signals.to_dict().items():
            assert 0.0 <= val <= 1.0, f"{name} = {val} out of range"

    def test_grayscale_input(self, extractor: VisualSignalExtractor) -> None:
        """Should handle grayscale (H, W) frames."""
        frame1 = np.zeros((50, 50), dtype=np.uint8)
        frame2 = np.full((50, 50), 128, dtype=np.uint8)
        extractor.extract(frame1)
        signals = extractor.extract(frame2)
        assert signals.motion > 0
        assert signals.color == 0.0  # No color info in grayscale

    def test_downsample(self) -> None:
        """Downsampling should reduce processing without errors."""
        extractor = VisualSignalExtractor(downsample=4)
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        signals = extractor.extract(frame)
        assert isinstance(signals, VisualSignals)

    def test_reset(self, extractor: VisualSignalExtractor) -> None:
        """Reset should clear history."""
        frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        extractor.extract(frame)
        extractor.reset()
        # After reset, next frame should be treated as first
        signals = extractor.extract(frame)
        assert signals.motion == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Perception Ensemble
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceptionEnsemble:
    @pytest.fixture
    def ensemble(self) -> PerceptionEnsemble:
        return PerceptionEnsemble()

    def test_no_signals_no_spikes(self, ensemble: PerceptionEnsemble) -> None:
        """Zero signals → no spikes → NONE processing."""
        signals = VisualSignals()
        decision = ensemble.step(signals)
        assert not decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.NONE

    def test_strong_motion_triggers_spike(self, ensemble: PerceptionEnsemble) -> None:
        """Sustained strong motion should eventually trigger motion spike."""
        strong_motion = VisualSignals(motion=1.0)
        any_processed = False
        for _ in range(20):
            decision = ensemble.step(strong_motion)
            if decision.should_process:
                any_processed = True
                break
        assert any_processed, "Strong motion should trigger processing"

    def test_static_scene_no_processing(self, ensemble: PerceptionEnsemble) -> None:
        """Static scene (zero signals) should never trigger processing."""
        static = VisualSignals()
        for _ in range(100):
            decision = ensemble.step(static)
            assert not decision.should_process

    def test_scene_change_triggers_high_processing(
        self,
        ensemble: PerceptionEnsemble,
    ) -> None:
        """Major scene change should trigger high/urgent processing."""
        scene_change = VisualSignals(
            motion=0.8,
            intensity=0.9,
            edge=0.5,
            color=0.7,
        )
        # Drive for many frames to accumulate
        any_high = False
        for _ in range(50):
            decision = ensemble.step(scene_change)
            if decision.processing_level in (
                PerceptionProcessingLevel.HIGH,
                PerceptionProcessingLevel.URGENT,
            ):
                any_high = True
                break
        assert any_high, "Scene change should trigger high-level processing"

    def test_reset(self, ensemble: PerceptionEnsemble) -> None:
        """Reset should clear all neuron state."""
        ensemble.step(VisualSignals(motion=0.5))
        ensemble.reset()
        assert all(v <= 0 for v in ensemble.state.values())

    def test_state_returns_potentials(self, ensemble: PerceptionEnsemble) -> None:
        """State property should return all channel potentials."""
        state = ensemble.state
        assert "scene" in state
        assert "entity" in state
        assert "motion" in state
        assert "novelty" in state
        assert "stability" in state


# ═══════════════════════════════════════════════════════════════════════════
# Visual Perception Stream
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualPerceptionStream:
    def test_stream_without_perception(self) -> None:
        """Stream without perception should still gate properly."""
        stream = VisualPerceptionStream(perception=None)
        assert stream.stats.total_frames == 0

    @pytest.mark.asyncio
    async def test_static_scene_mostly_skipped(self) -> None:
        """Static scene should skip most frames."""
        stream = VisualPerceptionStream(perception=None)

        # Send 50 identical frames
        frame = np.full((50, 50, 3), 128, dtype=np.uint8)
        for _ in range(50):
            await stream.process_frame(frame)

        assert stream.stats.total_frames == 50
        # Most frames should be skipped (no perception needed)
        assert stream.stats.skipped_frames > 40

    @pytest.mark.asyncio
    async def test_scene_change_triggers_processing(self) -> None:
        """Dramatic scene change should trigger some processing."""
        stream = VisualPerceptionStream(perception=None)

        # 10 dark frames
        dark = np.zeros((50, 50, 3), dtype=np.uint8)
        for _ in range(10):
            await stream.process_frame(dark)

        # Then 10 bright frames — should trigger processing eventually
        bright = np.full((50, 50, 3), 255, dtype=np.uint8)
        any_triggered = False
        for _ in range(10):
            result = await stream.process_frame(bright)
            if result.decision.should_process:
                any_triggered = True

        assert any_triggered, "Scene change should trigger processing"

    @pytest.mark.asyncio
    async def test_stats_tracking(self) -> None:
        """Stats should track frame counts."""
        stream = VisualPerceptionStream(perception=None)
        frame = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)

        for _ in range(5):
            await stream.process_frame(frame)

        assert stream.stats.total_frames == 5
        assert stream.stats.skipped_frames + stream.stats.processed_frames == 5

    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        """Reset should clear all accumulated state."""
        stream = VisualPerceptionStream(perception=None)
        frame = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        await stream.process_frame(frame)

        stream.reset()
        assert stream.stats.total_frames == 0
        assert stream._frame_index == 0
