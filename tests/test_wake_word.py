"""Tests for WakeWordDetector — voice activation detection."""

import struct


from hbllm.perception.wake_word import (
    EnergyWakeWordEngine,
    WakeWordConfig,
    WakeWordDetector,
    WakeWordEvent,
)


class TestWakeWordEvent:
    def test_to_dict(self):
        ev = WakeWordEvent(wake_word="hey sentra", confidence=0.92)
        d = ev.to_dict()
        assert d["wake_word"] == "hey sentra"
        assert d["confidence"] == 0.92
        assert d["timestamp"] > 0


class TestWakeWordConfig:
    def test_defaults(self):
        cfg = WakeWordConfig()
        assert "hey sentra" in cfg.wake_words
        assert cfg.confidence_threshold == 0.7
        assert cfg.cooldown_seconds == 2.0


class TestEnergyWakeWordEngine:
    def _make_pcm(self, amplitude: int, n_samples: int = 1600) -> bytes:
        """Generate PCM audio with a given amplitude."""
        return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))

    def test_no_detection_on_silence(self):
        cfg = WakeWordConfig(energy_threshold=500.0, energy_min_duration_ms=80)
        engine = EnergyWakeWordEngine(cfg)
        silence = self._make_pcm(0)
        events = engine.process_audio(silence)
        assert events == []

    def test_detection_on_loud_audio(self):
        cfg = WakeWordConfig(
            energy_threshold=100.0,
            energy_min_duration_ms=80,
            frame_length_ms=80,
        )
        engine = EnergyWakeWordEngine(cfg)
        loud = self._make_pcm(5000)
        events = engine.process_audio(loud)
        assert len(events) == 1
        assert events[0].wake_word == "hey sentra"
        assert events[0].confidence > 0

    def test_min_duration_requirement(self):
        cfg = WakeWordConfig(
            energy_threshold=100.0,
            energy_min_duration_ms=320,  # Needs 4 frames
            frame_length_ms=80,
        )
        engine = EnergyWakeWordEngine(cfg)
        loud = self._make_pcm(5000)

        # First three frames should not trigger
        for _ in range(3):
            events = engine.process_audio(loud)
            assert events == []

        # Fourth frame should trigger
        events = engine.process_audio(loud)
        assert len(events) == 1

    def test_reset_on_silence(self):
        cfg = WakeWordConfig(
            energy_threshold=100.0,
            energy_min_duration_ms=160,
            frame_length_ms=80,
        )
        engine = EnergyWakeWordEngine(cfg)
        loud = self._make_pcm(5000)
        silence = self._make_pcm(0)

        engine.process_audio(loud)  # 1 frame
        engine.process_audio(silence)  # Reset
        engine.process_audio(loud)  # 1 frame again
        events = engine.process_audio(loud)  # 2nd frame → trigger
        assert len(events) == 1

    def test_cleanup(self):
        cfg = WakeWordConfig()
        engine = EnergyWakeWordEngine(cfg)
        engine._speech_frames = 10
        engine.cleanup()
        assert engine._speech_frames == 0


class TestWakeWordDetector:
    def test_instantiation(self):
        detector = WakeWordDetector()
        assert "wake_word_detection" in detector.capabilities
        assert detector.config.wake_words == ["hey sentra"]

    def test_stats(self):
        detector = WakeWordDetector()
        stats = detector.stats()
        assert stats["total_detections"] == 0
        assert stats["active"]
        assert "hey sentra" in stats["wake_words"]

    def test_set_active(self):
        detector = WakeWordDetector()
        detector.set_active(False)
        assert not detector._active
        detector.set_active(True)
        assert detector._active

    def test_custom_config(self):
        cfg = WakeWordConfig(
            wake_words=["hello computer"],
            confidence_threshold=0.9,
            cooldown_seconds=5.0,
        )
        detector = WakeWordDetector(config=cfg)
        assert detector.config.wake_words == ["hello computer"]
        assert detector.config.confidence_threshold == 0.9
