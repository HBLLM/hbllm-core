"""Environmental Simulation Module — external perception plugins for non-symbolic streams."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from hbllm.perception.reality_bus import (
    EventOrigin,
    PerceptionEvent,
    PerceptionModality,
    RealityEventBus,
)

logger = logging.getLogger(__name__)


class EnvironmentSimulator:
    """Simulates physical/external environments and generates perception streams.

    Acts as an external perception hook for non-symbolic streams (e.g. synthetic audio,
    vision frames, and sensor inputs), feeding them into the RealityEventBus.
    """

    def __init__(self, bus: RealityEventBus, tick_rate_seconds: float = 1.0) -> None:
        self.bus = bus
        self.tick_rate_seconds = tick_rate_seconds
        self.is_running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._last_event: PerceptionEvent | None = None

    async def start(self) -> None:
        """Start the background environmental perception generation loop."""
        if self.is_running:
            return
        self.is_running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Environment Simulator started with tick rate %.2fs", self.tick_rate_seconds)

    async def stop(self) -> None:
        """Stop the background loop."""
        if not self.is_running:
            return
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("Environment Simulator stopped")

    async def _run_loop(self) -> None:
        """Periodic tick generator to simulate sensory input."""
        while self.is_running:
            try:
                await asyncio.sleep(self.tick_rate_seconds)
                # Randomly fire vision, audio, or sensor events
                event_choice = random.choice(["vision", "audio", "sensor"])
                if event_choice == "vision":
                    await self.simulate_vision_event()
                elif event_choice == "audio":
                    await self.simulate_audio_event()
                else:
                    await self.simulate_sensor_event()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in environment simulation loop: %s", e)

    async def simulate_vision_event(
        self, detected_objects: list[str] | None = None, confidence: float = 0.95
    ) -> PerceptionEvent:
        """Inject a synthetic vision frame perception event."""
        objects = detected_objects or random.choice(
            [
                ["user_face", "keyboard"],
                ["coding_editor", "terminal_output"],
                ["coffee_mug", "user_hands"],
                ["documents", "pen"],
            ]
        )
        event = PerceptionEvent(
            entity_id="camera_0",
            event_type="vision_frame",
            sub_type="object_detection",
            modality=PerceptionModality.SENSOR,
            origin=EventOrigin.EXTERNAL,
            confidence=confidence,
            payload={
                "frame_resolution": "1920x1080",
                "detected_objects": objects,
                "scene_brightness": round(random.uniform(0.3, 0.9), 2),
                "is_motion_detected": True,
            },
        )
        await self.bus.ingest(event)
        self._last_event = event
        return event

    async def simulate_audio_event(
        self, transcription: str | None = None, duration: float = 2.5
    ) -> PerceptionEvent:
        """Inject a synthetic audio waveform/activity perception event."""
        text = transcription or random.choice(
            [
                "hbllm please run tests",
                "optimize thought budget for next step",
                "are the causal nodes aligned",
                "hello system",
            ]
        )
        event = PerceptionEvent(
            entity_id="microphone_0",
            event_type="audio_waveform",
            sub_type="voice_activity",
            modality=PerceptionModality.SENSOR,
            origin=EventOrigin.EXTERNAL,
            confidence=0.92,
            payload={
                "duration_seconds": duration,
                "voice_detected": True,
                "transcription": text,
                "decibels": random.randint(45, 75),
            },
        )
        await self.bus.ingest(event)
        self._last_event = event
        return event

    async def simulate_sensor_event(self) -> PerceptionEvent:
        """Inject a synthetic device/environment state perception event."""
        event = PerceptionEvent(
            entity_id="device_host",
            event_type="sensor_telemetry",
            sub_type="resource_metrics",
            modality=PerceptionModality.SYSTEM,
            origin=EventOrigin.SYSTEM,
            confidence=1.0,
            payload={
                "cpu_usage_percent": random.randint(15, 85),
                "memory_available_bytes": random.randint(2 * 1024**3, 8 * 1024**3),
                "device_temperature_celsius": round(random.uniform(35.0, 65.0), 1),
            },
        )
        await self.bus.ingest(event)
        self._last_event = event
        return event
