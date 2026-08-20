"""Perception Fuser — cross-modal temporal alignment and fusion.

Collects perception events from multiple modalities (audio, visual, system)
in a sliding time window and produces fused context with correlation candidates
for ComprehensionStream and HCIR.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.network.messages import Message, MessageType
from hbllm.perception.correlation_engine import (
    CorrelationCandidate,
    CorrelationEngine,
    ObservationEnvelope,
)

if TYPE_CHECKING:
    from hbllm.perception.correlation_transaction import CorrelationTransaction

logger = logging.getLogger(__name__)


@dataclass
class PerceptionEvent:
    """A single perception event from any modality."""

    modality: str  # "audio", "visual", "system", "text", etc.
    content: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


@dataclass
class FusedContext:
    """Cross-modal fused perception context.

    Groups temporally aligned events from different modalities
    into a single context object for downstream processing.
    """

    events: list[PerceptionEvent] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    modalities: set[str] = field(default_factory=set)
    correlations: list[CorrelationCandidate] = field(default_factory=list)

    @property
    def is_multimodal(self) -> bool:
        """True if events span multiple modalities."""
        return len(self.modalities) > 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict suitable for bus payload."""
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "modalities": sorted(self.modalities),
            "is_multimodal": self.is_multimodal,
            "event_count": len(self.events),
            "correlation_count": len(self.correlations),
        }

        # Group by modality
        for modality in self.modalities:
            modal_events = [e for e in self.events if e.modality == modality]
            result[modality] = [
                {
                    "content": e.content,
                    "confidence": e.confidence,
                    "source": e.source,
                    "age_s": round(e.age_seconds(), 2),
                }
                for e in modal_events
            ]

        return result

    def summary_text(self) -> str:
        """Generate a natural-language summary of fused context."""
        parts: list[str] = []
        for modality in sorted(self.modalities):
            modal_events = [e for e in self.events if e.modality == modality]
            for event in modal_events:
                parts.append(f"[{modality}] {event.content}")
        return " | ".join(parts) if parts else ""


class PerceptionFuser:
    """Sliding-window cross-modal perception fusion and correlation.

    Collects events from different perception sources (audio, visual,
    system state) and when multiple modalities fire within the fusion
    window, computes correlation candidates and produces a FusedContext.

    Usage::

        fuser = PerceptionFuser(window_seconds=5.0)

        # Subscribe to perception topics
        bus.subscribe("perception.*", fuser.on_perception_event)

        # Fuser publishes to "perception.fused" when multimodal context is ready
    """

    def __init__(
        self,
        window_seconds: float = 5.0,
        min_modalities: int = 2,
        max_events: int = 50,
        bus: Any = None,
        correlation_engine: CorrelationEngine | None = None,
        correlation_transaction: CorrelationTransaction | None = None,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_modalities = min_modalities
        self.max_events = max_events
        self.bus = bus
        self._correlation_engine = correlation_engine or CorrelationEngine(
            max_temporal_gap=window_seconds,
        )
        self._correlation_transaction = correlation_transaction

        # Sliding window of recent perception events
        self._window: deque[PerceptionEvent] = deque(maxlen=max_events)

        # Debounce timer for fusion
        self._fusion_task: asyncio.Task[None] | None = None
        self._fusion_delay = 0.5  # Wait 500ms after last event before fusing

        # Stats
        self._total_events = 0
        self._total_fusions = 0
        self._total_correlations = 0

    def correlate_window(self) -> list[CorrelationCandidate]:
        """Compute cross-modal correlations among events in the window."""
        self._prune_window()
        envelopes: list[ObservationEnvelope] = []

        for event in self._window:
            obs_id = str(event.metadata.get("observation_id", "") or f"{event.modality}_{id(event)}")
            duration = float(event.metadata.get("duration", 0.5))
            direction = event.metadata.get("direction_degrees")
            direction_val = float(direction) if direction is not None else None

            envelopes.append(
                ObservationEnvelope(
                    observation_id=obs_id,
                    modality=event.modality,
                    start_time=event.timestamp,
                    end_time=event.timestamp + duration,
                    direction_degrees=direction_val,
                )
            )

        # Group envelopes by modality
        candidates: list[CorrelationCandidate] = []
        modalities = {e.modality for e in envelopes}
        mod_list = sorted(modalities)

        for i in range(len(mod_list)):
            for j in range(i + 1, len(mod_list)):
                m1_envs = [e for e in envelopes if e.modality == mod_list[i]]
                m2_envs = [e for e in envelopes if e.modality == mod_list[j]]
                matches = self._correlation_engine.correlate_batch(m1_envs, m2_envs)
                candidates.extend(matches)

        if self._correlation_transaction is not None and candidates:
            self._correlation_transaction.commit_batch(candidates)

        self._total_correlations += len(candidates)
        return candidates

    def _prune_window(self) -> None:
        """Remove events older than the window."""
        cutoff = time.time() - self.window_seconds
        while self._window and self._window[0].timestamp < cutoff:
            self._window.popleft()

    async def on_perception_event(self, message: Message) -> None:
        """Process incoming perception events from the bus.

        Maps bus message topics to modality types and adds them to the
        sliding window. Triggers fusion check after a debounce delay.
        """
        topic = message.topic
        payload = message.payload

        # Map topic to modality
        modality = self._topic_to_modality(topic)
        content = self._extract_content(payload, modality)

        if not content:
            return

        event = PerceptionEvent(
            modality=modality,
            content=content,
            confidence=payload.get("confidence", 1.0),
            source=message.source_node_id,
            metadata={k: v for k, v in payload.items() if k != "text"},
        )

        self._window.append(event)
        self._total_events += 1

        # Debounced fusion check
        if self._fusion_task is not None and not self._fusion_task.done():
            self._fusion_task.cancel()
        self._fusion_task = asyncio.create_task(self._delayed_fusion())

    def _topic_to_modality(self, topic: str) -> str:
        """Map a bus topic to a modality name."""
        modality_map = {
            "sensory.audio.in": "audio",
            "sensory.audio.out": "audio",
            "perception.audio": "audio",
            "sensory.vision.in": "visual",
            "perception.vision": "visual",
            "perception.screen": "visual",
            "perception.filesystem.changes": "system",
            "perception.system.health_alert": "system",
            "system.user_idle": "system",
            "perception.calendar.upcoming": "system",
        }

        for prefix, modality in modality_map.items():
            if topic.startswith(prefix):
                return modality

        # Default: infer from topic structure
        if "audio" in topic:
            return "audio"
        elif "vision" in topic or "screen" in topic or "image" in topic:
            return "visual"
        elif "system" in topic or "filesystem" in topic:
            return "system"
        else:
            return "text"

    def _extract_content(self, payload: dict[str, Any], modality: str) -> str:
        """Extract human-readable content from a perception payload."""
        # Try common payload fields
        for key in ("text", "summary", "description", "content"):
            if key in payload and payload[key]:
                return str(payload[key])[:300]

        # Modality-specific extraction
        if modality == "visual":
            if "objects" in payload:
                return f"Detected objects: {', '.join(str(o) for o in payload['objects'][:5])}"
            if "screen_text" in payload:
                return str(payload["screen_text"])[:300]
        elif modality == "system":
            if "type" in payload:
                return f"System event: {payload['type']}"
            if "path" in payload:
                return f"File change: {payload['path']}"

        return ""

    async def _delayed_fusion(self) -> None:
        """Wait for debounce period, then attempt fusion."""
        try:
            await asyncio.sleep(self._fusion_delay)
            await self._try_fuse()
        except asyncio.CancelledError:
            pass

    async def _try_fuse(self) -> None:
        """Check if we have enough modalities for fusion."""
        self._prune_window()

        if not self._window:
            return

        # Count distinct modalities in the current window
        modalities = {e.modality for e in self._window}

        if len(modalities) >= self.min_modalities:
            correlations = self.correlate_window()
            fused = FusedContext(
                events=list(self._window),
                timestamp=time.time(),
                modalities=modalities,
                correlations=correlations,
            )

            self._total_fusions += 1

            logger.info(
                "[PerceptionFuser] Fused %d events across %s (%d correlations)",
                len(fused.events),
                sorted(modalities),
                len(correlations),
            )

            # Publish fused context
            if self.bus is not None:
                await self.bus.publish(
                    "perception.fused",
                    Message(
                        type=MessageType.EVENT,
                        source_node_id="perception_fuser",
                        topic="perception.fused",
                        payload=fused.to_dict(),
                    ),
                )

            # Clear window after fusion to avoid re-fusing same events
            self._window.clear()

    def ingest_event(self, event: PerceptionEvent) -> FusedContext | None:
        """Directly ingest a PerceptionEvent and return FusedContext if threshold met."""
        self._total_events += 1
        self._window.append(event)
        self._prune_window()

        modalities = {e.modality for e in self._window}
        if len(modalities) >= self.min_modalities:
            correlations = self.correlate_window()
            fused = FusedContext(
                events=list(self._window),
                timestamp=time.time(),
                modalities=modalities,
                correlations=correlations,
            )
            self._total_fusions += 1
            return fused
        return None

    def ingest_audio_assessment(self, assessment: Any) -> FusedContext | None:
        """Ingest AudioAssessment into cross-modal fusion window."""
        content_parts = []
        if getattr(assessment, "speech", None) is not None:
            content_parts.append(f"speech: {assessment.speech.transcript}")
        for ev in getattr(assessment, "events", []):
            content_parts.append(f"sound: {ev.event_type} ({ev.confidence:.2f})")

        content = " | ".join(content_parts) or "audio event"
        event = PerceptionEvent(
            modality="audio",
            content=content,
            confidence=getattr(assessment.epistemic_profile, "combined", 1.0),
            source="audio_perception",
            timestamp=getattr(assessment.observation.temporal, "start_time", time.time()),
            metadata={"observation_id": assessment.observation.observation_id},
        )
        return self.ingest_event(event)

    def ingest_audio_observation(self, node: Any) -> FusedContext | None:
        """Ingest AudioObservationNode into cross-modal fusion window."""
        content = getattr(node, "transcript", "") or getattr(node, "event_type", "audio")
        event = PerceptionEvent(
            modality="audio",
            content=f"audio: {content}",
            confidence=1.0,
            source=getattr(node, "embedding_model", "audio"),
            timestamp=getattr(node, "start_time", time.time()) or time.time(),
            metadata={"observation_id": getattr(node, "id", "")},
        )
        return self.ingest_event(event)

    def ingest_visual_assessment(self, assessment: Any) -> FusedContext | None:
        """Ingest VisualAssessment into cross-modal fusion window."""
        label = getattr(assessment, "proposed_label", "") or "visual entity"
        event = PerceptionEvent(
            modality="visual",
            content=f"visual: {label}",
            confidence=1.0,
            source="visual_perception",
            timestamp=time.time(),
        )
        return self.ingest_event(event)

    def get_current_window(self) -> list[PerceptionEvent]:
        """Get current events in the sliding window."""
        self._prune_window()
        return list(self._window)

    def get_modality_summary(self) -> dict[str, int]:
        """Get count of events per modality in current window."""
        self._prune_window()
        counts: dict[str, int] = {}
        for event in self._window:
            counts[event.modality] = counts.get(event.modality, 0) + 1
        return counts

    def snapshot(self) -> dict[str, Any]:
        """Introspection snapshot."""
        self._prune_window()
        return {
            "window_size": len(self._window),
            "window_seconds": self.window_seconds,
            "modalities_in_window": sorted({e.modality for e in self._window}),
            "total_events_processed": self._total_events,
            "total_fusions": self._total_fusions,
            "min_modalities": self.min_modalities,
        }
