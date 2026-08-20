"""SNN Perception Gate — HBLLM Grounded Perception §V0.

Defines the typed decision output from the SNN temporal salience
mechanism.  The SNN isn't just a binary "embed or not" filter — it
determines both **whether** and **how much** perceptual computation
to spend.

Architecture:
    The SNN is a temporal salience mechanism deciding which
    physical-world changes deserve cognitive processing.

    cheap_signals → SNN_ensemble → PerceptionGateDecision
                                     ├── should_process
                                     ├── processing_level
                                     ├── urgency
                                     ├── event_type
                                     └── channels_fired
"""

from __future__ import annotations

import time
from enum import StrEnum

from pydantic import BaseModel, Field

from hbllm.brain.snn.neurons import SpikeEvent

# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════


class PerceptionProcessingLevel(StrEnum):
    """How much perceptual computation to spend.

    The SNN doesn't just decide *when* to embed — it decides
    *how much* to spend:

        NONE     — No processing (static scene, no change).
        LOW      — Cheap tracker update / crop (motion only).
        STANDARD — Full image embedding (scene change).
        HIGH     — Embedding + detection (novel appearance).
        URGENT   — Embedding + detection + OCR + active perception
                   (multiple channels, high novelty).
    """

    NONE = "none"
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    URGENT = "urgent"


class PerceptionEventType(StrEnum):
    """Why the SNN decided this input deserves processing.

    Multiple channels can fire on a single sample. The event type
    is determined by channel priority ordering.
    """

    # Visual events
    SCENE_CHANGE = "scene_change"
    NOVEL_APPEARANCE = "novel_appearance"
    ENTITY_CHANGE = "entity_change"
    MOTION_EVENT = "motion_event"
    STABILITY_SHIFT = "stability_shift"

    # Audio events
    SPEECH_ONSET = "speech_onset"
    ACOUSTIC_EVENT = "acoustic_event"
    AMBIENT_CHANGE = "ambient_change"
    TRANSIENT_BURST = "transient_burst"

    # Common
    HEARTBEAT = "heartbeat"  # Periodic confirmation (prevents blind spots)


# ═══════════════════════════════════════════════════════════════════════════
# Gate Decision
# ═══════════════════════════════════════════════════════════════════════════


class PerceptionGateDecision(BaseModel):
    """SNN temporal salience decision with processing level.

    Not a binary filter.  Determines both WHETHER and HOW MUCH
    perceptual computation to spend.

    The SNN outputs a **salience profile**: each channel contributes
    its spike strength.  The decision considers all channels, not
    just a single signal.

    Attributes:
        modality: Perception modality ("visual", "audio", etc.).
        should_process: Whether any processing is warranted.
        processing_level: How much compute to spend (NONE → URGENT).
        urgency: Highest channel spike strength [0.0, 1.0].
        event_type: Primary reason for the decision.
        channels_fired: Map of channel_name → spike strength.
        novelty: Novelty/change channel signal strength.
        temporal_significance: Average across fired channels.
        frame_index: Index of the frame/sample in the stream.
        timestamp: When the decision was made.

    """

    modality: str = "visual"
    should_process: bool
    processing_level: PerceptionProcessingLevel
    urgency: float = 0.0
    event_type: PerceptionEventType = PerceptionEventType.SCENE_CHANGE
    channels_fired: dict[str, float] = Field(default_factory=dict)
    novelty: float = 0.0
    temporal_significance: float = 0.0
    frame_index: int = 0
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def no_action(cls, frame_index: int) -> PerceptionGateDecision:
        """Convenience: no processing needed."""
        return cls(
            should_process=False,
            processing_level=PerceptionProcessingLevel.NONE,
            frame_index=frame_index,
        )

    @classmethod
    def heartbeat(cls, frame_index: int) -> PerceptionGateDecision:
        """Convenience: periodic confirmation to prevent blind spots."""
        return cls(
            should_process=True,
            processing_level=PerceptionProcessingLevel.LOW,
            urgency=0.1,
            event_type=PerceptionEventType.HEARTBEAT,
            frame_index=frame_index,
        )

    @classmethod
    def from_spikes(
        cls,
        fired: list[tuple[str, SpikeEvent]],
        frame_index: int,
    ) -> PerceptionGateDecision:
        """Construct decision from SNN spike output.

        Decision logic:
            1. If no channels fired → NONE.
            2. Event type from channel priority:
               scene > novelty > entity > motion > stability.
            3. Processing level from channel count + urgency:
               3+ channels or (novelty + high urgency) → URGENT
               scene or (novelty + medium urgency) → HIGH
               novelty or entity alone → STANDARD
               motion only → LOW
        """
        channels = {ch: spike.strength for ch, spike in fired}

        if not channels:
            return cls.no_action(frame_index)

        # ── Determine event type from channel priority ──
        if "scene" in channels:
            event_type = PerceptionEventType.SCENE_CHANGE
        elif "novelty" in channels:
            event_type = PerceptionEventType.NOVEL_APPEARANCE
        elif "entity" in channels:
            event_type = PerceptionEventType.ENTITY_CHANGE
        elif "motion" in channels:
            event_type = PerceptionEventType.MOTION_EVENT
        else:
            event_type = PerceptionEventType.STABILITY_SHIFT

        urgency = max(channels.values())
        num_channels = len(channels)
        has_novelty = "novelty" in channels

        # ── Determine processing level ──
        if num_channels >= 3 or (has_novelty and urgency > 0.8):
            level = PerceptionProcessingLevel.URGENT
        elif "scene" in channels or (has_novelty and urgency > 0.5):
            level = PerceptionProcessingLevel.HIGH
        elif has_novelty or "entity" in channels:
            level = PerceptionProcessingLevel.STANDARD
        else:
            level = PerceptionProcessingLevel.LOW

        return cls(
            should_process=True,
            processing_level=level,
            urgency=urgency,
            event_type=event_type,
            channels_fired=channels,
            novelty=channels.get("novelty", 0.0),
            temporal_significance=sum(channels.values()) / num_channels,
            frame_index=frame_index,
        )
