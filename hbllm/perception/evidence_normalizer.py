"""Evidence Normalizer — Observation → PerceptualEvidenceNode.

Converts modality-specific observations into the universal
``PerceptualEvidenceNode`` contract.  This is the critical boundary
that prevents perception providers from becoming coupled to HCIR.

Architecture::

    Provider → PerceptualObservation → EvidenceNormalizer → PerceptualEvidenceNode → HCIR

Providers produce typed observations (``SpeechEvidence``,
``SoundEventEvidence``, ``VisualAssessment``).  The normalizer
converts these into ``PerceptualEvidenceNode`` instances with
proper propositions, spatial context, and temporal validity.

Design invariants:
    - Providers NEVER construct HCIR nodes.
    - HCIR should own the canonical representation.
    - Raw media never enters HCIR — only propositions about it.
    - The normalizer is extensible for new modalities.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from hbllm.hcir.graph import PerceptualEvidenceNode
from hbllm.hcir.proposition import (
    BoundingBox,
    Proposition,
    SpatialContext,
    TemporalValidity,
)
from hbllm.hcir.types import EvidenceStrength, PerceptualEpistemicProfile
from hbllm.perception.providers.audio_evidence import (
    AcousticSceneEvidence,
    AudioAssessment,
    SoundEventEvidence,
    SpeechEvidence,
)
from hbllm.perception.providers.evidence import VisualAssessment
from hbllm.perception.providers.provider_provenance import ProviderProvenance

logger = logging.getLogger(__name__)


class EvidenceNormalizer:
    """Converts modality-specific observations into PerceptualEvidenceNodes.

    Providers should NOT know about HCIR graph structure.
    This normalizer owns the canonical representation boundary.

    Architecture::

        Whisper → SpeechEvidence      → normalize_speech()      → PerceptualEvidenceNode
        YAMNet  → SoundEventEvidence  → normalize_sound_event() → PerceptualEvidenceNode
        YOLO    → VisualAssessment    → normalize_visual()       → PerceptualEvidenceNode
        IMU     → dict                → normalize_sensor()       → PerceptualEvidenceNode

    Usage::

        normalizer = EvidenceNormalizer()

        # Audio speech
        nodes = normalizer.normalize_speech(speech_evidence)

        # Visual detection
        nodes = normalizer.normalize_visual(visual_assessment)

        # Generic sensor
        node = normalizer.normalize_sensor(
            sensor_id="imu_01",
            predicate="acceleration",
            value=[0.1, -0.3, 9.8],
            value_type="vector3",
        )
    """

    # ── Speech Evidence ──────────────────────────────────────────────────

    def normalize_speech(
        self,
        evidence: SpeechEvidence,
        *,
        observed_at: float | None = None,
    ) -> PerceptualEvidenceNode:
        """Normalize speech transcription evidence.

        Produces a proposition like::

            subject="utterance_42"
            predicate="transcribed_as"
            object_value="turn on the lights"
        """
        obs = evidence.observation
        prov = evidence.provider_provenance
        now = time.time()

        proposition = Proposition(
            subject=obs.observation_id,
            predicate="transcribed_as",
            object_value=evidence.transcript,
            object_type="transcript",
        )

        temporal = TemporalValidity(
            observed_at=observed_at or obs.temporal.start_time or now,
            received_at=now,
            valid_from=obs.temporal.start_time or None,
            valid_until=None,  # Transcripts don't expire
        )

        # Build modality-specific payload (opaque to HCIR)
        payload: dict[str, Any] = {
            "language": evidence.language,
            "is_partial": evidence.is_partial,
        }
        if evidence.speaker_ref is not None:
            payload["speaker_id"] = evidence.speaker_ref.speaker_id
            payload["speaker_confidence"] = evidence.speaker_ref.confidence
        if evidence.paralinguistic is not None:
            payload["tone"] = getattr(evidence.paralinguistic, "tone", "")
            payload["tone_confidence"] = getattr(evidence.paralinguistic, "confidence", 0.0)

        return PerceptualEvidenceNode(
            proposition=proposition,
            temporal_validity=temporal,
            modality="audio",
            evidence_type=EvidenceStrength.OBSERVATIONAL,
            strength=evidence.confidence,
            provider_provenance=_provenance_to_dict(prov),
            payload=payload,
        )

    # ── Sound Event Evidence ─────────────────────────────────────────────

    def normalize_sound_event(
        self,
        evidence: SoundEventEvidence,
        *,
        observed_at: float | None = None,
    ) -> PerceptualEvidenceNode:
        """Normalize acoustic event classification evidence.

        Produces a proposition like::

            subject="audio_event_82"
            predicate="indicates"
            object_value="doorbell"
        """
        obs = evidence.observation
        prov = evidence.provider_provenance
        now = time.time()

        proposition = Proposition(
            subject=obs.observation_id,
            predicate="indicates",
            object_value=evidence.event_type,
            object_type="event_class",
        )

        temporal = TemporalValidity(
            observed_at=observed_at or obs.temporal.start_time or now,
            received_at=now,
        )

        payload: dict[str, Any] = {
            "is_critical": evidence.is_critical,
            "event_state": evidence.event_state.value
            if hasattr(evidence.event_state, "value")
            else str(evidence.event_state),
        }
        if evidence.top_classes:
            payload["top_classes"] = [
                {"label": label, "confidence": conf} for label, conf in evidence.top_classes
            ]

        # Build candidates for epistemic evaluation
        candidates = [{"label": label, "confidence": conf} for label, conf in evidence.top_classes]

        return PerceptualEvidenceNode(
            proposition=proposition,
            temporal_validity=temporal,
            modality="audio",
            evidence_type=EvidenceStrength.OBSERVATIONAL,
            strength=evidence.confidence,
            provider_provenance=_provenance_to_dict(prov),
            candidates=candidates,
            payload=payload,
        )

    # ── Acoustic Scene Evidence ──────────────────────────────────────────

    def normalize_acoustic_scene(
        self,
        evidence: AcousticSceneEvidence,
        *,
        observed_at: float | None = None,
    ) -> PerceptualEvidenceNode:
        """Normalize acoustic scene characterization evidence.

        Produces a proposition like::

            subject="acoustic_scene"
            predicate="characterized_as"
            object_value={"indoor": True, "noise_level": 0.3, ...}
        """
        obs = evidence.observation
        prov = evidence.provider_provenance
        now = time.time()

        scene_value: dict[str, Any] = {
            "indoor": evidence.indoor,
            "speech_present": evidence.speech_present,
            "noise_level": evidence.noise_level,
            "estimated_activity": evidence.estimated_activity,
            "scene_tags": evidence.scene_tags,
        }

        proposition = Proposition(
            subject=obs.observation_id,
            predicate="characterized_as",
            object_value=scene_value,
            object_type="acoustic_scene",
        )

        temporal = TemporalValidity(
            observed_at=observed_at or obs.temporal.start_time or now,
            received_at=now,
        )

        return PerceptualEvidenceNode(
            proposition=proposition,
            temporal_validity=temporal,
            modality="audio",
            evidence_type=EvidenceStrength.OBSERVATIONAL,
            strength=evidence.confidence,
            provider_provenance=_provenance_to_dict(prov),
            payload=scene_value,
        )

    # ── Audio Assessment (compound) ──────────────────────────────────────

    def normalize_audio_assessment(
        self,
        assessment: AudioAssessment,
        *,
        observed_at: float | None = None,
    ) -> list[PerceptualEvidenceNode]:
        """Normalize a full audio assessment into multiple evidence nodes.

        An ``AudioAssessment`` can contain speech, events, scene, and
        source evidence.  Each produces a separate
        ``PerceptualEvidenceNode``.
        """
        nodes: list[PerceptualEvidenceNode] = []

        if assessment.speech is not None:
            nodes.append(self.normalize_speech(assessment.speech, observed_at=observed_at))

        for event in assessment.events:
            nodes.append(self.normalize_sound_event(event, observed_at=observed_at))

        if assessment.scene is not None:
            nodes.append(self.normalize_acoustic_scene(assessment.scene, observed_at=observed_at))

        return nodes

    # ── Visual Assessment ────────────────────────────────────────────────

    def normalize_visual(
        self,
        assessment: VisualAssessment,
        *,
        observed_at: float | None = None,
        frame_id: str = "",
        bounding_box: BoundingBox | None = None,
        depth_meters: float | None = None,
    ) -> PerceptualEvidenceNode:
        """Normalize visual detection/recognition evidence.

        Produces a proposition like::

            subject="visual_obs_abc123"
            predicate="classified_as"
            object_value="screwdriver"
        """
        now = time.time()
        ev = assessment.evidence
        ep = assessment.epistemic_profile

        # Best candidate label
        best_label = "unknown"
        best_confidence = 0.0
        if assessment.candidate_concepts:
            best = assessment.candidate_concepts[0]
            best_label = best.label
            best_confidence = best.best_similarity

        subject = (
            getattr(ev.provenance, "source_node", "")
            or getattr(ev.provenance, "created_by", "")
            or "visual_obs"
        )

        proposition = Proposition(
            subject=subject,
            predicate="classified_as",
            object_value=best_label,
            object_type="visual_concept",
        )

        spatial = None
        if bounding_box or depth_meters:
            spatial = SpatialContext(
                frame_id=frame_id,
                bounding_box=bounding_box,
                depth_meters=depth_meters,
            )

        temporal = TemporalValidity(
            observed_at=observed_at if observed_at is not None else float(getattr(ev.provenance, "timestamp", now)),
            received_at=now,
        )

        # Build candidates from concept candidates
        candidates = [
            {
                "label": c.label,
                "confidence": c.best_similarity,
                "mean_similarity": c.mean_similarity,
                "matching_observations": c.matching_observations,
            }
            for c in assessment.candidate_concepts
        ]

        # Map visual epistemic profile
        epistemic_profile = PerceptualEpistemicProfile(
            sensory_clarity=ep.perceptual_similarity,
            model_confidence=ep.evidence_strength,
            temporal_stability=ep.source_reliability,
        )

        payload: dict[str, Any] = {
            "image_hash": ev.image_hash,
            "embedding_ref": ev.embedding.ref if hasattr(ev.embedding, "ref") else "",
        }

        return PerceptualEvidenceNode(
            proposition=proposition,
            spatial=spatial,
            temporal_validity=temporal,
            modality="visual",
            evidence_type=EvidenceStrength.OBSERVATIONAL,
            strength=best_confidence,
            epistemic_profile=epistemic_profile,
            candidates=candidates,
            payload=payload,
        )

    # ── Generic Sensor ───────────────────────────────────────────────────

    def normalize_sensor(
        self,
        *,
        sensor_id: str,
        predicate: str,
        value: Any,
        value_type: str = "",
        modality: str = "sensor",
        confidence: float = 1.0,
        spatial: SpatialContext | None = None,
        observed_at: float | None = None,
        provider_provenance: ProviderProvenance | None = None,
    ) -> PerceptualEvidenceNode:
        """Normalize a generic sensor reading.

        Supports any sensor type: IMU, temperature, GPS, pressure,
        humidity, LiDAR point clouds, etc.

        Examples::

            # IMU
            normalizer.normalize_sensor(
                sensor_id="imu_01",
                predicate="acceleration",
                value=[0.1, -0.3, 9.8],
                value_type="vector3",
            )

            # Temperature
            normalizer.normalize_sensor(
                sensor_id="sensor_4",
                predicate="temperature",
                value=22.5,
                value_type="celsius",
            )

            # Door state
            normalizer.normalize_sensor(
                sensor_id="door_7",
                predicate="state",
                value="OPEN",
                value_type="enum",
            )
        """
        now = time.time()

        proposition = Proposition(
            subject=sensor_id,
            predicate=predicate,
            object_value=value,
            object_type=value_type,
        )

        temporal = TemporalValidity(
            observed_at=observed_at or now,
            received_at=now,
        )

        return PerceptualEvidenceNode(
            proposition=proposition,
            spatial=spatial,
            temporal_validity=temporal,
            modality=modality,
            evidence_type=EvidenceStrength.OBSERVATIONAL,
            strength=confidence,
            provider_provenance=_provenance_to_dict(provider_provenance)
            if provider_provenance
            else None,
        )


# ─── Helpers ──────────────────────────────────────────────────────────────


def _provenance_to_dict(prov: ProviderProvenance | None) -> dict[str, Any] | None:
    """Convert ProviderProvenance to dict for HCIR node field.

    Reuses the existing ``ProviderProvenance`` type rather than
    creating a competing provenance system.
    """
    if prov is None:
        return None
    result: dict[str, Any] = {
        "provider": prov.provider,
        "model": prov.model,
        "version": prov.version,
    }
    if prov.device:
        result["device"] = prov.device
    if prov.extra:
        result["extra"] = prov.extra
    return result
