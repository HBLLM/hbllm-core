"""Correlation Engine — pure geometry/time cross-modal alignment.

Produces CorrelationCandidate from pairs of PerceptualObservation.
Contains NO semantic interpretation — only measurable relationships.

The Correlation Engine answers:
    "These two observations happened near each other in time/space."

It does NOT answer:
    "The person made the footsteps."
    "The alarm caused the visual alert."

Those are beliefs. Beliefs belong to the epistemic layer.

Architecture:
    VisualObservation ──?
                            └── CorrelationCandidate (score, Δtime, overlap)
    AudioObservation  ───?

    Then HCIR transaction creates:
        VISUAL_OBSERVATION ── CORRELATES_WITH ── AUDIO_OBSERVATION

Usage::

    engine = CorrelationEngine(max_temporal_gap=5.0)
    candidate = engine.correlate(visual_obs, audio_obs)
    if candidate and candidate.score > 0.5:
        transaction.commit_correlation(candidate)
"""

from __future__ import annotations

from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════
# Correlation Candidate — measurable association, NOT semantic identity
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CorrelationCandidate:
    """Measurable association between two perceptual observations.

    Contains NO MEANING. Only measurable relationships.

    Attributes:
        source_observation_id: First observation.
        target_observation_id: Second observation.
        source_modality: Modality of the first observation.
        target_modality: Modality of the second observation.
        temporal_overlap: How much the observations overlap in time (0.0-1.0).
        spatial_overlap: Spatial proximity if available (0.0-1.0), or None.
        delta_time_ms: Temporal gap in milliseconds (signed: + means target after source).
        score: Combined correlation strength (0.0-1.0).

    """

    source_observation_id: str
    target_observation_id: str
    source_modality: str
    target_modality: str
    temporal_overlap: float
    spatial_overlap: float | None
    delta_time_ms: float
    score: float

    @property
    def is_cross_modal(self) -> bool:
        """True if the two observations are from different modalities."""
        return self.source_modality != self.target_modality


# ═══════════════════════════════════════════════════════════════════════════
# Observation Envelope — lightweight temporal/spatial descriptor
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ObservationEnvelope:
    """Temporal and optional spatial envelope for an observation.

    This is the minimal information the correlation engine needs.
    It does NOT contain the full observation — only what's needed
    for geometry/time alignment.

    Attributes:
        observation_id: Unique observation identifier.
        modality: Perception modality ("audio", "vision", ...).
        start_time: Observation start (epoch seconds).
        end_time: Observation end (epoch seconds).
        direction_degrees: Optional spatial direction (0-360), or None.

    """

    observation_id: str
    modality: str
    start_time: float
    end_time: float
    direction_degrees: float | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Correlation Engine — pure stateless function
# ═══════════════════════════════════════════════════════════════════════════


class CorrelationEngine:
    """Pure geometry/time correlation between perceptual observations.

    Stateless. Takes pairs of ObservationEnvelope and produces
    CorrelationCandidate if they are temporally (and optionally
    spatially) aligned.

    Args:
        max_temporal_gap: Maximum time gap (seconds) for correlation.
        spatial_threshold: Maximum angular difference (degrees) for spatial
            correlation. Only applied if both observations have direction.
        temporal_weight: Weight of temporal overlap in combined score.
        spatial_weight: Weight of spatial overlap in combined score.

    """

    def __init__(
        self,
        max_temporal_gap: float = 5.0,
        spatial_threshold: float = 30.0,
        temporal_weight: float = 0.7,
        spatial_weight: float = 0.3,
    ) -> None:
        self.max_temporal_gap = max_temporal_gap
        self.spatial_threshold = spatial_threshold
        self.temporal_weight = temporal_weight
        self.spatial_weight = spatial_weight

    def correlate(
        self,
        source: ObservationEnvelope,
        target: ObservationEnvelope,
    ) -> CorrelationCandidate | None:
        """Attempt to correlate two observations.

        Returns None if the observations are too far apart in
        time or space to be meaningfully associated.

        Args:
            source: First observation envelope.
            target: Second observation envelope.

        Returns:
            CorrelationCandidate if aligned, None otherwise.

        """
        # ── Temporal alignment ──
        temporal_overlap = self._compute_temporal_overlap(source, target)
        delta_time_ms = (target.start_time - source.start_time) * 1000.0

        # Check temporal gap
        gap = self._temporal_gap(source, target)
        if gap > self.max_temporal_gap:
            return None

        # ── Spatial alignment (optional) ──
        spatial_overlap = self._compute_spatial_overlap(source, target)

        # ── Combined score ──
        if spatial_overlap is not None:
            score = self.temporal_weight * temporal_overlap + self.spatial_weight * spatial_overlap
        else:
            # No spatial data — score based on temporal alone
            score = temporal_overlap

        if score <= 0.0:
            return None

        return CorrelationCandidate(
            source_observation_id=source.observation_id,
            target_observation_id=target.observation_id,
            source_modality=source.modality,
            target_modality=target.modality,
            temporal_overlap=temporal_overlap,
            spatial_overlap=spatial_overlap,
            delta_time_ms=delta_time_ms,
            score=score,
        )

    def correlate_batch(
        self,
        sources: list[ObservationEnvelope],
        targets: list[ObservationEnvelope],
    ) -> list[CorrelationCandidate]:
        """Correlate all source/target pairs, returning non-None matches.

        Args:
            sources: Source observation envelopes.
            targets: Target observation envelopes.

        Returns:
            List of CorrelationCandidates sorted by score (descending).

        """
        candidates: list[CorrelationCandidate] = []
        for src in sources:
            for tgt in targets:
                if src.observation_id == tgt.observation_id:
                    continue
                result = self.correlate(src, tgt)
                if result is not None:
                    candidates.append(result)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    # ── Internal computations ────────────────────────────────────────────

    def _temporal_gap(
        self,
        a: ObservationEnvelope,
        b: ObservationEnvelope,
    ) -> float:
        """Compute the temporal gap between two observations (seconds)."""
        if a.end_time <= b.start_time:
            return b.start_time - a.end_time
        if b.end_time <= a.start_time:
            return a.start_time - b.end_time
        return 0.0  # Overlapping

    def _compute_temporal_overlap(
        self,
        a: ObservationEnvelope,
        b: ObservationEnvelope,
    ) -> float:
        """Compute temporal overlap ratio (0.0-1.0).

        0.0 = no overlap (but within max_temporal_gap)
        1.0 = perfect overlap
        Values between = partial overlap, scaled by gap proximity
        """
        # Overlap duration
        overlap_start = max(a.start_time, b.start_time)
        overlap_end = min(a.end_time, b.end_time)
        overlap_duration = max(0.0, overlap_end - overlap_start)

        if overlap_duration > 0:
            # Actual overlap: ratio of overlap to shorter observation
            shorter = min(
                a.end_time - a.start_time,
                b.end_time - b.start_time,
            )
            if shorter <= 0:
                shorter = 0.001  # Prevent division by zero
            return min(1.0, overlap_duration / shorter)

        # No overlap — decay score by gap distance
        gap = self._temporal_gap(a, b)
        if gap >= self.max_temporal_gap:
            return 0.0
        return max(0.0, 1.0 - (gap / self.max_temporal_gap))

    def _compute_spatial_overlap(
        self,
        a: ObservationEnvelope,
        b: ObservationEnvelope,
    ) -> float | None:
        """Compute spatial overlap (0.0-1.0), or None if no spatial data.

        Uses angular difference between direction estimates.
        """
        if a.direction_degrees is None or b.direction_degrees is None:
            return None

        # Angular difference (handling wraparound)
        diff = abs(a.direction_degrees - b.direction_degrees) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff

        if diff >= self.spatial_threshold:
            return 0.0

        return max(0.0, 1.0 - (diff / self.spatial_threshold))
