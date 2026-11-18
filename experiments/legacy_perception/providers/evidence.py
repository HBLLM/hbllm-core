"""Visual Evidence and Assessment Types — HBLLM Grounded Perception §V0.

Separates raw evidence from derived interpretation:

    ``VisualEvidence``   — What was observed (immutable perceptual measurement).
    ``VisualAssessment``  — What HBLLM currently thinks about it (mutable interpretation).

This separation is critical because interpretation can change
(today: "unknown", tomorrow: "screwdriver") while the original
evidence remains the same.

Also defines:
    ``EpistemicEvidenceProfile`` — Multi-dimensional epistemic confidence.
    ``ObservationMatch``         — A similar observation found in memory.
    ``CandidateRanking``         — Ranking with ambiguity signal.
    ``ConceptCandidate``         — A concept derived from observation matches.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hbllm.hcir.types import Provenance
from hbllm.perception.providers.types import VisualEmbedding

# ═══════════════════════════════════════════════════════════════════════════
# Epistemic Dimensions
# ═══════════════════════════════════════════════════════════════════════════


class EpistemicEvidenceProfile(BaseModel):
    """Structured epistemic dimensions for visual evidence.

    Separates confidence into independently updatable dimensions rather
    than collapsing everything into a single float or packing it into
    a reason string.

    Each dimension has a distinct meaning:

        label_provenance:     How reliable is the label source?
                              (user-labeled → 1.0, inferred → 0.5, unknown → 0.0)
        perceptual_similarity: How visually similar to known concepts?
                              (cosine similarity of best match)
        evidence_strength:    How much supporting evidence exists?
                              (observation count, exemplar coverage)
        source_reliability:   How trustworthy is the perception source?
                              (camera quality, lighting, occlusion)
    """

    label_provenance: float = 0.0
    perceptual_similarity: float = 0.0
    evidence_strength: float = 0.0
    source_reliability: float = 1.0

    @property
    def combined(self) -> float:
        """Weighted combination — individual dimensions remain accessible."""
        return (
            self.label_provenance * 0.3
            + self.perceptual_similarity * 0.3
            + self.evidence_strength * 0.25
            + self.source_reliability * 0.15
        )


# ═══════════════════════════════════════════════════════════════════════════
# Observation Matches (from Visual Memory)
# ═══════════════════════════════════════════════════════════════════════════


class ObservationMatch(BaseModel):
    """A similar observation found in visual memory.

    This is raw evidence — which stored observations are visually
    similar to a query embedding.  Concept resolution is a derived step.

    Attributes:
        observation_ref: EmbeddingRef ID in the vector store.
        similarity: Cosine similarity to the query.
        concept_node_id: Which concept this observation belongs to (if any).
        label: Label associated with the observation.
        timestamp: When the original observation was made.

    """

    observation_ref: str
    similarity: float
    concept_node_id: str | None = None
    label: str = ""
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Candidate Ranking (with Ambiguity)
# ═══════════════════════════════════════════════════════════════════════════


class CandidateRanking(BaseModel):
    """Candidate ranking with ambiguity signal.

    A small ``margin`` (e.g., screwdriver=0.83, wrench=0.81, margin=0.02)
    should be treated very differently from a large margin
    (screwdriver=0.83, wrench=0.55, margin=0.28).

    Attributes:
        best_score: Similarity of the top candidate.
        second_score: Similarity of the second-best candidate.
        margin: ``best_score - second_score``.
        ambiguity: ``1.0 - margin`` — higher means more ambiguous.

    """

    best_score: float = 0.0
    second_score: float = 0.0
    margin: float = 0.0
    ambiguity: float = 1.0

    @classmethod
    def from_scores(cls, scores: list[float]) -> CandidateRanking:
        """Build ranking from a sorted (descending) list of scores."""
        if not scores:
            return cls()
        best = scores[0]
        second = scores[1] if len(scores) > 1 else 0.0
        margin = best - second
        return cls(
            best_score=best,
            second_score=second,
            margin=margin,
            ambiguity=max(0.0, 1.0 - margin),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Concept Candidates (derived from observation grouping)
# ═══════════════════════════════════════════════════════════════════════════


class ConceptCandidate(BaseModel):
    """A candidate concept derived from grouping observation matches.

    Not a direct search result — this is derived by grouping
    ``ObservationMatch`` instances by their ``concept_node_id``.

    Attributes:
        concept_node_id: HCIR VisualConceptNode ID.
        label: Label of the concept.
        mean_similarity: Average similarity across matched observations.
        best_similarity: Best single-observation similarity.
        matching_observations: How many observations matched.
        total_observations: Total observations the concept has.

    """

    concept_node_id: str
    label: str
    mean_similarity: float
    best_similarity: float
    matching_observations: int
    total_observations: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Visual Evidence — Immutable Perceptual Measurement
# ═══════════════════════════════════════════════════════════════════════════


class VisualEvidence(BaseModel):
    """What was observed — raw, immutable perceptual measurement.

    This is PURE EVIDENCE.  No mutable interpretation.
    Remains valid even if interpretation changes later.

    For example:
        Today: embedding → "unknown"
        Tomorrow (after learning): same embedding → "screwdriver"

    The original evidence didn't change.

    Attributes:
        embedding: The visual embedding produced by VisionProvider.
        provenance: Origin metadata (who observed, when, how).
        image_hash: SHA-256 of the source image for dedup.

    """

    embedding: VisualEmbedding
    provenance: Provenance = Field(default_factory=Provenance)
    image_hash: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Visual Assessment — Current Interpretation (Mutable)
# ═══════════════════════════════════════════════════════════════════════════


class VisualAssessment(BaseModel):
    """What HBLLM currently thinks about the evidence.

    Separated from ``VisualEvidence`` because interpretation can change.
    The raw evidence is embedded and immutable; the assessment is the
    current cognitive interpretation.

    Attributes:
        evidence: The immutable perceptual measurement.
        candidate_observations: Similar observations found in memory.
        candidate_concepts: Concepts derived from observation grouping.
        ranking: Candidate ranking with ambiguity/margin signal.
        epistemic_profile: Multi-dimensional epistemic confidence.
        proposed_label: Set during learn() — the user's label.
        proposed_context: Set during learn() — contextual information.

    """

    evidence: VisualEvidence
    candidate_observations: list[ObservationMatch] = Field(default_factory=list)
    candidate_concepts: list[ConceptCandidate] = Field(default_factory=list)
    ranking: CandidateRanking = Field(default_factory=CandidateRanking)
    epistemic_profile: EpistemicEvidenceProfile = Field(default_factory=EpistemicEvidenceProfile)
    proposed_label: str | None = None
    proposed_context: str | None = None
