"""Tests for Grounded Perception Contracts — V0.

Validates the stable contracts defined before any implementation:
    - VisualEmbedding space compatibility
    - EpistemicEvidenceProfile dimension independence
    - RecognitionPolicy match/ambiguous/novel decisions
    - CandidateRanking margin/ambiguity calculation
    - PerceptionGateDecision level assignment
    - Evidence vs assessment separation
"""

from __future__ import annotations

import time

from hbllm.brain.snn.neurons import SpikeEvent
from hbllm.brain.snn.perception.gate import (
    PerceptionEventType,
    PerceptionGateDecision,
    PerceptionProcessingLevel,
)
from hbllm.perception.providers.evidence import (
    CandidateRanking,
    ConceptCandidate,
    EpistemicEvidenceProfile,
    ObservationMatch,
    VisualAssessment,
    VisualEvidence,
)
from hbllm.perception.providers.policy import RecognitionPolicy
from hbllm.perception.providers.types import EmbeddingRef, VisualEmbedding, VisualRegion

# ═══════════════════════════════════════════════════════════════════════════
# VisualEmbedding
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualEmbedding:
    def _make_embedding(
        self, space_id: str = "siglip-image", model_id: str = "siglip"
    ) -> VisualEmbedding:
        return VisualEmbedding(
            vector=[0.1, 0.2, 0.3],
            model_id=model_id,
            space_id=space_id,
            embedding_type="semantic",
            dimensions=3,
            normalization="l2",
            source="image",
            image_hash="abc123",
        )

    def test_compatible_same_space(self) -> None:
        a = self._make_embedding(space_id="siglip-image")
        b = self._make_embedding(space_id="siglip-image")
        assert a.is_compatible_with(b)

    def test_incompatible_different_space(self) -> None:
        a = self._make_embedding(space_id="siglip-image")
        b = self._make_embedding(space_id="dino-image")
        assert not a.is_compatible_with(b)

    def test_incompatible_image_vs_text(self) -> None:
        a = self._make_embedding(space_id="siglip-image")
        b = VisualEmbedding(
            vector=[0.1, 0.2, 0.3],
            model_id="siglip",
            space_id="siglip-text",
            embedding_type="semantic",
            dimensions=3,
            source="text",
        )
        assert not a.is_compatible_with(b)

    def test_timestamp_auto_set(self) -> None:
        before = time.time()
        emb = self._make_embedding()
        after = time.time()
        assert before <= emb.timestamp <= after

    def test_fields_preserved(self) -> None:
        emb = self._make_embedding()
        assert emb.vector == [0.1, 0.2, 0.3]
        assert emb.normalization == "l2"
        assert emb.image_hash == "abc123"


# ═══════════════════════════════════════════════════════════════════════════
# EmbeddingRef
# ═══════════════════════════════════════════════════════════════════════════


class TestEmbeddingRef:
    def test_lightweight_reference(self) -> None:
        ref = EmbeddingRef(
            ref_id="vobs_abc123",
            space_id="siglip-image",
            model_id="siglip",
            image_hash="hash123",
        )
        assert ref.ref_id == "vobs_abc123"
        assert ref.space_id == "siglip-image"


# ═══════════════════════════════════════════════════════════════════════════
# VisualRegion
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualRegion:
    def test_normalized_bbox(self) -> None:
        region = VisualRegion(
            bbox=(0.1, 0.2, 0.5, 0.8),
            label="cup",
            confidence=0.95,
        )
        assert region.bbox == (0.1, 0.2, 0.5, 0.8)
        assert region.label == "cup"
        assert region.embedding_ref is None


# ═══════════════════════════════════════════════════════════════════════════
# EpistemicEvidenceProfile
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicEvidenceProfile:
    def test_dimensions_independent(self) -> None:
        """Each dimension should be independently settable."""
        profile = EpistemicEvidenceProfile(
            label_provenance=1.0,
            perceptual_similarity=0.0,
            evidence_strength=0.0,
            source_reliability=1.0,
        )
        assert profile.label_provenance == 1.0
        assert profile.perceptual_similarity == 0.0
        assert profile.evidence_strength == 0.0
        assert profile.source_reliability == 1.0

    def test_combined_weighted(self) -> None:
        profile = EpistemicEvidenceProfile(
            label_provenance=1.0,
            perceptual_similarity=0.8,
            evidence_strength=0.5,
            source_reliability=1.0,
        )
        # 1.0*0.3 + 0.8*0.3 + 0.5*0.25 + 1.0*0.15 = 0.3 + 0.24 + 0.125 + 0.15 = 0.815
        assert abs(profile.combined - 0.815) < 1e-6

    def test_combined_zero_dimensions(self) -> None:
        profile = EpistemicEvidenceProfile()
        # 0*0.3 + 0*0.3 + 0*0.25 + 1.0*0.15 = 0.15
        assert abs(profile.combined - 0.15) < 1e-6

    def test_user_label_vs_visual_confidence(self) -> None:
        """User label gives high label_provenance but visual identity may be unknown."""
        profile = EpistemicEvidenceProfile(
            label_provenance=1.0,  # User explicitly labeled
            perceptual_similarity=0.0,  # First time seeing this object
            evidence_strength=0.1,  # Single observation
            source_reliability=1.0,
        )
        # The combined score should NOT be near 1.0 — the system
        # has high label confidence but low perceptual evidence.
        assert profile.combined < 0.55
        assert profile.label_provenance == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# CandidateRanking
# ═══════════════════════════════════════════════════════════════════════════


class TestCandidateRanking:
    def test_from_scores_clear_winner(self) -> None:
        ranking = CandidateRanking.from_scores([0.83, 0.55])
        assert ranking.best_score == 0.83
        assert ranking.second_score == 0.55
        assert abs(ranking.margin - 0.28) < 1e-6
        assert abs(ranking.ambiguity - 0.72) < 1e-6

    def test_from_scores_ambiguous(self) -> None:
        ranking = CandidateRanking.from_scores([0.83, 0.81])
        assert abs(ranking.margin - 0.02) < 1e-6
        # High ambiguity: margin close to zero
        assert ranking.ambiguity > 0.95

    def test_from_scores_single_candidate(self) -> None:
        ranking = CandidateRanking.from_scores([0.9])
        assert ranking.best_score == 0.9
        assert ranking.second_score == 0.0
        assert ranking.margin == 0.9

    def test_from_scores_empty(self) -> None:
        ranking = CandidateRanking.from_scores([])
        assert ranking.best_score == 0.0
        assert ranking.ambiguity == 1.0

    def test_from_scores_sorted_descending(self) -> None:
        ranking = CandidateRanking.from_scores([0.9, 0.7, 0.5])
        assert ranking.best_score == 0.9
        assert ranking.second_score == 0.7


# ═══════════════════════════════════════════════════════════════════════════
# RecognitionPolicy
# ═══════════════════════════════════════════════════════════════════════════


class TestRecognitionPolicy:
    def test_clear_match(self) -> None:
        policy = RecognitionPolicy(minimum_similarity=0.7, ambiguity_margin=0.1)
        ranking = CandidateRanking(best_score=0.85, second_score=0.55, margin=0.3)
        assert policy.is_match(ranking)
        assert not policy.is_ambiguous(ranking)
        assert not policy.is_novel(ranking)

    def test_ambiguous(self) -> None:
        policy = RecognitionPolicy(minimum_similarity=0.7, ambiguity_margin=0.1)
        ranking = CandidateRanking(best_score=0.83, second_score=0.81, margin=0.02)
        assert not policy.is_match(ranking)
        assert policy.is_ambiguous(ranking)
        assert not policy.is_novel(ranking)

    def test_novel(self) -> None:
        policy = RecognitionPolicy(novelty_threshold=0.5)
        ranking = CandidateRanking(best_score=0.3, margin=0.3)
        assert not policy.is_match(ranking)
        assert policy.is_novel(ranking)

    def test_below_similarity_not_match(self) -> None:
        policy = RecognitionPolicy(minimum_similarity=0.7)
        ranking = CandidateRanking(best_score=0.6, margin=0.5)
        assert not policy.is_match(ranking)

    def test_custom_thresholds(self) -> None:
        """Policy can be customized per model/domain."""
        strict = RecognitionPolicy(
            minimum_similarity=0.9,
            ambiguity_margin=0.2,
            novelty_threshold=0.7,
        )
        ranking = CandidateRanking(best_score=0.85, margin=0.15)
        assert not strict.is_match(ranking)  # Below 0.9
        assert not strict.is_ambiguous(ranking)  # Below 0.9


# ═══════════════════════════════════════════════════════════════════════════
# Visual Evidence vs Assessment Separation
# ═══════════════════════════════════════════════════════════════════════════


class TestEvidenceAssessmentSeparation:
    def _make_evidence(self) -> VisualEvidence:
        return VisualEvidence(
            embedding=VisualEmbedding(
                vector=[0.1, 0.2, 0.3],
                model_id="siglip",
                space_id="siglip-image",
                embedding_type="semantic",
                dimensions=3,
            ),
            image_hash="hash123",
        )

    def test_evidence_is_pure(self) -> None:
        """Evidence should contain only the raw measurement."""
        evidence = self._make_evidence()
        assert evidence.embedding.vector == [0.1, 0.2, 0.3]
        assert evidence.image_hash == "hash123"
        # Evidence should NOT have candidate_observations, candidate_concepts, etc.
        assert not hasattr(evidence, "candidate_observations")
        assert not hasattr(evidence, "candidate_concepts")

    def test_assessment_wraps_evidence(self) -> None:
        """Assessment contains evidence plus interpretation."""
        evidence = self._make_evidence()
        assessment = VisualAssessment(
            evidence=evidence,
            candidate_observations=[
                ObservationMatch(
                    observation_ref="vobs_1",
                    similarity=0.85,
                    concept_node_id="vcpt_1",
                    label="screwdriver",
                ),
            ],
            candidate_concepts=[
                ConceptCandidate(
                    concept_node_id="vcpt_1",
                    label="screwdriver",
                    mean_similarity=0.85,
                    best_similarity=0.85,
                    matching_observations=1,
                ),
            ],
            ranking=CandidateRanking.from_scores([0.85]),
            epistemic_profile=EpistemicEvidenceProfile(
                perceptual_similarity=0.85,
            ),
        )
        # Evidence is embedded inside assessment
        assert assessment.evidence.embedding.vector == [0.1, 0.2, 0.3]
        # But assessment carries interpretation
        assert len(assessment.candidate_observations) == 1
        assert assessment.candidate_concepts[0].label == "screwdriver"

    def test_assessment_interpretation_independent_of_evidence(self) -> None:
        """Changing assessment doesn't affect evidence."""
        evidence = self._make_evidence()
        assessment = VisualAssessment(evidence=evidence)
        assessment.proposed_label = "toolbox"
        # Evidence is unaffected
        assert evidence.image_hash == "hash123"
        # Assessment carries the label
        assert assessment.proposed_label == "toolbox"


# ═══════════════════════════════════════════════════════════════════════════
# Perception Gate Decision
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceptionGateDecision:
    def _spike(self, strength: float) -> SpikeEvent:
        return SpikeEvent(fired=True, strength=strength, timestamp=time.time())

    def test_no_spikes(self) -> None:
        decision = PerceptionGateDecision.from_spikes([], frame_index=0)
        assert not decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.NONE

    def test_scene_spike_high_level(self) -> None:
        fired = [("scene", self._spike(0.7))]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=10)
        assert decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.HIGH
        assert decision.event_type == PerceptionEventType.SCENE_CHANGE
        assert decision.frame_index == 10

    def test_motion_only_low_level(self) -> None:
        fired = [("motion", self._spike(0.6))]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=5)
        assert decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.LOW
        assert decision.event_type == PerceptionEventType.MOTION_EVENT

    def test_novelty_without_scene_standard_level(self) -> None:
        """Novelty spike alone should still trigger processing."""
        fired = [("novelty", self._spike(0.4))]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=20)
        assert decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.STANDARD
        assert decision.event_type == PerceptionEventType.NOVEL_APPEARANCE

    def test_novelty_high_urgency_triggers_urgent(self) -> None:
        """Novelty at urgency > 0.8 triggers URGENT per gate decision logic."""
        fired = [("novelty", self._spike(0.9))]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=30)
        assert decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.URGENT
        assert decision.novelty == 0.9

    def test_multi_channel_urgent(self) -> None:
        fired = [
            ("scene", self._spike(0.7)),
            ("novelty", self._spike(0.8)),
            ("entity", self._spike(0.6)),
        ]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=40)
        assert decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.URGENT
        # scene has priority for event type
        assert decision.event_type == PerceptionEventType.SCENE_CHANGE

    def test_entity_spike_standard_level(self) -> None:
        fired = [("entity", self._spike(0.65))]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=15)
        assert decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.STANDARD
        assert decision.event_type == PerceptionEventType.ENTITY_CHANGE

    def test_stability_spike(self) -> None:
        fired = [("stability", self._spike(0.5))]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=50)
        assert decision.should_process
        assert decision.event_type == PerceptionEventType.STABILITY_SHIFT

    def test_heartbeat_convenience(self) -> None:
        decision = PerceptionGateDecision.heartbeat(frame_index=300)
        assert decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.LOW
        assert decision.event_type == PerceptionEventType.HEARTBEAT
        assert decision.urgency == 0.1

    def test_no_action_convenience(self) -> None:
        decision = PerceptionGateDecision.no_action(frame_index=99)
        assert not decision.should_process
        assert decision.processing_level == PerceptionProcessingLevel.NONE
        assert decision.frame_index == 99

    def test_urgency_is_max_strength(self) -> None:
        fired = [
            ("motion", self._spike(0.3)),
            ("entity", self._spike(0.75)),
        ]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=0)
        assert decision.urgency == 0.75

    def test_temporal_significance_is_mean(self) -> None:
        fired = [
            ("motion", self._spike(0.4)),
            ("scene", self._spike(0.8)),
        ]
        decision = PerceptionGateDecision.from_spikes(fired, frame_index=0)
        assert abs(decision.temporal_significance - 0.6) < 1e-6
