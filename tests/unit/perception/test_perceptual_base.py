"""Tests for Perceptual Base Abstractions — §A8.1.

Verifies PerceptualObservation and PerceptualAssessment contracts,
and that audio and visual types satisfy the shared structural pattern.
"""

from __future__ import annotations

from hbllm.perception.providers.audio_evidence import (
    AcousticObservation,
    AudioAssessment,
)
from hbllm.perception.providers.perceptual_base import (
    PerceptualAssessment,
    PerceptualObservation,
)
from hbllm.perception.providers.provider_provenance import ProviderProvenance


class TestPerceptualObservation:
    """PerceptualObservation structural contracts."""

    def test_auto_ids(self) -> None:
        obs = PerceptualObservation()
        assert obs.observation_id.startswith("pobs_")
        assert obs.segment_id.startswith("pseg_")
        assert obs.timestamp > 0

    def test_unique_ids(self) -> None:
        obs1 = PerceptualObservation()
        obs2 = PerceptualObservation()
        assert obs1.observation_id != obs2.observation_id
        assert obs1.segment_id != obs2.segment_id

    def test_duration(self) -> None:
        obs = PerceptualObservation(duration=2.5)
        assert obs.duration == 2.5

    def test_provenance(self) -> None:
        obs = PerceptualObservation()
        assert obs.provenance is not None
        assert obs.provenance.created_by == ""


class TestPerceptualAssessment:
    """PerceptualAssessment structural contracts."""

    def test_modality(self) -> None:
        assessment = PerceptualAssessment(modality="audio")
        assert assessment.modality == "audio"

    def test_candidates_list(self) -> None:
        assessment = PerceptualAssessment(
            modality="vision",
            candidates=["candidate1", "candidate2"],
        )
        assert len(assessment.candidates) == 2

    def test_provider_provenance(self) -> None:
        assessment = PerceptualAssessment(
            provider_provenance=ProviderProvenance(
                provider="moonshine", model="base",
            ),
        )
        assert assessment.provider_provenance.provider == "moonshine"

    def test_proposed_label(self) -> None:
        assessment = PerceptualAssessment(proposed_label="doorbell")
        assert assessment.proposed_label == "doorbell"

    def test_default_empty(self) -> None:
        assessment = PerceptualAssessment()
        assert assessment.modality == ""
        assert assessment.candidates == []
        assert assessment.proposed_label is None


class TestAudioTypesConformToBase:
    """Audio types should have equivalent fields to PerceptualObservation."""

    def test_acoustic_observation_has_observation_id(self) -> None:
        obs = AcousticObservation()
        assert obs.observation_id.startswith("aobs_")

    def test_acoustic_observation_has_provenance(self) -> None:
        obs = AcousticObservation()
        assert obs.provenance is not None

    def test_audio_assessment_has_observation(self) -> None:
        assessment = AudioAssessment()
        assert assessment.observation is not None
        assert assessment.observation.observation_id.startswith("aobs_")

    def test_audio_assessment_has_candidates(self) -> None:
        """AudioAssessment uses 'events' as its candidate list."""
        assessment = AudioAssessment()
        assert isinstance(assessment.events, list)

    def test_audio_assessment_has_epistemic_profile(self) -> None:
        assessment = AudioAssessment()
        assert assessment.epistemic_profile is not None
