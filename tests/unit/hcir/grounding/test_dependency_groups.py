from __future__ import annotations

from hbllm.hcir.grounding.contracts import (
    EvidenceRecord,
    ModalitySource,
)


def test_dependency_group_clustering() -> None:
    """Verify that multiple models evaluating the same image share a dependency group."""
    obs_id = "obs_kitchen_frame_42"
    dep_group_camera = f"dep_{obs_id}"

    # Model A: Object Detector (YOLO)
    ev_yolo = EvidenceRecord(
        evidence_id="ev_yolo_42",
        observation_id=obs_id,
        source_channel_id="cam_kitchen",
        modality=ModalitySource.VISION,
        model_name="yolov8",
        model_version="1.0.0",
        dependency_group=dep_group_camera,
        perceptual_score=0.92,
        calibrated_likelihood=0.90,
        evidence_strength=0.95,
        provenance_hash="h1",
    )

    # Model B: Segment Anything (SAM2) on the same frame
    ev_sam = EvidenceRecord(
        evidence_id="ev_sam_42",
        observation_id=obs_id,
        source_channel_id="cam_kitchen",
        modality=ModalitySource.VISION,
        model_name="sam2",
        model_version="2.0.0",
        dependency_group=dep_group_camera,
        perceptual_score=0.89,
        calibrated_likelihood=0.88,
        evidence_strength=0.95,
        provenance_hash="h1",
    )

    # Model C: Independent Audio Whisper detector on acoustic stream
    ev_whisper = EvidenceRecord(
        evidence_id="ev_whisper_10",
        observation_id="obs_audio_chunk_10",
        source_channel_id="mic_kitchen",
        modality=ModalitySource.AUDIO,
        model_name="whisper_v3",
        model_version="3.0.0",
        dependency_group="dep_obs_audio_chunk_10",
        perceptual_score=0.85,
        calibrated_likelihood=0.87,
        evidence_strength=0.80,
        provenance_hash="h2",
    )

    all_evidence = [ev_yolo, ev_sam, ev_whisper]

    # Cluster evidence records by dependency group
    clusters: dict[str, list[EvidenceRecord]] = {}
    for ev in all_evidence:
        clusters.setdefault(ev.dependency_group, []).append(ev)

    # Must find 2 distinct independent dependency groups, not 3 independent observations
    assert len(clusters) == 2
    assert len(clusters[dep_group_camera]) == 2
    assert len(clusters["dep_obs_audio_chunk_10"]) == 1

    # Verify that the two vision models share the same group
    assert ev_yolo.dependency_group == ev_sam.dependency_group
    assert ev_yolo.dependency_group != ev_whisper.dependency_group


def test_independent_score_separation_in_evidence() -> None:
    """Verify that perceptual score and calibrated likelihood are completely independent."""
    ev = EvidenceRecord(
        evidence_id="ev_calib_test",
        observation_id="obs_test",
        source_channel_id="sensor_test",
        modality=ModalitySource.VISION,
        model_name="overconfident_detector",
        model_version="0.1",
        dependency_group="grp_test",
        # Model is overconfident (raw score 0.99), but calibrated likelihood is only 0.70
        perceptual_score=0.99,
        calibrated_likelihood=0.70,
        evidence_strength=0.85,
        provenance_hash="test_hash",
    )

    assert ev.perceptual_score == 0.99
    assert ev.calibrated_likelihood == 0.70
    assert ev.evidence_strength == 0.85
    assert ev.perceptual_score != ev.calibrated_likelihood
