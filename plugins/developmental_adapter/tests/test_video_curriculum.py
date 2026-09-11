"""Unit tests for the Multimodal Video Curriculum Ingestion Pipeline (A23)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from plugins.developmental_adapter.types import BabyActionType, BabyObjectType
from plugins.developmental_adapter.video_curriculum import (
    VideoCurriculumCurator,
    VideoFeatureExtractor,
)


def test_synthetic_video_and_manifest_generation():
    """Verify generation of self-contained synthetic video manifest and frame structures."""
    manifest, video_path = VideoCurriculumCurator.create_synthetic_educational_video(
        title="Test Tabletop Video",
        duration_sec=6.0,
        fps=30.0,
    )

    assert manifest.title == "Test Tabletop Video"
    assert manifest.duration_sec == 6.0
    assert manifest.fps == 30.0
    assert len(manifest.audio_segments) == 4
    assert len(manifest.frame_segments) > 0

    # Verify frame segments contain detected visual entities
    first_frame = manifest.frame_segments[0]
    assert len(first_frame.detections) >= 2
    colors = {d.color for d in first_frame.detections}
    assert "red" in colors
    assert "blue" in colors


def test_temporal_cross_modal_alignment():
    """Verify temporal aligner links speech segments with concurrent visual detections."""
    manifest, _ = VideoCurriculumCurator.create_synthetic_educational_video(
        title="Alignment Demo",
        duration_sec=6.0,
        fps=30.0,
    )

    curator = VideoCurriculumCurator(temporal_tolerance_sec=0.8)
    aligned_pairs = curator.align_manifest(manifest)

    assert len(aligned_pairs) == 4
    utterances = [p.utterance for p in aligned_pairs]
    assert "red" in utterances
    assert "ball" in utterances
    assert "push" in utterances
    assert "inside" in utterances

    # Push utterance occurred between 3.0s and 4.2s, where active_action was PUSH
    push_pair = next(p for p in aligned_pairs if p.utterance == "push")
    assert push_pair.action_type == BabyActionType.PUSH


def test_blank_brain_learns_grounded_lexicon_from_video():
    """Verify a completely blank infant brain learns grounded words from video demonstrations."""
    from plugins.developmental_adapter.school import CognitiveSchool

    student = CognitiveSchool(seed=42).student
    assert len(student.grounding_engine.lexicon) == 0

    manifest, _ = VideoCurriculumCurator.create_synthetic_educational_video(
        title="Language Grounding Video",
        duration_sec=6.0,
        fps=30.0,
    )

    curator = VideoCurriculumCurator()
    # Process multiple repeats to establish strong cross-situational associative confidence
    for _ in range(3):
        curator.teach_kindergarten_from_manifest(student, manifest)

    # Verify blank brain successfully fast-mapped words
    assert "red" in student.grounding_engine.lexicon
    assert "ball" in student.grounding_engine.lexicon

    red_entry = student.grounding_engine.lexicon["red"]
    assert red_entry.grounded_symbol == "red"
    assert red_entry.confidence >= 0.70

    ball_entry = student.grounding_engine.lexicon["ball"]
    assert ball_entry.grounded_symbol == BabyObjectType.BALL.value
    assert ball_entry.confidence >= 0.70


def test_opencv_video_file_extraction():
    """Verify OpenCV reads keyframes and detects colored entities from an actual video file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_file = Path(tmpdir) / "test_demo.avi"
        manifest, path = VideoCurriculumCurator.create_synthetic_educational_video(
            title="File Extraction Test",
            output_path=video_file,
            duration_sec=2.0,
            fps=30.0,
        )
        assert path is not None
        assert video_file.exists()
        assert video_file.stat().st_size > 0

        # Test feature extractor
        extractor = VideoFeatureExtractor(target_width=320, target_height=240)
        keyframes = extractor.extract_keyframes(video_file, sample_interval_sec=0.5)
        assert len(keyframes) >= 3

        # Test color blob detection on first keyframe
        first_ts, first_img = keyframes[0]
        dets = extractor.detect_color_blobs(first_img, first_ts)
        assert len(dets) >= 1
        colors = {d.color for d in dets}
        assert "red" in colors or "blue" in colors
