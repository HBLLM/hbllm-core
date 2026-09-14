"""Multimodal Video Curriculum Ingestion Pipeline for HBLLM Developmental Learning (A23).

Enables observational pedagogical learning from video demonstrations (e.g. YouTube,
lecture clips, or camera streams). Implements temporal cross-modal alignment between
spoken narration (time-stamped utterances) and visual scene dynamics (bounding boxes,
color segmentation, optical velocities, and contact states via OpenCV).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .blank_brain import BlankBrainSubstrate
from .types import (
    BabyActionType,
    BabyObjectType,
    BabyRelationType,
    Vector2D,
)

logger = logging.getLogger(__name__)


class VideoModality(str, Enum):
    """Sensory track modality in video demonstrations."""

    VISUAL_FRAMES = "visual_frames"
    AUDIO_NARRATION = "audio_narration"
    SUBTITLES = "subtitles"
    SENSORIMOTOR_TELEMETRY = "sensorimotor_telemetry"


@dataclass
class VisualDetection:
    """A detected physical entity in a video frame."""

    entity_id: str
    object_type: BabyObjectType
    color: str
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixels
    center: Vector2D  # Normalized table/screen coordinates [-1.0, 1.0]
    velocity: Vector2D = field(default_factory=lambda: Vector2D(0.0, 0.0))
    is_moving: bool = False
    is_contacting: str | None = None  # ID of entity in contact
    confidence: float = 0.95


@dataclass
class VideoFrameSegment:
    """A single processed frame or temporal keyframe interval."""

    frame_index: int
    timestamp_sec: float
    detections: list[VisualDetection] = field(default_factory=list)
    active_action: BabyActionType | None = None
    spatial_relations: list[tuple[str, BabyRelationType, str]] = field(default_factory=list)


@dataclass
class AudioNarrationSegment:
    """A time-aligned spoken narration segment or transcript."""

    text: str
    start_sec: float
    end_sec: float
    speaker: str = "Teacher"
    confidence: float = 0.98

    @property
    def midpoint_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


@dataclass
class TemporalDemonstrationPair:
    """Cross-modally aligned speech and sensory context for ostensive grounding."""

    utterance: str
    visual_context: dict[str, Any]
    timestamp_sec: float
    duration_sec: float
    detected_entity_ids: list[str] = field(default_factory=list)
    action_type: BabyActionType | None = None


@dataclass
class VideoDemonstrationManifest:
    """Self-contained multimodal demonstration manifest for offline/online replay."""

    title: str
    video_path_or_url: str
    duration_sec: float
    fps: float
    audio_segments: list[AudioNarrationSegment] = field(default_factory=list)
    frame_segments: list[VideoFrameSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "video_path_or_url": self.video_path_or_url,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "audio_segments": [
                {
                    "text": a.text,
                    "start_sec": a.start_sec,
                    "end_sec": a.end_sec,
                    "speaker": a.speaker,
                    "confidence": a.confidence,
                }
                for a in self.audio_segments
            ],
            "frame_segments": [
                {
                    "frame_index": f.frame_index,
                    "timestamp_sec": f.timestamp_sec,
                    "active_action": f.active_action.value if f.active_action else None,
                    "detections": [
                        {
                            "entity_id": d.entity_id,
                            "object_type": d.object_type.value,
                            "color": d.color,
                            "bbox": list(d.bbox),
                            "center": [d.center.x, d.center.y],
                            "velocity": [d.velocity.x, d.velocity.y],
                            "is_moving": d.is_moving,
                            "confidence": d.confidence,
                        }
                        for d in f.detections
                    ],
                }
                for f in self.frame_segments
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoDemonstrationManifest:
        audio = [
            AudioNarrationSegment(
                text=a["text"],
                start_sec=float(a["start_sec"]),
                end_sec=float(a["end_sec"]),
                speaker=a.get("speaker", "Teacher"),
                confidence=float(a.get("confidence", 0.98)),
            )
            for a in data.get("audio_segments", [])
        ]
        frames = []
        for f in data.get("frame_segments", []):
            action_val = f.get("active_action")
            action = BabyActionType(action_val) if action_val else None
            dets = [
                VisualDetection(
                    entity_id=d["entity_id"],
                    object_type=BabyObjectType(d["object_type"]),
                    color=d["color"],
                    bbox=tuple(d["bbox"]),  # type: ignore
                    center=Vector2D(d["center"][0], d["center"][1]),
                    velocity=Vector2D(d["velocity"][0], d["velocity"][1]),
                    is_moving=bool(d.get("is_moving", False)),
                    confidence=float(d.get("confidence", 0.95)),
                )
                for d in f.get("detections", [])
            ]
            frames.append(
                VideoFrameSegment(
                    frame_index=int(f["frame_index"]),
                    timestamp_sec=float(f["timestamp_sec"]),
                    detections=dets,
                    active_action=action,
                )
            )
        return cls(
            title=data.get("title", "Untitled Demo"),
            video_path_or_url=data.get("video_path_or_url", ""),
            duration_sec=float(data.get("duration_sec", 0.0)),
            fps=float(data.get("fps", 30.0)),
            audio_segments=audio,
            frame_segments=frames,
            metadata=data.get("metadata", {}),
        )


class VideoFeatureExtractor:
    """Extracts bounding boxes, motion velocities, and contact states from video files."""

    def __init__(self, target_width: int = 640, target_height: int = 480) -> None:
        self.target_width = target_width
        self.target_height = target_height

    def extract_keyframes(
        self,
        video_path: str | Path,
        sample_interval_sec: float = 0.5,
        max_duration_sec: float = 60.0,
    ) -> list[tuple[float, np.ndarray]]:
        """Sample keyframes from a video file at a fixed temporal interval."""
        path_str = str(video_path)
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            logger.warning(f"Unable to open video file at {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps * sample_interval_sec))
        max_frames = int(fps * max_duration_sec)

        keyframes: list[tuple[float, np.ndarray]] = []
        frame_idx = 0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                resized = cv2.resize(frame, (self.target_width, self.target_height))
                keyframes.append((timestamp, resized))
            frame_idx += 1

        cap.release()
        return keyframes

    def detect_color_blobs(self, frame: np.ndarray, timestamp_sec: float) -> list[VisualDetection]:
        """Detect salient colored objects on a tabletop surface using HSV thresholds."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections: list[VisualDetection] = []

        # Color range definitions in HSV space
        color_ranges = {
            "red": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([170, 100, 100]), np.array([180, 255, 255])),
            ],
            "blue": [
                (np.array([100, 100, 100]), np.array([130, 255, 255])),
            ],
            "green": [
                (np.array([40, 100, 100]), np.array([80, 255, 255])),
            ],
            "yellow": [
                (np.array([20, 100, 100]), np.array([35, 255, 255])),
            ],
        }

        det_idx = 0
        h, w = frame.shape[:2]

        for color_name, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            # Morphological smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 400:  # Ignore tiny noise artifacts
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                # Estimate shape aspect ratio
                aspect = bw / float(bh)
                obj_type = BabyObjectType.BALL if 0.8 <= aspect <= 1.2 else BabyObjectType.BLOCK
                if area > 10000:
                    obj_type = BabyObjectType.BOX

                # Normalize to [-1.0, 1.0]
                norm_x = (x + bw / 2.0 - w / 2.0) / (w / 2.0)
                norm_y = (y + bh / 2.0 - h / 2.0) / (h / 2.0)

                det_idx += 1
                detections.append(
                    VisualDetection(
                        entity_id=f"vid_ent_{color_name}_{det_idx}",
                        object_type=obj_type,
                        color=color_name,
                        bbox=(x, y, bw, bh),
                        center=Vector2D(round(norm_x, 3), round(norm_y, 3)),
                    )
                )

        return detections


class VideoCurriculumCurator:
    """Curator that digests video demonstrations into pedagogical teaching sequences.

    Cross-modally aligns spoken commentary with visual scene representations,
    feeding clean ostensive pairs into Kindergarten Lexicon grounding and
    interventional state transitions into Elementary Causal Physics.
    """

    def __init__(self, temporal_tolerance_sec: float = 0.2) -> None:
        self.temporal_tolerance_sec = temporal_tolerance_sec
        self.feature_extractor = VideoFeatureExtractor()

    def align_manifest(
        self, manifest: VideoDemonstrationManifest
    ) -> list[TemporalDemonstrationPair]:
        """Align narration segments with corresponding visual frame states."""
        aligned_pairs: list[TemporalDemonstrationPair] = []

        for audio in manifest.audio_segments:
            # Find frame segments that fall within [audio.start - tol, audio.end + tol]
            matching_frames = [
                f
                for f in manifest.frame_segments
                if (audio.start_sec - self.temporal_tolerance_sec)
                <= f.timestamp_sec
                <= (audio.end_sec + self.temporal_tolerance_sec)
            ]

            if not matching_frames:
                continue

            # Prioritize moving objects or objects undergoing action during this speech interval
            active_detections: list[VisualDetection] = []
            detected_action: BabyActionType | None = None

            for frame in matching_frames:
                if frame.active_action:
                    detected_action = frame.active_action
                for det in frame.detections:
                    if det not in active_detections:
                        active_detections.append(det)

            # Build rich visual context dictionary compatible with observe_paired_demonstration
            visual_context: dict[str, Any] = {}

            # Primary focal object (e.g., first moving or primary detected entity)
            focal_det = next((d for d in active_detections if d.is_moving), None) or (
                active_detections[0] if active_detections else None
            )

            utt = audio.text.strip().lower()
            if focal_det:
                if utt in ("red", "blue", "green", "yellow", "color"):
                    visual_context["color"] = focal_det.color
                elif utt in ("ball", "block", "box", "tool", "stick", "toy", "object"):
                    visual_context["entity_type"] = focal_det.object_type
                else:
                    visual_context["color"] = focal_det.color
                    visual_context["entity_type"] = focal_det.object_type
                    visual_context["mass"] = (
                        1.0 if focal_det.object_type == BabyObjectType.BALL else 2.0
                    )

            if utt in ("push", "pull", "roll", "move"):
                visual_context["action"] = detected_action or BabyActionType(utt.upper())
            elif (
                detected_action
                and "entity_type" not in visual_context
                and "color" not in visual_context
            ):
                visual_context["action"] = detected_action

            if utt == "inside":
                visual_context["relation"] = BabyRelationType.INSIDE

            aligned_pairs.append(
                TemporalDemonstrationPair(
                    utterance=audio.text.strip().lower(),
                    visual_context=visual_context,
                    timestamp_sec=audio.start_sec,
                    duration_sec=audio.end_sec - audio.start_sec,
                    detected_entity_ids=[d.entity_id for d in active_detections],
                    action_type=detected_action,
                )
            )

        return aligned_pairs

    def teach_kindergarten_from_manifest(
        self,
        student: BlankBrainSubstrate,
        manifest: VideoDemonstrationManifest,
    ) -> list[TemporalDemonstrationPair]:
        """Teach infant language grounding using time-aligned video demonstrations."""
        pairs = self.align_manifest(manifest)
        logger.info(
            f"[VideoCurator] Teaching Kindergarten: processing {len(pairs)} aligned video demonstrations from '{manifest.title}'."
        )

        for pair in pairs:
            if pair.visual_context:
                student.grounding_engine.observe_paired_demonstration(
                    pair.utterance,
                    pair.visual_context,
                )

        return pairs

    def extract_causal_interventions_from_manifest(
        self,
        manifest: VideoDemonstrationManifest,
    ) -> list[tuple[dict[str, Any], BabyActionType, dict[str, Any]]]:
        """Extract before/after state transitions for interventional causal learning."""
        transitions: list[tuple[dict[str, Any], BabyActionType, dict[str, Any]]] = []

        frames = sorted(manifest.frame_segments, key=lambda f: f.timestamp_sec)
        for i in range(len(frames) - 1):
            f_before = frames[i]
            f_after = frames[i + 1]

            if f_before.active_action:
                act = f_before.active_action
                # Construct pre-state and post-state snapshots
                pre_state = {
                    d.entity_id: {"pos": d.center, "vel": d.velocity, "moving": d.is_moving}
                    for d in f_before.detections
                }
                post_state = {
                    d.entity_id: {"pos": d.center, "vel": d.velocity, "moving": d.is_moving}
                    for d in f_after.detections
                }
                transitions.append((pre_state, act, post_state))

        return transitions

    @staticmethod
    def create_synthetic_educational_video(
        title: str = "Introductory Physics & Tabletop Actions",
        output_path: str | Path | None = None,
        duration_sec: float = 6.0,
        fps: float = 30.0,
    ) -> tuple[VideoDemonstrationManifest, Path | None]:
        """Create a self-contained OpenCV-generated video file and corresponding manifest.

        Renders real moving 2D shapes (red ball rolling, blue block resting, yellow box)
        with timestamped spoken commentary tracks for zero-external-dependency validation.
        """
        width, height = 640, 480
        total_frames = int(duration_sec * fps)
        video_out_path = Path(output_path) if output_path else None

        writer = None
        if video_out_path:
            video_out_path.parent.mkdir(parents=True, exist_ok=True)
            # Use MJPG for reliable cross-platform encoding on macOS / Linux
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

        # Define script events: (start_sec, end_sec, text, action, ball_x_vel, box_pos)
        audio_script = [
            AudioNarrationSegment("red", 0.5, 1.5, speaker="Teacher"),
            AudioNarrationSegment("ball", 1.8, 2.8, speaker="Teacher"),
            AudioNarrationSegment("push", 3.0, 4.2, speaker="Teacher"),
            AudioNarrationSegment("inside", 4.5, 5.8, speaker="Teacher"),
        ]

        frame_segments: list[VideoFrameSegment] = []

        ball_x = 100.0
        ball_y = 240.0
        ball_radius = 30

        for frame_idx in range(total_frames):
            t_sec = frame_idx / fps
            img = np.full((height, width, 3), 245, dtype=np.uint8)  # White/gray table surface

            # Draw static blue block
            cv2.rectangle(img, (480, 200), (560, 280), (220, 60, 20), -1)  # BGR: Blue

            # Draw static yellow container / box
            cv2.rectangle(img, (380, 180), (460, 300), (30, 220, 240), 3)  # Yellow border

            # Active pushing event between t=3.0 and t=4.5
            is_pushing = 3.0 <= t_sec <= 4.5
            action = BabyActionType.PUSH if is_pushing else None

            if is_pushing:
                ball_x += 4.0  # Moves rightward towards yellow box
                ball_moving = True
            else:
                ball_moving = False

            # Draw red ball
            cv2.circle(img, (int(ball_x), int(ball_y)), ball_radius, (20, 20, 220), -1)  # BGR: Red

            if writer:
                writer.write(img)

            # Sample keyframes every 15 frames (~0.5 sec)
            if frame_idx % 15 == 0:
                norm_ball_x = (ball_x - width / 2.0) / (width / 2.0)
                norm_ball_y = (ball_y - height / 2.0) / (height / 2.0)

                dets = [
                    VisualDetection(
                        entity_id="synth_red_ball",
                        object_type=BabyObjectType.BALL,
                        color="red",
                        bbox=(int(ball_x - ball_radius), int(ball_y - ball_radius), 60, 60),
                        center=Vector2D(round(norm_ball_x, 3), round(norm_ball_y, 3)),
                        velocity=Vector2D(4.0 if ball_moving else 0.0, 0.0),
                        is_moving=ball_moving,
                    ),
                    VisualDetection(
                        entity_id="synth_blue_block",
                        object_type=BabyObjectType.BLOCK,
                        color="blue",
                        bbox=(480, 200, 80, 80),
                        center=Vector2D(0.625, -0.083),
                        is_moving=False,
                    ),
                    VisualDetection(
                        entity_id="synth_yellow_box",
                        object_type=BabyObjectType.BOX,
                        color="yellow",
                        bbox=(380, 180, 80, 120),
                        center=Vector2D(0.312, 0.0),
                        is_moving=False,
                    ),
                ]

                relations = []
                if ball_x > 380:
                    relations.append(
                        ("synth_red_ball", BabyRelationType.INSIDE, "synth_yellow_box")
                    )

                frame_segments.append(
                    VideoFrameSegment(
                        frame_index=frame_idx,
                        timestamp_sec=round(t_sec, 3),
                        detections=dets,
                        active_action=action,
                        spatial_relations=relations,
                    )
                )

        if writer:
            writer.release()

        manifest = VideoDemonstrationManifest(
            title=title,
            video_path_or_url=str(video_out_path) if video_out_path else "synthetic_in_memory",
            duration_sec=duration_sec,
            fps=fps,
            audio_segments=audio_script,
            frame_segments=frame_segments,
            metadata={
                "generator": "OpenCV Synthetic Tabletop Engine",
                "resolution": [width, height],
            },
        )

        return manifest, video_out_path
