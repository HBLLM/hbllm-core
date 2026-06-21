"""Gesture Recognition Node — hand/body gesture detection.

Detects user gestures from camera feed for non-verbal interaction:
    - Wave (attention/greeting)
    - Thumbs up/down (approval/rejection)
    - Point (directional intent)
    - Stop/palm (interrupt)
    - Custom gestures (extensible)

Backends:
    - MediaPipe Hands (primary, lightweight)
    - ONNX gesture model (for custom gestures)
    - Heuristic motion-based detection (fallback)

Publishes detected gestures to `perception.gesture.detected`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class GestureType(str, Enum):
    """Recognized gesture categories."""

    WAVE = "wave"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    POINT = "point"
    STOP = "stop"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    PINCH = "pinch"
    UNKNOWN = "unknown"


# Gestures that map to system actions
GESTURE_ACTION_MAP: dict[GestureType, str] = {
    GestureType.WAVE: "attention",  # Get the AI's attention
    GestureType.THUMBS_UP: "approve",  # Confirm pending action
    GestureType.THUMBS_DOWN: "reject",  # Reject pending action
    GestureType.STOP: "interrupt",  # Stop current operation
    GestureType.SWIPE_LEFT: "dismiss",  # Dismiss notification
    GestureType.SWIPE_RIGHT: "accept",  # Accept suggestion
}


@dataclass
class GestureEvent:
    """A detected gesture event."""

    gesture_type: GestureType
    confidence: float = 0.0
    hand: str = "unknown"  # "left", "right", "unknown"
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)
    action: str = ""  # Mapped system action
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gesture_type": self.gesture_type.value,
            "confidence": round(self.confidence, 3),
            "hand": self.hand,
            "action": self.action,
            "timestamp": self.timestamp,
        }


class GestureNode(Node):
    """Gesture recognition perception node.

    Processes camera frames to detect hand gestures for non-verbal
    interaction with the AI.
    """

    def __init__(
        self,
        node_id: str = "gesture_node",
        min_confidence: float = 0.7,
        cooldown_s: float = 2.0,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["gesture_recognition", "hand_tracking"],
        )
        self.min_confidence = min_confidence
        self.cooldown_s = cooldown_s

        self._hands_detector: Any = None
        self._use_mediapipe = False
        self._last_gesture: dict[str, float] = {}  # gesture_type → timestamp

        # Telemetry
        self._frames_analyzed = 0
        self._gestures_detected = 0

    async def on_start(self) -> None:
        """Initialize gesture detection backend."""
        logger.info("Starting GestureNode")

        # Try MediaPipe
        try:
            import mediapipe as mp  # type: ignore[import-not-found]

            self._hands_detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._use_mediapipe = True
            logger.info("GestureNode: MediaPipe Hands initialized")
        except ImportError:
            logger.info(
                "GestureNode: MediaPipe not installed, using heuristic detector. "
                "Install with: pip install mediapipe"
            )

        # Subscribe to video frames
        await self.bus.subscribe("perception.vision.frame", self._on_frame)

    async def on_stop(self) -> None:
        """Release resources."""
        if self._hands_detector and hasattr(self._hands_detector, "close"):
            self._hands_detector.close()
        logger.info(
            "GestureNode stopped (analyzed=%d, detected=%d)",
            self._frames_analyzed,
            self._gestures_detected,
        )

    async def handle_message(self, message: Message) -> Message | None:
        """Handle explicit gesture detection requests."""
        return None

    async def _on_frame(self, msg: Message) -> None:
        """Process a video frame for gesture detection."""
        frame_data = msg.payload.get("frame")
        if frame_data is None:
            return

        self._frames_analyzed += 1

        if isinstance(frame_data, list):
            frame = np.array(frame_data, dtype=np.uint8)
        elif isinstance(frame_data, np.ndarray):
            frame = frame_data
        else:
            return

        # Detect gestures
        if self._use_mediapipe:
            gestures = self._detect_mediapipe(frame)
        else:
            gestures = self._detect_heuristic(frame)

        # Publish detected gestures
        for gesture in gestures:
            if gesture.confidence >= self.min_confidence:
                # Cooldown check
                last = self._last_gesture.get(gesture.gesture_type.value, 0)
                if time.time() - last < self.cooldown_s:
                    continue

                self._last_gesture[gesture.gesture_type.value] = time.time()
                self._gestures_detected += 1

                # Map to system action
                gesture.action = GESTURE_ACTION_MAP.get(gesture.gesture_type, "")

                if self.bus:
                    await self.bus.publish(
                        "perception.gesture.detected",
                        Message(
                            type=MessageType.EVENT,
                            source_node_id=self.node_id,
                            topic="perception.gesture.detected",
                            payload=gesture.to_dict(),
                        ),
                    )

    def _detect_mediapipe(self, frame: np.ndarray) -> list[GestureEvent]:
        """Detect gestures using MediaPipe Hands."""
        gestures: list[GestureEvent] = []

        try:
            import cv2  # type: ignore[import-not-found]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands_detector.process(rgb)

            if not results.multi_hand_landmarks:
                return []

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract landmark positions
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # Determine handedness
                hand = "unknown"
                if results.multi_handedness and i < len(results.multi_handedness):
                    hand = results.multi_handedness[i].classification[0].label.lower()

                # Classify gesture from landmarks
                gesture_type, confidence = self._classify_hand_pose(landmarks)

                gestures.append(
                    GestureEvent(
                        gesture_type=gesture_type,
                        confidence=confidence,
                        hand=hand,
                        landmarks=landmarks,
                    )
                )

        except Exception as e:
            logger.debug("MediaPipe detection error: %s", e)

        return gestures

    def _classify_hand_pose(
        self,
        landmarks: list[tuple[float, float, float]],
    ) -> tuple[GestureType, float]:
        """Classify a hand pose from MediaPipe landmarks.

        Uses finger extension heuristics based on landmark positions.
        MediaPipe hand landmarks: 0=wrist, 4=thumb_tip, 8=index_tip,
        12=middle_tip, 16=ring_tip, 20=pinky_tip.
        """
        if len(landmarks) < 21:
            return GestureType.UNKNOWN, 0.0

        # Check which fingers are extended
        # Finger tip y < finger pip y means finger is extended (image coords)
        fingers_extended = [
            landmarks[8][1] < landmarks[6][1],  # Index
            landmarks[12][1] < landmarks[10][1],  # Middle
            landmarks[16][1] < landmarks[14][1],  # Ring
            landmarks[20][1] < landmarks[18][1],  # Pinky
        ]

        # Thumb (use x-axis since it moves sideways)
        thumb_extended = abs(landmarks[4][0] - landmarks[3][0]) > 0.04

        extended_count = sum(fingers_extended) + (1 if thumb_extended else 0)

        # Classification rules
        if extended_count >= 4:
            # All fingers open
            return GestureType.OPEN_PALM, 0.8

        if extended_count == 0:
            return GestureType.FIST, 0.8

        if thumb_extended and not any(fingers_extended):
            # Only thumb up
            if landmarks[4][1] < landmarks[3][1]:  # Thumb pointing up
                return GestureType.THUMBS_UP, 0.75
            else:
                return GestureType.THUMBS_DOWN, 0.75

        if fingers_extended[0] and not any(fingers_extended[1:]):
            # Only index finger extended
            return GestureType.POINT, 0.8

        if (
            fingers_extended[0]
            and fingers_extended[1]
            and not fingers_extended[2]
            and not fingers_extended[3]
        ):
            # Index + middle (peace/stop)
            return GestureType.STOP, 0.7

        return GestureType.UNKNOWN, 0.3

    def _detect_heuristic(self, frame: np.ndarray) -> list[GestureEvent]:
        """Simple motion-based gesture detection (no ML required).

        Detects large movements in the frame as potential gestures.
        Very low accuracy — meant as a placeholder.
        """
        # This is a stub — heuristic gesture detection from raw pixels
        # is not reliable. MediaPipe is strongly recommended.
        return []

    def stats(self) -> dict[str, Any]:
        """Node statistics."""
        return {
            "backend": "mediapipe" if self._use_mediapipe else "heuristic",
            "frames_analyzed": self._frames_analyzed,
            "gestures_detected": self._gestures_detected,
            "min_confidence": self.min_confidence,
        }
