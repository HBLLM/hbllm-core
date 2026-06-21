"""Tests for VideoStreamNode and GestureNode — visual perception."""

import numpy as np
import pytest

from hbllm.perception.gesture_node import (
    GESTURE_ACTION_MAP,
    GestureEvent,
    GestureNode,
    GestureType,
)
from hbllm.perception.video_stream_node import VideoStreamNode


class TestVideoStreamNode:
    """Tests for camera/screen capture node."""

    def test_initialization(self):
        node = VideoStreamNode("test_vision", source="camera")
        assert node.source == "camera"
        assert node.capture_fps == 0.5

    def test_stats_initial(self):
        node = VideoStreamNode("test_vision")
        s = node.stats()
        assert s["frames_captured"] == 0
        assert s["running"] is False

    def test_change_detection_first_frame(self):
        """First frame always counts as a significant change."""
        node = VideoStreamNode("test_vision")
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert node._has_significant_change(frame)

    def test_change_detection_identical_frames(self):
        """Identical frames are NOT a significant change."""
        node = VideoStreamNode("test_vision")
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        node._last_frame = frame.copy()
        assert not node._has_significant_change(frame)

    def test_change_detection_different_frames(self):
        """Very different frames ARE a significant change."""
        node = VideoStreamNode("test_vision", change_threshold=0.01)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        node._last_frame = frame1
        assert node._has_significant_change(frame2)

    def test_change_detection_shape_mismatch(self):
        """Differently shaped frames count as a change."""
        node = VideoStreamNode("test_vision")
        node._last_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        new_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        assert node._has_significant_change(new_frame)

    def test_capabilities(self):
        node = VideoStreamNode("test_vision")
        assert "vision" in node.capabilities
        assert "scene_description" in node.capabilities

    @pytest.mark.asyncio
    async def test_scene_description_fallback(self):
        """Without vision provider, returns frame stats."""
        node = VideoStreamNode("test_vision")
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        desc = await node._describe_scene(frame)
        assert "100" in desc


class TestGestureNode:
    """Tests for gesture recognition node."""

    def test_initialization(self):
        node = GestureNode("test_gesture")
        assert node.min_confidence == 0.7

    def test_gesture_action_map(self):
        """Gesture→action mappings exist."""
        assert GESTURE_ACTION_MAP[GestureType.WAVE] == "attention"
        assert GESTURE_ACTION_MAP[GestureType.THUMBS_UP] == "approve"
        assert GESTURE_ACTION_MAP[GestureType.STOP] == "interrupt"

    def test_gesture_event_to_dict(self):
        e = GestureEvent(
            gesture_type=GestureType.THUMBS_UP,
            confidence=0.9,
            hand="right",
            action="approve",
        )
        d = e.to_dict()
        assert d["gesture_type"] == "thumbs_up"
        assert d["confidence"] == 0.9
        assert d["action"] == "approve"

    def test_classify_open_palm(self):
        """All fingers extended → open palm."""
        node = GestureNode()
        landmarks = [(0.5, 0.5, 0)] * 21
        landmarks[3] = (0.4, 0.5, 0)
        landmarks[4] = (0.3, 0.4, 0)
        landmarks[6] = (0.5, 0.6, 0)
        landmarks[8] = (0.5, 0.3, 0)
        landmarks[10] = (0.5, 0.6, 0)
        landmarks[12] = (0.5, 0.3, 0)
        landmarks[14] = (0.5, 0.6, 0)
        landmarks[16] = (0.5, 0.3, 0)
        landmarks[18] = (0.5, 0.6, 0)
        landmarks[20] = (0.5, 0.3, 0)

        gesture_type, confidence = node._classify_hand_pose(landmarks)
        assert gesture_type == GestureType.OPEN_PALM
        assert confidence > 0.5

    def test_classify_fist(self):
        """All fingers curled → fist."""
        node = GestureNode()
        landmarks = [(0.5, 0.5, 0)] * 21
        landmarks[3] = (0.5, 0.5, 0)
        landmarks[4] = (0.5, 0.5, 0)
        landmarks[6] = (0.5, 0.4, 0)
        landmarks[8] = (0.5, 0.6, 0)
        landmarks[10] = (0.5, 0.4, 0)
        landmarks[12] = (0.5, 0.6, 0)
        landmarks[14] = (0.5, 0.4, 0)
        landmarks[16] = (0.5, 0.6, 0)
        landmarks[18] = (0.5, 0.4, 0)
        landmarks[20] = (0.5, 0.6, 0)

        gesture_type, confidence = node._classify_hand_pose(landmarks)
        assert gesture_type == GestureType.FIST

    def test_classify_insufficient_landmarks(self):
        """Too few landmarks → unknown gesture."""
        node = GestureNode()
        landmarks = [(0.5, 0.5, 0)] * 5
        gesture_type, confidence = node._classify_hand_pose(landmarks)
        assert gesture_type == GestureType.UNKNOWN

    def test_heuristic_detector_returns_empty(self):
        """Heuristic detector (no ML) returns empty list."""
        node = GestureNode()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        gestures = node._detect_heuristic(frame)
        assert gestures == []

    def test_stats(self):
        node = GestureNode()
        s = node.stats()
        assert s["frames_analyzed"] == 0
        assert s["gestures_detected"] == 0
