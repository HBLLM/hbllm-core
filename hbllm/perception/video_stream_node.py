"""Video Stream Node — visual perception via camera/screen capture.

Provides the AI's "eyes" by processing visual input:
    - Camera feed processing (frame capture, motion detection)
    - Screen capture for visual context
    - Scene description via vision LLM (e.g., LLaVA, GPT-4V)
    - Object/face detection hooks

Architecture:
    1. Captures frames at configurable FPS (default 0.5 FPS for ambient)
    2. Runs change detection to avoid processing static scenes
    3. Sends significant frames to vision LLM for description
    4. Publishes scene descriptions to `perception.vision.scene`

Backends:
    - OpenCV (cv2) for camera/screen capture
    - PIL/Pillow as fallback for screenshots
    - Vision LLM for scene understanding (via existing provider)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class VideoStreamNode(Node):
    """Visual perception node — the AI's eyes.

    Captures visual input, detects changes, and generates scene
    descriptions for the cognitive brain.

    Usage::

        node = VideoStreamNode("vision_node")
        await node.on_start()
        # Frame processing runs in background
    """

    def __init__(
        self,
        node_id: str = "video_stream",
        source: str = "camera",  # "camera", "screen", "file"
        camera_index: int = 0,
        capture_fps: float = 0.5,  # Ambient monitoring: 1 frame per 2 seconds
        change_threshold: float = 0.05,  # Min pixel change to process frame
        vision_provider: Any | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["vision", "scene_description", "motion_detection"],
        )
        self.source = source
        self.camera_index = camera_index
        self.capture_fps = capture_fps
        self.change_threshold = change_threshold
        self.vision_provider = vision_provider

        self._capture: Any = None  # cv2.VideoCapture
        self._last_frame: np.ndarray | None = None
        self._capture_task: asyncio.Task[None] | None = None
        self._running = False

        # Telemetry
        self._frames_captured = 0
        self._frames_processed = 0  # Frames that had significant changes
        self._scenes_described = 0

    async def on_start(self) -> None:
        """Initialize capture device and start background loop."""
        logger.info("Starting VideoStreamNode (source=%s, fps=%.1f)", self.source, self.capture_fps)

        if self.source == "camera":
            self._init_camera()
        elif self.source == "screen":
            self._init_screen_capture()

        if self._capture is not None or self.source == "screen":
            self._running = True
            self._capture_task = asyncio.create_task(self._capture_loop())
            logger.info("VideoStreamNode capture loop started")
        else:
            logger.warning("VideoStreamNode: No capture device available")

    async def on_stop(self) -> None:
        """Stop capture and release resources."""
        self._running = False
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
        logger.info(
            "VideoStreamNode stopped (captured=%d, processed=%d, described=%d)",
            self._frames_captured,
            self._frames_processed,
            self._scenes_described,
        )

    async def handle_message(self, message: Message) -> Message | None:
        """Handle explicit capture requests."""
        if message.topic == "vision.capture":
            frame = await self._capture_frame()
            if frame is not None:
                description = await self._describe_scene(frame)
                return message.create_response(
                    {
                        "description": description,
                        "frame_shape": list(frame.shape),
                        "timestamp": time.time(),
                    }
                )
        return None

    def _init_camera(self) -> None:
        """Initialize camera via OpenCV."""
        try:
            import cv2  # type: ignore[import-not-found]

            self._capture = cv2.VideoCapture(self.camera_index)
            if not self._capture.isOpened():
                logger.warning("Camera index %d not available", self.camera_index)
                self._capture = None
            else:
                logger.info("Camera %d opened successfully", self.camera_index)
        except ImportError:
            logger.warning(
                "OpenCV not installed — camera capture unavailable. "
                "Install with: pip install opencv-python-headless"
            )

    def _init_screen_capture(self) -> None:
        """Verify screen capture capability."""
        try:
            from PIL import ImageGrab  # type: ignore[import-not-found]  # noqa: F401

            logger.info("Screen capture initialized via PIL")
        except ImportError:
            logger.warning(
                "Pillow not installed — screen capture unavailable. "
                "Install with: pip install Pillow"
            )

    async def _capture_loop(self) -> None:
        """Background frame capture loop."""
        interval = 1.0 / self.capture_fps

        while self._running:
            try:
                frame = await self._capture_frame()
                if frame is not None:
                    self._frames_captured += 1

                    # Check for significant change
                    if self._has_significant_change(frame):
                        self._frames_processed += 1

                        # Describe the scene
                        description = await self._describe_scene(frame)

                        # Publish to bus
                        if self.bus and description:
                            await self.bus.publish(
                                "perception.vision.scene",
                                Message(
                                    type=MessageType.EVENT,
                                    source_node_id=self.node_id,
                                    topic="perception.vision.scene",
                                    payload={
                                        "description": description,
                                        "frame_shape": list(frame.shape),
                                        "timestamp": time.time(),
                                        "source": self.source,
                                    },
                                ),
                            )

                    self._last_frame = frame

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Frame capture error: %s", e)
                await asyncio.sleep(interval * 2)

    async def _capture_frame(self) -> np.ndarray | None:
        """Capture a single frame from the configured source."""
        if self.source == "camera" and self._capture is not None:
            return await asyncio.to_thread(self._read_camera_frame)
        elif self.source == "screen":
            return await asyncio.to_thread(self._read_screen_frame)
        return None

    def _read_camera_frame(self) -> np.ndarray | None:
        """Read a frame from camera (runs in thread)."""
        try:
            ret, frame = self._capture.read()
            if ret and frame is not None:
                return frame
        except Exception as e:
            logger.debug("Camera read failed: %s", e)
        return None

    def _read_screen_frame(self) -> np.ndarray | None:
        """Capture a screenshot (runs in thread)."""
        try:
            from PIL import ImageGrab  # type: ignore[import-not-found]

            screenshot = ImageGrab.grab()
            # Resize to reduce processing cost (max 640px wide)
            w, h = screenshot.size
            if w > 640:
                ratio = 640 / w
                screenshot = screenshot.resize((640, int(h * ratio)))
            return np.array(screenshot)
        except Exception as e:
            logger.debug("Screen capture failed: %s", e)
            return None

    def _has_significant_change(self, frame: np.ndarray) -> bool:
        """Check if the frame differs significantly from the last one."""
        if self._last_frame is None:
            return True

        if frame.shape != self._last_frame.shape:
            return True

        # Compute mean absolute difference (normalized)
        try:
            diff = np.mean(np.abs(frame.astype(float) - self._last_frame.astype(float))) / 255.0
            return diff > self.change_threshold
        except Exception:
            return True

    async def _describe_scene(self, frame: np.ndarray) -> str:
        """Generate a text description of the visual scene.

        Uses vision LLM if available, otherwise returns basic frame stats.
        """
        if self.vision_provider:
            try:
                # Encode frame as base64 for vision API
                import base64
                import io

                from PIL import Image  # type: ignore[import-not-found]

                img = Image.fromarray(frame)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=50)
                b64 = base64.b64encode(buffer.getvalue()).decode()

                response = await self.vision_provider.generate(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this scene briefly in 1-2 sentences.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=100,
                )

                self._scenes_described += 1
                content = (
                    response.get("content", "") if isinstance(response, dict) else str(response)
                )
                return content.strip()
            except Exception as e:
                logger.debug("Vision LLM description failed: %s", e)

        # Fallback: basic frame statistics
        h, w = frame.shape[:2]
        brightness = float(np.mean(frame))
        return f"Frame {w}×{h}, brightness={brightness:.0f}/255"

    def stats(self) -> dict[str, Any]:
        """Vision node statistics."""
        return {
            "source": self.source,
            "frames_captured": self._frames_captured,
            "frames_processed": self._frames_processed,
            "scenes_described": self._scenes_described,
            "capture_fps": self.capture_fps,
            "running": self._running,
        }
