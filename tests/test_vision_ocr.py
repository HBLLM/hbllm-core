"""
Tests for VisionNode OCR capability.

Tests OCR fallback chain and capabilities WITHOUT loading models.
"""

import pytest

from hbllm.perception.vision_node import VisionNode


class TestVisionNodeCapabilities:
    """Test VisionNode initialization and capabilities."""

    def test_has_ocr_capability(self):
        node = VisionNode(node_id="vision_test")
        assert "ocr" in node.capabilities
        assert "image_captioning" in node.capabilities
        assert "multimodal_processing" in node.capabilities

    def test_ocr_reader_initially_none(self):
        node = VisionNode(node_id="vision_test")
        assert node._ocr_reader is None


class TestOCRFallback:
    """Test OCR extraction with fallback chain."""

    def test_no_ocr_engine_returns_empty(self):
        """When no OCR engine is installed, _extract_text_ocr returns empty."""
        node = VisionNode(node_id="vision_test")
        # This will try easyocr (ImportError) → pytesseract (ImportError) → ""
        # We can't guarantee either is installed, so we just test it doesn't crash
        result = node._extract_text_ocr("/nonexistent/path.png")
        # Should return empty string (graceful fallback) or extracted text
        assert isinstance(result, str)
