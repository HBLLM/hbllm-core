import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Dynamically mock riva/riva.client module if not installed to prevent import errors in CI
try:
    import riva.client  # noqa: F401
except ImportError:
    import importlib.machinery

    riva_spec = importlib.machinery.ModuleSpec("riva", None)
    riva_mock = MagicMock()
    riva_mock.__spec__ = riva_spec

    riva_client_spec = importlib.machinery.ModuleSpec("riva.client", None)
    riva_client_mock = MagicMock()
    riva_client_mock.__spec__ = riva_client_spec

    # Link the submodule onto the parent package so attribute access works correctly
    riva_mock.client = riva_client_mock

    sys.modules["riva"] = riva_mock
    sys.modules["riva.client"] = riva_client_mock

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.perception.audio_in_node import AudioInputNode
from hbllm.perception.audio_out_node import AudioOutputNode


@pytest.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_audio_in_nvidia_rest_success(bus):
    """Verify that AudioInputNode makes the correct REST POST request to NVIDIA."""
    node = AudioInputNode(node_id="test_audio_in")
    await node.start(bus)

    # Set up environment variables
    with patch.dict(
        os.environ,
        {
            "NVIDIA_API_KEY": "test-key-123",
            "NVIDIA_ASR_URL": "https://test.nvidia.com/v1/audio/transcriptions",
        },
    ):
        # Mock httpx AsyncClient
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Hello world from NVIDIA ASR"}
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient.post", mock_post), patch("builtins.open", MagicMock()):
            # We create a dummy wav file path
            transcription = await node._transcribe_file("dummy.wav")

            assert transcription == "Hello world from NVIDIA ASR"
            mock_post.assert_called_once()

            # Check arguments
            url, kwargs = mock_post.call_args
            assert url[0] == "https://test.nvidia.com/v1/audio/transcriptions"
            assert kwargs["headers"]["Authorization"] in ("Bearer test-key-123", "***")
            assert kwargs["data"]["model"] == "openai/whisper-large-v3"


@pytest.mark.asyncio
async def test_audio_in_nvidia_rest_fallback(bus):
    """Verify that AudioInputNode falls back to local Moonshine on REST failure."""
    node = AudioInputNode(node_id="test_audio_in")
    await node.start(bus)

    # Set up environment variables
    with patch.dict(
        os.environ,
        {
            "NVIDIA_API_KEY": "test-key-123",
            "NVIDIA_ASR_URL": "https://test.nvidia.com/v1/audio/transcriptions",
        },
    ):
        # Mock httpx client post to raise an error (NVIDIA fails)
        mock_post = AsyncMock(side_effect=Exception("Connection Refused"))

        # Mock the Moonshine fallback transcription
        async def mock_moonshine(file_path):
            return "local transcription fallback"

        node._transcribe_file_moonshine = mock_moonshine

        with patch("httpx.AsyncClient.post", mock_post), patch("builtins.open", MagicMock()):
            transcription = await node._transcribe_file("dummy.wav")

            assert transcription == "local transcription fallback"
            mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_audio_out_nvidia_tts_cloud(bus):
    """Verify that AudioOutputNode correctly configures riva.client for cloud."""
    node = AudioOutputNode(node_id="test_audio_out")
    await node.start(bus)

    with patch.dict(
        os.environ,
        {
            "NVIDIA_API_KEY": "test-key-tts",
            "NVIDIA_TTS_URI": "grpc.nvcf.nvidia.com:443",
            "NVIDIA_TTS_FUNCTION_ID": "func-id-123",
            "NVIDIA_TTS_VOICE_NAME": "Magpie-Aria",
        },
    ):
        # Mock riva client
        mock_auth = MagicMock()
        mock_speech_client = MagicMock()

        mock_response = MagicMock()
        mock_response.audio = b"dummy_audio_bytes"
        mock_speech_client.synthesize.return_value = mock_response

        with (
            patch("riva.client.Auth", return_value=mock_auth) as mock_auth_cls,
            patch(
                "riva.client.SpeechSynthesisService", return_value=mock_speech_client
            ) as mock_client_cls,
            patch("builtins.open", MagicMock()),
        ):
            out_file = await node._synthesize_nvidia("Test text", "test.wav")

            assert out_file == "workspace/audio/test.wav"
            assert mock_auth_cls.call_count == 1
            call_kwargs = mock_auth_cls.call_args[1]
            assert call_kwargs["use_ssl"] is True
            assert call_kwargs["uri"] == "grpc.nvcf.nvidia.com:443"
            metadata = call_kwargs["metadata_args"]
            assert len(metadata) == 2
            assert metadata[0] == ["function-id", "func-id-123"]
            assert metadata[1][0] == "authorization"
            assert metadata[1][1] in ("Bearer test-key-tts", "***")

            mock_client_cls.assert_called_once_with(mock_auth)
            mock_speech_client.synthesize.assert_called_once_with(
                text="Test text", voice_name="Magpie-Aria", language_code="en-US"
            )


@pytest.mark.asyncio
async def test_audio_out_nvidia_tts_local(bus):
    """Verify that AudioOutputNode correctly configures riva.client for local without authorization/SSL."""
    node = AudioOutputNode(node_id="test_audio_out")
    await node.start(bus)

    with patch.dict(
        os.environ,
        {
            "NVIDIA_TTS_URI": "localhost:50051",
        },
    ):
        # Clear out other env vars
        if "NVIDIA_API_KEY" in os.environ:
            del os.environ["NVIDIA_API_KEY"]
        if "NVIDIA_TTS_FUNCTION_ID" in os.environ:
            del os.environ["NVIDIA_TTS_FUNCTION_ID"]

        # Mock riva client
        mock_auth = MagicMock()
        mock_speech_client = MagicMock()

        mock_response = MagicMock()
        mock_response.audio = b"dummy_audio_bytes_local"
        mock_speech_client.synthesize.return_value = mock_response

        with (
            patch("riva.client.Auth", return_value=mock_auth) as mock_auth_cls,
            patch(
                "riva.client.SpeechSynthesisService", return_value=mock_speech_client
            ) as mock_client_cls,
            patch("builtins.open", MagicMock()),
        ):
            out_file = await node._synthesize_nvidia("Local test", "test_local.wav")

            assert out_file == "workspace/audio/test_local.wav"
            mock_auth_cls.assert_called_once_with(use_ssl=False, uri="localhost:50051")
            mock_client_cls.assert_called_once_with(mock_auth)


@pytest.mark.asyncio
async def test_audio_out_nvidia_tts_fallback(bus):
    """Verify that AudioOutputNode falls back to local SpeechT5 on Riva failure."""
    node = AudioOutputNode(node_id="test_audio_out")
    await node.start(bus)

    with patch.dict(
        os.environ,
        {
            "NVIDIA_API_KEY": "test-key-tts",
            "NVIDIA_TTS_URI": "grpc.nvcf.nvidia.com:443",
            "NVIDIA_TTS_FUNCTION_ID": "func-id-123",
        },
    ):
        # Mock synthesis to raise exception
        mock_speech_client = MagicMock()
        mock_speech_client.synthesize.side_effect = Exception("gRPC Failed")

        with (
            patch("riva.client.Auth", MagicMock()),
            patch("riva.client.SpeechSynthesisService", return_value=mock_speech_client),
        ):
            out_file = await node._synthesize_nvidia("Test text", "test.wav")
            assert out_file is None  # Should return None and let the main handler fall back
