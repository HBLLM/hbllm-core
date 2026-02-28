"""
Audio Input Node (Speech-to-Text).

Listens for `sensory.audio.in` payloads (file paths or raw bytes),
transcribes the audio using a native Whisper transformer, and immediately
dispatches a router query so the system can respond to spoken word.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

class AudioInputNode(Node):
    """
    Service node that acts as the model's "ears".
    """

    def __init__(self, node_id: str, model_size: str = "tiny.en"):
        super().__init__(node_id=node_id, node_type=NodeType.PERCEPTION, capabilities=["speech_to_text"])
        self.model_size = model_size
        self.model = None

    def _load_model(self):
        if self.model is None:
            import whisper
            logger.info("Loading Whisper %s model for AudioInput...", self.model_size)
            self.model = whisper.load_model(self.model_size)

    async def on_start(self) -> None:
        """Subscribe to sensory audio streams and workspace evaluation."""
        logger.info("Starting AudioInputNode")
        await self.bus.subscribe("sensory.audio.in", self.handle_transcribe)
        # Multi-modal workspace: participate as a competing thought source
        await self.bus.subscribe("module.evaluate", self.handle_workspace_query)

    async def on_stop(self) -> None:
        logger.info("Stopping AudioInputNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_transcribe(self, message: Message) -> Message | None:
        """
        Handles `sensory.audio.in` messages.
        Payload expects:
            file_path: Optional[str] -> path to wav/mp3 file
        """
        payload = message.payload
        file_path = payload.get("file_path")
        
        if not file_path:
            return message.create_error("Missing 'file_path'")
            
        try:
            import asyncio
            
            # Offload heavy whisper STT inference to thread
            def _transcribe():
                self._load_model()
                logger.info("Transcribing audio file: %s", file_path)
                result = self.model.transcribe(file_path, fp16=False)
                return result["text"].strip()
                
            transcription = await asyncio.to_thread(_transcribe)
            logger.info("Transcribed text: '%s'", transcription)
            
            # Formulate the response so the Caller knows we succeeded
            resp = message.create_response({"text": transcription})
            
            # Automatically forward what the user said to the brain
            query_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="router.query",
                payload={"text": transcription},
                correlation_id=message.id  # Maintain chain
            )
            # Fire and forget the internal brain query
            import asyncio
            asyncio.create_task(self.bus.publish("router.query", query_msg))
            
            return resp
            
        except Exception as e:
            logger.error("Audio transcription failed: %s", e)
            return message.create_error(str(e))

    async def handle_workspace_query(self, message: Message) -> Message | None:
        """
        Multi-modal workspace participation: if the query references audio,
        transcribe it and post a speech_perception thought to the blackboard.
        """
        payload = message.payload
        audio_path = payload.get("audio_path") or payload.get("file_path")
        
        if not audio_path:
            return None  # Not an audio-relevant query
        
        try:
            import asyncio
            
            def _transcribe():
                self._load_model()
                result = self.model.transcribe(audio_path, fp16=False)
                return result["text"].strip()
            
            transcription = await asyncio.to_thread(_transcribe)
            
            # Post as a competing thought on the Workspace blackboard
            thought_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": "speech_perception",
                    "confidence": 0.80,
                    "content": f"[Audio Transcription] {transcription}",
                    "modality": "audio",
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", thought_msg)
        except Exception as e:
            logger.warning("AudioInputNode workspace thought failed: %s", e)
        
        return None
