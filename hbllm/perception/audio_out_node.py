"""
Audio Output Node (Text-to-Speech).

Listens for `sensory.audio.out` payloads (the generated text response from Domain),
uses SpeechT5 to synthesize a waveform, and saves it to disk (or plays it).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hbllm.network.messages import Message
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

class AudioOutputNode(Node):
    """
    Service node that acts as the model's "voice".
    Supports per-tenant voice customization via custom speaker embeddings.
    """

    def __init__(self, node_id: str, output_dir: str = "workspace/audio"):
        super().__init__(node_id=node_id, node_type=NodeType.PERCEPTION, capabilities=["text_to_speech", "voice_customization"])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processor = None
        self.model = None
        self.vocoder = None
        self.default_speaker_embeddings = None
        # Cache of tenant-specific speaker embeddings
        self._tenant_voices: dict[str, Any] = {}

    def _load_model(self):
        if self.model is None:
            logger.info("Loading SpeechT5 TTS models for AudioOutput...")
            import torch
            from datasets import load_dataset
            from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Load default speaker embedding
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.default_speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def _get_speaker_embedding(self, tenant_id: str, payload: dict) -> Any:
        """Get speaker embedding: custom per-tenant or default."""
        import torch

        # 1. Check if payload includes a custom voice embedding
        custom_embedding = payload.get("voice_embedding")
        if custom_embedding and isinstance(custom_embedding, list):
            return torch.tensor(custom_embedding).unsqueeze(0)

        # 2. Check tenant cache
        if tenant_id in self._tenant_voices:
            return self._tenant_voices[tenant_id]

        # 3. Fall back to default
        return self.default_speaker_embeddings

    async def on_start(self) -> None:
        """Subscribe to sensory audio streams and voice config."""
        logger.info("Starting AudioOutputNode (per-tenant voice enabled)")
        await self.bus.subscribe("sensory.audio.out", self.handle_synthesize)
        await self.bus.subscribe("voice.config", self._handle_voice_config)

    async def on_stop(self) -> None:
        logger.info("Stopping AudioOutputNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_synthesize(self, message: Message) -> Message | None:
        """
        Handles `sensory.audio.out` messages.
        Payload expects:
            text: str -> The response text to vocalize
            voice_embedding: list[float] -> Optional custom speaker xvector
        """
        payload = message.payload
        text = payload.get("text")
        tenant_id = message.tenant_id or "default"

        if not text:
            return message.create_error("Missing 'text' in payload")

        # Clean the text (SpeechT5 has limited character support)
        import re
        clean_text = re.sub(r'[^A-Za-z0-9 .,?!\'-]', '', text)

        # Chunk long text into segments for better quality
        chunks = self._chunk_text(clean_text, max_len=450)

        try:
            import asyncio

            import numpy as np
            import soundfile as sf

            def _synthesize():
                import torch
                self._load_model()

                speaker_emb = self._get_speaker_embedding(tenant_id, payload)
                all_speech = []

                for chunk in chunks:
                    inputs = self.processor(text=chunk, return_tensors="pt")
                    with torch.no_grad():
                        speech = self.model.generate_speech(
                            inputs["input_ids"],
                            speaker_emb,
                            vocoder=self.vocoder
                        )
                    all_speech.append(speech.numpy())

                # Concatenate all chunks
                combined = np.concatenate(all_speech) if len(all_speech) > 1 else all_speech[0]

                filename = f"response_{message.id[:8]}.wav"
                out_path = self.output_dir / filename

                sf.write(str(out_path), combined, samplerate=16000)
                logger.info("Saved TTS Audio to: %s (%d chunks)", out_path, len(chunks))

                # Attempt to play natively on macOS
                import os
                if os.uname().sysname == 'Darwin':
                    os.system(f"afplay {out_path} &")

                return str(out_path)

            out_file = await asyncio.to_thread(_synthesize)

            return message.create_response({"audio_path": out_file})

        except Exception as e:
            logger.error("Audio Synthesis failed: %s", e)
            return message.create_error(str(e))

    async def _handle_voice_config(self, message: Message) -> Message | None:
        """Configure a per-tenant voice embedding."""
        import torch
        tenant_id = message.tenant_id or message.payload.get("tenant_id", "")
        embedding = message.payload.get("voice_embedding")

        if tenant_id and embedding and isinstance(embedding, list):
            self._tenant_voices[tenant_id] = torch.tensor(embedding).unsqueeze(0)
            logger.info("Configured custom voice for tenant: %s", tenant_id)
        return None

    @staticmethod
    def _chunk_text(text: str, max_len: int = 450) -> list[str]:
        """Split text into sentence-aligned chunks for TTS quality."""
        if len(text) <= max_len:
            return [text] if text.strip() else ["..."]

        chunks = []
        current = ""
        # Split on sentence boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_len:
                current = f"{current} {sentence}".strip() if current else sentence
            else:
                if current:
                    chunks.append(current)
                current = sentence[:max_len]  # Truncate very long sentences

        if current:
            chunks.append(current)

        return chunks if chunks else [text[:max_len]]
