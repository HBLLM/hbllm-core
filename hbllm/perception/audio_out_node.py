"""
Audio Output Node (Text-to-Speech).

Listens for `sensory.audio.out` payloads (the generated text response from Domain),
uses SpeechT5 to synthesize a waveform, and saves it to disk (or plays it).
"""

from __future__ import annotations

import logging
from typing import Any
import os
from pathlib import Path

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

class AudioOutputNode(Node):
    """
    Service node that acts as the model's "voice".
    """

    def __init__(self, node_id: str, output_dir: str = "workspace/audio"):
        super().__init__(node_id=node_id, node_type=NodeType.PERCEPTION, capabilities=["text_to_speech"])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processor = None
        self.model = None
        self.vocoder = None
        self.speaker_embeddings = None

    def _load_model(self):
        if self.model is None:
            logger.info("Loading SpeechT5 TTS models for AudioOutput...")
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            from datasets import load_dataset
            import torch
            
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Load a standard speaker embedding
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    async def on_start(self) -> None:
        """Subscribe to sensory audio streams."""
        logger.info("Starting AudioOutputNode")
        await self.bus.subscribe("sensory.audio.out", self.handle_synthesize)

    async def on_stop(self) -> None:
        logger.info("Stopping AudioOutputNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_synthesize(self, message: Message) -> Message | None:
        """
        Handles `sensory.audio.out` messages.
        Payload expects:
            text: str -> The response text to vocalize
        """
        payload = message.payload
        text = payload.get("text")
        
        if not text:
            return message.create_error("Missing 'text' in payload")
            
        # Clean the text (SpeechT5 has a limited vocabulary)
        # For a true production pipeline, we'd chunk long strings.
        import re
        clean_text = re.sub(r'[^A-Za-z .,?!-]', '', text)
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
            
        try:
            import asyncio
            import soundfile as sf
            
            def _synthesize():
                import torch
                self._load_model()
                
                inputs = self.processor(text=clean_text, return_tensors="pt")
                
                # Inference
                with torch.no_grad():
                    speech = self.model.generate_speech(
                        inputs["input_ids"], 
                        self.speaker_embeddings, 
                        vocoder=self.vocoder
                    )
                
                filename = f"response_{message.id[:8]}.wav"
                out_path = self.output_dir / filename
                
                sf.write(str(out_path), speech.numpy(), samplerate=16000)
                logger.info("Saved TTS Audio to: %s", out_path)
                
                # Attempt to play it natively on MacOS
                import os
                if os.uname().sysname == 'Darwin':
                    os.system(f"afplay {out_path} &")
                    
                return str(out_path)
                
            out_file = await asyncio.to_thread(_synthesize)
            
            return message.create_response({"audio_path": out_file})
            
        except Exception as e:
            logger.error("Audio Synthesis failed: %s", e)
            return message.create_error(str(e))
