import asyncio
import logging
import os
from typing import Any

from hbllm.network.bus import MessageBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger("hbllm.plugins.local_audio")

__plugin__ = {
    "name": "local_audio",
    "version": "0.1.0",
    "description": "Local microphone and speaker integration for direct interaction.",
}


class LocalMicNode(Node):
    """Listens to the local microphone and streams chunks to the core."""
    
    def __init__(self, node_id: str = "local_mic", always_listen: bool = True, device: str | int | None = None, loc_device_id: str = "default"):
        super().__init__(
            node_id=node_id, 
            node_type=NodeType.PERCEPTION,
            capabilities=["audio_capture"]
        )
        self.always_listen = always_listen
        self.device = device
        self.loc_device_id = loc_device_id
        self._stream = None
        self._queue = asyncio.Queue()
        self._task = None

    async def on_start(self) -> None:
        import sounddevice as sd
        import numpy as np

        loop = asyncio.get_running_loop()

        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"[{self.node_id}] status: {status}")
            if self.always_listen:
                # Debug: Check if we are receiving pure silence
                rms = float(np.sqrt(np.mean(indata**2)))
                if rms < 0.0001 and len(self._queue._queue) % 20 == 0:
                    logger.debug(f"[{self.node_id}] Mic input is extremely quiet/silent (RMS: {rms:.5f}) - Check OS permissions!")

                # Assuming default float32 from sounddevice, converting to int16 PCM
                pcm = (indata * 32767).astype(np.int16).tobytes()
                loop.call_soon_threadsafe(self._queue.put_nowait, pcm)

        # 16kHz is expected by Whisper and Moonshine models
        # Set blocksize to 512 to perfectly match Silero VAD chunk sizes
        self._stream = sd.InputStream(
            samplerate=16000,
            blocksize=512,
            channels=1,
            dtype='float32',
            device=self.device,
            callback=audio_callback
        )
        self._stream.start()

        self._task = asyncio.create_task(self._process_queue())
        logger.info(f"[{self.node_id}] Local Microphone started (always_listen={self.always_listen})")

    async def on_stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._stream:
            self._stream.stop()
            self._stream.close()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def _process_queue(self):
        while True:
            pcm_bytes = await self._queue.get()
            if self.bus:
                msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    device_id=self.loc_device_id,
                    topic="sensory.audio.stream",
                    payload={
                        "chunk": pcm_bytes.hex(),
                        "sample_rate": 16000,
                        "is_final": False
                    }
                )
                await self.publish("sensory.audio.stream", msg)


class LocalSpeakerNode(Node):
    """Plays audio chunks received from the core through the local speaker."""
    
    def __init__(self, node_id: str = "local_speaker", device: str | int | None = None, loc_device_id: str = "default"):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PERCEPTION,
            capabilities=["audio_playback"]
        )
        self.device = device
        self.loc_device_id = loc_device_id
        self._queue = asyncio.Queue()
        self._task = None

    async def on_start(self) -> None:
        await self.bus.subscribe("sensory.audio.chunk", self.process_message)
        self._task = asyncio.create_task(self._play_queue())
        logger.info(f"[{self.node_id}] Local Speaker started")

    async def on_stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        # Filter out chunks meant for other physical devices in the house
        # Accept if it's explicitly for this device, a broadcast, or the system default
        if message.device_id not in (self.loc_device_id, "broadcast", "default"):
            return None

        payload = message.payload
        audio_hex = payload.get("audio")
        # TTS models typically output 24kHz (Kokoro) or 22kHz
        sample_rate = payload.get("sample_rate", 24000) 
        if audio_hex:
            try:
                audio_bytes = bytes.fromhex(audio_hex)
                await self._queue.put((audio_bytes, sample_rate))
            except ValueError as e:
                logger.error(f"[{self.node_id}] Failed to decode audio bytes: {e}")
        return None

    async def _play_queue(self):
        import sounddevice as sd
        import numpy as np

        while True:
            audio_bytes, sample_rate = await self._queue.get()
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                # play is non-blocking, so we wait for it in a separate thread to avoid blocking the event loop
                sd.play(audio_array, samplerate=sample_rate, device=self.device)
                await asyncio.to_thread(sd.wait)
            except Exception as e:
                logger.error(f"[{self.node_id}] Error playing audio: {e}")


async def register(bus: Any, registry: Any = None) -> Any:
    """Registers the local audio nodes if dependencies are met."""
    try:
        import sounddevice
        import numpy
    except ImportError:
        logger.warning("[hbllm_local_audio] sounddevice or numpy not installed. Local audio disabled.")
        return []

    always_listen = os.environ.get("HBLLM_MIC_ALWAYS_LISTEN", "true").lower() == "true"

    def parse_device(env_val: str | None) -> str | int | None:
        if not env_val:
            return None
        if env_val.isdigit():
            return int(env_val)
        return env_val

    mic_device = parse_device(os.environ.get("HBLLM_MIC_DEVICE"))
    speaker_device = parse_device(os.environ.get("HBLLM_SPEAKER_DEVICE"))
    loc_device_id = os.environ.get("HBLLM_LOGICAL_DEVICE_ID", "default")

    mic_node = LocalMicNode(always_listen=always_listen, device=mic_device, loc_device_id=loc_device_id)
    speaker_node = LocalSpeakerNode(device=speaker_device, loc_device_id=loc_device_id)

    # Note: PluginManager handles registration with the registry by calling node.get_info().
    # Here we just start the node. The plugin manager takes the returned nodes and registers them.
    await mic_node.start(bus)
    await speaker_node.start(bus)

    return [mic_node, speaker_node]
