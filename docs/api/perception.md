---
title: "Perception API — HBLLM Sensor & Input Processing"
description: "API reference for perception modules: audio, vision, speaker identification, event logging, reflex arc, and multi-modal fusion."
---

# Perception API

The perception layer transforms raw sensory input (audio, vision, IoT) into
structured messages for the cognitive core. All perception nodes publish to
the MessageBus and operate asynchronously.

## Module Overview

| Module | Class | Purpose |
|--------|-------|---------|
| `audio_in_node.py` | `AudioInputNode` | Speech-to-Text with VAD |
| `audio_out_node.py` | `AudioOutputNode` | Text-to-Speech synthesis |
| `speaker_id_node.py` | `SpeakerIDNode` | Voice print identification |
| `vision_node.py` | `VisionNode` | Image captioning, OCR |
| `perception_fuser.py` | `PerceptionFuser` | Multi-modal input fusion |
| `reflex_arc.py` | `ReflexArc` | Fast-path reflexive responses |
| `reality_bus.py` | `RealityBus` | World-state event aggregation |
| `event_log.py` | `PerceptionEventLog` | Auditable event trail |
| `normalizer.py` | `InputNormalizer` | Text normalization pipeline |
| `vector_projector.py` | `VectorProjector` | Embedding projection |
| `voice_config.py` | `VoiceConfig` | Audio pipeline configuration |
| `voice_profile_store.py` | `VoiceProfileStore` | Speaker profile persistence |

## AudioInputNode

```python
from hbllm.perception.audio_in_node import AudioInputNode

node = AudioInputNode(
    stt_engine="whisper",
    model_size="base",
    vad_threshold=0.5,
)
await node.start()

# Process an audio file
result = await node.transcribe_file("audio.wav")
# result.text = "Hello, how are you?"
# result.language = "en"
# result.confidence = 0.95
```

### Key Methods

| Method | Description |
|--------|-------------|
| `start()` | Initialize STT engine and start listening |
| `stop()` | Shutdown and release audio resources |
| `transcribe_file(path)` | Transcribe an audio file |
| `process_chunk(audio_bytes)` | Process a raw audio chunk (streaming) |

## SpeakerIDNode

```python
from hbllm.perception.speaker_id_node import SpeakerIDNode

node = SpeakerIDNode()

# Enroll a new speaker
await node.enroll("alice", audio_sample)

# Identify a speaker from audio
speaker = await node.identify(audio_chunk)
# speaker.id = "alice"
# speaker.confidence = 0.92
```

## VisionNode

```python
from hbllm.perception.vision_node import VisionNode

node = VisionNode()

# Caption an image
result = await node.process_image("screenshot.png")
# result.caption = "A terminal window showing Python code"
# result.ocr_text = "def hello_world():..."
```

## PerceptionFuser

Combines multi-modal inputs into a unified perception event:

```python
from hbllm.perception.perception_fuser import PerceptionFuser

fuser = PerceptionFuser()

# Fuse text + image context
event = fuser.fuse(
    text="What's wrong with this code?",
    image_caption="A Python traceback showing IndexError",
    speaker_id="alice",
)
# event.combined_context includes all modalities
```

## ReflexArc

Fast-path bypass for time-critical responses that skip full
cognitive processing:

```python
from hbllm.perception.reflex_arc import ReflexArc

arc = ReflexArc()
arc.register_reflex(
    trigger="emergency stop",
    action=emergency_stop_fn,
    priority=10,
)

# Returns a reflex response if triggered, None otherwise
response = arc.check(input_text)
```

## Bus Topics

| Topic | Publisher | Payload |
|-------|----------|---------|
| `perception.text` | InputNormalizer | Normalized text input |
| `perception.vision` | VisionNode | Image captions, OCR results |
| `perception.audio` | AudioInputNode | Transcription results |
| `perception.speaker` | SpeakerIDNode | Speaker identification |
| `perception.fused` | PerceptionFuser | Multi-modal fused event |
| `perception.reflex` | ReflexArc | Triggered reflex action |
| `perception.event` | PerceptionEventLog | All perception events |
