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
| `conversation_turn.py` | `ConversationTurnManager` | Full-duplex voice turn management |

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
| `sensory.audio.start` | ConversationTurnManager | User started speaking |
| `sensory.audio.stop` | ConversationTurnManager | User stopped speaking |
| `sensory.audio.interrupt` | ConversationTurnManager | Barge-in detected |

---

## ConversationTurnManager

**Module:** `hbllm.perception.conversation_turn.ConversationTurnManager`

Full-duplex voice conversation state machine that manages turn-taking between
the user and the system. Handles barge-in interrupts, silence timeouts, and
continuous listening mode.

### State Machine

```
IDLE → (wake word) → LISTENING → (transcription) → PROCESSING → (response) → SPEAKING
                                                                     ↑ (barge-in) ↓
                                                                     ← INTERRUPTED ←
```

### Usage

```python
from hbllm.perception.conversation_turn import ConversationTurnManager, TurnState

mgr = ConversationTurnManager(
    silence_timeout_s=3.0,  # Max silence before end-of-turn
    processing_timeout_s=30.0,  # Max time waiting for LLM response
    continuous_listen=True,  # Auto-resume listening after speaking
)

# Query current state
snap = mgr.snapshot()
# {"state": "idle", "turn_count": 0, "continuous_listen": true}
```

### TurnState Enum

| State | Description |
|-------|-------------|
| `IDLE` | Waiting for wake word or input |
| `LISTENING` | Actively capturing user speech |
| `PROCESSING` | Transcription received, waiting for LLM |
| `SPEAKING` | Playing TTS response |
| `INTERRUPTED` | User barged in during playback |

### Bus Topics

| Topic | Direction | Payload |
|-------|-----------|---------|
| `sensory.audio.start` | Subscribe | User started speaking |
| `sensory.audio.stop` | Subscribe | End of speech detected |
| `sensory.audio.interrupt` | Subscribe | Barge-in during TTS |
| `cognitive.response.ready` | Subscribe | LLM response available |
| `tts.playback.done` | Subscribe | TTS finished playing |

---

## Visual Cognition Runtime

The visual cognition system extends perception with typed evidence production,
one-shot visual learning, and SNN-gated temporal attention.

**Core invariant:** Perception produces evidence. HCIR commits state.

### Architecture

```
Image → VisionProvider.encode() → VisualEmbedding
      → VisualPerceptionRuntime.perceive() → VisualAssessment
      → VisualPerceptionTransaction.commit_*() → HCIR nodes + edges + beliefs

Video → VisualSignalExtractor → VisualSignals (cheap, ~0.1ms)
      → PerceptionEnsemble (SNN) → PerceptionGateDecision
      → (if should_process) → full perception pipeline
```

### Vision Providers

| Module | Class | Purpose |
|--------|-------|---------|
| `providers/base.py` | `VisionProvider` | Protocol for embedding providers |
| `providers/base.py` | `VisionDetector` | Protocol for object detection |
| `providers/base.py` | `VisionOCR` | Protocol for OCR |
| `providers/siglip_provider.py` | `SigLIPVisionProvider` | SigLIP model (lazy loading, CPU/CUDA/MPS) |
| `providers/mock_provider.py` | `MockVisionProvider` | Deterministic mock for testing |

```python
from hbllm.perception.providers.siglip_provider import SigLIPVisionProvider

provider = SigLIPVisionProvider()  # Lazy loads on first encode
embedding = await provider.encode(image)
# embedding.vector: list[float], L2-normalized
# embedding.space_id: "google/siglip-base-patch16-224-image"
```

### Visual Perception Pipeline

```python
from hbllm.perception.visual_perception import VisualPerception
from hbllm.perception.visual_perception_runtime import VisualPerceptionRuntime
from hbllm.perception.visual_perception_transaction import VisualPerceptionTransaction
from hbllm.perception.visual_memory import VisualMemory
from hbllm.hcir.graph import CognitiveGraph

provider = SigLIPVisionProvider()
memory = VisualMemory()
graph = CognitiveGraph()

runtime = VisualPerceptionRuntime(provider, memory)
transaction = VisualPerceptionTransaction(graph=graph, memory=memory)
perception = VisualPerception(runtime, transaction)

# One-shot learning
concept = await perception.learn(image, "screwdriver")

# Recognition
result = await perception.recognize(test_image)
# result.matched, result.label, result.confidence
```

### Visual Memory

| Method | Purpose |
|--------|---------|
| `search_observations()` | Primary evidence retrieval (cosine similarity) |
| `search_prototypes()` | Fast coarse retrieval via concept centroids |
| `derive_concept_candidates()` | Group observations → ranked concept candidates |
| `add_exemplar()` | Add observation with diversity enforcement |
| `update_prototype()` | Running-average centroid update |

### HCIR Visual Nodes

| Node Type | Parent | Purpose |
|-----------|--------|---------|
| `VisualObservationNode` | `ObservationNode` | Visual evidence with `embedding_ref` (not vector) |
| `VisualConceptNode` | `ConceptNode` | Learned concept with `prototype_ref` + `exemplar_refs` |

### SNN-Gated Perception Stream

```python
from hbllm.perception.visual_perception_stream import VisualPerceptionStream

stream = VisualPerceptionStream(perception=perception)

for frame in video_frames:
    result = await stream.process_frame(frame)
    if result.processed:
        print(f"Recognized: {result.recognition_label}")

print(f"Process rate: {stream.stats.process_rate:.1%}")  # e.g., 12%
```

### Processing Levels (SNN Gate)

| Level | Trigger | Action |
|-------|---------|--------|
| `NONE` | Static scene | Skip |
| `LOW` | Minor motion | Log only |
| `STANDARD` | Object movement | Full perception |
| `HIGH` | Scene change | Perception + context |
| `URGENT` | Multiple channels | Immediate + alert |

### Evaluation

```python
from hbllm.perception.evaluation.one_shot_eval import OneShotEvaluator

evaluator = OneShotEvaluator(perception)
await evaluator.teach("cup", [cup_images])
result = await evaluator.evaluate(test_cases)
# result.accuracy, result.ambiguity_rate
```

### Evidence Types

| Type | Mutability | Purpose |
|------|-----------|---------|
| `VisualEvidence` | Immutable | Raw perceptual measurement |
| `VisualAssessment` | Mutable | Current interpretation (candidates, ranking) |
| `EpistemicEvidenceProfile` | Mutable | Multi-dimensional confidence |
| `RecognitionPolicy` | Immutable | Configurable thresholds |

---

## Grounded Audio Perception Runtime (Waves A1–A6)

Audio is an evidence-producing perception modality where providers produce raw/typed observations, the `AudioPerceptionRuntime` normalizes them into structured evidence, SNN gates determine processing compute, and the `AudioPerceptionTransaction` commits observations to HCIR.

```
Microphone → AudioSignals → AudioPerceptionEnsemble (SNN Gate)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼ (should_process=True)       ▼ (should_process=False)
               Providers                        Skip (Silence/Steady)
         (Speech, Event, Scene, Speaker)
                    │
                    ▼
          AudioPerceptionRuntime
                    │
                    ▼
              AudioEvidence (composition model)
                    │
                    ▼
       AudioPerceptionTransaction (HCIR)
                    │
        AUDIO_OBSERVATION / ACOUSTIC_CONCEPT
```

### Invariants

1. **Audio perception produces evidence. HCIR commits state.** Providers never know about HCIR.
2. **SNN is an attention controller, not a classifier.** It decides *how much compute to spend*, not *what the sound is*.
3. **Composition over inheritance.** `AcousticObservation` ("the microphone received this") is composed with `SpeechEvidence`, `SoundEventEvidence`, and `AcousticSceneEvidence`.
4. **Learning creates cognitive artifacts.** `commit_learning()` creates `AcousticConceptNode` in HCIR without synchronous model training.

### Audio Provider Protocols

```python
from hbllm.perception.providers.audio_base import (
    SpeechProvider,
    AcousticEventProvider,
    AcousticSceneProvider,
    SpeakerProvider,
    SoundLocalizationProvider,
)
```

### Full Audio Pipeline Usage

```python
from hbllm.hcir.graph import CognitiveGraph
from hbllm.perception.audio_memory import AudioMemory
from hbllm.perception.audio_perception import AudioPerception
from hbllm.perception.audio_perception_runtime import AudioPerceptionRuntime
from hbllm.perception.audio_perception_transaction import AudioPerceptionTransaction
from hbllm.perception.providers.mock_audio_provider import MockAudioProvider

# 1. Initialize providers & memory
provider = MockAudioProvider()
memory = AudioMemory()
graph = CognitiveGraph()

# 2. Runtime & Transaction
runtime = AudioPerceptionRuntime(
    speech=provider,
    events=provider,
    scene=provider,
    speaker=provider,
    memory=memory,
)
transaction = AudioPerceptionTransaction(graph=graph, memory=memory)
perception = AudioPerception(runtime, transaction)

# 3. Listen (evidence only — no HCIR mutation)
assessment = await perception.listen(audio_bytes)

# 4. Learn a sound (creates AcousticConceptNode in HCIR)
concept = await perception.learn_sound(audio_bytes, "doorbell")

# 5. Recognize & commit speech
obs_node = await perception.recognize_speech(speech_audio)
```

### SNN Audio Gating Stream

```python
from hbllm.perception.audio_perception_stream import AudioPerceptionStream

stream = AudioPerceptionStream(runtime=runtime, sample_rate=16000)

for chunk in audio_stream:
    decision, assessment = await stream.process_chunk(chunk)
    if decision.should_process:
        print(f"[{decision.event_type}] Processing level: {decision.processing_level}")
```

### Audio HCIR Nodes

| Node Type | Parent | Purpose |
|-----------|--------|---------|
| `AudioObservationNode` | `ObservationNode` | Audio telemetry with `embedding_ref` + temporal identity |
| `AcousticConceptNode` | `ConceptNode` | Learned acoustic concept with prototype & exemplar refs |

### Temporal & Cross-Modal Integration

- **`TemporalFuser`**: Ingests `AudioObservationNode` and `AudioAssessment` to produce `TemporalPatternCandidate`s without premature factual commitment.
- **`PerceptionFuser`**: Aligns visual detections and audio speech/events across sliding time windows into multimodal `FusedContext`.
- **`WorldStateEngine`**: Ingests typed `AudioAssessment` and authoritative HCIR `CognitiveGraph` (dual-source gradual migration).

