---
title: "API Reference — Neuromorphic SNN Subsystem"
description: "API reference for Spiking Neural Network (SNN) modules: LIFNeuron, SNNNetwork, ComprehensionStream, ExpressionStream, BrocaEncoder, TrainedPRM, and Neuromodulation."
---

# SNN Neuromorphic Engine API

The **SNN Neuromorphic Engine** provides spike-rate neural computation, low-power continuous perception, thought planning, and 3-tier prompt compression.

**Package:** `hbllm.brain.snn`

---

## Subsystem Index

| Class | Module | Role |
|---|---|---|
| `LIFNeuron` / `LIFPopulation` | `lif.py` | Leaky Integrate-and-Fire spiking neuron models |
| `SNNNetwork` | `network.py` | Multi-layer recurrent spiking neural network graph |
| `ComprehensionStream` | `comprehension/stream.py` | 5-channel real-time continuous comprehension engine |
| `ExpressionStream` | `expression/expression_stream.py` | 3-tier adaptive thought articulation pipeline |
| `ThoughtController` | `expression/thought_controller.py` | Cognitive planning and rendering tier selector |
| `TrainedPRM` | `expression/trained_prm.py` | 4-layer spiking Process Reward Model ($6 \to 8 \to 4 \to 2$) |
| `BrocaEncoder` | `expression/broca_encoder.py` | Ultra-compact ~80-token prompt builder for LLM |
| `ShallowRenderer` | `expression/shallow_renderer.py` | Structured ~300-token prompt builder for LLM |
| `STDP` | `plasticity.py` | Spike-Timing-Dependent Plasticity learning engine |
| `WTACircuit` | `wta.py` | Winner-Take-All lateral inhibition competitive circuit |
| `NeuromodulationEngine` | `neuromodulation.py` | Global dopamine, serotonin, ACh, and noradrenaline dynamics |
| `OscillationManager` | `oscillations.py` | Cortical oscillation clocks ($\theta, \alpha, \beta, \gamma$) |
| `PerceptionEnsemble` | `perception/visual_ensemble.py` | 5-channel visual perception SNN gating |
| `VisualSignalExtractor` | `perception/visual_signals.py` | Cheap frame-level feature extraction |
| `PerceptionGateDecision` | `perception/gate.py` | SNN-driven processing level decision |

---

## Key Interfaces

### `LIFPopulation`

```python
import numpy as np
from hbllm.brain.snn.lif import LIFConfig, LIFPopulation

config = LIFConfig(
    num_neurons=64,
    tau_m=20.0,  # Membrane time constant (ms)
    v_thresh=-50.0,  # Threshold potential (mV)
    v_reset=-70.0,  # Reset potential (mV)
    refractory_period=2.0,  # ms
)

pop = LIFPopulation(config)

# Inject current tensor (shape: [num_neurons])
currents = np.random.uniform(0.0, 5.0, size=(64,))
spikes = pop.step(currents, dt=1.0)
print(f"Active firing neurons: {np.sum(spikes)}")
```

---

### `ComprehensionStream`

```python
from hbllm.brain.snn.comprehension.stream import ComprehensionStream

stream = ComprehensionStream()

# Ingest streaming text chunks
state = await stream.process_chunk("Critical alert: High latency on node-3")
print(f"Affective Urgency: {state.urgency_rate:.2f}")
print(f"Syntactic Complexity: {state.syntactic_complexity:.2f}")
```

---

### `TrainedPRM` & `BrocaEncoder`

```python
from hbllm.brain.snn.expression.broca_encoder import BrocaEncoder
from hbllm.brain.snn.expression.trained_prm import TrainedPRM

prm = TrainedPRM()
encoder = BrocaEncoder()

# Evaluate candidate thought step
score = prm.evaluate_step(
    goal="Fix cluster",
    action="Drain node-3",
    features=[0.9, 0.8, 0.95, 0.2, 0.9, 1.0],
)
print(f"PRM Step Reward Score: {score:.3f}")

# Compress thought to minimal prompt
compressed_prompt = encoder.encode(intent="drain_node", parameters={"node": "node-3"})
print(f"Compressed Broca tokens length: ~{len(compressed_prompt.split())}")
```

---

### Visual Perception Gating

The SNN perception subsystem decides **when** and **how much** visual
computation to spend on incoming video frames.

```python
from hbllm.brain.snn.perception.visual_signals import VisualSignalExtractor
from hbllm.brain.snn.perception.visual_ensemble import PerceptionEnsemble

extractor = VisualSignalExtractor(downsample=4)
ensemble = PerceptionEnsemble()

signals = extractor.extract(frame)  # ~0.1ms, numpy only
decision = ensemble.step(signals)  # ~0.01ms, 5 LIF neurons

if decision.should_process:
    # Run expensive VisionProvider encoding
    assessment = await runtime.perceive(frame)
```

**Channels:** scene (slow), entity (medium), motion (fast), novelty (medium-slow), stability (very slow).

**Outputs:** `PerceptionProcessingLevel` — NONE / LOW / STANDARD / HIGH / URGENT.
