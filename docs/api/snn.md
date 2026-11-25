---
title: "API Reference — Neuromorphic SNN Subsystem"
description: "API reference for HBLLM's Spiking Neural Network modules (hbllm.hcir.snn): LIFPopulation, SpikingNetwork, ComprehensionStream, STDP, WTACircuit, NeuromodulationEngine, OscillationManager, and SNN perception gating."
---

# SNN Neuromorphic Engine API

The **SNN Neuromorphic Engine** provides spike-rate neural computation, low-power continuous perception gating, neuromodulation, and STDP plasticity inside the LLM-free HCIR substrate.

**Canonical Package:** `hbllm.hcir.snn` *(compatibility re-exports at `hbllm.brain.snn`)*

---

## Subsystem Index

| Class | Module | Role |
|---|---|---|
| `LIFConfig` / `LIFNeuron` / `SpikingAccumulator` | `hbllm.hcir.snn.lif` | Leaky Integrate-and-Fire neuron dynamics and spike accumulation |
| `IzhikevichNeuron` / `SpikeEvent` | `hbllm.hcir.snn.neurons` | Bursting/chattering neuron models and spike events |
| `DendriticNeuron` | `hbllm.hcir.snn.dendrite` | Predictive-coding dendritic compartments |
| `SpikingNetwork` | `hbllm.hcir.snn.network` | Multi-layer recurrent spiking neural network graph |
| `STDP` / `STDPRule` | `hbllm.hcir.snn.plasticity` | Spike-Timing-Dependent Plasticity learning engine |
| `STPManager` | `hbllm.hcir.snn.stp` | Short-Term Plasticity (depression/facilitation) |
| `WTACircuit` | `hbllm.hcir.snn.wta` | Winner-Take-All lateral inhibition competitive circuit |
| `PopulationEncoder` | `hbllm.hcir.snn.population` | Continuous state to population spike train encoder |
| `NeuromodulationEngine` | `hbllm.hcir.snn.neuromodulation` | Global dopamine, serotonin, ACh, and noradrenaline dynamics |
| `OscillationManager` | `hbllm.hcir.snn.oscillations` | Cortical oscillation clocks ($\theta, \alpha, \beta, \gamma$) |
| `ComprehensionStream` | `hbllm.hcir.snn.comprehension.stream` | Continuous streaming comprehension ensemble |
| `PerceptionEnsemble` | `hbllm.hcir.snn.perception.visual_ensemble` | 5-channel visual perception SNN gating |
| `VisualSignalExtractor` | `hbllm.hcir.snn.perception.visual_signals` | Fast NumPy visual frame feature extraction |
| `AudioPerceptionEnsemble` | `hbllm.hcir.snn.perception.audio_ensemble` | Acoustic perception gating ensemble |
| `SNNPerceptualFeatureExtractor` | `hbllm.hcir.grounding.features.feature_extractor` | Spiking feature extractor for `perception_mode="snn"` |

---

## Key Interfaces

### `LIFNeuron` & `SpikingAccumulator`

```python
import numpy as np
from hbllm.hcir.snn.lif import LIFConfig, LIFNeuron

config = LIFConfig(
    tau_m=20.0,
    v_thresh=-50.0,
    v_reset=-70.0,
    refractory_ms=2.0,
)
neuron = LIFNeuron(config=config)
fired = neuron.step(input_current=2.5, dt_ms=1.0)
print(f"Fired: {fired}, V_m={neuron.v_m:.2f}")
```

### `ComprehensionStream`

```python
from hbllm.hcir.snn.comprehension.stream import ComprehensionStream

stream = ComprehensionStream()
state = await stream.process_chunk("Critical alert: high latency on node-3")
print(f"Understanding state: {state}")
```

### `NeuromodulationEngine`

```python
from hbllm.hcir.snn.neuromodulation import NeuromodulationEngine

neuro = NeuromodulationEngine()
neuro.signal_novelty(0.5)
neuro.signal_reward(0.3)
```

### Visual Perception Gating

```python
from hbllm.hcir.snn.perception.visual_ensemble import PerceptionEnsemble
from hbllm.hcir.snn.perception.visual_signals import VisualSignalExtractor

extractor = VisualSignalExtractor(downsample=4)
ensemble = PerceptionEnsemble()

signals = extractor.extract(frame)
decision = ensemble.step(signals)
if decision.should_process:
    print("High-salience frame -> forward to grounding")
```
