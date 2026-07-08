"""
Spiking Neural Network (SNN) modules for cognitive modeling.
"""

from hbllm.brain.neuromodulation import NeuromodulationEngine, NeuromodulatorState
from hbllm.brain.snn.dendrite import DendriticConfig, DendriticNeuron
from hbllm.brain.snn.lif import LIFConfig, LIFNeuron, SpikeEvent, SpikingAccumulator
from hbllm.brain.snn.network import LayerProjection, NeuronLayer, ProjectionType, SpikingNetwork
from hbllm.brain.snn.neurons import (
    BaseNeuron,
    IzhikevichConfig,
    IzhikevichNeuron,
    create_neuron_from_dict,
    register_neuron_type,
)
from hbllm.brain.snn.oscillations import OscillationBand, OscillationManager
from hbllm.brain.snn.plasticity import PlasticWeightMatrix, STDPRule, SynapticConnection
from hbllm.brain.snn.population import CognitiveStateEncoder, PopulationEncoder

__all__ = [
    # Base neuron interface
    "BaseNeuron",
    "create_neuron_from_dict",
    "register_neuron_type",
    # LIF model (original)
    "LIFConfig",
    "LIFNeuron",
    "SpikeEvent",
    "SpikingAccumulator",
    # Izhikevich model (v4)
    "IzhikevichConfig",
    "IzhikevichNeuron",
    # Dendritic model (M3 — predictive coding)
    "DendriticConfig",
    "DendriticNeuron",
    # Multi-layer network framework
    "NeuronLayer",
    "LayerProjection",
    "ProjectionType",
    "SpikingNetwork",
    # Plasticity (STDP)
    "SynapticConnection",
    "STDPRule",
    "PlasticWeightMatrix",
    # Neuromodulation
    "NeuromodulationEngine",
    "NeuromodulatorState",
    # Population coding (M3)
    "PopulationEncoder",
    "CognitiveStateEncoder",
    # Neural oscillations (M5)
    "OscillationManager",
    "OscillationBand",
]
