"""
Spiking Neural Network (SNN) modules for cognitive modeling.
"""

from hbllm.brain.snn.lif import LIFConfig, LIFNeuron, SpikeEvent, SpikingAccumulator
from hbllm.brain.snn.network import LayerProjection, NeuronLayer, SpikingNetwork
from hbllm.brain.snn.plasticity import PlasticWeightMatrix, STDPRule, SynapticConnection

__all__ = [
    "LIFConfig",
    "LIFNeuron",
    "SpikeEvent",
    "SpikingAccumulator",
    # Multi-layer network framework
    "NeuronLayer",
    "LayerProjection",
    "SpikingNetwork",
    # Plasticity (STDP)
    "SynapticConnection",
    "STDPRule",
    "PlasticWeightMatrix",
    # Comprehension subpackage
    "comprehension",
    # Expression subpackage
    "expression",
    # Reasoning subpackage
    "reasoning",
]
