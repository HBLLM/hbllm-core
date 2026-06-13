"""
Spiking Neural Network (SNN) modules for cognitive modeling.
"""

from hbllm.brain.snn.lif import LIFConfig, LIFNeuron, SpikeEvent, SpikingAccumulator

__all__ = [
    "LIFConfig",
    "LIFNeuron",
    "SpikeEvent",
    "SpikingAccumulator",
    # Comprehension subpackage
    "comprehension",
]
