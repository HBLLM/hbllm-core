"""
Cognitive Stream — Comprehension Pipeline.

SNN-driven input comprehension that decouples expensive ONNX embeddings
from the spiking neural network.  Cheap lexical signals feed the SNN;
embeddings fire only when a concept boundary is detected via spike.

Modules:
    lexical   — LexicalBuffer (Layer 0) + LexicalSignals (Layer 1)
    ensemble  — ComprehensionEnsemble (5-channel LIF neurons)
    stream    — ComprehensionStream (full pipeline)
    models    — ComprehensionUnit, UnderstandingState, ActivatedMemory
    calibrator — SNNCalibrator (parameter tuning)
"""

from hbllm.brain.snn.comprehension.calibrator import SNNCalibrator
from hbllm.brain.snn.comprehension.ensemble import (
    DOMAIN_PARAMS,
    ComprehensionEnsemble,
    ConceptSpike,
)
from hbllm.brain.snn.comprehension.lexical import (
    LexicalBuffer,
    LexicalSignals,
    populate_from_registry,
)
from hbllm.brain.snn.comprehension.models import (
    ActivatedMemory,
    ComprehensionUnit,
    UnderstandingState,
)
from hbllm.brain.snn.comprehension.stream import ComprehensionStream

__all__ = [
    # Layer 0-1
    "LexicalBuffer",
    "LexicalSignals",
    "populate_from_registry",
    # Ensemble
    "ComprehensionEnsemble",
    "ConceptSpike",
    "DOMAIN_PARAMS",
    # Stream
    "ComprehensionStream",
    # Models
    "ComprehensionUnit",
    "UnderstandingState",
    "ActivatedMemory",
    # Calibrator
    "SNNCalibrator",
]
