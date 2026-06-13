"""
SNN Reasoning modules.

Provides higher-level reasoning layers built on top of the
multi-layer SpikingNetwork framework.

Components:
    AssociationLayer   — discovers relationships between comprehension concepts
    ReasoningNetwork   — SNN for evaluating causal chain quality
    CausalReasoner     — multi-hop causal graph traversal with SNN evaluation
"""

from hbllm.brain.snn.reasoning.association import (
    AssociationLayer,
    ConceptAssociation,
)
from hbllm.brain.snn.reasoning.reasoner import (
    CausalChain,
    CausalReasoner,
)
from hbllm.brain.snn.reasoning.reasoning_network import ReasoningNetwork

__all__ = [
    "AssociationLayer",
    "ConceptAssociation",
    "ReasoningNetwork",
    "CausalReasoner",
    "CausalChain",
]

