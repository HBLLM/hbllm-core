"""
SNN Reasoning modules.

Provides higher-level reasoning layers built on top of the
multi-layer SpikingNetwork framework.

Components:
    AssociationLayer — discovers relationships between comprehension concepts
"""

from hbllm.brain.snn.reasoning.association import (
    AssociationLayer,
    ConceptAssociation,
)

__all__ = [
    "AssociationLayer",
    "ConceptAssociation",
]
