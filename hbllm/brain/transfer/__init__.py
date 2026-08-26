"""A20 Relational Generalization & Analogical Transfer Package.

Provides RelationalSchema representation, schema induction from grounded episodes,
deterministic Gentner Structure Mapping, constraint-gated negative transfer rejection,
and zero-shot analogical plan synthesis.
"""

from hbllm.brain.transfer.engine import (
    AnalogicalTransfer,
    AnalogicalTransferEngine,
    ConditionalAnalogicalHypothesis,
)
from hbllm.brain.transfer.extractor import (
    GroundedEpisode,
    RelationalSchemaExtractor,
)
from hbllm.brain.transfer.mapper import (
    MappingStatus,
    StructuralMappingResult,
    StructureMappingEngine,
)
from hbllm.brain.transfer.schema import (
    ActionTemplate,
    ConsequenceTemplate,
    RelationalSchema,
    SchemaConstraint,
    SchemaLifecycleStatus,
    SchemaRelation,
    SchemaRole,
)

__all__ = [
    "ActionTemplate",
    "AnalogicalTransfer",
    "AnalogicalTransferEngine",
    "ConditionalAnalogicalHypothesis",
    "ConsequenceTemplate",
    "GroundedEpisode",
    "MappingStatus",
    "RelationalSchema",
    "RelationalSchemaExtractor",
    "SchemaConstraint",
    "SchemaLifecycleStatus",
    "SchemaRelation",
    "SchemaRole",
    "StructuralMappingResult",
    "StructureMappingEngine",
]
