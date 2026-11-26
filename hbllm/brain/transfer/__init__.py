"""A20 Relational Generalization & Analogical Transfer Package.

Provides RelationalSchema representation, schema induction from grounded episodes,
deterministic Gentner Structure Mapping, constraint-gated negative transfer rejection,
and zero-shot analogical plan synthesis.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AnalogicalTransfer": ("hbllm.brain.transfer.engine", "AnalogicalTransfer"),
    "AnalogicalTransferEngine": ("hbllm.brain.transfer.engine", "AnalogicalTransferEngine"),
    "ConditionalAnalogicalHypothesis": (
        "hbllm.brain.transfer.engine",
        "ConditionalAnalogicalHypothesis",
    ),
    "GroundedEpisode": ("hbllm.brain.transfer.extractor", "GroundedEpisode"),
    "RelationalSchemaExtractor": ("hbllm.brain.transfer.extractor", "RelationalSchemaExtractor"),
    "MappingStatus": ("hbllm.brain.transfer.mapper", "MappingStatus"),
    "StructuralMappingResult": ("hbllm.brain.transfer.mapper", "StructuralMappingResult"),
    "StructureMappingEngine": ("hbllm.brain.transfer.mapper", "StructureMappingEngine"),
    "ActionTemplate": ("hbllm.brain.transfer.schema", "ActionTemplate"),
    "ConsequenceTemplate": ("hbllm.brain.transfer.schema", "ConsequenceTemplate"),
    "RelationalSchema": ("hbllm.brain.transfer.schema", "RelationalSchema"),
    "SchemaConstraint": ("hbllm.brain.transfer.schema", "SchemaConstraint"),
    "SchemaLifecycleStatus": ("hbllm.brain.transfer.schema", "SchemaLifecycleStatus"),
    "SchemaRelation": ("hbllm.brain.transfer.schema", "SchemaRelation"),
    "SchemaRole": ("hbllm.brain.transfer.schema", "SchemaRole"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_path, attr = _EXPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
