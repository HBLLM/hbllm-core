"""Memory system — working, episodic, semantic, and procedural memory."""

from __future__ import annotations

import importlib
from typing import Any

from hbllm.memory.interface import MemoryType, SearchResult, UnifiedMemoryInterface

_EXPORTS: dict[str, tuple[str, str]] = {
    "MemoryNode": ("hbllm.memory.memory_node", "MemoryNode"),
    "ConceptExtractor": ("hbllm.memory.concept_extractor", "ConceptExtractor"),
}


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


__all__ = ["MemoryNode", "ConceptExtractor", "UnifiedMemoryInterface", "MemoryType", "SearchResult"]
