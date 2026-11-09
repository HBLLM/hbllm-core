"""Domain modules — specialized nodes with LoRA adapters."""

from __future__ import annotations

from pathlib import Path

_exp_path = Path(__file__).resolve().parents[2] / "experiments" / "legacy_llm" / "modules"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))


from .adapter_registry import AdapterRegistryConfig, AdapterSource


def __getattr__(name: str):
    if name == "AdapterRegistry":
        from .adapter_registry import AdapterRegistry

        return AdapterRegistry
    if name == "DomainModuleNode":
        from .base_module import DomainModuleNode

        return DomainModuleNode
    if name == "LoRAManager":
        from .lora import LoRAManager

        return LoRAManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdapterRegistry",
    "AdapterRegistryConfig",
    "AdapterSource",
    "DomainModuleNode",
    "LoRAManager",
]
