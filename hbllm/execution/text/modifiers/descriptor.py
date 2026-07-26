"""
Modifier Descriptor — typed, versioned modifier references.

Replaces string identifiers with rich descriptors that support:
    - Signed modifiers
    - Downloadable modifiers
    - Tenant-specific modifiers
    - Version pinning
    - Configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModifierDescriptor:
    """
    Typed, versioned modifier reference.

    Used in ExecutionPlan.modifiers instead of plain strings.
    Supports signed, downloadable, tenant-specific, and
    version-pinned modifiers.
    """

    id: str  # e.g. "lora-medical-v2"
    modifier_type: str  # "lora", "prompt", "grammar", "safety", "watermark"
    version: str = "1.0.0"
    priority: int = 0  # Higher priority → earlier in pipeline
    required_capabilities: tuple[str, ...] = ()
    configuration: dict[str, Any] = field(default_factory=dict)
    signature: str | None = None  # For signed modifiers
    source: str | None = None  # For downloadable modifiers (URL or registry path)
    tenant_id: str | None = None  # For tenant-specific modifiers

    @staticmethod
    def lora(
        adapter_name: str,
        version: str = "1.0.0",
        priority: int = 100,
        **config: Any,
    ) -> ModifierDescriptor:
        """Create a LoRA modifier descriptor."""
        return ModifierDescriptor(
            id=f"lora-{adapter_name}",
            modifier_type="lora",
            version=version,
            priority=priority,
            required_capabilities=("lora",),
            configuration={"adapter_name": adapter_name, **config},
        )

    @staticmethod
    def prompt(
        style: str,
        version: str = "1.0.0",
        priority: int = 50,
        **config: Any,
    ) -> ModifierDescriptor:
        """Create a prompt modifier descriptor."""
        return ModifierDescriptor(
            id=f"prompt-{style}",
            modifier_type="prompt",
            version=version,
            priority=priority,
            configuration={"style": style, **config},
        )

    @staticmethod
    def grammar(
        schema_name: str = "json",
        version: str = "1.0.0",
        priority: int = 75,
        **config: Any,
    ) -> ModifierDescriptor:
        """Create a grammar modifier descriptor."""
        return ModifierDescriptor(
            id=f"grammar-{schema_name}",
            modifier_type="grammar",
            version=version,
            priority=priority,
            required_capabilities=("grammar",),
            configuration={"schema": schema_name, **config},
        )

    @staticmethod
    def no_modifier() -> ModifierDescriptor:
        """Create a pass-through (no-op) modifier descriptor."""
        return ModifierDescriptor(
            id="none",
            modifier_type="none",
            version="1.0.0",
            priority=0,
        )
