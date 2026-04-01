"""Domain modules — specialized nodes with LoRA adapters."""

from .adapter_registry import AdapterRegistry, AdapterRegistryConfig, AdapterSource
from .base_module import DomainModuleNode
from .lora import LoRAManager
