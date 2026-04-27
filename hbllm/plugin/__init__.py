"""
HBLLM Plugin SDK & Manager.

Provides:
  - ``HBLLMPlugin`` / ``@subscribe`` — declarative base for code-only plugins
  - ``PluginBundle`` / ``PluginManifest`` / ``PluginAssets`` — bundle data loader
  - ``PluginManager`` / ``LoadedBundle`` / ``PromptStore`` — lifecycle manager
"""

from hbllm.plugin.bundle import PluginAssets, PluginBundle, PluginManifest
from hbllm.plugin.manager import LoadedBundle, PluginManager, PromptStore
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

__all__ = [
    "HBLLMPlugin",
    "subscribe",
    "PluginBundle",
    "PluginManifest",
    "PluginAssets",
    "PluginManager",
    "LoadedBundle",
    "PromptStore",
]
