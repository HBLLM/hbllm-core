"""
HBLLM Plugin SDK.
Provides declarative interfaces to write third-party extensions.
"""

from hbllm.plugin.bundle import PluginAssets, PluginBundle, PluginManifest
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

__all__ = [
    "HBLLMPlugin",
    "subscribe",
    "PluginBundle",
    "PluginManifest",
    "PluginAssets",
]
