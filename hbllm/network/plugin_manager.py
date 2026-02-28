"""
Plugin Manager — dynamic node discovery and hot-loading.

Discovers and loads HBLLM nodes from a plugins directory. Each plugin
is a Python file with a `register(bus, registry)` async function that
creates and starts nodes.

Plugin format:
    plugins/my_domain.py:
        __plugin__ = {"name": "my_domain", "version": "1.0", "description": "Custom domain expert"}

        async def register(bus, registry):
            from hbllm.modules.base_module import DomainModuleNode
            node = DomainModuleNode(node_id="domain_custom", domain_name="custom", ...)
            await node.start(bus)
            return [node]
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.network.bus import MessageBus
from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Metadata about a discovered plugin."""
    name: str
    path: str
    version: str = "0.0.0"
    description: str = ""
    loaded: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "version": self.version,
            "description": self.description,
            "loaded": self.loaded,
            "error": self.error,
        }


class PluginManager:
    """
    Discovers and manages HBLLM plugins.

    Plugins are Python files in configured directories that follow the
    plugin convention (define `register` function and optional `__plugin__` dict).

    Lifecycle:
      1. discover() — scan directories for plugin files
      2. load_all() — import and register all discovered plugins
      3. unload_all() — stop all plugin-created nodes
    """

    def __init__(
        self,
        plugin_dirs: list[str | Path] | None = None,
        bus: MessageBus | None = None,
        registry: ServiceRegistry | None = None,
    ):
        self._plugin_dirs = [Path(d) for d in (plugin_dirs or ["plugins"])]
        self._bus = bus
        self._registry = registry
        self._plugins: dict[str, PluginInfo] = {}
        self._loaded_nodes: list[Any] = []

    @property
    def plugins(self) -> dict[str, PluginInfo]:
        """All discovered plugins."""
        return dict(self._plugins)

    @property
    def loaded_count(self) -> int:
        """Number of successfully loaded plugins."""
        return sum(1 for p in self._plugins.values() if p.loaded)

    def discover(self) -> list[PluginInfo]:
        """
        Scan plugin directories for Python files with register() functions.
        Returns list of discovered plugins.
        """
        discovered = []

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                logger.debug("Plugin directory does not exist: %s", plugin_dir)
                continue

            for py_file in sorted(plugin_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue

                plugin_name = py_file.stem
                info = PluginInfo(
                    name=plugin_name,
                    path=str(py_file),
                )

                # Try to extract __plugin__ metadata without full import
                try:
                    source = py_file.read_text()
                    if "def register" not in source and "async def register" not in source:
                        logger.debug("Skipping %s — no register() function", py_file)
                        continue

                    # Extract __plugin__ dict if present
                    if "__plugin__" in source:
                        spec = importlib.util.spec_from_file_location(plugin_name, py_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            try:
                                spec.loader.exec_module(module)
                                plugin_meta = getattr(module, "__plugin__", {})
                                info.name = plugin_meta.get("name", plugin_name)
                                info.version = plugin_meta.get("version", "0.0.0")
                                info.description = plugin_meta.get("description", "")
                            except Exception as e:
                                logger.warning("Could not load metadata from %s: %s", py_file, e)

                except Exception as e:
                    info.error = str(e)
                    logger.warning("Error scanning plugin %s: %s", py_file, e)

                self._plugins[plugin_name] = info
                discovered.append(info)
                logger.info("Discovered plugin: %s (%s)", info.name, py_file)

        return discovered

    async def load_all(
        self,
        bus: MessageBus | None = None,
        registry: ServiceRegistry | None = None,
    ) -> list[PluginInfo]:
        """
        Load and register all discovered plugins.

        Args:
            bus: MessageBus to pass to plugin register() functions.
            registry: ServiceRegistry for node registration.

        Returns:
            List of all plugins with updated load status.
        """
        bus = bus or self._bus
        registry = registry or self._registry

        if not bus:
            raise RuntimeError("MessageBus required to load plugins")

        results = []
        for name, info in self._plugins.items():
            if info.loaded:
                results.append(info)
                continue

            try:
                spec = importlib.util.spec_from_file_location(name, info.path)
                if not spec or not spec.loader:
                    info.error = "Could not create module spec"
                    results.append(info)
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                register_fn = getattr(module, "register", None)
                if not register_fn:
                    info.error = "No register() function found"
                    results.append(info)
                    continue

                # Call register(bus, registry) or register(bus)
                import inspect
                sig = inspect.signature(register_fn)
                if len(sig.parameters) >= 2 and registry:
                    nodes = await register_fn(bus, registry)
                else:
                    nodes = await register_fn(bus)

                if nodes:
                    if isinstance(nodes, list):
                        self._loaded_nodes.extend(nodes)
                    else:
                        self._loaded_nodes.append(nodes)

                    # Register with service registry
                    if registry:
                        for node in (nodes if isinstance(nodes, list) else [nodes]):
                            if hasattr(node, "get_info"):
                                await registry.register(node.get_info())

                info.loaded = True
                info.error = None
                logger.info("Loaded plugin: %s (%d nodes)", name, len(nodes) if isinstance(nodes, list) else 1)

            except Exception as e:
                info.error = str(e)
                logger.error("Failed to load plugin %s: %s", name, e)

            results.append(info)

        return results

    async def unload_all(self) -> None:
        """Stop all plugin-created nodes."""
        for node in self._loaded_nodes:
            try:
                if hasattr(node, "stop"):
                    await node.stop()
            except Exception as e:
                logger.warning("Error stopping plugin node: %s", e)

        self._loaded_nodes.clear()
        for info in self._plugins.values():
            info.loaded = False

        logger.info("All plugins unloaded")

    def list_plugins(self) -> list[dict[str, Any]]:
        """Return all plugins as dicts for CLI/API output."""
        return [info.to_dict() for info in self._plugins.values()]
