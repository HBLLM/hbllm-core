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
import json
import logging
from dataclasses import dataclass
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
        app: Any | None = None,
        brain: Any | None = None,
    ):
        """
        Initialize the plugin manager.

        Args:
            plugin_dirs: List of directories to scan for plugins.
            bus: Default message bus to pass to plugins.
            registry: Default service registry.
            app: Default FastAPI app instance to pass to plugins desiring API endpoints.
            brain: The running Brain instance to coordinate with its plugin bundle system.
        """
        self._plugin_dirs = [Path(d) for d in (plugin_dirs or ["plugins"])]
        self._bus = bus
        self._registry = registry
        self._app = app
        self._brain = brain
        self._plugins: dict[str, PluginInfo] = {}
        self._loaded_nodes: list[Any] = []
        self._plugin_nodes: dict[str, list[Any]] = {}

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
        Scan plugin directories for Python files with register() functions
        or directories with plugin.json manifests (v2 bundles).
        Returns list of discovered plugins.
        """
        discovered = []
        discovered_names = set()

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                logger.debug("Plugin directory does not exist: %s", plugin_dir)
                continue

            for item in sorted(plugin_dir.iterdir()):
                if item.name.startswith("_"):
                    continue

                is_v2 = False
                py_file = None
                plugin_name = None

                if item.is_file() and item.suffix == ".py":
                    py_file = item
                    plugin_name = item.stem
                elif item.is_dir():
                    manifest_file = item / "plugin.json"
                    if manifest_file.exists():
                        is_v2 = True
                        plugin_name = item.name
                    elif (item / "__init__.py").exists():
                        py_file = item / "__init__.py"
                        plugin_name = item.name
                    else:
                        continue
                else:
                    continue

                existing = self._plugins.get(plugin_name)
                info = PluginInfo(
                    name=plugin_name,
                    path=str(py_file or item),
                )
                if existing:
                    info.loaded = existing.loaded

                # Check load status from brain's plugin_manager for v2 bundles
                if (
                    is_v2
                    and self._brain
                    and hasattr(self._brain, "plugin_manager")
                    and self._brain.plugin_manager
                ):
                    if plugin_name in self._brain.plugin_manager.bundles:
                        info.loaded = True

                if is_v2:
                    # Read metadata from plugin.json
                    try:
                        manifest_file = item / "plugin.json"
                        manifest_data = json.loads(manifest_file.read_text())
                        info.name = manifest_data.get("name", plugin_name)
                        info.version = manifest_data.get("version", "0.0.0")
                        info.description = manifest_data.get("description", "")
                    except Exception as e:
                        logger.warning("Could not read manifest from %s: %s", item, e)
                        info.error = f"Manifest error: {e}"
                else:
                    # Try to extract __plugin__ metadata without full import for v1 py files
                    try:
                        source = py_file.read_text()
                        if (
                            "def register" not in source
                            and "async def register" not in source
                            and "HBLLMPlugin" not in source
                        ):
                            logger.debug(
                                "Skipping %s — no register() function or SDK subclass", py_file
                            )
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
                                    logger.warning(
                                        "Could not load metadata from %s: %s", py_file, e
                                    )

                    except Exception as e:
                        info.error = str(e)
                        logger.warning("Error scanning plugin %s: %s", py_file, e)

                self._plugins[plugin_name] = info
                discovered.append(info)
                discovered_names.add(plugin_name)
                logger.info("Discovered plugin: %s (%s)", info.name, py_file or item)

        # Remove deleted/uninstalled plugins from memory cache
        for name in list(self._plugins.keys()):
            if name not in discovered_names:
                del self._plugins[name]

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
        results = []
        for name in list(self._plugins.keys()):
            await self.load_single_plugin(name, bus=bus, registry=registry)
            results.append(self._plugins[name])
        return results

    async def load_single_plugin(
        self,
        name: str,
        bus: MessageBus | None = None,
        registry: ServiceRegistry | None = None,
    ) -> bool:
        """Load a single plugin by name if discovered."""
        info = self._plugins.get(name)
        if not info or info.loaded:
            return False

        bus = bus or self._bus
        registry = registry or self._registry

        if not bus:
            raise RuntimeError("MessageBus required to load plugins")

        try:
            import inspect

            from hbllm.plugin.sdk import HBLLMPlugin

            spec = importlib.util.spec_from_file_location(name, info.path)
            if not spec or not spec.loader:
                info.error = "Could not create module spec"
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            nodes = []
            register_fn = getattr(module, "register", None)

            if register_fn:
                # Call register() with dynamic injection
                sig = inspect.signature(register_fn)
                kwargs = {}
                if "bus" in sig.parameters:
                    kwargs["bus"] = bus
                if "registry" in sig.parameters and registry is not None:
                    kwargs["registry"] = registry
                if "app" in sig.parameters and self._app is not None:
                    kwargs["app"] = self._app

                res = await register_fn(**kwargs)
                if res:
                    nodes.extend(res if isinstance(res, list) else [res])

            # Find HBLLMPlugin subclasses
            for obj_name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, HBLLMPlugin) and obj is not HBLLMPlugin:
                    instance = obj(node_id=f"{name}_{obj_name.lower()}")
                    await instance.start(bus)
                    nodes.append(instance)

            if not nodes:
                info.error = "No register() hook or SDK subclasses found"
                return False

            self._loaded_nodes.extend(nodes)
            self._plugin_nodes[name] = nodes

            # Register with service registry
            if registry:
                for node in nodes:
                    if hasattr(node, "get_info"):
                        await registry.register(node.get_info())

            info.loaded = True
            info.error = None
            logger.info("Loaded plugin: %s (%d nodes)", name, len(nodes))
            return True

        except Exception as e:
            info.error = str(e)
            logger.error("Failed to load plugin %s: %s", name, e)
            return False

    async def toggle_plugin(self, name: str) -> bool:
        """Toggle a plugin between loaded and unloaded states."""
        info = self._plugins.get(name)
        if not info:
            return False

        # Check if this is a v2 plugin bundle
        plugin_path = Path(info.path)
        is_v2 = False
        if plugin_path.is_dir():
            is_v2 = (plugin_path / "plugin.json").exists()
        elif plugin_path.name == "__init__.py":
            is_v2 = (plugin_path.parent / "plugin.json").exists()

        if is_v2:
            if (
                self._brain
                and hasattr(self._brain, "plugin_manager")
                and self._brain.plugin_manager
            ):
                bpm = self._brain.plugin_manager
                if name in bpm.bundles:
                    # Unload it
                    await bpm.unload_bundle(name)
                    info.loaded = False
                    logger.info("Unloaded v2 plugin: %s via brain.plugin_manager", name)
                    return False
                else:
                    # Load it
                    plugin_dir = plugin_path if plugin_path.is_dir() else plugin_path.parent
                    await bpm.load_bundle(plugin_dir)
                    info.loaded = True
                    logger.info("Loaded v2 plugin: %s via brain.plugin_manager", name)
                    return True
            else:
                logger.warning(
                    "Cannot toggle v2 plugin %s: brain.plugin_manager not available", name
                )
                return False

        if info.loaded:
            # Unload it
            nodes = self._plugin_nodes.pop(name, [])
            for node in nodes:
                try:
                    if hasattr(node, "stop"):
                        await node.stop()
                    if node in self._loaded_nodes:
                        self._loaded_nodes.remove(node)
                except Exception as e:
                    logger.warning("Error stopping plugin node: %s", e)
            info.loaded = False
            logger.info("Unloaded plugin: %s", name)
            return False
        else:
            # Load it
            return await self.load_single_plugin(name)

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
