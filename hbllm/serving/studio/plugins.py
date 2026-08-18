"""
Studio Plugin Management Endpoints.

Exposes plugin listing, toggling, marketplace discovery, install/uninstall.
Includes the ``PluginMarketplace`` helper class for registry-based
plugin distribution.

Extracted from ``_legacy.py`` — see Work Stream 1.
"""

from __future__ import annotations

import json
import logging
import pathlib
import shutil
import urllib.request
import zipfile
from io import BytesIO
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hbllm.serving.state import _state
from hbllm.serving.studio.helpers import get_brain

logger = logging.getLogger(__name__)

router = APIRouter()


class PluginMarketplace:
    def __init__(self, plugins_dir: pathlib.Path):
        self.plugins_dir = plugins_dir
        self.registry_path = (
            pathlib.Path(__file__).resolve().parent.parent.parent.parent
            / "sentra-plugins"
            / "registry.json"
        )

    async def list_available(self) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            logger.warning("Marketplace registry file not found: %s", self.registry_path)
            return []
        try:
            with open(self.registry_path) as f:
                data = json.load(f)
            plugins = data.get("plugins", [])
            for p in plugins:
                p["installed"] = (self.plugins_dir / p["name"]).exists()
            return plugins
        except Exception as e:
            logger.error("Failed to read registry: %s", e)
            return []

    async def install(self, plugin_name: str) -> dict[str, Any]:
        plugins = await self.list_available()
        plugin_info = next((p for p in plugins if p["name"] == plugin_name), None)
        if not plugin_info:
            return {"status": "error", "error": f"Plugin '{plugin_name}' not found"}

        download_url = plugin_info.get("download_url")

        try:
            local_source = self.registry_path.parent / "plugins" / plugin_name
            target_dir = self.plugins_dir / plugin_name
            if target_dir.exists():
                shutil.rmtree(target_dir)

            if local_source.exists() and local_source.is_dir():
                shutil.copytree(local_source, target_dir)
                logger.info("Installed plugin %s locally from sentra-plugins", plugin_name)
            elif download_url:
                req = urllib.request.Request(
                    download_url, headers={"User-Agent": "HBLLM-Marketplace/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()

                with zipfile.ZipFile(BytesIO(data)) as zf:
                    zf.extractall(target_dir)
                logger.info("Installed plugin %s from URL", plugin_name)
            else:
                return {"status": "error", "error": "No download source available"}

            return {
                "status": "installed",
                "name": plugin_name,
                "version": plugin_info.get("version"),
            }
        except Exception as e:
            logger.error("Install failed for %s: %s", plugin_name, e)
            return {"status": "error", "error": str(e)}

    def uninstall(self, plugin_name: str) -> dict[str, Any]:
        target_dir = self.plugins_dir / plugin_name
        if not target_dir.exists():
            return {"status": "error", "error": "Not installed"}
        try:
            shutil.rmtree(target_dir)
            return {"status": "uninstalled", "name": plugin_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}


@router.get("/api/plugins")
async def list_plugins():
    pm = _state.get("plugin_manager")
    if not pm:
        return {"plugins": []}

    # Call discover() to pick up any newly installed/deleted folders
    pm.discover()

    raw_plugins = pm.list_plugins()
    mapped = []
    for p in raw_plugins:
        mapped.append(
            {
                "name": p["name"],
                "enabled": p["loaded"],
                "loaded": p["loaded"],
                "description": p["description"] or "No description available",
                "version": p["version"] or "0.1.0",
                "path": p["path"],
                "error": p["error"],
            }
        )
    return {"plugins": mapped}


@router.post("/api/plugins/{plugin_name}/toggle")
async def toggle_plugin(plugin_name: str):
    pm = _state.get("plugin_manager")
    if not pm:
        raise HTTPException(status_code=503, detail="PluginManager not initialized")

    enabled = await pm.toggle_plugin(plugin_name)
    return {"plugin": plugin_name, "enabled": enabled}


@router.get("/api/plugins/marketplace")
async def marketplace_list():
    pm = _state.get("plugin_manager")
    if not pm or not pm._plugin_dirs:
        raise HTTPException(status_code=503, detail="PluginManager not initialized")
    plugins_dir = pathlib.Path(pm._plugin_dirs[0])
    mp = PluginMarketplace(plugins_dir)
    available = await mp.list_available()
    return {"plugins": available}


@router.post("/api/plugins/install")
async def marketplace_install(request: Request):
    pm = _state.get("plugin_manager")
    if not pm or not pm._plugin_dirs:
        raise HTTPException(status_code=503, detail="PluginManager not initialized")
    body = await request.json()
    plugin_name = body.get("name")
    if not plugin_name:
        return {"status": "error", "error": "Missing 'name'"}

    plugins_dir = pathlib.Path(pm._plugin_dirs[0])
    mp = PluginMarketplace(plugins_dir)
    result = await mp.install(plugin_name)

    if result.get("status") == "installed":
        brain = get_brain()
        if brain and hasattr(brain, "plugin_manager") and brain.plugin_manager:
            try:
                await brain.plugin_manager.load_bundle(plugins_dir / plugin_name)
            except Exception as e:
                logger.error("Failed to hot-load installed plugin %s: %s", plugin_name, e)

    pm.discover()
    return result


@router.delete("/api/plugins/uninstall")
async def marketplace_uninstall(request: Request):
    pm = _state.get("plugin_manager")
    if not pm or not pm._plugin_dirs:
        raise HTTPException(status_code=503, detail="PluginManager not initialized")
    body = await request.json()
    plugin_name = body.get("name")
    if not plugin_name:
        return {"status": "error", "error": "Missing 'name'"}

    info = pm._plugins.get(plugin_name)
    if info and info.loaded:
        await pm.toggle_plugin(plugin_name)
    else:
        brain = get_brain()
        if brain and hasattr(brain, "plugin_manager") and brain.plugin_manager:
            if plugin_name in brain.plugin_manager.bundles:
                try:
                    await brain.plugin_manager.unload_bundle(plugin_name)
                except Exception as e:
                    logger.error("Failed to unload bundle %s during uninstall: %s", plugin_name, e)

    plugins_dir = pathlib.Path(pm._plugin_dirs[0])
    mp = PluginMarketplace(plugins_dir)
    result = mp.uninstall(plugin_name)

    pm.discover()
    return result
