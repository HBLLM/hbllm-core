import asyncio
import logging
from pathlib import Path

from hbllm.network.plugin_manager import PluginManager

logging.basicConfig(level=logging.INFO)


async def test_plugins() -> None:
    plugin_dir = Path(__file__).resolve().parent.parent / "plugins"
    print(f"Scanning directory: {plugin_dir}")

    # We do not strictly need a MessageBus just to discover,
    # but to load_all we do. For simple discovery, we can just call discover().
    pm = PluginManager(plugin_dirs=[plugin_dir])

    discovered = pm.discover()
    print(f"Discovered {len(discovered)} plugins.")
    for p in discovered:
        print(f" - {p.name}: {p.path}")

    # To test actual load, we need a mock bus
    from hbllm.network.bus import InProcessBus

    bus = InProcessBus()

    loaded = await pm.load_all(bus=bus)

    success_count = 0
    for p in loaded:
        if p.loaded:
            success_count += 1
            print(f"[OK] Plugin {p.name} loaded successfully.")
        else:
            print(f"[FAIL] Plugin {p.name} failed to load: {p.error}")

    print(f"\nSuccessfully loaded {success_count}/{len(discovered)} plugins.")


async def test_v2_plugin_integration() -> None:
    from unittest.mock import MagicMock, AsyncMock
    from hbllm.network.plugin_manager import PluginManager

    # Create mock brain and plugin manager
    mock_brain = MagicMock()
    mock_bpm = MagicMock()
    mock_bpm.bundles = {}
    mock_bpm.load_bundle = AsyncMock()
    mock_bpm.unload_bundle = AsyncMock()
    mock_brain.plugin_manager = mock_bpm

    import tempfile
    import json
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        plugin_dir = tmp_path / "test_v2_plugin"
        plugin_dir.mkdir()
        manifest_file = plugin_dir / "plugin.json"
        manifest_data = {
            "name": "test_v2_plugin",
            "version": "1.2.3",
            "description": "Test v2 bundle description",
            "manifest_version": 2
        }
        manifest_file.write_text(json.dumps(manifest_data))

        # Discover
        pm = PluginManager(plugin_dirs=[tmp_path], brain=mock_brain)
        discovered = pm.discover()
        
        assert len(discovered) == 1
        info = discovered[0]
        assert info.name == "test_v2_plugin"
        assert info.version == "1.2.3"
        assert info.description == "Test v2 bundle description"
        assert info.loaded is False

        # Load it (toggle from False to True)
        loaded = await pm.toggle_plugin("test_v2_plugin")
        assert loaded is True
        mock_bpm.load_bundle.assert_called_once_with(plugin_dir)

        # Mock that it's now loaded
        mock_bpm.bundles["test_v2_plugin"] = MagicMock()
        pm.discover()
        assert pm.plugins["test_v2_plugin"].loaded is True

        # Unload it (toggle from True to False)
        loaded = await pm.toggle_plugin("test_v2_plugin")
        assert loaded is False
        mock_bpm.unload_bundle.assert_called_once_with("test_v2_plugin")

        # Simulate deleting the plugin directory (uninstall)
        manifest_file.unlink()
        plugin_dir.rmdir()
        
        # Discover again
        discovered_after = pm.discover()
        assert len(discovered_after) == 0
        assert "test_v2_plugin" not in pm.plugins


if __name__ == "__main__":
    asyncio.run(test_plugins())
    asyncio.run(test_v2_plugin_integration())
