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


if __name__ == "__main__":
    asyncio.run(test_plugins())
