"""
Tests for core PluginManager — standalone plugin lifecycle without Sentra.

Verifies that hbllm core users can load/unload bundles, discover plugins,
and use the background watcher independently.
"""

import asyncio
import json
from pathlib import Path

import pytest

from hbllm.plugin.manager import LoadedBundle, PluginManager, PromptStore


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _create_bundle(tmp_path: Path, name: str = "test-core-plugin") -> Path:
    """Create a minimal v2 plugin bundle."""
    plugin_dir = tmp_path / name
    plugin_dir.mkdir(parents=True)

    (plugin_dir / "plugin.json").write_text(json.dumps({
        "name": name,
        "version": "1.0.0",
        "manifest_version": 2,
        "capabilities": ["testing"],
    }))
    (plugin_dir / "__init__.py").write_text('"""Test."""\n')

    # Skills
    skills_dir = plugin_dir / "skills"
    skills_dir.mkdir()
    (skills_dir / "skills.yaml").write_text("""
skills:
  - name: "Core Skill"
    description: "A test skill"
    steps: ["step1", "step2"]
""")

    # Policies
    policies_dir = plugin_dir / "policies"
    policies_dir.mkdir()
    (policies_dir / "policies.yaml").write_text("""
policies:
  - name: "core_test_policy"
    type: "deny"
    action: "warn"
    pattern: "test"
    description: "Test policy"
""")

    # Prompts
    prompts_dir = plugin_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "templates.yaml").write_text("""
prompts:
  greeting: "Hello {name}!"
""")

    return plugin_dir


# ── Standalone Core Usage ────────────────────────────────────────────────────


class TestCorePluginManagerStandalone:
    """Core PluginManager works without Sentra."""

    @pytest.mark.asyncio
    async def test_load_bundle_standalone(self, tmp_path):
        """Standalone load without any brain subsystems."""
        bundle_dir = _create_bundle(tmp_path)
        manager = PluginManager(plugin_dirs=[])

        loaded = await manager.load_bundle(bundle_dir)
        assert loaded.name == "test-core-plugin"
        assert manager.bundle_count == 1

    @pytest.mark.asyncio
    async def test_load_with_policy_engine(self, tmp_path):
        """Load with a PolicyEngine for policy ingestion."""
        from hbllm.brain.policy_engine import PolicyEngine

        bundle_dir = _create_bundle(tmp_path)
        engine = PolicyEngine()
        manager = PluginManager(plugin_dirs=[], policy_engine=engine)

        await manager.load_bundle(bundle_dir)
        assert engine.policy_count == 1

    @pytest.mark.asyncio
    async def test_unload_soft_deactivation(self, tmp_path):
        """Unload deactivates policies, archives prompts."""
        from hbllm.brain.policy_engine import PolicyEngine

        bundle_dir = _create_bundle(tmp_path)
        engine = PolicyEngine()
        manager = PluginManager(plugin_dirs=[], policy_engine=engine)

        await manager.load_bundle(bundle_dir)
        assert engine.policy_count == 1
        assert manager.prompt_store.count == 1

        await manager.unload_bundle("test-core-plugin")
        # Policy deactivated, not deleted
        assert engine.policy_count == 1
        policies = engine.list_policies()
        assert all(not p["enabled"] for p in policies)
        # Prompt archived
        assert manager.prompt_store.count == 0
        assert manager.prompt_store.get("archived:test-core-plugin:greeting") is not None


class TestCoreDiscovery:
    """Plugin discovery from configured directories."""

    @pytest.mark.asyncio
    async def test_discover_from_directory(self, tmp_path):
        """discover_plugins() finds and loads v2 bundles."""
        _create_bundle(tmp_path, "plugin-a")
        _create_bundle(tmp_path, "plugin-b")

        manager = PluginManager(plugin_dirs=[tmp_path])
        discovered = await manager.discover_plugins()
        assert len(discovered) == 2
        assert manager.bundle_count == 2

    @pytest.mark.asyncio
    async def test_discover_skips_already_loaded(self, tmp_path):
        """Calling discover_plugins() twice doesn't double-load."""
        _create_bundle(tmp_path, "plugin-a")

        manager = PluginManager(plugin_dirs=[tmp_path])
        first = await manager.discover_plugins()
        second = await manager.discover_plugins()
        assert len(first) == 1
        assert len(second) == 0  # Already loaded
        assert manager.bundle_count == 1

    @pytest.mark.asyncio
    async def test_discover_ignores_hidden_dirs(self, tmp_path):
        """Hidden directories and __pycache__ are skipped."""
        _create_bundle(tmp_path, "plugin-a")
        (tmp_path / ".hidden-dir").mkdir()
        (tmp_path / "__pycache__").mkdir()

        manager = PluginManager(plugin_dirs=[tmp_path])
        discovered = await manager.discover_plugins()
        assert len(discovered) == 1


class TestCoreWatcher:
    """Background directory watcher for hot-loading."""

    @pytest.mark.asyncio
    async def test_watch_and_stop(self, tmp_path):
        """Watcher starts and stops cleanly."""
        manager = PluginManager(plugin_dirs=[tmp_path])
        await manager.watch_directories(interval=1)
        assert manager._watch_task is not None

        await manager.stop_watching()
        assert manager._watch_task is None

    @pytest.mark.asyncio
    async def test_watcher_detects_new_plugin(self, tmp_path):
        """Watcher auto-loads plugins dropped into directory."""
        manager = PluginManager(plugin_dirs=[tmp_path])
        assert manager.bundle_count == 0

        # Start watcher with fast interval
        await manager.watch_directories(interval=0.5)

        # Drop a new plugin while watcher is running
        await asyncio.sleep(0.1)
        _create_bundle(tmp_path, "hot-plugin")

        # Wait for watcher to pick it up
        await asyncio.sleep(1.5)

        assert manager.bundle_count == 1
        assert "hot-plugin" in manager.bundles

        await manager.stop_watching()


class TestCoreStats:
    """Plugin manager stats."""

    @pytest.mark.asyncio
    async def test_stats(self, tmp_path):
        bundle_dir = _create_bundle(tmp_path)
        manager = PluginManager(plugin_dirs=[tmp_path])
        await manager.load_bundle(bundle_dir)

        stats = manager.stats()
        assert stats["bundles_loaded"] == 1
        assert stats["prompts"] == 1
        assert stats["watcher_active"] is False


class TestAddPluginDir:
    """Dynamic plugin directory management."""

    def test_add_plugin_dir(self, tmp_path):
        manager = PluginManager(plugin_dirs=[])
        manager.add_plugin_dir(tmp_path)
        assert tmp_path.resolve() in [d.resolve() for d in manager.plugin_dirs]
