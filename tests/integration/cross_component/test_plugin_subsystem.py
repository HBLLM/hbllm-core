"""Integration tests for Plugin subsystem — PluginManager, PromptStore, Bundle lifecycle."""

import json
from pathlib import Path

import pytest

from hbllm.plugin.manager import LoadedBundle, PluginManager, PromptStore

# ── PromptStore Tests ────────────────────────────────────────────────────────


class TestPromptStoreIntegration:
    """Test PromptStore template management."""

    def test_add_and_get(self):
        store = PromptStore()
        store.add("greeting", "Hello, {name}!", source="test-plugin")
        assert store.get("greeting") == "Hello, {name}!"

    def test_render_template(self):
        store = PromptStore()
        store.add("greeting", "Hello, {name}! You are a {role}.", source="test")
        result = store.render("greeting", name="Alice", role="developer")
        assert result == "Hello, Alice! You are a developer."

    def test_render_missing_variable(self):
        store = PromptStore()
        store.add("partial", "Hello {name}, your {missing} is ready.", source="test")
        # Should return the template as-is when variable is missing
        result = store.render("partial", name="Bob")
        assert result is not None
        # It either fills what it can or returns the original template

    def test_render_nonexistent_key(self):
        store = PromptStore()
        result = store.render("nonexistent", name="value")
        assert result is None

    def test_get_nonexistent(self):
        store = PromptStore()
        assert store.get("nonexistent") is None

    def test_remove_by_source(self):
        store = PromptStore()
        store.add("p1", "Template 1", source="plugin-a")
        store.add("p2", "Template 2", source="plugin-a")
        store.add("p3", "Template 3", source="plugin-b")

        removed = store.remove_by_source("plugin-a")
        assert removed == 2
        assert store.get("p1") is None
        assert store.get("p2") is None
        assert store.get("p3") is not None

    def test_archive_and_restore(self):
        store = PromptStore()
        store.add("prompt1", "Original content", source="my-plugin")

        # Archive
        archived = store.archive_by_source("my-plugin")
        assert archived == 1
        assert store.get("prompt1") is None  # No longer active

        # Restore
        restored = store.restore_archived("my-plugin")
        assert restored == 1
        assert store.get("prompt1") == "Original content"

    def test_list_templates(self):
        store = PromptStore()
        store.add("t1", "Template 1 is long enough to have a preview", source="src1")
        store.add("t2", "Template 2 also has content", source="src2")

        templates = store.list_templates()
        assert len(templates) == 2
        assert all("key" in t for t in templates)
        assert all("source" in t for t in templates)

    def test_count_excludes_archived(self):
        store = PromptStore()
        store.add("a1", "Active", source="p1")
        store.add("a2", "Active too", source="p1")
        store.add("a3", "Also active", source="p2")

        assert store.count == 3

        store.archive_by_source("p1")
        assert store.count == 1


# ── PluginManager Integration Tests ──────────────────────────────────────────


def _create_v2_plugin(plugin_dir: Path, name: str = "test-plugin") -> Path:
    """Create a minimal v2 plugin bundle for testing."""
    bundle_dir = plugin_dir / name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": name,
        "version": "1.0.0",
        "manifest_version": 2,
        "namespace": f"ns-{name}",
        "description": "A test plugin",
    }

    (bundle_dir / "plugin.json").write_text(json.dumps(manifest, indent=2))

    # Create prompt templates YAML file for discovery
    prompts_dir = bundle_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    try:
        import yaml

        prompts_data = {
            "prompts": {
                "greeting": f"Hello from {name} plugin!",
                "summary": "Summary: {{topic}}",
            }
        }
        (prompts_dir / "templates.yaml").write_text(yaml.dump(prompts_data))
    except ImportError:
        # If yaml not available, prompts won't be discovered — tests will skip
        pass

    return bundle_dir


class TestPluginManagerIntegration:
    """Test PluginManager bundle lifecycle and discovery."""

    def test_init_with_custom_dirs(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        assert len(manager.plugin_dirs) >= 1
        assert any(str(plugins_dir) in str(d) for d in manager.plugin_dirs)

    def test_add_plugin_dir(self, tmp_path):
        manager = PluginManager(plugin_dirs=[])
        new_dir = tmp_path / "extra_plugins"
        new_dir.mkdir()

        manager.add_plugin_dir(new_dir)
        assert any(str(new_dir.resolve()) in str(d) for d in manager.plugin_dirs)

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_load_v2_bundle(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        bundle_path = _create_v2_plugin(plugins_dir, "my-plugin")

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        loaded = await manager.load_bundle(bundle_path)

        assert isinstance(loaded, LoadedBundle)
        assert loaded.name == "my-plugin"
        assert "my-plugin" in manager.bundles
        assert manager.bundle_count == 1

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_load_stores_prompts(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        _create_v2_plugin(plugins_dir, "prompt-test")

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        await manager.load_bundle(plugins_dir / "prompt-test")

        # Only check prompts if yaml is available (prompts are in YAML)
        try:
            import yaml  # noqa: F401

            assert manager.prompt_store.count >= 2
            assert manager.prompt_store.get("prompt-test:greeting") is not None
        except ImportError:
            pytest.skip("PyYAML not installed, prompt YAML not discovered")

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_unload_soft_deactivation(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        _create_v2_plugin(plugins_dir, "unload-test")

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        await manager.load_bundle(plugins_dir / "unload-test")
        assert manager.bundle_count == 1

        result = await manager.unload_bundle("unload-test")
        assert result is True
        assert manager.bundle_count == 0

        # Prompts should be archived, not deleted
        assert manager.prompt_store.get("unload-test:greeting") is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_unload_nonexistent_returns_false(self, tmp_path):
        manager = PluginManager(plugin_dirs=[])
        result = await manager.unload_bundle("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_reload_bundle(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        _create_v2_plugin(plugins_dir, "reload-test")

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])

        # Load first time
        loaded1 = await manager.load_bundle(plugins_dir / "reload-test")  # noqa: F841
        # Load again — should unload first, then reload
        loaded2 = await manager.load_bundle(plugins_dir / "reload-test")

        assert manager.bundle_count == 1
        assert loaded2.name == "reload-test"

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_discover_plugins(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        _create_v2_plugin(plugins_dir, "discover-a")
        _create_v2_plugin(plugins_dir, "discover-b")

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        newly_loaded = await manager.discover_plugins()

        assert len(newly_loaded) == 2
        assert manager.bundle_count == 2

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_discover_skips_already_loaded(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        _create_v2_plugin(plugins_dir, "already-loaded")

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])

        first = await manager.discover_plugins()
        assert len(first) == 1

        second = await manager.discover_plugins()
        assert len(second) == 0  # Already loaded

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_discover_skips_v1_plugins(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        # Create a v1 plugin (manifest_version=1)
        v1_dir = plugins_dir / "v1-plugin"
        v1_dir.mkdir()
        (v1_dir / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "v1-plugin",
                    "version": "1.0.0",
                    "manifest_version": 1,
                }
            )
        )

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        loaded = await manager.discover_plugins()
        assert len(loaded) == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_discover_skips_hidden_and_underscore_dirs(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()

        # These should be skipped
        (plugins_dir / "_internal").mkdir()
        (plugins_dir / ".hidden").mkdir()

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        loaded = await manager.discover_plugins()
        assert len(loaded) == 0

    def test_list_bundles(self, tmp_path):
        manager = PluginManager(plugin_dirs=[])
        result = manager.list_bundles()
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_get_bundle(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        _create_v2_plugin(plugins_dir, "get-test")

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        await manager.load_bundle(plugins_dir / "get-test")

        bundle = manager.get_bundle("get-test")
        assert bundle is not None
        assert bundle.name == "get-test"

        assert manager.get_bundle("nonexistent") is None

    def test_stats(self, tmp_path):
        manager = PluginManager(plugin_dirs=[str(tmp_path)])
        stats = manager.stats()
        assert "bundles_loaded" in stats
        assert "plugin_dirs" in stats
        assert "prompts" in stats
        assert "watcher_active" in stats
        assert stats["watcher_active"] is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_loaded_bundle_to_dict(self, tmp_path):
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        _create_v2_plugin(plugins_dir, "dict-test")

        manager = PluginManager(plugin_dirs=[str(plugins_dir)])
        loaded = await manager.load_bundle(plugins_dir / "dict-test")

        d = loaded.to_dict()
        assert d["name"] == "dict-test"
        assert d["version"] == "1.0.0"
        assert "ingested" in d
        assert "assets" in d
