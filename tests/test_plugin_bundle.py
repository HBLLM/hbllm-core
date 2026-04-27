"""
Tests for the Plugin Bundle System (hbllm.plugin.bundle).
"""

import json
import os
from pathlib import Path

import pytest

from hbllm.plugin.bundle import (
    KNOWLEDGE_EXTENSIONS,
    PluginAssets,
    PluginBundle,
    PluginManifest,
    _load_json_safe,
    _load_yaml_safe,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _create_plugin_dir(tmp_path: Path, manifest: dict, v2_assets: bool = False) -> Path:
    """Create a minimal plugin directory for testing."""
    plugin_dir = tmp_path / manifest.get("name", "test-plugin")
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest))

    # Write entry point
    (plugin_dir / "__init__.py").write_text('"""Test plugin."""\n')

    if v2_assets:
        # Knowledge
        knowledge_dir = plugin_dir / "knowledge"
        knowledge_dir.mkdir()
        (knowledge_dir / "guide.md").write_text("# Test Guide\n\nSome knowledge content.")
        (knowledge_dir / "data.json").write_text('{"key": "value"}')
        (knowledge_dir / "notes.txt").write_text("Plain text notes.")
        (knowledge_dir / "ignored.exe").write_text("should be skipped")

        # Skills (YAML)
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        skills_yaml = """
skills:
  - name: "Test Skill One"
    category: "testing"
    description: "A test skill"
    steps:
      - "Step 1"
      - "Step 2"
    tools_used: ["tool_a"]
  - name: "Test Skill Two"
    category: "testing"
    description: "Another test skill"
    steps:
      - "Do something"
    tools_used: []
"""
        (skills_dir / "skills.yaml").write_text(skills_yaml)

        # Policies (YAML)
        policies_dir = plugin_dir / "policies"
        policies_dir.mkdir()
        policies_yaml = """
policies:
  - name: "test_deny_rule"
    type: "deny"
    action: "warn"
    pattern: "bad_word"
    description: "Test deny policy"
    severity: "low"
  - name: "test_transform_rule"
    type: "transform"
    action: "append"
    content: "Disclaimer text"
    description: "Test transform policy"
"""
        (policies_dir / "policies.yaml").write_text(policies_yaml)

        # Prompts (YAML)
        prompts_dir = plugin_dir / "prompts"
        prompts_dir.mkdir()
        prompts_yaml = """
prompts:
  greeting: "Hello {name}, welcome to {plugin}!"
  farewell: "Goodbye {name}."
"""
        (prompts_dir / "templates.yaml").write_text(prompts_yaml)

        # Config (YAML)
        config_dir = plugin_dir / "config"
        config_dir.mkdir()
        config_yaml = """
enabled: true
scan_interval: 30
max_retries: 3
"""
        (config_dir / "defaults.yaml").write_text(config_yaml)

    return plugin_dir


# ── PluginManifest ────────────────────────────────────────────────────────────


class TestPluginManifest:
    def test_from_dict_v1(self):
        """v1 manifest loads with default v2 fields."""
        data = {
            "name": "my-plugin",
            "version": "1.0.0",
            "description": "Test plugin",
        }
        m = PluginManifest.from_dict(data)
        assert m.name == "my-plugin"
        assert m.version == "1.0.0"
        assert m.manifest_version == 1
        assert not m.is_v2
        assert m.namespace == "plugin:my-plugin"

    def test_from_dict_v2(self):
        """v2 manifest loads all enhanced fields."""
        data = {
            "name": "sentinel-shield",
            "version": "1.1.0",
            "manifest_version": 2,
            "capabilities": ["threat_detection"],
            "permissions": ["subprocess"],
            "knowledge_dir": "knowledge",
            "skills_file": "skills/skills.yaml",
        }
        m = PluginManifest.from_dict(data)
        assert m.is_v2
        assert m.capabilities == ["threat_detection"]
        assert m.permissions == ["subprocess"]
        assert m.namespace == "plugin:sentinel-shield"

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            PluginManifest.from_dict({"version": "1.0.0"})

    def test_missing_version_raises(self):
        with pytest.raises(ValueError, match="version"):
            PluginManifest.from_dict({"name": "test"})

    def test_to_dict_roundtrip(self):
        m = PluginManifest(name="test", version="1.0.0", manifest_version=2)
        d = m.to_dict()
        m2 = PluginManifest.from_dict(d)
        assert m.name == m2.name
        assert m.version == m2.version
        assert m.manifest_version == m2.manifest_version

    def test_defaults(self):
        m = PluginManifest(name="test", version="0.1.0")
        assert m.entry_point == "__init__.py"
        assert m.dependencies == []
        assert m.tags == []
        assert m.license == "MIT"
        assert m.knowledge_dir == "knowledge"
        assert m.skills_file == "skills/skills.yaml"

    def test_unknown_fields_ignored(self):
        data = {
            "name": "test",
            "version": "1.0.0",
            "unknown_field": "should be ignored",
            "another_thing": 42,
        }
        m = PluginManifest.from_dict(data)
        assert m.name == "test"
        assert not hasattr(m, "unknown_field")


# ── PluginAssets ──────────────────────────────────────────────────────────────


class TestPluginAssets:
    def test_empty_assets(self):
        a = PluginAssets()
        assert a.is_empty
        assert not a.has_knowledge
        assert not a.has_skills
        assert not a.has_policies
        assert not a.has_prompts
        assert not a.has_config

    def test_summary(self):
        a = PluginAssets(
            knowledge_files=[Path("/a.md"), Path("/b.txt")],
            skills=[{"name": "s1"}],
            policies=[{"name": "p1"}, {"name": "p2"}],
            prompts={"key": "value"},
            config_defaults={"enabled": True},
        )
        s = a.summary()
        assert s["knowledge_files"] == 2
        assert s["skills"] == 1
        assert s["policies"] == 2
        assert s["prompts"] == 1
        assert s["config_keys"] == 1

    def test_not_empty_with_knowledge(self):
        a = PluginAssets(knowledge_files=[Path("/a.md")])
        assert not a.is_empty
        assert a.has_knowledge


# ── PluginBundle ──────────────────────────────────────────────────────────────


class TestPluginBundle:
    def test_v1_code_only(self, tmp_path):
        """v1 manifest loads as code-only — no assets discovered."""
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {
                "name": "basic-plugin",
                "version": "1.0.0",
            },
        )
        bundle = PluginBundle(plugin_dir)
        assert bundle.manifest.name == "basic-plugin"
        assert bundle.manifest.manifest_version == 1
        assert not bundle.manifest.is_v2
        assert bundle.assets.is_empty
        assert bundle.has_code

    def test_v2_full_bundle(self, tmp_path):
        """v2 manifest discovers all asset types."""
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {
                "name": "rich-plugin",
                "version": "2.0.0",
                "manifest_version": 2,
            },
            v2_assets=True,
        )
        bundle = PluginBundle(plugin_dir)
        assert bundle.manifest.is_v2
        assert bundle.manifest.name == "rich-plugin"

        # Knowledge files (3 valid, .exe skipped)
        assert len(bundle.assets.knowledge_files) == 3
        names = [f.name for f in bundle.assets.knowledge_files]
        assert "guide.md" in names
        assert "data.json" in names
        assert "notes.txt" in names
        assert "ignored.exe" not in names

        # Skills
        assert len(bundle.assets.skills) == 2
        assert bundle.assets.skills[0]["name"] == "Test Skill One"
        assert bundle.assets.skills[0]["source"] == "plugin:rich-plugin"
        assert bundle.assets.skills[1]["name"] == "Test Skill Two"

        # Policies
        assert len(bundle.assets.policies) == 2
        assert bundle.assets.policies[0]["name"] == "test_deny_rule"
        assert bundle.assets.policies[0]["source"] == "plugin:rich-plugin"
        assert bundle.assets.policies[0]["priority"] == -10  # Default for plugins

        # Prompts (namespaced)
        assert len(bundle.assets.prompts) == 2
        assert "rich-plugin:greeting" in bundle.assets.prompts
        assert "rich-plugin:farewell" in bundle.assets.prompts
        assert "{name}" in bundle.assets.prompts["rich-plugin:greeting"]

        # Config
        assert bundle.assets.config_defaults["enabled"] is True
        assert bundle.assets.config_defaults["scan_interval"] == 30

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            PluginBundle(Path("/nonexistent/plugin"))

    def test_no_manifest_fallback(self, tmp_path):
        """Plugin directory without plugin.json gets a minimal manifest."""
        plugin_dir = tmp_path / "no-manifest"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("# plugin\n")

        bundle = PluginBundle(plugin_dir)
        assert bundle.manifest.name == "no-manifest"
        assert bundle.manifest.version == "0.0.0"

    def test_invalid_manifest_json(self, tmp_path):
        """Malformed JSON raises ValueError."""
        plugin_dir = tmp_path / "bad-json"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text("{invalid json")

        with pytest.raises(ValueError, match="Failed to parse"):
            PluginBundle(plugin_dir)

    def test_manifest_missing_name(self, tmp_path):
        """Manifest with missing 'name' field raises ValueError."""
        plugin_dir = tmp_path / "no-name"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text('{"version": "1.0.0"}')

        with pytest.raises(ValueError, match="name"):
            PluginBundle(plugin_dir)

    def test_empty_asset_dirs(self, tmp_path):
        """v2 bundle with empty asset directories produces empty assets."""
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {"name": "empty-assets", "version": "1.0.0", "manifest_version": 2},
        )
        # Create empty dirs
        (plugin_dir / "knowledge").mkdir()
        (plugin_dir / "skills").mkdir()

        bundle = PluginBundle(plugin_dir)
        assert bundle.assets.has_knowledge is False
        assert bundle.assets.has_skills is False

    def test_missing_optional_dirs(self, tmp_path):
        """v2 bundle works fine when optional dirs don't exist."""
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {"name": "minimal-v2", "version": "1.0.0", "manifest_version": 2},
        )
        bundle = PluginBundle(plugin_dir)
        assert bundle.assets.is_empty

    def test_malformed_skills_yaml(self, tmp_path):
        """Malformed skills YAML is handled gracefully."""
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {"name": "bad-skills", "version": "1.0.0", "manifest_version": 2},
        )
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "skills.yaml").write_text("skills: not_a_list")

        bundle = PluginBundle(plugin_dir)
        assert bundle.assets.has_skills is False

    def test_skills_missing_name_skipped(self, tmp_path):
        """Skills without a 'name' field are skipped."""
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {"name": "nameless-skill", "version": "1.0.0", "manifest_version": 2},
        )
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "skills.yaml").write_text(
            'skills:\n  - description: "no name"\n    category: "test"\n'
        )

        bundle = PluginBundle(plugin_dir)
        assert len(bundle.assets.skills) == 0

    def test_policies_tagged_with_namespace(self, tmp_path):
        """Policies are tagged with the plugin namespace."""
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {"name": "tagged-policies", "version": "1.0.0", "manifest_version": 2},
            v2_assets=True,
        )
        bundle = PluginBundle(plugin_dir)
        for policy in bundle.assets.policies:
            assert policy["source"] == "plugin:tagged-policies"

    def test_prompts_namespaced(self, tmp_path):
        """Prompt keys are namespaced by plugin name."""
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {"name": "my-prompts", "version": "1.0.0", "manifest_version": 2},
            v2_assets=True,
        )
        bundle = PluginBundle(plugin_dir)
        for key in bundle.assets.prompts:
            assert key.startswith("my-prompts:")

    def test_entry_point_path(self, tmp_path):
        plugin_dir = _create_plugin_dir(
            tmp_path,
            {"name": "ep-test", "version": "1.0.0", "entry_point": "__init__.py"},
        )
        bundle = PluginBundle(plugin_dir)
        assert bundle.entry_point_path == plugin_dir / "__init__.py"
        assert bundle.has_code

    def test_repr(self, tmp_path):
        plugin_dir = _create_plugin_dir(tmp_path, {"name": "repr-test", "version": "1.0.0"})
        bundle = PluginBundle(plugin_dir)
        r = repr(bundle)
        assert "repr-test" in r
        assert "v1.0.0" in r


# ── Helper Functions ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_load_json_safe_valid(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        data = _load_json_safe(f)
        assert data == {"key": "value"}

    def test_load_json_safe_nonexistent(self):
        assert _load_json_safe(Path("/nonexistent/file.json")) is None

    def test_load_json_safe_invalid(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{invalid}")
        assert _load_json_safe(f) is None

    def test_load_yaml_safe_nonexistent(self):
        assert _load_yaml_safe(Path("/nonexistent/file.yaml")) is None


# ── Real Plugin Integration ──────────────────────────────────────────────────


class TestSentinelShieldBundle:
    """Test loading the actual sentinel-shield plugin as a bundle."""

    SHIELD_PATH = (
        Path(__file__).parent.parent.parent / "sentra-plugins" / "plugins" / "sentinel-shield"
    )

    @pytest.mark.skipif(
        not (
            Path(__file__).parent.parent.parent
            / "sentra-plugins"
            / "plugins"
            / "sentinel-shield"
            / "plugin.json"
        ).exists(),
        reason="sentinel-shield plugin not available",
    )
    def test_load_real_shield_bundle(self):
        bundle = PluginBundle(self.SHIELD_PATH)
        assert bundle.manifest.name == "sentinel-shield"
        assert bundle.manifest.is_v2
        assert bundle.manifest.manifest_version == 2

        # Should have knowledge
        assert bundle.assets.has_knowledge
        assert any("threat-guide" in f.name for f in bundle.assets.knowledge_files)

        # Should have skills
        assert bundle.assets.has_skills
        assert len(bundle.assets.skills) >= 3

        # Should have policies
        assert bundle.assets.has_policies
        assert len(bundle.assets.policies) >= 2

        # Should have prompts
        assert bundle.assets.has_prompts
        assert any("threat_explanation" in k for k in bundle.assets.prompts)

        # All skills should be namespaced
        for skill in bundle.assets.skills:
            assert skill["source"] == "plugin:sentinel-shield"

        # All policies should be namespaced
        for policy in bundle.assets.policies:
            assert policy["source"] == "plugin:sentinel-shield"
