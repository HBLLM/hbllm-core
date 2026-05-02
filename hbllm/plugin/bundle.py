"""
Plugin Bundle System — Rich plugin packages for HBLLM.

A Plugin Bundle is a directory that follows a standardized layout, allowing
plugins to ship knowledge documents, pre-built skills, governance policies,
prompt templates, and configuration alongside their code.

Existing code-only plugins continue to work unchanged — the loader auto-detects
manifest version and falls back to legacy behavior.

Bundle layout::

    my-plugin/
    ├── plugin.json          # Manifest v2 (enhanced)
    ├── __init__.py          # Code entry point
    ├── knowledge/           # Auto-ingested into KnowledgeBase
    │   ├── domain-guide.md
    │   └── reference-data.json
    ├── skills/              # Auto-registered in SkillRegistry
    │   └── skills.yaml
    ├── policies/            # Auto-loaded into PolicyEngine
    │   └── policies.yaml
    ├── prompts/             # Reusable prompt templates
    │   └── templates.yaml
    ├── config/              # Plugin-specific configuration
    │   └── defaults.yaml
    └── tests/
        └── test_plugin.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional YAML support
try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_yaml_safe(path: Path) -> Any:
    """Load a YAML file safely, returning None on failure."""
    if not path.exists():
        return None

    if yaml is None:
        logger.warning(
            "PyYAML is not installed. Cannot load %s. Install with: pip install pyyaml",
            path,
        )
        return None

    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        logger.error("Failed to parse YAML %s: %s", path, e)
        return None
    except OSError as e:
        logger.error("Failed to read %s: %s", path, e)
        return None


def _load_json_safe(path: Path) -> Any:
    """Load a JSON file safely, returning None on failure."""
    if not path.exists():
        return None

    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load JSON %s: %s", path, e)
        return None


# ── Plugin Manifest ───────────────────────────────────────────────────────────


@dataclass
class PluginManifest:
    """
    Enhanced plugin manifest (v2).

    Backward-compatible with v1 manifests — missing v2 fields use sensible
    defaults, and the ``manifest_version`` field distinguishes the two.
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    entry_point: str = "__init__.py"
    sentra_version: str = ">=0.1.0"
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    license: str = "MIT"

    # v2 additions
    manifest_version: int = 1
    knowledge_dir: str = "knowledge"
    skills_file: str = "skills/skills.yaml"
    policies_file: str = "policies/policies.yaml"
    prompts_file: str = "prompts/templates.yaml"
    config_file: str = "config/defaults.yaml"
    capabilities: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)

    @property
    def is_v2(self) -> bool:
        """Check if this is a v2 manifest with bundle support."""
        return self.manifest_version >= 2

    @property
    def namespace(self) -> str:
        """Plugin namespace for scoped asset storage."""
        return f"plugin:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "version": self.version,
            "manifest_version": self.manifest_version,
            "description": self.description,
            "author": self.author,
            "entry_point": self.entry_point,
            "sentra_version": self.sentra_version,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "license": self.license,
            "capabilities": self.capabilities,
            "permissions": self.permissions,
            "knowledge_dir": self.knowledge_dir,
            "skills_file": self.skills_file,
            "policies_file": self.policies_file,
            "prompts_file": self.prompts_file,
            "config_file": self.config_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        """
        Create a PluginManifest from a dictionary.

        Handles both v1 and v2 manifests gracefully.

        Raises:
            ValueError: If required fields (name, version) are missing.
        """
        if "name" not in data:
            raise ValueError("Plugin manifest missing required field: 'name'")
        if "version" not in data:
            raise ValueError("Plugin manifest missing required field: 'version'")

        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            entry_point=data.get("entry_point", "__init__.py"),
            sentra_version=data.get("sentra_version", ">=0.1.0"),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            license=data.get("license", "MIT"),
            manifest_version=data.get("manifest_version", 1),
            knowledge_dir=data.get("knowledge_dir", "knowledge"),
            skills_file=data.get("skills_file", "skills/skills.yaml"),
            policies_file=data.get("policies_file", "policies/policies.yaml"),
            prompts_file=data.get("prompts_file", "prompts/templates.yaml"),
            config_file=data.get("config_file", "config/defaults.yaml"),
            capabilities=data.get("capabilities", []),
            permissions=data.get("permissions", []),
        )


# ── Plugin Assets ─────────────────────────────────────────────────────────────


@dataclass
class PluginAssets:
    """
    All discovered assets from a plugin bundle.

    Each field is populated by scanning the corresponding directory/file
    in the plugin bundle. Empty collections mean no assets of that type.
    """

    knowledge_files: list[Path] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    policies: list[dict[str, Any]] = field(default_factory=list)
    prompts: dict[str, str] = field(default_factory=dict)
    config_defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def has_knowledge(self) -> bool:
        return len(self.knowledge_files) > 0

    @property
    def has_skills(self) -> bool:
        return len(self.skills) > 0

    @property
    def has_policies(self) -> bool:
        return len(self.policies) > 0

    @property
    def has_prompts(self) -> bool:
        return len(self.prompts) > 0

    @property
    def has_config(self) -> bool:
        return len(self.config_defaults) > 0

    @property
    def is_empty(self) -> bool:
        """Check if no assets were discovered."""
        return not any(
            [
                self.has_knowledge,
                self.has_skills,
                self.has_policies,
                self.has_prompts,
                self.has_config,
            ]
        )

    def summary(self) -> dict[str, int]:
        """Summarize asset counts."""
        return {
            "knowledge_files": len(self.knowledge_files),
            "skills": len(self.skills),
            "policies": len(self.policies),
            "prompts": len(self.prompts),
            "config_keys": len(self.config_defaults),
        }


# ── Plugin Bundle ─────────────────────────────────────────────────────────────


KNOWLEDGE_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}


class PluginBundle:
    """
    Loads and validates a plugin bundle from a directory.

    Discovers the manifest, code entry point, and all optional asset
    directories (knowledge, skills, policies, prompts, config).

    Usage::

        bundle = PluginBundle(Path("plugins/sentinel-shield"))
        print(bundle.manifest.name)       # "sentinel-shield"
        print(bundle.assets.has_skills)   # True
        print(bundle.assets.summary())    # {"knowledge_files": 2, ...}
    """

    def __init__(self, plugin_path: Path | str) -> None:
        """
        Initialize a PluginBundle from a directory path.

        Args:
            plugin_path: Path to the plugin directory.

        Raises:
            FileNotFoundError: If the plugin directory doesn't exist.
            ValueError: If plugin.json is missing or invalid.
        """
        self.path = Path(plugin_path).resolve()

        if not self.path.is_dir():
            raise FileNotFoundError(f"Plugin directory not found: {self.path}")

        self.manifest = self._load_manifest()
        self.assets = self._discover_assets()

        logger.info(
            "Loaded plugin bundle: %s v%s (manifest v%d) — %s",
            self.manifest.name,
            self.manifest.version,
            self.manifest.manifest_version,
            self.assets.summary(),
        )

    # ── Manifest Loading ─────────────────────────────────────────────

    def _load_manifest(self) -> PluginManifest:
        """
        Load and validate the plugin manifest.

        Raises:
            ValueError: If plugin.json is missing or contains invalid data.
        """
        manifest_path = self.path / "plugin.json"

        if not manifest_path.exists():
            # Fallback: try to construct a minimal manifest from the directory name
            logger.warning(
                "No plugin.json found in %s — creating minimal manifest",
                self.path,
            )
            return PluginManifest(
                name=self.path.name,
                version="0.0.0",
                description=f"Plugin from {self.path.name} (no manifest)",
            )

        data = _load_json_safe(manifest_path)
        if data is None:
            raise ValueError(f"Failed to parse plugin.json in {self.path}")

        try:
            return PluginManifest.from_dict(data)
        except ValueError as e:
            raise ValueError(f"Invalid plugin.json in {self.path}: {e}") from e

    # ── Asset Discovery ──────────────────────────────────────────────

    def _discover_assets(self) -> PluginAssets:
        """
        Discover all optional assets from the plugin bundle.

        For v1 manifests, returns empty assets (code-only plugin).
        For v2 manifests, scans all configured asset directories.
        """
        if not self.manifest.is_v2:
            return PluginAssets()

        return PluginAssets(
            knowledge_files=self._discover_knowledge(),
            skills=self._discover_skills(),
            policies=self._discover_policies(),
            prompts=self._discover_prompts(),
            config_defaults=self._discover_config(),
        )

    def _discover_knowledge(self) -> list[Path]:
        """Discover knowledge documents from the knowledge directory."""
        knowledge_dir = self.path / self.manifest.knowledge_dir

        if not knowledge_dir.is_dir():
            return []

        files: list[Path] = []
        try:
            for item in sorted(knowledge_dir.iterdir()):
                if item.is_file() and item.suffix.lower() in KNOWLEDGE_EXTENSIONS:
                    files.append(item)
        except OSError as e:
            logger.error(
                "Failed to scan knowledge directory %s: %s",
                knowledge_dir,
                e,
            )

        if files:
            logger.debug(
                "[%s] Discovered %d knowledge files",
                self.manifest.name,
                len(files),
            )

        return files

    def _discover_skills(self) -> list[dict[str, Any]]:
        """Discover pre-built skill definitions from skills YAML."""
        skills_path = self.path / self.manifest.skills_file

        data = _load_yaml_safe(skills_path)
        if data is None:
            return []

        skills = data.get("skills", [])
        if not isinstance(skills, list):
            logger.warning(
                "[%s] skills.yaml 'skills' key is not a list — skipping",
                self.manifest.name,
            )
            return []

        # Validate and tag each skill with the plugin source
        valid_skills: list[dict[str, Any]] = []
        for i, skill in enumerate(skills):
            if not isinstance(skill, dict):
                logger.warning(
                    "[%s] Skipping invalid skill entry at index %d",
                    self.manifest.name,
                    i,
                )
                continue

            if "name" not in skill:
                logger.warning(
                    "[%s] Skipping skill at index %d: missing 'name'",
                    self.manifest.name,
                    i,
                )
                continue

            # Tag with plugin source for ownership tracking
            skill["source"] = self.manifest.namespace
            valid_skills.append(skill)

        if valid_skills:
            logger.debug(
                "[%s] Discovered %d pre-built skills",
                self.manifest.name,
                len(valid_skills),
            )

        return valid_skills

    def _discover_policies(self) -> list[dict[str, Any]]:
        """Discover governance policies from policies YAML."""
        policies_path = self.path / self.manifest.policies_file

        data = _load_yaml_safe(policies_path)
        if data is None:
            return []

        policies = data.get("policies", [])
        if not isinstance(policies, list):
            logger.warning(
                "[%s] policies.yaml 'policies' key is not a list — skipping",
                self.manifest.name,
            )
            return []

        # Validate and tag each policy
        valid_policies: list[dict[str, Any]] = []
        for i, policy in enumerate(policies):
            if not isinstance(policy, dict):
                logger.warning(
                    "[%s] Skipping invalid policy entry at index %d",
                    self.manifest.name,
                    i,
                )
                continue

            if "name" not in policy:
                logger.warning(
                    "[%s] Skipping policy at index %d: missing 'name'",
                    self.manifest.name,
                    i,
                )
                continue

            # Tag with plugin source, default to lower priority than user policies
            policy.setdefault("source", self.manifest.namespace)
            policy.setdefault("priority", -10)  # Below user-defined (default 0)
            valid_policies.append(policy)

        if valid_policies:
            logger.debug(
                "[%s] Discovered %d governance policies",
                self.manifest.name,
                len(valid_policies),
            )

        return valid_policies

    def _discover_prompts(self) -> dict[str, str]:
        """Discover named prompt templates from prompts YAML."""
        prompts_path = self.path / self.manifest.prompts_file

        data = _load_yaml_safe(prompts_path)
        if data is None:
            return {}

        prompts = data.get("prompts", {})
        if not isinstance(prompts, dict):
            logger.warning(
                "[%s] prompts.yaml 'prompts' key is not a dict — skipping",
                self.manifest.name,
            )
            return {}

        # Namespace the prompt keys to avoid collisions
        namespaced: dict[str, str] = {}
        for key, template in prompts.items():
            if not isinstance(template, str):
                logger.warning(
                    "[%s] Skipping non-string prompt '%s'",
                    self.manifest.name,
                    key,
                )
                continue

            namespaced_key = f"{self.manifest.name}:{key}"
            namespaced[namespaced_key] = template

        if namespaced:
            logger.debug(
                "[%s] Discovered %d prompt templates",
                self.manifest.name,
                len(namespaced),
            )

        return namespaced

    def _discover_config(self) -> dict[str, Any]:
        """Discover default configuration from config YAML."""
        config_path = self.path / self.manifest.config_file

        data = _load_yaml_safe(config_path)
        if data is None:
            return {}

        if not isinstance(data, dict):
            logger.warning(
                "[%s] config/defaults.yaml root is not a dict — skipping",
                self.manifest.name,
            )
            return {}

        if data:
            logger.debug(
                "[%s] Discovered %d config defaults",
                self.manifest.name,
                len(data),
            )

        return data

    # ── Utility ──────────────────────────────────────────────────────

    @property
    def entry_point_path(self) -> Path:
        """Absolute path to the code entry point."""
        return self.path / self.manifest.entry_point

    @property
    def has_code(self) -> bool:
        """Check if the plugin has a code entry point."""
        return self.entry_point_path.exists()

    def __repr__(self) -> str:
        return (
            f"<PluginBundle name={self.manifest.name!r} "
            f"v{self.manifest.version} "
            f"assets={self.assets.summary()}>"
        )
