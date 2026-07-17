"""
HBPKG Manifest — Schema for the .hbpkg agent package format.

An ``.hbpkg`` archive is a distributable cognitive agent package containing
everything needed to install new capabilities into HBLLM:

Layout::

    my-agent.hbpkg/
    ├── hbpkg.yaml            ← Package manifest (this schema)
    ├── skills/               ← SKILL.md files auto-registered
    │   ├── deploy.md
    │   └── debug.md
    ├── knowledge/            ← Documents auto-ingested into KnowledgeBase
    │   ├── api-reference.md
    │   └── best-practices.json
    ├── actions/              ← Custom action implementations (Python)
    │   └── custom_tool.py
    ├── models/               ← LoRA adapters or model configs
    │   └── adapter_config.json
    ├── prompts/              ← Reusable prompt templates
    │   └── templates.yaml
    ├── policies/             ← Governance rules (value alignment)
    │   └── safety.yaml
    └── config/               ← Package-specific config overrides
        └── defaults.yaml

The manifest schema extends the existing PluginManifest (v2) with:
  - Explicit capability declarations with semantic metadata
  - Permission scope requests (mobile-style sandbox model)
  - Dependency resolution (other .hbpkg packages)
  - Profile compatibility (which BrainProfiles this package supports)
  - Lifecycle hooks (on_install, on_activate, on_deactivate)

Usage::

    from hbllm.hbpkg.manifest import HBPkgManifest

    manifest = HBPkgManifest.from_yaml(Path("hbpkg.yaml"))
    print(manifest.name, manifest.version)
    print(manifest.capabilities)
    print(manifest.permissions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional YAML support
try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    yaml = None


# ═══════════════════════════════════════════════════════════════════════════
# Permission Scopes
# ═══════════════════════════════════════════════════════════════════════════


class PermissionScope(StrEnum):
    """Sandbox permission scopes that packages can request.

    Modeled after mobile app permission systems — explicit, reviewable,
    and revocable. The PermissionEngine enforces these at runtime.
    """

    # Filesystem
    READ_FILES = "read_files"  # Read workspace files
    WRITE_FILES = "write_files"  # Create/modify files
    READ_HOME = "read_home"  # Access ~/
    EXECUTE_CODE = "execute_code"  # Run code in sandbox

    # Network
    INTERNET = "internet"  # Outbound HTTP/HTTPS
    LOCALHOST = "localhost"  # Access localhost services
    DNS_LOOKUP = "dns_lookup"  # DNS resolution

    # System
    SHELL = "shell"  # Execute shell commands
    ENVIRONMENT = "environment"  # Read environment variables
    PROCESS_SPAWN = "process_spawn"  # Spawn subprocesses

    # Communication
    SEND_EMAIL = "send_email"
    SEND_NOTIFICATION = "send_notification"
    WEBHOOK = "webhook"  # Send/receive webhooks

    # Memory
    READ_MEMORY = "read_memory"  # Query episodic/semantic memory
    WRITE_MEMORY = "write_memory"  # Store into memory systems
    READ_KNOWLEDGE = "read_knowledge"  # Query knowledge base
    WRITE_KNOWLEDGE = "write_knowledge"  # Ingest into knowledge base

    # Cognition
    INVOKE_LLM = "invoke_llm"  # Make LLM inference calls
    SPAWN_AGENT = "spawn_agent"  # Create sub-agents
    MODIFY_GOALS = "modify_goals"  # Alter goal hierarchy
    MODIFY_SKILLS = "modify_skills"  # Register/modify skills

    # Hardware
    MICROPHONE = "microphone"
    CAMERA = "camera"
    GPIO = "gpio"  # IoT/Raspberry Pi GPIO
    MQTT = "mqtt"  # MQTT broker access


# ═══════════════════════════════════════════════════════════════════════════
# Capability Declaration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CapabilityDeclaration:
    """A capability provided by this package.

    Declares what this package can do, enabling automatic registration
    into the SemanticCapabilityRegistry at install time.
    """

    name: str  # e.g., "deploy.staging"
    purpose: str  # Natural language description
    domains: list[str] = field(default_factory=list)
    estimated_latency_ms: float = 5000
    confidence: float = 0.8
    side_effects: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityDeclaration:
        return cls(
            name=data.get("name", ""),
            purpose=data.get("purpose", ""),
            domains=data.get("domains", []),
            estimated_latency_ms=data.get("estimated_latency_ms", 5000),
            confidence=data.get("confidence", 0.8),
            side_effects=data.get("side_effects", []),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle Hooks
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LifecycleHooks:
    """Optional lifecycle hook scripts."""

    on_install: str = ""  # Script to run after installation
    on_activate: str = ""  # Script to run on activation
    on_deactivate: str = ""  # Script to run on deactivation
    on_uninstall: str = ""  # Script to run before removal

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LifecycleHooks:
        if not data:
            return cls()
        return cls(
            on_install=data.get("on_install", ""),
            on_activate=data.get("on_activate", ""),
            on_deactivate=data.get("on_deactivate", ""),
            on_uninstall=data.get("on_uninstall", ""),
        )


# ═══════════════════════════════════════════════════════════════════════════
# HBPkg Manifest
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HBPkgManifest:
    """Manifest for a .hbpkg agent package.

    This is the evolution of PluginManifest (v2) into a full-featured
    package manifest. It adds:
      - Semantic capability declarations
      - Permission scope requests
      - Profile compatibility
      - Package dependencies
      - Lifecycle hooks

    Example ``hbpkg.yaml``::

        name: devops-agent
        version: "1.0.0"
        description: "DevOps automation skills for CI/CD pipelines"
        author: "dumith"
        license: MIT

        # Directories within the package
        skills_dir: skills
        knowledge_dir: knowledge
        actions_dir: actions
        prompts_dir: prompts
        policies_dir: policies
        config_dir: config

        # What this package provides
        capabilities:
          - name: deploy.staging
            purpose: "Deploy current branch to staging environment"
            domains: [devops, ci-cd]
            confidence: 0.9
          - name: deploy.production
            purpose: "Deploy to production with approval workflow"
            domains: [devops, ci-cd]
            side_effects: [production_deployment]
            confidence: 0.85

        # Sandbox permissions requested
        permissions:
          - shell
          - internet
          - read_files
          - write_files
          - send_notification

        # Which brain profiles support this package
        compatible_profiles:
          - full
          - lite
          - research

        # Package dependencies
        dependencies:
          - name: git-tools
            version: ">=1.0.0"

        # Lifecycle hooks
        hooks:
          on_install: "scripts/setup.sh"
          on_activate: "scripts/activate.py"

        tags: [devops, deployment, ci-cd, docker]
    """

    # Identity
    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    license: str = "MIT"
    manifest_version: int = 3  # v3 = .hbpkg format

    # Package directories
    skills_dir: str = "skills"
    knowledge_dir: str = "knowledge"
    actions_dir: str = "actions"
    models_dir: str = "models"
    prompts_dir: str = "prompts"
    policies_dir: str = "policies"
    config_dir: str = "config"

    # Capability declarations
    capabilities: list[CapabilityDeclaration] = field(default_factory=list)

    # Permission requests
    permissions: list[str] = field(default_factory=list)

    # Profile compatibility
    compatible_profiles: list[str] = field(default_factory=lambda: ["full", "lite"])

    # Dependencies on other .hbpkg packages
    dependencies: list[dict[str, str]] = field(default_factory=list)

    # Lifecycle hooks
    hooks: LifecycleHooks = field(default_factory=LifecycleHooks)

    # Tags for search / categorization
    tags: list[str] = field(default_factory=list)

    # Source tracking
    source_path: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HBPkgManifest:
        """Create from a parsed YAML/JSON dict."""
        if not data.get("name"):
            raise ValueError("hbpkg.yaml missing required field: 'name'")

        capabilities = [CapabilityDeclaration.from_dict(c) for c in data.get("capabilities", [])]

        return cls(
            name=data["name"],
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            license=data.get("license", "MIT"),
            manifest_version=data.get("manifest_version", 3),
            skills_dir=data.get("skills_dir", "skills"),
            knowledge_dir=data.get("knowledge_dir", "knowledge"),
            actions_dir=data.get("actions_dir", "actions"),
            models_dir=data.get("models_dir", "models"),
            prompts_dir=data.get("prompts_dir", "prompts"),
            policies_dir=data.get("policies_dir", "policies"),
            config_dir=data.get("config_dir", "config"),
            capabilities=capabilities,
            permissions=data.get("permissions", []),
            compatible_profiles=data.get("compatible_profiles", ["full", "lite"]),
            dependencies=data.get("dependencies", []),
            hooks=LifecycleHooks.from_dict(data.get("hooks")),
            tags=data.get("tags", []),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> HBPkgManifest:
        """Load manifest from a YAML file."""
        if yaml is None:
            raise ImportError("PyYAML required to load hbpkg.yaml")

        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        manifest = cls.from_dict(data)
        manifest.source_path = str(path)
        return manifest

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for YAML/JSON output."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "manifest_version": self.manifest_version,
            "skills_dir": self.skills_dir,
            "knowledge_dir": self.knowledge_dir,
            "actions_dir": self.actions_dir,
            "models_dir": self.models_dir,
            "prompts_dir": self.prompts_dir,
            "policies_dir": self.policies_dir,
            "config_dir": self.config_dir,
            "capabilities": [
                {
                    "name": c.name,
                    "purpose": c.purpose,
                    "domains": c.domains,
                    "estimated_latency_ms": c.estimated_latency_ms,
                    "confidence": c.confidence,
                    "side_effects": c.side_effects,
                }
                for c in self.capabilities
            ],
            "permissions": self.permissions,
            "compatible_profiles": self.compatible_profiles,
            "dependencies": self.dependencies,
            "hooks": {
                "on_install": self.hooks.on_install,
                "on_activate": self.hooks.on_activate,
                "on_deactivate": self.hooks.on_deactivate,
                "on_uninstall": self.hooks.on_uninstall,
            },
            "tags": self.tags,
        }

    def validate(self) -> list[str]:
        """Validate the manifest, returning a list of error messages."""
        errors: list[str] = []
        if not self.name:
            errors.append("Package name is required")
        if not self.version:
            errors.append("Package version is required")
        for cap in self.capabilities:
            if not cap.name:
                errors.append("Capability declaration missing 'name'")
            if not cap.purpose:
                errors.append(f"Capability '{cap.name}' missing 'purpose'")
        for perm in self.permissions:
            try:
                PermissionScope(perm)
            except ValueError:
                errors.append(f"Unknown permission scope: '{perm}'")
        return errors
