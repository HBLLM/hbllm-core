"""
Skill Installer — Discovers, imports, and manages SKILL.md packages.

Provides the ``hbllm skill install`` pipeline::

    Source (path / URL / git repo)
        ↓
    SkillParser.parse_file()
        ↓
    Validation (name, steps, tools)
        ↓
    SkillRegistry.store()  (persisted to DB)
        ↓
    SemanticCapabilityRegistry.register()  (discoverable by planner)

Supports:
    - Local file paths (``hbllm skill install ./deploy.md``)
    - Directories (``hbllm skill install ./skills/``)
    - Future: URLs, git repos, .hbpkg archives

Usage::

    from hbllm.skills.installer import SkillInstaller

    installer = SkillInstaller(
        skill_registry=brain.skill_registry,
        capability_registry=brain.capability_registry,
    )

    # Install from a file
    result = await installer.install_file(Path("skills/deploy.md"))

    # Install all skills from a directory
    results = await installer.install_directory(Path("skills/"))

    # List installed skills
    skills = await installer.list_installed()

    # Uninstall a skill
    await installer.uninstall("deploy-to-staging")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.skills.parser import ParsedSkill, SkillParser

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Installation Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class InstallResult:
    """Result of a skill installation attempt."""

    name: str
    success: bool
    message: str = ""
    skill_id: str = ""
    source_path: str = ""

    @property
    def is_error(self) -> bool:
        return not self.success


@dataclass
class BatchInstallResult:
    """Aggregate result of batch installation."""

    results: list[InstallResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.is_error)

    @property
    def all_success(self) -> bool:
        return all(r.success for r in self.results)


# ═══════════════════════════════════════════════════════════════════════════
# Skill Installer
# ═══════════════════════════════════════════════════════════════════════════


class SkillInstaller:
    """Discovers, validates, and installs SKILL.md files into the registry.

    Integrates two registries:
      1. ``SkillRegistry`` — Persistent storage for skill execution.
      2. ``SemanticCapabilityRegistry`` — Planner-visible capability metadata.

    Both registries are optional — pass ``None`` to skip either.
    """

    def __init__(
        self,
        skill_registry: Any = None,  # SkillRegistry
        capability_registry: Any = None,  # SemanticCapabilityRegistry
        parser: SkillParser | None = None,
    ) -> None:
        self._skill_registry = skill_registry
        self._capability_registry = capability_registry
        self._parser = parser or SkillParser()

    # ── Install Methods ──────────────────────────────────────────────────

    async def install_file(self, path: Path) -> InstallResult:
        """Install a single SKILL.md file.

        Args:
            path: Path to the skill file.

        Returns:
            InstallResult with success/failure details.
        """
        # 1. Parse
        parsed = self._parser.parse_file(path)
        if parsed is None:
            return InstallResult(
                name=path.name,
                success=False,
                message=f"Failed to parse skill file: {path}",
                source_path=str(path),
            )

        # 2. Validate
        error = self._validate(parsed)
        if error:
            return InstallResult(
                name=parsed.name,
                success=False,
                message=error,
                source_path=str(path),
            )

        # 3. Register in SkillRegistry
        if self._skill_registry:
            try:
                self._skill_registry.store_skill(
                    name=parsed.name,
                    description=parsed.description,
                    category=parsed.category,
                    steps=parsed.steps,
                    tools_used=parsed.tools_used,
                    success_criteria=parsed.success_criteria,
                    examples=parsed.examples,
                )
                logger.info("Installed skill '%s' into SkillRegistry", parsed.name)
            except Exception as e:
                return InstallResult(
                    name=parsed.name,
                    success=False,
                    message=f"SkillRegistry.store_skill() failed: {e}",
                    skill_id=parsed.skill_id,
                    source_path=str(path),
                )

        # 4. Register in SemanticCapabilityRegistry
        if self._capability_registry:
            try:
                from hbllm.brain.planning.semantic_capability_registry import (
                    CapabilityDescriptor,
                )

                descriptor = CapabilityDescriptor(
                    name=f"skill.{parsed.name}",
                    provider_id="skill-registry",
                    purpose=parsed.description,
                    domains=[parsed.category] + parsed.tags,
                    estimated_cost=0.0,  # Skills are "free" (just LLM calls)
                    latency_ms=5000,  # Skills typically take a few seconds
                    confidence=0.8,
                    required_permissions=[],
                    side_effects=[],
                    supported_modalities=["text"],
                )
                self._capability_registry.register(descriptor)
                logger.info(
                    "Registered capability 'skill.%s' in SemanticCapabilityRegistry",
                    parsed.name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to register capability for skill '%s': %s",
                    parsed.name,
                    e,
                )

        return InstallResult(
            name=parsed.name,
            success=True,
            message=f"Installed skill '{parsed.name}' ({len(parsed.steps)} steps)",
            skill_id=parsed.skill_id,
            source_path=str(path),
        )

    async def install_directory(self, directory: Path) -> BatchInstallResult:
        """Install all SKILL.md files in a directory.

        Args:
            directory: Directory to scan for .md files.

        Returns:
            BatchInstallResult with per-file results.
        """
        batch = BatchInstallResult()

        if not directory.is_dir():
            batch.results.append(
                InstallResult(
                    name=str(directory),
                    success=False,
                    message=f"Not a directory: {directory}",
                )
            )
            return batch

        for path in sorted(directory.glob("*.md")):
            result = await self.install_file(path)
            batch.results.append(result)

        logger.info(
            "Batch install from %s: %d success, %d errors",
            directory,
            batch.success_count,
            batch.error_count,
        )
        return batch

    async def uninstall(self, skill_name: str) -> bool:
        """Uninstall a skill by name.

        Removes from both SkillRegistry and SemanticCapabilityRegistry.

        Args:
            skill_name: Name of the skill to remove.

        Returns:
            True if the skill was found and removed.
        """
        removed = False

        if self._skill_registry and hasattr(self._skill_registry, "delete_skill"):
            try:
                self._skill_registry.delete_skill(skill_name)
                removed = True
                logger.info("Removed skill '%s' from SkillRegistry", skill_name)
            except Exception as e:
                logger.warning("Failed to remove skill '%s': %s", skill_name, e)

        if self._capability_registry:
            cap_name = f"skill.{skill_name}"
            if self._capability_registry.unregister(cap_name):
                removed = True
                logger.info(
                    "Removed capability '%s' from SemanticCapabilityRegistry",
                    cap_name,
                )

        return removed

    # ── Validation ───────────────────────────────────────────────────────

    @staticmethod
    def _validate(skill: ParsedSkill) -> str | None:
        """Validate a parsed skill.

        Returns an error message string if invalid, or None if valid.
        """
        if not skill.name:
            return "Skill has no name"
        if not skill.description:
            return f"Skill '{skill.name}' has no description"
        if not skill.steps:
            return f"Skill '{skill.name}' has no steps"
        return None
