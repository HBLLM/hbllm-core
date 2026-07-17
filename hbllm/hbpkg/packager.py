"""
HBPKG Packager — Pack and unpack .hbpkg agent packages.

Provides the ``hbllm install`` pipeline::

    .hbpkg archive (tar.gz with hbpkg.yaml)
        ↓
    Packager.unpack()
        ↓
    Validate manifest + permissions
        ↓
    Install skills/ → SkillInstaller
        ↓
    Install knowledge/ → KnowledgeBase
        ↓
    Register capabilities → SemanticCapabilityRegistry
        ↓
    Run lifecycle hooks

Also supports ``hbllm pack`` for creating distributable .hbpkg archives::

    my-agent/
        ↓
    Packager.pack()
        ↓
    my-agent-1.0.0.hbpkg  (tar.gz)

Usage::

    from hbllm.hbpkg.packager import Packager

    packager = Packager(install_dir=Path("~/.hbllm/packages"))

    # Install from a directory
    result = await packager.install(Path("my-agent/"))

    # Install from an archive
    result = await packager.install(Path("my-agent-1.0.0.hbpkg"))

    # Pack a directory into an archive
    archive_path = await packager.pack(Path("my-agent/"))

    # List installed packages
    packages = await packager.list_installed()

    # Uninstall
    await packager.uninstall("my-agent")
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hbpkg.manifest import HBPkgManifest, PermissionScope
from hbllm.skills.installer import SkillInstaller
from hbllm.skills.parser import SkillParser

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Installation Result
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PackageInstallResult:
    """Result of a package installation."""

    name: str
    version: str
    success: bool
    message: str = ""
    skills_installed: int = 0
    knowledge_files_ingested: int = 0
    capabilities_registered: int = 0
    warnings: list[str] = field(default_factory=list)
    install_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Packager
# ═══════════════════════════════════════════════════════════════════════════


class Packager:
    """Packs and unpacks .hbpkg agent packages.

    The Packager is the central controller for the package lifecycle:
      1. ``pack()`` — Create a distributable .hbpkg archive.
      2. ``install()`` — Unpack, validate, and install a package.
      3. ``uninstall()`` — Remove a package and its assets.
      4. ``list_installed()`` — List all installed packages.

    Dependencies (all optional — pass None to skip):
      - ``skill_registry`` — For skill installation.
      - ``capability_registry`` — For capability registration.
      - ``permission_engine`` — For permission grant/deny.
    """

    MANIFEST_FILENAME = "hbpkg.yaml"
    ARCHIVE_EXTENSION = ".hbpkg"

    def __init__(
        self,
        install_dir: Path | None = None,
        skill_registry: Any = None,
        capability_registry: Any = None,
        permission_engine: Any = None,
    ) -> None:
        self._install_dir = install_dir or Path.home() / ".hbllm" / "packages"
        self._install_dir.mkdir(parents=True, exist_ok=True)

        self._skill_registry = skill_registry
        self._capability_registry = capability_registry
        self._permission_engine = permission_engine
        self._skill_installer = SkillInstaller(
            skill_registry=skill_registry,
            capability_registry=capability_registry,
            parser=SkillParser(),
        )

    # ── Pack ─────────────────────────────────────────────────────────────

    async def pack(self, source_dir: Path, output_dir: Path | None = None) -> Path:
        """Create a .hbpkg archive from a directory.

        Args:
            source_dir: Directory containing hbpkg.yaml and assets.
            output_dir: Where to write the archive. Defaults to CWD.

        Returns:
            Path to the created .hbpkg archive.

        Raises:
            FileNotFoundError: If the source directory or manifest is missing.
        """
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        manifest_path = source_dir / self.MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No {self.MANIFEST_FILENAME} found in {source_dir}"
            )

        manifest = HBPkgManifest.from_yaml(manifest_path)
        errors = manifest.validate()
        if errors:
            raise ValueError(f"Manifest validation failed: {'; '.join(errors)}")

        # Build archive name
        archive_name = f"{manifest.name}-{manifest.version}{self.ARCHIVE_EXTENSION}"
        out_dir = output_dir or Path.cwd()
        archive_path = out_dir / archive_name

        # Create tar.gz
        with tarfile.open(str(archive_path), "w:gz") as tar:
            tar.add(str(source_dir), arcname=manifest.name)

        logger.info("Packed %s → %s", source_dir, archive_path)
        return archive_path

    # ── Install ──────────────────────────────────────────────────────────

    async def install(self, source: Path) -> PackageInstallResult:
        """Install a .hbpkg package from a directory or archive.

        Args:
            source: Path to either a directory or a .hbpkg archive.

        Returns:
            PackageInstallResult with details.
        """
        start_time = time.monotonic()

        # If archive, unpack first
        if source.suffix == self.ARCHIVE_EXTENSION or (
            source.is_file() and tarfile.is_tarfile(str(source))
        ):
            pkg_dir = await self._unpack_archive(source)
        elif source.is_dir():
            pkg_dir = source
        else:
            return PackageInstallResult(
                name=source.name,
                version="",
                success=False,
                message=f"Source is neither a directory nor a .hbpkg archive: {source}",
            )

        # Load manifest
        manifest_path = pkg_dir / self.MANIFEST_FILENAME
        if not manifest_path.exists():
            # Try looking one level deeper (archive root might be a subdirectory)
            subdirs = [d for d in pkg_dir.iterdir() if d.is_dir()]
            for sub in subdirs:
                if (sub / self.MANIFEST_FILENAME).exists():
                    pkg_dir = sub
                    manifest_path = sub / self.MANIFEST_FILENAME
                    break

        if not manifest_path.exists():
            return PackageInstallResult(
                name=source.name,
                version="",
                success=False,
                message=f"No {self.MANIFEST_FILENAME} found in package",
            )

        try:
            manifest = HBPkgManifest.from_yaml(manifest_path)
        except Exception as e:
            return PackageInstallResult(
                name=source.name,
                version="",
                success=False,
                message=f"Failed to parse manifest: {e}",
            )

        # Validate manifest
        errors = manifest.validate()
        if errors:
            return PackageInstallResult(
                name=manifest.name,
                version=manifest.version,
                success=False,
                message=f"Manifest validation failed: {'; '.join(errors)}",
            )

        warnings: list[str] = []

        # Check permissions
        if self._permission_engine:
            denied = self._check_permissions(manifest)
            if denied:
                return PackageInstallResult(
                    name=manifest.name,
                    version=manifest.version,
                    success=False,
                    message=f"Permissions denied: {', '.join(denied)}",
                )

        # Install skills
        skills_count = 0
        skills_dir = pkg_dir / manifest.skills_dir
        if skills_dir.is_dir():
            batch = await self._skill_installer.install_directory(skills_dir)
            skills_count = batch.success_count
            if batch.error_count > 0:
                for r in batch.results:
                    if r.is_error:
                        warnings.append(f"Skill '{r.name}': {r.message}")

        # Register capabilities from manifest
        caps_count = 0
        if self._capability_registry and manifest.capabilities:
            caps_count = await self._register_capabilities(manifest)

        # Copy to install directory for persistence
        install_dest = self._install_dir / manifest.name
        if pkg_dir != install_dest:
            if install_dest.exists():
                shutil.rmtree(install_dest)
            shutil.copytree(pkg_dir, install_dest)

        elapsed = (time.monotonic() - start_time) * 1000

        logger.info(
            "Installed package '%s' v%s (%d skills, %d capabilities, %.0fms)",
            manifest.name,
            manifest.version,
            skills_count,
            caps_count,
            elapsed,
        )

        return PackageInstallResult(
            name=manifest.name,
            version=manifest.version,
            success=True,
            message=f"Package '{manifest.name}' installed successfully",
            skills_installed=skills_count,
            capabilities_registered=caps_count,
            warnings=warnings,
            install_time_ms=elapsed,
        )

    # ── Uninstall ────────────────────────────────────────────────────────

    async def uninstall(self, package_name: str) -> bool:
        """Uninstall a package and remove its assets.

        Args:
            package_name: Name of the package to remove.

        Returns:
            True if the package was found and removed.
        """
        pkg_dir = self._install_dir / package_name
        if not pkg_dir.exists():
            logger.warning("Package '%s' not found in %s", package_name, self._install_dir)
            return False

        # Load manifest for cleanup
        manifest_path = pkg_dir / self.MANIFEST_FILENAME
        if manifest_path.exists():
            try:
                manifest = HBPkgManifest.from_yaml(manifest_path)

                # Unregister capabilities
                if self._capability_registry:
                    for cap in manifest.capabilities:
                        self._capability_registry.unregister(cap.name)

            except Exception as e:
                logger.warning("Error during cleanup of '%s': %s", package_name, e)

        # Remove package directory
        shutil.rmtree(pkg_dir)
        logger.info("Uninstalled package '%s'", package_name)
        return True

    # ── List Installed ───────────────────────────────────────────────────

    async def list_installed(self) -> list[HBPkgManifest]:
        """List all installed packages.

        Returns:
            List of manifests from installed packages.
        """
        packages: list[HBPkgManifest] = []
        if not self._install_dir.exists():
            return packages

        for pkg_dir in sorted(self._install_dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            manifest_path = pkg_dir / self.MANIFEST_FILENAME
            if manifest_path.exists():
                try:
                    manifest = HBPkgManifest.from_yaml(manifest_path)
                    manifest.source_path = str(pkg_dir)
                    packages.append(manifest)
                except Exception as e:
                    logger.warning(
                        "Failed to load manifest for '%s': %s",
                        pkg_dir.name,
                        e,
                    )

        return packages

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _unpack_archive(self, archive_path: Path) -> Path:
        """Unpack a .hbpkg archive to a temporary directory."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="hbpkg-"))
        with tarfile.open(str(archive_path), "r:gz") as tar:
            tar.extractall(str(tmp_dir))
        logger.debug("Unpacked %s → %s", archive_path, tmp_dir)
        return tmp_dir

    def _check_permissions(self, manifest: HBPkgManifest) -> list[str]:
        """Check if the requested permissions can be granted.

        Returns a list of denied permission names.
        """
        denied: list[str] = []
        for perm in manifest.permissions:
            if self._permission_engine and hasattr(
                self._permission_engine, "is_allowed"
            ):
                if not self._permission_engine.is_allowed(manifest.name, perm):
                    denied.append(perm)
        return denied

    async def _register_capabilities(self, manifest: HBPkgManifest) -> int:
        """Register all capability declarations from the manifest."""
        count = 0
        try:
            from hbllm.brain.planning.semantic_capability_registry import (
                CapabilityDescriptor,
            )

            for cap in manifest.capabilities:
                descriptor = CapabilityDescriptor(
                    name=cap.name,
                    provider_id=f"hbpkg:{manifest.name}",
                    purpose=cap.purpose,
                    domains=cap.domains,
                    estimated_cost=0.0,
                    latency_ms=cap.estimated_latency_ms,
                    confidence=cap.confidence,
                    required_permissions=manifest.permissions,
                    side_effects=cap.side_effects,
                    supported_modalities=["text"],
                )
                self._capability_registry.register(descriptor)
                count += 1
        except Exception as e:
            logger.warning("Failed to register capabilities: %s", e)

        return count
