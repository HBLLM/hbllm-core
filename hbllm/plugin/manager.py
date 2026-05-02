"""
HBLLM Plugin Manager — Runtime lifecycle for plugin bundles.

Provides the full plugin lifecycle: discovery, loading, asset ingestion,
soft-deactivation (graduation/orphaning), and runtime hot-loading.

Designed to work standalone with hbllm core, or extended by higher-level
frameworks like Sentra that add additional scan paths and agent hooks.

Usage::

    from hbllm.plugin.manager import PluginManager

    # Create with brain subsystems
    manager = PluginManager(
        skill_registry=brain.skill_registry,
        policy_engine=brain.policy_engine,
    )

    # Discover and load all plugins from default paths
    await manager.discover_plugins()

    # Hot-load a specific plugin at runtime (no restart needed)
    loaded = await manager.load_bundle(Path("~/.hbllm/plugins/my-plugin"))

    # Soft-unload (graduate/orphan skills, deactivate policies)
    await manager.unload_bundle("my-plugin")

    # Start background watcher for new plugins
    await manager.watch_directories()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from hbllm.plugin.bundle import PluginAssets, PluginBundle, PluginManifest

logger = logging.getLogger(__name__)


# ── Prompt Store ──────────────────────────────────────────────────────────────


class PromptStore:
    """
    Lightweight store for named prompt templates contributed by plugins.

    Templates are namespaced by plugin name (e.g., "sentinel-shield:threat_explanation").
    """

    def __init__(self) -> None:
        self._templates: dict[str, str] = {}
        self._sources: dict[str, str] = {}  # template_key → plugin_name

    def add(self, key: str, template: str, source: str = "") -> None:
        """Add a named prompt template."""
        self._templates[key] = template
        if source:
            self._sources[key] = source

    def get(self, key: str) -> str | None:
        """Get a prompt template by key."""
        return self._templates.get(key)

    def render(self, key: str, **kwargs: Any) -> str | None:
        """Get a template and format it with the given kwargs."""
        template = self._templates.get(key)
        if template is None:
            return None
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning("Prompt template '%s' missing variable: %s", key, e)
            return template

    def remove_by_source(self, source: str) -> int:
        """Remove all templates from a given source. Returns count removed."""
        keys_to_remove = [k for k, s in self._sources.items() if s == source]
        for key in keys_to_remove:
            del self._templates[key]
            del self._sources[key]
        return len(keys_to_remove)

    def archive_by_source(self, source: str) -> int:
        """
        Archive templates from a source (soft removal).

        Templates are moved to an 'archived:' prefix — they stop appearing
        in active lookups but remain retrievable for restoration.
        """
        keys_to_archive = [k for k, s in self._sources.items() if s == source]
        for key in keys_to_archive:
            archived_key = f"archived:{key}"
            self._templates[archived_key] = self._templates.pop(key)
            self._sources[archived_key] = self._sources.pop(key)
        if keys_to_archive:
            logger.info(
                "Archived %d prompt templates from '%s'",
                len(keys_to_archive),
                source,
            )
        return len(keys_to_archive)

    def restore_archived(self, source: str) -> int:
        """Restore archived templates from a source back to active."""
        prefix = "archived:"
        keys_to_restore = [
            k for k, s in self._sources.items() if s == source and k.startswith(prefix)
        ]
        for key in keys_to_restore:
            original_key = key[len(prefix) :]
            self._templates[original_key] = self._templates.pop(key)
            self._sources[original_key] = self._sources.pop(key)
        return len(keys_to_restore)

    def list_templates(self) -> list[dict[str, str]]:
        """List all active (non-archived) templates with their sources."""
        return [
            {"key": k, "source": self._sources.get(k, ""), "preview": v[:80]}
            for k, v in self._templates.items()
            if not k.startswith("archived:")
        ]

    @property
    def count(self) -> int:
        return sum(1 for k in self._templates if not k.startswith("archived:"))


# ── Loaded Bundle State ──────────────────────────────────────────────────────


class LoadedBundle:
    """Tracks a loaded plugin bundle and the assets it contributed."""

    def __init__(self, bundle: PluginBundle) -> None:
        self.bundle = bundle
        self.name = bundle.manifest.name
        self.namespace = bundle.manifest.namespace
        self.loaded_at = time.time()
        self.knowledge_source_ids: list[str] = []
        self.skill_ids: list[str] = []
        self.policy_names: list[str] = []
        self.prompt_keys: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        """Serialize bundle state for API responses."""
        return {
            "name": self.name,
            "version": self.bundle.manifest.version,
            "manifest_version": self.bundle.manifest.manifest_version,
            "namespace": self.namespace,
            "loaded_at": self.loaded_at,
            "assets": self.bundle.assets.summary(),
            "ingested": {
                "knowledge_sources": len(self.knowledge_source_ids),
                "skills": len(self.skill_ids),
                "policies": len(self.policy_names),
                "prompts": len(self.prompt_keys),
            },
        }


# ── Plugin Manager ────────────────────────────────────────────────────────────

# Default scan interval for background watcher
_WATCH_INTERVAL_SECONDS = 30


class PluginManager:
    """
    Manages plugin lifecycle, bundle loading, and asset dispatch.

    Supports:
      - v1 code-only plugins (register/unregister)
      - v2 rich bundles (load/unload with full asset ingestion)
      - Runtime hot-loading (no restart needed)
      - Background directory watching for auto-discovery
      - Soft deactivation (skills graduated/orphaned, policies deactivated)

    Args:
        plugin_dirs: Directories to scan for plugins. Defaults to
            ``~/.hbllm/plugins/`` and the package-level plugins dir.
        skill_registry: SkillRegistry for ingesting plugin skills.
        policy_engine: PolicyEngine for ingesting plugin policies.
        knowledge_base: KnowledgeBase for ingesting plugin knowledge.
    """

    def __init__(
        self,
        plugin_dirs: list[Path | str] | None = None,
        skill_registry: Any = None,
        policy_engine: Any = None,
        knowledge_base: Any = None,
    ) -> None:
        # Brain subsystems for asset ingestion
        self._skill_registry = skill_registry
        self._policy_engine = policy_engine
        self._knowledge_base = knowledge_base

        # Storage
        self._plugins: dict[str, Any] = {}  # v1 code-only plugins
        self._bundles: dict[str, LoadedBundle] = {}  # v2 rich bundles
        self.prompt_store = PromptStore()

        # Plugin directories
        self._plugin_dirs: list[Path] = []
        if plugin_dirs:
            self._plugin_dirs.extend(Path(p).expanduser() for p in plugin_dirs)

        # Always include default paths
        default_user_dir = Path.home() / ".hbllm" / "plugins"
        if default_user_dir not in self._plugin_dirs:
            self._plugin_dirs.append(default_user_dir)

        # Package-level plugins (shipped with hbllm)
        try:
            import hbllm

            pkg_plugins = Path(hbllm.__file__).parent.parent / "plugins"
            if pkg_plugins.exists() and pkg_plugins not in self._plugin_dirs:
                self._plugin_dirs.append(pkg_plugins)
        except Exception:
            pass

        # Background watcher
        self._watch_task: asyncio.Task[None] | None = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def plugin_dirs(self) -> list[Path]:
        """Configured plugin scan directories."""
        return list(self._plugin_dirs)

    @property
    def bundles(self) -> dict[str, LoadedBundle]:
        """Loaded v2 rich bundles."""
        return self._bundles

    @property
    def bundle_count(self) -> int:
        """Total number of loaded bundles."""
        return len(self._bundles)

    # ── Plugin Directory Management ──────────────────────────────────

    def add_plugin_dir(self, path: Path | str) -> None:
        """Add an additional directory to scan for plugins."""
        resolved = Path(path).expanduser().resolve()
        if resolved not in self._plugin_dirs:
            self._plugin_dirs.append(resolved)
            logger.info("Added plugin directory: %s", resolved)

    # ── v2: Rich Bundle Loading ──────────────────────────────────────

    async def load_bundle(
        self,
        plugin_path: Path | str,
        *,
        knowledge_base: Any = None,
        skill_registry: Any = None,
        policy_engine: Any = None,
    ) -> LoadedBundle:
        """
        Load a plugin bundle and ingest all bundled assets.

        Can be called at startup or at runtime for hot-loading.

        Args:
            plugin_path: Path to the plugin directory.
            knowledge_base: Override for KnowledgeBase (uses instance default if None).
            skill_registry: Override for SkillRegistry (uses instance default if None).
            policy_engine: Override for PolicyEngine (uses instance default if None).

        Returns:
            LoadedBundle tracking all contributed assets.
        """
        # Use instance defaults if not explicitly overridden
        kb = knowledge_base if knowledge_base is not None else self._knowledge_base
        sr = skill_registry if skill_registry is not None else self._skill_registry
        pe = policy_engine if policy_engine is not None else self._policy_engine

        bundle = PluginBundle(plugin_path)
        loaded = LoadedBundle(bundle)

        # Check for duplicate — unload first
        if bundle.manifest.name in self._bundles:
            logger.warning(
                "Plugin '%s' is already loaded — unloading first",
                bundle.manifest.name,
            )
            await self.unload_bundle(bundle.manifest.name)

        # Ingest assets if this is a v2 bundle
        if bundle.manifest.is_v2 and not bundle.assets.is_empty:
            await self._ingest_knowledge(loaded, kb)
            self._register_skills(loaded, sr)
            self._load_policies(loaded, pe)
            self._store_prompts(loaded)

        self._bundles[bundle.manifest.name] = loaded

        logger.info(
            "Bundle loaded: %s v%s — ingested %s",
            bundle.manifest.name,
            bundle.manifest.version,
            loaded.to_dict()["ingested"],
        )

        return loaded

    async def unload_bundle(self, plugin_name: str) -> bool:
        """
        Unload a plugin bundle using soft deactivation.

        The brain never forgets what it learned. When a plugin is removed:
          - **Knowledge**: Stays permanently — once learned, never deleted.
          - **Skills**: Graduated (if used) or orphaned (if not) — never deleted.
          - **Policies**: Deactivated (not deleted) — records persist.
          - **Prompts**: Moved to archive — still retrievable but not active.

        Returns True if the bundle was found and unloaded.
        """
        loaded = self._bundles.pop(plugin_name, None)
        if loaded is None:
            logger.warning("Bundle '%s' not found for unloading", plugin_name)
            return False

        graduated_count = 0
        orphaned_count = 0
        deactivated_count = 0

        sr = self._skill_registry
        pe = self._policy_engine

        # ── Knowledge: NEVER remove ──────────────────────────────────
        if loaded.knowledge_source_ids:
            logger.info(
                "[%s] Retaining %d knowledge sources (learned knowledge persists)",
                plugin_name,
                len(loaded.knowledge_source_ids),
            )

        # ── Skills: Graduate or Orphan ───────────────────────────────
        if sr and loaded.skill_ids:
            if hasattr(sr, "graduate_experienced_skills"):
                graduated_ids = sr.graduate_experienced_skills(loaded.namespace, min_invocations=1)
                graduated_count = len(graduated_ids)

            if hasattr(sr, "find_by_source"):
                remaining = sr.find_by_source(loaded.namespace)
                for skill in remaining:
                    try:
                        if hasattr(sr, "_store"):
                            skill.source = f"orphaned:{loaded.namespace}"
                            sr._store(skill)
                            orphaned_count += 1
                    except Exception as e:
                        logger.error(
                            "[%s] Failed to orphan skill %s: %s",
                            plugin_name,
                            skill.skill_id,
                            e,
                        )

            logger.info(
                "[%s] Skills: %d graduated to core, %d orphaned (all retained)",
                plugin_name,
                graduated_count,
                orphaned_count,
            )

        # ── Policies: Deactivate (not delete) ────────────────────────
        if pe and loaded.policy_names:
            for policy_name in loaded.policy_names:
                try:
                    if hasattr(pe, "get_policy"):
                        policy = pe.get_policy(policy_name)
                        if policy:
                            policy.enabled = False
                            deactivated_count += 1
                    else:
                        pe.remove_policy(policy_name)
                except Exception as e:
                    logger.error(
                        "[%s] Failed to deactivate policy %s: %s",
                        plugin_name,
                        policy_name,
                        e,
                    )

            logger.info(
                "[%s] Deactivated %d policies (records retained)",
                plugin_name,
                deactivated_count,
            )

        # ── Prompts: Archive ─────────────────────────────────────────
        self.prompt_store.archive_by_source(plugin_name)

        logger.info(
            "Bundle unloaded (soft): %s — graduated=%d orphaned=%d deactivated=%d knowledge=retained",
            plugin_name,
            graduated_count,
            orphaned_count,
            deactivated_count,
        )
        return True

    # ── Plugin Discovery ─────────────────────────────────────────────

    async def discover_plugins(self) -> list[LoadedBundle]:
        """
        Scan all configured plugin directories and load new bundles.

        Can be called at startup or at runtime. Only loads bundles that
        aren't already loaded. Returns list of newly loaded bundles.
        """
        newly_loaded: list[LoadedBundle] = []

        for plugins_path in self._plugin_dirs:
            if not plugins_path.exists():
                continue

            for entry in sorted(plugins_path.iterdir()):
                if not entry.is_dir() or entry.name.startswith(("_", ".")):
                    continue

                # Skip already loaded
                if entry.name in self._bundles:
                    continue

                # Check for v2 manifest
                manifest_path = entry / "plugin.json"
                if not manifest_path.exists():
                    continue

                try:
                    with open(manifest_path) as f:
                        manifest_data = json.load(f)

                    if manifest_data.get("manifest_version", 1) >= 2:
                        loaded = await self.load_bundle(entry)
                        newly_loaded.append(loaded)
                except Exception as e:
                    logger.error(
                        "Failed to load plugin from '%s': %s",
                        entry.name,
                        e,
                    )

        if newly_loaded:
            logger.info(
                "Plugin discovery complete: %d new bundles loaded",
                len(newly_loaded),
            )

        return newly_loaded

    # ── Background Watcher ───────────────────────────────────────────

    async def watch_directories(self, interval: float = _WATCH_INTERVAL_SECONDS) -> None:
        """
        Start background polling for new plugins.

        Periodically scans plugin directories and auto-loads any new
        bundles found. This enables runtime hot-loading without restart.
        """
        if self._watch_task is not None:
            logger.warning("Plugin watcher is already running")
            return

        async def _poll_loop() -> None:
            logger.info(
                "Plugin watcher started (interval=%ds, dirs=%d)",
                interval,
                len(self._plugin_dirs),
            )
            while True:
                try:
                    await asyncio.sleep(interval)
                    new_bundles = await self.discover_plugins()
                    if new_bundles:
                        logger.info(
                            "Watcher: auto-loaded %d new plugins: %s",
                            len(new_bundles),
                            [b.name for b in new_bundles],
                        )
                except asyncio.CancelledError:
                    logger.info("Plugin watcher stopped")
                    break
                except Exception as e:
                    logger.error("Plugin watcher error: %s", e)

        self._watch_task = asyncio.create_task(_poll_loop())

    async def stop_watching(self) -> None:
        """Stop the background plugin watcher."""
        if self._watch_task is not None:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

    # ── Asset Ingestion ──────────────────────────────────────────────

    async def _ingest_knowledge(self, loaded: LoadedBundle, knowledge_base: Any) -> None:
        """Ingest knowledge documents into the KnowledgeBase."""
        if not loaded.bundle.assets.has_knowledge:
            return

        if knowledge_base is None:
            logger.debug(
                "[%s] No KnowledgeBase provided — skipping knowledge ingestion",
                loaded.name,
            )
            return

        for knowledge_file in loaded.bundle.assets.knowledge_files:
            try:
                source_name = f"{loaded.namespace}/{knowledge_file.name}"

                if hasattr(knowledge_base, "add_source"):
                    source = knowledge_base.add_source(
                        str(knowledge_file),
                        source_type="file",
                    )
                    source_id = getattr(source, "source_id", source_name)
                    loaded.knowledge_source_ids.append(source_id)

                    if hasattr(knowledge_base, "ingest_source"):
                        knowledge_base.ingest_source(source_id)

                    logger.info(
                        "[%s] Ingested knowledge: %s",
                        loaded.name,
                        knowledge_file.name,
                    )
                else:
                    logger.warning(
                        "[%s] KnowledgeBase missing 'add_source' method",
                        loaded.name,
                    )

            except FileNotFoundError:
                logger.error(
                    "[%s] Knowledge file not found: %s",
                    loaded.name,
                    knowledge_file,
                )
            except Exception as e:
                logger.error(
                    "[%s] Failed to ingest knowledge file %s: %s",
                    loaded.name,
                    knowledge_file.name,
                    e,
                )

    def _register_skills(self, loaded: LoadedBundle, skill_registry: Any) -> None:
        """Register pre-built skills into the SkillRegistry."""
        if not loaded.bundle.assets.has_skills:
            return

        if skill_registry is None:
            logger.debug(
                "[%s] No SkillRegistry provided — skipping skill registration",
                loaded.name,
            )
            return

        for skill_def in loaded.bundle.assets.skills:
            try:
                skill_id = f"plugin_{loaded.name}_{skill_def['name'].lower().replace(' ', '_')}"

                # Check if this skill already exists
                existing = None
                if hasattr(skill_registry, "get_skill"):
                    existing = skill_registry.get_skill(skill_id)

                if existing:
                    existing_source = getattr(existing, "source", "")

                    # Graduated skills are preserved — they've earned their place
                    if existing_source == "graduated":
                        logger.debug(
                            "[%s] Skill '%s' is graduated — preserving as-is",
                            loaded.name,
                            skill_def["name"],
                        )
                        loaded.skill_ids.append(skill_id)
                        continue

                    # Orphaned skills get re-activated with the plugin source
                    if existing_source.startswith("orphaned:"):
                        logger.info(
                            "[%s] Re-activating orphaned skill '%s'",
                            loaded.name,
                            skill_def["name"],
                        )
                        # Fall through to re-store with active source

                    # User-forked skills are never overwritten
                    elif existing_source not in ("", loaded.namespace):
                        logger.debug(
                            "[%s] Skill '%s' was forked by user — skipping overwrite",
                            loaded.name,
                            skill_def["name"],
                        )
                        continue

                # Build the skill and store it
                if hasattr(skill_registry, "_store"):
                    from hbllm.brain.skill_registry import Skill

                    skill = Skill(
                        skill_id=skill_id,
                        name=skill_def["name"],
                        description=skill_def.get("description", ""),
                        category=skill_def.get("category", "general"),
                        steps=skill_def.get("steps", []),
                        tools_used=skill_def.get("tools_used", []),
                        success_criteria=skill_def.get("success_criteria", ""),
                        examples=skill_def.get("examples", []),
                        tenant_id=loaded.namespace,
                        source=loaded.namespace,
                    )

                    # Preserve invocation metrics if re-activating
                    if existing and existing.invocations > 0:
                        skill.invocations = existing.invocations
                        skill.success_rate = existing.success_rate
                        skill.confidence_score = existing.confidence_score
                        skill.avg_latency_ms = existing.avg_latency_ms

                    skill_registry._store(skill)
                    loaded.skill_ids.append(skill_id)

                    logger.info(
                        "[%s] Registered skill: %s (%s)",
                        loaded.name,
                        skill_def["name"],
                        skill_id,
                    )

            except Exception as e:
                logger.error(
                    "[%s] Failed to register skill '%s': %s",
                    loaded.name,
                    skill_def.get("name", "?"),
                    e,
                )

    def _load_policies(self, loaded: LoadedBundle, policy_engine: Any) -> None:
        """Load governance policies into the PolicyEngine."""
        if not loaded.bundle.assets.has_policies:
            return

        if policy_engine is None:
            logger.debug(
                "[%s] No PolicyEngine provided — skipping policy loading",
                loaded.name,
            )
            return

        for policy_def in loaded.bundle.assets.policies:
            try:
                from hbllm.brain.policy_engine import (
                    Policy,
                    PolicyAction,
                    PolicyCondition,
                    PolicyType,
                )

                # Parse runtime conditions if present
                conditions = []
                for cond_data in policy_def.get("conditions", []):
                    try:
                        conditions.append(
                            PolicyCondition(
                                key=cond_data["key"],
                                operator=cond_data["operator"],
                                value=cond_data["value"],
                            )
                        )
                    except (KeyError, TypeError) as ce:
                        logger.warning(
                            "[%s] Invalid condition in policy '%s': %s",
                            loaded.name,
                            policy_def.get("name", "?"),
                            ce,
                        )

                policy = Policy(
                    name=policy_def["name"],
                    type=PolicyType(policy_def.get("type", "deny")),
                    action=PolicyAction(policy_def.get("action", "warn")),
                    description=policy_def.get("description", ""),
                    pattern=policy_def.get("pattern", ""),
                    content=policy_def.get("content", ""),
                    domains=policy_def.get("domains", []),
                    tenant_ids=policy_def.get("tenant_ids", ["*"]),
                    priority=policy_def.get("priority", -10),
                    enabled=policy_def.get("enabled", True),
                    severity=policy_def.get("severity", "medium"),
                    conditions=conditions,
                    source=policy_def.get("source", loaded.namespace),
                )

                policy_engine.add_policy(policy)
                loaded.policy_names.append(policy.name)

                logger.info(
                    "[%s] Loaded policy: %s (%s/%s)",
                    loaded.name,
                    policy.name,
                    policy.type,
                    policy.action,
                )

            except (KeyError, ValueError) as e:
                logger.error(
                    "[%s] Failed to load policy '%s': %s",
                    loaded.name,
                    policy_def.get("name", "?"),
                    e,
                )
            except Exception as e:
                logger.error(
                    "[%s] Unexpected error loading policy '%s': %s",
                    loaded.name,
                    policy_def.get("name", "?"),
                    e,
                )

    def _store_prompts(self, loaded: LoadedBundle) -> None:
        """Store prompt templates in the PromptStore."""
        if not loaded.bundle.assets.has_prompts:
            return

        for key, template in loaded.bundle.assets.prompts.items():
            try:
                self.prompt_store.add(key, template, source=loaded.name)
                loaded.prompt_keys.append(key)
            except Exception as e:
                logger.error(
                    "[%s] Failed to store prompt '%s': %s",
                    loaded.name,
                    key,
                    e,
                )

        if loaded.prompt_keys:
            logger.info(
                "[%s] Stored %d prompt templates",
                loaded.name,
                len(loaded.prompt_keys),
            )

    # ── Query ────────────────────────────────────────────────────────

    def list_bundles(self) -> list[dict[str, Any]]:
        """List all loaded bundles with their asset summaries."""
        return [loaded.to_dict() for loaded in self._bundles.values()]

    def get_bundle(self, name: str) -> LoadedBundle | None:
        """Get a loaded bundle by plugin name."""
        return self._bundles.get(name)

    def stats(self) -> dict[str, Any]:
        """Plugin manager statistics."""
        return {
            "bundles_loaded": self.bundle_count,
            "plugin_dirs": [str(d) for d in self._plugin_dirs],
            "prompts": self.prompt_store.count,
            "watcher_active": self._watch_task is not None,
        }
