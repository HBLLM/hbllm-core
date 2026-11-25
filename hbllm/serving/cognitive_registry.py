"""Per-tenant CognitiveSystem registry for HTTP/serving endpoints.

Provides thread-safe, tenant-isolated Pure-HCIR runtimes constructed via the
single canonical composition root (`build_cognitive_system`) and executed off
the asyncio event loop via `run_in_threadpool`.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

_UNSET = object()
_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _sanitize_tenant_slug(tenant_id: str) -> str:
    """Return a path-safe, collision-resistant directory/world slug for *tenant_id*."""
    cleaned = _SAFE_SLUG_RE.sub("_", tenant_id.strip()).strip("_") or "tenant"
    digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:32]}_{digest}"


class CognitiveSystemRegistry:
    """Thread-safe LRU registry of per-tenant `CognitiveSystem` runtimes."""

    def __init__(
        self,
        base_memory_dir: Path | str | None | object = _UNSET,
        *,
        max_tenants: int = 64,
        enable_grounding: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._systems: OrderedDict[str, Any] = OrderedDict()
        self._max_tenants = max(1, max_tenants)
        self._enable_grounding = enable_grounding
        self.hcir_fallback_count: int = 0

        if base_memory_dir is _UNSET:
            env_dir = os.environ.get("HBLLM_TENANT_MEMORY_DIR", "").strip()
            if env_dir == ":memory:":
                self._base_dir: Path | None = None
            elif env_dir:
                self._base_dir = Path(env_dir).resolve()
            else:
                self._base_dir = Path.home() / ".hbllm" / "tenants"
        elif base_memory_dir is None or str(base_memory_dir) == ":memory:":
            self._base_dir = None
        else:
            self._base_dir = Path(str(base_memory_dir)).resolve()

    def get_or_create(self, tenant_id: str) -> Any:
        """Return the isolated `CognitiveSystem` bound to *tenant_id*."""
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string")

        with self._lock:
            existing = self._systems.get(tenant_id)
            if existing is not None:
                self._systems.move_to_end(tenant_id)
                return existing

            from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

            slug = _sanitize_tenant_slug(tenant_id)
            mem_dir = (self._base_dir / slug) if self._base_dir is not None else None
            cfg = CognitiveSystemConfig(
                world_id=f"tenant_{slug}",
                memory_dir=mem_dir,
                tenant_id=tenant_id,
                enable_logging=False,
                enable_grounding=self._enable_grounding,
                enable_brain_llm=False,
            )
            system = build_cognitive_system(cfg)
            self._systems[tenant_id] = system
            while len(self._systems) > self._max_tenants:
                self._systems.popitem(last=False)
            return system

    async def process_dialogue(
        self,
        tenant_id: str,
        text: str,
        *,
        speaker_id: str = "human",
    ) -> Any:
        """Execute synchronous HCIR dialogue turn in FastAPI's worker threadpool."""
        system = self.get_or_create(tenant_id)
        return await run_in_threadpool(system.process_dialogue, text, speaker_id=speaker_id)

    def record_fallback(self) -> int:
        """Increment and return the count of Pure-HCIR fast-path exceptions."""
        with self._lock:
            self.hcir_fallback_count += 1
            return self.hcir_fallback_count

    def clear(self) -> None:
        """Drop all cached per-tenant runtimes (primarily for test isolation)."""
        with self._lock:
            self._systems.clear()
            self.hcir_fallback_count = 0


_default_registry_lock = threading.Lock()
_default_registry: CognitiveSystemRegistry | None = None


def get_cognitive_registry() -> CognitiveSystemRegistry:
    """Return the process-wide default `CognitiveSystemRegistry`."""
    global _default_registry
    with _default_registry_lock:
        if _default_registry is None:
            _default_registry = CognitiveSystemRegistry()
        return _default_registry
