"""
Memory Migration Proxy — 5-phase migration from legacy stores to HCIR.

Wraps both the legacy MemoryNode and HCIRMemoryBackend, routing store/recall
operations through the appropriate backend(s) depending on migration phase:

    Phase 1 (READ_THROUGH):   Legacy authoritative. HCIR populated from legacy
                               results for warming on every recall.
    Phase 2 (DUAL_WRITE):     Writes go to both. Divergence is logged as warnings.
    Phase 3 (SHADOW_READ):    Both read; legacy returned. HCIR results compared
                               in background; divergence logged.
    Phase 4 (HCIR_PRIMARY):   HCIR authoritative. Legacy kept for rollback.
    Phase 5 (LEGACY_REMOVED): Legacy stores fully retired.

Usage::

    proxy = MemoryMigrationProxy(
        legacy=memory_node,
        hcir=hcir_backend,
        phase=MigrationPhase.DUAL_WRITE,
    )
    await proxy.store(MemoryType.EPISODIC, "User asked about weather", ...)
    results = await proxy.recall(MemoryType.EPISODIC, "weather", ...)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from hbllm.hcir.adapters.hcir_memory_backend import HCIRMemoryBackend, MigrationPhase
from hbllm.memory.interface import MemoryType

logger = logging.getLogger(__name__)


class MemoryMigrationProxy:
    """Routes memory operations through legacy and/or HCIR backends.

    Each phase progressively shifts authority from legacy to HCIR:

    - **READ_THROUGH**: Legacy handles all reads/writes. HCIR is populated
      from legacy recall results (cache warming).
    - **DUAL_WRITE**: Both backends receive writes. Reads come from legacy.
      HCIR writes are fire-and-forget; divergence is logged.
    - **SHADOW_READ**: Both backends are read in parallel. Legacy result is
      returned, but HCIR result is compared for divergence tracking.
    - **HCIR_PRIMARY**: HCIR handles all reads/writes. Legacy receives
      writes for rollback safety (fire-and-forget).
    - **LEGACY_REMOVED**: HCIR only. Legacy is not called at all.
    """

    def __init__(
        self,
        legacy: Any | None,
        hcir: HCIRMemoryBackend,
        phase: MigrationPhase = MigrationPhase.READ_THROUGH,
    ) -> None:
        self._legacy = legacy
        self._hcir = hcir
        self._phase = phase
        self._store_count: int = 0
        self._recall_count: int = 0
        self._divergences: int = 0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def phase(self) -> MigrationPhase:
        return self._phase

    @phase.setter
    def phase(self, value: MigrationPhase) -> None:
        old = self._phase
        self._phase = value
        self._hcir.migration_phase = value
        logger.info("[MigrationProxy] Phase changed: %s → %s", old, value)

    @property
    def divergence_count(self) -> int:
        return self._divergences

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "phase": self._phase.value,
            "stores": self._store_count,
            "recalls": self._recall_count,
            "divergences": self._divergences,
        }

    def advance_phase(self) -> MigrationPhase:
        """Advance to the next migration phase."""
        phases = list(MigrationPhase)
        idx = phases.index(self._phase)
        if idx < len(phases) - 1:
            self.phase = phases[idx + 1]
        return self._phase

    # ── Store (Write Path) ────────────────────────────────────────────

    async def store(
        self,
        memory_type: MemoryType,
        data: Any,
        *,
        tenant_id: str = "default",
        session_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Store memory entry via appropriate backend(s) for current phase."""
        self._store_count += 1

        if self._phase == MigrationPhase.READ_THROUGH:
            # Legacy only — HCIR not written
            return await self._store_legacy(
                memory_type, data, tenant_id=tenant_id, session_id=session_id, **kwargs
            )

        elif self._phase == MigrationPhase.DUAL_WRITE:
            # Write to both; legacy is authoritative
            legacy_result = await self._store_legacy(
                memory_type, data, tenant_id=tenant_id, session_id=session_id, **kwargs
            )
            # HCIR write is fire-and-forget
            try:
                await self._store_hcir(
                    memory_type, data, tenant_id=tenant_id, session_id=session_id, **kwargs
                )
            except Exception as exc:
                logger.warning("[MigrationProxy] HCIR dual-write failed: %s", exc)
            return legacy_result

        elif self._phase == MigrationPhase.SHADOW_READ:
            # Same as dual-write for stores
            legacy_result = await self._store_legacy(
                memory_type, data, tenant_id=tenant_id, session_id=session_id, **kwargs
            )
            try:
                await self._store_hcir(
                    memory_type, data, tenant_id=tenant_id, session_id=session_id, **kwargs
                )
            except Exception as exc:
                logger.warning("[MigrationProxy] HCIR shadow-write failed: %s", exc)
            return legacy_result

        elif self._phase == MigrationPhase.HCIR_PRIMARY:
            # HCIR authoritative; legacy for rollback
            hcir_result = await self._store_hcir(
                memory_type, data, tenant_id=tenant_id, session_id=session_id, **kwargs
            )
            try:
                await self._store_legacy(
                    memory_type, data, tenant_id=tenant_id, session_id=session_id, **kwargs
                )
            except Exception as exc:
                logger.debug("[MigrationProxy] Legacy rollback write failed: %s", exc)
            return hcir_result

        else:
            # LEGACY_REMOVED — HCIR only
            return await self._store_hcir(
                memory_type, data, tenant_id=tenant_id, session_id=session_id, **kwargs
            )

    # ── Recall (Read Path) ────────────────────────────────────────────

    async def recall(
        self,
        memory_type: MemoryType,
        query: str = "",
        *,
        tenant_id: str = "default",
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Recall memory entries via appropriate backend(s) for current phase."""
        self._recall_count += 1

        if self._phase == MigrationPhase.READ_THROUGH:
            # Legacy only, but populate HCIR from results (cache warming)
            results = await self._recall_legacy(
                memory_type, query, tenant_id=tenant_id, limit=limit, **kwargs
            )
            # Warm HCIR cache in background
            asyncio.ensure_future(self._warm_hcir(memory_type, results, tenant_id))
            return results

        elif self._phase == MigrationPhase.DUAL_WRITE:
            # Legacy read only
            return await self._recall_legacy(
                memory_type, query, tenant_id=tenant_id, limit=limit, **kwargs
            )

        elif self._phase == MigrationPhase.SHADOW_READ:
            # Read both in parallel; legacy is authoritative
            legacy_task = asyncio.ensure_future(
                self._recall_legacy(memory_type, query, tenant_id=tenant_id, limit=limit, **kwargs)
            )
            hcir_task = asyncio.ensure_future(
                self._recall_hcir(memory_type, query, tenant_id=tenant_id, limit=limit, **kwargs)
            )
            legacy_results = await legacy_task
            try:
                hcir_results = await hcir_task
                self._compare_results(memory_type.value, legacy_results, hcir_results)
            except Exception as exc:
                logger.warning("[MigrationProxy] HCIR shadow-read failed: %s", exc)
            return legacy_results

        elif self._phase == MigrationPhase.HCIR_PRIMARY:
            # HCIR authoritative
            return await self._recall_hcir(
                memory_type, query, tenant_id=tenant_id, limit=limit, **kwargs
            )

        else:
            # LEGACY_REMOVED
            return await self._recall_hcir(
                memory_type, query, tenant_id=tenant_id, limit=limit, **kwargs
            )

    # ── Cross-Memory Search ───────────────────────────────────────────

    async def search_all(
        self,
        query: str,
        tenant_id: str = "default",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Cross-memory-type search — uses HCIR when it's authoritative."""
        if self._hcir.is_hcir_authoritative:
            return await self._hcir.search_across_memory_types(
                query, tenant_id=tenant_id, limit=limit
            )
        # Legacy doesn't have cross-memory search natively
        return await self._hcir.search_across_memory_types(query, tenant_id=tenant_id, limit=limit)

    # ── Private: Legacy Store/Recall ──────────────────────────────────

    async def _store_legacy(self, memory_type: MemoryType, data: Any, **kwargs: Any) -> str:
        """Delegate store to legacy MemoryNode."""
        if self._legacy is None:
            return ""
        return await self._legacy.store(memory_type, data, **kwargs)

    async def _recall_legacy(
        self, memory_type: MemoryType, query: str = "", **kwargs: Any
    ) -> list[Any]:
        """Delegate recall to legacy MemoryNode."""
        if self._legacy is None:
            return []
        return await self._legacy.retrieve(memory_type, query, **kwargs)

    # ── Private: HCIR Store/Recall ────────────────────────────────────

    async def _store_hcir(self, memory_type: MemoryType, data: Any, **kwargs: Any) -> str:
        """Delegate store to HCIR backend."""
        tenant_id = kwargs.get("tenant_id", "default")
        session_id = kwargs.get("session_id", "")

        if memory_type == MemoryType.EPISODIC:
            return await self._hcir.store_episode(
                summary=str(data), tenant_id=tenant_id, session_id=session_id
            )
        elif memory_type == MemoryType.SEMANTIC:
            return await self._hcir.store_concept(label=str(data), tenant_id=tenant_id)
        elif memory_type == MemoryType.PROCEDURAL:
            return await self._hcir.store_skill(
                skill_name=kwargs.get("name", str(data)), description=str(data), tenant_id=tenant_id
            )
        elif memory_type == MemoryType.VALUE:
            return await self._hcir.store_value(
                dimension=kwargs.get("topic", "general"),
                weight=float(data) if isinstance(data, (int, float)) else 0.5,
                tenant_id=tenant_id,
            )
        elif memory_type == MemoryType.KNOWLEDGE_GRAPH:
            return await self._hcir.store_belief(claim=str(data), tenant_id=tenant_id)
        return ""

    async def _recall_hcir(
        self, memory_type: MemoryType, query: str = "", **kwargs: Any
    ) -> list[Any]:
        """Delegate recall to HCIR backend."""
        tenant_id = kwargs.get("tenant_id", "default")
        limit = kwargs.get("limit", 10)

        if memory_type == MemoryType.EPISODIC:
            return await self._hcir.recall_episodes(query=query, tenant_id=tenant_id, limit=limit)
        elif memory_type == MemoryType.SEMANTIC:
            return await self._hcir.recall_concepts(query=query, tenant_id=tenant_id, limit=limit)
        elif memory_type == MemoryType.PROCEDURAL:
            return await self._hcir.recall_skills(query=query, tenant_id=tenant_id, limit=limit)
        return []

    # ── Private: Cache Warming ────────────────────────────────────────

    async def _warm_hcir(self, memory_type: MemoryType, results: list[Any], tenant_id: str) -> None:
        """Populate HCIR from legacy recall results (read-through warming)."""
        try:
            for item in results[:5]:  # Cap warming to avoid overload
                if memory_type == MemoryType.EPISODIC and isinstance(item, dict):
                    summary = item.get("content") or item.get("summary") or str(item)
                    await self._hcir.store_episode(
                        summary=summary,
                        tenant_id=tenant_id,
                    )
                elif memory_type == MemoryType.SEMANTIC and isinstance(item, dict):
                    await self._hcir.store_concept(
                        label=item.get("content", str(item)),
                        tenant_id=tenant_id,
                    )
        except Exception as exc:
            logger.debug("[MigrationProxy] HCIR warming failed: %s", exc)

    # ── Private: Divergence Detection ─────────────────────────────────

    def _compare_results(
        self,
        memory_type: str,
        legacy_results: list[Any],
        hcir_results: list[Any],
    ) -> None:
        """Compare legacy and HCIR results, log divergence."""
        legacy_count = len(legacy_results)
        hcir_count = len(hcir_results)

        if legacy_count != hcir_count:
            self._divergences += 1
            self._hcir.record_divergence(memory_type, hcir_results, legacy_results)
            logger.warning(
                "[MigrationProxy] Divergence in %s: legacy=%d, hcir=%d",
                memory_type,
                legacy_count,
                hcir_count,
            )
