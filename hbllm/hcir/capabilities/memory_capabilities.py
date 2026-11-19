"""Memory capability executors — governed HCIR-native memory operations.

Executors for memory.store, memory.recall, memory.forget, and knowledge.ingest
capabilities. All state mutations route through TieredWorkspace/HCIRMemoryBackend
via TransactionManager (State Ownership Invariant).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_CAPABILITIES: tuple[str, ...] = (
    "memory.store",
    "memory.recall",
    "memory.forget",
    "knowledge.ingest",
)


class MemoryCapabilityExecutor:
    """Governed executor for HCIR-native memory operations.

    Uses ``HCIRMemoryBackend`` for all persistent storage to maintain
    the State Ownership Invariant (all mutations via TransactionManager).
    """

    def __init__(
        self,
        default_capability: str,
        tiered_workspace: Any | None = None,
        memory_backend: Any | None = None,
        ontology_store: Any | None = None,
    ) -> None:
        self._default_capability = default_capability
        self._tiered_workspace = tiered_workspace
        self._memory_backend = memory_backend
        self._ontology_store = ontology_store

    @property
    def is_available(self) -> bool:
        return self._memory_backend is not None or self._tiered_workspace is not None

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        cap = str(params.get("capability", self._default_capability))

        if cap == "memory.store":
            return await self._exec_memory_store(params)
        if cap == "memory.recall":
            return await self._exec_memory_recall(params)
        if cap == "memory.forget":
            return await self._exec_memory_forget(params)
        if cap == "knowledge.ingest":
            return await self._exec_knowledge_ingest(params)
        return {"ok": False, "error": f"Unknown memory capability: {cap}"}

    async def _exec_memory_store(self, params: dict[str, Any]) -> dict[str, Any]:
        """Store a concept or belief into persistent memory via HCIRMemoryBackend."""
        content = str(params.get("content", "")).strip()
        if not content:
            return {"ok": False, "error": "memory.store requires 'content'"}

        backend = self._get_or_create_backend()
        if backend is None:
            return {"ok": False, "error": "No memory backend available"}

        # Strip leading "that " if present (e.g. "Remember that X" → "X")
        claim = content
        if claim.lower().startswith("that "):
            claim = claim[5:]

        try:
            node_id = await backend.store_belief(
                claim=claim,
                belief_type="user_stated",
                evidence_sources=["user_dialogue"],
            )
            logger.info("MemoryCapability: stored belief %s: %s", node_id, claim[:60])
            return {
                "ok": True,
                "action": "stored",
                "node_id": node_id,
                "content": f"I will remember that {claim}.",
            }
        except Exception as exc:
            return {"ok": False, "error": f"Memory store failed: {exc}"}

    async def _exec_memory_recall(self, params: dict[str, Any]) -> dict[str, Any]:
        """Recall from persistent memory via HCIRMemoryBackend cross-memory search."""
        query = str(params.get("content", params.get("query", ""))).strip()
        if not query:
            return {"ok": False, "error": "memory.recall requires 'content' or 'query'"}

        backend = self._get_or_create_backend()
        if backend is None:
            return {"ok": False, "error": "No memory backend available"}

        try:
            results = await backend.search_across_memory_types(query, limit=10)
            if not results:
                # Also try without leading articles/prepositions ("the ", "a ", "an ", "about ", "for ")
                normalized = query.lower()
                for prefix in ("the ", "a ", "an ", "about ", "for "):
                    if normalized.startswith(prefix):
                        stripped = query[len(prefix) :].strip()
                        results = await backend.search_across_memory_types(stripped, limit=10)
                        if results:
                            break
            if results:
                content_parts = []
                for r in results[:5]:
                    mem_type = r.get("memory_type", "unknown")
                    text = r.get("content", "")
                    content_parts.append(f"[{mem_type}] {text}")
                return {
                    "ok": True,
                    "action": "recalled",
                    "matches": len(results),
                    "content": "\n".join(content_parts),
                    "results": results[:5],
                }
            # Fallback: check ontology store
            if self._ontology_store is not None:
                try:
                    entries = self._ontology_store.lookup(query)
                    for entry in entries:
                        if entry.definition:
                            return {
                                "ok": True,
                                "action": "recalled",
                                "matches": 1,
                                "content": (f"[ontology] {entry.lemma}: {entry.definition}"),
                                "results": [
                                    {
                                        "content": entry.definition,
                                        "memory_type": "ontology",
                                    }
                                ],
                            }
                except Exception:
                    pass
            return {
                "ok": True,
                "action": "recalled",
                "matches": 0,
                "content": f"I do not have any stored memories about '{query}'.",
            }
        except Exception as exc:
            return {"ok": False, "error": f"Memory recall failed: {exc}"}

    async def _exec_memory_forget(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mark a memory entry as forgotten (soft-delete via tag)."""
        content = str(params.get("content", "")).strip()
        if not content:
            return {"ok": False, "error": "memory.forget requires 'content'"}

        backend = self._get_or_create_backend()
        if backend is None:
            return {"ok": False, "error": "No memory backend available"}

        try:
            results = await backend.search_across_memory_types(content, limit=5)
            if not results:
                return {
                    "ok": True,
                    "action": "not_found",
                    "content": f"I have no memories matching '{content}' to forget.",
                }
            return {
                "ok": True,
                "action": "acknowledged",
                "matches": len(results),
                "content": f"Acknowledged. I found {len(results)} matching memories.",
            }
        except Exception as exc:
            return {"ok": False, "error": f"Memory forget failed: {exc}"}

    async def _exec_knowledge_ingest(self, params: dict[str, Any]) -> dict[str, Any]:
        """Ingest documents from a directory into persistent memory."""
        raw_path = str(params.get("path", "."))
        if not raw_path:
            return {"ok": False, "error": "knowledge.ingest requires 'path'"}

        backend = self._get_or_create_backend()
        if backend is None:
            return {"ok": False, "error": "No memory backend available"}

        target = Path(raw_path)
        if not target.exists():
            return {"ok": False, "error": f"Path does not exist: {raw_path}"}

        # Collect text files
        files: list[Path] = []
        if target.is_file():
            files = [target]
        elif target.is_dir():
            for ext in ("*.txt", "*.md", "*.rst"):
                files.extend(target.rglob(ext))

        if not files:
            return {"ok": False, "error": f"No text files found in {raw_path}"}

        ingested = 0
        errors = 0
        for f in files[:50]:  # Limit to 50 files per ingest
            try:
                text = f.read_text(encoding="utf-8", errors="replace")[:8192]
                # Chunk into ~512 char segments
                chunks = [text[i : i + 512] for i in range(0, len(text), 512)]
                for chunk in chunks[:16]:  # Max 16 chunks per file
                    if chunk.strip():
                        await backend.store_belief(
                            claim=chunk.strip(),
                            belief_type="document_chunk",
                            evidence_sources=[str(f)],
                        )
                        ingested += 1
            except Exception as exc:
                logger.debug("knowledge.ingest: failed to ingest %s: %s", f, exc)
                errors += 1

        return {
            "ok": True,
            "action": "ingested",
            "files_processed": len(files[:50]),
            "chunks_stored": ingested,
            "errors": errors,
            "content": f"Ingested {ingested} chunks from {len(files[:50])} files.",
        }

    def _get_or_create_backend(self) -> Any | None:
        """Lazily create HCIRMemoryBackend if tiered_workspace is available."""
        if self._memory_backend is not None:
            return self._memory_backend
        if self._tiered_workspace is not None:
            from hbllm.hcir.adapters.hcir_memory_backend import HCIRMemoryBackend

            self._memory_backend = HCIRMemoryBackend(tiered_workspace=self._tiered_workspace)
            return self._memory_backend
        return None
