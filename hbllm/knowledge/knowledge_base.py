"""
Knowledge Base — Connect files, folders, and code to HBLLM.

Manages source registration, file ingestion, chunking, and semantic
search powered by HBLLM's SemanticMemory vector database.

All data stays local in the configured data directory.

Usage::

    kb = KnowledgeBase()
    source_id = kb.add_source("/path/to/project", "folder")
    kb.ingest_source(source_id)
    results = kb.search("how does the auth system work?")
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Supported file extensions ────────────────────────────────────────────────

# ── Optional markitdown import ───────────────────────────────────────────────

try:
    from markitdown import MarkItDown

    _HAS_MARKITDOWN = True
except ImportError:
    _HAS_MARKITDOWN = False

# Base file types (always supported via plain read)
SUPPORTED_EXTENSIONS: dict[str, str] = {
    # Text
    ".txt": "text",
    ".md": "text",
    ".rst": "text",
    ".log": "text",
    # Code
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".rs": "code",
    ".go": "code",
    ".java": "code",
    ".cpp": "code",
    ".c": "code",
    ".h": "code",
    ".hpp": "code",
    ".rb": "code",
    ".php": "code",
    ".swift": "code",
    ".kt": "code",
    ".dart": "code",
    ".css": "code",
    ".html": "code",
    ".sql": "code",
    ".sh": "code",
    # Data
    ".json": "data",
    ".yaml": "data",
    ".yml": "data",
    ".toml": "data",
    ".csv": "data",
    ".xml": "data",
    ".env": "data",
}

# Rich formats supported via markitdown
MARKITDOWN_EXTENSIONS: dict[str, str] = {
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",
    ".pptx": "document",
    ".xlsx": "document",
    ".xls": "document",
    ".epub": "document",
    ".rtf": "document",
    ".ipynb": "document",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".bmp": "image",
    ".webp": "image",
    ".wav": "audio",
    ".mp3": "audio",
    ".zip": "archive",
}


def get_all_supported_extensions() -> dict[str, str]:
    """Return all supported extensions based on available libraries."""
    exts = dict(SUPPORTED_EXTENSIONS)
    if _HAS_MARKITDOWN:
        exts.update(MARKITDOWN_EXTENSIONS)
    return exts


# Directories to always skip
SKIP_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    ".idea",
    ".vscode",
    ".DS_Store",
    "vendor",
    "coverage",
    ".tox",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ── Source Model ─────────────────────────────────────────────────────────────


class Source:
    """A registered knowledge source (file or folder)."""

    def __init__(
        self,
        source_id: str,
        path: str,
        source_type: str,  # "file" | "folder"
        name: str,
        file_type: str = "",
        chunk_count: int = 0,
        file_count: int = 0,
        status: str = "pending",  # "pending" | "ingesting" | "ready" | "error"
        added_at: float = 0,
        last_synced: float = 0,
        error: str = "",
    ):
        self.source_id = source_id
        self.path = path
        self.source_type = source_type
        self.name = name
        self.file_type = file_type
        self.chunk_count = chunk_count
        self.file_count = file_count
        self.status = status
        self.added_at = added_at or time.time()
        self.last_synced = last_synced
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "source_type": self.source_type,
            "name": self.name,
            "file_type": self.file_type,
            "chunk_count": self.chunk_count,
            "file_count": self.file_count,
            "status": self.status,
            "added_at": self.added_at,
            "last_synced": self.last_synced,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Source:
        return cls(**d)


# ── Knowledge Base ───────────────────────────────────────────────────────────


class KnowledgeBase:
    """
    Manages knowledge sources and their vector embeddings.

    Uses HBLLM's SemanticMemory for the vector store and a JSON
    manifest for source tracking. All data persisted to disk.
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or os.path.expanduser("~/.hbllm/knowledge"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._sources: dict[str, Source] = {}
        self._chunk_map: dict[str, list[str]] = {}  # source_id -> [doc_ids]
        self._memory = None  # Lazy-loaded SemanticMemory
        self._markitdown = MarkItDown() if _HAS_MARKITDOWN else None

        self._load_manifest()

    @property
    def memory(self):
        """Lazy-load SemanticMemory to avoid import overhead."""
        if self._memory is None:
            try:
                from hbllm.memory.semantic import SemanticMemory

                vector_dir = self.data_dir / "vectors"
                if (vector_dir / "documents.json").exists():
                    self._memory = SemanticMemory.load_from_disk(str(vector_dir))
                else:
                    self._memory = SemanticMemory()
                logger.info("Knowledge base vector store initialized (%d docs)", self._memory.count)
            except ImportError:
                logger.warning("HBLLM SemanticMemory not available, using stub")
                self._memory = _StubMemory()
        return self._memory

    # ── Source Management ────────────────────────────────────────────────────

    def add_source(self, path: str, source_type: str = "auto") -> Source:
        """
        Register a new knowledge source.

        Args:
            path: Absolute path to file or folder
            source_type: "file", "folder", or "auto" (detect)

        Returns:
            The created Source object
        """
        abs_path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Path does not exist: {abs_path}")

        # Auto-detect type
        if source_type == "auto":
            source_type = "folder" if os.path.isdir(abs_path) else "file"

        # Check for duplicates
        for s in self._sources.values():
            if s.path == abs_path:
                return s

        name = os.path.basename(abs_path)
        file_type = ""
        all_exts = get_all_supported_extensions()
        if source_type == "file":
            ext = os.path.splitext(abs_path)[1].lower()
            file_type = all_exts.get(ext, "unknown")

        source = Source(
            source_id=str(uuid.uuid4())[:8],
            path=abs_path,
            source_type=source_type,
            name=name,
            file_type=file_type,
            status="pending",
        )

        self._sources[source.source_id] = source
        self._save_manifest()

        logger.info("Added knowledge source: %s (%s)", name, source_type)
        return source

    def remove_source(self, source_id: str) -> bool:
        """Remove a source and all its chunks from the vector store."""
        if source_id not in self._sources:
            return False

        # Remove chunks from vector store
        doc_ids = self._chunk_map.get(source_id, [])
        for doc_id in doc_ids:
            self.memory.delete(doc_id)

        self._chunk_map.pop(source_id, None)
        del self._sources[source_id]

        self._save_manifest()
        self._save_vectors()

        logger.info("Removed source %s and %d chunks", source_id, len(doc_ids))
        return True

    def list_sources(self) -> list[dict[str, Any]]:
        """Return all registered sources as dicts."""
        return [s.to_dict() for s in self._sources.values()]

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        total_chunks = sum(s.chunk_count for s in self._sources.values())
        total_files = sum(s.file_count for s in self._sources.values())
        return {
            "total_sources": len(self._sources),
            "total_chunks": total_chunks,
            "total_files": total_files,
            "ready_sources": sum(1 for s in self._sources.values() if s.status == "ready"),
            "data_dir": str(self.data_dir),
        }

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_source(self, source_id: str) -> int:
        """
        Ingest a source — chunk and embed all files.

        Returns:
            Number of chunks created
        """
        source = self._sources.get(source_id)
        if not source:
            raise ValueError(f"Source not found: {source_id}")

        source.status = "ingesting"
        source.error = ""
        self._save_manifest()

        try:
            # Clear old chunks for this source
            old_doc_ids = self._chunk_map.get(source_id, [])
            for doc_id in old_doc_ids:
                self.memory.delete(doc_id)

            # Collect files
            if source.source_type == "folder":
                files = self._collect_files(source.path)
            else:
                files = [source.path]

            # Ingest each file
            total_chunks = 0
            doc_ids = []

            for file_path in files:
                try:
                    chunks = self._read_and_chunk(file_path)
                    for chunk_text in chunks:
                        doc_id = self.memory.store(
                            content=chunk_text,
                            metadata={
                                "source_id": source_id,
                                "file_path": file_path,
                                "file_name": os.path.basename(file_path),
                            },
                        )
                        if doc_id:
                            doc_ids.append(doc_id)
                            total_chunks += 1
                except Exception as e:
                    logger.warning("Failed to ingest %s: %s", file_path, e)

            # Update source stats
            source.chunk_count = total_chunks
            source.file_count = len(files)
            source.status = "ready"
            source.last_synced = time.time()

            self._chunk_map[source_id] = doc_ids
            self._save_manifest()
            self._save_vectors()

            logger.info(
                "Ingested source '%s': %d files, %d chunks",
                source.name,
                len(files),
                total_chunks,
            )
            return total_chunks

        except Exception as e:
            source.status = "error"
            source.error = str(e)
            self._save_manifest()
            logger.error("Ingestion failed for %s: %s", source_id, e)
            raise

    def _collect_files(self, directory: str) -> list[str]:
        """Walk a directory and collect supported files."""
        all_exts = get_all_supported_extensions()
        files = []
        for root, dirs, filenames in os.walk(directory):
            # Filter out skip directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in all_exts:
                    full_path = os.path.join(root, fname)
                    # Skip large files
                    if os.path.getsize(full_path) <= MAX_FILE_SIZE:
                        files.append(full_path)
        return sorted(files)

    def _read_and_chunk(self, file_path: str) -> list[str]:
        """Read a file and split into chunks, using markitdown for rich formats."""
        ext = os.path.splitext(file_path)[1].lower()
        all_exts = get_all_supported_extensions()
        file_cat = all_exts.get(ext, "text")

        # Use markitdown for rich document formats (PDF, DOCX, images, etc.)
        if ext in MARKITDOWN_EXTENSIONS and self._markitdown is not None:
            try:
                result = self._markitdown.convert(file_path)
                content = result.text_content
                if not content or not content.strip():
                    return []
                logger.info("Converted %s via markitdown (%d chars)", file_path, len(content))
            except Exception as e:
                logger.warning("markitdown failed for %s: %s, skipping", file_path, e)
                return []
        else:
            # Plain text read for code/text/data files
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                return []

        if not content.strip():
            return []

        # Add file context header
        rel_name = os.path.basename(file_path)
        header = f"[File: {rel_name}]\n"

        if file_cat == "code":
            return self._chunk_code(header, content)
        else:
            return self._chunk_text(header, content)

    def _chunk_text(
        self, header: str, text: str, chunk_size: int = 512, overlap: int = 50
    ) -> list[str]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if len(words) <= chunk_size:
            return [header + text]

        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(header + chunk)
        return chunks

    def _chunk_code(
        self, header: str, code: str, max_lines: int = 80, overlap: int = 10
    ) -> list[str]:
        """Split code into line-based chunks preserving structure."""
        lines = code.split("\n")
        if len(lines) <= max_lines:
            return [header + code]

        chunks = []
        for i in range(0, len(lines), max_lines - overlap):
            chunk = "\n".join(lines[i : i + max_lines])
            if chunk.strip():
                chunks.append(header + chunk)
        return chunks

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Semantic search across all ingested sources.

        Returns:
            List of results with content, score, and source metadata
        """
        if not query.strip():
            return []

        results = self.memory.search(query, top_k=top_k)

        # Enrich results with source info
        enriched = []
        for r in results:
            meta = r.get("metadata", {})
            source_id = meta.get("source_id", "")
            source = self._sources.get(source_id)
            enriched.append(
                {
                    "content": r.get("content", ""),
                    "score": r.get("score", 0),
                    "file_name": meta.get("file_name", "unknown"),
                    "file_path": meta.get("file_path", ""),
                    "source_name": source.name if source else "unknown",
                    "source_id": source_id,
                }
            )

        return enriched

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_manifest(self):
        """Save sources and chunk map to disk."""
        manifest = {
            "sources": {sid: s.to_dict() for sid, s in self._sources.items()},
            "chunk_map": self._chunk_map,
        }
        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _load_manifest(self):
        """Load sources and chunk map from disk."""
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path) as f:
                data = json.load(f)

            for sid, sdata in data.get("sources", {}).items():
                self._sources[sid] = Source.from_dict(sdata)

            self._chunk_map = data.get("chunk_map", {})
            logger.info("Loaded %d knowledge sources from manifest", len(self._sources))
        except Exception as e:
            logger.warning("Failed to load manifest: %s", e)

    def _save_vectors(self):
        """Persist the vector store to disk."""
        try:
            vector_dir = self.data_dir / "vectors"
            self.memory.save_to_disk(str(vector_dir))
        except Exception as e:
            logger.warning("Failed to save vectors: %s", e)


# ── Stub Memory (when HBLLM SemanticMemory is not available) ─────────────────


class _StubMemory:
    """Minimal stub when SemanticMemory is not importable."""

    def __init__(self):
        self.count = 0
        self._docs: dict[str, dict] = {}

    def store(self, content, metadata=None, **kwargs):
        doc_id = str(uuid.uuid4())
        self._docs[doc_id] = {"content": content, "metadata": metadata or {}}
        self.count += 1
        return doc_id

    def search(self, query, top_k=5, **kwargs):
        # Simple keyword matching fallback
        results = []
        query_lower = query.lower()
        for doc_id, doc in self._docs.items():
            content = doc["content"].lower()
            score = sum(1 for w in query_lower.split() if w in content) / max(
                len(query_lower.split()), 1
            )
            if score > 0:
                results.append(
                    {"content": doc["content"], "score": score, "metadata": doc["metadata"]}
                )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete(self, doc_id):
        if doc_id in self._docs:
            del self._docs[doc_id]
            self.count -= 1
            return True
        return False

    def save_to_disk(self, path):
        import json

        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "documents.json", "w") as f:
            json.dump({"documents": self._docs, "ids": list(self._docs.keys())}, f)
