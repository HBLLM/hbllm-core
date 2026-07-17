"""
HBLLM Core Constants — shared values used across subsystems.

Centralises constants that were previously duplicated in multiple modules.
"""

from __future__ import annotations

# ── Directory Skip List ──────────────────────────────────────────────────────
# Directories to always skip during file-system traversal (ingestion, extraction).
# Previously duplicated in knowledge_base.py, extractor.py (×2).

SKIP_DIRS: frozenset[str] = frozenset(
    {
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
)
