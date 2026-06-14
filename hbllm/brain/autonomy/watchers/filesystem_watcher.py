"""Filesystem Watcher — detects file changes in watched directories.

Uses ``os.scandir`` + mtime tracking (no ``watchdog`` dependency).
Registers as an AutonomyCore proactive handler.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """A detected file system change."""

    path: str
    change_type: str  # "created", "modified", "deleted"
    size: int = 0
    mtime: float = 0.0


class FilesystemWatcher:
    """Proactive handler that monitors directories for file changes.

    Uses mtime-based polling (no external dependencies). On each tick,
    compares current file mtimes with the previous snapshot and reports
    changes.

    Usage::

        watcher = FilesystemWatcher(watch_dirs=["/path/to/project"])
        autonomy_core.add_proactive_handler("fs_watcher", watcher.check)
    """

    def __init__(
        self,
        watch_dirs: list[str | Path] | None = None,
        ignore_patterns: list[str] | None = None,
        max_depth: int = 3,
        min_change_interval: float = 5.0,
    ) -> None:
        self.watch_dirs = [Path(d) for d in (watch_dirs or [])]
        self.ignore_patterns = ignore_patterns or [
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "*.pyc",
            ".DS_Store",
        ]
        self.max_depth = max_depth
        self.min_change_interval = min_change_interval

        # State: path → (mtime, size)
        self._snapshot: dict[str, tuple[float, int]] = {}
        self._last_check: float = 0.0
        self._initialized = False

    def _should_ignore(self, path: str) -> bool:
        """Check if a path matches any ignore pattern."""
        basename = os.path.basename(path)
        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                if basename.endswith(pattern[1:]):
                    return True
            elif pattern in path.split(os.sep):
                return True
        return False

    def _scan_directory(self, root: Path, depth: int = 0) -> dict[str, tuple[float, int]]:
        """Recursively scan a directory and return path → (mtime, size)."""
        result: dict[str, tuple[float, int]] = {}
        if depth > self.max_depth:
            return result

        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if self._should_ignore(entry.path):
                        continue
                    try:
                        if entry.is_file(follow_symlinks=False):
                            stat = entry.stat(follow_symlinks=False)
                            result[entry.path] = (stat.st_mtime, stat.st_size)
                        elif entry.is_dir(follow_symlinks=False):
                            result.update(self._scan_directory(Path(entry.path), depth + 1))
                    except (PermissionError, OSError):
                        continue
        except (PermissionError, OSError):
            pass

        return result

    def _detect_changes(self, current: dict[str, tuple[float, int]]) -> list[FileChange]:
        """Compare current snapshot with previous and detect changes."""
        changes: list[FileChange] = []

        # New or modified files
        for path, (mtime, size) in current.items():
            if path not in self._snapshot:
                changes.append(FileChange(path=path, change_type="created", size=size, mtime=mtime))
            elif self._snapshot[path][0] < mtime:
                changes.append(
                    FileChange(path=path, change_type="modified", size=size, mtime=mtime)
                )

        # Deleted files
        for path in self._snapshot:
            if path not in current:
                changes.append(FileChange(path=path, change_type="deleted"))

        return changes

    async def check(self) -> list[Message] | None:
        """Proactive handler callback — check for file changes.

        Returns a list of Messages for each detected change, or None
        if no changes occurred.
        """
        now = time.monotonic()
        if now - self._last_check < self.min_change_interval:
            return None
        self._last_check = now

        if not self.watch_dirs:
            return None

        # Scan all watched directories
        current: dict[str, tuple[float, int]] = {}
        for watch_dir in self.watch_dirs:
            if watch_dir.exists():
                current.update(self._scan_directory(watch_dir))

        if not self._initialized:
            # First scan — just record baseline, don't report
            self._snapshot = current
            self._initialized = True
            logger.info(
                "[FilesystemWatcher] Initialized with %d files in %d directories",
                len(current),
                len(self.watch_dirs),
            )
            return None

        changes = self._detect_changes(current)
        self._snapshot = current

        if not changes:
            return None

        # Group changes into a summary message
        messages: list[Message] = []
        created = [c for c in changes if c.change_type == "created"]
        modified = [c for c in changes if c.change_type == "modified"]
        deleted = [c for c in changes if c.change_type == "deleted"]

        summary_parts = []
        if created:
            summary_parts.append(f"{len(created)} created")
        if modified:
            summary_parts.append(f"{len(modified)} modified")
        if deleted:
            summary_parts.append(f"{len(deleted)} deleted")

        messages.append(
            Message(
                type=MessageType.EVENT,
                source_node_id="autonomy.watcher.filesystem",
                topic="perception.filesystem.changes",
                payload={
                    "summary": ", ".join(summary_parts),
                    "created": [{"path": c.path, "size": c.size} for c in created[:10]],
                    "modified": [{"path": c.path, "size": c.size} for c in modified[:10]],
                    "deleted": [{"path": c.path} for c in deleted[:10]],
                    "total_changes": len(changes),
                    "_urgency": 0.3,
                },
            )
        )

        logger.info(
            "[FilesystemWatcher] Detected %d changes: %s",
            len(changes),
            ", ".join(summary_parts),
        )
        return messages

    def snapshot(self) -> dict[str, Any]:
        """Introspection snapshot."""
        return {
            "watch_dirs": [str(d) for d in self.watch_dirs],
            "tracked_files": len(self._snapshot),
            "initialized": self._initialized,
        }
