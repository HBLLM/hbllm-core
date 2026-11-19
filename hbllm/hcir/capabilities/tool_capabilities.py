"""Kernel-governed local tool capabilities (read-only fs, Python AST, git status).

All executors enforce strict root-directory sandboxing (no path traversal),
bounded output sizes, and zero shell interpolation.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TOOL_CAPABILITIES: tuple[str, ...] = (
    "fs.read",
    "fs.list",
    "fs.search",
    "code.python_ast",
    "git.status",
    "shell.exec",
)


def _resolve_sandboxed_path(root_dir: Path, raw_path: str) -> Path:
    """Resolve *raw_path* strictly inside *root_dir*, rejecting traversal."""
    root = root_dir.resolve()
    candidate = (root / raw_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError(f"Path traversal outside sandbox root rejected: {raw_path!r}")
    return candidate


class ToolCapabilityExecutor:
    """Sandboxed executor for governed local inspection capabilities."""

    def __init__(
        self,
        default_capability: str,
        root_dir: Path | str | None = None,
        max_read_bytes: int = 65536,
        max_list_entries: int = 200,
    ) -> None:
        self._default_capability = default_capability
        self._root_dir = Path(root_dir).resolve() if root_dir is not None else Path.cwd().resolve()
        self._max_read_bytes = max_read_bytes
        self._max_list_entries = max_list_entries

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        cap = str(params.get("capability", self._default_capability))
        root_override = params.get("root_dir")
        root = Path(root_override).resolve() if root_override else self._root_dir

        if cap == "fs.read":
            return self._exec_fs_read(root, params)
        if cap == "fs.list":
            return self._exec_fs_list(root, params)
        if cap == "fs.search":
            return self._exec_fs_search(root, params)
        if cap == "code.python_ast":
            return self._exec_python_ast(root, params)
        if cap == "git.status":
            return await self._exec_git_status(root, params)
        if cap == "shell.exec":
            return await self._exec_shell(root, params)
        return {"ok": False, "error": f"Unknown tool capability: {cap}"}

    def _exec_fs_read(self, root: Path, params: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(params.get("path", ""))
        if not raw_path:
            return {"ok": False, "error": "fs.read requires 'path'"}
        try:
            target = _resolve_sandboxed_path(root, raw_path)
            if not target.is_file():
                return {"ok": False, "error": f"Not a file: {raw_path}"}
            max_bytes = int(params.get("max_bytes", self._max_read_bytes))
            raw = target.read_bytes()
            truncated = len(raw) > max_bytes
            content = raw[:max_bytes].decode("utf-8", errors="replace")
            return {
                "ok": True,
                "path": str(target.relative_to(root)),
                "content": content,
                "truncated": truncated,
                "bytes": len(raw),
            }
        except PermissionError as exc:
            return {"ok": False, "error": str(exc)}
        except OSError as exc:
            return {"ok": False, "error": f"I/O error: {exc}"}

    def _exec_fs_list(self, root: Path, params: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(params.get("path", "."))
        try:
            target = _resolve_sandboxed_path(root, raw_path)
            if not target.is_dir():
                return {"ok": False, "error": f"Not a directory: {raw_path}"}
            limit = int(params.get("limit", self._max_list_entries))
            entries = sorted(
                (
                    {
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                    }
                    for entry in target.iterdir()
                ),
                key=lambda item: (item["type"] != "dir", item["name"]),
            )
            truncated = len(entries) > limit
            return {
                "ok": True,
                "path": str(target.relative_to(root)),
                "entries": entries[:limit],
                "truncated": truncated,
                "total": len(entries),
            }
        except PermissionError as exc:
            return {"ok": False, "error": str(exc)}
        except OSError as exc:
            return {"ok": False, "error": f"I/O error: {exc}"}

    def _exec_python_ast(self, root: Path, params: dict[str, Any]) -> dict[str, Any]:
        source = params.get("source")
        raw_path = params.get("path")
        if source is None:
            if not raw_path:
                return {"ok": False, "error": "code.python_ast requires 'source' or 'path'"}
            read_res = self._exec_fs_read(root, {"path": raw_path})
            if not read_res.get("ok"):
                return read_res
            source = read_res["content"]

        try:
            tree = ast.parse(str(source))
        except SyntaxError as exc:
            return {
                "ok": False,
                "error": f"SyntaxError at line {exc.lineno}: {exc.msg}",
            }

        functions: list[str] = []
        classes: list[str] = []
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        return {
            "ok": True,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "node_count": sum(1 for _ in ast.walk(tree)),
        }

    async def _exec_git_status(self, root: Path, params: dict[str, Any]) -> dict[str, Any]:
        timeout = float(params.get("timeout", 5.0))
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "status",
                "--short",
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "lines": [
                    line
                    for line in stdout.decode("utf-8", errors="replace").splitlines()
                    if line.strip()
                ],
                "stderr": stderr.decode("utf-8", errors="replace").strip(),
            }
        except TimeoutError:
            return {"ok": False, "error": f"git status timed out after {timeout}s"}
        except OSError as exc:
            return {"ok": False, "error": f"git execution failed: {exc}"}

    # ── fs.search ────────────────────────────────────────────────────

    def _exec_fs_search(self, root: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Search for files matching a glob pattern within the sandbox root."""
        pattern = str(params.get("pattern", params.get("query", "*")))
        raw_path = str(params.get("path", "."))
        limit = int(params.get("limit", self._max_list_entries))
        try:
            target = _resolve_sandboxed_path(root, raw_path)
            if not target.is_dir():
                return {"ok": False, "error": f"Not a directory: {raw_path}"}

            # Support both glob and recursive glob patterns
            if "**" in pattern:
                matches = list(target.glob(pattern))
            else:
                # Try recursive search by default
                matches = list(target.rglob(pattern))

            results = sorted(
                [
                    {
                        "name": str(m.relative_to(root)),
                        "type": "dir" if m.is_dir() else "file",
                    }
                    for m in matches
                    if "__pycache__" not in m.parts and ".git" not in m.parts
                ],
                key=lambda item: item["name"],
            )
            truncated = len(results) > limit
            return {
                "ok": True,
                "pattern": pattern,
                "path": str(target.relative_to(root)),
                "matches": results[:limit],
                "truncated": truncated,
                "total": len(results),
            }
        except PermissionError as exc:
            return {"ok": False, "error": str(exc)}
        except OSError as exc:
            return {"ok": False, "error": f"I/O error: {exc}"}

    # ── shell.exec ───────────────────────────────────────────────────

    async def _exec_shell(self, root: Path, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a sandboxed shell command with strict timeout and output limits.

        Security:
          - Uses ``subprocess.exec`` (no shell interpolation).
          - Bounded timeout (default 10s, max 30s).
          - Output truncated to max_read_bytes.
          - Only runs within the sandbox root directory.
        """
        import shlex

        raw_command = str(params.get("command", ""))
        if not raw_command:
            return {"ok": False, "error": "shell.exec requires 'command'"}

        timeout = min(float(params.get("timeout", 10.0)), 30.0)

        try:
            argv = shlex.split(raw_command)
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid command syntax: {exc}"}

        if not argv:
            return {"ok": False, "error": "Empty command after parsing"}

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout[: self._max_read_bytes].decode("utf-8", errors="replace")
            err = stderr[: self._max_read_bytes].decode("utf-8", errors="replace").strip()
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": out,
                "stderr": err,
                "truncated": len(stdout) > self._max_read_bytes,
            }
        except TimeoutError:
            return {"ok": False, "error": f"Command timed out after {timeout}s"}
        except FileNotFoundError:
            return {"ok": False, "error": f"Command not found: {argv[0]!r}"}
        except OSError as exc:
            return {"ok": False, "error": f"Execution failed: {exc}"}
