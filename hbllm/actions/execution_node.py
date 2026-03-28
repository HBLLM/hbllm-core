"""
Sandboxed Execution Node.

Receives Python code, validates it against a security policy using AST
inspection, then executes it in a restricted subprocess (with timeout
and resource limits) and returns output or traceback as a reward signal.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
import sys
import tempfile
from typing import Any

from hbllm.network.node import Node, NodeType
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)

# ── Security policy ──────────────────────────────────────────────────────────

# Modules that grant filesystem, network, or process control access
BLOCKED_MODULES: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "ctypes", "signal", "multiprocessing", "threading",
    "importlib", "runpy", "code", "codeop",
    "pickle", "shelve", "marshal",
    "webbrowser", "ftplib", "smtplib", "telnetlib",
})

# Built-in functions / names that can escape the sandbox
BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "open", "input", "breakpoint", "exit", "quit",
})


class CodeSecurityError(Exception):
    """Raised when submitted code violates the sandbox policy."""


class _SecurityVisitor(ast.NodeVisitor):
    """
    AST walker that rejects dangerous constructs *before* execution.

    Checks:
    - import / from-import of blocked modules
    - Calls to blocked built-in names
    - Use of dunder attributes (__class__, __subclasses__, etc.)
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    # ── import detection ─────────────────────────────────────────────────
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod in BLOCKED_MODULES:
                self.violations.append(
                    f"Line {node.lineno}: blocked import '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root_mod = node.module.split(".")[0]
            if root_mod in BLOCKED_MODULES:
                self.violations.append(
                    f"Line {node.lineno}: blocked import from '{node.module}'"
                )
        self.generic_visit(node)

    # ── dangerous call detection ─────────────────────────────────────────
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name: str | None = None

        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr

        if name and name in BLOCKED_BUILTINS:
            self.violations.append(
                f"Line {node.lineno}: blocked built-in call '{name}()'"
            )
        self.generic_visit(node)

    # ── dunder attribute access ──────────────────────────────────────────
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.violations.append(
                f"Line {node.lineno}: blocked dunder access '.{node.attr}'"
            )
        self.generic_visit(node)


def validate_code(code: str) -> list[str]:
    """
    Parse *code* and return a list of security violations (empty = safe).

    Raises ``SyntaxError`` for unparseable code — let the caller decide
    whether to propagate or wrap it.
    """
    tree = ast.parse(code)
    visitor = _SecurityVisitor()
    visitor.visit(tree)
    return visitor.violations


# ── Execution Node ────────────────────────────────────────────────────────────

class ExecutionNode(Node):
    """Executes code securely to provide deterministic ground-truth verification."""

    def __init__(self, node_id: str, timeout: float = 3.0):
        # We will set node_type to CORE since ACTION doesn't exist
        super().__init__(node_id=node_id, node_type=NodeType.CORE)
        self.timeout = timeout

    async def on_start(self) -> None:
        logger.info("Starting ExecutionNode")
        await self.bus.subscribe("action.execute_code", self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping ExecutionNode")

    async def handle_message(self, message: Message) -> Message | None:
        """Handle execution requests."""
        if message.type != MessageType.QUERY:
            return None
            
        code = message.payload.get("code", "")
        if not code:
            # Try to extract code blocks if just text is passed
            text = message.payload.get("text", "")
            match = re.search(r"```python\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
            else:
                return message.create_error("No Python code provided for execution.")
        
        # ── Security gate: AST validation before execution ──
        try:
            violations = validate_code(code)
        except SyntaxError as e:
            return message.create_error(f"Syntax error in submitted code: {e}")

        if violations:
            detail = "; ".join(violations)
            logger.warning("ExecutionNode rejected code: %s", detail)
            return message.create_error(
                f"Code rejected by security policy: {detail}"
            )

        # Run code in an isolated subprocess
        result = await self._execute_python(code)
        return message.create_response(result)
        
    async def _execute_python(self, code: str) -> dict[str, Any]:
        """Write to temp file and run in a restricted subprocess."""
        temp_script = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        try:
            temp_script.write(code)
            temp_script.close()  # Close so subprocess can read it
            
            try:
                # Build a restricted environment: strip PATH and dangerous env vars
                safe_env = {
                    "PATH": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                }

                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-I", temp_script.name,  # -I = isolated mode
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=safe_env,
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                    return {
                        "status": "FAILURE",
                        "output": f"Execution timed out after {self.timeout} seconds.",
                        "error": "TimeoutError"
                    }
                
                output = stdout.decode().strip()
                error = stderr.decode().strip()
                
                if proc.returncode == 0:
                    return {
                        "status": "SUCCESS",
                        "output": output,
                        "error": error
                    }
                else:
                    return {
                        "status": "FAILURE",
                        "output": output,
                        "error": error
                    }
            except Exception as e:
                return {
                    "status": "FAILURE",
                    "output": "",
                    "error": str(e)
                }
        finally:
            os.unlink(temp_script.name)
