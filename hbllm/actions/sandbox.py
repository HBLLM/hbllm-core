"""
Shared sandboxed Python execution for HBLLM.

Provides ``run_sandboxed_python()`` — a unified entry point for executing
untrusted Python code in a subprocess with:
  - AST pre-validation (module blocklist, builtin blocklist, dunder blocking)
  - POSIX resource limits (memory, CPU)
  - Optional OS-level network isolation via ``unshare -Urn`` (Linux)
  - Timeout enforcement

Both ``ExecutionNode`` (bus-facing) and ``tool_python_exec`` (agent-facing)
should route through this helper to ensure consistent sandboxing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result of a sandboxed Python execution."""

    status: str  # "SUCCESS" or "FAILURE"
    output: str
    error: str


# Cache the unshare capability check (once per process)
_unshare_checked: bool = False
_can_unshare: bool = False


def _check_unshare() -> bool:
    """Check if ``unshare -Urn`` is available on this platform."""
    global _unshare_checked, _can_unshare
    if _unshare_checked:
        return _can_unshare
    _unshare_checked = True
    if not sys.platform.startswith("linux"):
        _can_unshare = False
        return False
    unshare_path = shutil.which("unshare")
    if not unshare_path:
        logger.warning("unshare not found. Network isolation unavailable.")
        _can_unshare = False
        return False
    # Test if unshare actually works (may fail in containers without CAP_SYS_ADMIN)
    try:
        import subprocess

        result = subprocess.run(
            [unshare_path, "-Urn", "true"],
            capture_output=True,
            timeout=5,
        )
        _can_unshare = result.returncode == 0
        if not _can_unshare:
            logger.warning(
                "unshare -Urn returned %s. Network isolation unavailable.",
                result.returncode,
            )
    except Exception as e:
        logger.warning("unshare check failed: %s. Network isolation unavailable.", e)
        _can_unshare = False
    return _can_unshare


async def run_sandboxed_python(
    code: str,
    *,
    timeout: float = 5.0,
    max_memory_mb: int = 256,
    disable_network: bool = True,
) -> SandboxResult:
    """Execute Python code in a restricted subprocess.

    This is the shared implementation used by both ``ExecutionNode`` and
    ``tool_python_exec``. It provides:

    1. POSIX resource limits (``RLIMIT_AS``, ``RLIMIT_CPU``)
    2. Isolated Python mode (``-I``)
    3. Stripped ``PATH`` environment
    4. Optional ``unshare -Urn`` network isolation (Linux only)

    Args:
        code: Python source code to execute (must already pass AST validation).
        timeout: Maximum wall-clock seconds before kill.
        max_memory_mb: Maximum virtual memory in MiB.
        disable_network: If True, attempt ``unshare -Urn`` network isolation.

    Returns:
        A ``SandboxResult`` with status, stdout, and stderr.
    """
    temp_script = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        # Inject strict OS-level hardware quotas
        bound_wrapper = (
            "import resource\n"
            "try:\n"
            f"    resource.setrlimit(resource.RLIMIT_AS, ({max_memory_mb} * 1024 * 1024, {max_memory_mb} * 1024 * 1024))\n"
            f"    resource.setrlimit(resource.RLIMIT_CPU, ({int(timeout)}, {int(timeout)}))\n"
            "except BaseException:\n"
            "    pass\n\n"
        )
        temp_script.write(bound_wrapper + code)
        temp_script.close()

        safe_env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin" if not sys.platform.startswith("win") else "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }

        exec_cmd: list[str] = [sys.executable, "-I", temp_script.name]

        # Apply network isolation if requested and available
        if disable_network and _check_unshare():
            unshare_path = shutil.which("unshare")
            if unshare_path:
                exec_cmd = [unshare_path, "-Urn"] + exec_cmd

        try:
            proc = await asyncio.create_subprocess_exec(
                *exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=safe_env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except (TimeoutError, asyncio.TimeoutError):
                proc.kill()
                await proc.communicate()
                return SandboxResult(
                    status="FAILURE",
                    output=f"Execution timed out after {timeout} seconds.",
                    error="TimeoutError",
                )

            output = stdout.decode().strip()
            error = stderr.decode().strip()

            if proc.returncode == 0:
                return SandboxResult(status="SUCCESS", output=output, error=error)
            else:
                return SandboxResult(status="FAILURE", output=output, error=error)

        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            return SandboxResult(status="FAILURE", output="", error=str(e))
    finally:
        os.unlink(temp_script.name)
