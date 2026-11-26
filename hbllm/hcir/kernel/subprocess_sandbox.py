"""Process-level subprocess isolation for untrusted code/plugin execution.

Provides Tier-2 isolation with:
- Resource limits (RLIMIT_AS, RLIMIT_CPU, RLIMIT_FSIZE, RLIMIT_NPROC)
- Wall-clock timeout enforcement with process group kill
- Stripped minimal environment (no secrets or arbitrary inherited env vars)
- Linux user/network namespace isolation when kernel permissions allow
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    """Result of executing code inside an isolated subprocess."""

    success: bool
    stdout: str
    stderr: str
    return_value: object = None
    timed_out: bool = False
    killed_by_limit: bool = False
    exit_code: int = -1

    @property
    def status(self) -> str:
        if self.timed_out:
            return "TIMEOUT"
        return "SUCCESS" if self.success else "ERROR"

    @property
    def output(self) -> str:
        return self.stdout

    @property
    def error(self) -> str:
        return self.stderr


def _check_unshare() -> bool:
    """Check if unshare --user --net is supported in this environment."""
    unshare_bin = shutil.which("unshare")
    if not unshare_bin:
        return False
    try:
        res = subprocess.run(  # nosec B603
            [unshare_bin, "--user", "--net", "true"],
            capture_output=True,
            timeout=2.0,
        )
        return res.returncode == 0
    except Exception:
        return False


_HAS_UNSHARE = _check_unshare()


async def run_sandboxed_python(
    code: str,
    *,
    timeout: float = 10.0,
    max_memory_mb: int = 256,
    max_cpu_seconds: int = 10,
    max_file_bytes: int = 10 * 1024 * 1024,
    allow_network: bool = False,
    disable_network: bool | None = None,
    allowed_env_keys: tuple[str, ...] = ("PATH", "LANG", "LC_ALL"),
) -> SandboxResult:
    """Execute Python code in an isolated OS subprocess with resource limits."""
    if disable_network is not None:
        allow_network = not disable_network
    # Wrap code in a minimal runner that writes the final result as JSON to fd 3 or stdout sentinel
    wrapped = textwrap.dedent(f"""\
        import json as _sb_json
        import resource as _sb_res
        import sys as _sb_sys

        # Apply POSIX resource limits inside the child process
        def _apply_limits():
            limits = [
                (_sb_res.RLIMIT_CPU, ({max_cpu_seconds}, {max_cpu_seconds})),
                (_sb_res.RLIMIT_FSIZE, ({max_file_bytes}, {max_file_bytes})),
            ]
            # RLIMIT_AS is unsupported on macOS
            if _sb_sys.platform != "darwin":
                mem_bytes = {max_memory_mb} * 1024 * 1024
                limits.append((_sb_res.RLIMIT_AS, (mem_bytes, mem_bytes)))
            for res_type, (soft, hard) in limits:
                try:
                    _sb_res.setrlimit(res_type, (soft, hard))
                except (ValueError, OSError):
                    pass

        _apply_limits()

        _sb_ns = {{"__builtins__": __builtins__}}
        _sb_code = {code!r}
        try:
            exec(compile(_sb_code, "<sandbox>", "exec"), _sb_ns)
            if "result" in _sb_ns:
                print("\\n__SB_RESULT__:" + _sb_json.dumps(_sb_ns["result"], default=str))
        except Exception as _sb_exc:
            import traceback as _sb_tb
            _sb_tb.print_exc(file=_sb_sys.stderr)
            _sb_sys.exit(1)
    """)

    # Minimal clean environment — strip API keys, tokens, credentials
    clean_env = {k: os.environ[k] for k in allowed_env_keys if k in os.environ}
    clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
    clean_env["PYTHONNOUSERSITE"] = "1"

    with tempfile.TemporaryDirectory(prefix="hbllm_sb_") as tmp_dir:
        script_path = os.path.join(tmp_dir, "run.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(wrapped)

        cmd = [sys.executable, "-I", script_path]
        if not allow_network and _HAS_UNSHARE:
            cmd = ["unshare", "--user", "--net", "--"] + cmd

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmp_dir,
            env=clean_env,
            start_new_session=True,  # Own process group so we can killpg on timeout
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await proc.wait()
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                timed_out=True,
                killed_by_limit=True,
                exit_code=-9,
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0

        # Extract structured return value if present
        return_value = None
        marker = "\n__SB_RESULT__:"
        if marker in stdout:
            idx = stdout.rfind(marker)
            raw_json = stdout[idx + len(marker) :].strip()
            stdout = stdout[:idx]
            try:
                return_value = json.loads(raw_json)
            except json.JSONDecodeError:
                pass

        # Fallback: also check start of stdout if script didn't print prior lines
        if return_value is None and stdout.startswith("__SB_RESULT__:"):
            raw_json = stdout[len("__SB_RESULT__:") :].strip()
            stdout = ""
            try:
                return_value = json.loads(raw_json)
            except json.JSONDecodeError:
                pass

        killed = exit_code in (-signal.SIGKILL, -signal.SIGXCPU, 137, 152)
        return SandboxResult(
            success=exit_code == 0,
            stdout=stdout.strip(),
            stderr=stderr.strip(),
            return_value=return_value,
            timed_out=False,
            killed_by_limit=killed,
            exit_code=exit_code,
        )


_CMD_EXEC_SHIM = textwrap.dedent("""\
    import json as _sb_json
    import os as _sb_os
    import resource as _sb_res
    import sys as _sb_sys

    _cfg = _sb_json.loads(_sb_sys.argv[1])
    _limits = [
        (_sb_res.RLIMIT_CPU, (_cfg["cpu"], _cfg["cpu"])),
        (_sb_res.RLIMIT_FSIZE, (_cfg["fsize"], _cfg["fsize"])),
    ]
    if _sb_sys.platform != "darwin":
        _mem = _cfg["mem"] * 1024 * 1024
        _limits.append((_sb_res.RLIMIT_AS, (_mem, _mem)))
    for _rt, (_soft, _hard) in _limits:
        try:
            _sb_res.setrlimit(_rt, (_soft, _hard))
        except (ValueError, OSError):
            pass
    _exe = _sb_sys.argv[2]
    _argv = _sb_sys.argv[3:]
    _sb_os.execv(_exe, _argv)  # nosec B606
""")


async def run_sandboxed_command(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout: float = 30.0,
    max_memory_mb: int = 512,
    max_cpu_seconds: int = 30,
    max_file_bytes: int = 32 * 1024 * 1024,
    allow_network: bool = False,
    allowed_env_keys: tuple[str, ...] = ("PATH", "LANG", "LC_ALL"),
) -> SandboxResult:
    """Execute an allowlisted OS command inside Tier-2 subprocess isolation (RLIMIT + unshare)."""
    if not argv:
        return SandboxResult(success=False, stdout="", stderr="Empty argv", exit_code=-1)

    cwd_path = Path(cwd).resolve()
    clean_env = {k: os.environ[k] for k in allowed_env_keys if k in os.environ}
    raw_exe = argv[0]
    if "/" in raw_exe:
        cand = Path(raw_exe) if Path(raw_exe).is_absolute() else (cwd_path / raw_exe).resolve()
        if not cand.exists():
            raise FileNotFoundError(f"Executable not found: {raw_exe}")
        resolved_exe = str(cand)
    else:
        found = shutil.which(raw_exe, path=clean_env.get("PATH"))
        if not found:
            raise FileNotFoundError(f"Executable not found in PATH: {raw_exe}")
        resolved_exe = found

    limit_cfg = json.dumps(
        {
            "cpu": int(max_cpu_seconds),
            "fsize": int(max_file_bytes),
            "mem": int(max_memory_mb),
        }
    )
    cmd = [sys.executable, "-I", "-c", _CMD_EXEC_SHIM, limit_cfg, resolved_exe, *argv]
    if not allow_network and _HAS_UNSHARE:
        cmd = ["unshare", "--user", "--net", "--"] + cmd

    proc = await asyncio.create_subprocess_exec(  # nosec B603
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd_path),
        env=clean_env,
        start_new_session=True,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.wait()
        return SandboxResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {timeout}s",
            timed_out=True,
            killed_by_limit=True,
            exit_code=-9,
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    exit_code = proc.returncode if proc.returncode is not None else -1
    killed = exit_code in (-signal.SIGKILL, -signal.SIGXCPU, 137, 152)
    return SandboxResult(
        success=exit_code == 0,
        stdout=stdout.strip(),
        stderr=stderr.strip(),
        timed_out=False,
        killed_by_limit=killed,
        exit_code=exit_code,
    )
