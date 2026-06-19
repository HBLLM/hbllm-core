"""
Built-in Tools — standard tool implementations for HBLLM.

Provides sandboxed Python execution, safe shell commands, file I/O,
and web search as reusable tool functions.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

from hbllm.actions.execution_node import validate_code
from hbllm.actions.tool_registry import ToolResult

logger = logging.getLogger(__name__)


# ── Safe Shell Commands ────────────────────────────────────────────────────────

SAFE_SHELL_COMMANDS: frozenset[str] = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "find",
        "echo",
        "pwd",
        "date",
        "whoami",
        "uname",
        "df",
        "du",
        "sort",
        "uniq",
        "tr",
        "cut",
        "awk",
        "sed",
        "diff",
        "tree",
        "file",
        "which",
    }
)


# ── Python Execution ──────────────────────────────────────────────────────────


async def tool_python_exec(code: str) -> ToolResult:
    """Execute Python code in a sandboxed subprocess."""
    try:
        violations = validate_code(code)
    except SyntaxError as e:
        return ToolResult(tool="python_exec", success=False, output="", error=f"Syntax error: {e}")

    if violations:
        return ToolResult(
            tool="python_exec",
            success=False,
            output="",
            error=f"Security policy violation: {'; '.join(violations)}",
        )

    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        wrapper = (
            "import resource\n"
            "try:\n"
            "    resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))\n"
            "    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))\n"
            "except BaseException:\n"
            "    pass\n\n"
        )
        tmp.write(wrapper + code)
        tmp.close()

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            tmp.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except (TimeoutError, asyncio.TimeoutError):
            proc.kill()
            await proc.communicate()
            return ToolResult(tool="python_exec", success=False, output="", error="Timed out (5s)")

        out = stdout.decode().strip()
        err = stderr.decode().strip()
        return ToolResult(
            tool="python_exec",
            success=(proc.returncode == 0),
            output=out,
            error=err,
        )
    finally:
        os.unlink(tmp.name)


# ── Shell Execution ───────────────────────────────────────────────────────────


async def tool_shell_exec(command: str) -> ToolResult:
    """Execute a safe shell command."""
    parts = command.strip().split()
    if not parts:
        return ToolResult(tool="shell_exec", success=False, output="", error="Empty command")

    cmd_name = parts[0]
    if cmd_name not in SAFE_SHELL_COMMANDS:
        return ToolResult(
            tool="shell_exec",
            success=False,
            output="",
            error=f"Command '{cmd_name}' not in safe allowlist. Allowed: {', '.join(sorted(SAFE_SHELL_COMMANDS))}",
        )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path.home()),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        out = stdout.decode().strip()
        err = stderr.decode().strip()
        # Truncate large outputs
        if len(out) > 5000:
            out = out[:5000] + f"\n... (truncated, {len(out)} chars total)"
        return ToolResult(tool="shell_exec", success=(proc.returncode == 0), output=out, error=err)
    except (TimeoutError, asyncio.TimeoutError):
        return ToolResult(tool="shell_exec", success=False, output="", error="Timed out (10s)")
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
        return ToolResult(tool="shell_exec", success=False, output="", error=str(e))


# ── File Operations ───────────────────────────────────────────────────────────


async def tool_file_read(path: str) -> ToolResult:
    """Read a file from the local filesystem."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return ToolResult(
                tool="file_read", success=False, output="", error=f"File not found: {path}"
            )
        if not p.is_file():
            return ToolResult(
                tool="file_read", success=False, output="", error=f"Not a file: {path}"
            )
        if p.stat().st_size > 500_000:
            return ToolResult(
                tool="file_read", success=False, output="", error="File too large (>500KB)"
            )
        content = p.read_text(errors="replace")
        return ToolResult(tool="file_read", success=True, output=content)
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
        return ToolResult(tool="file_read", success=False, output="", error=str(e))


async def tool_file_write(path: str, content: str) -> ToolResult:
    """Write content to a file."""
    try:
        p = Path(path).expanduser().resolve()
        # Safety: only allow writes within home directory
        home = Path.home()
        if not str(p).startswith(str(home)):
            return ToolResult(
                tool="file_write",
                success=False,
                output="",
                error="Writes only allowed within home directory",
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolResult(
            tool="file_write", success=True, output=f"Written {len(content)} bytes to {p}"
        )
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
        return ToolResult(tool="file_write", success=False, output="", error=str(e))


# ── Web Search ────────────────────────────────────────────────────────────────


async def tool_web_search(query: str) -> ToolResult:
    """Search the web using DuckDuckGo HTML endpoint (no API key required).

    Bypasses the duckduckgo_search DDGS library which has a bug that
    hardcodes the 'bing' backend and returns empty results.  Instead we
    POST directly to https://html.duckduckgo.com/html and parse with lxml.
    """
    try:
        import requests as _requests  # type: ignore
        from lxml.html import document_fromstring  # type: ignore

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://html.duckduckgo.com/",
        }
        resp = _requests.post(
            "https://html.duckduckgo.com/html",
            data={"q": query, "b": ""},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()

        tree = document_fromstring(resp.content)
        results: list[dict[str, str]] = []
        cache: set[str] = set()
        for div in tree.xpath("//div[h2]"):
            hrefs = div.xpath("./a/@href")
            href = str(hrefs[0]) if hrefs else ""
            if (
                not href
                or href in cache
                or href.startswith(
                    ("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")
                )
            ):
                continue
            cache.add(href)
            titles = div.xpath("./h2/a/text()")
            title = str(titles[0]).strip() if titles else ""
            bodies = div.xpath("./a//text()")
            body = "".join(str(x) for x in bodies).strip() if bodies else ""
            results.append({"title": title, "href": href, "body": body})
            if len(results) >= 5:
                break

        formatted = []
        for r in results:
            formatted.append(
                f"**{r.get('title', '')}**\n{r.get('body', '')}\nURL: {r.get('href', '')}"
            )
        output = "\n\n---\n\n".join(formatted) if formatted else "No results found."
        return ToolResult(tool="web_search", success=True, output=output)
    except ImportError:
        return ToolResult(
            tool="web_search",
            success=False,
            output="",
            error="Missing dependencies. Run: pip install requests lxml",
        )
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
        return ToolResult(tool="web_search", success=False, output="", error=str(e))
