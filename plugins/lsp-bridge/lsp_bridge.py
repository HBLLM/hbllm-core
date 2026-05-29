"""
LSP Bridge Plugin — connects HBLLM to Language Servers (Pyright, rust-analyzer, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from typing import Any

from hbllm.network.messages import Message
from hbllm.plugin.sdk import HBLLMPlugin, subscribe

logger = logging.getLogger(__name__)

# Default language to LSP server commands mapping
DEFAULT_SERVERS: dict[str, list[str]] = {
    "python": ["pyright", "--stdio"],
    "rust": ["rust-analyzer"],
    "javascript": ["typescript-language-server", "--stdio"],
    "typescript": ["typescript-language-server", "--stdio"],
    "go": ["gopls"],
}


class LspClient:
    """A single Language Server Protocol client session running stdio."""

    def __init__(self, command: list[str], workspace_dir: str) -> None:
        self.command = command
        self.workspace_dir = workspace_dir
        self.process: asyncio.subprocess.Process | None = None
        self.read_task: asyncio.Task[None] | None = None
        self.request_id = 0
        self.pending: dict[int, asyncio.Future[Any]] = {}
        self.diagnostics: dict[str, list[dict[str, Any]]] = {}
        self.running = False

    async def start(self) -> None:
        """Spawn the language server and initialize it."""
        if not shutil.which(self.command[0]):
            raise FileNotFoundError(
                f"LSP server executable '{self.command[0]}' not found in system PATH."
            )

        self.process = await asyncio.create_subprocess_exec(
            self.command[0],
            *self.command[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self.workspace_dir,
        )
        self.running = True
        self.read_task = asyncio.create_task(self._read_loop())

        # Perform initialize handshake
        await self._send_request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootPath": self.workspace_dir,
                "rootUri": f"file://{self.workspace_dir}",
                "capabilities": {
                    "textDocument": {
                        "definition": {"dynamicRegistration": True},
                        "references": {"dynamicRegistration": True},
                        "publishDiagnostics": {"relatedInformation": True},
                    }
                },
            },
        )
        await self._send_notification("initialized", {})
        logger.info("LSP server '%s' initialized successfully.", self.command[0])

    async def _send_request(
        self, method: str, params: dict[str, Any], timeout: float = 10.0
    ) -> Any:
        if not self.running or not self.process or not self.process.stdin:
            raise RuntimeError("LSP server is not running")

        self.request_id += 1
        req_id = self.request_id
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        future = asyncio.get_event_loop().create_future()
        self.pending[req_id] = future

        body = json.dumps(request).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.process.stdin.write(header + body)
        await self.process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            self.pending.pop(req_id, None)
            raise TimeoutError(f"LSP request '{method}' timed out after {timeout}s")

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        if not self.running or not self.process or not self.process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        body = json.dumps(notification).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.process.stdin.write(header + body)
        await self.process.stdin.drain()

    async def _read_loop(self) -> None:
        if not self.process or not self.process.stdout:
            return

        reader = self.process.stdout

        while self.running:
            try:
                line = await reader.readline()
                if not line:
                    break

                line_str = line.decode("ascii", errors="replace").strip()
                if line_str.startswith("Content-Length:"):
                    content_length = int(line_str.split(":")[1].strip())
                    await reader.readline()  # empty separator line
                    body = await reader.readexactly(content_length)
                    msg = json.loads(body.decode("utf-8"))

                    # Handle Responses
                    if "id" in msg:
                        msg_id = msg["id"]
                        if msg_id in self.pending:
                            future = self.pending.pop(msg_id)
                            if not future.done():
                                if "error" in msg:
                                    future.set_exception(
                                        RuntimeError(
                                            f"LSP error: {msg['error'].get('message', 'Unknown')}"
                                        )
                                    )
                                else:
                                    future.set_result(msg.get("result"))

                    # Handle Notifications (e.g. Diagnostics)
                    elif msg.get("method") == "textDocument/publishDiagnostics":
                        params = msg.get("params", {})
                        uri = params.get("uri", "")
                        diags = params.get("diagnostics", [])
                        self.diagnostics[uri] = diags

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in LSP read loop: %s", e)
                break

    async def stop(self) -> None:
        self.running = False
        if self.read_task:
            self.read_task.cancel()
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
            except Exception:
                if self.process:
                    self.process.kill()


class LspBridge(HBLLMPlugin):
    """
    HBLLM Plugin wrapping LSP language server processes.
    """

    def __init__(self, node_id: str = "lsp_bridge", workspace_dir: str | None = None) -> None:
        super().__init__(
            node_id=node_id,
            capabilities=["lsp_diagnostics", "lsp_definition", "lsp_references"],
        )
        self.workspace_dir = os.path.abspath(workspace_dir or os.getcwd())
        self.clients: dict[str, LspClient] = {}

    def _get_lang_from_file(self, filepath: str) -> str | None:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".py":
            return "python"
        elif ext == ".rs":
            return "rust"
        elif ext in (".js", ".jsx"):
            return "javascript"
        elif ext in (".ts", ".tsx"):
            return "typescript"
        elif ext == ".go":
            return "go"
        return None

    async def _get_client(self, lang: str) -> LspClient | None:
        if lang in self.clients:
            return self.clients[lang]

        cmd = DEFAULT_SERVERS.get(lang)
        if not cmd:
            return None

        try:
            client = LspClient(cmd, self.workspace_dir)
            await client.start()
            self.clients[lang] = client
            return client
        except Exception as e:
            logger.error("Failed to start LSP client for '%s': %s", lang, e)
            return None

    @subscribe("lsp.diagnostics")
    async def on_lsp_diagnostics(self, message: Message) -> Message:
        """Retrieve diagnostics for a given file."""
        filepath = message.payload.get("filepath", "")
        if not filepath:
            return message.create_error("Missing 'filepath' in lsp.diagnostics payload")

        lang = self._get_lang_from_file(filepath)
        if not lang:
            return message.create_error(f"Unsupported file type for LSP: {filepath}")

        client = await self._get_client(lang)
        if not client:
            return message.create_error(f"LSP Server not available/installed for language: {lang}")

        # Send open notification to trigger diagnostic collection if not already opened
        file_uri = f"file://{os.path.abspath(filepath)}"
        try:
            with open(filepath, errors="replace") as fh:
                file_text = fh.read()
        except OSError as e:
            return message.create_error(f"Cannot read file '{filepath}': {e}")

        await client._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": file_uri,
                    "languageId": lang,
                    "version": 1,
                    "text": file_text,
                }
            },
        )

        # Allow brief time for diagnostics to populate
        await asyncio.sleep(0.5)

        diags = client.diagnostics.get(file_uri, [])
        return message.create_response({"status": "SUCCESS", "diagnostics": diags})

    @subscribe("lsp.definition")
    async def on_lsp_definition(self, message: Message) -> Message:
        """Find definition of symbol at file, line, column."""
        filepath = message.payload.get("filepath", "")
        line = message.payload.get("line")  # 0-indexed
        column = message.payload.get("column")  # 0-indexed

        if not filepath or line is None or column is None:
            return message.create_error(
                "Missing 'filepath', 'line', or 'column' in lsp.definition payload"
            )

        lang = self._get_lang_from_file(filepath)
        if not lang:
            return message.create_error(f"Unsupported file type for LSP: {filepath}")

        client = await self._get_client(lang)
        if not client:
            return message.create_error(f"LSP Server not available/installed for language: {lang}")

        file_uri = f"file://{os.path.abspath(filepath)}"
        try:
            res = await client._send_request(
                "textDocument/definition",
                {
                    "textDocument": {"uri": file_uri},
                    "position": {"line": line, "character": column},
                },
            )
            return message.create_response({"status": "SUCCESS", "result": res})
        except Exception as e:
            return message.create_error(f"LSP definition query failed: {e}")

    @subscribe("lsp.references")
    async def on_lsp_references(self, message: Message) -> Message:
        """Find references of symbol at file, line, column."""
        filepath = message.payload.get("filepath", "")
        line = message.payload.get("line")
        column = message.payload.get("column")

        if not filepath or line is None or column is None:
            return message.create_error(
                "Missing 'filepath', 'line', or 'column' in lsp.references payload"
            )

        lang = self._get_lang_from_file(filepath)
        if not lang:
            return message.create_error(f"Unsupported file type for LSP: {filepath}")

        client = await self._get_client(lang)
        if not client:
            return message.create_error(f"LSP Server not available/installed for language: {lang}")

        file_uri = f"file://{os.path.abspath(filepath)}"
        try:
            res = await client._send_request(
                "textDocument/references",
                {
                    "textDocument": {"uri": file_uri},
                    "position": {"line": line, "character": column},
                    "context": {"includeDeclaration": True},
                },
            )
            return message.create_response({"status": "SUCCESS", "result": res})
        except Exception as e:
            return message.create_error(f"LSP references query failed: {e}")

    async def on_stop(self) -> None:
        for client in self.clients.values():
            await client.stop()
        self.clients.clear()
