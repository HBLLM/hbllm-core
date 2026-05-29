"""
Unit tests for LspBridge plugin.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

# Add the lsp-bridge plugin path to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lsp_bridge import LspBridge, LspClient

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest_asyncio.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def lsp_plugin(bus):
    plugin = LspBridge(node_id="test_lsp")
    await plugin.start(bus)
    yield plugin
    await plugin.stop()


@pytest.mark.asyncio
class TestLspBridge:
    async def test_language_detection_python(self, lsp_plugin):
        assert lsp_plugin._get_lang_from_file("main.py") == "python"

    async def test_language_detection_rust(self, lsp_plugin):
        assert lsp_plugin._get_lang_from_file("lib.rs") == "rust"

    async def test_language_detection_typescript(self, lsp_plugin):
        assert lsp_plugin._get_lang_from_file("index.ts") == "typescript"
        assert lsp_plugin._get_lang_from_file("App.tsx") == "typescript"

    async def test_language_detection_go(self, lsp_plugin):
        assert lsp_plugin._get_lang_from_file("main.go") == "go"

    async def test_language_detection_javascript(self, lsp_plugin):
        assert lsp_plugin._get_lang_from_file("app.js") == "javascript"
        assert lsp_plugin._get_lang_from_file("Component.jsx") == "javascript"

    async def test_language_detection_unsupported(self, lsp_plugin):
        assert lsp_plugin._get_lang_from_file("styles.css") is None
        assert lsp_plugin._get_lang_from_file("README.md") is None

    async def test_diagnostics_missing_filepath(self, lsp_plugin, bus):
        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="lsp.diagnostics",
            payload={},
        )
        resp = await bus.request("lsp.diagnostics", query, timeout=2.0)
        assert resp.type == MessageType.ERROR
        assert "filepath" in resp.payload["error"].lower()

    async def test_diagnostics_unsupported_file(self, lsp_plugin, bus):
        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="lsp.diagnostics",
            payload={"filepath": "styles.css"},
        )
        resp = await bus.request("lsp.diagnostics", query, timeout=2.0)
        assert resp.type == MessageType.ERROR
        assert "Unsupported" in resp.payload["error"]

    async def test_definition_missing_params(self, lsp_plugin, bus):
        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="lsp.definition",
            payload={"filepath": "main.py"},  # missing line and column
        )
        resp = await bus.request("lsp.definition", query, timeout=2.0)
        assert resp.type == MessageType.ERROR
        assert "line" in resp.payload["error"].lower() or "column" in resp.payload["error"].lower()

    async def test_references_missing_params(self, lsp_plugin, bus):
        query = Message(
            type=MessageType.QUERY,
            source_node_id="tester",
            topic="lsp.references",
            payload={"filepath": "main.py", "line": 10},  # missing column
        )
        resp = await bus.request("lsp.references", query, timeout=2.0)
        assert resp.type == MessageType.ERROR
        assert "column" in resp.payload["error"].lower()

    async def test_diagnostics_server_not_installed(self, lsp_plugin, bus):
        """When the LSP executable is not found, return a helpful error."""
        with patch("shutil.which", return_value=None):
            query = Message(
                type=MessageType.QUERY,
                source_node_id="tester",
                topic="lsp.diagnostics",
                payload={"filepath": "main.py"},
            )
            resp = await bus.request("lsp.diagnostics", query, timeout=2.0)
            assert resp.type == MessageType.ERROR
            assert (
                "not available" in resp.payload["error"].lower()
                or "not found" in resp.payload["error"].lower()
            )

    async def test_lsp_client_stop_graceful(self):
        """Verify LspClient.stop() handles no-process case gracefully."""
        client = LspClient(["fake-server"], "/tmp")
        # Should not raise even if never started
        await client.stop()
        assert not client.running
