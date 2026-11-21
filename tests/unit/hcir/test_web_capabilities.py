"""Unit tests for HCIR-native governed Web Capabilities (web.search, web.fetch).

Tests cover:
  1. SSRF and URL safety guards (loopback, private, link-local metadata, schemes, ports).
  2. SimpleHTMLTextExtractor (HTML sanitization, tag filtering, title and paragraph parsing).
  3. WebCapabilityExecutor execution with mock search and fetch responses.
  4. Automatic proposition extraction and transactional ingestion into persistent memory.
  5. CommandDispatcher routing (contextual search vs fs.search, fetch, google, browse).
  6. End-to-end cognitive dialogue processing with web capabilities.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from hbllm.hcir.capabilities.web_capabilities import (
    SimpleHTMLTextExtractor,
    WebCapabilityExecutor,
    is_safe_url,
    strip_html_tags,
)
from hbllm.hcir.kernel.command_dispatcher import CommandDispatcher
from hbllm.hcir.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.runtime.bootstrap import build_cognitive_system

# ═══════════════════════════════════════════════════════════════════════════
# 1. SSRF & URL Safety Validation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSSRFValidation:
    """Verifies that all SSRF attack vectors and non-web endpoints are blocked."""

    @pytest.mark.parametrize(
        "bad_url",
        [
            "file:///etc/passwd",
            "ftp://example.com/file",
            "gopher://example.com/",
            "data:text/plain;base64,SGVsbG8=",
            "javascript:alert(1)",
        ],
    )
    def test_reject_non_http_schemes(self, bad_url: str) -> None:
        safe, msg = is_safe_url(bad_url)
        assert safe is False
        assert "Disallowed scheme" in msg or "Malformed URL" in msg

    @pytest.mark.parametrize(
        "loopback_url",
        [
            "http://127.0.0.1/admin",
            "http://127.0.0.2:80/status",
            "http://localhost/metrics",
        ],
    )
    def test_reject_loopback_addresses(self, loopback_url: str) -> None:
        safe, msg = is_safe_url(loopback_url)
        assert safe is False
        assert "loopback" in msg.lower() or "rejected" in msg.lower()

    @pytest.mark.parametrize(
        "private_url",
        [
            "http://10.0.0.1/secret",
            "http://192.168.1.1/router",
            "http://172.16.0.1/internal",
            "http://172.31.255.255/internal",
        ],
    )
    def test_reject_private_rfc1918_addresses(self, private_url: str) -> None:
        safe, msg = is_safe_url(private_url)
        assert safe is False
        assert "private" in msg.lower()

    def test_reject_cloud_metadata_link_local(self) -> None:
        safe, msg = is_safe_url("http://169.254.169.254/latest/meta-data")
        assert safe is False
        assert "metadata" in msg.lower() or "link-local" in msg.lower() or "private" in msg.lower()

    @pytest.mark.parametrize(
        "bad_port_url",
        [
            "http://example.com:22/",
            "http://example.com:25/",
            "http://example.com:3306/",
            "http://example.com:5432/",
            "http://example.com:6379/",
        ],
    )
    def test_reject_non_web_ports(self, bad_port_url: str) -> None:
        safe, msg = is_safe_url(bad_port_url)
        assert safe is False
        assert "Port" in msg and "not permitted" in msg

    def test_accept_public_https_url(self) -> None:
        safe, msg = is_safe_url("https://en.wikipedia.org/wiki/Quantum_computing")
        assert safe is True
        assert msg == "OK"


# ═══════════════════════════════════════════════════════════════════════════
# 2. HTML Text Extraction Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHTMLTextExtraction:
    """Verifies that SimpleHTMLTextExtractor extracts clean text and title."""

    def test_extract_clean_text_and_title(self) -> None:
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Computing - Overview</title>
            <style>body { font-size: 14px; }</style>
            <script>console.log('tracking');</script>
        </head>
        <body>
            <header><p>Site Header</p></header>
            <nav><a href="/">Home</a></nav>
            <h1>Quantum Computing</h1>
            <p>Quantum computing is a rapidly-emerging technology.</p>
            <p>It solves computational problems beyond classical physics.</p>
            <aside>Related Links</aside>
            <footer>Copyright 2026</footer>
        </body>
        </html>
        """
        extractor = SimpleHTMLTextExtractor()
        extractor.feed(sample_html)
        title, text = extractor.get_result()

        assert title == "Quantum Computing - Overview"
        assert "Quantum computing is a rapidly-emerging technology." in text
        assert "beyond classical physics" in text
        # Ignored sections must not appear
        assert "tracking" not in text
        assert "font-size" not in text
        assert "Site Header" not in text
        assert "Copyright 2026" not in text

    def test_strip_html_tags(self) -> None:
        raw = 'A <span class="searchmatch">quantum</span> computer is a <span class="searchmatch">machine</span>.'
        clean = strip_html_tags(raw)
        assert clean == "A quantum computer is a machine."


# ═══════════════════════════════════════════════════════════════════════════
# 3. WebCapabilityExecutor Mock Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWebCapabilityExecutor:
    """Tests WebCapabilityExecutor execution with mock network responses."""

    @pytest.mark.asyncio
    async def test_web_search_with_mock_results(self) -> None:
        mock_payload = {
            "query": {
                "search": [
                    {
                        "title": "Quantum computing",
                        "snippet": 'A <span class="searchmatch">quantum</span> computing system.',
                    },
                    {
                        "title": "Superconducting quantum computing",
                        "snippet": "Branch of quantum physics.",
                    },
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        executor = WebCapabilityExecutor(default_capability="web.search")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = await executor.execute({"query": "quantum computing", "auto_ingest": False})

        assert res["ok"] is True
        assert res["action"] == "searched"
        assert res["total_results"] == 2
        assert "Quantum computing" in res["content"]
        assert "https://en.wikipedia.org/wiki/Quantum_computing" in res["content"]

    @pytest.mark.asyncio
    async def test_web_fetch_with_mock_html(self) -> None:
        html_doc = """
        <html>
        <head><title>Photosynthesis - Wikipedia</title></head>
        <body>
            <h1>Photosynthesis</h1>
            <p>Photosynthesis is a biological process used by plants.</p>
            <p>It converts light energy into chemical energy.</p>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.read.return_value = html_doc.encode("utf-8")
        mock_resp.headers.get.return_value = "text/html; charset=utf-8"
        mock_resp.headers.get_content_charset.return_value = "utf-8"
        mock_resp.__enter__.return_value = mock_resp

        executor = WebCapabilityExecutor(default_capability="web.fetch")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = await executor.execute(
                {
                    "url": "https://en.wikipedia.org/wiki/Photosynthesis",
                    "auto_ingest": False,
                }
            )

        assert res["ok"] is True
        assert res["action"] == "fetched"
        assert res["title"] == "Photosynthesis - Wikipedia"
        assert "biological process used by plants" in res["content"]

    @pytest.mark.asyncio
    async def test_auto_ingestion_commits_propositions(self) -> None:
        tiered = TieredWorkspace()
        executor = WebCapabilityExecutor(
            default_capability="web.fetch",
            tiered_workspace=tiered,
        )

        html_with_propositions = """
        <html>
        <head><title>Scientific Concepts</title></head>
        <body>
            <p>A microscope is a magnifier.</p>
            <p>Gravity causes attraction.</p>
            <p>Combustion requires oxygen.</p>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.read.return_value = html_with_propositions.encode("utf-8")
        mock_resp.headers.get.return_value = "text/html"
        mock_resp.headers.get_content_charset.return_value = "utf-8"
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = await executor.execute(
                {
                    "url": "https://example.org/science",
                    "auto_ingest": True,
                }
            )

        assert res["ok"] is True
        assert res["propositions_ingested"] >= 2
        # Check that nodes are committed in persistent graph
        p_graph = tiered.persistent.graph
        assert any("microscope" in n.id for n in p_graph.nodes.values())
        assert any("gravity" in n.id for n in p_graph.nodes.values())


# ═══════════════════════════════════════════════════════════════════════════
# 4. CommandDispatcher Contextual Routing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCommandDispatcherRouting:
    """Tests intelligent contextual routing for web commands vs fs commands."""

    def test_routing_search_web_vs_search_filesystem(self) -> None:
        mock_resolver = MagicMock()
        mock_resolver.has_capability.side_effect = lambda cap: (
            cap
            in (
                "fs.search",
                "web.search",
                "web.fetch",
            )
        )

        disp = CommandDispatcher(mock_resolver, None)

        # "search the web for quantum" -> web.search
        web_frame = SemanticFrame(
            frame_type=FrameType.COMMAND,
            predicate="search",
            metadata=LanguageMetadata(raw_text="search the web for quantum computing"),
        )
        web_frame.set_role(ThematicRole.PATIENT, EntityReference(concept_name="web"))
        assert disp._resolve_capability("search", frame=web_frame) == "web.search"

        # "search pattern *.py" -> fs.search
        fs_frame = SemanticFrame(
            frame_type=FrameType.COMMAND,
            predicate="search",
            metadata=LanguageMetadata(raw_text="search for pattern in files"),
        )
        fs_frame.set_role(ThematicRole.PATIENT, EntityReference(concept_name="pattern"))
        assert disp._resolve_capability("search", frame=fs_frame) == "fs.search"

    def test_routing_web_verbs(self) -> None:
        mock_resolver = MagicMock()
        mock_resolver.has_capability.return_value = True
        disp = CommandDispatcher(mock_resolver, None)

        assert disp._resolve_capability("websearch") == "web.search"
        assert disp._resolve_capability("google") == "web.search"
        assert disp._resolve_capability("fetch") == "web.fetch"
        assert disp._resolve_capability("browse") == "web.fetch"

    def test_param_building_for_web_search(self) -> None:
        disp = CommandDispatcher(MagicMock(), None)
        frame = SemanticFrame(
            frame_type=FrameType.COMMAND,
            predicate="search",
            metadata=LanguageMetadata(raw_text="search the web for artificial intelligence"),
        )
        params = disp._build_params(frame, "web.search", "web")
        assert params["query"] == "artificial intelligence"

    def test_param_building_for_web_fetch(self) -> None:
        disp = CommandDispatcher(MagicMock(), None)
        frame = SemanticFrame(
            frame_type=FrameType.COMMAND,
            predicate="fetch",
            metadata=LanguageMetadata(
                raw_text="fetch https://en.wikipedia.org/wiki/Photosynthesis"
            ),
        )
        params = disp._build_params(
            frame, "web.fetch", "https://en.wikipedia.org/wiki/Photosynthesis"
        )
        assert params["url"] == "https://en.wikipedia.org/wiki/Photosynthesis"


# ═══════════════════════════════════════════════════════════════════════════
# 5. End-to-End Dialogue Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWebCapabilitiesDialogue:
    """Tests complete turn execution through CognitiveSystem.process_dialogue."""

    def test_e2e_search_the_web_dialogue(self) -> None:
        sys = build_cognitive_system()

        mock_payload = {
            "query": {
                "search": [
                    {
                        "title": "Quantum computing",
                        "snippet": "Quantum computers process information using qubits.",
                    }
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = sys.process_dialogue("search the web for quantum computing")

        assert res.is_success is True
        assert (
            res.epistemic_state is not None and res.epistemic_state.target_predicate == "web.search"
        )
        assert "Quantum computing" in res.response_text

    def test_e2e_fetch_url_dialogue(self) -> None:
        sys = build_cognitive_system()

        html_doc = """
        <html>
        <head><title>Quantum Teleportation</title></head>
        <body>
            <h1>Quantum Teleportation</h1>
            <p>Quantum teleportation is a technique for transferring quantum information.</p>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.read.return_value = html_doc.encode("utf-8")
        mock_resp.headers.get.return_value = "text/html"
        mock_resp.headers.get_content_charset.return_value = "utf-8"
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = sys.process_dialogue("fetch https://en.wikipedia.org/wiki/Quantum_teleportation")

        assert res.is_success is True
        assert (
            res.epistemic_state is not None and res.epistemic_state.target_predicate == "web.fetch"
        )
        assert "Quantum Teleportation" in res.response_text
