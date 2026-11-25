"""Kernel-governed web capabilities (web.search, web.fetch).

Enforces strict SSRF protection, clean HTML extraction without external dependencies,
optional automatic proposition extraction and ingestion into TieredWorkspace.persistent,
and structured discourse formatting for conversational turns.

Invariants:
  1. Zero LLM dependency in the capability core.
  2. Complete SSRF prevention: RFC1918 private IPs, loopback, link-local (cloud metadata),
     multicast, and non-web ports are strictly rejected.
  3. All knowledge mutations route transactionally through TieredWorkspace / HCIRMemoryBackend.
  4. Fully sandboxed execution adhering to SandboxedCapabilityPolicy (allow_network=True).
"""

from __future__ import annotations

import html
import ipaddress
import json
import logging
import os
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from hbllm.hcir.graph import BeliefNode, ConceptNode, HCIREdge, HCIREdgeType
from hbllm.hcir.language.ontology_bootstrap import LexicalOntologyRecord
from hbllm.hcir.language.reading_curriculum import (
    ExtractedProposition,
    extract_propositions_from_text,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

logger = logging.getLogger(__name__)

WEB_CAPABILITIES: tuple[str, ...] = (
    "web.search",
    "web.fetch",
)

DEFAULT_USER_AGENT: str = "HBLLM-CognitiveOS/1.0 (https://github.com/hbllm; contact@hbllm.ai)"
WIKIPEDIA_API_ENDPOINT: str = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&utf8=1&srlimit={limit}"
ALLOWED_SCHEMES: tuple[str, ...] = ("http", "https")
ALLOWED_PORTS: tuple[int, ...] = (80, 443, 8080, 8443)


def is_safe_url(url: str) -> tuple[bool, str]:
    """Validate a target URL against SSRF vulnerabilities before connecting.

    Rejects:
      - Non-http/https schemes (file, ftp, gopher, data, etc.)
      - Missing or empty hostnames
      - Non-standard web ports (22, 25, 3306, 5432, 6379, etc.)
      - Loopback addresses (127.0.0.0/8, ::1)
      - RFC1918 private subnets (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
      - Link-local and cloud metadata addresses (169.254.0.0/16, fe80::/10)
      - Multicast, broadcast, unspecified, and reserved IPs
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        return False, f"Malformed URL: {exc}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return False, f"Disallowed scheme '{scheme}' (only http and https are permitted)"

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL"

    port = parsed.port or (443 if scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        return False, f"Port {port} not permitted (only standard web ports permitted)"

    try:
        addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return False, f"Failed to resolve hostname '{hostname}': {exc}"

    for info in addr_info:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"Invalid resolved IP address: {ip_str}"

        if ip.is_loopback:
            return False, f"Access to loopback address ({ip}) is rejected"
        if ip.is_private:
            return False, f"Access to private IP address ({ip}) is rejected"
        if ip.is_link_local:
            return False, f"Access to link-local / metadata address ({ip}) is rejected"
        if ip.is_multicast:
            return False, f"Access to multicast address ({ip}) is rejected"
        if ip.is_reserved or ip.is_unspecified:
            return False, f"Access to reserved IP address ({ip}) is rejected"

    return True, "OK"


class SimpleHTMLTextExtractor(HTMLParser):
    """Clean text extractor from HTML with zero external dependencies.

    Filters out script, style, navigation, header, and footer tags,
    extracts the page title, and normalizes paragraph breaks.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ignored_tags: set[str] = {
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
            "aside",
        }
        self._ignored_depth: int = 0
        self._title_tokens: list[str] = []
        self._in_title: bool = False
        self._text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self._ignored_tags:
            self._ignored_depth += 1
        elif t == "title":
            self._in_title = True
        elif t in ("p", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._text_chunks.append("\n\n")
        elif t in ("li", "br", "tr", "div"):
            self._text_chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._ignored_tags:
            if self._ignored_depth > 0:
                self._ignored_depth -= 1
        elif t == "title":
            self._in_title = False
        elif t in ("p", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._text_chunks.append("\n\n")
        elif t in ("li", "div"):
            self._text_chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        if self._in_title:
            self._title_tokens.append(data.strip())
        else:
            self._text_chunks.append(data)

    def get_result(self) -> tuple[str, str]:
        title = " ".join(self._title_tokens).strip()
        raw_text = "".join(self._text_chunks)
        # Collapse excessive whitespace within lines while preserving paragraph breaks
        paragraphs: list[str] = []
        for raw_p in raw_text.split("\n\n"):
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_p.splitlines()]
            joined = " ".join(line for line in lines if line)
            if joined:
                paragraphs.append(joined)
        cleaned = "\n\n".join(paragraphs)
        return title, cleaned


def strip_html_tags(raw: str) -> str:
    """Fast regex-based tag stripper and unescaper for short snippets."""
    no_tags = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(no_tags).strip()


class WebCapabilityExecutor:
    """Kernel-governed executor for web search and fetch operations."""

    def __init__(
        self,
        default_capability: str,
        tiered_workspace: Any | None = None,
        memory_backend: Any | None = None,
        ontology_store: Any | None = None,
        grant: Any | None = None,
        search_endpoint: str | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 15.0,
        max_read_bytes: int = 131072,
    ) -> None:
        self._default_capability = default_capability
        self._tiered_workspace = tiered_workspace
        self._memory_backend = memory_backend
        self._ontology_store = ontology_store
        self._grant = grant
        self._search_endpoint = search_endpoint or os.environ.get("HBLLM_SEARCH_ENDPOINT")
        self._user_agent = user_agent
        self._timeout = timeout_seconds
        self._max_read_bytes = max_read_bytes

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        cap = str(params.get("capability", self._default_capability))
        if cap == "web.search":
            return self._exec_web_search(params)
        if cap == "web.fetch":
            return self._exec_web_fetch(params)
        return {"ok": False, "error": f"Unknown web capability: {cap}"}

    def _exec_web_search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a web search and format results into structured discourse."""
        query = str(params.get("query", params.get("content", params.get("target", "")))).strip()
        if not query:
            return {
                "ok": False,
                "error": "web.search requires a search query ('query' or 'content')",
            }

        limit = min(max(1, int(params.get("limit", 5))), 10)
        auto_ingest = bool(params.get("auto_ingest", True))
        timeout = float(params.get("timeout", self._timeout))

        if self._search_endpoint:
            url = f"{self._search_endpoint}?q={urllib.parse.quote(query)}&format=json"
        else:
            url = WIKIPEDIA_API_ENDPOINT.format(query=urllib.parse.quote(query), limit=limit)

        safe, reason = is_safe_url(url)
        if not safe:
            return {"ok": False, "error": f"Search URL rejected by SSRF guard: {reason}"}

        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read(self._max_read_bytes)
                data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        except Exception as exc:
            return {"ok": False, "error": f"Web search request failed: {exc}"}

        results: list[dict[str, str]] = []
        snippets_for_ingestion: list[str] = []

        if "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                title = str(item.get("title", ""))
                raw_snippet = str(item.get("snippet", ""))
                clean_snippet = strip_html_tags(raw_snippet)
                wiki_url = (
                    f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                )
                results.append(
                    {
                        "title": title,
                        "snippet": clean_snippet,
                        "url": wiki_url,
                    }
                )
                snippets_for_ingestion.append(f"{title}. {clean_snippet}")
        elif isinstance(data, list):
            # Generic JSON search result format
            for item in data[:limit]:
                if isinstance(item, dict):
                    title = str(item.get("title", ""))
                    clean_snippet = strip_html_tags(str(item.get("snippet", item.get("body", ""))))
                    res_url = str(item.get("url", item.get("link", "")))
                    results.append({"title": title, "snippet": clean_snippet, "url": res_url})
                    snippets_for_ingestion.append(f"{title}. {clean_snippet}")

        ingested_count = 0
        if auto_ingest and snippets_for_ingestion:
            combined_text = "\n".join(snippets_for_ingestion)
            propositions = extract_propositions_from_text(
                combined_text,
                source_file=f"search_{query}",
                include_extended=True,
                ontology_store=self._ontology_store,
            )
            if propositions:
                ingested_count = self._ingest_propositions(propositions)

        # Build structured human-readable discourse content
        if not results:
            content = f"No search results found for '{query}'."
        else:
            lines = [f'Found {len(results)} results for "{query}":\n']
            for idx, r in enumerate(results, start=1):
                lines.append(f"{idx}. {r['title']}")
                if r["snippet"]:
                    lines.append(f"   Summary: {r['snippet']}")
                if r["url"]:
                    lines.append(f"   URL: {r['url']}")
                lines.append("")
            if ingested_count > 0:
                lines.append(
                    f"Ingested {ingested_count} new proposition(s) into persistent knowledge."
                )
            content = "\n".join(lines).strip()

        return {
            "ok": True,
            "action": "searched",
            "query": query,
            "total_results": len(results),
            "results": results,
            "propositions_ingested": ingested_count,
            "content": content,
        }

    def _exec_web_fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Safely fetch and parse clean text from a web page."""
        raw_url = str(params.get("url", params.get("target", params.get("content", "")))).strip()
        if not raw_url:
            return {"ok": False, "error": "web.fetch requires a target URL ('url' or 'target')"}

        # Extract URL if surrounded by extra words
        m = re.search(r"https?://[^\s\"'<>]+", raw_url)
        target_url = m.group(0).rstrip(".,;:)[]") if m else raw_url

        safe, reason = is_safe_url(target_url)
        if not safe:
            return {"ok": False, "error": f"URL safety rejection: {reason}"}

        max_bytes = int(params.get("max_bytes", self._max_read_bytes))
        auto_ingest = bool(params.get("auto_ingest", True))
        timeout = float(params.get("timeout", self._timeout))

        headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        }
        req = urllib.request.Request(target_url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "").lower()
                if not any(t in content_type for t in ("text/", "html", "json", "xml")):
                    return {
                        "ok": False,
                        "error": f"Non-text content type '{content_type}' rejected for ingestion",
                    }
                raw_bytes = resp.read(max_bytes)
                charset = resp.headers.get_content_charset() or "utf-8"
                html_text = raw_bytes.decode(charset, errors="replace")
        except Exception as exc:
            return {"ok": False, "error": f"HTTP fetch failed for '{target_url}': {exc}"}

        extractor = SimpleHTMLTextExtractor()
        try:
            extractor.feed(html_text)
            title, clean_text = extractor.get_result()
        except Exception:
            title = target_url
            clean_text = strip_html_tags(html_text)

        ingested_count = 0
        if auto_ingest and clean_text:
            propositions = extract_propositions_from_text(
                clean_text,
                source_file=target_url,
                include_extended=True,
                ontology_store=self._ontology_store,
            )
            if propositions:
                ingested_count = self._ingest_propositions(propositions)

        # Build clean summary discourse for the user
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        substantive = [
            p
            for p in paragraphs
            if len(p) >= 30
            and not p.startswith(
                ("Jump to", "From Wikipedia", "Coordinates:", "Main article:", "See also")
            )
        ]
        lead_summary = (
            "\n\n".join(substantive[:4])
            if substantive
            else ("\n\n".join(paragraphs[:3]) if paragraphs else clean_text[:1200])
        )

        lines = []
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"URL: {target_url}\n")
        lines.append(lead_summary)
        if ingested_count > 0:
            lines.append(
                f"\nIngested {ingested_count} new proposition(s) into persistent knowledge."
            )
        content = "\n".join(lines).strip()

        return {
            "ok": True,
            "action": "fetched",
            "url": target_url,
            "title": title,
            "bytes": len(raw_bytes),
            "propositions_ingested": ingested_count,
            "content": content,
        }

    def _ingest_propositions(
        self,
        propositions: list[ExtractedProposition],
    ) -> int:
        """Transactionally commit extracted propositions to persistent memory."""
        if not propositions:
            return 0

        workspace = None
        if self._tiered_workspace is not None:
            workspace = (
                getattr(self._tiered_workspace, "persistent", None) or self._tiered_workspace
            )
        elif self._memory_backend is not None:
            workspace = getattr(self._memory_backend, "_workspace", None)

        if workspace is None:
            return 0

        grant = self._grant or getattr(workspace, "_system_grant", None)

        node_ops: list[TransactionOperation] = []
        edge_ops: list[TransactionOperation] = []
        revision_ops: list[TransactionOperation] = []
        created_concepts: set[str] = set()
        graph = getattr(workspace, "graph", None)
        existing_edges = getattr(graph, "edges", {})
        store_records: list[LexicalOntologyRecord] = []

        from hbllm.hcir.capabilities.belief_revision import BeliefConflictResolver

        conflict_resolver = BeliefConflictResolver(ontology_store=self._ontology_store)

        edge_type_map = {
            "is_a": HCIREdgeType.GENERALIZES,
            "causes": HCIREdgeType.CAUSES,
            "requires": HCIREdgeType.REQUIRES,
        }

        for prop in propositions:
            src_id = f"web_concept_{prop.subject}"
            dst_id = f"web_concept_{prop.obj}"
            for term, cid in ((prop.subject, src_id), (prop.obj, dst_id)):
                if cid not in created_concepts:
                    created_concepts.add(cid)
                    cnode = ConceptNode(
                        id=cid,
                        label=term,
                        definition=f"Concept extracted from web ingestion ({prop.source_file})",
                        domain="web_ingestion",
                        tags=["web_ingestion"],
                    )
                    node_ops.append(
                        TransactionOperation(
                            op=TransactionOp.UPSERT_NODE,
                            node_id=cnode.id,
                            node_data=cnode.model_dump(),
                        )
                    )

            belief = BeliefNode(
                id=f"web_belief_{prop.subject}_{prop.relation}_{prop.obj}",
                statement=f"{prop.subject} {prop.relation} {prop.obj}",
                confidence=0.80,
                tags=["web_ingestion", prop.relation, prop.source_file],
                properties={"source_file": prop.source_file, "sentence": prop.sentence[:120]},
            )
            rev_report = conflict_resolver.evaluate_belief(belief, graph)
            if rev_report.has_conflicts:
                belief = belief.model_copy(
                    update={
                        "confidence": rev_report.adjusted_confidence,
                        "properties": {
                            **belief.properties,
                            "contested": rev_report.contested,
                            "contradiction_count": len(rev_report.conflicts),
                        },
                    }
                )
                revision_ops.extend(rev_report.operations)

            node_ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=belief.id,
                    node_data=belief.model_dump(),
                )
            )

            edge_id = f"web_{prop.relation}_{src_id}__{dst_id}"
            if edge_id not in existing_edges:
                edge = HCIREdge(
                    id=edge_id,
                    edge_type=edge_type_map.get(prop.relation, HCIREdgeType.GENERALIZES),
                    sources=[src_id],
                    targets=[dst_id],
                    weight=0.80,
                    properties={"source_file": prop.source_file, "relation": prop.relation},
                )
                edge_ops.append(
                    TransactionOperation(
                        op=TransactionOp.ADD_EDGE,
                        edge_data=edge.model_dump(),
                    )
                )

            if prop.relation == "is_a":
                store_records.append(
                    LexicalOntologyRecord(
                        lemma=prop.subject,
                        pos="noun",
                        language="en",
                        definition=prop.sentence[:200],
                        hypernyms=(prop.obj,),
                        domain="web_ingestion",
                    )
                )

        if self._ontology_store is not None and store_records:
            try:
                self._ontology_store.upsert_many(
                    store_records, source="web_ingestion", preserve_existing=True
                )
            except Exception as exc:
                logger.warning("WebCapabilityExecutor: failed to update ontology store: %s", exc)

        all_ops = node_ops + edge_ops + revision_ops
        if all_ops:
            try:
                workspace.commit_operations(all_ops, author="web_ingestion", grant=grant)
            except Exception as exc:
                logger.warning("WebCapabilityExecutor: failed to commit operations: %s", exc)
                return 0

        return len(propositions)
