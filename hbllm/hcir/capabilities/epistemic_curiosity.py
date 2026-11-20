"""Autonomous Epistemic Curiosity Controller for HCIR.

Closes the knowledge-gap loop without any LLM dependency:
  1. Detects high epistemic uncertainty (e.g., unknown concept / missing causal chain).
  2. Dispatches governed ``web.search`` (and optional ``web.fetch``) through
     ``CapabilityResolver`` so all SSRF and sandbox policies remain enforced.
  3. Commits newly discovered facts/summaries transactionally into
     ``TieredWorkspace.persistent`` with explicit source provenance.
  4. Re-synthesizes the answer from the freshly updated persistent graph.

Invariants:
  - Zero LLM dependency.
  - All external network access goes strictly through governed ``CapabilityResolver``.
  - Bounded session deduplication prevents infinite research loops.
  - All state mutations route transactionally via ``commit_operations()``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import BeliefNode, ConceptNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

logger = logging.getLogger(__name__)


@dataclass
class CuriosityResult:
    """Outcome of an autonomous epistemic curiosity research cycle."""

    researched: bool = False
    concept: str = ""
    propositions_ingested: int = 0
    sources: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""


class EpistemicCuriosityController:
    """Uncertainty-gated autonomous research loop using governed web capabilities."""

    def __init__(
        self,
        capability_resolver: Any,
        tiered_workspace: Any | None = None,
        *,
        uncertainty_threshold: float = 0.75,
        max_session_queries: int = 64,
        fetch_top_result: bool = True,
        grant: Any | None = None,
    ) -> None:
        self._resolver = capability_resolver
        self._tiered_workspace = tiered_workspace
        self._uncertainty_threshold = uncertainty_threshold
        self._fetch_top_result = fetch_top_result
        self._grant = grant
        self._recent_queries: deque[str] = deque(maxlen=max_session_queries)

    def should_research(self, concept: str, uncertainty: float = 1.0) -> bool:
        """Return True if epistemic uncertainty justifies an autonomous web lookup."""
        clean = concept.strip().lower()
        if not clean or len(clean) < 2:
            return False
        if uncertainty < self._uncertainty_threshold:
            return False
        if clean in self._recent_queries:
            return False
        if self._resolver is None or not self._resolver.has_capability("web.search"):
            return False
        return True

    async def research_concept(
        self,
        concept: str,
        *,
        query_mode: str = "definition",
        uncertainty: float = 1.0,
    ) -> CuriosityResult:
        """Execute a governed web search (+ optional fetch) and ingest findings.

        Args:
            concept: Normalized concept/topic to research.
            query_mode: ``"definition"`` | ``"why"`` | ``"how"``.
            uncertainty: Epistemic uncertainty score in ``[0.0, 1.0]``.
        """
        clean = concept.strip().lower()
        if not self.should_research(clean, uncertainty=uncertainty):
            return CuriosityResult(researched=False, concept=clean)

        self._recent_queries.append(clean)

        search_query = clean
        if query_mode == "why":
            search_query = f"cause of {clean}"
        elif query_mode == "how":
            search_query = f"how {clean} works"

        try:
            search_res = await self._resolver.resolve_and_execute(
                "web.search",
                {"query": search_query, "limit": 3, "auto_ingest": True},
            )
        except Exception as exc:
            logger.warning("EpistemicCuriosity: web.search failed for '%s': %s", clean, exc)
            return CuriosityResult(researched=False, concept=clean, error=str(exc))

        if not isinstance(search_res, dict) or not search_res.get("ok"):
            err = (
                search_res.get("error", "search failed")
                if isinstance(search_res, dict)
                else "invalid search result"
            )
            return CuriosityResult(researched=False, concept=clean, error=str(err))

        total_ingested = int(search_res.get("propositions_ingested", 0))
        results = search_res.get("results", []) or []
        sources: list[str] = []
        snippets: list[str] = []

        for item in results[:3]:
            if isinstance(item, dict):
                url = str(item.get("url", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                if url:
                    sources.append(url)
                if snippet:
                    snippets.append(snippet)

        # Optionally fetch the top result for deeper proposition extraction
        fetched_summary = ""
        if (
            self._fetch_top_result
            and sources
            and self._resolver.has_capability("web.fetch")
            and total_ingested == 0
        ):
            try:
                fetch_res = await self._resolver.resolve_and_execute(
                    "web.fetch",
                    {"url": sources[0], "auto_ingest": True},
                )
                if isinstance(fetch_res, dict) and fetch_res.get("ok"):
                    total_ingested += int(fetch_res.get("propositions_ingested", 0))
                    fetched_summary = str(fetch_res.get("content", "")).strip()
            except Exception as exc:
                logger.debug("EpistemicCuriosity: web.fetch skipped for %s: %s", sources[0], exc)

        lead_snippet = snippets[0] if snippets else fetched_summary[:400]

        # If structured triple extraction yielded 0 propositions, still ground the
        # retrieved lead snippet as a provenance-backed BeliefNode & ConceptNode
        if total_ingested == 0 and lead_snippet:
            stored = self._commit_summary_belief(
                clean, lead_snippet, sources[0] if sources else "web.search"
            )
            if stored:
                total_ingested = 1

        return CuriosityResult(
            researched=True,
            concept=clean,
            propositions_ingested=total_ingested,
            sources=sources,
            summary=lead_snippet,
        )

    def _commit_summary_belief(self, concept: str, snippet: str, source_url: str) -> bool:
        """Transactionally store a provenance-backed summary belief for ``concept``."""
        if self._tiered_workspace is None:
            return False

        workspace = getattr(self._tiered_workspace, "persistent", None) or self._tiered_workspace
        if not hasattr(workspace, "commit_operations"):
            return False

        grant = self._grant or getattr(workspace, "_system_grant", None)
        cid = f"web_concept_{concept.replace(' ', '_')}"
        cnode = ConceptNode(
            id=cid,
            label=concept,
            definition=snippet[:300],
            domain="web_curiosity",
            tags=["web_curiosity", "autonomous_research"],
        )
        belief = BeliefNode(
            id=f"web_curiosity_belief_{concept.replace(' ', '_')}",
            statement=f"{concept}: {snippet[:260]}",
            confidence=0.75,
            tags=["web_curiosity", "autonomous_research", source_url],
            properties={"source_uri": source_url, "concept": concept},
        )

        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=cnode.id,
                node_data=cnode.model_dump(),
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=belief.id,
                node_data=belief.model_dump(),
            ),
        ]
        try:
            workspace.commit_operations(ops, author="epistemic_curiosity", grant=grant)
            return True
        except Exception as exc:
            logger.warning("EpistemicCuriosity: failed to commit summary belief: %s", exc)
            return False

    def reset_session(self) -> None:
        """Clear session deduplication cache."""
        self._recent_queries.clear()
