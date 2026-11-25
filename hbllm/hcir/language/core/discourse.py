"""Multi-Clause Discourse Segmentation & Expository Prose Processor.

Zero-LLM deterministic segmentation of paragraphs, instruction lists, and
compound utterances, plus transactional expository knowledge extraction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.language.english.normalizer import CLOSED_CLASS_VOCAB, COMMAND_VERB_VOCAB
from hbllm.hcir.language.reading_curriculum import (
    ExtractedProposition,
    commit_propositions,
    extract_propositions_from_text,
)

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s<>\"'()]+[^\s<>\"'().,!?;:]")
_FILE_OR_DECIMAL_RE = re.compile(
    r"\b[A-Za-z0-9_./-]*[A-Za-z0-9_/-]+\.[A-Za-z0-9]{1,8}\b|\b\d+\.\d+\b"
)
_ABBREV_RE = re.compile(r"\b(?:e\.g\.|i\.e\.|etc\.|vs\.|dr\.|mr\.|mrs\.|ms\.|prof\.)", flags=re.I)

_CLAUSE_STARTERS: frozenset[str] = COMMAND_VERB_VOCAB | frozenset(
    {
        "what",
        "where",
        "why",
        "how",
        "who",
        "when",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "will",
        "there",
    }
)


@dataclass
class ExpositoryIngestionSummary:
    """Result of deterministically ingesting expository prose into HCIR."""

    propositions: list[ExtractedProposition] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    nodes_committed: int = 0
    edges_committed: int = 0
    summary_text: str = ""


class DiscourseSegmenter:
    """Segments multi-sentence paragraphs, lists, and sequential instruction sets."""

    @classmethod
    def segment(cls, text: str) -> list[str]:
        """Split *text* into ordered, standalone clauses while protecting URLs/paths/decimals."""
        raw = text.strip()
        if not raw:
            return []

        placeholders: dict[str, str] = {}

        def _protect(pattern: re.Pattern[str], source: str, prefix: str) -> str:
            def _sub(match: re.Match[str]) -> str:
                token = f"__{prefix}_{len(placeholders)}__"
                placeholders[token] = match.group(0)
                return token

            return pattern.sub(_sub, source)

        def _restore(piece: str) -> str:
            out = piece
            for token, original in placeholders.items():
                out = out.replace(token, original)
            return out

        protected = _protect(_URL_RE, raw, "URL")
        protected = _protect(_ABBREV_RE, protected, "ABBR")
        protected = _protect(_FILE_OR_DECIMAL_RE, protected, "PATH")

        # Turn inline numbered steps ("1. ... 2. ...") and bullet lines into newline boundaries
        protected = re.sub(r"(?:^|(?<=[\s.!?;]))\s*(?:\d+[.)])\s+", "\n", protected)
        protected = re.sub(r"(?:^|\n)\s*[-*•]\s+", "\n", protected)

        # First split on newlines and semicolons
        primary_chunks: list[str] = []
        for line in re.split(r"[\n;]+", protected):
            line = line.strip()
            if not line:
                continue
            # Split sentences ending with . ! ? followed by whitespace
            for sent in re.split(r"(?<=[.!?])\s+", line):
                s = sent.strip()
                if s:
                    primary_chunks.append(s)

        # Next split sequential instruction connectors ('and then', 'after that', 'then')
        sequential_chunks: list[str] = []
        for chunk in primary_chunks:
            parts = re.split(
                r"\s*,?\s*\b(?:and\s+then|after\s+that|then)\b\s*,?\s*",
                chunk,
                flags=re.IGNORECASE,
            )
            if len(parts) > 1 and all(len(p.strip().split()) >= 2 for p in parts):
                sequential_chunks.extend(p.strip() for p in parts if p.strip())
            else:
                sequential_chunks.append(chunk)

        # Finally split clause-level 'and' ONLY when RHS starts a new action/query clause
        final_segments: list[str] = []
        for chunk in sequential_chunks:
            and_parts = re.split(r"\s*,?\s*\band\b\s+", chunk, flags=re.IGNORECASE)
            if len(and_parts) > 1:
                merged: list[str] = [and_parts[0].strip()]
                for rhs in and_parts[1:]:
                    rhs_clean = rhs.strip()
                    rhs_words = rhs_clean.split()
                    if not rhs_words:
                        continue
                    first_tok = re.sub(r"[^a-zA-Z]", "", rhs_words[0]).lower()
                    # Check if previous part had >=2 words and RHS looks like a standalone clause
                    if (
                        len(merged[-1].split()) >= 2
                        and len(rhs_words) >= 2
                        and (
                            first_tok in _CLAUSE_STARTERS
                            or (
                                first_tok in CLOSED_CLASS_VOCAB
                                and len(rhs_words) >= 4
                                and "?" in rhs_clean
                            )
                        )
                    ):
                        merged.append(rhs_clean)
                    else:
                        merged[-1] = f"{merged[-1]} and {rhs_clean}"
                final_segments.extend(m for m in merged if m)
            else:
                final_segments.append(chunk)

        return [_restore(seg).strip() for seg in final_segments if _restore(seg).strip()]


def _humanize_relation(rel: str) -> str:
    """Format an internal relation key into natural English prose (no internal tokens)."""
    r = rel.strip().lower()
    if r == "is_a":
        return "is"
    if r == "has_property":
        return "is"
    if r == "not_is_a":
        return "is not"
    if r == "not_has_property":
        return "is not"
    if r.startswith("not_"):
        return f"does not {r[4:].replace('_', ' ')}"
    return r.replace("_", " ")


class ExpositoryProseProcessor:
    """Extracts and transactionally commits propositions from long expository prose/essays."""

    def ingest_prose(
        self,
        text: str,
        *,
        workspace: Any,
        tiered_workspace: Any | None = None,
        ontology_store: Any | None = None,
        reference_resolver: Any | None = None,
        parser: Any | None = None,
        source_label: str = "dialogue_prose",
    ) -> ExpositoryIngestionSummary:
        """Extract extended propositions and commit Concept/Belief/Edge nodes transactionally."""
        props = extract_propositions_from_text(
            text,
            source_file=source_label,
            include_extended=True,
            parser=parser,
            ontology_store=ontology_store,
        )
        if not props:
            return ExpositoryIngestionSummary()

        target_ws = (
            getattr(tiered_workspace, "persistent", None)
            if tiered_workspace is not None
            else workspace
        ) or workspace
        grant = getattr(tiered_workspace, "_system_grant", None) or getattr(
            workspace, "_system_grant", None
        )

        nodes_committed, edges_committed, _ = commit_propositions(
            props,
            target_ws,
            ontology_store=ontology_store,
            grant=grant,
            author="discourse_expository",
        )
        if tiered_workspace is not None and hasattr(tiered_workspace, "notify_commit"):
            tiered_workspace.notify_commit()

        # Collect distinct concepts and register them for follow-up coreference
        concepts: list[str] = []
        seen_c: set[str] = set()
        for p in props:
            for raw_term in (p.subject, p.obj):
                clean = raw_term.replace("_", " ").strip()
                if clean and clean not in seen_c:
                    seen_c.add(clean)
                    concepts.append(clean)

        if reference_resolver is not None:
            for concept in reversed(concepts[:5]):
                reference_resolver.register_topic(concept)

        sample_str = "; ".join(
            f"{p.subject.replace('_', ' ')} {_humanize_relation(p.relation)} {p.obj.replace('_', ' ')}"
            for p in props[:4]
        )
        summary = f"Understood. Extracted and stored {len(props)} structured facts ({sample_str})."
        return ExpositoryIngestionSummary(
            propositions=props,
            concepts=concepts,
            nodes_committed=nodes_committed,
            edges_committed=edges_committed,
            summary_text=summary,
        )
