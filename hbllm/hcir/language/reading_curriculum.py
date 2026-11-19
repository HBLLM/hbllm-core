"""Track T4: Open-Ontology Reading Curriculum Ingestor for HCIR.

Reads local documentation, textbooks, and prose corpora, deterministically
extracts taxonomic (``is-a``), causal (``causes`` / ``triggers`` / ``leads to``),
and dependency (``requires`` / ``depends on``) propositions, and commits them
transactionally into ``TieredWorkspace.persistent`` and ``LexicalOntologyStore``.

Zero LLM dependencies; 100% deterministic CPU execution.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hcir.graph import HCIREdge
from hbllm.hcir.language.ontology_bootstrap import LexicalOntologyRecord
from hbllm.hcir.nodes import BeliefNode, ConceptNode, HCIREdgeType
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

logger = logging.getLogger(__name__)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_IS_A_RE = re.compile(
    r"^(?:the\s+|a\s+|an\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\s+is\s+(?:a|an)\s+(?:[a-zA-Z-]+\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\b",
    re.IGNORECASE,
)
_CAUSES_RE = re.compile(
    r"^(?:the\s+|a\s+|an\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\s+(?:causes|triggers|leads\s+to|produces)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\b",
    re.IGNORECASE,
)
_REQUIRES_RE = re.compile(
    r"^(?:the\s+|a\s+|an\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\s+(?:requires|depends\s+on|needs)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\b",
    re.IGNORECASE,
)
_GLOSSARY_IS_A_RE = re.compile(
    r"^\*?\s*\*\*([a-zA-Z][a-zA-Z0-9_\s-]{2,28})\*\*:\s+(?:the\s+|a\s+|an\s+)?(?:[a-zA-Z-]+\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\b",
    re.IGNORECASE,
)
_INTRO_STRIP_RE = re.compile(
    r"^(?:however|furthermore|moreover|therefore|in\s+addition|in\s+fact|typically|generally)\s*,\s*",
    re.IGNORECASE,
)

_SKIP_SUBJECTS = frozenset(
    {
        "this",
        "that",
        "there",
        "here",
        "which",
        "what",
        "when",
        "where",
        "they",
        "these",
        "those",
        "it",
        "he",
        "she",
        "we",
        "you",
        "who",
        "whom",
        "whose",
        "i",
        "me",
        "him",
        "her",
        "us",
        "them",
        "and",
        "or",
        "but",
        "with",
        "from",
        "for",
        "to",
        "at",
        "by",
        "about",
        "more",
        "some",
        "many",
        "such",
        "each",
        "every",
        "all",
        "both",
        "either",
        "neither",
        "very",
        "also",
        "then",
        "thus",
    }
)


@dataclass(frozen=True)
class ExtractedProposition:
    """A structured taxonomic, causal, or dependency proposition extracted from text."""

    subject: str
    relation: str  # "is_a" | "causes" | "requires"
    obj: str
    source_file: str
    sentence: str


@dataclass
class ReadingCurriculumStats:
    """Summary metrics for a Track T4 reading run."""

    files_read: int = 0
    sentences_scanned: int = 0
    taxonomic_extracted: int = 0
    causal_extracted: int = 0
    dependency_extracted: int = 0
    nodes_committed: int = 0
    edges_committed: int = 0
    store_hypernyms_added: int = 0
    elapsed_ms: float = 0.0
    sample_propositions: list[dict[str, str]] = field(default_factory=list)


def _clean_term(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "", raw.strip().lower())


def extract_propositions_from_text(text: str, source_file: str = "") -> list[ExtractedProposition]:
    """Deterministically extract ``is_a``, ``causes``, and ``requires`` propositions."""
    found: list[ExtractedProposition] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_sent in _SENT_SPLIT_RE.split(text):
        raw_trimmed = raw_sent.strip()
        # 1. Check structured glossary definition pattern first
        gm = _GLOSSARY_IS_A_RE.match(raw_trimmed)
        if gm:
            subj = _clean_term(gm.group(1))
            obj = _clean_term(gm.group(2))
            if (
                subj
                and obj
                and subj != obj
                and subj not in _SKIP_SUBJECTS
                and obj not in _SKIP_SUBJECTS
            ):
                key = (subj, "is_a", obj)
                if key not in seen:
                    seen.add(key)
                    found.append(
                        ExtractedProposition(
                            subject=subj,
                            relation="is_a",
                            obj=obj,
                            source_file=source_file,
                            sentence=raw_trimmed[:120],
                        )
                    )
            continue

        sent = raw_trimmed.lstrip("#>-*` ")
        if len(sent) < 12 or len(sent) > 220:
            continue

        sent_clean = _INTRO_STRIP_RE.sub("", sent)

        for rel, pattern in (
            ("is_a", _IS_A_RE),
            ("causes", _CAUSES_RE),
            ("requires", _REQUIRES_RE),
        ):
            m = pattern.match(sent_clean)
            if not m:
                continue
            subj = _clean_term(m.group(1))
            obj = _clean_term(m.group(2))
            if (
                not subj
                or not obj
                or subj == obj
                or subj in _SKIP_SUBJECTS
                or obj in _SKIP_SUBJECTS
            ):
                continue
            key = (subj, rel, obj)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                ExtractedProposition(
                    subject=subj,
                    relation=rel,
                    obj=obj,
                    source_file=source_file,
                    sentence=sent,
                )
            )
            break

    return found


class ReadingCurriculumIngestor:
    """Ingests structured knowledge from docs/books into HCIR transactionally."""

    def ingest_directories(
        self,
        directories: list[Path | str],
        workspace: Any,
        *,
        ontology_store: Any | None = None,
        grant: Any | None = None,
        max_files: int = 100,
        max_propositions: int = 300,
        batch_size: int = 100,
    ) -> ReadingCurriculumStats:
        t0 = time.perf_counter()
        stats = ReadingCurriculumStats()
        propositions: list[ExtractedProposition] = []
        seen_global: set[tuple[str, str, str]] = set()

        for raw_dir in directories:
            root = Path(raw_dir)
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if stats.files_read >= max_files or len(propositions) >= max_propositions:
                    break
                if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                stats.files_read += 1
                stats.sentences_scanned += len(_SENT_SPLIT_RE.split(text))
                for prop in extract_propositions_from_text(text, source_file=path.name):
                    key = (prop.subject, prop.relation, prop.obj)
                    if key in seen_global:
                        continue
                    seen_global.add(key)
                    propositions.append(prop)
                    if prop.relation == "is_a":
                        stats.taxonomic_extracted += 1
                    elif prop.relation == "causes":
                        stats.causal_extracted += 1
                    else:
                        stats.dependency_extracted += 1
                    if len(propositions) >= max_propositions:
                        break

        # Commit extracted concepts, beliefs, and causal/dependency/taxonomic edges
        node_ops: list[TransactionOperation] = []
        edge_ops: list[TransactionOperation] = []
        created_concepts: set[str] = set()
        existing_edges = getattr(getattr(workspace, "graph", None), "edges", {})
        store_records: list[LexicalOntologyRecord] = []

        edge_type_map = {
            "is_a": HCIREdgeType.GENERALIZES,
            "causes": HCIREdgeType.CAUSES,
            "requires": HCIREdgeType.REQUIRES,
        }

        for idx, prop in enumerate(propositions):
            src_id = f"t4_concept_{prop.subject}"
            dst_id = f"t4_concept_{prop.obj}"
            for term, cid in ((prop.subject, src_id), (prop.obj, dst_id)):
                if cid not in created_concepts:
                    created_concepts.add(cid)
                    cnode = ConceptNode(
                        id=cid,
                        label=term,
                        definition=f"Concept extracted by Track T4 reading ({prop.source_file})",
                        domain="t4_reading_curriculum",
                        tags=["t4_reading"],
                    )
                    node_ops.append(
                        TransactionOperation(
                            op=TransactionOp.UPSERT_NODE,
                            node_id=cnode.id,
                            node_data=cnode.model_dump(),
                        )
                    )

            belief = BeliefNode(
                id=f"t4_belief_{prop.subject}_{prop.relation}_{prop.obj}",
                statement=f"{prop.subject} {prop.relation} {prop.obj}",
                confidence=0.82,
                tags=["t4_reading_curriculum", prop.relation, prop.source_file],
                properties={"source_file": prop.source_file, "sentence": prop.sentence[:120]},
            )
            node_ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=belief.id,
                    node_data=belief.model_dump(),
                )
            )

            edge_id = f"t4_{prop.relation}_{src_id}__{dst_id}"
            if edge_id not in existing_edges:
                edge = HCIREdge(
                    id=edge_id,
                    edge_type=edge_type_map[prop.relation],
                    sources=[src_id],
                    targets=[dst_id],
                    weight=0.82,
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
                        domain="t4_reading",
                    )
                )

            if idx < 10:
                stats.sample_propositions.append(
                    {
                        "subject": prop.subject,
                        "relation": prop.relation,
                        "object": prop.obj,
                        "source": prop.source_file,
                    }
                )

        if ontology_store is not None and store_records:
            _, edges_added = ontology_store.upsert_many(
                store_records, source="t4_reading_curriculum"
            )
            stats.store_hypernyms_added = edges_added

        for i in range(0, len(node_ops), batch_size):
            batch = node_ops[i : i + batch_size]
            receipt = workspace.commit_operations(
                batch, author="t4_reading_curriculum", grant=grant
            )
            if getattr(receipt, "is_committed", False):
                stats.nodes_committed += len(batch)

        for i in range(0, len(edge_ops), batch_size):
            batch = edge_ops[i : i + batch_size]
            receipt = workspace.commit_operations(
                batch, author="t4_reading_curriculum", grant=grant
            )
            if getattr(receipt, "is_committed", False):
                stats.edges_committed += len(batch)

        stats.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return stats
