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
    r"^(?:however|furthermore|moreover|therefore|consequently|additionally|"
    r"in\s+addition|in\s+fact|in\s+general|typically|generally)\s*,\s*",
    re.IGNORECASE,
)
_EXEMPLIFY_RE = re.compile(
    r"\b([a-zA-Z][a-zA-Z0-9_-]{2,28}(?:\s+[a-zA-Z][a-zA-Z0-9_-]{2,28})?)\s+"
    r"(?:such\s+as|including|like)\s+"
    r"([^.!?;]+?)(?:\s+\bto\s+(?:maintain|ensure|achieve|prevent|enable|improve|support|reduce)\b|$)",
    re.IGNORECASE,
)
_PURPOSE_RE = re.compile(
    r"\b([a-zA-Z][a-zA-Z0-9_-]{2,28}(?:\s+[a-zA-Z][a-zA-Z0-9_-]{2,28})?)\s+"
    r"(?:(?:such\s+as|including|like)\s+[^.!?;]+?\s+)?"
    r"\bto\s+(?:maintain|ensure|achieve|prevent|enable|improve|support|reduce)\s+"
    r"(?:the\s+|a\s+|an\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\b",
    re.IGNORECASE,
)
_ENABLES_RE = re.compile(
    r"^(?:the\s+|a\s+|an\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\s+"
    r"(?:(?:can|may|will)\s+)?(?:improves?|enables?|maintains?|supports?|reduces?|prevents?|faces?|designs?)\s+"
    r"(?:the\s+|a\s+|an\s+|their\s+|its\s+|many\s+|often\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\b",
    re.IGNORECASE,
)
_REL_CLAUSE_ENABLES_RE = re.compile(
    r"\b(?:who|that|which)\s+(?:practice|use|apply|build|design)\s+([a-zA-Z][a-zA-Z0-9_-]{2,28})\s+"
    r"(?:improves?|enables?|maintains?|supports?|reduces?|prevents?)\s+"
    r"(?:the\s+|a\s+|an\s+|their\s+|its\s+)?([a-zA-Z][a-zA-Z0-9_-]{2,28})\b",
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
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip().lower()).strip("_-")


def _singularize_basic(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _strip_intro_clause(sent: str) -> str:
    """Strip single leading adverb/PP or subordinate clause before the first comma."""
    stripped = _INTRO_STRIP_RE.sub("", sent).strip()
    if "," in stripped:
        prefix, rest = stripped.split(",", 1)
        first_word = prefix.strip().split()[0].lower() if prefix.strip() else ""
        if (
            first_word
            in {
                "in",
                "on",
                "at",
                "by",
                "for",
                "with",
                "during",
                "while",
                "when",
                "because",
                "since",
                "although",
                "though",
                "if",
                "unless",
            }
            and len(rest.strip().split()) >= 3
        ):
            return rest.strip()
    return stripped


def extract_propositions_from_text(
    text: str,
    source_file: str = "",
    *,
    include_extended: bool = False,
) -> list[ExtractedProposition]:
    """Deterministically extract structured propositions from prose.

    With ``include_extended=False`` (default), extracts core ``is_a``, ``causes``,
    and ``requires`` triples for high-precision batch reading. With
    ``include_extended=True``, also extracts exemplification (``such as`` / ``like``),
    purpose (``to maintain ...``), enablement verbs (``improves`` / ``maintains``),
    and subordinate clauses for live multi-sentence dialogue & essay comprehension.
    """
    found: list[ExtractedProposition] = []
    seen: set[tuple[str, str, str]] = set()
    max_len = 600 if include_extended else 220

    def _add(subj_raw: str, rel: str, obj_raw: str, sentence: str) -> bool:
        subj = _clean_term(subj_raw)
        obj = _clean_term(obj_raw)
        if (
            not subj
            or not obj
            or subj == obj
            or subj in _SKIP_SUBJECTS
            or obj in _SKIP_SUBJECTS
            or len(subj) < 2
            or len(obj) < 2
        ):
            return False
        key = (subj, rel, obj)
        if key in seen:
            return False
        seen.add(key)
        found.append(
            ExtractedProposition(
                subject=subj,
                relation=rel,
                obj=obj,
                source_file=source_file,
                sentence=sentence[:200],
            )
        )
        return True

    for raw_sent in _SENT_SPLIT_RE.split(text):
        raw_trimmed = raw_sent.strip()
        # 1. Check structured glossary definition pattern first
        gm = _GLOSSARY_IS_A_RE.match(raw_trimmed)
        if gm:
            _add(gm.group(1), "is_a", gm.group(2), raw_trimmed[:120])
            continue

        sent = raw_trimmed.lstrip("#>-*` ")
        if len(sent) < 12 or len(sent) > max_len:
            continue

        sent_clean = (
            _strip_intro_clause(sent) if include_extended else _INTRO_STRIP_RE.sub("", sent)
        )

        matched_primary = False
        for rel, pattern in (
            ("is_a", _IS_A_RE),
            ("causes", _CAUSES_RE),
            ("requires", _REQUIRES_RE),
        ):
            m = pattern.match(sent_clean)
            if not m:
                continue
            if _add(m.group(1), rel, m.group(2), sent):
                matched_primary = True
            break

        if not include_extended:
            continue

        # Extended patterns for long paragraphs / essays
        if not matched_primary:
            m_rel = _REL_CLAUSE_ENABLES_RE.search(sent_clean)
            if m_rel:
                _add(m_rel.group(1), "enables", m_rel.group(2), sent)
            else:
                m_en = _ENABLES_RE.match(sent_clean)
                if m_en:
                    _add(m_en.group(1), "enables", m_en.group(2), sent)

        # Exemplification: "many challenges such as X, Y, and Z" / "consensus algorithms like Raft and Paxos"
        for m_ex in _EXEMPLIFY_RE.finditer(sent_clean):
            raw_head = m_ex.group(1).strip()
            # Filter leading quantifiers ("many challenges" -> "challenges")
            head_words = [
                w
                for w in raw_head.split()
                if w.lower()
                not in {"many", "several", "various", "some", "these", "those", "the", "other"}
            ]
            if not head_words:
                continue
            head_singular = _singularize_basic("_".join(head_words).lower())
            head_tail = _singularize_basic(head_words[-1].lower())
            items_chunk = re.split(r",|\band\b|\bor\b", m_ex.group(2), flags=re.IGNORECASE)
            for item_raw in items_chunk:
                item_clean = item_raw.strip().strip(".,;:")
                # Drop leading determiners inside items
                item_words = [
                    w
                    for w in item_clean.split()
                    if w.lower() not in {"a", "an", "the", "both", "such", "other"}
                ]
                if not item_words or len(item_words) > 4:
                    continue
                item_term = "_".join(item_words).lower()
                _add(item_term, "is_a", head_singular, sent)
                if head_tail != head_singular:
                    _add(item_term, "is_a", head_tail, sent)

        # Purpose clauses: "<mechanism> ... to maintain <target>"
        for m_purp in _PURPOSE_RE.finditer(sent_clean):
            mech_words = [
                w
                for w in m_purp.group(1).strip().split()
                if w.lower() not in {"a", "an", "the", "many", "several", "design", "use", "apply"}
            ]
            if mech_words:
                mech_term = _singularize_basic("_".join(mech_words).lower())
                _add(mech_term, "enables", m_purp.group(2), sent)

        # Subordinate Because clauses: "Because <clause1>, <clause2>"
        m_bec = re.match(r"^because\s+([^,]+),\s*(.+)$", sent.strip(), flags=re.IGNORECASE)
        if m_bec:
            c1_words = [
                w.lower()
                for w in re.findall(r"[a-zA-Z]+", m_bec.group(1))
                if w.lower() not in _SKIP_SUBJECTS
                and w.lower() not in {"can", "may", "will", "be", "is", "are"}
            ]
            c2_words = [
                w.lower()
                for w in re.findall(r"[a-zA-Z]+", m_bec.group(2))
                if w.lower() not in _SKIP_SUBJECTS
                and w.lower() not in {"can", "may", "will", "be", "is", "are", "to", "like", "and"}
            ]
            if c1_words and len(c2_words) >= 2:
                # Link antecedent subject/event to consequent mechanism
                _add(c1_words[0], "causes", c2_words[1] if len(c2_words) > 2 else c2_words[0], sent)

    return found


def commit_propositions(
    propositions: list[ExtractedProposition],
    workspace: Any,
    *,
    ontology_store: Any | None = None,
    grant: Any | None = None,
    author: str = "t4_reading_curriculum",
    batch_size: int = 100,
) -> tuple[int, int, int]:
    """Commit extracted propositions into HCIR workspace & optional ontology store atomically.

    Returns ``(nodes_committed, edges_committed, store_hypernyms_added)``.
    """
    node_ops: list[TransactionOperation] = []
    edge_ops: list[TransactionOperation] = []
    created_concepts: set[str] = set()
    existing_edges = getattr(getattr(workspace, "graph", None), "edges", {})
    store_records: list[LexicalOntologyRecord] = []

    edge_type_map = {
        "is_a": HCIREdgeType.GENERALIZES,
        "causes": HCIREdgeType.CAUSES,
        "requires": HCIREdgeType.REQUIRES,
        "enables": HCIREdgeType.CAUSES,
    }

    for prop in propositions:
        src_id = f"t4_concept_{prop.subject}"
        dst_id = f"t4_concept_{prop.obj}"
        for term, cid in ((prop.subject, src_id), (prop.obj, dst_id)):
            if cid not in created_concepts:
                created_concepts.add(cid)
                display = term.replace("_", " ")
                defn = ""
                if term == prop.subject and prop.relation == "is_a":
                    defn = f"a {prop.obj.replace('_', ' ')} ({prop.sentence[:120]})"
                cnode = ConceptNode(
                    id=cid,
                    label=display,
                    definition=defn,
                    domain="t4_reading_curriculum",
                    tags=["t4_reading", display],
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
            statement=f"{prop.subject.replace('_', ' ')} {prop.relation} {prop.obj.replace('_', ' ')}",
            confidence=0.82,
            tags=["t4_reading_curriculum", prop.relation, prop.source_file],
            properties={"source_file": prop.source_file, "sentence": prop.sentence[:160]},
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
                edge_type=edge_type_map.get(prop.relation, HCIREdgeType.SUPPORTS),
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
            subj_display = prop.subject.replace("_", " ")
            obj_display = prop.obj.replace("_", " ")
            store_records.append(
                LexicalOntologyRecord(
                    lemma=subj_display,
                    pos="noun",
                    language="en",
                    definition=prop.sentence[:200],
                    hypernyms=(obj_display,),
                    domain="t4_reading",
                )
            )

    store_hypernyms_added = 0
    if ontology_store is not None and store_records:
        _, store_hypernyms_added = ontology_store.upsert_many(store_records, source=author)

    nodes_committed = 0
    for i in range(0, len(node_ops), batch_size):
        batch = node_ops[i : i + batch_size]
        receipt = workspace.commit_operations(batch, author=author, grant=grant)
        if getattr(receipt, "is_committed", False):
            nodes_committed += len(batch)

    edges_committed = 0
    for i in range(0, len(edge_ops), batch_size):
        batch = edge_ops[i : i + batch_size]
        receipt = workspace.commit_operations(batch, author=author, grant=grant)
        if getattr(receipt, "is_committed", False):
            edges_committed += len(batch)

    return nodes_committed, edges_committed, store_hypernyms_added


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

        nodes_committed, edges_committed, store_hypernyms_added = commit_propositions(
            propositions,
            workspace,
            ontology_store=ontology_store,
            grant=grant,
            author="t4_reading_curriculum",
            batch_size=batch_size,
        )
        stats.nodes_committed = nodes_committed
        stats.edges_committed = edges_committed
        stats.store_hypernyms_added = store_hypernyms_added

        for idx, prop in enumerate(propositions[:10]):
            stats.sample_propositions.append(
                {
                    "subject": prop.subject,
                    "relation": prop.relation,
                    "object": prop.obj,
                    "source": prop.source_file,
                }
            )

        stats.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return stats
