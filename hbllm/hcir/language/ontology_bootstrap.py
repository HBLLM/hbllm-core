"""Track T1: Open-Vocabulary Lexicon & Ontology Bootstrap for HCIR.

Deterministically ingests lexical entries, definitions, thematic verb frames,
and taxonomic hypernym relations into ``TieredWorkspace`` strictly through
``TransactionOperation``s, and hydrates ``EnglishLexicon`` (and multilingual
lexicons) directly from the persisted HCIR graph.

Zero LLM dependencies; runs 100% on CPU.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, HCIREdge
from hbllm.hcir.language.english.lexicon import EnglishLexicalEntry, EnglishLexicon, EnglishPOS
from hbllm.hcir.nodes import ConceptNode, HCIREdgeType, HCIRNodeType
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z]{3,24}")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "have",
        "was",
        "were",
        "are",
        "not",
        "but",
        "you",
        "all",
        "can",
        "had",
        "her",
        "his",
        "one",
        "our",
        "out",
        "they",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "into",
        "more",
        "than",
        "then",
        "them",
        "these",
        "those",
        "there",
        "their",
        "about",
        "would",
        "could",
        "should",
        "been",
        "being",
        "some",
        "such",
        "only",
        "very",
        "also",
        "just",
        "over",
        "after",
        "before",
        "between",
    }
)

_POS_MAP: dict[str, EnglishPOS] = {
    "n": EnglishPOS.NOUN,
    "noun": EnglishPOS.NOUN,
    "v": EnglishPOS.VERB,
    "verb": EnglishPOS.VERB,
    "a": EnglishPOS.ADJ,
    "s": EnglishPOS.ADJ,
    "adj": EnglishPOS.ADJ,
    "adjective": EnglishPOS.ADJ,
    "r": EnglishPOS.ADV,
    "adv": EnglishPOS.ADV,
    "adverb": EnglishPOS.ADV,
    "prep": EnglishPOS.PREP,
    "preposition": EnglishPOS.PREP,
}


@dataclass(frozen=True)
class LexicalOntologyRecord:
    """Normalized open-vocabulary lexical & ontological record."""

    lemma: str
    pos: str = "noun"
    language: str = "en"
    definition: str = ""
    hypernyms: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    thematic_roles: tuple[str, ...] = ()
    domain: str = "general"


@dataclass
class BootstrapStats:
    """Metrics returned by Track T1 ingestion runs."""

    source: str
    records_processed: int = 0
    concepts_committed: int = 0
    edges_committed: int = 0
    lexicon_entries_added: int = 0
    elapsed_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _concept_id(language: str, pos: str, lemma: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\u0D80-\u0DFF\u0B80-\u0BFF]+", "_", lemma.strip().lower()).strip(
        "_"
    )
    return f"lex_{language}_{pos}_{clean}"


def _infer_pos_from_morphology(word: str) -> str:
    if word.endswith(("tion", "sion", "ment", "ness", "ity", "ship", "ism", "ist")):
        return "noun"
    if word.endswith(("able", "ible", "ous", "ful", "less", "ive", "al")):
        return "adj"
    if word.endswith("ly") and len(word) > 4:
        return "adv"
    if word.endswith(("ize", "ise", "ate", "ify")):
        return "verb"
    return "noun"


# Built-in open-domain + software/OS/reasoning starter ontology (zero external download needed)
CORE_ONTOLOGY_RECORDS: tuple[LexicalOntologyRecord, ...] = (
    # Abstract / taxonomic roots
    LexicalOntologyRecord("entity", "noun", "en", "Something that exists independently"),
    LexicalOntologyRecord("object", "noun", "en", "A tangible physical thing", ("entity",)),
    LexicalOntologyRecord("artifact", "noun", "en", "An object created by an agent", ("object",)),
    LexicalOntologyRecord("agent", "noun", "en", "An entity capable of acting", ("entity",)),
    LexicalOntologyRecord("location", "noun", "en", "A place or region in space", ("entity",)),
    LexicalOntologyRecord("event", "noun", "en", "Something that occurs in time", ("entity",)),
    LexicalOntologyRecord("information", "noun", "en", "Structured knowledge or data", ("entity",)),
    # Software, OS & engineering nouns
    LexicalOntologyRecord(
        "file",
        "noun",
        "en",
        "A named unit of stored digital data",
        ("artifact",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "directory",
        "noun",
        "en",
        "A container organizing files",
        ("location",),
        ("folder",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "folder", "noun", "en", "A directory containing files", ("directory",), domain="software"
    ),
    LexicalOntologyRecord(
        "repository",
        "noun",
        "en",
        "A version-controlled project store",
        ("directory",),
        ("repo",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "code",
        "noun",
        "en",
        "Instructions written in a programming language",
        ("information",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "program", "noun", "en", "An executable software system", ("artifact",), domain="software"
    ),
    LexicalOntologyRecord(
        "module", "noun", "en", "A self-contained unit of code", ("artifact",), domain="software"
    ),
    LexicalOntologyRecord(
        "function", "noun", "en", "A callable block of code", ("artifact",), domain="software"
    ),
    LexicalOntologyRecord(
        "class",
        "noun",
        "en",
        "A blueprint defining objects and methods",
        ("artifact",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "variable",
        "noun",
        "en",
        "A named storage binding for a value",
        ("information",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "test", "noun", "en", "An automated verification check", ("event",), domain="software"
    ),
    LexicalOntologyRecord(
        "error",
        "noun",
        "en",
        "A failure or invalid runtime condition",
        ("event",),
        ("bug",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "bug", "noun", "en", "A defect in software behavior", ("error",), domain="software"
    ),
    LexicalOntologyRecord(
        "server",
        "noun",
        "en",
        "A computer system or program that provides data or services to other computers over a network",
        ("computer", "artifact"),
        domain="software",
    ),
    LexicalOntologyRecord(
        "microscope",
        "noun",
        "en",
        "An optical instrument with lenses used as a magnifier for viewing very small objects",
        ("instrument", "magnifier"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "telescope",
        "noun",
        "en",
        "An optical instrument that serves as a magnifier for viewing distant objects",
        ("instrument", "magnifier"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "algorithm",
        "noun",
        "en",
        "A finite rule or set of steps designed to solve a problem or perform a computation",
        ("rule", "information"),
        domain="software",
    ),
    LexicalOntologyRecord(
        "focus",
        "noun",
        "en",
        "The state of producing clear attention or visual definition, caused by spiking stimulus",
        ("state", "information"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "stimulus",
        "noun",
        "en",
        "An event or agent evoking a functional reaction",
        ("event", "information"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "spiking",
        "noun",
        "en",
        "Rapid change in membrane potential of a neuron",
        ("event", "information"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "photosynthesis",
        "noun",
        "en",
        "A biological process where plants convert light energy into chemical energy",
        ("process", "event"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "gravity",
        "noun",
        "en",
        "The universal force of attraction between masses of matter",
        ("force", "phenomenon"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "neuron",
        "noun",
        "en",
        "A specialized brain cell transmitting nerve impulses",
        ("cell", "artifact"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "atom",
        "noun",
        "en",
        "The basic unit of a chemical element containing a nucleus surrounded by electrons",
        ("entity", "particle"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "energy",
        "noun",
        "en",
        "The capacity of a physical system to do work or produce heat and power",
        ("property", "quantity"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "molecule",
        "noun",
        "en",
        "A group of atoms bonded together representing the smallest fundamental unit of a chemical compound",
        ("particle", "entity"),
        domain="science",
    ),
    LexicalOntologyRecord(
        "cluster",
        "noun",
        "en",
        "A group of coordinated compute nodes",
        ("location",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "database",
        "noun",
        "en",
        "A persistent structured data store",
        ("artifact",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "query",
        "noun",
        "en",
        "A request for information from a store",
        ("information",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "endpoint",
        "noun",
        "en",
        "An addressable network API route",
        ("location",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "token",
        "noun",
        "en",
        "A security credential or lexical unit",
        ("information",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "tenant",
        "noun",
        "en",
        "An isolated customer scope in a system",
        ("agent",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "user", "noun", "en", "A human interacting with a system", ("agent",), domain="software"
    ),
    LexicalOntologyRecord(
        "log",
        "noun",
        "en",
        "A chronological record of system events",
        ("information",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "config",
        "noun",
        "en",
        "Settings governing system behavior",
        ("information",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "branch",
        "noun",
        "en",
        "An independent line of version history",
        ("information",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "commit",
        "noun",
        "en",
        "An atomic snapshot of repository changes",
        ("event",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "patch",
        "noun",
        "en",
        "A targeted set of file modifications",
        ("artifact",),
        domain="software",
    ),
    LexicalOntologyRecord(
        "workspace",
        "noun",
        "en",
        "Active working environment for tasks",
        ("location",),
        domain="software",
    ),
    # Cognitive & everyday nouns
    LexicalOntologyRecord(
        "memory",
        "noun",
        "en",
        "Stored beliefs, episodes, or skills",
        ("information",),
        domain="cognition",
    ),
    LexicalOntologyRecord(
        "belief",
        "noun",
        "en",
        "An epistemic proposition held with confidence",
        ("information",),
        domain="cognition",
    ),
    LexicalOntologyRecord(
        "goal", "noun", "en", "A desired target state", ("information",), domain="cognition"
    ),
    LexicalOntologyRecord(
        "plan",
        "noun",
        "en",
        "An ordered sequence of actions to reach a goal",
        ("information",),
        domain="cognition",
    ),
    LexicalOntologyRecord(
        "skill",
        "noun",
        "en",
        "A reusable procedural capability",
        ("information",),
        domain="cognition",
    ),
    LexicalOntologyRecord(
        "rule",
        "noun",
        "en",
        "A constraint or causal regularity",
        ("information",),
        domain="cognition",
    ),
    LexicalOntologyRecord(
        "evidence",
        "noun",
        "en",
        "An observation supporting or refuting a belief",
        ("information",),
        domain="cognition",
    ),
    LexicalOntologyRecord(
        "question",
        "noun",
        "en",
        "An utterance requesting information",
        ("information",),
        domain="cognition",
    ),
    LexicalOntologyRecord(
        "answer",
        "noun",
        "en",
        "A response addressing a question",
        ("information",),
        domain="cognition",
    ),
    LexicalOntologyRecord(
        "book", "noun", "en", "A written work of text", ("artifact",), domain="general"
    ),
    LexicalOntologyRecord(
        "document", "noun", "en", "A structured written record", ("artifact",), domain="general"
    ),
    LexicalOntologyRecord(
        "room", "noun", "en", "An enclosed physical space", ("location",), domain="general"
    ),
    LexicalOntologyRecord(
        "door",
        "noun",
        "en",
        "A movable barrier controlling passage",
        ("artifact",),
        domain="general",
    ),
    LexicalOntologyRecord(
        "key", "noun", "en", "An instrument that unlocks a barrier", ("artifact",), domain="general"
    ),
    LexicalOntologyRecord(
        "sensor",
        "noun",
        "en",
        "A device measuring environmental state",
        ("artifact",),
        domain="general",
    ),
    LexicalOntologyRecord(
        "robot", "noun", "en", "An embodied autonomous machine", ("agent",), domain="general"
    ),
    # Action & reasoning verbs with thematic frames
    LexicalOntologyRecord(
        "read",
        "verb",
        "en",
        "Inspect content from a file or document",
        (),
        (),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "write",
        "verb",
        "en",
        "Store content to a file or target",
        (),
        (),
        ("AGENT", "PATIENT", "DESTINATION"),
        "software",
    ),
    LexicalOntologyRecord(
        "create",
        "verb",
        "en",
        "Bring a new artifact into existence",
        (),
        (),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "delete",
        "verb",
        "en",
        "Remove an existing artifact",
        (),
        ("remove",),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "update",
        "verb",
        "en",
        "Modify an existing artifact or state",
        (),
        (),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "fix",
        "verb",
        "en",
        "Repair an error or broken state",
        (),
        (),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "run",
        "verb",
        "en",
        "Execute a command, test, or program",
        (),
        ("execute",),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "execute",
        "verb",
        "en",
        "Carry out a program or instruction",
        (),
        ("run",),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "search",
        "verb",
        "en",
        "Locate matching items in a scope",
        (),
        ("find",),
        ("AGENT", "PATIENT", "LOCATION"),
        "software",
    ),
    LexicalOntologyRecord(
        "find",
        "verb",
        "en",
        "Discover the location or value of a target",
        (),
        ("search",),
        ("AGENT", "PATIENT"),
        "general",
    ),
    LexicalOntologyRecord(
        "list",
        "verb",
        "en",
        "Enumerate items inside a container",
        (),
        (),
        ("AGENT", "LOCATION"),
        "software",
    ),
    LexicalOntologyRecord(
        "check",
        "verb",
        "en",
        "Inspect or verify a condition",
        (),
        ("verify",),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "verify",
        "verb",
        "en",
        "Confirm correctness against constraints",
        (),
        ("check",),
        ("AGENT", "PATIENT"),
        "cognition",
    ),
    LexicalOntologyRecord(
        "explain",
        "verb",
        "en",
        "Describe causes or structure of a concept",
        (),
        (),
        ("AGENT", "PATIENT"),
        "cognition",
    ),
    LexicalOntologyRecord(
        "summarize",
        "verb",
        "en",
        "Condense information to key points",
        (),
        (),
        ("AGENT", "PATIENT"),
        "cognition",
    ),
    LexicalOntologyRecord(
        "compare",
        "verb",
        "en",
        "Evaluate similarities and differences",
        (),
        (),
        ("AGENT", "PATIENT"),
        "cognition",
    ),
    LexicalOntologyRecord(
        "learn",
        "verb",
        "en",
        "Acquire knowledge or skills from experience",
        (),
        (),
        ("AGENT", "PATIENT"),
        "cognition",
    ),
    LexicalOntologyRecord(
        "train",
        "verb",
        "en",
        "Improve a model or skill on data",
        (),
        (),
        ("AGENT", "PATIENT"),
        "cognition",
    ),
    LexicalOntologyRecord(
        "open",
        "verb",
        "en",
        "Make an object or barrier accessible",
        (),
        (),
        ("AGENT", "PATIENT"),
        "general",
    ),
    LexicalOntologyRecord(
        "close", "verb", "en", "Shut an object or resource", (), (), ("AGENT", "PATIENT"), "general"
    ),
    LexicalOntologyRecord(
        "unlock",
        "verb",
        "en",
        "Release a lock using an instrument",
        (),
        (),
        ("AGENT", "PATIENT", "INSTRUMENT"),
        "general",
    ),
    LexicalOntologyRecord(
        "build",
        "verb",
        "en",
        "Compile or assemble an artifact",
        (),
        (),
        ("AGENT", "PATIENT"),
        "software",
    ),
    LexicalOntologyRecord(
        "deploy",
        "verb",
        "en",
        "Release a system to an environment",
        (),
        (),
        ("AGENT", "PATIENT", "DESTINATION"),
        "software",
    ),
    # Adjectives
    LexicalOntologyRecord(
        "active", "adj", "en", "Currently running or operational", domain="general"
    ),
    LexicalOntologyRecord("broken", "adj", "en", "Failing to function correctly", domain="general"),
    LexicalOntologyRecord(
        "valid", "adj", "en", "Satisfying required constraints", domain="cognition"
    ),
    LexicalOntologyRecord(
        "invalid", "adj", "en", "Violating required constraints", domain="cognition"
    ),
    LexicalOntologyRecord("fast", "adj", "en", "Operating with low latency", domain="general"),
    LexicalOntologyRecord("slow", "adj", "en", "Operating with high latency", domain="general"),
    LexicalOntologyRecord("empty", "adj", "en", "Containing no elements", domain="general"),
    LexicalOntologyRecord("locked", "adj", "en", "Secured against access", domain="general"),
    LexicalOntologyRecord(
        "safe", "adj", "en", "Free from hazard or policy violation", domain="cognition"
    ),
    LexicalOntologyRecord(
        "local", "adj", "en", "Residing on the current machine", domain="software"
    ),
    LexicalOntologyRecord("remote", "adj", "en", "Residing across a network", domain="software"),
)


class OntologyBootstrapper:
    """Deterministic, transaction-governed Track T1 lexicon & ontology ingestor."""

    def commit_records(
        self,
        records: Sequence[LexicalOntologyRecord],
        workspace: Any | None,
        *,
        source: str = "custom",
        batch_size: int = 200,
        author: str = "t1_ontology_bootstrap",
        lexicon: EnglishLexicon | None = None,
        grant: Any | None = None,
        ontology_store: Any | None = None,
        workspace_limit: int | None = None,
    ) -> BootstrapStats:
        """Commit lexical ontology records to ``ontology_store`` and/or ``workspace``."""
        t0 = time.perf_counter()
        stats = BootstrapStats(source=source, records_processed=len(records))

        if ontology_store is not None:
            stored_entries, stored_edges = ontology_store.upsert_many(records, source=source)
            stats.extra["store_entries_upserted"] = stored_entries
            stats.extra["store_hypernym_edges"] = stored_edges

        ws_records = records[:workspace_limit] if workspace_limit is not None else records
        if workspace is None or not ws_records:
            stats.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return stats

        ops: list[TransactionOperation] = []
        created_ids: dict[tuple[str, str], str] = {}

        # Pass 1: ConceptNodes
        for rec in ws_records:
            lemma = rec.lemma.strip().lower()
            if not lemma:
                continue
            pos = rec.pos.strip().lower() or "noun"
            cid = _concept_id(rec.language, pos, lemma)
            created_ids[(rec.language, lemma)] = cid
            node = ConceptNode(
                id=cid,
                label=lemma,
                definition=rec.definition or f"{lemma} ({pos})",
                domain="lexical_ontology",
                properties={
                    "lemma": lemma,
                    "pos": pos,
                    "language": rec.language,
                    "hypernyms": list(rec.hypernyms),
                    "synonyms": list(rec.synonyms),
                    "thematic_roles": list(rec.thematic_roles),
                    "semantic_domain": rec.domain,
                    "source": source,
                },
                tags=["t1_ontology", rec.language, pos, rec.domain],
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node.id,
                    node_data=node.model_dump(),
                )
            )

        # Pass 2: Taxonomic GENERALIZES edges (child -> hypernym)
        existing_edges = getattr(getattr(workspace, "graph", None), "edges", {})
        edge_ops: list[TransactionOperation] = []
        for rec in ws_records:
            lemma = rec.lemma.strip().lower()
            pos = rec.pos.strip().lower() or "noun"
            src_id = _concept_id(rec.language, pos, lemma)
            for hyp in rec.hypernyms:
                hyp_clean = hyp.strip().lower()
                dst_id = created_ids.get((rec.language, hyp_clean))
                edge_id = f"gen_{src_id}__{dst_id}"
                if dst_id and dst_id != src_id and edge_id not in existing_edges:
                    edge = HCIREdge(
                        id=edge_id,
                        edge_type=HCIREdgeType.GENERALIZES,
                        sources=[src_id],
                        targets=[dst_id],
                        weight=1.0,
                        properties={"source": source, "relation": "hypernym"},
                    )
                    edge_ops.append(
                        TransactionOperation(
                            op=TransactionOp.ADD_EDGE,
                            edge_data=edge.model_dump(),
                        )
                    )

        # Commit nodes first in batches, then edges in batches
        for i in range(0, len(ops), batch_size):
            batch = ops[i : i + batch_size]
            receipt = workspace.commit_operations(batch, author=author, grant=grant)
            if getattr(receipt, "is_committed", False):
                stats.concepts_committed += len(batch)

        for i in range(0, len(edge_ops), batch_size):
            batch = edge_ops[i : i + batch_size]
            receipt = workspace.commit_operations(batch, author=author, grant=grant)
            if getattr(receipt, "is_committed", False):
                stats.edges_committed += len(batch)

        if lexicon is not None:
            stats.lexicon_entries_added = self.hydrate_lexicon(workspace, lexicon, language="en")

        stats.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return stats

    def ingest_core_ontology(
        self,
        workspace: Any | None,
        *,
        lexicon: EnglishLexicon | None = None,
        grant: Any | None = None,
        ontology_store: Any | None = None,
    ) -> BootstrapStats:
        """Ingest the built-in open-domain + software/OS/reasoning starter ontology."""
        return self.commit_records(
            CORE_ONTOLOGY_RECORDS,
            workspace,
            source="core_ontology",
            lexicon=lexicon,
            grant=grant,
            ontology_store=ontology_store,
        )

    def ingest_kaikki_jsonl(
        self,
        path: Path | str,
        workspace: Any | None,
        *,
        language: str = "en",
        limit: int = 10_000,
        lexicon: EnglishLexicon | None = None,
        grant: Any | None = None,
        ontology_store: Any | None = None,
        workspace_limit: int | None = None,
    ) -> BootstrapStats:
        """Ingest a Kaikki / Wiktionary JSONL dump for any language (en/si/ta)."""
        file_path = Path(path)
        records: list[LexicalOntologyRecord] = []
        seen: set[tuple[str, str]] = set()

        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(records) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                word = str(item.get("word", "")).strip().lower()
                raw_pos = str(item.get("pos", "noun")).strip().lower()
                if not word or " " in word or (word, raw_pos) in seen:
                    continue
                seen.add((word, raw_pos))

                senses = item.get("senses") or []
                gloss = ""
                hypernyms: list[str] = []
                synonyms: list[str] = []
                if senses and isinstance(senses, list):
                    first = senses[0] if isinstance(senses[0], dict) else {}
                    glosses = first.get("glosses") or []
                    if glosses:
                        gloss = str(glosses[0])[:240]
                    for hyp in first.get("hypernyms") or []:
                        if isinstance(hyp, dict) and hyp.get("word"):
                            hypernyms.append(str(hyp["word"]).lower())
                    for syn in first.get("synonyms") or []:
                        if isinstance(syn, dict) and syn.get("word"):
                            synonyms.append(str(syn["word"]).lower())

                records.append(
                    LexicalOntologyRecord(
                        lemma=word,
                        pos=raw_pos,
                        language=language,
                        definition=gloss,
                        hypernyms=tuple(hypernyms[:3]),
                        synonyms=tuple(synonyms[:5]),
                        domain="wiktionary",
                    )
                )

        return self.commit_records(
            records,
            workspace,
            source=f"kaikki:{file_path.name}",
            lexicon=lexicon,
            grant=grant,
            ontology_store=ontology_store,
            workspace_limit=workspace_limit,
        )

    def ingest_wordnet(
        self,
        workspace: Any | None,
        *,
        limit: int = 120_000,
        lexicon: EnglishLexicon | None = None,
        grant: Any | None = None,
        ontology_store: Any | None = None,
        workspace_limit: int | None = 0,
    ) -> BootstrapStats:
        """Ingest Open English WordNet via ``wn`` or ``nltk.corpus.wordnet``."""
        records: list[LexicalOntologyRecord] = []
        seen: set[tuple[str, str]] = set()

        try:
            import wn  # type: ignore[import-not-found]

            for synset in wn.synsets():
                if len(records) >= limit:
                    break
                lemmas = synset.lemmas()
                if not lemmas:
                    continue
                lemma = lemmas[0].lower().replace(" ", "_")
                pos = str(synset.pos or "n")
                if (lemma, pos) in seen:
                    continue
                seen.add((lemma, pos))
                hyps = tuple(
                    h.lemmas()[0].lower().replace(" ", "_")
                    for h in synset.hypernyms()[:3]
                    if h.lemmas()
                )
                records.append(
                    LexicalOntologyRecord(
                        lemma=lemma,
                        pos=pos,
                        language="en",
                        definition=(synset.definition() or "")[:240],
                        hypernyms=hyps,
                        synonyms=tuple(x.lower() for x in lemmas[1:5]),
                        domain="wordnet",
                    )
                )
            return self.commit_records(
                records,
                workspace,
                source="open_english_wordnet",
                lexicon=lexicon,
                grant=grant,
                ontology_store=ontology_store,
                workspace_limit=workspace_limit,
            )
        except ImportError:
            pass

        try:
            from nltk.corpus import wordnet as nltk_wn  # type: ignore[import-not-found]

            for synset in nltk_wn.all_synsets():
                if len(records) >= limit:
                    break
                lemmas = synset.lemma_names()
                if not lemmas:
                    continue
                lemma = lemmas[0].lower()
                pos = str(synset.pos())
                if (lemma, pos) in seen:
                    continue
                seen.add((lemma, pos))
                hyps = tuple(
                    h.lemma_names()[0].lower() for h in synset.hypernyms()[:3] if h.lemma_names()
                )
                records.append(
                    LexicalOntologyRecord(
                        lemma=lemma,
                        pos=pos,
                        language="en",
                        definition=(synset.definition() or "")[:240],
                        hypernyms=hyps,
                        synonyms=tuple(x.lower() for x in lemmas[1:5]),
                        domain="wordnet",
                    )
                )
            return self.commit_records(
                records,
                workspace,
                source="nltk_wordnet",
                lexicon=lexicon,
                grant=grant,
                ontology_store=ontology_store,
                workspace_limit=workspace_limit,
            )
        except (ImportError, LookupError) as exc:
            raise RuntimeError(
                "WordNet not installed. Run `pip install wn && python -m wn download oewn:2024` "
                "or use --source core / --source corpus / --source kaikki."
            ) from exc

    def ingest_corpus_directory(
        self,
        directory: Path | str,
        workspace: Any | None,
        *,
        limit: int = 5_000,
        min_freq: int = 3,
        max_files: int = 100,
        lexicon: EnglishLexicon | None = None,
        grant: Any | None = None,
        ontology_store: Any | None = None,
        workspace_limit: int | None = None,
    ) -> BootstrapStats:
        """Extract high-frequency open vocabulary from local text/markdown files."""
        root = Path(directory)
        counts: Counter[str] = Counter()
        files_scanned = 0

        for path in sorted(root.rglob("*")):
            if files_scanned >= max_files:
                break
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            files_scanned += 1
            for match in _WORD_RE.findall(text):
                w = match.lower()
                if w not in _STOPWORDS:
                    counts[w] += 1

        records = [
            LexicalOntologyRecord(
                lemma=word,
                pos=_infer_pos_from_morphology(word),
                language="en",
                definition=f"Corpus-grounded vocabulary item (freq={freq})",
                domain="corpus",
            )
            for word, freq in counts.most_common(limit)
            if freq >= min_freq
        ]
        stats = self.commit_records(
            records,
            workspace,
            source=f"corpus:{root.name}",
            lexicon=lexicon,
            grant=grant,
            ontology_store=ontology_store,
            workspace_limit=workspace_limit,
        )
        stats.extra["files_scanned"] = files_scanned
        return stats

    @staticmethod
    def hydrate_lexicon(
        graph_or_workspace: CognitiveGraph | Any,
        lexicon: EnglishLexicon,
        *,
        language: str = "en",
    ) -> int:
        """Load persisted ``lexical_ontology`` ConceptNodes into *lexicon*."""
        graph = getattr(graph_or_workspace, "graph", graph_or_workspace)
        if not isinstance(graph, CognitiveGraph):
            return 0

        added = 0
        for node in graph.all_nodes():
            if getattr(node, "node_type", None) != HCIRNodeType.CONCEPT:
                continue
            if getattr(node, "domain", "") != "lexical_ontology":
                continue
            props: dict[str, Any] = getattr(node, "properties", {}) or {}
            if props.get("language", "en") != language:
                continue
            lemma = str(props.get("lemma") or getattr(node, "label", "")).strip().lower()
            if not lemma:
                continue
            raw_pos = str(props.get("pos", "noun")).strip().lower()
            pos = _POS_MAP.get(raw_pos, EnglishPOS.NOUN)
            existing = lexicon.lookup(lemma)
            if any(entry.pos == pos for entry in existing):
                continue
            lexicon.add_entry(
                EnglishLexicalEntry(
                    lemma=lemma,
                    pos=pos,
                    semantic_predicate=lemma,
                    properties={
                        "thematic_roles": props.get("thematic_roles", []),
                        "hypernyms": props.get("hypernyms", []),
                    },
                )
            )
            added += 1
        return added


def hydrate_lexicon_from_iter(
    records: Iterable[LexicalOntologyRecord],
    lexicon: EnglishLexicon,
    *,
    language: str = "en",
) -> int:
    """Directly hydrate *lexicon* from an in-memory sequence of ontology records."""
    added = 0
    for rec in records:
        if rec.language != language:
            continue
        lemma = rec.lemma.strip().lower()
        pos = _POS_MAP.get(rec.pos.strip().lower(), EnglishPOS.NOUN)
        if any(e.pos == pos for e in lexicon.lookup(lemma)):
            continue
        lexicon.add_entry(
            EnglishLexicalEntry(
                lemma=lemma,
                pos=pos,
                semantic_predicate=lemma,
                properties={
                    "thematic_roles": list(rec.thematic_roles),
                    "hypernyms": list(rec.hypernyms),
                },
            )
        )
        added += 1
    return added
