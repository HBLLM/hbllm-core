"""Indexed SQLite Open-Vocabulary Lexical & Ontology Store for HCIR (Track T1).

Stores 100K+ grounded lemmas, POS tags, definitions, thematic verb frames,
synonyms, and hypernym links on disk with O(1) indexed lookup (< 0.1ms) and
zero working-graph RAM bloat, keeping ``TieredWorkspace`` within its active
node budget while giving ``hcir/language`` full open-vocabulary coverage.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ONTOLOGY_DB = Path("checkpoints/t1_ontology/lexical_ontology.db")


@dataclass(frozen=True)
class StoredLexicalEntry:
    """A lexical/ontological entry fetched from ``LexicalOntologyStore``."""

    lemma: str
    pos: str
    language: str
    definition: str = ""
    hypernyms: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    thematic_roles: tuple[str, ...] = ()
    domain: str = "general"
    source: str = ""


class LexicalOntologyStore:
    """Fast, read-mostly SQLite store for 100K+ open-vocabulary lexical entries."""

    def __init__(self, db_path: Path | str = DEFAULT_ONTOLOGY_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lexical_entries (
                language TEXT NOT NULL,
                lemma TEXT NOT NULL,
                pos TEXT NOT NULL,
                definition TEXT NOT NULL DEFAULT '',
                hypernyms_json TEXT NOT NULL DEFAULT '[]',
                synonyms_json TEXT NOT NULL DEFAULT '[]',
                thematic_roles_json TEXT NOT NULL DEFAULT '[]',
                domain TEXT NOT NULL DEFAULT 'general',
                source TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (language, lemma, pos)
            );
            CREATE INDEX IF NOT EXISTS idx_lex_lang_lemma
                ON lexical_entries (language, lemma);
            CREATE TABLE IF NOT EXISTS hypernym_edges (
                language TEXT NOT NULL,
                child TEXT NOT NULL,
                parent TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (language, child, parent)
            );
            CREATE INDEX IF NOT EXISTS idx_hyp_lang_child
                ON hypernym_edges (language, child);
            """
        )
        self._conn.commit()

    def upsert_many(
        self,
        records: Iterable[Any],
        *,
        source: str = "",
    ) -> tuple[int, int]:
        """Bulk-upsert lexical records and hypernym edges in one SQLite transaction."""
        entry_rows: list[tuple[str, str, str, str, str, str, str, str, str]] = []
        edge_rows: list[tuple[str, str, str, str]] = []

        for rec in records:
            lemma = str(rec.lemma).strip().lower()
            if not lemma:
                continue
            lang = str(rec.language).strip().lower() or "en"
            pos = str(rec.pos).strip().lower() or "noun"
            hyps = [str(h).strip().lower() for h in rec.hypernyms if str(h).strip()]
            syns = [str(s).strip().lower() for s in rec.synonyms if str(s).strip()]
            roles = [str(r).strip() for r in rec.thematic_roles if str(r).strip()]
            entry_rows.append(
                (
                    lang,
                    lemma,
                    pos,
                    str(rec.definition or ""),
                    json.dumps(hyps, ensure_ascii=False),
                    json.dumps(syns, ensure_ascii=False),
                    json.dumps(roles, ensure_ascii=False),
                    str(rec.domain or "general"),
                    source,
                )
            )
            for parent in hyps:
                if parent != lemma:
                    edge_rows.append((lang, lemma, parent, source))

        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO lexical_entries (
                    language, lemma, pos, definition,
                    hypernyms_json, synonyms_json, thematic_roles_json, domain, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(language, lemma, pos) DO UPDATE SET
                    definition = CASE WHEN excluded.definition != '' THEN excluded.definition ELSE lexical_entries.definition END,
                    hypernyms_json = CASE WHEN excluded.hypernyms_json != '[]' THEN excluded.hypernyms_json ELSE lexical_entries.hypernyms_json END,
                    synonyms_json = CASE WHEN excluded.synonyms_json != '[]' THEN excluded.synonyms_json ELSE lexical_entries.synonyms_json END,
                    thematic_roles_json = CASE WHEN excluded.thematic_roles_json != '[]' THEN excluded.thematic_roles_json ELSE lexical_entries.thematic_roles_json END,
                    domain = excluded.domain,
                    source = excluded.source
                """,
                entry_rows,
            )
            if edge_rows:
                self._conn.executemany(
                    """
                    INSERT OR IGNORE INTO hypernym_edges (language, child, parent, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    edge_rows,
                )

        return len(entry_rows), len(edge_rows)

    def lookup(self, lemma: str, *, language: str = "en") -> list[StoredLexicalEntry]:
        """O(1) indexed lookup for *lemma* in *language*."""
        key = lemma.strip().lower()
        if not key:
            return []
        cur = self._conn.execute(
            """
            SELECT lemma, pos, language, definition,
                   hypernyms_json, synonyms_json, thematic_roles_json, domain, source
            FROM lexical_entries
            WHERE language = ? AND lemma = ?
            """,
            (language, key),
        )
        results: list[StoredLexicalEntry] = []
        for row in cur.fetchall():
            results.append(
                StoredLexicalEntry(
                    lemma=row[0],
                    pos=row[1],
                    language=row[2],
                    definition=row[3],
                    hypernyms=tuple(json.loads(row[4])),
                    synonyms=tuple(json.loads(row[5])),
                    thematic_roles=tuple(json.loads(row[6])),
                    domain=row[7],
                    source=row[8],
                )
            )
        return results

    def has_word(self, lemma: str, *, language: str = "en") -> bool:
        key = lemma.strip().lower()
        if not key:
            return False
        cur = self._conn.execute(
            "SELECT 1 FROM lexical_entries WHERE language = ? AND lemma = ? LIMIT 1",
            (language, key),
        )
        return cur.fetchone() is not None

    def get_hypernyms(self, lemma: str, *, language: str = "en") -> list[str]:
        key = lemma.strip().lower()
        cur = self._conn.execute(
            "SELECT parent FROM hypernym_edges WHERE language = ? AND child = ?",
            (language, key),
        )
        return [row[0] for row in cur.fetchall()]

    def is_a(
        self,
        child: str,
        ancestor: str,
        *,
        language: str = "en",
        max_depth: int = 8,
    ) -> bool:
        """Transitive taxonomic check: is *child* a subtype of *ancestor*?"""
        start = child.strip().lower()
        target = ancestor.strip().lower()
        if not start or not target:
            return False
        if start == target:
            return True

        visited: set[str] = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for parent in self.get_hypernyms(curr, language=language):
                if parent == target:
                    return True
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth + 1))
        return False

    def stats(self) -> dict[str, Any]:
        entries = self._conn.execute("SELECT COUNT(*) FROM lexical_entries").fetchone()[0]
        unique_lemmas = self._conn.execute(
            "SELECT COUNT(DISTINCT lemma) FROM lexical_entries"
        ).fetchone()[0]
        edges = self._conn.execute("SELECT COUNT(*) FROM hypernym_edges").fetchone()[0]
        by_lang = dict(
            self._conn.execute(
                "SELECT language, COUNT(*) FROM lexical_entries GROUP BY language"
            ).fetchall()
        )
        return {
            "entries": entries,
            "unique_lemmas": unique_lemmas,
            "hypernym_edges": edges,
            "by_language": by_lang,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        self._conn.close()
