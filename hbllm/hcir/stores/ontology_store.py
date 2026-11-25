"""Indexed SQLite Open-Vocabulary Lexical & Ontology Store for HCIR (Track T1).

Stores 100K+ grounded lemmas, POS tags, definitions, thematic verb frames,
synonyms, and hypernym links on disk with O(1) indexed lookup (< 0.1ms) and
zero working-graph RAM bloat, keeping ``TieredWorkspace`` within its active
node budget while giving ``hcir/language`` full open-vocabulary coverage.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ONTOLOGY_DB = (
    Path(__file__).resolve().parents[3] / "checkpoints" / "t1_ontology" / "lexical_ontology.db"
)
_CORPUS_FREQ_RE = re.compile(r"\(freq=(\d+)\)")


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
    frequency: int = 0


class LexicalOntologyStore:
    """Fast, read-mostly SQLite store for 100K+ open-vocabulary lexical entries."""

    def __init__(self, db_path: Path | str = DEFAULT_ONTOLOGY_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA cache_size=-32768;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA mmap_size=268435456;")
        self._trusted_cache: dict[tuple[str, str], bool] = {}
        self._freq_cache: dict[tuple[str, str], int] = {}
        self._lookup_cache: dict[tuple[str, str], list[StoredLexicalEntry]] = {}
        self._hypernyms_cache: dict[tuple[str, str], list[str]] = {}
        self._is_a_cache: dict[tuple[str, str, str], bool] = {}
        self._relations_cache: dict[tuple[str, str, str], list[tuple[str, str, float]]] = {}
        self._antonym_cache: dict[tuple[str, str, str], bool] = {}
        self._align_cache: dict[tuple[str, str], str | None] = {}
        self._surface_cache: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._translit_cache: dict[tuple[str, str], list[str]] = {}
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
                frequency INTEGER NOT NULL DEFAULT 0,
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
            CREATE TABLE IF NOT EXISTS semantic_relations (
                language TEXT NOT NULL,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (language, subject, relation, object)
            );
            CREATE INDEX IF NOT EXISTS idx_semrel_lang_subj_rel
                ON semantic_relations (language, subject, relation);
            CREATE INDEX IF NOT EXISTS idx_semrel_lang_obj_rel
                ON semantic_relations (language, object, relation);
            CREATE TABLE IF NOT EXISTS concept_alignments (
                language TEXT NOT NULL,
                lemma TEXT NOT NULL,
                en_concept TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (language, lemma, en_concept)
            );
            CREATE INDEX IF NOT EXISTS idx_align_lang_lemma
                ON concept_alignments (language, lemma);
            CREATE INDEX IF NOT EXISTS idx_align_lang_en
                ON concept_alignments (language, en_concept);
            CREATE TABLE IF NOT EXISTS surface_forms (
                language TEXT NOT NULL,
                surface TEXT NOT NULL,
                lemma TEXT NOT NULL,
                pos TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (language, surface, lemma, pos)
            );
            CREATE INDEX IF NOT EXISTS idx_surface_lang_form
                ON surface_forms (language, surface);
            CREATE TABLE IF NOT EXISTS transliterations (
                language TEXT NOT NULL,
                romanized TEXT NOT NULL,
                native TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (language, romanized, native)
            );
            CREATE INDEX IF NOT EXISTS idx_translit_lang_rom
                ON transliterations (language, romanized);
            """
        )
        cols = {
            row[1] for row in self._conn.execute("PRAGMA table_info(lexical_entries);").fetchall()
        }
        if "frequency" not in cols:
            self._conn.execute(
                "ALTER TABLE lexical_entries ADD COLUMN frequency INTEGER NOT NULL DEFAULT 0;"
            )
        self._conn.commit()

        cur = self._conn.execute("SELECT count(*) FROM lexical_entries;")
        row = cur.fetchone()
        if row and row[0] == 0:
            from hbllm.hcir.language.ontology_bootstrap import CORE_ONTOLOGY_RECORDS

            self.upsert_many(CORE_ONTOLOGY_RECORDS, source="core_bootstrap")

    def upsert_many(
        self,
        records: Iterable[Any],
        *,
        source: str = "",
        preserve_existing: bool = False,
    ) -> tuple[int, int]:
        """Bulk-upsert lexical records and hypernym edges in one SQLite transaction."""
        entry_rows: list[tuple[str, str, str, str, str, str, str, str, str, int]] = []
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
            defn = str(rec.definition or "")
            freq = int(getattr(rec, "frequency", 0) or 0)
            if freq == 0 and defn:
                m_freq = _CORPUS_FREQ_RE.search(defn)
                if m_freq:
                    freq = int(m_freq.group(1))
            entry_rows.append(
                (
                    lang,
                    lemma,
                    pos,
                    defn,
                    json.dumps(hyps, ensure_ascii=False),
                    json.dumps(syns, ensure_ascii=False),
                    json.dumps(roles, ensure_ascii=False),
                    str(rec.domain or "general"),
                    source,
                    freq,
                )
            )
            for parent in hyps:
                if parent != lemma:
                    edge_rows.append((lang, lemma, parent, source))

        on_conflict_sql = (
            """
            ON CONFLICT(language, lemma, pos) DO UPDATE SET
                definition = CASE WHEN lexical_entries.definition != '' THEN lexical_entries.definition ELSE excluded.definition END,
                hypernyms_json = CASE WHEN lexical_entries.hypernyms_json != '[]' THEN lexical_entries.hypernyms_json ELSE excluded.hypernyms_json END,
                synonyms_json = CASE WHEN lexical_entries.synonyms_json != '[]' THEN lexical_entries.synonyms_json ELSE excluded.synonyms_json END,
                thematic_roles_json = CASE WHEN lexical_entries.thematic_roles_json != '[]' THEN lexical_entries.thematic_roles_json ELSE excluded.thematic_roles_json END,
                domain = CASE WHEN lexical_entries.source IN ('core_ontology', 'core_bootstrap', 't3_llm_scaffold', 't4_reading_curriculum', 'wordnet') THEN lexical_entries.domain ELSE excluded.domain END,
                source = CASE WHEN lexical_entries.source IN ('core_ontology', 'core_bootstrap', 't3_llm_scaffold', 't4_reading_curriculum', 'wordnet') OR lexical_entries.definition != '' THEN lexical_entries.source ELSE excluded.source END,
                frequency = MAX(lexical_entries.frequency, excluded.frequency)
            """
            if preserve_existing
            else """
            ON CONFLICT(language, lemma, pos) DO UPDATE SET
                definition = CASE WHEN excluded.definition != '' THEN excluded.definition ELSE lexical_entries.definition END,
                hypernyms_json = CASE WHEN excluded.hypernyms_json != '[]' THEN excluded.hypernyms_json ELSE lexical_entries.hypernyms_json END,
                synonyms_json = CASE WHEN excluded.synonyms_json != '[]' THEN excluded.synonyms_json ELSE lexical_entries.synonyms_json END,
                thematic_roles_json = CASE WHEN excluded.thematic_roles_json != '[]' THEN excluded.thematic_roles_json ELSE lexical_entries.thematic_roles_json END,
                domain = excluded.domain,
                source = excluded.source,
                frequency = MAX(lexical_entries.frequency, excluded.frequency)
            """
        )

        with self._conn:
            self._conn.executemany(
                f"""
                INSERT INTO lexical_entries (
                    language, lemma, pos, definition,
                    hypernyms_json, synonyms_json, thematic_roles_json, domain, source, frequency
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                {on_conflict_sql}
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
        self._trusted_cache.clear()
        self._freq_cache.clear()
        self._lookup_cache.clear()
        self._hypernyms_cache.clear()
        self._is_a_cache.clear()
        self._relations_cache.clear()
        self._antonym_cache.clear()
        self._align_cache.clear()
        self._surface_cache.clear()
        self._translit_cache.clear()
        return len(entry_rows), len(edge_rows)

    def lookup(self, lemma: str, *, language: str = "en") -> list[StoredLexicalEntry]:
        """O(1) indexed lookup for *lemma* in *language*."""
        key = lemma.strip().lower()
        if not key:
            return []
        cache_key = (language, key)
        if cache_key in self._lookup_cache:
            return self._lookup_cache[cache_key]
        cur = self._conn.execute(
            """
            SELECT lemma, pos, language, definition,
                   hypernyms_json, synonyms_json, thematic_roles_json, domain, source, frequency
            FROM lexical_entries
            WHERE language = ? AND lemma = ?
            ORDER BY
                CASE
                    WHEN source = 'core_ontology' THEN 0
                    WHEN source IN ('open_english_wordnet', 't4_reading_curriculum', 't3_llm_scaffold') THEN 1
                    WHEN definition LIKE 'Corpus-grounded%' OR source = 'corpus:gutenberg_cache' THEN 3
                    ELSE 2
                END,
                CASE pos
                    WHEN 'noun' THEN 0
                    WHEN 'n' THEN 0
                    WHEN 'verb' THEN 1
                    WHEN 'v' THEN 1
                    WHEN 'adj' THEN 2
                    WHEN 'a' THEN 2
                    WHEN 's' THEN 2
                    ELSE 3
                END
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
                    frequency=int(row[9] or 0),
                )
            )
        if len(self._lookup_cache) < 32768:
            self._lookup_cache[cache_key] = results
        return results

    def has_word(self, lemma: str, *, language: str = "en") -> bool:
        key = lemma.strip().lower()
        if not key:
            return False
        cache_key = (language, key)
        if cache_key in self._lookup_cache:
            return bool(self._lookup_cache[cache_key])
        cur = self._conn.execute(
            "SELECT 1 FROM lexical_entries WHERE language = ? AND lemma = ? LIMIT 1",
            (language, key),
        )
        return cur.fetchone() is not None

    def get_word_frequency(self, lemma: str, *, language: str = "en") -> int:
        """Return max recorded corpus frequency for *lemma* (0 if unknown)."""
        key = lemma.strip().lower()
        if not key:
            return 0
        cache_key = (language, key)
        if cache_key in self._freq_cache:
            return self._freq_cache[cache_key]
        row = self._conn.execute(
            "SELECT COALESCE(MAX(frequency), 0), definition FROM lexical_entries "
            "WHERE language = ? AND lemma = ?",
            (language, key),
        ).fetchone()
        freq = int(row[0] or 0) if row else 0
        if freq == 0 and row and row[1]:
            m = _CORPUS_FREQ_RE.search(str(row[1]))
            if m:
                freq = int(m.group(1))
        if len(self._freq_cache) < 16384:
            self._freq_cache[cache_key] = freq
        return freq

    def record_frequencies(self, counts: Mapping[str, int], *, language: str = "en") -> None:
        """Persist or increment word frequencies for existing entries in SQLite."""
        rows = [
            (int(freq), language, str(word).strip().lower())
            for word, freq in counts.items()
            if word and int(freq) > 0
        ]
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "UPDATE lexical_entries SET frequency = MAX(frequency, ?) "
                "WHERE language = ? AND lemma = ?",
                rows,
            )
        self._freq_cache.clear()
        self._trusted_cache.clear()

    def is_trusted_word(
        self,
        lemma: str,
        *,
        language: str = "en",
        min_corpus_freq: int = 5,
    ) -> bool:
        """Return True if *lemma* is curated/WordNet/curriculum OR meets corpus frequency threshold."""
        key = lemma.strip().lower()
        if not key:
            return False
        cache_key = (language, key)
        if cache_key in self._trusted_cache:
            return self._trusted_cache[cache_key]
        cur = self._conn.execute(
            "SELECT 1 FROM lexical_entries WHERE language = ? AND lemma = ? "
            "AND (source NOT IN ('corpus:gutenberg_cache', 'hcir_language_gateway', 'discourse_expository') OR frequency >= ?) "
            "LIMIT 1",
            (language, key, int(min_corpus_freq)),
        )
        trusted = cur.fetchone() is not None
        if len(self._trusted_cache) < 8192:
            self._trusted_cache[cache_key] = trusted
        return trusted

    def fuzzy_lookup(
        self,
        word: str,
        *,
        language: str = "en",
        limit: int = 5,
        include_replacements: bool = False,
    ) -> list[str]:
        """Fast O(edits) indexed 1-edit fuzzy lookup against trusted lexical entries.

        Uses B-tree point lookups (``lemma IN (...)``) rather than range scans.
        Defaults to high-precision Phase A edits (transpositions, deletions, missing
        vowels / trailing endings) for sub-5ms lookup and minimal false positives.
        """
        key = word.strip().lower()
        if len(key) < 4 or not key.isalpha():
            return []

        def _query_candidates(cands: list[str]) -> list[tuple[str, str]]:
            hits: list[tuple[str, str]] = []
            for start in range(0, len(cands), 850):
                batch = cands[start : start + 850]
                placeholders = ",".join("?" for _ in batch)
                rows = self._conn.execute(
                    f"SELECT DISTINCT lemma, source FROM lexical_entries "  # nosec B608
                    f"WHERE language = ? AND lemma IN ({placeholders}) "
                    f"AND source != 'corpus:gutenberg_cache'",
                    (language, *batch),
                ).fetchall()
                hits.extend((str(r[0]), str(r[1])) for r in rows)
            return hits

        splits = [(key[:i], key[i:]) for i in range(len(key) + 1)]
        transposes = [
            left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1
        ]
        deletes = [
            left + right[1:] for left, right in splits if right and len(left + right[1:]) >= 3
        ]
        fast_inserts = [key + c for c in "esydnl"] + [
            left + c + right for left, right in splits for c in "aeiou"
        ]
        phase_a = sorted(
            {c for c in (*transposes, *deletes, *fast_inserts) if c != key and len(c) >= 3}
        )
        raw_hits = _query_candidates(phase_a) if phase_a else []

        if not raw_hits and include_replacements:
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            replaces = [left + c + right[1:] for left, right in splits if right for c in alphabet]
            seen_a = set(phase_a)
            phase_b = sorted({c for c in replaces if c != key and len(c) >= 3 and c not in seen_a})
            if phase_b:
                raw_hits = _query_candidates(phase_b)

        if not raw_hits:
            return []

        key_sorted = sorted(key)
        trusted_sources = {"core_ontology", "t3_llm_scaffold", "t4_reading_curriculum"}
        # Deduplicate keeping best source per lemma
        best_source: dict[str, str] = {}
        for lemma, src in raw_hits:
            if lemma not in best_source or src in trusted_sources:
                best_source[lemma] = src

        ranked = sorted(
            best_source.items(),
            key=lambda item: (
                0 if sorted(item[0]) == key_sorted else 1,
                0 if item[0][0] == key[0] else 1,
                0 if item[1] in trusted_sources else 1,
                abs(len(item[0]) - len(key)),
                item[0],
            ),
        )
        return [lemma for lemma, _ in ranked[:limit]]

    def get_hypernyms(self, lemma: str, *, language: str = "en") -> list[str]:
        key = lemma.strip().lower()
        if not key:
            return []
        cache_key = (language, key)
        if cache_key in self._hypernyms_cache:
            return self._hypernyms_cache[cache_key]
        cur = self._conn.execute(
            "SELECT parent FROM hypernym_edges WHERE language = ? AND child = ?",
            (language, key),
        )
        res = [row[0] for row in cur.fetchall()]
        if len(self._hypernyms_cache) < 32768:
            self._hypernyms_cache[cache_key] = res
        return res

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
        cache_key = (language, start, target)
        if cache_key in self._is_a_cache:
            return self._is_a_cache[cache_key]

        visited: set[str] = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for parent in self.get_hypernyms(curr, language=language):
                if parent == target:
                    if len(self._is_a_cache) < 32768:
                        self._is_a_cache[cache_key] = True
                    return True
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth + 1))
        if len(self._is_a_cache) < 32768:
            self._is_a_cache[cache_key] = False
        return False

    def upsert_hypernyms(
        self,
        edges: Iterable[tuple[str, str, str, str]],
    ) -> int:
        """Bulk-insert `(language, child, parent, source)` hypernym edges."""
        rows = [
            (
                str(lang).strip().lower() or "en",
                str(child).strip().lower(),
                str(parent).strip().lower(),
                str(src or ""),
            )
            for lang, child, parent, src in edges
            if str(child).strip()
            and str(parent).strip()
            and str(child).strip().lower() != str(parent).strip().lower()
        ]
        if not rows:
            return 0
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO hypernym_edges (language, child, parent, source) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
        self._hypernyms_cache.clear()
        self._is_a_cache.clear()
        return len(rows)

    def upsert_semantic_relations(
        self,
        rows: Iterable[tuple[str, str, str, str, float, str]],
    ) -> int:
        """Bulk-upsert `(language, subject, relation, object, weight, source)` commonsense edges."""
        cleaned = [
            (
                str(lang).strip().lower() or "en",
                str(subj).strip().lower(),
                str(rel).strip(),
                str(obj).strip().lower(),
                float(wt),
                str(src or ""),
            )
            for lang, subj, rel, obj, wt, src in rows
            if str(subj).strip() and str(rel).strip() and str(obj).strip()
        ]
        if not cleaned:
            return 0
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO semantic_relations (language, subject, relation, object, weight, source)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(language, subject, relation, object) DO UPDATE SET
                    weight = MAX(semantic_relations.weight, excluded.weight),
                    source = CASE WHEN excluded.weight >= semantic_relations.weight
                             THEN excluded.source ELSE semantic_relations.source END
                """,
                cleaned,
            )
        self._relations_cache.clear()
        self._antonym_cache.clear()
        return len(cleaned)

    def get_relations(
        self,
        concept: str,
        *,
        relation: str | None = None,
        language: str = "en",
        limit: int = 12,
    ) -> list[tuple[str, str, float]]:
        """Return `(relation, object, weight)` commonsense triples for *concept*."""
        key = concept.strip().lower()
        if not key:
            return []
        rel_key = relation.strip() if relation else ""
        cache_key = (language, key, f"{rel_key}:{limit}")
        if cache_key in self._relations_cache:
            return self._relations_cache[cache_key]
        if rel_key:
            cur = self._conn.execute(
                """
                SELECT relation, object, weight
                FROM semantic_relations
                WHERE language = ? AND subject = ? AND relation = ?
                ORDER BY weight DESC, object ASC
                LIMIT ?
                """,
                (language, key, rel_key, int(limit)),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT relation, object, weight
                FROM semantic_relations
                WHERE language = ? AND subject = ?
                ORDER BY weight DESC, relation ASC, object ASC
                LIMIT ?
                """,
                (language, key, int(limit)),
            )
        res = [(str(r[0]), str(r[1]), float(r[2])) for r in cur.fetchall()]
        if len(self._relations_cache) < 32768:
            self._relations_cache[cache_key] = res
        return res

    def are_antonyms(
        self,
        term_a: str,
        term_b: str,
        *,
        language: str = "en",
    ) -> bool:
        """Return True if *term_a* and *term_b* are linked via Antonym/DistinctFrom."""
        a = term_a.strip().lower()
        b = term_b.strip().lower()
        if not a or not b or a == b:
            return False
        pair = (a, b) if a < b else (b, a)
        cache_key = (language, pair[0], pair[1])
        if cache_key in self._antonym_cache:
            return self._antonym_cache[cache_key]
        row = self._conn.execute(
            """
            SELECT 1 FROM semantic_relations
            WHERE language = ? AND relation IN ('Antonym', 'DistinctFrom')
              AND ((subject = ? AND object = ?) OR (subject = ? AND object = ?))
            LIMIT 1
            """,
            (language, a, b, b, a),
        ).fetchone()
        matched = row is not None
        if len(self._antonym_cache) < 32768:
            self._antonym_cache[cache_key] = matched
        return matched

    def upsert_concept_alignments(
        self,
        rows: Iterable[tuple[str, str, str, float, str]],
    ) -> int:
        """Bulk-upsert `(language, lemma, en_concept, confidence, source)` alignments."""
        cleaned = [
            (
                str(lang).strip().lower(),
                str(lemma).strip().lower(),
                str(en).strip().lower(),
                float(conf),
                str(src or ""),
            )
            for lang, lemma, en, conf, src in rows
            if str(lang).strip() and str(lemma).strip() and str(en).strip()
        ]
        if not cleaned:
            return 0
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO concept_alignments (language, lemma, en_concept, confidence, source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(language, lemma, en_concept) DO UPDATE SET
                    confidence = MAX(concept_alignments.confidence, excluded.confidence),
                    source = CASE WHEN excluded.confidence >= concept_alignments.confidence
                             THEN excluded.source ELSE concept_alignments.source END
                """,
                cleaned,
            )
        self._align_cache.clear()
        return len(cleaned)

    def align_to_english(self, lemma: str, *, language: str) -> str | None:
        """Return the highest-confidence English HCIR concept aligned to *lemma*."""
        key = lemma.strip().lower()
        if not key:
            return None
        cache_key = (language, key)
        if cache_key in self._align_cache:
            return self._align_cache[cache_key]
        row = self._conn.execute(
            """
            SELECT ca.en_concept
            FROM concept_alignments AS ca
            LEFT JOIN lexical_entries AS le
              ON le.language = 'en' AND le.lemma = ca.en_concept
            WHERE ca.language = ? AND ca.lemma = ?
            GROUP BY ca.en_concept, ca.confidence
            ORDER BY ca.confidence DESC, COALESCE(MAX(le.frequency), 0) DESC, LENGTH(ca.en_concept) ASC, ca.en_concept ASC
            LIMIT 1
            """,
            (language, key),
        ).fetchone()
        res = str(row[0]) if row else None
        if len(self._align_cache) < 32768:
            self._align_cache[cache_key] = res
        return res

    def align_from_english(
        self,
        en_concept: str,
        *,
        language: str,
        limit: int = 5,
    ) -> list[str]:
        """Return target-language lemmas aligned to *en_concept* ordered by confidence."""
        key = en_concept.strip().lower()
        if not key:
            return []
        rows = self._conn.execute(
            """
            SELECT lemma FROM concept_alignments
            WHERE language = ? AND en_concept = ?
            ORDER BY confidence DESC, lemma ASC
            LIMIT ?
            """,
            (language, key, int(limit)),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def upsert_surface_forms(
        self,
        rows: Iterable[tuple[str, str, str, str, str]],
    ) -> int:
        """Bulk-insert `(language, surface, lemma, pos, tags_json)` inflection mappings."""
        cleaned = [
            (
                str(lang).strip().lower() or "en",
                str(surface).strip().lower(),
                str(lemma).strip().lower(),
                str(pos or "").strip().lower(),
                str(tags or "[]"),
            )
            for lang, surface, lemma, pos, tags in rows
            if str(surface).strip() and str(lemma).strip()
        ]
        if not cleaned:
            return 0
        with self._conn:
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO surface_forms (language, surface, lemma, pos, tags_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                cleaned,
            )
        self._surface_cache.clear()
        return len(cleaned)

    def lemmatize_surface(self, surface: str, *, language: str = "en") -> list[tuple[str, str]]:
        """Return `(lemma, pos)` pairs for an inflected *surface* form."""
        key = surface.strip().lower()
        if not key:
            return []
        cache_key = (language, key)
        if cache_key in self._surface_cache:
            return self._surface_cache[cache_key]
        rows = self._conn.execute(
            "SELECT lemma, pos FROM surface_forms WHERE language = ? AND surface = ?",
            (language, key),
        ).fetchall()
        res = [(str(r[0]), str(r[1])) for r in rows]
        if len(self._surface_cache) < 32768:
            self._surface_cache[cache_key] = res
        return res

    def upsert_transliterations(
        self,
        rows: Iterable[tuple[str, str, str, int, str]],
    ) -> int:
        """Bulk-upsert `(language, romanized, native, count, source)` mappings."""
        cleaned = [
            (
                str(lang).strip().lower(),
                str(rom).strip().lower(),
                str(nat).strip(),
                int(cnt),
                str(src or ""),
            )
            for lang, rom, nat, cnt, src in rows
            if str(rom).strip() and str(nat).strip()
        ]
        if not cleaned:
            return 0
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO transliterations (language, romanized, native, count, source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(language, romanized, native) DO UPDATE SET
                    count = MAX(transliterations.count, excluded.count),
                    source = excluded.source
                """,
                cleaned,
            )
        self._translit_cache.clear()
        return len(cleaned)

    def transliterate_romanized(
        self,
        token: str,
        *,
        language: str,
        limit: int = 3,
    ) -> list[str]:
        """Return top native-script candidates for a romanized *token* ordered by attestation count."""
        key = token.strip().lower()
        if not key:
            return []
        cache_key = (language, key)
        if cache_key in self._translit_cache:
            return self._translit_cache[cache_key][:limit]
        rows = self._conn.execute(
            """
            SELECT native FROM transliterations
            WHERE language = ? AND romanized = ?
            ORDER BY count DESC, LENGTH(native) ASC, native ASC
            LIMIT 8
            """,
            (language, key),
        ).fetchall()
        res = [str(r[0]) for r in rows]
        if len(self._translit_cache) < 32768:
            self._translit_cache[cache_key] = res
        return res[:limit]

    def record_thematic_roles(
        self,
        role_map: Mapping[str, Iterable[str]],
        *,
        language: str = "en",
        source: str = "verb_frames",
    ) -> int:
        """Merge thematic role lists into verb entries in `lexical_entries`."""
        if not role_map:
            return 0
        rows_to_upsert: list[tuple[str, str, str, str, str, str, str, str, str, int]] = []
        norm_map = {
            str(verb).strip().lower(): sorted({str(r).strip() for r in roles if str(r).strip()})
            for verb, roles in role_map.items()
            if str(verb).strip()
        }
        verbs = list(norm_map.keys())
        existing_roles: dict[tuple[str, str], list[str]] = {}
        existing_pos: dict[str, str] = {}
        for start in range(0, len(verbs), 850):
            batch = verbs[start : start + 850]
            placeholders = ",".join("?" for _ in batch)
            cur = self._conn.execute(
                f"SELECT lemma, pos, thematic_roles_json FROM lexical_entries "  # nosec B608
                f"WHERE language = ? AND pos IN ('v', 'verb') AND lemma IN ({placeholders})",
                (language, *batch),
            )
            for lemma, pos, raw_roles in cur.fetchall():
                try:
                    parsed = json.loads(raw_roles) if raw_roles else []
                except json.JSONDecodeError:
                    parsed = []
                existing_roles[(str(lemma), str(pos))] = [str(r) for r in parsed]
                existing_pos.setdefault(str(lemma), str(pos))

        updated = 0
        with self._conn:
            for verb, new_roles in norm_map.items():
                if not new_roles:
                    continue
                pos = existing_pos.get(verb, "verb")
                prev = existing_roles.get((verb, pos), [])
                merged = sorted(set(prev) | set(new_roles))
                if merged == prev and (verb, pos) in existing_roles:
                    continue
                rows_to_upsert.append(
                    (
                        language,
                        verb,
                        pos,
                        "",
                        "[]",
                        "[]",
                        json.dumps(merged, ensure_ascii=False),
                        "general",
                        source,
                        0,
                    )
                )
            if rows_to_upsert:
                self._conn.executemany(
                    """
                    INSERT INTO lexical_entries (
                        language, lemma, pos, definition,
                        hypernyms_json, synonyms_json, thematic_roles_json, domain, source, frequency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(language, lemma, pos) DO UPDATE SET
                        thematic_roles_json = excluded.thematic_roles_json
                    """,
                    rows_to_upsert,
                )
                updated = len(rows_to_upsert)
        self._lookup_cache.clear()
        return updated

    def stats(self) -> dict[str, Any]:
        entries = self._conn.execute("SELECT COUNT(*) FROM lexical_entries").fetchone()[0]
        unique_lemmas = self._conn.execute(
            "SELECT COUNT(DISTINCT lemma) FROM lexical_entries"
        ).fetchone()[0]
        edges = self._conn.execute("SELECT COUNT(*) FROM hypernym_edges").fetchone()[0]
        semantic_relations = self._conn.execute(
            "SELECT COUNT(*) FROM semantic_relations"
        ).fetchone()[0]
        concept_alignments = self._conn.execute(
            "SELECT COUNT(*) FROM concept_alignments"
        ).fetchone()[0]
        surface_forms = self._conn.execute("SELECT COUNT(*) FROM surface_forms").fetchone()[0]
        transliterations = self._conn.execute("SELECT COUNT(*) FROM transliterations").fetchone()[0]
        by_lang = dict(
            self._conn.execute(
                "SELECT language, COUNT(*) FROM lexical_entries GROUP BY language"
            ).fetchall()
        )
        return {
            "entries": entries,
            "unique_lemmas": unique_lemmas,
            "hypernym_edges": edges,
            "semantic_relations": semantic_relations,
            "concept_alignments": concept_alignments,
            "surface_forms": surface_forms,
            "transliterations": transliterations,
            "by_language": by_lang,
            "db_path": str(self.db_path),
        }

    def iter_external_sqlite_rows(
        self,
        db_path: Path | str,
        *,
        table: str,
        columns: tuple[str, ...],
    ) -> list[tuple[Any, ...]]:
        """Read rows from an external SQLite file via ``ATTACH DATABASE`` within this store."""
        ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        if not ident_re.fullmatch(table):
            raise ValueError(f"Invalid SQLite table name: {table!r}")
        if not columns or not all(ident_re.fullmatch(c) for c in columns):
            raise ValueError(f"Invalid SQLite column list: {columns!r}")

        if self._conn.in_transaction:
            self._conn.commit()
        col_sql = ", ".join(columns)
        self._conn.execute("ATTACH DATABASE ? AS ext_dict", (str(db_path),))
        try:
            return list(
                self._conn.execute(f"SELECT {col_sql} FROM ext_dict.{table}").fetchall()  # nosec B608
            )
        finally:
            if self._conn.in_transaction:
                self._conn.commit()
            self._conn.execute("DETACH DATABASE ext_dict")

    def optimize(self, *, vacuum: bool = False) -> None:
        """Checkpoint WAL, refresh SQLite query-planner statistics, and optionally compact."""
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        self._conn.execute("PRAGMA optimize;")
        self._conn.execute("ANALYZE;")
        if vacuum:
            self._conn.execute("VACUUM;")
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    def close(self) -> None:
        self._conn.close()
