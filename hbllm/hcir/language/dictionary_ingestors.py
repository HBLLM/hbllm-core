"""Track T1.5: Streaming Multilingual Dictionary, Frame & Commonsense Ingestors.

Deterministically ingests curated local linguistic datasets (frequency lists,
Kaikki/Wiktionary dumps, VerbNet, PropBank, ConceptNet, bilingual Sinhala/Tamil
dictionaries, Dakshina transliterations, and Webster/Moby enrichment) directly into
``LexicalOntologyStore`` with zero working-graph RAM bloat and zero LLM deps.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET  # nosec B405
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hcir.language.ontology_bootstrap import LexicalOntologyRecord

logger = logging.getLogger(__name__)

_TOKEN_CLEAN_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9 _-]{0,31}$")
_POS_PREFIX_RE = re.compile(
    r"^(?:(?:n|v|a|adj|adv|prep|pron|conj|int|interj|part)\s*\.\s*)+",
    re.IGNORECASE,
)
_IGNORED_FORM_TAGS: frozenset[str] = frozenset(
    {
        "romanization",
        "table-tags",
        "inflection-template",
        "class",
        "obsolete",
        "archaic",
        "misspelling",
        "error-unrecognized-form",
    }
)
_PROPBANK_ROLE_MAP: dict[str, str] = {
    "PAG": "Agent",
    "PPT": "Patient",
    "GOL": "Goal",
    "INS": "Instrument",
    "LOC": "Location",
    "SRC": "Source",
    "DIR": "Direction",
    "MNR": "Manner",
    "CAU": "Cause",
    "PRD": "Attribute",
    "COM": "Comitative",
    "REC": "Recipient",
    "EXT": "Extent",
    "TMP": "Time",
}
_CONCEPTNET_SEMANTIC_RELATIONS: frozenset[str] = frozenset(
    {
        "Causes",
        "HasPrerequisite",
        "MotivatedByGoal",
        "UsedFor",
        "CapableOf",
        "HasProperty",
        "PartOf",
        "AtLocation",
        "Antonym",
        "DistinctFrom",
        "MadeOf",
    }
)
_CONCEPTNET_URI_RE = re.compile(r"^/c/([a-z]{2,3})/([^/]+)")


@dataclass
class IngestionReport:
    """Structured summary emitted by each Track T1.5 ingestor."""

    source: str
    records_read: int = 0
    entries_written: int = 0
    relations_written: int = 0
    alignments_written: int = 0
    forms_written: int = 0
    transliterations_written: int = 0
    elapsed_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _normalize_gloss_candidates(raw: str, *, max_words: int = 2) -> list[str]:
    """Extract clean short English concept candidates from a bilingual gloss string."""
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", raw)
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[;,/|]", cleaned):
        cand = " ".join(part.strip().lower().split())
        if not cand or len(cand.split()) > max_words:
            continue
        if _TOKEN_CLEAN_RE.match(cand) and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _flush_kaikki_batch(
    store: Any,
    records: list[LexicalOntologyRecord],
    forms: list[tuple[str, str, str, str, str]],
    alignments: list[tuple[str, str, str, float, str]],
    source: str,
) -> tuple[int, int, int]:
    """Flush batched Kaikki records, forms, and alignments into SQLite."""
    written = 0
    forms_written = 0
    align_written = 0
    if records:
        written, _ = store.upsert_many(records, source=source, preserve_existing=True)
        records.clear()
    if forms:
        forms_written = store.upsert_surface_forms(forms)
        forms.clear()
    if alignments:
        align_written = store.upsert_concept_alignments(alignments)
        alignments.clear()
    return written, forms_written, align_written


def ingest_frequency_list(
    path: Path | str,
    store: Any,
    *,
    language: str = "en",
    max_entries: int = 250_000,
) -> tuple[dict[str, int], IngestionReport]:
    """Ingest word frequency counts from ``<word> <count>`` text files."""
    t0 = time.perf_counter()
    file_path = Path(path)
    counts: dict[str, int] = {}
    records_read = 0
    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if len(counts) >= max_entries:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0].strip().lower()
            try:
                freq = int(parts[-1])
            except ValueError:
                continue
            records_read += 1
            if not word or freq <= 0 or " " in word:
                continue
            if word not in counts or freq > counts[word]:
                counts[word] = freq

    store.record_frequencies(counts, language=language)
    return counts, IngestionReport(
        source=f"frequency:{file_path.name}",
        records_read=records_read,
        entries_written=len(counts),
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        extra={"language": language},
    )


def _extract_kaikki_item(
    item: dict[str, Any],
    *,
    language: str,
    freq_map: Mapping[str, int] | None,
    include_forms: bool,
    source_name: str,
) -> tuple[
    LexicalOntologyRecord | None,
    list[tuple[str, str, str, str, str]],
    list[tuple[str, str, str, float, str]],
]:
    """Normalize a single Kaikki JSON object into ontology rows."""
    word = str(item.get("word", "")).strip().lower()
    raw_pos = str(item.get("pos", "noun")).strip().lower() or "noun"
    if not word or " " in word or len(word) > 48:
        return None, [], []

    senses = item.get("senses") or []
    dict_senses = [s for s in senses if isinstance(s, dict)] if isinstance(senses, list) else []
    if dict_senses:
        all_misspelling = all(
            any(str(t).strip().lower() in ("misspelling", "typo") for t in (s.get("tags") or []))
            for s in dict_senses
        )
        if all_misspelling:
            return None, [], []

        all_form_of = all(
            any(str(t).strip().lower() == "form-of" for t in (s.get("tags") or []))
            or bool(s.get("form_of"))
            for s in dict_senses
        )
        if all_form_of:
            forms_only: list[tuple[str, str, str, str, str]] = []
            if include_forms:
                for s in dict_senses:
                    s_tags = [
                        str(t).strip().lower() for t in (s.get("tags") or []) if str(t).strip()
                    ]
                    tag_json = json.dumps(s_tags[:4], ensure_ascii=False)
                    for fo in s.get("form_of") or []:
                        if not isinstance(fo, dict):
                            continue
                        base_lemma = str(fo.get("word", "")).strip().lower()
                        if (
                            not base_lemma
                            or base_lemma == word
                            or " " in base_lemma
                            or len(base_lemma) > 48
                        ):
                            continue
                        forms_only.append((language, word, base_lemma, raw_pos, tag_json))
            return None, forms_only, []

    gloss = ""
    hypernyms: list[str] = []
    synonyms: list[str] = []
    alignments: list[tuple[str, str, str, float, str]] = []
    if dict_senses:
        first = dict_senses[0]
        glosses = first.get("glosses") or []
        if glosses:
            gloss = str(glosses[0]).strip()[:280]
        for hyp in first.get("hypernyms") or []:
            if isinstance(hyp, dict) and hyp.get("word"):
                hypernyms.append(str(hyp["word"]).strip().lower())
        for syn in first.get("synonyms") or []:
            if isinstance(syn, dict) and syn.get("word"):
                synonyms.append(str(syn["word"]).strip().lower())
        if language in ("si", "ta"):
            for g in glosses[:2]:
                for cand in _normalize_gloss_candidates(str(g), max_words=2)[:3]:
                    alignments.append((language, word, cand, 0.78, source_name))

    for hyp in item.get("hypernyms") or []:
        if isinstance(hyp, dict) and hyp.get("word") and len(hypernyms) < 5:
            hypernyms.append(str(hyp["word"]).strip().lower())
    for syn in item.get("synonyms") or []:
        if isinstance(syn, dict) and syn.get("word") and len(synonyms) < 8:
            synonyms.append(str(syn["word"]).strip().lower())

    freq = int(freq_map.get(word, 0)) if freq_map else 0
    rec = LexicalOntologyRecord(
        lemma=word,
        pos=raw_pos,
        language=language,
        definition=gloss,
        hypernyms=tuple(list(dict.fromkeys(hypernyms))[:4]),
        synonyms=tuple(list(dict.fromkeys(synonyms))[:6]),
        domain="wiktionary",
    )
    # Attach frequency via wrapper tuple cleanly using getattr compatibility
    if freq > 0:
        rec = LexicalOntologyRecordWithFreq.from_record(rec, frequency=freq)

    forms_out: list[tuple[str, str, str, str, str]] = []
    if include_forms:
        for f_obj in item.get("forms") or []:
            if not isinstance(f_obj, dict):
                continue
            form = str(f_obj.get("form", "")).strip().lower()
            tags = [str(t).strip().lower() for t in (f_obj.get("tags") or []) if str(t).strip()]
            if not form or form == word or " " in form or len(form) > 48:
                continue
            if any(t in _IGNORED_FORM_TAGS for t in tags):
                continue
            forms_out.append(
                (language, form, word, raw_pos, json.dumps(tags[:4], ensure_ascii=False))
            )

    return rec, forms_out, alignments


@dataclass(frozen=True)
class LexicalOntologyRecordWithFreq(LexicalOntologyRecord):
    """LexicalOntologyRecord carrying corpus frequency for ``store.upsert_many``."""

    frequency: int = 0

    @classmethod
    def from_record(
        cls, rec: LexicalOntologyRecord, *, frequency: int
    ) -> LexicalOntologyRecordWithFreq:
        return cls(
            lemma=rec.lemma,
            pos=rec.pos,
            language=rec.language,
            definition=rec.definition,
            hypernyms=rec.hypernyms,
            synonyms=rec.synonyms,
            thematic_roles=rec.thematic_roles,
            domain=rec.domain,
            frequency=frequency,
        )


def ingest_kaikki_multilingual(
    path: Path | str,
    store: Any,
    *,
    language: str = "en",
    limit: int = 150_000,
    freq_map: Mapping[str, int] | None = None,
    include_forms: bool = True,
    batch_size: int = 20_000,
) -> IngestionReport:
    """Stream Kaikki/Wiktionary JSONL line-by-line with optional frequency filtering."""
    t0 = time.perf_counter()
    file_path = Path(path)
    source_name = f"kaikki:{file_path.name}"

    target_words: set[str] | None = None
    if language == "en" and freq_map:
        ranked = sorted(freq_map.items(), key=lambda kv: kv[1], reverse=True)
        target_words = {w for w, _ in ranked[: max(limit, 50_000)] if w.isalpha()}

    records_batch: list[LexicalOntologyRecord] = []
    forms_batch: list[tuple[str, str, str, str, str]] = []
    align_batch: list[tuple[str, str, str, float, str]] = []
    seen: set[tuple[str, str]] = set()
    read_count = 0
    entries_written = 0
    forms_written = 0
    align_written = 0

    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if entries_written + len(records_batch) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            read_count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            word = str(item.get("word", "")).strip().lower()
            raw_pos = str(item.get("pos", "noun")).strip().lower() or "noun"
            if not word or " " in word or (word, raw_pos) in seen:
                continue
            if target_words is not None and word not in target_words:
                senses = item.get("senses") or []
                has_rich_links = isinstance(senses, list) and bool(
                    senses and isinstance(senses[0], dict) and senses[0].get("hypernyms")
                )
                if not has_rich_links:
                    continue
            seen.add((word, raw_pos))
            rec, forms, aligns = _extract_kaikki_item(
                item,
                language=language,
                freq_map=freq_map,
                include_forms=include_forms,
                source_name=source_name,
            )
            if forms:
                forms_batch.extend(forms)
            if aligns:
                align_batch.extend(aligns)
            if rec is not None:
                records_batch.append(rec)

            if len(records_batch) >= batch_size or len(forms_batch) >= batch_size * 2:
                w, fw, aw = _flush_kaikki_batch(
                    store, records_batch, forms_batch, align_batch, source_name
                )
                entries_written += w
                forms_written += fw
                align_written += aw

    w, fw, aw = _flush_kaikki_batch(store, records_batch, forms_batch, align_batch, source_name)
    entries_written += w
    forms_written += fw
    align_written += aw

    return IngestionReport(
        source=source_name,
        records_read=read_count,
        entries_written=entries_written,
        alignments_written=align_written,
        forms_written=forms_written,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        extra={"language": language, "frequency_guided": target_words is not None},
    )


def ingest_verbnet_zip(path: Path | str, store: Any) -> IngestionReport:
    """Ingest VerbNet XML classes from ``verbnet-master.zip`` into thematic verb frames."""
    t0 = time.perf_counter()
    zip_path = Path(path)
    role_map: dict[str, set[str]] = {}
    files_read = 0

    with zipfile.ZipFile(zip_path) as zf:
        v34 = [n for n in zf.namelist() if "/verbnet3.4/" in n and n.endswith(".xml")]
        xml_names = v34 or [n for n in zf.namelist() if n.endswith(".xml")]
        for name in xml_names:
            files_read += 1
            try:
                root = ET.fromstring(zf.read(name))  # nosec B314
            except ET.ParseError:
                continue
            roles = [
                str(el.attrib.get("type", "")).strip()
                for el in root.findall(".//THEMROLE")
                if el.attrib.get("type")
            ]
            if not roles:
                continue
            for mem in root.findall(".//MEMBER"):
                lemma = str(mem.attrib.get("name", "")).strip().lower().replace("_", " ")
                if lemma and " " not in lemma:
                    role_map.setdefault(lemma, set()).update(roles)

    updated = store.record_thematic_roles(
        role_map, language="en", source=f"verbnet:{zip_path.name}"
    )
    return IngestionReport(
        source=f"verbnet:{zip_path.name}",
        records_read=files_read,
        entries_written=updated,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        extra={"verbs_with_roles": len(role_map)},
    )


def ingest_propbank_zip(path: Path | str, store: Any) -> IngestionReport:
    """Ingest PropBank predicate framesets from ``propbank-frames-main.zip``."""
    t0 = time.perf_counter()
    zip_path = Path(path)
    role_map: dict[str, set[str]] = {}
    files_read = 0

    with zipfile.ZipFile(zip_path) as zf:
        xml_names = [n for n in zf.namelist() if "/frames/" in n and n.endswith(".xml")]
        for name in xml_names:
            files_read += 1
            try:
                root = ET.fromstring(zf.read(name))  # nosec B314
            except ET.ParseError:
                continue
            for pred in root.findall(".//predicate"):
                lemma = str(pred.attrib.get("lemma", "")).strip().lower()
                if not lemma or "_" in lemma or " " in lemma:
                    continue
                roles: set[str] = set()
                for role in pred.findall(".//role"):
                    func = str(role.attrib.get("f", "")).strip().upper()
                    mapped = _PROPBANK_ROLE_MAP.get(func)
                    if mapped:
                        roles.add(mapped)
                if roles:
                    role_map.setdefault(lemma, set()).update(roles)

    updated = store.record_thematic_roles(
        role_map, language="en", source=f"propbank:{zip_path.name}"
    )
    return IngestionReport(
        source=f"propbank:{zip_path.name}",
        records_read=files_read,
        entries_written=updated,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        extra={"verbs_with_roles": len(role_map)},
    )


def _parse_conceptnet_term(uri: str, allowed_langs: frozenset[str]) -> tuple[str, str] | None:
    m = _CONCEPTNET_URI_RE.match(uri.strip())
    if not m:
        return None
    lang, raw_term = m.group(1), m.group(2)
    if lang not in allowed_langs:
        return None
    term = raw_term.replace("_", " ").strip().lower()
    if not term or len(term) > 32 or len(term.split()) > 2:
        return None
    return lang, term


def ingest_conceptnet_gz(
    path: Path | str,
    store: Any,
    *,
    languages: Sequence[str] = ("en", "si", "ta"),
    min_weight: float = 2.0,
    max_edges: int = 250_000,
    batch_size: int = 25_000,
) -> IngestionReport:
    """Stream ConceptNet 5.7 assertions into hypernyms, semantic_relations, and alignments."""
    t0 = time.perf_counter()
    gz_path = Path(path)
    allowed = frozenset(languages)
    src_name = f"conceptnet:{gz_path.name}"

    hyp_batch: list[tuple[str, str, str, str]] = []
    rel_batch: list[tuple[str, str, str, str, float, str]] = []
    align_batch: list[tuple[str, str, str, float, str]] = []

    read_count = 0
    hyp_written = 0
    rel_written = 0
    align_written = 0

    def _flush() -> None:
        nonlocal hyp_written, rel_written, align_written
        if hyp_batch:
            hyp_written += store.upsert_hypernyms(hyp_batch)
            hyp_batch.clear()
        if rel_batch:
            rel_written += store.upsert_semantic_relations(rel_batch)
            rel_batch.clear()
        if align_batch:
            align_written += store.upsert_concept_alignments(align_batch)
            align_batch.clear()

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if hyp_written + rel_written + len(hyp_batch) + len(rel_batch) >= max_edges:
                break
            read_count += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            rel = parts[1].removeprefix("/r/")
            if rel != "IsA" and rel != "Synonym" and rel not in _CONCEPTNET_SEMANTIC_RELATIONS:
                continue
            s_parsed = _parse_conceptnet_term(parts[2], allowed)
            if s_parsed is None:
                continue
            o_parsed = _parse_conceptnet_term(parts[3], allowed)
            if o_parsed is None:
                continue
            s_lang, s_term = s_parsed
            o_lang, o_term = o_parsed
            if s_term == o_term and s_lang == o_lang:
                continue
            try:
                meta = json.loads(parts[4])
                weight = float(meta.get("weight", 1.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                weight = 1.0
            if weight < min_weight:
                continue

            if s_lang == o_lang:
                if rel == "IsA":
                    hyp_batch.append((s_lang, s_term, o_term, src_name))
                elif rel in _CONCEPTNET_SEMANTIC_RELATIONS:
                    rel_batch.append((s_lang, s_term, rel, o_term, weight, src_name))
            elif rel == "Synonym":
                if s_lang in ("si", "ta") and o_lang == "en":
                    align_batch.append((s_lang, s_term, o_term, min(1.0, weight / 3.0), src_name))
                elif o_lang in ("si", "ta") and s_lang == "en":
                    align_batch.append((o_lang, o_term, s_term, min(1.0, weight / 3.0), src_name))

            if len(hyp_batch) + len(rel_batch) + len(align_batch) >= batch_size:
                _flush()

    _flush()
    return IngestionReport(
        source=src_name,
        records_read=read_count,
        entries_written=hyp_written,
        relations_written=rel_written,
        alignments_written=align_written,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        extra={"hypernyms_written": hyp_written, "min_weight": min_weight},
    )


def _commit_bilingual_chunk(
    store: Any,
    records: list[LexicalOntologyRecord],
    alignments: list[tuple[str, str, str, float, str]],
    source: str,
) -> tuple[int, int]:
    written = 0
    aligned = 0
    if records:
        written, _ = store.upsert_many(records, source=source, preserve_existing=True)
    if alignments:
        aligned = store.upsert_concept_alignments(alignments)
    return written, aligned


def _ingest_osdb_pair(sinhala_dir: Path, store: Any) -> IngestionReport | None:
    sn2en_path = sinhala_dir / "osdb-sn2en.json"
    en2sn_path = sinhala_dir / "osdb-en2sn.json"
    if not sn2en_path.exists() and not en2sn_path.exists():
        return None
    t0 = time.perf_counter()
    records: list[LexicalOntologyRecord] = []
    alignments: list[tuple[str, str, str, float, str]] = []
    read_count = 0

    if sn2en_path.exists():
        items = json.loads(sn2en_path.read_text(encoding="utf-8"))
        for obj in items:
            if not isinstance(obj, dict):
                continue
            read_count += 1
            si_word = str(obj.get("word", "")).strip()
            defs = [str(d).strip() for d in (obj.get("definitions") or []) if str(d).strip()]
            if not si_word or " " in si_word or not defs:
                continue
            records.append(
                LexicalOntologyRecord(
                    lemma=si_word,
                    pos="noun",
                    language="si",
                    definition="; ".join(defs[:3])[:240],
                    domain="bilingual",
                )
            )
            for d in defs[:4]:
                for en_c in _normalize_gloss_candidates(d, max_words=2)[:3]:
                    alignments.append(("si", si_word, en_c, 0.90, "osdb-sn2en"))

    if en2sn_path.exists():
        items = json.loads(en2sn_path.read_text(encoding="utf-8"))
        for obj in items:
            if not isinstance(obj, dict):
                continue
            read_count += 1
            en_word = str(obj.get("word", "")).strip().lower()
            if not _TOKEN_CLEAN_RE.match(en_word) or len(en_word.split()) > 2:
                continue
            defs = [str(d).strip() for d in (obj.get("definitions") or []) if str(d).strip()]
            for si_term in defs[:4]:
                if si_term and " " not in si_term:
                    alignments.append(("si", si_term, en_word, 0.85, "osdb-en2sn"))

    w, a = _commit_bilingual_chunk(store, records, alignments, "osdb")
    return IngestionReport(
        source="bilingual:osdb",
        records_read=read_count,
        entries_written=w,
        alignments_written=a,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
    )


def _ingest_en_si_filtered_tsv(sinhala_dir: Path, store: Any) -> IngestionReport | None:
    tsv_path = sinhala_dir / "En-Si-dict-filtered-V2.tsv"
    if not tsv_path.exists():
        return None
    t0 = time.perf_counter()
    alignments: list[tuple[str, str, str, float, str]] = []
    records: list[LexicalOntologyRecord] = []
    seen_si: set[str] = set()
    read_count = 0

    with tsv_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            read_count += 1
            en_term = parts[0].strip().lower()
            si_term = parts[1].strip()
            if not _TOKEN_CLEAN_RE.match(en_term) or len(en_term.split()) > 2:
                continue
            if not si_term or " " in si_term or len(si_term) > 48:
                continue
            alignments.append(("si", si_term, en_term, 0.72, "en-si-filtered"))
            if si_term not in seen_si:
                seen_si.add(si_term)
                records.append(
                    LexicalOntologyRecord(
                        lemma=si_term,
                        pos="noun",
                        language="si",
                        definition=en_term,
                        domain="bilingual",
                    )
                )

    w, a = _commit_bilingual_chunk(store, records, alignments, "en-si-filtered")
    return IngestionReport(
        source="bilingual:En-Si-dict-filtered-V2",
        records_read=read_count,
        entries_written=w,
        alignments_written=a,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
    )


def _ingest_se_dictionary_zip(sinhala_dir: Path, store: Any) -> IngestionReport | None:
    zip_path = sinhala_dir / "SEDictionary-master.zip"
    if not zip_path.exists():
        return None
    t0 = time.perf_counter()
    alignments: list[tuple[str, str, str, float, str]] = []
    records: list[LexicalOntologyRecord] = []
    seen_si: set[str] = set()
    read_count = 0

    with zipfile.ZipFile(zip_path) as zf:
        raw = json.loads(zf.read("SEDictionary-master/ENtoSI.json"))
        for obj in raw:
            if not isinstance(obj, dict):
                continue
            for en_key, si_val in obj.items():
                read_count += 1
                en_clean = str(en_key).strip().lower()
                si_clean = str(si_val).strip()
                if not _TOKEN_CLEAN_RE.match(en_clean) or len(en_clean.split()) > 2:
                    continue
                if not si_clean or " " in si_clean or len(si_clean) > 48:
                    continue
                alignments.append(("si", si_clean, en_clean, 0.76, "sedictionary"))
                if si_clean not in seen_si:
                    seen_si.add(si_clean)
                    records.append(
                        LexicalOntologyRecord(
                            lemma=si_clean,
                            pos="noun",
                            language="si",
                            definition=en_clean,
                            domain="bilingual",
                        )
                    )

    w, a = _commit_bilingual_chunk(store, records, alignments, "sedictionary")
    return IngestionReport(
        source="bilingual:SEDictionary",
        records_read=read_count,
        entries_written=w,
        alignments_written=a,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
    )


def _ingest_eng2tamil_zip(tamil_dir: Path, store: Any) -> IngestionReport | None:
    zip_path = tamil_dir / "eng2tamildictionary-master.zip"
    if not zip_path.exists():
        return None
    t0 = time.perf_counter()
    alignments: list[tuple[str, str, str, float, str]] = []
    records: list[LexicalOntologyRecord] = []
    seen_ta: set[str] = set()
    read_count = 0

    with zipfile.ZipFile(zip_path) as zf:
        items = json.loads(zf.read("eng2tamildictionary-master/dictionary.json"))
        for obj in items:
            if not isinstance(obj, dict):
                continue
            read_count += 1
            en_term = str(obj.get("eng", "")).strip().lower()
            ta_raw = _POS_PREFIX_RE.sub("", str(obj.get("tamil", "")).strip()).rstrip(".")
            if not _TOKEN_CLEAN_RE.match(en_term) or len(en_term.split()) > 2:
                continue
            for cand in [c.strip() for c in re.split(r"[,;]", ta_raw) if c.strip()][:4]:
                if " " in cand or len(cand) > 48:
                    continue
                alignments.append(("ta", cand, en_term, 0.85, "eng2tamildict"))
                if cand not in seen_ta:
                    seen_ta.add(cand)
                    records.append(
                        LexicalOntologyRecord(
                            lemma=cand,
                            pos="noun",
                            language="ta",
                            definition=en_term,
                            domain="bilingual",
                        )
                    )

    w, a = _commit_bilingual_chunk(store, records, alignments, "eng2tamildict")
    return IngestionReport(
        source="bilingual:eng2tamildict",
        records_read=read_count,
        entries_written=w,
        alignments_written=a,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
    )


def _ingest_powertamil_and_rprabhu(tamil_dir: Path, store: Any) -> list[IngestionReport]:
    reports: list[IngestionReport] = []
    pt_zip = tamil_dir / "powertamil-dictionary-master.zip"
    if pt_zip.exists():
        t0 = time.perf_counter()
        records: list[LexicalOntologyRecord] = []
        with zipfile.ZipFile(pt_zip) as zf:
            fd, tmp_path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
            try:
                Path(tmp_path).write_bytes(zf.read("powertamil-dictionary-master/PTTamil.db"))
                for word, meanings in store.iter_external_sqlite_rows(
                    tmp_path,
                    table="dictionary",
                    columns=("word", "meanings"),
                ):
                    w = str(word or "").strip()
                    m = str(meanings or "").strip()[:240]
                    if w and " " not in w and m:
                        records.append(
                            LexicalOntologyRecord(
                                lemma=w,
                                pos="noun",
                                language="ta",
                                definition=m,
                                domain="monolingual",
                            )
                        )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        w_cnt, _ = _commit_bilingual_chunk(store, records, [], "powertamil")
        reports.append(
            IngestionReport(
                source="bilingual:powertamil",
                records_read=len(records),
                entries_written=w_cnt,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        )

    rp_zip = tamil_dir / "rprabhu-TamilDictionary-master.zip"
    if rp_zip.exists():
        t0 = time.perf_counter()
        records_rp: list[LexicalOntologyRecord] = []
        with zipfile.ZipFile(rp_zip) as zf:
            for name in zf.namelist():
                if not (name.endswith(".json") and "/json/" in name):
                    continue
                data = json.loads(zf.read(name))
                if not isinstance(data, dict):
                    continue
                for word, defn in data.items():
                    w = str(word).strip()
                    m = str(defn).strip()[:240]
                    if w and " " not in w and m:
                        records_rp.append(
                            LexicalOntologyRecord(
                                lemma=w,
                                pos="noun",
                                language="ta",
                                definition=m,
                                domain="monolingual",
                            )
                        )
        w_cnt, _ = _commit_bilingual_chunk(store, records_rp, [], "rprabhu-tamil")
        reports.append(
            IngestionReport(
                source="bilingual:rprabhu",
                records_read=len(records_rp),
                entries_written=w_cnt,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        )
    return reports


def _ingest_tamilinaiyam_zip(tamil_dir: Path, store: Any) -> IngestionReport | None:
    zip_path = tamil_dir / "Tamilinaiyam-Dictionary-master.zip"
    if not zip_path.exists():
        return None
    t0 = time.perf_counter()
    records: list[LexicalOntologyRecord] = []
    alignments: list[tuple[str, str, str, float, str]] = []
    read_count = 0

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json") or name.endswith("/DB.json"):
                continue
            try:
                data = json.loads(zf.read(name))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            for word, entry in data.items():
                read_count += 1
                w = str(word).strip()
                if not w or " " in w or len(w) > 48:
                    continue
                gloss_src = ""
                if isinstance(entry, dict):
                    gloss_src = str(entry.get("g") or entry.get("h") or "")
                elif isinstance(entry, str):
                    gloss_src = entry
                if not gloss_src:
                    continue
                clean_gloss = re.sub(r"[\x00-\x1f]+", " ", gloss_src.split("¦", 1)[-1]).strip()
                records.append(
                    LexicalOntologyRecord(
                        lemma=w,
                        pos="noun",
                        language="ta",
                        definition=clean_gloss[:240],
                        domain="lexicon",
                    )
                )
                for en_c in _normalize_gloss_candidates(clean_gloss.split(".")[0], max_words=2)[:2]:
                    alignments.append(("ta", w, en_c, 0.80, "tamilinaiyam"))

    w_cnt, a_cnt = _commit_bilingual_chunk(store, records, alignments, "tamilinaiyam")
    return IngestionReport(
        source="bilingual:Tamilinaiyam",
        records_read=read_count,
        entries_written=w_cnt,
        alignments_written=a_cnt,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
    )


def ingest_bilingual_dictionaries(dict_root: Path | str, store: Any) -> list[IngestionReport]:
    """Ingest curated licensed Sinhala and Tamil bilingual/monolingual dictionaries."""
    root = Path(dict_root)
    reports: list[IngestionReport] = []
    si_dir = root / "sinhala"
    ta_dir = root / "tamil"
    for rep in (
        _ingest_osdb_pair(si_dir, store) if si_dir.exists() else None,
        _ingest_en_si_filtered_tsv(si_dir, store) if si_dir.exists() else None,
        _ingest_se_dictionary_zip(si_dir, store) if si_dir.exists() else None,
        _ingest_eng2tamil_zip(ta_dir, store) if ta_dir.exists() else None,
        _ingest_tamilinaiyam_zip(ta_dir, store) if ta_dir.exists() else None,
    ):
        if rep is not None:
            reports.append(rep)
    if ta_dir.exists():
        reports.extend(_ingest_powertamil_and_rprabhu(ta_dir, store))
    return reports


def ingest_dakshina_transliterations(
    dakshina_dir: Path | str,
    store: Any,
    *,
    languages: Sequence[str] = ("si", "ta"),
    split: str = "train",
) -> IngestionReport:
    """Ingest Dakshina ``<native>\\t<romanized>\\t<count>`` lexicon split (default: train)."""
    t0 = time.perf_counter()
    base = Path(dakshina_dir)
    if (base / "dakshina_dataset_v1.0").exists():
        base = base / "dakshina_dataset_v1.0"

    rows: list[tuple[str, str, str, int, str]] = []
    read_count = 0
    for lang in languages:
        tsv = base / lang / "lexicons" / f"{lang}.translit.sampled.{split}.tsv"
        if not tsv.exists():
            continue
        with tsv.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                read_count += 1
                native = parts[0].strip()
                romanized = parts[1].strip().lower()
                try:
                    cnt = int(parts[2])
                except ValueError:
                    cnt = 1
                if native and romanized and " " not in romanized:
                    rows.append((lang, romanized, native, max(1, cnt), f"dakshina:{split}"))

    written = store.upsert_transliterations(rows)
    return IngestionReport(
        source=f"dakshina:{split}",
        records_read=read_count,
        transliterations_written=written,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        extra={"languages": list(languages), "split": split},
    )


def evaluate_dakshina_transliteration(
    dakshina_dir: Path | str,
    store: Any,
    *,
    language: str,
    split: str = "test",
    limit: int = 5000,
) -> dict[str, Any]:
    """Evaluate romanized -> native transliteration top-1 / top-3 recall on held-out split."""
    base = Path(dakshina_dir)
    if (base / "dakshina_dataset_v1.0").exists():
        base = base / "dakshina_dataset_v1.0"
    tsv = base / language / "lexicons" / f"{language}.translit.sampled.{split}.tsv"
    if not tsv.exists():
        return {"language": language, "split": split, "total": 0, "top1": 0.0, "top3": 0.0}

    targets_by_rom: dict[str, set[str]] = {}
    with tsv.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            nat, rom = parts[0].strip(), parts[1].strip().lower()
            if nat and rom:
                targets_by_rom.setdefault(rom, set()).add(nat)
            if len(targets_by_rom) >= limit:
                break

    top1_hits = 0
    top3_hits = 0
    total = len(targets_by_rom)
    for rom, valid_natives in targets_by_rom.items():
        preds = store.transliterate_romanized(rom, language=language, limit=3)
        if preds and preds[0] in valid_natives:
            top1_hits += 1
        if any(p in valid_natives for p in preds):
            top3_hits += 1

    return {
        "language": language,
        "split": split,
        "total": total,
        "top1": round(top1_hits / max(1, total), 4),
        "top3": round(top3_hits / max(1, total), 4),
    }


def ingest_gutenberg_enrichment(
    gutenberg_dir: Path | str,
    store: Any,
    *,
    max_webster: int = 100_000,
    max_moby: int = 40_000,
) -> list[IngestionReport]:
    """Fill missing English definitions (Webster 1913) and synonyms (Moby Thesaurus)."""
    root = Path(gutenberg_dir)
    reports: list[IngestionReport] = []

    webster_path = root / "websters-1913.json"
    if webster_path.exists():
        t0 = time.perf_counter()
        raw = json.loads(webster_path.read_text(encoding="utf-8"))
        rows: list[tuple[str, str, str, str, str, str, str, str, str, int]] = []
        read_count = 0
        for word, defn in raw.items():
            if len(rows) >= max_webster:
                break
            read_count += 1
            w = str(word).strip().lower()
            d = str(defn).strip()[:280]
            if not w or not d or " " in w or not w.isalpha():
                continue
            rows.append(("en", w, "noun", d, "[]", "[]", "[]", "general", "websters-1913", 0))
        with store._conn:
            store._conn.executemany(
                """
                INSERT INTO lexical_entries (
                    language, lemma, pos, definition,
                    hypernyms_json, synonyms_json, thematic_roles_json, domain, source, frequency
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(language, lemma, pos) DO UPDATE SET
                    definition = CASE
                        WHEN lexical_entries.definition = '' THEN excluded.definition
                        ELSE lexical_entries.definition
                    END
                """,
                rows,
            )
        store._lookup_cache.clear()
        reports.append(
            IngestionReport(
                source="enrichment:websters-1913",
                records_read=read_count,
                entries_written=len(rows),
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        )

    moby_path = root / "moby-thesaurus.txt"
    if moby_path.exists():
        t0 = time.perf_counter()
        syn_rows: list[tuple[str, str, str]] = []
        read_count = 0
        with moby_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if len(syn_rows) >= max_moby:
                    break
                parts = [p.strip().lower() for p in line.split(",") if p.strip()]
                if len(parts) < 2:
                    continue
                read_count += 1
                head = parts[0]
                if " " in head or not head.isalpha():
                    continue
                syns = [s for s in parts[1:] if " " not in s and s.isalpha() and s != head][:8]
                if syns:
                    syn_rows.append((json.dumps(syns, ensure_ascii=False), "en", head))
        with store._conn:
            store._conn.executemany(
                """
                UPDATE lexical_entries
                SET synonyms_json = ?
                WHERE language = ? AND lemma = ? AND synonyms_json = '[]'
                """,
                syn_rows,
            )
        store._lookup_cache.clear()
        reports.append(
            IngestionReport(
                source="enrichment:moby-thesaurus",
                records_read=read_count,
                entries_written=len(syn_rows),
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
        )

    return reports
