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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import ConstructionNode, HCIREdge, SkillNode
from hbllm.hcir.language.english.morphology import (
    INVARIANT_S_NOUNS,
    singularize_noun,
)
from hbllm.hcir.language.ontology_bootstrap import LexicalOntologyRecord
from hbllm.hcir.nodes import BeliefNode, ConceptNode, HCIREdgeType, NodeLifecycle
from hbllm.hcir.seed.loader import load_constructions_spec
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope

logger = logging.getLogger(__name__)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_IS_A_RE = re.compile(
    r"^(?:the\s+|a\s+|an\s+)?([a-zA-Z][a-zA-Z0-9_\s-]{1,35}?)\s+(?:is|are)\s+(?:a|an)?\s*(.+)$",
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
    r"^\*?\s*\*\*([a-zA-Z0-9_\s-]{2,40})\*\*:\s*(.+)$",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(?:is|are|was|were)\s+not\b|\bnever\b", re.IGNORECASE)
_HEDGE_RE = re.compile(
    r"\b(?:may|might|could|can)\s+(?:cause|trigger|lead|produce|require|need)\b",
    re.IGNORECASE,
)
_GARBAGE_OBJ = frozenset(
    "the a an if when must determines requiring every only no is are deep neural level high type kind form".split()
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
    "this that there here which what when where they these those it he she we you who whom "
    "whose i me him her us them and or but with from for to at by about more some many such "
    "each every all both either neither very also then thus".split()
)


@dataclass(frozen=True)
class ExtractedProposition:
    """A structured taxonomic, causal, dependency, or SVO proposition extracted from text."""

    subject: str
    relation: (
        str  # "is_a" | "causes" | "requires" | "enables" | "has_property" | <verb> | "not_<rel>"
    )
    obj: str
    source_file: str
    sentence: str
    polarity: bool = True
    modifiers: tuple[tuple[str, str], ...] = ()
    paragraph_idx: int = 0
    sentence_idx: int = 0


@dataclass
class ReadingCurriculumStats:
    """Summary metrics for a Track T4 reading run."""

    files_read: int = 0
    sentences_scanned: int = 0
    taxonomic_extracted: int = 0
    causal_extracted: int = 0
    dependency_extracted: int = 0
    relational_extracted: int = 0
    nodes_committed: int = 0
    edges_committed: int = 0
    store_hypernyms_added: int = 0
    elapsed_ms: float = 0.0
    sample_propositions: list[dict[str, str]] = field(default_factory=list)


def _clean_term(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\u0D80-\u0DFF\u0B80-\u0BFF-]+", "_", raw.strip().lower()).strip(
        "_-"
    )


def _singularize_basic(term: str) -> str:
    return singularize_noun(term)


_NP_BOUNDARY_RE = re.compile(
    r"\s+(?:that|which|who|whom|whose|where|when|because|while|although|though|"
    r"with|toward|towards|from|for|by|in|on|at|into|to|through|across|between|under|over)\b|[,;]",
    re.IGNORECASE,
)


def _extract_head_from_np(np_text: str) -> str:
    """Extract semantic head noun from noun phrase, truncating relative/PP clauses & 'type of'."""
    m_type = re.search(
        r"\b(?:type|kind|form|sort)\s+of\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_-]+)",
        np_text,
        re.IGNORECASE,
    )
    if m_type:
        return _singularize_basic(_clean_term(m_type.group(1)))
    head_span = _NP_BOUNDARY_RE.split(np_text, maxsplit=1)[0]
    if " of " in head_span:
        head_span = head_span.split(" of ", 1)[0]
    words = re.findall(r"[a-zA-Z0-9_-]+", head_span)
    content_words = [
        w
        for w in words
        if w.lower() not in {"a", "an", "the", "this", "that", "these", "those", "very", "quite"}
    ]
    if not content_words:
        return ""
    return _singularize_basic(_clean_term(content_words[-1]))


def _extract_np_modifiers(np_text: str) -> tuple[tuple[str, str], ...]:
    """Extract pre-head adjectival/nominal modifiers from a noun phrase."""
    head_span = _NP_BOUNDARY_RE.split(np_text, maxsplit=1)[0]
    if " of " in head_span:
        head_span = head_span.split(" of ", 1)[0]
    words = [
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9_-]+", head_span)
        if w.lower()
        not in {
            "a",
            "an",
            "the",
            "this",
            "that",
            "these",
            "those",
            "very",
            "quite",
            "type",
            "kind",
            "form",
            "sort",
            "of",
        }
    ]
    if len(words) <= 1:
        return ()
    return tuple(("adj", _clean_term(w)) for w in words[:-1] if len(w) >= 2)


_PROTECTED_ABBREV_RE = re.compile(
    r"\b(mr|mrs|ms|dr|prof|st|e\.g|i\.e|vs|etc)\.",
    re.IGNORECASE,
)


def _split_sentences_safe(text: str) -> list[str]:
    """Split prose into sentences while preserving honorifics & abbreviations (Mr., Dr., e.g.)."""
    holders: dict[str, str] = {}

    def _mask(match: re.Match[str]) -> str:
        tok = f"__ABBR{len(holders)}__"
        holders[tok] = match.group(0)
        return tok

    masked = _PROTECTED_ABBREV_RE.sub(_mask, text)
    parts = _SENT_SPLIT_RE.split(masked)
    out: list[str] = []
    for part in parts:
        restored = part
        for tok, orig in holders.items():
            restored = restored.replace(tok, orig)
        if restored.strip():
            out.append(restored.strip())
    return out


_CORE_PARSER_SINGLETON: Any | None = None
_PARSER_CACHE: dict[tuple[str, int], Any] = {}


def _detect_script_language(text: str) -> str:
    """Detect whether *text* is predominantly Sinhala, Tamil, or English script."""
    si_count = 0
    ta_count = 0
    for ch in text:
        cp = ord(ch)
        if 0x0D80 <= cp <= 0x0DFF:
            si_count += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            ta_count += 1
    if si_count > 0 and si_count >= ta_count:
        return "si"
    if ta_count > 0:
        return "ta"
    return "en"


def _get_parser(
    parser: Any | None,
    *,
    ontology_store: Any | None = None,
    language: str = "en",
) -> Any:
    global _CORE_PARSER_SINGLETON
    if language == "en" and parser is not None:
        return parser
    if language == "en" and ontology_store is None:
        if _CORE_PARSER_SINGLETON is None:
            from hbllm.hcir.language.english.parser import EnglishParser

            _CORE_PARSER_SINGLETON = EnglishParser()
        return _CORE_PARSER_SINGLETON

    store_key = id(ontology_store) if ontology_store is not None else 0
    cache_key = (language, store_key)
    cached = _PARSER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if len(_PARSER_CACHE) >= 8:
        _PARSER_CACHE.clear()

    if language == "si":
        from hbllm.hcir.language.sinhala.lexicon import SinhalaLexicon
        from hbllm.hcir.language.sinhala.parser import SinhalaParser

        built: Any = SinhalaParser(lexicon=SinhalaLexicon(ontology_store=ontology_store))
    elif language == "ta":
        from hbllm.hcir.language.tamil.lexicon import TamilLexicon
        from hbllm.hcir.language.tamil.parser import TamilParser

        built = TamilParser(lexicon=TamilLexicon(ontology_store=ontology_store))
    else:
        from hbllm.hcir.language.english.lexicon import EnglishLexicon
        from hbllm.hcir.language.english.parser import EnglishParser

        built = EnglishParser(lexicon=EnglishLexicon(ontology_store=ontology_store))

    _PARSER_CACHE[cache_key] = built
    return built


def _split_coordinated_vps(sent: str, parser: Any) -> list[str]:
    """Split 'Subj VP1 and VP2' into ['Subj VP1', 'Subj VP2'] when RHS starts with a verb."""
    if " and " not in sent.lower():
        return [sent]
    from hbllm.hcir.language.english.lexicon import EnglishPOS

    tokens = parser._morphology.lemmatize_sequence(sent)
    content = [t for t in tokens if t.pos != EnglishPOS.PUNCT]
    # Find subject NP span (Det? Adj* Noun+) before first VERB/AUX
    subj_end = -1
    for i, t in enumerate(content):
        if i > 0 and t.pos in (EnglishPOS.VERB, EnglishPOS.AUX):
            subj_end = i
            break
    if subj_end <= 0:
        return [sent]

    subject_surface = " ".join(t.surface for t in content[:subj_end])
    rest = content[subj_end:]
    # Pass 1: Prefer "and" followed strictly by a primary VERB (e.g. "... and releases oxygen")
    for i in range(1, len(rest) - 1):
        if rest[i].surface.lower() == "and" and rest[i + 1].pos == EnglishPOS.VERB:
            lhs = f"{subject_surface} " + " ".join(t.surface for t in rest[:i])
            rhs = f"{subject_surface} " + " ".join(t.surface for t in rest[i + 1 :])
            return [lhs.strip(), rhs.strip()]
    # Pass 2: Fall back to verb-capable token only if not a noun inside a coordinated NP
    for i in range(2, len(rest) - 1):
        if rest[i].surface.lower() == "and":
            nxt = rest[i + 1]
            if (
                "verb" in nxt.features.get("alt_pos", "")
                and nxt.pos != EnglishPOS.NOUN
                and i + 2 < len(rest)
                and rest[i + 2].pos != EnglishPOS.PREP
            ):
                lhs = f"{subject_surface} " + " ".join(t.surface for t in rest[:i])
                rhs = f"{subject_surface} " + " ".join(t.surface for t in rest[i + 1 :])
                return [lhs.strip(), rhs.strip()]
    return [sent]


def _extract_subject_label(ref: Any) -> str:
    if ref is None:
        return ""
    raw = (getattr(ref, "raw_text", "") or "").strip()
    concept = (getattr(ref, "concept_name", "") or "").strip()
    # Preserve multi-word proper names like "Elizabeth Bennet" or "Mr. Darcy"
    if concept and " " in concept:
        return _clean_term(concept)
    # If raw_text has 2+ capitalized tokens (e.g. "Elizabeth Bennet"), keep full proper name
    raw_words = [
        w
        for w in re.findall(r"[A-Za-z0-9_'-]+", raw)
        if w.lower() not in {"a", "an", "the", "this", "that"}
    ]
    if len(raw_words) >= 2 and all(w[0].isupper() for w in raw_words):
        return _clean_term("_".join(raw_words))
    # Preserve compound subjects when raw has Adj + Noun only for proper/capitalized compounds;
    # for common nouns like "heavy rain", include compound modifier when needed or keep concept_name.
    cleaned = _clean_term(concept or (raw_words[-1] if raw_words else ""))
    if cleaned in INVARIANT_S_NOUNS:
        return cleaned
    if raw_words and raw_words[-1][0].isupper() and cleaned.endswith(("as", "os")):
        return cleaned
    return _singularize_basic(cleaned)


def _extract_object_label(ref: Any, *, preserve_head: bool = False) -> str:
    if ref is None:
        return ""
    raw = (getattr(ref, "raw_text", "") or "").strip()
    concept = (getattr(ref, "concept_name", "") or "").strip()
    if preserve_head and raw:
        head = _extract_head_from_np(raw)
        if head:
            return head
    cleaned = _clean_term(concept or raw)
    if cleaned in INVARIANT_S_NOUNS:
        return cleaned
    return _singularize_basic(cleaned)


_INTRO_PREPS = frozenset(
    "in on at by for with during while when because since although though if unless".split()
)


def _strip_intro_clause(sent: str) -> str:
    """Strip single leading adverb/PP or subordinate clause before the first comma."""
    stripped = _INTRO_STRIP_RE.sub("", sent).strip()
    if "," in stripped:
        prefix, rest = stripped.split(",", 1)
        first_word = prefix.strip().split()[0].lower() if prefix.strip() else ""
        if first_word in _INTRO_PREPS and len(rest.strip().split()) >= 3:
            return rest.strip()
    return stripped


_TRAILING_DIGITS_RE = re.compile(r"_\d+$")
_NOISY_FRONTMATTER_RE = re.compile(
    r"^(?:copyright|chapter|part|book|volume|frontispiece|illustrations?\b|"
    r"contents\b|table\s+of\s+contents|appendix\b|index\b)\b",
    re.IGNORECASE,
)
_ISA_SUBJ_NOISE_WORDS = frozenset(
    "i me my myself you your we our he she they where when why how who but and said know".split()
)


def _is_noisy_prose_line(line: str) -> bool:
    """Return True for captions, frontmatter metadata, dividers, ALL-CAPS headers, or digit-heavy lines."""
    s = line.strip()
    if not s:
        return True
    if s.startswith("[") and "]" in s:
        return True
    if set(s) <= set("-*=~_ \t"):
        return True
    alpha_chars = [ch for ch in s if ch.isalpha()]
    if len(alpha_chars) >= 6 and s.isupper():
        return True
    digits = sum(1 for ch in s if ch.isdigit())
    if digits / max(1, len(s)) > 0.25:
        return True
    if _NOISY_FRONTMATTER_RE.match(s) and len(s) < 60:
        return True
    return False


def extract_propositions_from_text(
    text: str,
    source_file: str = "",
    *,
    include_extended: bool = False,
    parser: Any | None = None,
    ontology_store: Any | None = None,
) -> list[ExtractedProposition]:
    """Deterministically extract SVO, taxonomic, causal, and property propositions via parsers.

    Extracts subject-verb-object triples with explicit polarity (``not_<rel>`` when
    negated), thematic role modifiers (recipient, instrument, source, location),
    intra-paragraph pronoun coreference, and Sinhala/Tamil declarative assertions.
    """
    from hbllm.hcir.language.core.semantic_frame import FrameType, ThematicRole

    found: list[ExtractedProposition] = []
    seen: set[tuple[str, str, str, bool]] = set()
    max_len = 600 if include_extended else 240
    active_parser = _get_parser(parser, ontology_store=ontology_store, language="en")

    def _add(
        subj_raw: str,
        rel: str,
        obj_raw: str,
        sentence: str,
        *,
        polarity: bool = True,
        modifiers: tuple[tuple[str, str], ...] = (),
        paragraph_idx: int = 0,
        sentence_idx: int = 0,
    ) -> bool:
        subj = _clean_term(subj_raw)
        obj = _clean_term(obj_raw)
        if (
            not subj
            or not obj
            or subj == obj
            or subj in _SKIP_SUBJECTS
            or obj in _SKIP_SUBJECTS
            or subj.isdigit()
            or obj.isdigit()
            or _TRAILING_DIGITS_RE.search(subj)
            or _TRAILING_DIGITS_RE.search(obj)
            or len(subj) < 2
            or len(obj) < 2
        ):
            return False
        final_rel = rel if polarity or rel.startswith("not_") else f"not_{rel}"
        key = (subj, final_rel, obj, polarity)
        if key in seen:
            return False
        seen.add(key)
        found.append(
            ExtractedProposition(
                subject=subj,
                relation=final_rel,
                obj=obj,
                source_file=source_file,
                sentence=sentence[:200],
                polarity=polarity,
                modifiers=modifiers,
                paragraph_idx=paragraph_idx,
                sentence_idx=sentence_idx,
            )
        )
        return True

    def _extract_clause_via_parser(
        clause: str,
        orig_sent: str,
        *,
        active_lang_parser: Any | None = None,
        paragraph_idx: int = 0,
        sentence_idx: int = 0,
    ) -> bool:
        added_any = False
        clause_parser = active_lang_parser or active_parser
        frame = clause_parser.parse(clause)
        if frame.is_error or frame.frame_type != FrameType.ASSERTION:
            return False

        frame_lang = getattr(frame.metadata, "language", "en")
        is_non_english = frame_lang != "en"
        polarity = bool(getattr(frame.metadata, "polarity", True))
        subj_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        subj = _extract_subject_label(subj_ref)
        if not subj:
            return False

        pat_ref = frame.get_role(ThematicRole.PATIENT)
        rec_ref = frame.get_role(ThematicRole.RECIPIENT)
        src_ref = frame.get_role(ThematicRole.SOURCE)
        loc_ref = frame.get_role(ThematicRole.LOCATION)
        dest_ref = frame.get_role(ThematicRole.DESTINATION)
        inst_ref = frame.get_role(ThematicRole.INSTRUMENT)

        mod_list: list[tuple[str, str]] = []
        if rec_ref is not None and rec_ref.concept_name:
            rec_mod = _clean_term(rec_ref.concept_name)
            if rec_mod:
                mod_list.append(("recipient", rec_mod))
        if src_ref is not None and src_ref.concept_name:
            src_mod = _clean_term(src_ref.concept_name)
            if src_mod:
                mod_list.append(("from", src_mod))
        for role_ref in (loc_ref, dest_ref, inst_ref):
            if role_ref is not None and role_ref.concept_name and frame.predicate != "is_property":
                prep = str(role_ref.properties.get("prep", "")).strip().lower()
                tgt_mod = _clean_term(role_ref.concept_name)
                if prep and tgt_mod:
                    mod_list.append((prep, tgt_mod))

        if frame.predicate == "is_a" and pat_ref is not None:
            if is_non_english and pat_ref.concept_name:
                obj = _extract_object_label(pat_ref, preserve_head=False)
                adj_mods: tuple[tuple[str, str], ...] = ()
            else:
                pat_raw = pat_ref.raw_text or pat_ref.concept_name or ""
                obj = _extract_head_from_np(pat_raw) or _extract_object_label(pat_ref)
                adj_mods = _extract_np_modifiers(pat_raw)
            if obj and obj not in _GARBAGE_OBJ:
                if _add(
                    subj,
                    "is_a",
                    obj,
                    orig_sent,
                    polarity=polarity,
                    modifiers=tuple([*adj_mods, *mod_list]),
                    paragraph_idx=paragraph_idx,
                    sentence_idx=sentence_idx,
                ):
                    added_any = True
            return added_any

        def _emit(s: str, r: str, o: str, mods: tuple[tuple[str, str], ...] = ()) -> bool:
            return _add(
                s,
                r,
                o,
                orig_sent,
                polarity=polarity,
                modifiers=mods,
                paragraph_idx=paragraph_idx,
                sentence_idx=sentence_idx,
            )

        if frame.predicate == "is_property" and loc_ref is not None:
            prop_val = loc_ref.concept_name or (
                str(next(iter(loc_ref.properties.values()))) if loc_ref.properties else ""
            )
            prop_clean = _clean_term(prop_val)
            if prop_clean and prop_clean not in _GARBAGE_OBJ:
                added_any |= _emit(subj, "has_property", prop_clean)
            return added_any

        if frame.predicate in ("causes", "requires", "enables") and pat_ref is not None:
            obj = _extract_object_label(pat_ref, preserve_head=not is_non_english)
            inf_verb = str(pat_ref.properties.get("infinitive_verb", "")).strip().lower()
            if inf_verb and frame.predicate == "causes":
                raw_sc = _clean_term(subj_ref.raw_text) if (subj_ref and subj_ref.raw_text) else ""
                for s_lbl in (subj, *([raw_sc] if raw_sc and raw_sc != subj else [])):
                    if obj:
                        added_any |= _emit(s_lbl, "causes", f"{obj}_{inf_verb}")
                    added_any |= _emit(s_lbl, "causes", inf_verb)
            elif obj:
                added_any |= _emit(subj, frame.predicate, obj, tuple(mod_list))
            return added_any

        if frame.predicate.startswith("located_") or frame.predicate in ("below", "above", "near"):
            if loc_ref is not None and (loc_obj := _extract_object_label(loc_ref)):
                added_any |= _emit(subj, frame.predicate, loc_obj, tuple(mod_list))
            return added_any

        verb_rel = _clean_term(frame.predicate or "")
        if not verb_rel or verb_rel in ("be", "do", "have"):
            return False

        mods_tup = tuple(mod_list)
        if pat_ref is not None:
            pat_raw = pat_ref.raw_text or ""
            if " and " in pat_raw.lower():
                for op in (
                    p.strip() for p in re.split(r"\band\b", pat_raw, flags=re.I) if p.strip()
                ):
                    if op_head := _extract_head_from_np(op):
                        added_any |= _emit(subj, verb_rel, op_head, mods_tup)
            else:
                obj = _extract_object_label(pat_ref, preserve_head=not is_non_english)
                if obj and obj not in _GARBAGE_OBJ:
                    added_any |= _emit(subj, verb_rel, obj, mods_tup)
            for prep, tgt in mod_list:
                if prep == "into" and verb_rel in ("convert", "turn", "transform"):
                    added_any |= _emit(subj, "produce", tgt)
        elif loc_ref is not None or dest_ref is not None:
            if obl_obj := _extract_object_label(loc_ref or dest_ref):
                added_any |= _emit(subj, verb_rel, obl_obj, mods_tup)

        return added_any

    paragraphs = [p for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    for p_idx, para in enumerate(paragraphs):
        last_subject_raw = ""
        last_plural_subject_raw = ""
        last_person_subject_raw = ""
        last_object_raw = ""

        for s_idx, raw_sent in enumerate(_split_sentences_safe(para)):
            raw_trimmed = raw_sent.strip()
            if not raw_trimmed:
                continue

            lang = _detect_script_language(raw_trimmed)
            if lang in ("si", "ta"):
                lang_parser = _get_parser(None, ontology_store=ontology_store, language=lang)
                _extract_clause_via_parser(
                    raw_trimmed,
                    raw_trimmed,
                    active_lang_parser=lang_parser,
                    paragraph_idx=p_idx,
                    sentence_idx=s_idx,
                )
                continue

            if _is_noisy_prose_line(raw_trimmed):
                continue

            if _HEDGE_RE.search(raw_trimmed):
                continue

            gm = _GLOSSARY_IS_A_RE.match(raw_trimmed)
            if gm:
                term = gm.group(1).strip()
                body = gm.group(2).strip()
                first = body.split()[0].lower() if body.split() else ""
                if first not in _GARBAGE_OBJ and first not in {
                    "if",
                    "determines",
                    "the",
                    "must",
                    "when",
                }:
                    m_def = re.match(r"^(?:is\s+(?:a|an)\s+|(?:a|an)\s+)(.+)$", body, re.IGNORECASE)
                    if m_def:
                        obj = _extract_head_from_np(m_def.group(1))
                        if obj and obj not in _GARBAGE_OBJ:
                            _add(
                                _singularize_basic(_clean_term(term)),
                                "is_a",
                                obj,
                                raw_trimmed[:120],
                                paragraph_idx=p_idx,
                                sentence_idx=s_idx,
                            )
                continue

            sent = raw_trimmed.lstrip("#>-*` ")
            if len(sent) < 10 or len(sent) > max_len:
                continue

            sent_clean = (
                _strip_intro_clause(sent) if include_extended else _INTRO_STRIP_RE.sub("", sent)
            )

            # Intra-paragraph pronoun coreference (subject & mid-sentence object)
            if include_extended:
                m_pron = re.match(r"^(it|he|she|they)\s+(.+)$", sent_clean, re.IGNORECASE)
                if m_pron:
                    pron = m_pron.group(1).lower()
                    if pron in ("he", "she"):
                        antecedent = last_person_subject_raw or last_subject_raw
                    elif pron == "they":
                        antecedent = last_plural_subject_raw or last_subject_raw
                    else:
                        antecedent = last_subject_raw or last_object_raw
                    if antecedent:
                        sent_clean = f"{antecedent} {m_pron.group(2)}"

                sp_idx = sent_clean.find(" ")
                if sp_idx > 0:
                    head_part = sent_clean[:sp_idx]
                    tail_part = sent_clean[sp_idx:]
                    if last_object_raw and last_object_raw.lower() != head_part.lower():
                        tail_part = re.sub(
                            r"\b(?:it|them)\b",
                            last_object_raw,
                            tail_part,
                            flags=re.IGNORECASE,
                        )
                    if (
                        last_person_subject_raw
                        and last_person_subject_raw.lower() != head_part.lower()
                    ):
                        tail_part = re.sub(
                            r"\bhim\b",
                            last_person_subject_raw,
                            tail_part,
                            flags=re.IGNORECASE,
                        )
                        tail_part = re.sub(
                            r"\bher(?=\s*(?:$|[.,;!?]|\b(?:to|for|in|on|at|with|from|by|and|the|a|an)\b))",
                            last_person_subject_raw,
                            tail_part,
                            flags=re.IGNORECASE,
                        )
                    sent_clean = head_part + tail_part

            m_rc = re.match(
                r"^((?:the\s+|a\s+|an\s+)?[a-zA-Z][a-zA-Z0-9_\s-]{1,35}?)\s+(?:is|are)\s+(.+?)\s+(?:that|which|who)\s+(.+)$",
                sent_clean,
                re.IGNORECASE,
            )
            if m_rc:
                subj_phrase = m_rc.group(1).strip()
                head_np = m_rc.group(2).strip()
                rel_vp = m_rc.group(3).strip()
                subj_clean = _singularize_basic(
                    _clean_term(re.sub(r"^(?:the|a|an)\s+", "", subj_phrase, flags=re.I))
                )
                obj_head = _extract_head_from_np(head_np)
                if subj_clean and obj_head and obj_head not in _GARBAGE_OBJ:
                    _add(
                        subj_clean,
                        "is_a",
                        obj_head,
                        sent,
                        modifiers=_extract_np_modifiers(head_np),
                        paragraph_idx=p_idx,
                        sentence_idx=s_idx,
                    )
                    last_subject_raw = subj_phrase
                _extract_clause_via_parser(
                    f"{subj_phrase} {rel_vp}",
                    sent,
                    paragraph_idx=p_idx,
                    sentence_idx=s_idx,
                )
                continue

            sub_clauses = (
                _split_coordinated_vps(sent_clean, active_parser)
                if include_extended
                else [sent_clean]
            )

            clause_matched = False
            for clause in sub_clauses:
                if _extract_clause_via_parser(
                    clause,
                    sent,
                    paragraph_idx=p_idx,
                    sentence_idx=s_idx,
                ):
                    clause_matched = True

            if not clause_matched and not _NEGATION_RE.search(sent_clean):
                m_isa = _IS_A_RE.match(sent_clean)
                if m_isa:
                    subj_words = [
                        w.lower() for w in re.findall(r"[a-zA-Z0-9_'-]+", m_isa.group(1).strip())
                    ]
                    subj = _singularize_basic(_clean_term(m_isa.group(1)))
                    obj = _extract_head_from_np(m_isa.group(2))
                    if (
                        subj
                        and obj
                        and subj != obj
                        and obj not in _GARBAGE_OBJ
                        and len(subj_words) <= 4
                        and not (_ISA_SUBJ_NOISE_WORDS & set(subj_words))
                    ):
                        if _add(
                            subj,
                            "is_a",
                            obj,
                            sent,
                            modifiers=_extract_np_modifiers(m_isa.group(2)),
                            paragraph_idx=p_idx,
                            sentence_idx=s_idx,
                        ):
                            clause_matched = True

                if not clause_matched:
                    for rel, pattern in (
                        ("causes", _CAUSES_RE),
                        ("requires", _REQUIRES_RE),
                        ("enables", _ENABLES_RE),
                    ):
                        m = pattern.match(sent_clean)
                        if m and _add(
                            m.group(1),
                            rel,
                            m.group(2),
                            sent,
                            paragraph_idx=p_idx,
                            sentence_idx=s_idx,
                        ):
                            clause_matched = True
                            break

            if clause_matched and found:
                parsed_last = active_parser.parse(sent_clean)
                s_ref = parsed_last.get_role(ThematicRole.THEME) or parsed_last.get_role(
                    ThematicRole.AGENT
                )
                cand_subj = (
                    (s_ref.raw_text or s_ref.concept_name or "").strip()
                    if s_ref is not None
                    else ""
                )
                if cand_subj:
                    raw_split = cand_subj.split()
                    cand_words = [
                        w for w in raw_split if w.lower() not in {"a", "an", "the", "this", "that"}
                    ]
                    if cand_words and cand_words[0].lower() not in _SKIP_SUBJECTS:
                        last_subject_raw = cand_subj
                        if (
                            raw_split[0][:1].isupper()
                            and raw_split[0].lower() not in {"a", "an", "the", "this", "that"}
                            and not (len(cand_words) == 1 and cand_words[0].lower().endswith("s"))
                        ):
                            last_person_subject_raw = cand_subj
                        if (
                            cand_words[-1].lower().endswith("s")
                            and cand_words[-1].lower() not in INVARIANT_S_NOUNS
                        ):
                            last_plural_subject_raw = cand_subj

                p_ref = parsed_last.get_role(ThematicRole.PATIENT)
                if p_ref is not None:
                    obj_cand = (p_ref.raw_text or p_ref.concept_name or "").strip()
                    if obj_cand and obj_cand.lower() not in _SKIP_SUBJECTS:
                        last_object_raw = obj_cand

            if not include_extended:
                continue

            if not clause_matched:
                if (m_rel := _REL_CLAUSE_ENABLES_RE.search(sent_clean)) is not None:
                    _add(
                        m_rel.group(1),
                        "enables",
                        m_rel.group(2),
                        sent,
                        paragraph_idx=p_idx,
                        sentence_idx=s_idx,
                    )
                elif (m_en := _ENABLES_RE.match(sent_clean)) is not None:
                    _add(
                        m_en.group(1),
                        "enables",
                        m_en.group(2),
                        sent,
                        paragraph_idx=p_idx,
                        sentence_idx=s_idx,
                    )

            for m_ex in _EXEMPLIFY_RE.finditer(sent_clean):
                ex_skip = {"many", "several", "various", "some", "these", "those", "the", "other"}
                head_words = [w for w in m_ex.group(1).strip().split() if w.lower() not in ex_skip]
                if not head_words:
                    continue
                head_singular = _singularize_basic("_".join(head_words).lower())
                head_tail = _singularize_basic(head_words[-1].lower())
                for item_raw in re.split(r",|\band\b|\bor\b", m_ex.group(2), flags=re.IGNORECASE):
                    iw = [
                        w
                        for w in item_raw.strip().strip(".,;:").split()
                        if w.lower() not in {"a", "an", "the", "both", "such", "other"}
                    ]
                    if not iw or len(iw) > 4:
                        continue
                    iterm = "_".join(iw).lower()
                    _add(
                        iterm, "is_a", head_singular, sent, paragraph_idx=p_idx, sentence_idx=s_idx
                    )
                    if head_tail != head_singular:
                        _add(
                            iterm, "is_a", head_tail, sent, paragraph_idx=p_idx, sentence_idx=s_idx
                        )

            for m_purp in _PURPOSE_RE.finditer(sent_clean):
                mw = [
                    w
                    for w in m_purp.group(1).strip().split()
                    if w.lower()
                    not in {"a", "an", "the", "many", "several", "design", "use", "apply"}
                ]
                if mw:
                    _add(
                        _singularize_basic("_".join(mw).lower()),
                        "enables",
                        m_purp.group(2),
                        sent,
                        paragraph_idx=p_idx,
                        sentence_idx=s_idx,
                    )

            if m_bec := re.match(r"^because\s+([^,]+),\s*(.+)$", sent.strip(), flags=re.IGNORECASE):
                stop_b = {"can", "may", "will", "be", "is", "are"}
                c1w = [
                    w.lower()
                    for w in re.findall(r"[a-zA-Z]+", m_bec.group(1))
                    if w.lower() not in _SKIP_SUBJECTS and w.lower() not in stop_b
                ]
                c2w = [
                    w.lower()
                    for w in re.findall(r"[a-zA-Z]+", m_bec.group(2))
                    if w.lower() not in _SKIP_SUBJECTS
                    and w.lower() not in (stop_b | {"to", "like", "and"})
                ]
                if c1w and len(c2w) >= 2:
                    _add(
                        c1w[0],
                        "causes",
                        c2w[1] if len(c2w) > 2 else c2w[0],
                        sent,
                        paragraph_idx=p_idx,
                        sentence_idx=s_idx,
                    )

    return found


_STATIC_RELATIONS = frozenset(
    {"is_a", "has_property", "causes", "requires", "enables", "below", "above", "near"}
)


def commit_propositions(
    propositions: list[ExtractedProposition],
    workspace: Any,
    *,
    ontology_store: Any | None = None,
    grant: Any | None = None,
    author: str = "t4_reading_curriculum",
    batch_size: int = 100,
    link_temporal_events: bool = True,
) -> tuple[int, int, int]:
    """Commit extracted propositions into HCIR workspace & optional ontology store atomically.

    Returns ``(nodes_committed, edges_committed, store_hypernyms_added)``.
    """
    target_ws = (
        getattr(workspace, "persistent", None)
        if not hasattr(workspace, "commit_operations") and hasattr(workspace, "persistent")
        else workspace
    ) or workspace
    node_ops: list[TransactionOperation] = []
    edge_ops: list[TransactionOperation] = []
    created_concepts: set[str] = set()
    seen_edge_ids: set[str] = set()
    ws_graph = getattr(target_ws, "graph", None)
    existing_edges = getattr(ws_graph, "edges", {})
    store_records: list[LexicalOntologyRecord] = []
    last_event_by_para: dict[tuple[str, int], tuple[int, str]] = {}
    effective_grant = (
        grant
        if grant is not None
        else (
            getattr(workspace, "_system_grant", None) or getattr(target_ws, "_system_grant", None)
        )
    )

    def _qnode(nd: Any) -> None:
        node_ops.append(
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=nd.id, node_data=nd.model_dump()
            )
        )

    def _queue_edge(eid: str, edge: HCIREdge, *, force_upsert: bool = False) -> None:
        if eid in seen_edge_ids:
            return
        seen_edge_ids.add(eid)
        op_type = (
            TransactionOp.UPSERT_EDGE
            if force_upsert or eid in existing_edges
            else TransactionOp.ADD_EDGE
        )
        edge_ops.append(
            TransactionOperation(
                op=op_type,
                edge_id=eid,
                edge_data=edge.model_dump(),
            )
        )

    edge_type_map = {
        "is_a": HCIREdgeType.GENERALIZES,
        "causes": HCIREdgeType.CAUSES,
        "requires": HCIREdgeType.REQUIRES,
        "enables": HCIREdgeType.CAUSES,
        "has_property": HCIREdgeType.SUPPORTS,
    }

    for prop in propositions:
        src_id = f"t4_concept_{prop.subject}"
        dst_id = f"t4_concept_{prop.obj}"
        for term, cid in ((prop.subject, src_id), (prop.obj, dst_id)):
            if cid not in created_concepts:
                created_concepts.add(cid)
                display = term.replace("_", " ")
                defn = ""
                if term == prop.subject and prop.relation == "is_a" and prop.polarity:
                    from hbllm.hcir.language.english.realizer import MASS_NOUNS

                    clean_obj = prop.obj.replace("_", " ").strip()
                    lower_obj = clean_obj.lower()
                    if (
                        lower_obj in MASS_NOUNS
                        or (clean_obj and clean_obj[0].isupper())
                        or lower_obj.startswith(("a ", "an ", "the "))
                    ):
                        art_pfx = ""
                    elif lower_obj and lower_obj[0] in "aeiou":
                        art_pfx = "an "
                    else:
                        art_pfx = "a "
                    defn = f"{art_pfx}{clean_obj} ({prop.sentence[:120]})"

                existing_node = ws_graph.get_node(cid) if ws_graph is not None else None
                existing_def = (
                    getattr(existing_node, "definition", None)
                    if existing_node is not None
                    else None
                )
                if existing_node is not None and not defn and existing_def:
                    continue
                _qnode(
                    ConceptNode(
                        id=cid,
                        label=display,
                        definition=defn,
                        domain="t4_reading_curriculum",
                        tags=["t4_reading", display],
                    )
                )

        mod_dict: dict[str, Any] = {}
        adjs: list[str] = []
        for k, v in prop.modifiers:
            if k == "adj":
                adjs.append(v)
            else:
                mod_dict[k] = v
        if adjs:
            mod_dict["adj"] = adjs

        belief = BeliefNode(
            id=f"t4_belief_{prop.subject}_{prop.relation}_{prop.obj}",
            statement=f"{prop.subject.replace('_', ' ')} {prop.relation} {prop.obj.replace('_', ' ')}",
            confidence=0.88 if not prop.polarity else 0.82,
            tags=["t4_reading_curriculum", prop.relation, prop.source_file],
            properties={
                "source_file": prop.source_file,
                "sentence": prop.sentence[:160],
                "polarity": prop.polarity,
                "modifiers": mod_dict,
            },
        )
        _qnode(belief)

        def _make_edge(
            eid: str, etype: HCIREdgeType, src: str, dst: str, w: float, props_d: dict[str, Any]
        ) -> HCIREdge:
            return HCIREdge(
                id=eid, edge_type=etype, sources=[src], targets=[dst], weight=w, properties=props_d
            )

        is_neg = (not prop.polarity) or prop.relation.startswith("not_")
        base_props = {
            "source_file": prop.source_file,
            "relation": prop.relation,
            "polarity": not is_neg,
            "modifiers": mod_dict,
            "sentence": prop.sentence[:160],
        }
        if is_neg:
            pos_rel = prop.relation[4:] if prop.relation.startswith("not_") else prop.relation
            pos_eid = f"t4_{pos_rel}_{src_id}__{dst_id}"
            if pos_eid in existing_edges and pos_eid not in seen_edge_ids:
                seen_edge_ids.add(pos_eid)
                edge_ops.append(TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id=pos_eid))
            _queue_edge(
                f"t4_contradicts_{src_id}__{dst_id}__{prop.relation}",
                _make_edge(
                    f"t4_contradicts_{src_id}__{dst_id}__{prop.relation}",
                    HCIREdgeType.CONTRADICTS,
                    src_id,
                    dst_id,
                    0.95,
                    base_props,
                ),
                force_upsert=True,
            )
        else:
            edge_id = f"t4_{prop.relation}_{src_id}__{dst_id}"
            if edge_id not in existing_edges or mod_dict:
                _queue_edge(
                    edge_id,
                    _make_edge(
                        edge_id,
                        edge_type_map.get(prop.relation, HCIREdgeType.SUPPORTS),
                        src_id,
                        dst_id,
                        0.82,
                        base_props,
                    ),
                )

        if (recipient_term := str(mod_dict.get("recipient", "")).strip()) and not is_neg:
            rec_id = f"t4_concept_{recipient_term}"
            if rec_id not in created_concepts:
                created_concepts.add(rec_id)
                rec_display = recipient_term.replace("_", " ")
                if ws_graph is None or ws_graph.get_node(rec_id) is None:
                    _qnode(
                        ConceptNode(
                            id=rec_id,
                            label=rec_display,
                            definition="",
                            domain="t4_reading_curriculum",
                            tags=["t4_reading", rec_display],
                        )
                    )
            rec_eid = f"t4_{prop.relation}_recipient_{src_id}__{rec_id}"
            _queue_edge(
                rec_eid,
                _make_edge(
                    rec_eid,
                    HCIREdgeType.SUPPORTS,
                    src_id,
                    rec_id,
                    0.82,
                    {
                        "source_file": prop.source_file,
                        "relation": f"{prop.relation}_recipient",
                        "polarity": True,
                        "sentence": prop.sentence[:160],
                    },
                ),
            )

        if (
            link_temporal_events
            and prop.polarity
            and not prop.relation.startswith(("not_", "located_"))
            and prop.relation not in _STATIC_RELATIONS
        ):
            para_key = (prop.source_file, prop.paragraph_idx)
            if (
                (prev_ev := last_event_by_para.get(para_key)) is not None
                and prop.sentence_idx > prev_ev[0]
                and prev_ev[1] != belief.id
            ):
                beid = f"t4_before_{prev_ev[1]}__{belief.id}"
                _queue_edge(
                    beid,
                    _make_edge(
                        beid,
                        HCIREdgeType.BEFORE,
                        prev_ev[1],
                        belief.id,
                        0.78,
                        {
                            "relation": "before",
                            "source_file": prop.source_file,
                            "paragraph_idx": prop.paragraph_idx,
                        },
                    ),
                )
            last_event_by_para[para_key] = (prop.sentence_idx, belief.id)

        if prop.relation == "is_a" and prop.polarity:
            store_records.append(
                LexicalOntologyRecord(
                    lemma=prop.subject.replace("_", " "),
                    pos="noun",
                    language="en",
                    definition=prop.sentence[:200],
                    hypernyms=(prop.obj.replace("_", " "),),
                    domain="t4_reading",
                )
            )

    store_hypernyms_added = 0
    if ontology_store is not None and store_records:
        _, store_hypernyms_added = ontology_store.upsert_many(
            store_records, source=author, preserve_existing=True
        )

    def _commit_batches(ops_list: list[TransactionOperation]) -> int:
        cnt = 0
        for i in range(0, len(ops_list), batch_size):
            batch = ops_list[i : i + batch_size]
            if getattr(
                target_ws.commit_operations(batch, author=author, grant=effective_grant),
                "is_committed",
                False,
            ):
                cnt += len(batch)
        return cnt

    return _commit_batches(node_ops), _commit_batches(edge_ops), store_hypernyms_added


@dataclass(frozen=True)
class ProceduralDefinition:
    """Structured procedural/definitional instruction extracted from reading prose."""

    verb: str
    param_noun: str
    step_clauses: list[str]
    source_file: str
    sentence: str


def extract_procedural_definitions(
    text: str, source_file: str = "reading"
) -> list[ProceduralDefinition]:
    """Extract definitional/procedural rules (e.g. 'To X a Y means to A and then B')."""
    rules = load_constructions_spec("en").get("lexical_rules", {})
    patterns = [re.compile(str(p), re.IGNORECASE) for p in rules.get("definitional_patterns", ())]
    if not patterns:
        return []
    out: list[ProceduralDefinition] = []
    seen: set[str] = set()
    for sent in _SENT_SPLIT_RE.split(text):
        clean = sent.strip().rstrip(".!?")
        if not clean:
            continue
        for pat in patterns:
            if (m := pat.match(clean)) is None:
                continue
            verb = _clean_term(m.group(1))
            param_noun = _clean_term(m.group(2))
            rhs = m.group(3).strip()
            if not verb or verb in seen or not rhs:
                break
            raw_steps = re.split(
                r"\s*,?\s*\b(?:and\s+then|then|and)\b\s*,?\s*", rhs, flags=re.IGNORECASE
            )
            steps = [s.strip() for s in raw_steps if s.strip()]
            if steps:
                seen.add(verb)
                out.append(ProceduralDefinition(verb, param_noun, steps, source_file, clean))
            break
    return out


def _ins(op: Opcode, cost: int, **params: Any) -> Instruction:
    return Instruction(opcode=op, params=params, cost_estimate=cost)


def compile_procedural_definitions_to_ops(
    defs: Sequence[ProceduralDefinition],
    existing_graph: Any | None = None,
    *,
    tenant_id: str | None = "default",
    author: str = "t4_reading_curriculum",
) -> list[TransactionOperation]:
    """Compile `ProceduralDefinition`s into homoiconic `SkillNode` + `ConstructionNode` ops."""
    articles = list(load_constructions_spec("en").get("lexical_rules", {}).get("articles", ()))
    known_skills: dict[str, str] = {}
    if existing_graph is not None:
        for n in existing_graph.nodes_by_type("skill"):
            if isinstance(n, SkillNode) and n.skill_name:
                known_skills[n.skill_name.lower()] = n.id
    for d in defs:
        known_skills.setdefault(d.verb.lower(), f"t4_skill_{d.verb.lower()}")

    ops: list[TransactionOperation] = []
    prov, scope = Provenance(created_by=author), Scope(tenant_id=tenant_id or "default")
    for d in defs:
        v = d.verb.lower()
        skill_id = known_skills[v]
        ins_list: list[Instruction] = [_ins(Opcode.UNIFY, 5, pattern={"id": "?target"}, bind="$r0")]
        for clause in d.step_clauses:
            parts = clause.strip().lower().split(None, 1)
            sv = _clean_term(parts[0])
            sobj = parts[1].strip() if len(parts) > 1 else ""
            for art in articles:
                if sobj.startswith(art):
                    sobj = sobj[len(art) :].strip()
                    break
            tsk = known_skills.get(sv)
            if tsk and tsk != skill_id:
                ins_list.append(
                    _ins(
                        Opcode.CALL,
                        15,
                        skill_id=tsk,
                        args={"target": "$r0", "subject": sobj or d.param_noun},
                        bind="$r1",
                    )
                )
            else:
                ins_list.append(
                    _ins(
                        Opcode.EXECUTE,
                        25,
                        capability=f"motor.{sv}",
                        params={"action": sv, "target": "$r0", "subject": sobj},
                        bind="$r1",
                    )
                )
        ins_list.append(_ins(Opcode.RETURN, 0, **{"yield": ["$r1"]}))
        stream_dump = InstructionStream(
            author=author, description=d.sentence, instructions=ins_list
        ).model_dump()
        tpl: list[dict[str, Any]] = [
            {
                "opcode": "UNIFY",
                "cost_estimate": 5,
                "params": {"pattern": {"id": "{target_ref}"}, "bind": "$r0"},
            },
            {
                "opcode": "CALL",
                "cost_estimate": 15,
                "params": {
                    "skill_id": skill_id,
                    "args": {"target": "$r0", "target_ref": "{target_ref}", "subject": "{subject}"},
                    "bind": "$r1",
                },
            },
            {"opcode": "RETURN", "params": {"yield": ["$r1"]}},
        ]
        nodes = (
            SkillNode(
                id=skill_id,
                skill_name=v,
                description=d.sentence,
                parameters=[
                    {"name": "target", "role": d.param_noun, "required": False, "default": None}
                ],
                precondition_pattern={"intent": v, "param": d.param_noun},
                postcondition_pattern={"intent": v, "target": "?target"},
                bytecode_stream=stream_dump,
                alpha_success=3.0,
                beta_failure=1.0,
                lifecycle=NodeLifecycle.ACTIVE,
                provenance=prov,
                scope=scope,
                tags=["t4_skill", v],
            ),
            ConstructionNode(
                id=f"t4_cxn_en_{v}",
                construction_name=f"learned_{v}",
                language="en",
                intent_type="act",
                syntactic_template="{action} {target}",
                trigger_tokens=[v],
                slot_constraints={"target": d.param_noun},
                semantic_mapping={
                    "default_action": v,
                    "description_template": f"Execute {v} on {{target_ref}}",
                },
                compiled_hcir_template=tpl,
                bytecode_stream=stream_dump,
                alpha_success=3.0,
                beta_failure=1.0,
                lifecycle=NodeLifecycle.ACTIVE,
                provenance=prov,
                scope=scope,
                tags=["t4_construction", v],
            ),
        )
        ops.extend(
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=nd.id, node_data=nd.model_dump()
            )
            for nd in nodes
        )
    return ops


def commit_procedural_definitions(
    raw_text: str,
    workspace: Any,
    *,
    grant: Any | None = None,
    author: str = "t4_reading_curriculum",
    source_label: str = "reading",
    tenant_id: str | None = None,
) -> int:
    """Extract and transactionally commit procedural definitions as `SkillNode` + `ConstructionNode`."""
    from hbllm.hcir.workspace_tiers import TieredWorkspace

    defs = extract_procedural_definitions(raw_text, source_file=source_label)
    if not defs:
        return 0
    is_tw = isinstance(workspace, TieredWorkspace)
    target_ws = workspace.persistent if is_tw else workspace
    tid = tenant_id or getattr(target_ws, "tenant_id", None) or "default"
    ops = compile_procedural_definitions_to_ops(
        defs, existing_graph=target_ws.graph, tenant_id=tid, author=author
    )
    if not ops:
        return 0
    effective_grant = grant if grant is not None else (workspace._system_grant if is_tw else None)
    res = target_ws.commit_operations(ops, author=author, grant=effective_grant, tenant_id=tid)
    return len(ops) if res.is_committed else 0


def _record_prop_tally(prop: ExtractedProposition, stats: ReadingCurriculumStats) -> None:
    if not prop.polarity or prop.relation.startswith("not_"):
        stats.relational_extracted += 1
    elif prop.relation == "is_a":
        stats.taxonomic_extracted += 1
    elif prop.relation in ("causes", "enables"):
        stats.causal_extracted += 1
    elif prop.relation == "requires":
        stats.dependency_extracted += 1
    else:
        stats.relational_extracted += 1


def _finalize_sample_props(
    props: Sequence[ExtractedProposition], stats: ReadingCurriculumStats
) -> None:
    for prop in props[:10]:
        stats.sample_propositions.append(
            {
                "subject": prop.subject,
                "relation": prop.relation,
                "object": prop.obj,
                "source": prop.source_file,
            }
        )


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
        raw_texts: list[tuple[str, str]] = []

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
                raw_texts.append((path.name, text))
                for prop in extract_propositions_from_text(
                    text,
                    source_file=path.name,
                    include_extended=True,
                    ontology_store=ontology_store,
                ):
                    key = (prop.subject, prop.relation, prop.obj)
                    if key in seen_global:
                        continue
                    seen_global.add(key)
                    propositions.append(prop)
                    _record_prop_tally(prop, stats)
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
        proc_nodes = sum(
            commit_procedural_definitions(
                txt, workspace, grant=grant, author="t4_reading_curriculum", source_label=fname
            )
            for fname, txt in raw_texts
        )
        stats.nodes_committed = nodes_committed + proc_nodes
        stats.edges_committed, stats.store_hypernyms_added = edges_committed, store_hypernyms_added
        _finalize_sample_props(propositions, stats)
        stats.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return stats

    def ingest_book(
        self,
        text_or_path: Path | str,
        workspace: Any,
        *,
        ontology_store: Any | None = None,
        grant: Any | None = None,
        max_propositions: int = 500,
        batch_size: int = 100,
        source_name: str = "",
    ) -> ReadingCurriculumStats:
        """Read an entire book/text, consolidate by entity-salience, and commit chronologically."""
        t0 = time.perf_counter()
        stats = ReadingCurriculumStats()

        if isinstance(text_or_path, Path) or (
            isinstance(text_or_path, str)
            and "\n" not in text_or_path
            and len(text_or_path) < 512
            and Path(text_or_path).is_file()
        ):
            book_file = Path(text_or_path)
            raw_text, label, stats.files_read = (
                book_file.read_text(encoding="utf-8", errors="ignore"),
                source_name or book_file.name,
                1,
            )
        else:
            raw_text, label = str(text_or_path), source_name or "book_text"
            stats.files_read = 1 if raw_text.strip() else 0

        stats.sentences_scanned = len(_SENT_SPLIT_RE.split(raw_text))
        all_props: list[ExtractedProposition] = []
        seen_global: set[tuple[str, str, str]] = set()
        for prop in extract_propositions_from_text(
            raw_text, source_file=label, include_extended=True, ontology_store=ontology_store
        ):
            key = (prop.subject, prop.relation, prop.obj)
            if key not in seen_global:
                seen_global.add(key)
                all_props.append(prop)

        selected = all_props
        if len(all_props) > max_propositions:
            deg: dict[str, int] = {}
            for p_it in all_props:
                deg[p_it.subject] = deg.get(p_it.subject, 0) + 1
                deg[p_it.obj] = deg.get(p_it.obj, 0) + 1

            def _score(idx_prop: tuple[int, ExtractedProposition]) -> tuple[float, int]:
                idx, pel = idx_prop
                rel_w = (
                    3.0
                    if pel.relation in ("causes", "requires", "enables")
                    else 2.5
                    if pel.relation == "is_a"
                    else 1.5
                    if pel.modifiers
                    else 1.0
                )
                return (float(deg.get(pel.subject, 0) + deg.get(pel.obj, 0)) + rel_w, -idx)

            ranked = sorted(enumerate(all_props), key=_score, reverse=True)[:max_propositions]
            ranked.sort(key=lambda item: item[0])
            selected = [pel for _, pel in ranked]

        for prop in selected:
            _record_prop_tally(prop, stats)

        nodes_committed, edges_committed, store_hypernyms_added = commit_propositions(
            selected,
            workspace,
            ontology_store=ontology_store,
            grant=grant,
            author="t4_reading_curriculum",
            batch_size=batch_size,
            link_temporal_events=True,
        )
        proc_nodes = commit_procedural_definitions(
            raw_text, workspace, grant=grant, author="t4_reading_curriculum", source_label=label
        )
        stats.nodes_committed = nodes_committed + proc_nodes
        stats.edges_committed, stats.store_hypernyms_added = edges_committed, store_hypernyms_added
        _finalize_sample_props(selected, stats)
        stats.elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return stats

    @staticmethod
    def _normalize_triple(subj: str, rel: str, obj: str) -> tuple[str, str, str]:
        return (
            re.sub(r"\s+", "_", subj.strip().lower()),
            re.sub(r"\s+", "_", rel.strip().lower()),
            re.sub(r"\s+", "_", obj.strip().lower()),
        )

    def evaluate_extraction_recall(
        self, annotated_samples: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute precision, recall, and F1 against annotated ground-truth triples."""
        tot_exp = tot_ext = tot_match = 0
        per_sample: list[dict[str, Any]] = []
        for idx, sample in enumerate(annotated_samples):
            sid = str(sample.get("id", f"sample_{idx}"))
            expected = {
                self._normalize_triple(
                    str(it["subject"]) if isinstance(it, dict) else str(it[0]),
                    str(it["relation"]) if isinstance(it, dict) else str(it[1]),
                    str(it.get("object", it.get("obj", "")))
                    if isinstance(it, dict)
                    else str(it[2]),
                )
                for it in (sample.get("expected") or ())
            }
            extracted = {
                self._normalize_triple(p.subject, p.relation, p.obj)
                for p in extract_propositions_from_text(
                    str(sample.get("text", "")), source_file=sid
                )
            }
            n_exp, n_ext, n_m = len(expected), len(extracted), len(expected & extracted)
            tot_exp, tot_ext, tot_match = tot_exp + n_exp, tot_ext + n_ext, tot_match + n_m
            per_sample.append(
                {
                    "id": sid,
                    "expected": n_exp,
                    "extracted": n_ext,
                    "matched": n_m,
                    "precision": round(n_m / max(1, n_ext), 4),
                    "recall": round(n_m / max(1, n_exp), 4),
                    "missed": sorted(expected - extracted),
                }
            )
        prec, rec = tot_match / max(1, tot_ext), tot_match / max(1, tot_exp)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "matched": tot_match,
            "expected_count": tot_exp,
            "extracted_count": tot_ext,
            "per_sample": per_sample,
        }
