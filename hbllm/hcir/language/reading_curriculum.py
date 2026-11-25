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

from hbllm.hcir.graph import HCIREdge
from hbllm.hcir.language.english.morphology import (
    INVARIANT_S_NOUNS,
    singularize_noun,
)
from hbllm.hcir.language.ontology_bootstrap import LexicalOntologyRecord
from hbllm.hcir.nodes import BeliefNode, ConceptNode, HCIREdgeType
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

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
    {
        "the",
        "a",
        "an",
        "if",
        "when",
        "must",
        "determines",
        "requiring",
        "every",
        "only",
        "no",
        "is",
        "are",
        "deep",
        "neural",
        "level",
        "high",
        "type",
        "kind",
        "form",
    }
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
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip().lower()).strip("_-")


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


def _get_parser(parser: Any | None) -> Any:
    global _CORE_PARSER_SINGLETON
    if parser is not None:
        return parser
    if _CORE_PARSER_SINGLETON is None:
        from hbllm.hcir.language.english.parser import EnglishParser

        _CORE_PARSER_SINGLETON = EnglishParser()
    return _CORE_PARSER_SINGLETON


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
    parser: Any | None = None,
) -> list[ExtractedProposition]:
    """Deterministically extract SVO, taxonomic, causal, and property propositions via EnglishParser.

    Extracts subject-verb-object triples with explicit polarity (``not_<rel>`` when
    negated) and modifiers (adjectival/prepositional) directly from syntactic parses.
    """
    from hbllm.hcir.language.core.semantic_frame import FrameType, ThematicRole

    found: list[ExtractedProposition] = []
    seen: set[tuple[str, str, str, bool]] = set()
    max_len = 600 if include_extended else 240
    active_parser = _get_parser(parser)

    def _add(
        subj_raw: str,
        rel: str,
        obj_raw: str,
        sentence: str,
        *,
        polarity: bool = True,
        modifiers: tuple[tuple[str, str], ...] = (),
    ) -> bool:
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
            )
        )
        return True

    def _extract_clause_via_parser(clause: str, orig_sent: str) -> bool:
        added_any = False
        frame = active_parser.parse(clause)
        if frame.is_error or frame.frame_type != FrameType.ASSERTION:
            return False

        polarity = bool(getattr(frame.metadata, "polarity", True))
        subj_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        subj = _extract_subject_label(subj_ref)
        if not subj:
            return False

        pat_ref = frame.get_role(ThematicRole.PATIENT)
        loc_ref = frame.get_role(ThematicRole.LOCATION)
        dest_ref = frame.get_role(ThematicRole.DESTINATION)
        inst_ref = frame.get_role(ThematicRole.INSTRUMENT)

        mod_list: list[tuple[str, str]] = []
        for role_ref in (loc_ref, dest_ref, inst_ref):
            if role_ref is not None and role_ref.concept_name and frame.predicate != "is_property":
                prep = str(role_ref.properties.get("prep", "")).strip().lower()
                tgt_mod = _clean_term(role_ref.concept_name)
                if prep and tgt_mod:
                    mod_list.append((prep, tgt_mod))

        if frame.predicate == "is_a" and pat_ref is not None:
            pat_raw = pat_ref.raw_text or pat_ref.concept_name or ""
            obj = _extract_head_from_np(pat_raw) or _extract_object_label(pat_ref)
            adj_mods = _extract_np_modifiers(pat_raw)
            if obj and obj not in _GARBAGE_OBJ:
                if _add(subj, "is_a", obj, orig_sent, polarity=polarity, modifiers=adj_mods):
                    added_any = True
            return added_any

        if frame.predicate == "is_property" and loc_ref is not None:
            prop_val = loc_ref.concept_name or (
                str(next(iter(loc_ref.properties.values()))) if loc_ref.properties else ""
            )
            prop_clean = _clean_term(prop_val)
            if prop_clean and prop_clean not in _GARBAGE_OBJ:
                if _add(subj, "has_property", prop_clean, orig_sent, polarity=polarity):
                    added_any = True
            return added_any

        if frame.predicate in ("causes", "requires", "enables") and pat_ref is not None:
            obj = _extract_object_label(pat_ref, preserve_head=True)
            inf_verb = str(pat_ref.properties.get("infinitive_verb", "")).strip().lower()
            if inf_verb and frame.predicate == "causes":
                # "Heavy rain caused the river to flood" -> causes river_flood & flood
                if obj:
                    added_any |= _add(
                        subj, "causes", f"{obj}_{inf_verb}", orig_sent, polarity=polarity
                    )
                added_any |= _add(subj, "causes", inf_verb, orig_sent, polarity=polarity)
                # Also record compound subject if subj_ref has modifier (e.g. heavy_rain)
                if subj_ref and subj_ref.raw_text:
                    raw_subj_clean = _clean_term(subj_ref.raw_text)
                    if raw_subj_clean and raw_subj_clean != subj:
                        if obj:
                            added_any |= _add(
                                raw_subj_clean,
                                "causes",
                                f"{obj}_{inf_verb}",
                                orig_sent,
                                polarity=polarity,
                            )
                        added_any |= _add(
                            raw_subj_clean, "causes", inf_verb, orig_sent, polarity=polarity
                        )
            elif obj:
                added_any |= _add(
                    subj,
                    frame.predicate,
                    obj,
                    orig_sent,
                    polarity=polarity,
                    modifiers=tuple(mod_list),
                )
            return added_any

        # Spatial predicates ("on", "in", "under", ...)
        if frame.predicate.startswith("located_") or frame.predicate in (
            "below",
            "above",
            "near",
        ):
            if loc_ref is not None:
                loc_obj = _extract_object_label(loc_ref)
                if loc_obj:
                    added_any |= _add(
                        subj,
                        frame.predicate,
                        loc_obj,
                        orig_sent,
                        polarity=polarity,
                        modifiers=tuple(mod_list),
                    )
            return added_any

        # General SVO verb (eat, live, absorb, release, walk, sleep, hunt, attract, convert, ...)
        verb_rel = _clean_term(frame.predicate or "")
        if not verb_rel or verb_rel in ("be", "do", "have"):
            return False

        if pat_ref is not None:
            pat_raw = pat_ref.raw_text or ""
            # Handle coordinated direct objects ("sunlight and water")
            if " and " in pat_raw.lower():
                obj_parts = [
                    p.strip() for p in re.split(r"\band\b", pat_raw, flags=re.I) if p.strip()
                ]
                for op in obj_parts:
                    op_head = _extract_head_from_np(op)
                    if op_head:
                        added_any |= _add(
                            subj,
                            verb_rel,
                            op_head,
                            orig_sent,
                            polarity=polarity,
                            modifiers=tuple(mod_list),
                        )
            else:
                obj = _extract_object_label(pat_ref, preserve_head=True)
                if obj and obj not in _GARBAGE_OBJ:
                    added_any |= _add(
                        subj,
                        verb_rel,
                        obj,
                        orig_sent,
                        polarity=polarity,
                        modifiers=tuple(mod_list),
                    )
            # "convert X into Y" also records production/result edge -> Y
            for prep, tgt in mod_list:
                if prep == "into" and verb_rel in ("convert", "turn", "transform"):
                    added_any |= _add(subj, "produce", tgt, orig_sent, polarity=polarity)
        elif loc_ref is not None or dest_ref is not None:
            obl = loc_ref or dest_ref
            obl_obj = _extract_object_label(obl)
            if obl_obj:
                added_any |= _add(
                    subj,
                    verb_rel,
                    obl_obj,
                    orig_sent,
                    polarity=polarity,
                    modifiers=tuple(mod_list),
                )

        return added_any

    last_subject_raw = ""
    last_plural_subject_raw = ""

    for raw_sent in _split_sentences_safe(text):
        raw_trimmed = raw_sent.strip()
        if not raw_trimmed:
            continue

        # 0. Reject hedged causal claims (keep explicit negation handling via parser)
        if _HEDGE_RE.search(raw_trimmed):
            continue

        # 1. Check structured glossary definition pattern first
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
                        _add(_singularize_basic(_clean_term(term)), "is_a", obj, raw_trimmed[:120])
            continue

        sent = raw_trimmed.lstrip("#>-*` ")
        if len(sent) < 10 or len(sent) > max_len:
            continue

        sent_clean = (
            _strip_intro_clause(sent) if include_extended else _INTRO_STRIP_RE.sub("", sent)
        )

        # Intra-paragraph pronoun subject coreference ("It hunts beetles", "She was tired")
        if include_extended and last_subject_raw:
            m_pron = re.match(r"^(it|he|she|they)\s+(.+)$", sent_clean, re.IGNORECASE)
            if m_pron:
                pron = m_pron.group(1).lower()
                antecedent = (
                    last_plural_subject_raw
                    if pron == "they" and last_plural_subject_raw
                    else last_subject_raw
                )
                sent_clean = f"{antecedent} {m_pron.group(2)}"

        # 2. Relative clause decomposition: "<Subj> is/are <NP> that/which/who <RelVP>"
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
                )
                last_subject_raw = subj_phrase
            _extract_clause_via_parser(f"{subj_phrase} {rel_vp}", sent)
            continue

        # 3. Coordinated VP splitting ("Photosynthesis converts X and releases Y")
        sub_clauses = (
            _split_coordinated_vps(sent_clean, active_parser) if include_extended else [sent_clean]
        )

        clause_matched = False
        for clause in sub_clauses:
            if _extract_clause_via_parser(clause, sent):
                clause_matched = True

        # Regex fallback for legacy nominal/causal/enables structures if parser couldn't resolve
        if not clause_matched and not _NEGATION_RE.search(sent_clean):
            m_isa = _IS_A_RE.match(sent_clean)
            if m_isa:
                subj = _singularize_basic(_clean_term(m_isa.group(1)))
                obj = _extract_head_from_np(m_isa.group(2))
                if subj and obj and subj != obj and obj not in _GARBAGE_OBJ:
                    if _add(
                        subj, "is_a", obj, sent, modifiers=_extract_np_modifiers(m_isa.group(2))
                    ):
                        clause_matched = True

            if not clause_matched:
                for rel, pattern in (
                    ("causes", _CAUSES_RE),
                    ("requires", _REQUIRES_RE),
                    ("enables", _ENABLES_RE),
                ):
                    m = pattern.match(sent_clean)
                    if m and _add(m.group(1), rel, m.group(2), sent):
                        clause_matched = True
                        break

        # Update intra-paragraph subject memory from parsed subject (never swallowing the verb)
        if clause_matched and found:
            parsed_last = active_parser.parse(sent_clean)
            s_ref = parsed_last.get_role(ThematicRole.THEME) or parsed_last.get_role(
                ThematicRole.AGENT
            )
            cand_subj = (
                (s_ref.raw_text or s_ref.concept_name or "").strip() if s_ref is not None else ""
            )
            if cand_subj:
                cand_words = [
                    w
                    for w in cand_subj.split()
                    if w.lower() not in {"a", "an", "the", "this", "that"}
                ]
                if cand_words and cand_words[0].lower() not in _SKIP_SUBJECTS:
                    last_subject_raw = cand_subj
                    if (
                        cand_words[-1].lower().endswith("s")
                        and cand_words[-1].lower() not in INVARIANT_S_NOUNS
                    ):
                        last_plural_subject_raw = cand_subj

        if not include_extended:
            continue

        # Extended patterns for long paragraphs / essays
        if not clause_matched:
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
    ws_graph = getattr(workspace, "graph", None)
    existing_edges = getattr(ws_graph, "edges", {})
    store_records: list[LexicalOntologyRecord] = []

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
        node_ops.append(
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=belief.id,
                node_data=belief.model_dump(),
            )
        )

        is_neg = (not prop.polarity) or prop.relation.startswith("not_")
        if is_neg:
            pos_rel = prop.relation[4:] if prop.relation.startswith("not_") else prop.relation
            pos_edge_id = f"t4_{pos_rel}_{src_id}__{dst_id}"
            if pos_edge_id in existing_edges:
                edge_ops.append(
                    TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id=pos_edge_id)
                )
            neg_edge_id = f"t4_contradicts_{src_id}__{dst_id}__{prop.relation}"
            edge = HCIREdge(
                id=neg_edge_id,
                edge_type=HCIREdgeType.CONTRADICTS,
                sources=[src_id],
                targets=[dst_id],
                weight=0.95,
                properties={
                    "source_file": prop.source_file,
                    "relation": prop.relation,
                    "polarity": False,
                    "modifiers": mod_dict,
                    "sentence": prop.sentence[:160],
                },
            )
            edge_ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_EDGE,
                    edge_id=edge.id,
                    edge_data=edge.model_dump(),
                )
            )
        else:
            edge_id = f"t4_{prop.relation}_{src_id}__{dst_id}"
            if edge_id not in existing_edges or mod_dict:
                edge = HCIREdge(
                    id=edge_id,
                    edge_type=edge_type_map.get(prop.relation, HCIREdgeType.SUPPORTS),
                    sources=[src_id],
                    targets=[dst_id],
                    weight=0.82,
                    properties={
                        "source_file": prop.source_file,
                        "relation": prop.relation,
                        "polarity": True,
                        "modifiers": mod_dict,
                        "sentence": prop.sentence[:160],
                    },
                )
                edge_ops.append(
                    TransactionOperation(
                        op=TransactionOp.UPSERT_EDGE
                        if edge_id in existing_edges
                        else TransactionOp.ADD_EDGE,
                        edge_id=edge.id,
                        edge_data=edge.model_dump(),
                    )
                )

        if prop.relation == "is_a" and prop.polarity:
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
        """Compute precision, recall, and F1 against annotated ground-truth triples.

        Each item in ``annotated_samples`` maps ``"text"`` to prose and ``"expected"``
        to a sequence of ``(subject, relation, object)`` tuples or dicts.
        """
        total_expected = 0
        total_extracted = 0
        total_matched = 0
        per_sample: list[dict[str, Any]] = []

        for idx, sample in enumerate(annotated_samples):
            text = str(sample.get("text", ""))
            raw_expected = sample.get("expected") or ()
            expected_set: set[tuple[str, str, str]] = set()
            for item in raw_expected:
                if isinstance(item, dict):
                    s = str(item.get("subject", ""))
                    r = str(item.get("relation", ""))
                    o = str(item.get("object", item.get("obj", "")))
                else:
                    s, r, o = str(item[0]), str(item[1]), str(item[2])
                expected_set.add(self._normalize_triple(s, r, o))

            extracted_props = extract_propositions_from_text(
                text, source_file=str(sample.get("id", f"sample_{idx}"))
            )
            extracted_set = {
                self._normalize_triple(p.subject, p.relation, p.obj) for p in extracted_props
            }

            matched_set = expected_set & extracted_set
            n_exp = len(expected_set)
            n_ext = len(extracted_set)
            n_match = len(matched_set)
            total_expected += n_exp
            total_extracted += n_ext
            total_matched += n_match

            s_prec = n_match / max(1, n_ext)
            s_rec = n_match / max(1, n_exp)
            per_sample.append(
                {
                    "id": sample.get("id", f"sample_{idx}"),
                    "expected": n_exp,
                    "extracted": n_ext,
                    "matched": n_match,
                    "precision": round(s_prec, 4),
                    "recall": round(s_rec, 4),
                    "missed": sorted(expected_set - extracted_set),
                }
            )

        precision = total_matched / max(1, total_extracted)
        recall = total_matched / max(1, total_expected)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "matched": total_matched,
            "expected_count": total_expected,
            "extracted_count": total_extracted,
            "per_sample": per_sample,
        }
