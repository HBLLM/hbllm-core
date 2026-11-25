"""Shared SQLite-backed lookup resolution for Sinhala and Tamil lexicons."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

TEntry = TypeVar("TEntry")


def resolve_multilingual_lexicon_lookup(
    word: str,
    *,
    language: str,
    local_entries: Mapping[str, list[TEntry]],
    ontology_store: Any | None,
    pos_map: Mapping[str, Any],
    default_pos: Any,
    entry_factory: Callable[[str, Any, str], TEntry],
) -> list[TEntry]:
    """Resolve native/romanized/inflected tokens via local entries + LexicalOntologyStore."""
    key = word.strip()
    if not key:
        return []
    local = local_entries.get(key)
    if local:
        return local
    if ontology_store is None:
        return []

    candidates = [key]
    if key.isascii() and hasattr(ontology_store, "transliterate_romanized"):
        candidates.extend(ontology_store.transliterate_romanized(key, language=language, limit=2))

    for cand in candidates:
        cand_local = local_entries.get(cand)
        if cand_local:
            return cand_local
        stored = ontology_store.lookup(cand, language=language)
        if not stored and hasattr(ontology_store, "lemmatize_surface"):
            for lemma, _ in ontology_store.lemmatize_surface(cand, language=language):
                stored = ontology_store.lookup(lemma, language=language)
                if stored:
                    break
        if stored:
            out: list[TEntry] = []
            for s in stored:
                aligned = (
                    ontology_store.align_to_english(s.lemma, language=language)
                    if hasattr(ontology_store, "align_to_english")
                    else None
                )
                out.append(
                    entry_factory(s.lemma, pos_map.get(s.pos, default_pos), aligned or s.lemma)
                )
            return out
        if hasattr(ontology_store, "align_to_english"):
            aligned = ontology_store.align_to_english(cand, language=language)
            if aligned:
                return [entry_factory(cand, default_pos, aligned)]
    return []
