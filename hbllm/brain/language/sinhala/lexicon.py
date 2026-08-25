"""Sinhala Lexicon for A16.

Defines Sinhala vocabulary, POS tags, and semantic mappings.
Validates that HCIR semantics are language-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SinhalaPOS(StrEnum):
    """Sinhala Part of Speech categories."""

    NOUN = "noun"
    VERB = "verb"
    ADJ = "adj"
    POSTP = "postp"  # Postpositions (e.g. මත, තුළ, යට)
    PRON = "pron"
    WH = "wh"
    PUNCT = "punct"


@dataclass(frozen=True)
class SinhalaLexicalEntry:
    """A lexical entry in the Sinhala dictionary."""

    lemma: str
    pos: SinhalaPOS
    semantic_predicate: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


class SinhalaLexicon:
    """Dictionary of Sinhala words with POS tags and semantic mappings."""

    def __init__(self) -> None:
        self._entries: dict[str, list[SinhalaLexicalEntry]] = {}
        self._init_vocabulary()

    def add_entry(self, entry: SinhalaLexicalEntry) -> None:
        word = entry.lemma.strip()
        if word not in self._entries:
            self._entries[word] = []
        self._entries[word].append(entry)

    def lookup(self, word: str) -> list[SinhalaLexicalEntry]:
        return self._entries.get(word.strip(), [])

    def _init_vocabulary(self) -> None:
        # Nouns (Physical objects & concepts)
        nouns = {
            "බෝලය": "ball",
            "බෝල": "ball",
            "මේසය": "table",
            "මේසේ": "table",
            "පෙට්ටිය": "box",
            "කොටුව": "box",
            "කෝප්පය": "cup",
            "රොබෝ": "robot",
        }
        for noun, pred in nouns.items():
            self.add_entry(SinhalaLexicalEntry(lemma=noun, pos=SinhalaPOS.NOUN, semantic_predicate=pred))

        # Adjectives (Properties)
        adjs = {
            "රතු": {"color": "red"},
            "නිල්": {"color": "blue"},
            "කොළ": {"color": "green"},
            "විශාල": {"size": "large"},
            "කුඩා": {"size": "small"},
        }
        for adj, props in adjs.items():
            self.add_entry(SinhalaLexicalEntry(lemma=adj, pos=SinhalaPOS.ADJ, properties=props))

        # Postpositions (Spatial relations)
        postps = {
            "මත": "located_on",
            "උඩ": "located_on",
            "තුළ": "located_in",
            "ඇතුලේ": "located_in",
            "යට": "below",
            "ළඟ": "near",
        }
        for postp, pred in postps.items():
            self.add_entry(SinhalaLexicalEntry(lemma=postp, pos=SinhalaPOS.POSTP, semantic_predicate=pred))

        # Verbs & Copula
        verbs = {
            "තියෙනවා": "located_on",   # exists / is located
            "තිබේ": "located_on",
            "තල්ලු කළා": "push",
            "ගෙනයන්න": "move",
            "දමන්න": "put",
        }
        for verb, pred in verbs.items():
            self.add_entry(SinhalaLexicalEntry(lemma=verb, pos=SinhalaPOS.VERB, semantic_predicate=pred))

        # Question words
        wh_words = {
            "කොහෙද": "where",
            "මොකක්ද": "what",
            "තියෙනවද": "is_it",
        }
        for wh, pred in wh_words.items():
            self.add_entry(SinhalaLexicalEntry(lemma=wh, pos=SinhalaPOS.WH, semantic_predicate=pred))
