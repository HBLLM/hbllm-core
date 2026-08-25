"""English Lexicon for A16.

Defines Part of Speech tags, lexical entries, and base vocabulary for physical entities,
actions, spatial relations, and properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EnglishPOS(StrEnum):
    """English Part of Speech categories."""

    NOUN = "noun"
    VERB = "verb"
    ADJ = "adj"
    DET = "det"
    PREP = "prep"
    PRON = "pron"
    ADV = "adv"
    AUX = "aux"
    WH = "wh"
    PUNCT = "punct"


@dataclass(frozen=True)
class EnglishLexicalEntry:
    """A lexical entry in the English dictionary."""

    lemma: str
    pos: EnglishPOS
    semantic_predicate: str = ""  # Mapping to HCIR concept or relation
    properties: dict[str, Any] = field(default_factory=dict)
    irregular_forms: dict[str, str] = field(default_factory=dict)  # tense/number -> form


class EnglishLexicon:
    """Dictionary of English words with POS tags and semantic mappings.

    Extensible; comes pre-loaded with physical world concepts, spatial
    relations, and action predicates.
    """

    def __init__(self) -> None:
        self._entries: dict[str, list[EnglishLexicalEntry]] = {}
        self._init_vocabulary()

    def add_entry(self, entry: EnglishLexicalEntry) -> None:
        word = entry.lemma.lower()
        if word not in self._entries:
            self._entries[word] = []
        self._entries[word].append(entry)

    def lookup(self, word: str) -> list[EnglishLexicalEntry]:
        return self._entries.get(word.lower(), [])

    def has_word(self, word: str) -> bool:
        return word.lower() in self._entries

    def _init_vocabulary(self) -> None:
        # Determiners
        for det in ["the", "a", "an", "this", "that", "these", "those"]:
            self.add_entry(EnglishLexicalEntry(lemma=det, pos=EnglishPOS.DET))

        # Pronouns
        for pron in ["it", "them", "me", "you", "i", "what", "where"]:
            pos = EnglishPOS.WH if pron in ("what", "where") else EnglishPOS.PRON
            self.add_entry(EnglishLexicalEntry(lemma=pron, pos=pos))

        # Auxiliary verbs
        for aux in ["is", "are", "was", "were", "did", "does", "do", "can"]:
            self.add_entry(EnglishLexicalEntry(lemma=aux, pos=EnglishPOS.AUX))

        # Prepositions
        preps = {
            "on": "located_on",
            "in": "located_in",
            "under": "below",
            "above": "above",
            "near": "near",
            "from": "from",
            "to": "to",
            "into": "into",
        }
        for prep, pred in preps.items():
            self.add_entry(EnglishLexicalEntry(lemma=prep, pos=EnglishPOS.PREP, semantic_predicate=pred))

        # Nouns (Physical entities & concepts)
        nouns = {
            "ball": "ball",
            "box": "box",
            "table": "table",
            "cup": "cup",
            "block": "block",
            "robot": "robot",
            "hand": "hand",
            "shelf": "shelf",
            "chair": "chair",
            "floor": "floor",
            "color": "color",
            "location": "location",
        }
        for noun, pred in nouns.items():
            self.add_entry(EnglishLexicalEntry(lemma=noun, pos=EnglishPOS.NOUN, semantic_predicate=pred))

        # Adjectives (Properties)
        adjs = {
            "red": {"color": "red"},
            "blue": {"color": "blue"},
            "green": {"color": "green"},
            "yellow": {"color": "yellow"},
            "large": {"size": "large"},
            "small": {"size": "small"},
            "round": {"shape": "round"},
            "wooden": {"material": "wood"},
        }
        for adj, props in adjs.items():
            self.add_entry(EnglishLexicalEntry(lemma=adj, pos=EnglishPOS.ADJ, properties=props))

        # Verbs (Actions & Events)
        verbs = {
            "move": "move",
            "push": "push",
            "put": "put",
            "place": "place",
            "give": "give",
            "drop": "drop",
            "fall": "fall",
            "roll": "roll",
            "support": "supports",
        }
        for verb, pred in verbs.items():
            self.add_entry(EnglishLexicalEntry(lemma=verb, pos=EnglishPOS.VERB, semantic_predicate=pred))
