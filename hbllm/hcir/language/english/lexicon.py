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

    def __init__(self, ontology_store: Any | None = None) -> None:
        self._entries: dict[str, list[EnglishLexicalEntry]] = {}
        self._ontology_store = ontology_store
        self._init_vocabulary()

    def attach_ontology_store(self, store: Any) -> None:
        """Attach an indexed ``LexicalOntologyStore`` for O(1) open-vocabulary fallback."""
        self._ontology_store = store

    def add_entry(self, entry: EnglishLexicalEntry) -> None:
        word = entry.lemma.lower()
        if word not in self._entries:
            self._entries[word] = []
        self._entries[word].append(entry)

    def lookup(self, word: str) -> list[EnglishLexicalEntry]:
        key = word.lower()
        local = self._entries.get(key)
        if local:
            return local
        if self._ontology_store is not None:
            pos_map = {
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
            stored = self._ontology_store.lookup(key, language="en")
            if stored:
                return [
                    EnglishLexicalEntry(
                        lemma=s.lemma,
                        pos=pos_map.get(s.pos, EnglishPOS.NOUN),
                        semantic_predicate=s.lemma,
                        properties={
                            "thematic_roles": list(s.thematic_roles),
                            "hypernyms": list(s.hypernyms),
                        },
                    )
                    for s in stored
                ]
        return []

    def has_word(self, word: str) -> bool:
        key = word.lower()
        if key in self._entries:
            return True
        if self._ontology_store is not None:
            return bool(self._ontology_store.has_word(key, language="en"))
        return False

    def is_trusted_word(self, word: str) -> bool:
        """Return True if *word* is in core lexicon or non-noisy ontology entries."""
        key = word.lower()
        if key in self._entries:
            return True
        if self._ontology_store is not None:
            if hasattr(self._ontology_store, "is_trusted_word"):
                return bool(self._ontology_store.is_trusted_word(key, language="en"))
            return bool(self._ontology_store.has_word(key, language="en"))
        return False

    def base_words(self) -> frozenset[str]:
        """Return all words explicitly registered in the built-in lexicon."""
        return frozenset(self._entries.keys())

    def _init_vocabulary(self) -> None:
        # Determiners
        for det in [
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "no",
            "every",
            "each",
            "some",
            "any",
        ]:
            self.add_entry(EnglishLexicalEntry(lemma=det, pos=EnglishPOS.DET))

        # Adverbs (markers like please/not)
        for adv in ["please", "kindly", "not"]:
            self.add_entry(EnglishLexicalEntry(lemma=adv, pos=EnglishPOS.ADV))

        # Pronouns & Wh-words
        for pron in [
            "it",
            "them",
            "me",
            "us",
            "you",
            "i",
            "what",
            "where",
            "who",
            "why",
            "how",
            "which",
        ]:
            pos = (
                EnglishPOS.WH
                if pron in ("what", "where", "who", "why", "how", "which")
                else EnglishPOS.PRON
            )
            self.add_entry(EnglishLexicalEntry(lemma=pron, pos=pos))

        # Auxiliary verbs
        for aux in ["is", "are", "was", "were", "did", "does", "do", "can", "could", "would"]:
            self.add_entry(EnglishLexicalEntry(lemma=aux, pos=EnglishPOS.AUX))

        # Prepositions
        preps = {
            "on": "located_on",
            "in": "located_in",
            "inside": "located_in",
            "within": "located_in",
            "under": "below",
            "beneath": "below",
            "above": "above",
            "near": "near",
            "beside": "near",
            "from": "from",
            "to": "to",
            "into": "into",
            "about": "about",
            "for": "for",
            "of": "of",
            "with": "with",
            "by": "by",
        }
        for prep, pred in preps.items():
            self.add_entry(
                EnglishLexicalEntry(lemma=prep, pos=EnglishPOS.PREP, semantic_predicate=pred)
            )

        # Nouns (Physical entities & concepts)
        nouns = {
            "ball": "ball",
            "box": "box",
            "table": "table",
            "cup": "cup",
            "block": "block",
            "robot": "robot",
            "arm": "arm",
            "door": "door",
            "gate": "gate",
            "safe": "safe",
            "lock": "lock",
            "hand": "hand",
            "shelf": "shelf",
            "chair": "chair",
            "floor": "floor",
            "color": "color",
            "location": "location",
            "focus": "focus",
            "stimulus": "stimulus",
            "spiking": "spiking",
            "microscope": "microscope",
            "telescope": "telescope",
            "algorithm": "algorithm",
            "computer": "computer",
            "dog": "dog",
            "cat": "cat",
            "mammal": "mammal",
            "animal": "animal",
            "compiler": "compiler",
            "program": "program",
            "overflow": "overflow",
            "crash": "crash",
            "friction": "friction",
            "heat": "heat",
            "encryption": "encryption",
            "key": "key",
            "man": "man",
            "woman": "woman",
            "person": "person",
            "whale": "whale",
        }
        for noun, pred in nouns.items():
            self.add_entry(
                EnglishLexicalEntry(lemma=noun, pos=EnglishPOS.NOUN, semantic_predicate=pred)
            )

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
            "open": "open",
            "unlock": "unlock",
            "rotate": "rotate",
            "close": "close",
            "require": "requires",
            "cause": "causes",
            "trigger": "triggers",
            "produce": "produces",
            "improve": "improves",
            "enable": "enables",
            "maintain": "maintains",
        }
        for verb, pred in verbs.items():
            self.add_entry(
                EnglishLexicalEntry(lemma=verb, pos=EnglishPOS.VERB, semantic_predicate=pred)
            )
