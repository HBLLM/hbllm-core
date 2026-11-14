"""Tamil Lexicon for A16 non-LLM multilingual cognition.

Defines Tamil vocabulary, POS tags, and language-neutral semantic mappings to HCIR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TamilPOS(StrEnum):
    """Tamil Part of Speech categories."""

    NOUN = "noun"
    VERB = "verb"
    ADJ = "adj"
    POSTP = "postp"  # Postpositions (e.g. மீது, உள்ளே, அருகில்)
    PRON = "pron"
    WH = "wh"
    PUNCT = "punct"


@dataclass(frozen=True)
class TamilLexicalEntry:
    """A lexical entry in the Tamil dictionary."""

    lemma: str
    pos: TamilPOS
    semantic_predicate: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


class TamilLexicon:
    """Dictionary of Tamil words with POS tags and semantic mappings."""

    def __init__(self) -> None:
        self._entries: dict[str, list[TamilLexicalEntry]] = {}
        self._init_vocabulary()

    def add_entry(self, entry: TamilLexicalEntry) -> None:
        word = entry.lemma.strip()
        if word not in self._entries:
            self._entries[word] = []
        self._entries[word].append(entry)

    def lookup(self, word: str) -> list[TamilLexicalEntry]:
        return self._entries.get(word.strip(), [])

    def _init_vocabulary(self) -> None:
        # Nouns (Physical objects & concepts)
        nouns = {
            "பந்து": "ball",
            "மேசை": "table",
            "பெட்டி": "box",
            "கோப்பை": "cup",
            "ரோபோ": "robot",
            "கதவு": "door",
            "முன் கதவு": "front_door",
            "பின் கதவு": "back_door",
            "வாயில்": "gate",
            "கை": "arm",
            "ரோபோ கை": "robot_arm",
        }
        for noun, pred in nouns.items():
            self.add_entry(
                TamilLexicalEntry(lemma=noun, pos=TamilPOS.NOUN, semantic_predicate=pred)
            )

        # Adjectives (Properties)
        adjs = {
            "சிவப்பு": {"color": "red"},
            "சிகப்பு": {"color": "red"},
            "நீலம்": {"color": "blue"},
            "பச்சை": {"color": "green"},
            "பெரிய": {"size": "large"},
            "சிறிய": {"size": "small"},
        }
        for adj, props in adjs.items():
            self.add_entry(TamilLexicalEntry(lemma=adj, pos=TamilPOS.ADJ, properties=props))

        # Postpositions (Spatial relations)
        postps = {
            "மீது": "located_on",
            "மேல்": "located_on",
            "உள்ளே": "located_in",
            "கீழே": "below",
            "அடியில்": "below",
            "அருகில்": "near",
        }
        for postp, pred in postps.items():
            self.add_entry(
                TamilLexicalEntry(lemma=postp, pos=TamilPOS.POSTP, semantic_predicate=pred)
            )

        # Verbs & Copula
        verbs = {
            "இருக்கிறது": "located_on",
            "உள்ளது": "located_on",
            "தள்ளு": "push",
            "தள்ளவும்": "push",
            "நகர்த்து": "move",
            "நகர்த்தவும்": "move",
            "வை": "put",
            "வைக்கவும்": "put",
            "திற": "open",
            "திறக்க": "open",
            "திறக்கவும்": "open",
            "பூட்டு": "lock",
            "பூட்டவும்": "lock",
            "மூடு": "close",
            "மூடவும்": "close",
            "சுழற்று": "rotate",
            "சுழற்றவும்": "rotate",
            "நிறுத்து": "stop",
            "நிறுத்தவும்": "stop",
            "நிலை": "status",
            "பரிசோதி": "inspect",
        }
        for verb, pred in verbs.items():
            self.add_entry(
                TamilLexicalEntry(lemma=verb, pos=TamilPOS.VERB, semantic_predicate=pred)
            )

        # Question words
        wh_words = {
            "எங்கே": "where",
            "என்ன": "what",
            "இருக்கிறதா": "is_it",
        }
        for wh, pred in wh_words.items():
            self.add_entry(TamilLexicalEntry(lemma=wh, pos=TamilPOS.WH, semantic_predicate=pred))
