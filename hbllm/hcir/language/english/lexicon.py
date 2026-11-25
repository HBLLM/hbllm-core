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
    CONJ = "conj"
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
            if not stored and key.isalpha() and hasattr(self._ontology_store, "lemmatize_surface"):
                for lemma, _ in self._ontology_store.lemmatize_surface(key, language="en"):
                    local_lemma = self._entries.get(lemma)
                    if local_lemma:
                        return local_lemma
                    stored = self._ontology_store.lookup(lemma, language="en")
                    if stored:
                        break
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
            "their",
            "its",
            "his",
            "her",
            "my",
            "our",
            "your",
            "all",
            "both",
            "many",
            "most",
            "several",
            "few",
        ]:
            self.add_entry(EnglishLexicalEntry(lemma=det, pos=EnglishPOS.DET))

        # Adverbs (markers like please/not & universal frequency/manner adverbs)
        for adv in [
            "please",
            "kindly",
            "not",
            "never",
            "always",
            "often",
            "sometimes",
            "usually",
            "rarely",
            "very",
            "quite",
            "too",
            "also",
            "still",
            "already",
            "quickly",
            "slowly",
            "heavily",
            "quietly",
            "loudly",
            "well",
            "far",
            "away",
            "back",
            "here",
            "there",
        ]:
            self.add_entry(EnglishLexicalEntry(lemma=adv, pos=EnglishPOS.ADV))

        # Pronouns & Wh-words
        for pron in [
            "it",
            "them",
            "me",
            "us",
            "you",
            "i",
            "he",
            "she",
            "they",
            "we",
            "him",
            "someone",
            "everyone",
            "nobody",
            "something",
            "nothing",
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

        # Conjunctions (prevent 'and'/'or'/'but' from being swallowed as NOUNs inside NPs)
        for conj in ["and", "or", "but", "yet", "nor"]:
            self.add_entry(EnglishLexicalEntry(lemma=conj, pos=EnglishPOS.CONJ))

        # Auxiliary verbs & modals
        for aux in [
            "is",
            "are",
            "was",
            "were",
            "am",
            "did",
            "does",
            "do",
            "can",
            "could",
            "would",
            "will",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "has",
            "have",
            "had",
            "been",
            "being",
        ]:
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
            "through": "through",
            "across": "across",
            "toward": "toward",
            "towards": "toward",
            "around": "around",
            "over": "above",
            "between": "between",
            "among": "among",
            "during": "during",
            "without": "without",
            "against": "against",
            "along": "along",
            "behind": "behind",
            "beyond": "beyond",
            "before": "before",
            "after": "after",
            "until": "until",
            "since": "since",
            "at": "located_at",
            "as": "as",
            "out": "out",
            "off": "off",
            "up": "up",
            "down": "down",
            "like": "like",
        }
        for prep, pred in preps.items():
            self.add_entry(
                EnglishLexicalEntry(lemma=prep, pos=EnglishPOS.PREP, semantic_predicate=pred)
            )

        # Nouns (Physical entities & universal concepts)
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
            "plant": "plant",
            "leaf": "leaf",
            "tree": "tree",
            "river": "river",
            "rain": "rain",
            "flood": "flood",
            "home": "home",
            "cave": "cave",
            "moss": "moss",
            "meat": "meat",
            "creature": "creature",
            "bird": "bird",
            "beetle": "beetle",
            "night": "night",
            "sunlight": "sunlight",
            "water": "water",
            "glucose": "glucose",
            "oxygen": "oxygen",
            "photosynthesis": "photosynthesis",
            "gravity": "gravity",
            "force": "force",
            "object": "object",
            "mass": "mass",
            "orbit": "orbit",
            "light": "light",
            "energy": "energy",
            "air": "air",
            "soil": "soil",
            "root": "root",
            "seed": "seed",
            "berry": "berry",
            "fish": "fish",
            "reptile": "reptile",
            "insect": "insect",
            "forest": "forest",
            "ocean": "ocean",
            "lake": "lake",
            "mountain": "mountain",
            "earth": "earth",
            "sun": "sun",
            "moon": "moon",
            "star": "star",
            "time": "time",
            "day": "day",
            "body": "body",
            "food": "food",
            "blood": "blood",
            "temperature": "temperature",
            "cell": "cell",
            "system": "system",
            "process": "process",
            "atom": "atom",
            "matter": "matter",
            "rust": "rust",
        }
        for noun, pred in nouns.items():
            self.add_entry(
                EnglishLexicalEntry(lemma=noun, pos=EnglishPOS.NOUN, semantic_predicate=pred)
            )

        # Adjectives (Properties & universal descriptive adjectives)
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

        universal_adjs = (
            "big",
            "little",
            "tall",
            "short",
            "long",
            "wide",
            "narrow",
            "deep",
            "shallow",
            "thick",
            "thin",
            "heavy",
            "light",
            "fast",
            "slow",
            "quick",
            "hot",
            "cold",
            "warm",
            "cool",
            "wet",
            "dry",
            "bright",
            "dark",
            "loud",
            "quiet",
            "hard",
            "soft",
            "sharp",
            "dull",
            "smooth",
            "rough",
            "clean",
            "dirty",
            "strong",
            "weak",
            "young",
            "old",
            "new",
            "ancient",
            "fresh",
            "sweet",
            "sour",
            "bitter",
            "salty",
            "happy",
            "sad",
            "angry",
            "tired",
            "hungry",
            "thirsty",
            "afraid",
            "brave",
            "calm",
            "busy",
            "ready",
            "sick",
            "healthy",
            "alive",
            "dead",
            "wild",
            "tame",
            "rare",
            "common",
            "aquatic",
            "nocturnal",
            "diurnal",
            "terrestrial",
            "carnivorous",
            "herbivorous",
            "omnivorous",
            "venomous",
            "poisonous",
            "edible",
            "soluble",
            "magnetic",
            "electric",
            "metallic",
            "organic",
            "transparent",
            "opaque",
            "solid",
            "liquid",
            "gaseous",
            "tiny",
            "huge",
            "giant",
            "pale",
            "golden",
            "silver",
            "black",
            "white",
            "gray",
            "grey",
            "brown",
            "orange",
            "purple",
            "pink",
            "proud",
            "kind",
            "clever",
            "wise",
            "lonely",
            "curious",
            "safe",
            "dangerous",
            "empty",
            "full",
            "broken",
            "frozen",
            "flooded",
            "warm-blooded",
            "cold-blooded",
        )
        for adj in universal_adjs:
            self.add_entry(
                EnglishLexicalEntry(lemma=adj, pos=EnglishPOS.ADJ, properties={"property": adj})
            )

        # Verbs (Actions, processes & events)
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

        universal_verbs = (
            "eat",
            "live",
            "sleep",
            "drink",
            "walk",
            "run",
            "fly",
            "swim",
            "climb",
            "hunt",
            "feed",
            "nest",
            "migrate",
            "travel",
            "flow",
            "flood",
            "rain",
            "grow",
            "bloom",
            "die",
            "breathe",
            "digest",
            "absorb",
            "release",
            "emit",
            "convert",
            "transform",
            "attract",
            "repel",
            "reflect",
            "conduct",
            "dissolve",
            "evaporate",
            "freeze",
            "melt",
            "boil",
            "burn",
            "glow",
            "shine",
            "contain",
            "consist",
            "form",
            "generate",
            "create",
            "make",
            "build",
            "carry",
            "hold",
            "keep",
            "store",
            "transport",
            "transmit",
            "protect",
            "defend",
            "prevent",
            "reduce",
            "increase",
            "regulate",
            "connect",
            "separate",
            "combine",
            "divide",
            "break",
            "repair",
            "enter",
            "exit",
            "cross",
            "return",
            "arrive",
            "depart",
            "hide",
            "seek",
            "chase",
            "catch",
            "escape",
            "rest",
            "wake",
            "work",
            "play",
            "speak",
            "talk",
            "say",
            "tell",
            "ask",
            "answer",
            "think",
            "know",
            "learn",
            "teach",
            "understand",
            "believe",
            "feel",
            "see",
            "hear",
            "watch",
            "look",
            "find",
            "lose",
            "take",
            "bring",
            "send",
            "get",
            "become",
            "remain",
            "stay",
            "seem",
            "appear",
            "happen",
            "occur",
            "exist",
            "use",
            "help",
            "allow",
            "permit",
            "depend",
            "lead",
            "follow",
            "start",
            "stop",
            "begin",
            "end",
            "finish",
            "continue",
            "change",
            "turn",
            "sit",
            "stand",
            "lie",
            "rise",
            "sink",
            "pull",
            "lift",
            "throw",
            "hit",
            "cut",
            "wear",
            "buy",
            "sell",
            "pay",
            "write",
            "draw",
            "sing",
            "win",
            "try",
            "wait",
            "serve",
            "kill",
            "save",
            "plant",
            "heat",
            "water",
            "light",
            "focus",
        )
        for verb in universal_verbs:
            self.add_entry(
                EnglishLexicalEntry(lemma=verb, pos=EnglishPOS.VERB, semantic_predicate=verb)
            )
