"""English Morphology & Tokenizer for A16.

Provides deterministic rule-based inflection, lemmatization, and tokenization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hbllm.brain.language.english.lexicon import EnglishLexicon, EnglishPOS


@dataclass(frozen=True)
class MorphToken:
    """A single token with its surface form, lemma, POS, and morphological features."""

    surface: str
    lemma: str
    pos: EnglishPOS
    features: dict[str, str]  # e.g., {"number": "plural", "tense": "past"}


class EnglishMorphology:
    """Morphological analyzer and inflector for English."""

    def __init__(self, lexicon: EnglishLexicon | None = None) -> None:
        self._lexicon = lexicon or EnglishLexicon()
        self._irregular_verbs = {
            "is": ("be", "present", "3s"),
            "are": ("be", "present", "pl"),
            "was": ("be", "past", "3s"),
            "were": ("be", "past", "pl"),
            "fell": ("fall", "past", "any"),
            "fallen": ("fall", "past_participle", "any"),
            "dropped": ("drop", "past", "any"),
            "moved": ("move", "past", "any"),
            "pushed": ("push", "past", "any"),
            "rolled": ("roll", "past", "any"),
            "gave": ("give", "past", "any"),
            "given": ("give", "past_participle", "any"),
            "put": ("put", "present", "any"),
        }
        self._irregular_plurals = {
            "boxes": "box",
            "children": "child",
            "feet": "foot",
            "teeth": "tooth",
            "mice": "mouse",
        }

    def tokenize(self, text: str) -> list[str]:
        """Split text into normalized word and punctuation tokens."""
        # Normalize whitespace and separate punctuation
        cleaned = re.sub(r"([?.!,])", r" \1 ", text)
        tokens = cleaned.strip().split()
        return tokens

    def analyze(self, token: str) -> MorphToken:
        """Perform morphological analysis and POS identification for a token."""
        word = token.lower()

        # Check punctuation
        if word in ("?", ".", "!", ","):
            return MorphToken(surface=token, lemma=word, pos=EnglishPOS.PUNCT, features={})

        # Check irregular verbs
        if word in self._irregular_verbs:
            lemma, tense, person = self._irregular_verbs[word]
            pos = EnglishPOS.AUX if lemma == "be" else EnglishPOS.VERB
            return MorphToken(
                surface=token,
                lemma=lemma,
                pos=pos,
                features={"tense": tense, "person": person},
            )

        # Check irregular plurals
        if word in self._irregular_plurals:
            lemma = self._irregular_plurals[word]
            return MorphToken(
                surface=token,
                lemma=lemma,
                pos=EnglishPOS.NOUN,
                features={"number": "plural"},
            )

        # Direct lexicon lookup
        entries = self._lexicon.lookup(word)
        if entries:
            entry = entries[0]
            return MorphToken(
                surface=token,
                lemma=entry.lemma,
                pos=entry.pos,
                features={"number": "singular", "tense": "present"},
            )

        # Rule-based noun pluralization: -es, -s
        if word.endswith("es") and len(word) > 3:
            stem = word[:-2]
            if self._lexicon.has_word(stem):
                return MorphToken(surface=token, lemma=stem, pos=EnglishPOS.NOUN, features={"number": "plural"})
            stem_e = word[:-1]
            if self._lexicon.has_word(stem_e):
                return MorphToken(surface=token, lemma=stem_e, pos=EnglishPOS.NOUN, features={"number": "plural"})
        elif word.endswith("s") and len(word) > 2:
            stem = word[:-1]
            if self._lexicon.has_word(stem):
                return MorphToken(surface=token, lemma=stem, pos=EnglishPOS.NOUN, features={"number": "plural"})

        # Rule-based verb past tense: -ed
        if word.endswith("ed") and len(word) > 3:
            stem = word[:-2]
            if self._lexicon.has_word(stem):
                return MorphToken(surface=token, lemma=stem, pos=EnglishPOS.VERB, features={"tense": "past"})
            stem_d = word[:-1]
            if self._lexicon.has_word(stem_d):
                return MorphToken(surface=token, lemma=stem_d, pos=EnglishPOS.VERB, features={"tense": "past"})

        # Rule-based 3rd person singular verb: -s
        if word.endswith("s") and len(word) > 2:
            stem = word[:-1]
            if self._lexicon.has_word(stem):
                return MorphToken(surface=token, lemma=stem, pos=EnglishPOS.VERB, features={"tense": "present", "person": "3s"})

        # Unknown word fallback -> treat as noun
        return MorphToken(surface=token, lemma=word, pos=EnglishPOS.NOUN, features={})

    def lemmatize_sequence(self, text: str) -> list[MorphToken]:
        """Tokenize and morphologically analyze an entire sentence."""
        raw_tokens = self.tokenize(text)
        return [self.analyze(tok) for tok in raw_tokens]

    def inflect_verb(self, lemma: str, tense: str = "present", person: str = "3s") -> str:
        """Generate surface inflected verb form."""
        if lemma == "be":
            if tense == "past":
                return "was" if person in ("1s", "3s") else "were"
            return "is" if person == "3s" else "are"

        if tense == "past":
            if lemma.endswith("e"):
                return lemma + "d"
            elif lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
                return lemma[:-1] + "ied"
            elif lemma == "fall":
                return "fell"
            elif lemma == "give":
                return "gave"
            return lemma + "ed"

        if tense == "present" and person == "3s":
            if lemma.endswith(("sh", "ch", "s", "x", "z", "o")):
                return lemma + "es"
            elif lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
                return lemma[:-1] + "ies"
            return lemma + "s"

        return lemma

    def pluralize_noun(self, lemma: str) -> str:
        """Generate plural surface form of a noun."""
        if lemma in self._irregular_plurals.values():
            for pl, sg in self._irregular_plurals.items():
                if sg == lemma:
                    return pl
        if lemma.endswith(("s", "sh", "ch", "x", "z")):
            return lemma + "es"
        elif lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ies"
        return lemma + "s"
