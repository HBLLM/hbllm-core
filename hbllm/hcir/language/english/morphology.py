"""English Morphology & Tokenizer for A16.

Provides deterministic rule-based inflection, lemmatization, and tokenization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from hbllm.hcir.language.english.lexicon import EnglishLexicon, EnglishPOS
from hbllm.hcir.language.english.normalizer import OrthographicNormalizer

_URL_OR_PATH_TOKEN_RE = re.compile(
    r"""
    https?://[^\s<>"'()]*[^\s<>"'().,!?;:]
    | [A-Za-z0-9_./-]*[A-Za-z0-9_/-]+\.[A-Za-z0-9]{1,8}(?=/|\s|$|[.,!?;:])
    | [A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+
    | [A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*
    | [?.!,;:]
    """,
    re.VERBOSE,
)
_STRICT_PATH_RE = re.compile(
    r"^(?:https?://[^\s]+|[A-Za-z0-9_./-]*[A-Za-z0-9_/-]+\.[A-Za-z]{1,8}|[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)$"
)


def is_path_or_url(token: str) -> bool:
    """Return True if *token* looks like a file path, dotted filename, or URL."""
    return bool(_STRICT_PATH_RE.match(token))


@dataclass(frozen=True)
class MorphToken:
    """A single token with its surface form, lemma, POS, and morphological features."""

    surface: str
    lemma: str
    pos: EnglishPOS
    features: dict[str, str]  # e.g., {"number": "plural", "tense": "past"}


class EnglishMorphology:
    """Morphological analyzer and inflector for English."""

    def __init__(
        self,
        lexicon: EnglishLexicon | None = None,
        normalizer: OrthographicNormalizer | None = None,
    ) -> None:
        self._lexicon = lexicon or EnglishLexicon()
        self._normalizer = normalizer or OrthographicNormalizer(lexicon=self._lexicon)
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
        """Split text into normalized word, path/URL, and punctuation tokens."""
        tokens = _URL_OR_PATH_TOKEN_RE.findall(text.strip())
        if tokens:
            return tokens
        cleaned = re.sub(r"([?.!,;:])", r" \1 ", text)
        return cleaned.strip().split()

    def _trusted_lookup(self, stem: str) -> list[Any]:
        if hasattr(self._lexicon, "is_trusted_word"):
            if not self._lexicon.is_trusted_word(stem):
                return []
            return list(self._lexicon.lookup(stem))
        if self._lexicon.has_word(stem):
            return list(self._lexicon.lookup(stem))
        return []

    @staticmethod
    def _inflected_token(
        surface: str,
        stem: str,
        entries: list[Any],
        *,
        default_pos: EnglishPOS,
        features: dict[str, str],
    ) -> MorphToken:
        pos = entries[0].pos if entries else default_pos
        return MorphToken(surface=surface, lemma=stem, pos=pos, features=dict(features))

    def analyze(self, token: str, *, _allow_fuzzy: bool = True) -> MorphToken:
        """Perform morphological analysis, inflection stripping, and typo recovery."""
        word = token.lower()

        # Check punctuation
        if word in ("?", ".", "!", ",", ";", ":"):
            return MorphToken(surface=token, lemma=word, pos=EnglishPOS.PUNCT, features={})

        # Protect filenames, paths, and URLs with exact surface casing
        if is_path_or_url(token):
            return MorphToken(
                surface=token,
                lemma=token,
                pos=EnglishPOS.NOUN,
                features={"is_path": "true"},
            )

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

        # Direct trusted lexicon lookup
        direct_entries = self._trusted_lookup(word)
        if direct_entries:
            entry = direct_entries[0]
            return MorphToken(
                surface=token,
                lemma=entry.lemma,
                pos=entry.pos,
                features={"number": "singular", "tense": "present"},
            )

        # Rule-based -ies pluralization: directories -> directory, queries -> query
        if word.endswith("ies") and len(word) > 4:
            stem_y = word[:-3] + "y"
            entries = self._trusted_lookup(stem_y)
            if entries:
                return self._inflected_token(
                    token,
                    stem_y,
                    entries,
                    default_pos=EnglishPOS.NOUN,
                    features={"number": "plural"},
                )

        # Rule-based -es / -s: sibilant stems check -es before -e; others check -s (-e) FIRST
        # so "files" -> "file" (not "fil") and "boxes"/"focuses" -> "box"/"focus".
        if word.endswith("es") and len(word) > 3:
            if word.endswith(("shes", "ches", "xes", "zes", "ses", "oes")):
                stem_order = (word[:-2], word[:-1])
            else:
                stem_order = (word[:-1], word[:-2])
            for stem in stem_order:
                entries = self._trusted_lookup(stem)
                if entries:
                    pos = entries[0].pos
                    feats = (
                        {"tense": "present", "person": "3s"}
                        if pos == EnglishPOS.VERB
                        else {"number": "plural"}
                    )
                    return self._inflected_token(
                        token,
                        stem,
                        entries,
                        default_pos=EnglishPOS.NOUN,
                        features=feats,
                    )
        elif word.endswith("s") and len(word) > 2:
            stem = word[:-1]
            entries = self._trusted_lookup(stem)
            if entries:
                pos = entries[0].pos
                feats = (
                    {"tense": "present", "person": "3s"}
                    if pos == EnglishPOS.VERB
                    else {"number": "plural"}
                )
                return self._inflected_token(
                    token,
                    stem,
                    entries,
                    default_pos=EnglishPOS.NOUN,
                    features=feats,
                )

        # Rule-based verb past tense: -ed
        if word.endswith("ed") and len(word) > 3:
            for stem in (word[:-2], word[:-1]):
                if self._trusted_lookup(stem):
                    return MorphToken(
                        surface=token, lemma=stem, pos=EnglishPOS.VERB, features={"tense": "past"}
                    )

        # Rule-based progressive verb: -ing
        if word.endswith("ing") and len(word) > 5:
            for stem in (word[:-3], word[:-3] + "e"):
                entries = self._trusted_lookup(stem)
                if entries and any(e.pos == EnglishPOS.VERB for e in entries):
                    return MorphToken(
                        surface=token,
                        lemma=stem,
                        pos=EnglishPOS.VERB,
                        features={"aspect": "progressive"},
                    )

        # Orthographic normalization fallback for untrusted typos
        if _allow_fuzzy:
            corrected = self._normalizer.normalize_token(token)
            if corrected.lower() != word:
                reanalyzed = self.analyze(corrected, _allow_fuzzy=False)
                feats = dict(reanalyzed.features)
                feats["corrected_from"] = word
                return MorphToken(
                    surface=token,
                    lemma=reanalyzed.lemma,
                    pos=reanalyzed.pos,
                    features=feats,
                )

        # Final untrusted fallback -> treat as generic noun
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
