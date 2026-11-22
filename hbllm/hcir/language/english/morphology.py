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
    r"^(?:https?://[^\s]+|[A-Za-z0-9_./-]*[A-Za-z0-9_/-]+\.[A-Za-z0-9]{1,8}|[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)$"
)

INVARIANT_S_NOUNS: frozenset[str] = frozenset(
    {
        "photosynthesis",
        "analysis",
        "hypothesis",
        "thesis",
        "synthesis",
        "osmosis",
        "mitosis",
        "meiosis",
        "metamorphosis",
        "diagnosis",
        "prognosis",
        "oasis",
        "axis",
        "basis",
        "crisis",
        "genesis",
        "species",
        "series",
        "means",
        "news",
        "physics",
        "optics",
        "mathematics",
        "thermodynamics",
        "electronics",
        "mechanics",
        "genetics",
        "linguistics",
        "economics",
        "politics",
        "ethics",
        "statistics",
        "chaos",
        "cosmos",
        "gas",
        "lens",
        "focus",
        "status",
        "radius",
        "virus",
        "nucleus",
        "campus",
        "corpus",
        "stimulus",
        "consensus",
        "apparatus",
        "cactus",
        "fungus",
        "octopus",
        "syllabus",
        "celsius",
        "atlas",
        "canvas",
        "bias",
        "plus",
        "minus",
        "bus",
        "mars",
        "venus",
        "mass",
        "moss",
        "grass",
        "glass",
        "class",
        "pass",
        "loss",
        "cross",
        "boss",
    }
)

IRREGULAR_PLURALS: dict[str, str] = {
    "boxes": "box",
    "children": "child",
    "feet": "foot",
    "teeth": "tooth",
    "mice": "mouse",
    "men": "man",
    "women": "woman",
    "geese": "goose",
    "people": "person",
    "oxen": "ox",
    "leaves": "leaf",
    "knives": "knife",
    "lives": "life",
    "wives": "wife",
    "wolves": "wolf",
    "halves": "half",
    "shelves": "shelf",
    "calves": "calf",
    "loaves": "loaf",
    "thieves": "thief",
    "elves": "elf",
    "scarves": "scarf",
    "fungi": "fungus",
    "cacti": "cactus",
    "nuclei": "nucleus",
    "phenomena": "phenomenon",
    "criteria": "criterion",
    "bacteria": "bacterium",
}

_NON_VERB_ED_NOUNS: frozenset[str] = frozenset(
    {
        "seed",
        "speed",
        "reed",
        "weed",
        "deed",
        "need",
        "feed",
        "shed",
        "sled",
        "bed",
        "red",
        "hundred",
    }
)

_HYPHEN_ADJ_SUFFIXES: tuple[str, ...] = (
    "-blooded",
    "-legged",
    "-footed",
    "-handed",
    "-eyed",
    "-headed",
    "-minded",
    "-hearted",
    "-bodied",
    "-winged",
    "-tailed",
    "-moving",
    "-growing",
    "-living",
    "-acting",
    "-soluble",
    "-free",
    "-rich",
    "-based",
    "-proof",
    "-like",
    "-born",
    "-bred",
    "-ed",
)


def singularize_noun(term: str) -> str:
    """Deterministically singularize an English noun or underscore compound without mangling stems."""
    if not term:
        return term
    if "_" in term:
        parts = term.split("_")
        parts[-1] = singularize_noun(parts[-1])
        return "_".join(parts)
    if " " in term:
        parts = term.split(" ")
        parts[-1] = singularize_noun(parts[-1])
        return " ".join(parts)

    low = term.lower()
    if low in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[low]
    if low in INVARIANT_S_NOUNS:
        return low
    if len(low) <= 3:
        return low
    if low.endswith(("ss", "us", "is", "ics")):
        return low
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"
    if low.endswith(("shes", "ches", "xes", "zes", "oes")) and len(low) > 4:
        return low[:-2]
    if low.endswith("s"):
        return low[:-1]
    return low


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
            "am": ("be", "present", "1s"),
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
            "ate": ("eat", "past", "any"),
            "eaten": ("eat", "past_participle", "any"),
            "went": ("go", "past", "any"),
            "came": ("come", "past", "any"),
            "saw": ("see", "past", "any"),
            "took": ("take", "past", "any"),
            "made": ("make", "past", "any"),
            "built": ("build", "past", "any"),
            "kept": ("keep", "past", "any"),
            "slept": ("sleep", "past", "any"),
            "flew": ("fly", "past", "any"),
            "swam": ("swim", "past", "any"),
            "ran": ("run", "past", "any"),
            "sat": ("sit", "past", "any"),
            "stood": ("stand", "past", "any"),
            "found": ("find", "past", "any"),
            "felt": ("feel", "past", "any"),
            "left": ("leave", "past", "any"),
            "brought": ("bring", "past", "any"),
            "caught": ("catch", "past", "any"),
            "hid": ("hide", "past", "any"),
            "woke": ("wake", "past", "any"),
            "grew": ("grow", "past", "any"),
            "broke": ("break", "past", "any"),
            "spoke": ("speak", "past", "any"),
            "wrote": ("write", "past", "any"),
            "chose": ("choose", "past", "any"),
            "began": ("begin", "past", "any"),
            "drank": ("drink", "past", "any"),
            "rose": ("rise", "past", "any"),
            "froze": ("freeze", "past", "any"),
            "held": ("hold", "past", "any"),
            "fed": ("feed", "past", "any"),
            "led": ("lead", "past", "any"),
            "met": ("meet", "past", "any"),
            "sent": ("send", "past", "any"),
            "spent": ("spend", "past", "any"),
            "lost": ("lose", "past", "any"),
            "won": ("win", "past", "any"),
            "taught": ("teach", "past", "any"),
            "thought": ("think", "past", "any"),
            "bought": ("buy", "past", "any"),
            "heard": ("hear", "past", "any"),
            "said": ("say", "past", "any"),
            "told": ("tell", "past", "any"),
            "knew": ("know", "past", "any"),
            "became": ("become", "past", "any"),
            "got": ("get", "past", "any"),
            "had": ("have", "past", "any"),
            "did": ("do", "past", "any"),
        }
        self._irregular_plurals = dict(IRREGULAR_PLURALS)

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
    def _alt_pos_str(entries: list[Any]) -> str:
        seen: list[str] = []
        for e in entries:
            val = getattr(e.pos, "value", str(e.pos))
            if val not in seen:
                seen.append(val)
        return ",".join(seen)

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
        feats = dict(features)
        if len(entries) > 1:
            alt = EnglishMorphology._alt_pos_str(entries)
            if "," in alt:
                feats["alt_pos"] = alt
        return MorphToken(surface=surface, lemma=stem, pos=pos, features=feats)

    def analyze(
        self,
        token: str,
        *,
        _allow_fuzzy: bool = True,
        is_sentence_start: bool = True,
    ) -> MorphToken:
        """Perform morphological analysis, inflection stripping, and guarded typo recovery."""
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
            pos = EnglishPOS.AUX if lemma in ("be", "do", "have") else EnglishPOS.VERB
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

        # Protect invariant -s nouns (photosynthesis, species, moss, physics, etc.)
        if word in INVARIANT_S_NOUNS:
            direct = self._trusted_lookup(word)
            pos = direct[0].pos if direct else EnglishPOS.NOUN
            feats: dict[str, str] = {"number": "singular"}
            if direct and len(direct) > 1:
                alt = self._alt_pos_str(direct)
                if "," in alt:
                    feats["alt_pos"] = alt
            return MorphToken(surface=token, lemma=word, pos=pos, features=feats)

        # Protect hyphenated compound adjectives (e.g. warm-blooded, cold-blooded)
        if "-" in word and (
            word.endswith(_HYPHEN_ADJ_SUFFIXES)
            or word.endswith(("ed", "ing", "al", "ic", "ous", "ful", "less", "able", "ible", "ive"))
        ):
            return MorphToken(surface=token, lemma=word, pos=EnglishPOS.ADJ, features={})

        # Direct trusted lexicon lookup
        direct_entries = self._trusted_lookup(word)
        if direct_entries:
            entry = direct_entries[0]
            feats = {"number": "singular", "tense": "present"}
            if len(direct_entries) > 1:
                alt = self._alt_pos_str(direct_entries)
                if "," in alt:
                    feats["alt_pos"] = alt
            return MorphToken(
                surface=token,
                lemma=entry.lemma,
                pos=entry.pos,
                features=feats,
            )

        # Rule-based -ies pluralization/3s: directories -> directory, queries -> query
        if word.endswith("ies") and len(word) > 4:
            stem_y = word[:-3] + "y"
            entries = self._trusted_lookup(stem_y)
            if entries:
                pos = entries[0].pos
                feats = (
                    {"tense": "present", "person": "3s"}
                    if pos == EnglishPOS.VERB
                    else {"number": "plural"}
                )
                return self._inflected_token(
                    token,
                    stem_y,
                    entries,
                    default_pos=EnglishPOS.NOUN,
                    features=feats,
                )

        # Rule-based -es / -s against trusted lexicon (before fuzzy normalization!)
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
        elif word.endswith("s") and len(word) > 2 and not word.endswith(("ss", "us", "is")):
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

        # Rule-based verb past tense: -ed / -ied / doubled consonant
        if word.endswith("ed") and len(word) > 3 and word not in _NON_VERB_ED_NOUNS:
            cand_stems: list[str] = []
            if word.endswith("ied") and len(word) > 4:
                cand_stems.append(word[:-3] + "y")
            cand_stems.extend([word[:-2], word[:-1]])
            base_ed = word[:-2]
            if len(base_ed) >= 3 and base_ed[-1] == base_ed[-2]:
                cand_stems.append(base_ed[:-1])
            for stem in cand_stems:
                entries = self._trusted_lookup(stem)
                if entries and any(e.pos == EnglishPOS.VERB for e in entries):
                    return MorphToken(
                        surface=token,
                        lemma=stem,
                        pos=EnglishPOS.VERB,
                        features={"tense": "past", "alt_pos": "verb,adj"},
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
            corrected = self._normalizer.normalize_token(token, is_sentence_start=is_sentence_start)
            if corrected.lower() != word:
                reanalyzed = self.analyze(
                    corrected, _allow_fuzzy=False, is_sentence_start=is_sentence_start
                )
                feats = dict(reanalyzed.features)
                feats["corrected_from"] = word
                return MorphToken(
                    surface=token,
                    lemma=reanalyzed.lemma,
                    pos=reanalyzed.pos,
                    features=feats,
                )

        # Open-vocabulary regular past verb fallback (e.g. invented verb "glorked")
        if word.endswith("ed") and len(word) > 4 and word not in _NON_VERB_ED_NOUNS:
            stem = word[:-2] if not word.endswith("eed") else word
            return MorphToken(
                surface=token,
                lemma=stem,
                pos=EnglishPOS.VERB,
                features={"tense": "past", "alt_pos": "verb,adj"},
            )

        # Open-vocabulary regular plural noun fallback (skip capitalized proper names in -as/-os/-us/-is)
        if (
            word.endswith("s")
            and len(word) > 3
            and not word.endswith(("ss", "us", "is", "ics"))
            and not (token[0].isupper() and word.endswith(("as", "os")))
        ):
            stem = singularize_noun(word)
            return MorphToken(
                surface=token,
                lemma=stem,
                pos=EnglishPOS.NOUN,
                features={"number": "plural"},
            )

        # Final untrusted fallback -> treat as generic noun
        return MorphToken(surface=token, lemma=word, pos=EnglishPOS.NOUN, features={})

    def lemmatize_sequence(self, text: str) -> list[MorphToken]:
        """Tokenize, morphologically analyze, and contextually disambiguate POS tags."""
        raw_tokens = self.tokenize(text)
        tokens = [
            self.analyze(tok, is_sentence_start=(idx == 0)) for idx, tok in enumerate(raw_tokens)
        ]
        return self._disambiguate_sequence(tokens)

    @staticmethod
    def _has_alt_pos(tok: MorphToken, pos: EnglishPOS) -> bool:
        alt = tok.features.get("alt_pos", "")
        return bool(alt and pos.value in alt.split(","))

    @staticmethod
    def _with_pos(tok: MorphToken, pos: EnglishPOS) -> MorphToken:
        if tok.pos == pos:
            return tok
        return MorphToken(
            surface=tok.surface,
            lemma=tok.lemma,
            pos=pos,
            features=dict(tok.features),
        )

    def _disambiguate_sequence(self, tokens: list[MorphToken]) -> list[MorphToken]:
        """Resolve noun/verb/adj ambiguities using deterministic local syntactic context."""
        if not tokens:
            return tokens
        out = list(tokens)
        content_idxs = [i for i, t in enumerate(out) if t.pos != EnglishPOS.PUNCT]
        if not content_idxs:
            return out

        first_i = content_idxs[0]
        first = out[first_i]

        # 1. Sentence-initial plural noun ("Plants absorb...", "Leaves convert...")
        if first.features.get("number") == "plural" and first.pos == EnglishPOS.VERB:
            if self._has_alt_pos(first, EnglishPOS.NOUN) or first.surface.lower().endswith("s"):
                out[first_i] = self._with_pos(first, EnglishPOS.NOUN)

        # 2. Determiner + noun / inflected noun phrase head: word right after DET (or DET Adj*)
        for k, idx in enumerate(content_idxs):
            tok = out[idx]
            if k > 0 and out[content_idxs[k - 1]].pos == EnglishPOS.DET:
                if self._has_alt_pos(tok, EnglishPOS.NOUN):
                    # Unless followed immediately by another noun that this modifies as ADJ
                    next_is_noun = (
                        k + 1 < len(content_idxs)
                        and out[content_idxs[k + 1]].pos == EnglishPOS.NOUN
                        and tok.pos == EnglishPOS.ADJ
                    )
                    if not next_is_noun:
                        out[idx] = self._with_pos(tok, EnglishPOS.NOUN)

        # 3. Infinitive "to + <V>" (e.g., "caused the river to flood")
        for k in range(1, len(content_idxs)):
            prev_tok = out[content_idxs[k - 1]]
            idx = content_idxs[k]
            tok = out[idx]
            if prev_tok.lemma == "to" and self._has_alt_pos(tok, EnglishPOS.VERB):
                out[idx] = self._with_pos(tok, EnglishPOS.VERB)

        # 4. Aux (do/does/did/can/could/will/would/may/might/must) + ... + verb
        # Handles both declarative ("Heat does not cause rust", "Plants can absorb light")
        # and inverted questions ("What do florbs eat?", "Where do florbs live?", "Do florbs fly?")
        do_modal_lemmas = frozenset(
            {"do", "does", "did", "can", "could", "will", "would", "may", "might", "must"}
        )
        if len(content_idxs) >= 3:
            c0 = out[content_idxs[0]]
            c1 = out[content_idxs[1]]
            is_wh_do = c0.pos == EnglishPOS.WH and c1.lemma in do_modal_lemmas
            is_yn_do = c0.lemma in do_modal_lemmas or c0.surface.lower() in (
                "do",
                "does",
                "did",
                "can",
            )
            if is_wh_do or is_yn_do:
                start_k = 2 if is_wh_do else 1
                # Scan after subject NP for first token that can be a verb (or last token before end)
                verb_picked = False
                for k in range(start_k + 1, len(content_idxs)):
                    idx = content_idxs[k]
                    tok = out[idx]
                    if tok.pos == EnglishPOS.VERB:
                        verb_picked = True
                        break
                    if self._has_alt_pos(tok, EnglishPOS.VERB):
                        out[idx] = self._with_pos(tok, EnglishPOS.VERB)
                        verb_picked = True
                        break
                # Open-vocab fallback for WH + do/does/did + Subject + UnknownVerb ("What do florbs glork?")
                if not verb_picked and (
                    (is_wh_do and c1.lemma in ("do", "does", "did"))
                    or (is_yn_do and c0.lemma in ("do", "does", "did"))
                ):
                    last_k = len(content_idxs) - 1
                    # For YN or intransitive WH, check last token or token before first PREP
                    cand_k = last_k
                    for k in range(start_k + 1, len(content_idxs)):
                        if out[content_idxs[k]].pos == EnglishPOS.PREP:
                            cand_k = k - 1
                            break
                    if cand_k >= start_k + 1:
                        cand_tok = out[content_idxs[cand_k]]
                        if (
                            cand_tok.pos == EnglishPOS.NOUN
                            and cand_tok.features.get("is_path") != "true"
                        ):
                            out[content_idxs[cand_k]] = self._with_pos(cand_tok, EnglishPOS.VERB)

        # 5. Declarative modal/do + optional not + verb ("Mira did not walk", "Birds can fly")
        for k in range(len(content_idxs) - 1):
            tok = out[content_idxs[k]]
            if tok.lemma in do_modal_lemmas or tok.surface.lower() in ("do", "does", "did"):
                if k == 0 or (k == 1 and out[content_idxs[0]].pos == EnglishPOS.WH):
                    continue
                nxt_k = k + 1
                if nxt_k < len(content_idxs) and out[content_idxs[nxt_k]].lemma in ("not", "never"):
                    nxt_k += 1
                if nxt_k < len(content_idxs):
                    v_idx = content_idxs[nxt_k]
                    v_tok = out[v_idx]
                    if self._has_alt_pos(v_tok, EnglishPOS.VERB):
                        out[v_idx] = self._with_pos(v_tok, EnglishPOS.VERB)

        # 6. Subject NP + <dual N/V> -> VERB ("Florbs live in caves", "Plants absorb sunlight", "Heavy rain caused...")
        if len(content_idxs) >= 2 and out[content_idxs[0]].pos in (
            EnglishPOS.DET,
            EnglishPOS.ADJ,
            EnglishPOS.NOUN,
            EnglishPOS.PRON,
        ):
            # Walk across initial NP (Det? Adj* Noun+)
            k = 0
            if out[content_idxs[k]].pos == EnglishPOS.PRON:
                k = 1
            else:
                if k < len(content_idxs) and out[content_idxs[k]].pos == EnglishPOS.DET:
                    k += 1
                while k < len(content_idxs) and out[content_idxs[k]].pos == EnglishPOS.ADJ:
                    k += 1
                noun_count = 0
                while (
                    k < len(content_idxs)
                    and out[content_idxs[k]].pos == EnglishPOS.NOUN
                    and noun_count < 3
                ):
                    if noun_count >= 1 and self._has_alt_pos(out[content_idxs[k]], EnglishPOS.VERB):
                        break
                    noun_count += 1
                    k += 1
            if k < len(content_idxs):
                v_idx = content_idxs[k]
                v_tok = out[v_idx]
                if self._has_alt_pos(v_tok, EnglishPOS.VERB) and v_tok.pos != EnglishPOS.AUX:
                    out[v_idx] = self._with_pos(v_tok, EnglishPOS.VERB)

        # 7. Subject-seeking "Who + <V>" ("Who walked home?", "Who eats moss?")
        if len(content_idxs) >= 2 and out[content_idxs[0]].lemma == "who":
            v_idx = content_idxs[1]
            v_tok = out[v_idx]
            if v_tok.pos != EnglishPOS.AUX and self._has_alt_pos(v_tok, EnglishPOS.VERB):
                out[v_idx] = self._with_pos(v_tok, EnglishPOS.VERB)

        # 8. Predicative adjective after copula "be" ("Mira was tired", "A drell is warm-blooded")
        for k in range(len(content_idxs) - 1):
            tok = out[content_idxs[k]]
            if tok.lemma == "be":
                nxt_k = k + 1
                if nxt_k < len(content_idxs) and out[content_idxs[nxt_k]].lemma in (
                    "not",
                    "very",
                    "quite",
                ):
                    nxt_k += 1
                if nxt_k < len(content_idxs):
                    a_idx = content_idxs[nxt_k]
                    a_tok = out[a_idx]
                    if self._has_alt_pos(a_tok, EnglishPOS.ADJ):
                        out[a_idx] = self._with_pos(a_tok, EnglishPOS.ADJ)

        return out

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
