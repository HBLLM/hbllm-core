"""Tamil Semantic Parser for A16 non-LLM multilingual cognition.

Parses Tamil utterances (SOV word order + postpositions + inflectional suffixes)
into language-neutral SemanticFrames.
Implements the LanguageParser protocol.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hbllm.brain.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    LanguageErrorType,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)
from hbllm.brain.language.tamil.lexicon import TamilLexicon, TamilPOS

logger = logging.getLogger(__name__)


def _strip_tamil_case_suffix(word: str) -> str:
    """Strip common Tamil nominal case suffixes (accusative, dative, locative)."""
    # Accusative: -யை, -வை, -ஐ (e.g. கையை -> கை, கதவை -> கதவு, பந்தை -> பந்து)
    if word.endswith("யை") and len(word) > 2:
        return word[:-2]
    if word.endswith("வை") and len(word) > 2:
        # e.g., கதவை -> கதவு
        stem = word[:-2]
        return stem + "வு"
    if word.endswith("தை") and len(word) > 2:
        # e.g., பந்தை -> பந்து
        stem = word[:-2]
        return stem + "து"
    if word.endswith("ஐ") and len(word) > 1:
        return word[:-1]
    # Dative: -க்கு, -ற்கு, -உக்கு
    if word.endswith("க்கு") and len(word) > 3:
        return word[:-3]
    if word.endswith("உக்கு") and len(word) > 4:
        return word[:-4]
    return word


class TamilParser:
    """Parses Tamil utterances into language-neutral SemanticFrames."""

    def __init__(self, lexicon: TamilLexicon | None = None) -> None:
        self._lexicon = lexicon or TamilLexicon()

    def parse(self, text: str) -> SemanticFrame:
        """Parse raw Tamil text into a language-neutral SemanticFrame."""
        meta = LanguageMetadata(language="ta", raw_text=text)

        # Normalize and tokenize
        cleaned = re.sub(r"([?.!,])", r" \1 ", text)
        tokens = cleaned.strip().split()
        if not tokens:
            return SemanticFrame(
                frame_type=FrameType.ERROR,
                error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
                error_detail="Empty utterance.",
                metadata=meta,
            )

        clean_tokens = [t for t in tokens if t not in ("?", ".", "!", ",")]

        # 1. Wh-Question: "பந்து எங்கே?" / "பந்து எங்கே உள்ளது?" (Where is the ball?)
        if "எங்கே" in clean_tokens:
            subj_tokens = [
                t for t in clean_tokens if t not in ("எங்கே", "இருக்கிறது", "உள்ளது", "உள்ளது?")
            ]
            subject_ref = self._extract_entity_ref(subj_tokens)

            frame = SemanticFrame(
                frame_type=FrameType.QUERY,
                predicate="located_on",
                query_target="location",
                metadata=meta,
            )
            if subject_ref:
                frame.set_role(ThematicRole.THEME, subject_ref)
            return frame

        # 2. Yes/No Question: "பந்து மேசை மீது இருக்கிறதா?" (Is the ball on the table?)
        if any(t.endswith("ஆ") or t.endswith("இருக்கிறதா") or t == "இருக்கிறதா" for t in clean_tokens):
            theme_tokens, loc_tokens, postp_pred = self._extract_spatial_components(clean_tokens)
            theme_ref = self._extract_entity_ref(theme_tokens)
            loc_ref = self._extract_entity_ref(loc_tokens)

            frame = SemanticFrame(
                frame_type=FrameType.QUERY,
                predicate=postp_pred or "located_on",
                query_target="verification",
                metadata=meta,
            )
            if theme_ref:
                frame.set_role(ThematicRole.THEME, theme_ref)
            if loc_ref:
                frame.set_role(ThematicRole.LOCATION, loc_ref)
            return frame

        # 3. Imperative Command: "முன் கதவை திறக்கவும்" / "ரோபோ கையை சுழற்றவும்"
        command_verbs = {
            "திறக்கவும்": "open",
            "திறக்க": "open",
            "திற": "open",
            "பூட்டவும்": "lock",
            "பூட்டு": "lock",
            "மூடவும்": "close",
            "மூடு": "close",
            "சுழற்றவும்": "rotate",
            "சுழற்று": "rotate",
            "நகர்த்தவும்": "move",
            "நகர்த்து": "move",
            "தள்ளவும்": "push",
            "தள்ளு": "push",
            "வைக்கவும்": "put",
            "வை": "put",
            "நிறுத்தவும்": "stop",
            "நிறுத்து": "stop",
        }
        matched_cmd_verb = None
        matched_verb_phrase = ""
        for cv_key, cv_pred in command_verbs.items():
            if cv_key in text:
                matched_cmd_verb = cv_pred
                matched_verb_phrase = cv_key
                break

        if matched_cmd_verb:
            action_verb = matched_cmd_verb
            verb_tokens = matched_verb_phrase.split()
            target_tokens = [
                t
                for t in clean_tokens
                if t not in verb_tokens
                and t not in ("செய்க", "செய்யவும்", "மீதுக்கு", "நோக்கி", "அதை", "இதை")
            ]
            patient_ref = self._extract_entity_ref(target_tokens)

            frame = SemanticFrame(
                frame_type=FrameType.COMMAND,
                predicate=action_verb,
                metadata=meta,
            )
            if patient_ref:
                frame.set_role(ThematicRole.PATIENT, patient_ref)
            return frame

        # 4. Declarative Assertion: "சிவப்பு பந்து மேசை மீது இருக்கிறது" (The red ball is on the table)
        theme_tokens, loc_tokens, postp_pred = self._extract_spatial_components(clean_tokens)
        if theme_tokens:
            theme_ref = self._extract_entity_ref(theme_tokens)
            loc_ref = self._extract_entity_ref(loc_tokens) if loc_tokens else None

            frame = SemanticFrame(
                frame_type=FrameType.ASSERTION,
                predicate=postp_pred or "located_on",
                metadata=meta,
            )
            if theme_ref:
                frame.set_role(ThematicRole.THEME, theme_ref)
            if loc_ref:
                frame.set_role(ThematicRole.LOCATION, loc_ref)
            return frame

        return SemanticFrame(
            frame_type=FrameType.ERROR,
            error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
            error_detail=f"Tamil parsing failed for '{text}'",
            metadata=meta,
        )

    def _extract_spatial_components(self, tokens: list[str]) -> tuple[list[str], list[str], str]:
        """Split tokens into theme, location, and postposition predicate."""
        postpositions = {
            "மீது": "located_on",
            "மேல்": "located_on",
            "உள்ளே": "located_in",
            "கீழே": "below",
            "அடியில்": "below",
            "அருகில்": "near",
        }
        theme_tokens: list[str] = []
        loc_tokens: list[str] = []
        predicate = "located_on"

        split_idx = -1
        for i, t in enumerate(tokens):
            if t in postpositions:
                split_idx = i
                predicate = postpositions[t]
                break

        if split_idx != -1:
            loc_tokens = [tokens[split_idx - 1]] if split_idx > 0 else []
            theme_tokens = tokens[: split_idx - 1] if split_idx > 1 else tokens[:split_idx]
        else:
            # No postposition -> treat non-verbs as theme
            theme_tokens = [
                t for t in tokens if t not in ("இருக்கிறது", "உள்ளது", "இருக்கிறதா", "உள்ளதா")
            ]

        return theme_tokens, loc_tokens, predicate

    def _extract_entity_ref(self, tokens: list[str]) -> EntityReference | None:
        """Extract an EntityReference from Tamil tokens (e.g. 'சிவப்பு பந்து', 'முன் கதவை')."""
        if not tokens:
            return None

        # Try full phrase lookup first (with stem suffix removal)
        raw_phrase = " ".join(tokens)
        stemmed_phrase = " ".join(_strip_tamil_case_suffix(t) for t in tokens)
        for cand_phrase in (raw_phrase, stemmed_phrase):
            phrase_entries = self._lexicon.lookup(cand_phrase)
            if phrase_entries and phrase_entries[0].pos == TamilPOS.NOUN:
                return EntityReference(
                    concept_name=phrase_entries[0].semantic_predicate,
                    properties=phrase_entries[0].properties or {},
                    specifier="definite",
                    raw_text=raw_phrase,
                )

        props: dict[str, Any] = {}
        concept_name: str | None = None

        i = 0
        while i < len(tokens):
            matched = False
            # Check bigram
            if i + 1 < len(tokens):
                bigram_raw = f"{tokens[i]} {tokens[i + 1]}"
                bigram_stemmed = f"{tokens[i]} {_strip_tamil_case_suffix(tokens[i + 1])}"
                for bg in (bigram_raw, bigram_stemmed):
                    entries = self._lexicon.lookup(bg)
                    if entries and entries[0].pos == TamilPOS.NOUN:
                        concept_name = entries[0].semantic_predicate
                        matched = True
                        i += 2
                        break
                if matched:
                    continue

            # Check unigram
            tok = tokens[i]
            tok_stem = _strip_tamil_case_suffix(tok)
            entries = self._lexicon.lookup(tok) or self._lexicon.lookup(tok_stem)
            if entries:
                entry = entries[0]
                if entry.pos == TamilPOS.ADJ and entry.properties:
                    props.update(entry.properties)
                elif entry.pos == TamilPOS.NOUN:
                    concept_name = entry.semantic_predicate
            i += 1

        if not concept_name and tokens:
            concept_name = _strip_tamil_case_suffix(tokens[-1])

        return EntityReference(
            concept_name=concept_name,
            properties=props,
            specifier="definite",
            raw_text=raw_phrase,
        )
