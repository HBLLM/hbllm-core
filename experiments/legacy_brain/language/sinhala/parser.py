"""Sinhala Semantic Parser for A16.

Parses Sinhala utterances (SOV word order + postpositions) into
language-neutral SemanticFrames.
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
from hbllm.brain.language.sinhala.lexicon import SinhalaLexicon, SinhalaPOS

logger = logging.getLogger(__name__)


class SinhalaParser:
    """Parses Sinhala utterances into language-neutral SemanticFrames."""

    def __init__(self, lexicon: SinhalaLexicon | None = None) -> None:
        self._lexicon = lexicon or SinhalaLexicon()

    def parse(self, text: str) -> SemanticFrame:
        """Parse raw Sinhala text into a language-neutral SemanticFrame."""
        meta = LanguageMetadata(language="si", raw_text=text)

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

        # 1. Wh-Question: "බෝලය කොහෙද?" / "බෝලය කොහෙද තියෙන්නේ?" (Where is the ball?)
        if "කොහෙද" in clean_tokens:
            subj_tokens = [
                t for t in clean_tokens if t not in ("කොහෙද", "තියෙන්නේ", "තිබෙන්නේ", "තියෙනවා")
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

        # 2. Yes/No Question: "බෝලය මේසය මත තියෙනවද?" (Is the ball on the table?)
        if any(t.endswith("ද") or t == "තියෙනවද" for t in clean_tokens):
            # Spatial verification query
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

        # 3. Imperative Command: "බෝලය මේසය මතට ගෙනයන්න" / "ඉදිරිපස දොර අරින්න" / "රොබෝ අත කරකවන්න"
        command_verbs = {
            "විවෘත කරන්න": "open",
            "අගුළු අරින්න": "unlock",
            "අගුළුහරින්න": "unlock",
            "අගුළු දමන්න": "lock",
            "අගුළුලන්න": "lock",
            "තල්ලු කරන්න": "push",
            "ගෙනයන්න": "move",
            "දමන්න": "put",
            "තල්ලු": "push",
            "අරින්න": "open",
            "ඇරගන්න": "open",
            "අරින්නට": "open",
            "වහන්න": "close",
            "පියවන්න": "close",
            "කරකවන්න": "rotate",
            "නවතන්න": "stop",
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
                and t not in ("කරන්න", "කරපන්", "කරන්නට", "මතට", "වෙත", "එය", "මේක")
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

        # 4. Declarative Assertion: "රතු බෝලය මේසය මත තියෙනවා" (The red ball is on the table)
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
            error_detail=f"Sinhala parsing failed for '{text}'",
            metadata=meta,
        )

    def _extract_spatial_components(self, tokens: list[str]) -> tuple[list[str], list[str], str]:
        """Split tokens into theme, location, and postposition predicate."""
        postpositions = {
            "මත": "located_on",
            "උඩ": "located_on",
            "තුළ": "located_in",
            "ඇතුලේ": "located_in",
            "යට": "below",
            "ළඟ": "near",
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
            theme_tokens = [t for t in tokens if t not in ("තියෙනවා", "තිබේ", "තියෙනවද")]

        return theme_tokens, loc_tokens, predicate

    def _extract_entity_ref(self, tokens: list[str]) -> EntityReference | None:
        """Extract an EntityReference from Sinhala tokens (e.g. 'රතු බෝලය', 'ඉදිරිපස දොර')."""
        if not tokens:
            return None

        # Check full joined phrase in lexicon first
        full_phrase = " ".join(tokens)
        phrase_entries = self._lexicon.lookup(full_phrase)
        if phrase_entries and phrase_entries[0].pos == SinhalaPOS.NOUN:
            return EntityReference(
                concept_name=phrase_entries[0].semantic_predicate,
                properties=phrase_entries[0].properties or {},
                specifier="definite",
                raw_text=full_phrase,
            )

        props: dict[str, Any] = {}
        concept_name: str | None = None

        # Check two-word sliding window then individual tokens
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens):
                bigram = f"{tokens[i]} {tokens[i + 1]}"
                entries = self._lexicon.lookup(bigram)
                if entries:
                    entry = entries[0]
                    if entry.pos == SinhalaPOS.NOUN:
                        concept_name = entry.semantic_predicate
                        i += 2
                        continue
            entries = self._lexicon.lookup(tokens[i])
            if entries:
                entry = entries[0]
                if entry.pos == SinhalaPOS.ADJ and entry.properties:
                    props.update(entry.properties)
                elif entry.pos == SinhalaPOS.NOUN:
                    concept_name = entry.semantic_predicate
            i += 1

        if not concept_name and tokens:
            concept_name = tokens[-1]

        return EntityReference(
            concept_name=concept_name,
            properties=props,
            specifier="definite",
            raw_text=" ".join(tokens),
        )
