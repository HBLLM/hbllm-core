"""Universal Declarative Cognitive Language Engine.

Provides language-independent parsing and realization parameterized entirely
by declarative `LanguageProfile` instances, enabling training and seeding
of new human languages without modifying Python source code.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hbllm.hcir.language.core.epistemic_policy import (
    CognitiveEpistemicState,
    EpistemicRealizationPolicy,
)
from hbllm.hcir.language.core.profile import LanguageProfile
from hbllm.hcir.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    LanguageErrorType,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)

logger = logging.getLogger(__name__)


class UniversalLanguageParser:
    """Language-agnostic parser driven entirely by a declarative LanguageProfile."""

    def __init__(
        self,
        profile: LanguageProfile,
        *,
        lexicon: Any | None = None,
        ontology_store: Any | None = None,
    ) -> None:
        self.profile = profile
        self.lexicon = lexicon
        self._ontology_store = ontology_store
        self._dynamic_vocab: set[str] = set()

    def set_dynamic_vocabulary(self, words: set[str]) -> None:
        """Update active entities and concepts for vocabulary matching."""
        self._dynamic_vocab = {w.lower() for w in words if w}

    def parse(self, text: str) -> SemanticFrame:
        """Parse natural language text into a language-neutral SemanticFrame."""
        meta = LanguageMetadata(language=self.profile.iso_code, raw_text=text)

        cleaned = text.strip()
        if not cleaned:
            return SemanticFrame(
                frame_type=FrameType.ERROR,
                error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
                error_detail="Empty utterance.",
                metadata=meta,
            )

        # 1. Strip polite conversational wrappers (e.g. "s'il vous plaît", "please can you")
        cleaned_no_wrapper = self._strip_polite_wrappers(cleaned)

        # Tokenize preserving word and punctuation boundaries
        tokens = self._tokenize(cleaned_no_wrapper)
        if not tokens:
            return SemanticFrame(
                frame_type=FrameType.ERROR,
                error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
                error_detail="No content tokens after normalisation.",
                metadata=meta,
            )

        # 2. Wh-Question Detection
        wh_frame = self._try_parse_wh_question(tokens, meta)
        if wh_frame is not None:
            return wh_frame

        # 3. Yes/No Verification Query Detection
        yn_frame = self._try_parse_yes_no_question(tokens, meta)
        if yn_frame is not None:
            return yn_frame

        # 4. Command Detection (Imperatives, Capabilities, Tool dispatch)
        cmd_frame = self._try_parse_command(tokens, meta, cleaned_no_wrapper)
        if cmd_frame is not None:
            return cmd_frame

        # 5. Expository Causal/Dependency Assertions (SOV expository markers)
        exp_frame = self._try_parse_expository_assertion(tokens, meta)
        if exp_frame is not None:
            return exp_frame

        # 6. Spatial Assertion Detection (Prepositional or Postpositional)
        spatial_frame = self._try_parse_spatial_assertion(tokens, meta)
        if spatial_frame is not None:
            return spatial_frame

        # 7. Copular Definition Assertion (NP + BE + NP -> is_a)
        def_frame = self._try_parse_copular_assertion(tokens, meta)
        if def_frame is not None:
            return def_frame

        # Default fallback
        return SemanticFrame(
            frame_type=FrameType.ERROR,
            error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
            error_detail=f"Could not parse utterance under {self.profile.name} grammar.",
            metadata=meta,
        )

    # ── Internal Parsing Subroutines ──────────────────────────────────

    def _strip_polite_wrappers(self, text: str) -> str:
        s = text.strip()
        low = s.lower()
        for wrapper in sorted(self.profile.closed_class.polite_wrappers, key=len, reverse=True):
            w_low = wrapper.lower()
            if low.startswith(w_low):
                rem = s[len(wrapper) :].strip()
                rem = rem.lstrip(",:;").strip()
                return rem
            if low.endswith(w_low):
                rem = s[: -len(wrapper)].strip()
                rem = rem.rstrip(",:;").strip()
                return rem
        return s

    def _tokenize(self, text: str) -> list[str]:
        # Split on whitespace while separating terminal and inverted punctuation (e.g. ¿, ¡)
        s = re.sub(r"([?.!,;:¿¡«»\"'()])", r" \1 ", text)
        toks = []
        for t in s.split():
            clean = t.strip("¿¡?.!,;:\"'()")
            if clean:
                toks.append(clean)
            elif t in ("?", "!", "¿", "¡"):
                toks.append(t)
        return toks

    def _try_parse_wh_question(
        self, tokens: list[str], meta: LanguageMetadata
    ) -> SemanticFrame | None:
        q_markers = self.profile.closed_class.question_markers
        clean_tokens = [t for t in tokens if t not in ("?", ".", "!", ",", "¿", "¡", ";", ":")]

        if self.profile.typology.wh_movement == "front":
            if not clean_tokens:
                return None
            first_tok = clean_tokens[0].lower()
            # Multi-word marker check (e.g. "est-ce que")
            matched_marker: str | None = None
            if len(clean_tokens) >= 2:
                two_word = f"{clean_tokens[0]} {clean_tokens[1]}".lower()
                if two_word in q_markers:
                    matched_marker = two_word
            if matched_marker is None and first_tok in q_markers:
                matched_marker = first_tok

            if matched_marker is not None:
                spec = q_markers[matched_marker]
                rem_tokens = clean_tokens[2:] if " " in matched_marker else clean_tokens[1:]
                # Filter out copula
                copulas = set(self.profile.closed_class.copula_forms)
                filtered = [t for t in rem_tokens if t.lower() not in copulas]
                subj_ref = self._extract_entity_ref(filtered)

                target = spec.get("target", "location")
                pred = spec.get("predicate", "located_on")
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate=pred,
                    query_target=target,
                    metadata=meta,
                )
                if subj_ref is not None:
                    frame.set_role(ThematicRole.THEME, subj_ref)
                return frame

        else:
            # in_situ Wh-question (Sinhala, Tamil, Japanese)
            for tok in clean_tokens:
                tok_low = tok.lower()
                if tok_low in q_markers:
                    spec = q_markers[tok_low]
                    copulas = set(self.profile.closed_class.copula_forms)
                    subj_tokens = [
                        t for t in clean_tokens if t.lower() != tok_low and t.lower() not in copulas
                    ]
                    subj_ref = self._extract_entity_ref(subj_tokens)
                    target = spec.get("target", "location")
                    pred = spec.get("predicate", "located_on")
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate=pred,
                        query_target=target,
                        metadata=meta,
                    )
                    if subj_ref is not None:
                        frame.set_role(ThematicRole.THEME, subj_ref)
                    return frame

        return None

    def _try_parse_yes_no_question(
        self, tokens: list[str], meta: LanguageMetadata
    ) -> SemanticFrame | None:
        clean_tokens = [t for t in tokens if t not in ("?", ".", "!", ",")]
        if not clean_tokens:
            return None

        # Strategy A: Question particle (Sinhala "ද", Tamil "ஆ")
        if self.profile.typology.yes_no_strategy == "particle":
            particles = self.profile.closed_class.question_particles
            has_particle = any(
                any(tok.endswith(p) or tok == p for p in particles) for tok in clean_tokens
            )
            if has_particle:
                # Remove particle from token
                stripped_tokens: list[str] = []
                for t in clean_tokens:
                    cleaned_t = t
                    for p in particles:
                        if cleaned_t.endswith(p) and len(cleaned_t) > len(p):
                            cleaned_t = cleaned_t[: -len(p)]
                            break
                    if cleaned_t not in particles:
                        stripped_tokens.append(cleaned_t)

                theme_toks, loc_toks, pred = self._split_spatial_tokens(stripped_tokens)
                theme_ref = self._extract_entity_ref(theme_toks)
                loc_ref = self._extract_entity_ref(loc_toks)

                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate=pred or "located_on",
                    query_target="verification",
                    metadata=meta,
                )
                if theme_ref:
                    frame.set_role(ThematicRole.THEME, theme_ref)
                if loc_ref:
                    frame.set_role(ThematicRole.LOCATION, loc_ref)
                return frame

        # Strategy B: Aux/Copula Inversion (English, German)
        elif self.profile.typology.yes_no_strategy == "inversion":
            copulas = set(self.profile.closed_class.copula_forms)
            if clean_tokens[0].lower() in copulas:
                rem = clean_tokens[1:]
                theme_toks, loc_toks, pred = self._split_spatial_tokens(rem)
                theme_ref = self._extract_entity_ref(theme_toks)
                loc_ref = self._extract_entity_ref(loc_toks)
                if theme_ref and loc_ref:
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate=pred or "located_on",
                        query_target="verification",
                        metadata=meta,
                    )
                    frame.set_role(ThematicRole.THEME, theme_ref)
                    frame.set_role(ThematicRole.LOCATION, loc_ref)
                    return frame

        # Strategy C: Tag Question (French "est-ce que")
        elif self.profile.typology.yes_no_strategy == "tag":
            tag_markers = ["est-ce que", "est-ce qu'", "est-ce qu"]
            raw_low = " ".join(clean_tokens).lower()
            for tm in tag_markers:
                if raw_low.startswith(tm):
                    rem_words = clean_tokens[len(tm.split()) :]
                    theme_toks, loc_toks, pred = self._split_spatial_tokens(rem_words)
                    theme_ref = self._extract_entity_ref(theme_toks)
                    loc_ref = self._extract_entity_ref(loc_toks)
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate=pred or "located_on",
                        query_target="verification",
                        metadata=meta,
                    )
                    if theme_ref:
                        frame.set_role(ThematicRole.THEME, theme_ref)
                    if loc_ref:
                        frame.set_role(ThematicRole.LOCATION, loc_ref)
                    return frame

        return None

    def _try_parse_command(
        self, tokens: list[str], meta: LanguageMetadata, raw_text: str
    ) -> SemanticFrame | None:
        cmd_verbs = self.profile.closed_class.command_verbs
        clean_tokens = [t for t in tokens if t not in ("?", ".", "!", ",")]
        if not clean_tokens:
            return None

        # Check multi-word command verbs first (e.g. "විවෘත කරන්න", "qu'est-ce que")
        matched_verb: str | None = None
        matched_cap: str | None = None
        rem_tokens: list[str] = []

        is_sov = self.profile.typology.word_order == "SOV"

        if is_sov:
            # Check tail of sentence for command verbs (e.g. "ඉදිරිපස දොර අරින්න")
            for length in (3, 2, 1):
                if len(clean_tokens) >= length:
                    candidate = " ".join(clean_tokens[-length:]).lower()
                    if candidate in cmd_verbs:
                        matched_verb = candidate
                        matched_cap = cmd_verbs[candidate]
                        rem_tokens = clean_tokens[:-length]
                        break

        if matched_verb is None or matched_cap is None:
            # Check head of sentence for command verbs (e.g. "open the door")
            for length in (3, 2, 1):
                if len(clean_tokens) >= length:
                    candidate = " ".join(clean_tokens[:length]).lower()
                    if candidate in cmd_verbs:
                        matched_verb = candidate
                        matched_cap = cmd_verbs[candidate]
                        rem_tokens = clean_tokens[length:]
                        break

        if matched_verb is None or matched_cap is None:
            return None

        target_ref = self._extract_entity_ref(rem_tokens) if rem_tokens else None

        frame = SemanticFrame(
            frame_type=FrameType.COMMAND,
            predicate=matched_cap,
            metadata=meta,
        )
        if target_ref is not None:
            frame.set_role(ThematicRole.THEME, target_ref)
            frame.set_role(ThematicRole.PATIENT, target_ref)

        return frame

    def _try_parse_spatial_assertion(
        self, tokens: list[str], meta: LanguageMetadata
    ) -> SemanticFrame | None:
        clean_tokens = [t for t in tokens if t not in ("?", ".", "!", ",")]
        theme_toks, loc_toks, pred = self._split_spatial_tokens(clean_tokens)
        if not theme_toks or not loc_toks or not pred:
            return None

        theme_ref = self._extract_entity_ref(theme_toks)
        loc_ref = self._extract_entity_ref(loc_toks)
        if theme_ref is None or loc_ref is None:
            return None

        frame = SemanticFrame(
            frame_type=FrameType.ASSERTION,
            predicate=pred,
            metadata=meta,
        )
        frame.set_role(ThematicRole.THEME, theme_ref)
        frame.set_role(ThematicRole.LOCATION, loc_ref)
        return frame

    def _try_parse_expository_assertion(
        self, tokens: list[str], meta: LanguageMetadata
    ) -> SemanticFrame | None:
        if "?" in tokens or self.profile.typology.word_order != "SOV":
            return None
        clean_tokens = [t for t in tokens if t not in ("?", ".", "!", ",")]
        if len(clean_tokens) < 3:
            return None

        markers = self.profile.closed_class.expository_markers
        rel_map = {
            "cause": "causes",
            "require": "requires",
            "enable": "enables",
        }
        candidates: list[tuple[int, str, str]] = []
        for cue_key, pred in rel_map.items():
            for phrase in markers.get(cue_key, ()):
                phrase_parts = phrase.strip().lower().split()
                if phrase_parts:
                    candidates.append((len(phrase_parts), " ".join(phrase_parts), pred))
        candidates.sort(key=lambda item: item[0], reverse=True)

        matched_len = 0
        matched_pred = ""
        for m_len, m_phrase, pred in candidates:
            if len(clean_tokens) - m_len >= 2:
                tail = " ".join(t.lower() for t in clean_tokens[-m_len:])
                if tail == m_phrase:
                    matched_len = m_len
                    matched_pred = pred
                    break

        if not matched_pred:
            return None

        rem = clean_tokens[:-matched_len]
        split_idx = 1
        for i in range(1, len(rem) - 1):
            if rem[i] in ("යනු", "என்பது"):
                split_idx = i
                lhs = rem[:split_idx]
                rhs = rem[split_idx + 1 :]
                break
        else:
            lhs = rem[:1]
            rhs = rem[1:]

        theme_ref = self._extract_entity_ref(lhs)
        pat_ref = self._extract_entity_ref(rhs)
        if theme_ref is None or pat_ref is None:
            return None

        frame = SemanticFrame(
            frame_type=FrameType.ASSERTION,
            predicate=matched_pred,
            metadata=meta,
        )
        frame.set_role(ThematicRole.THEME, theme_ref)
        frame.set_role(ThematicRole.PATIENT, pat_ref)
        return frame

    def _try_parse_copular_assertion(
        self, tokens: list[str], meta: LanguageMetadata
    ) -> SemanticFrame | None:
        if "?" in tokens:
            return None
        clean_tokens = [t for t in tokens if t not in ("?", ".", "!", ",")]
        copulas = {c.lower() for c in self.profile.closed_class.copula_forms}

        cop_idx = -1
        for i, tok in enumerate(clean_tokens):
            if tok.lower() in copulas and 0 < i < len(clean_tokens) - 1:
                cop_idx = i
                break

        if cop_idx > 0:
            lhs = clean_tokens[:cop_idx]
            rhs = clean_tokens[cop_idx + 1 :]
        elif (
            self.profile.typology.word_order == "SOV"
            and len(clean_tokens) >= 3
            and clean_tokens[-1].lower() in copulas
        ):
            body = clean_tokens[:-1]
            topic_idx = -1
            for i in range(1, len(body) - 1):
                if body[i] in ("යනු", "என்பது"):
                    topic_idx = i
                    break
            if topic_idx > 0:
                lhs = body[:topic_idx]
                rhs = body[topic_idx + 1 :]
            else:
                lhs = body[:1]
                rhs = body[1:]
        else:
            return None

        theme_ref = self._extract_entity_ref(lhs)
        attr_ref = self._extract_entity_ref(rhs)
        if theme_ref is None or attr_ref is None:
            return None

        frame = SemanticFrame(
            frame_type=FrameType.ASSERTION,
            predicate="is_a",
            metadata=meta,
        )
        frame.set_role(ThematicRole.THEME, theme_ref)
        frame.set_role(ThematicRole.PATIENT, attr_ref)
        return frame

    def _split_spatial_tokens(self, tokens: list[str]) -> tuple[list[str], list[str], str | None]:
        adpositions = self.profile.closed_class.adpositions
        copulas = set(self.profile.closed_class.copula_forms)

        if self.profile.typology.adposition_type == "preposition":
            # [Theme NP] [Copula] [Preposition] [Location NP]
            adp_idx = -1
            pred: str | None = None
            for i, t in enumerate(tokens):
                t_low = t.lower()
                if t_low in adpositions:
                    adp_idx = i
                    pred = adpositions[t_low]
                    break

            if adp_idx <= 0:
                return [], [], None

            lhs = tokens[:adp_idx]
            rhs = tokens[adp_idx + 1 :]
            # Filter copula from lhs
            theme_toks = [t for t in lhs if t.lower() not in copulas]
            loc_toks = rhs
            return theme_toks, loc_toks, pred

        else:
            # Postposition: [Theme NP] [Location NP] [Postposition] [Copula]
            adp_idx = -1
            pred = None
            for i, t in enumerate(tokens):
                t_low = t.lower()
                if t_low in adpositions:
                    adp_idx = i
                    pred = adpositions[t_low]
                    break

            if adp_idx <= 0:
                return [], [], None

            # In postpositional SOV, the noun directly preceding the postposition is the location
            lhs = tokens[:adp_idx]
            if len(lhs) >= 2:
                theme_toks = lhs[:-1]
                loc_toks = [lhs[-1]]
            else:
                theme_toks = lhs
                loc_toks = []
            return theme_toks, loc_toks, pred

    def _apply_stemming(self, token: str) -> str:
        t_low = token.lower()
        for rule in self.profile.morphology.stemming_rules:
            if len(rule) == 2:
                suf, rep = rule
                if t_low.endswith(suf) and len(t_low) > len(suf):
                    return t_low[: -len(suf)] + rep
        for suf in self.profile.morphology.strip_suffixes:
            if t_low.endswith(suf) and len(t_low) > len(suf):
                return t_low[: -len(suf)]
        return t_low

    def _extract_entity_ref(self, tokens: list[str]) -> EntityReference | None:
        if not tokens:
            return None

        specifiers = self.profile.closed_class.specifiers
        spec_type = "definite" if self.profile.typology.has_articles else "generic"

        clean_words: list[str] = []
        for t in tokens:
            t_low = t.lower()
            if t_low in specifiers:
                spec_type = specifiers[t_low]
            else:
                clean_words.append(t)

        if not clean_words:
            return None

        # 1. Check attached lexicon if available (e.g. SinhalaLexicon / TamilLexicon)
        if self.lexicon is not None and hasattr(self.lexicon, "lookup"):
            full_phrase = " ".join(tokens)
            stemmed_phrase = " ".join(self._apply_stemming(t) for t in tokens)
            for cand_phrase in (full_phrase, stemmed_phrase):
                phrase_entries = self.lexicon.lookup(cand_phrase)
                if phrase_entries and "noun" in str(getattr(phrase_entries[0], "pos", "")).lower():
                    return EntityReference(
                        concept_name=phrase_entries[0].semantic_predicate,
                        properties=getattr(phrase_entries[0], "properties", {}) or {},
                        specifier=spec_type,
                        raw_text=full_phrase,
                    )

            lex_props: dict[str, Any] = {}
            lex_concept: str | None = None
            i = 0
            while i < len(clean_words):
                matched = False
                if i + 1 < len(clean_words):
                    bg_raw = f"{clean_words[i]} {clean_words[i + 1]}"
                    bg_stem = f"{clean_words[i]} {self._apply_stemming(clean_words[i + 1])}"
                    for bg in (bg_raw, bg_stem):
                        entries = self.lexicon.lookup(bg)
                        if entries and "noun" in str(getattr(entries[0], "pos", "")).lower():
                            lex_concept = entries[0].semantic_predicate
                            matched = True
                            i += 2
                            break
                    if matched:
                        continue
                w = clean_words[i]
                w_stem = self._apply_stemming(w)
                entries = self.lexicon.lookup(w) or self.lexicon.lookup(w_stem)
                if entries:
                    e = entries[0]
                    p_str = str(getattr(e, "pos", "")).lower()
                    if "noun" in p_str:
                        lex_concept = e.semantic_predicate
                    elif "adj" in p_str or hasattr(e, "properties"):
                        lex_props.update(getattr(e, "properties", {}) or {})
                        if e.semantic_predicate and "color" not in lex_props:
                            lex_props["color"] = e.semantic_predicate
                i += 1

            if lex_concept:
                return EntityReference(
                    concept_name=lex_concept,
                    properties=lex_props,
                    specifier=spec_type,
                    raw_text=" ".join(tokens),
                )

        # 2. Known colors & declarative concept map
        inv_concept = {v.lower(): k for k, v in self.profile.closed_class.concept_map.items()}
        inv_color = {v.lower(): k for k, v in self.profile.closed_class.color_map.items()}

        full_phrase = " ".join(clean_words).lower()
        full_stemmed = " ".join(self._apply_stemming(t) for t in clean_words).lower()
        if full_phrase in inv_concept:
            return EntityReference(concept_name=inv_concept[full_phrase], specifier=spec_type)
        if full_stemmed in inv_concept:
            return EntityReference(concept_name=inv_concept[full_stemmed], specifier=spec_type)

        # Known colors / properties
        color_candidates = {
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "brown",
            "silver",
            "rouge",
            "bleu",
            "vert",
            "jaune",
            "noir",
            "blanc",
            "marron",
            "argent",
            "rot",
            "blau",
            "grün",
            "gelb",
            "schwarz",
            "weiß",
            "braun",
            "රතු",
            "නිල්",
            "කොළ",
            "කහ",
            "කළු",
            "සුදු",
            "சிவப்பு",
            "நீலம்",
            "பச்சை",
            "மஞ்சள்",
            "கருப்பு",
            "வெள்ளை",
        }

        props: dict[str, Any] = {}
        noun_parts: list[str] = []
        mapped_concept: str | None = None

        i = 0
        while i < len(clean_words):
            matched = False
            if i + 1 < len(clean_words):
                bg_raw = f"{clean_words[i]} {clean_words[i + 1]}".lower()
                bg_stem = f"{clean_words[i]} {self._apply_stemming(clean_words[i + 1])}".lower()
                for bg in (bg_raw, bg_stem):
                    if bg in inv_concept:
                        mapped_concept = inv_concept[bg]
                        matched = True
                        i += 2
                        break
                if matched:
                    continue

            w = clean_words[i]
            w_low = w.lower()
            w_stem = self._apply_stemming(w_low)

            if w_low in inv_color:
                props["color"] = inv_color[w_low]
            elif w_stem in inv_color:
                props["color"] = inv_color[w_stem]
            elif w_low in color_candidates:
                props["color"] = w_low
            elif w_stem in color_candidates:
                props["color"] = w_stem
            elif w_low in inv_concept:
                mapped_concept = inv_concept[w_low]
            elif w_stem in inv_concept:
                mapped_concept = inv_concept[w_stem]
            else:
                noun_parts.append(w_low)
            i += 1

        concept_name = mapped_concept or (
            " ".join(noun_parts) if noun_parts else " ".join(clean_words)
        )
        return EntityReference(
            concept_name=concept_name,
            specifier=spec_type,
            properties=props,
        )


class UniversalLanguageRealizer:
    """Language-agnostic realizer verbalizing epistemic states using a LanguageProfile."""

    def __init__(
        self,
        profile: LanguageProfile,
        epistemic_policy: EpistemicRealizationPolicy | None = None,
    ) -> None:
        self.profile = profile
        self.policy = epistemic_policy or EpistemicRealizationPolicy()

    def realize(
        self,
        epistemic_state: CognitiveEpistemicState,
        original_frame: SemanticFrame | None = None,
    ) -> str:
        """Verbalize CognitiveEpistemicState into natural language text."""
        from hbllm.hcir.language.core.epistemic_policy import EpistemicVerbalizationLevel

        level = self.policy.evaluate(epistemic_state)
        subj = epistemic_state.target_subject
        obj = epistemic_state.target_object or ""
        pred = epistemic_state.target_predicate
        target = (
            original_frame.query_target
            if original_frame and original_frame.query_target
            else ("location" if obj else "definition")
        )

        subj_str = self.profile.closed_class.concept_map.get(subj, subj)
        obj_str = self.profile.closed_class.concept_map.get(obj, obj)
        adp_word = self._get_adposition_for_predicate(pred, location=obj_str or obj)

        # 0. Discourse explanation / long answers (definition, causes, requires)
        if pred in ("definition", "causes", "requires") and obj:
            if ("." in obj or "\n" in obj) and len(obj) > 30:
                return obj
            if target == "cause":
                tmpl = self.profile.templates.cause_template
                return tmpl.format(effect=subj_str, cause=obj_str)
            elif target == "procedure":
                tmpl = self.profile.templates.procedure_template
                return tmpl.format(target=subj_str, step=obj_str)
            else:
                tmpl = self.profile.templates.definition_template
                return tmpl.format(subject=subj_str, definition=obj_str)

        # 1. Contradiction / Insufficient Evidence
        if level == EpistemicVerbalizationLevel.CONTRADICTED:
            return self.profile.templates.contradicted_evidence.format(theme=subj_str)

        if (
            level == EpistemicVerbalizationLevel.INSUFFICIENT_EVIDENCE
            or not epistemic_state.is_known
        ):
            tmpl = (
                self.profile.templates.insufficient_evidence
                if self.profile.templates.insufficient_evidence
                else self.profile.templates.location_unknown
            )
            return tmpl.format(theme=subj_str)

        # 2. Verification queries (Yes / No)
        if target == "verification":
            if epistemic_state.raw_belief_value is True or epistemic_state.is_known:
                if level == EpistemicVerbalizationLevel.PROBABLE:
                    tmpl = self.profile.templates.verification_true_probable
                elif level == EpistemicVerbalizationLevel.PLAUSIBLE:
                    tmpl = self.profile.templates.verification_true_plausible
                else:
                    tmpl = self.profile.templates.verification_true
            else:
                tmpl = self.profile.templates.verification_false
            return tmpl.format(theme=subj_str, adposition=adp_word, location=obj_str)

        # 3. Location queries
        if level == EpistemicVerbalizationLevel.PROBABLE:
            tmpl = self.profile.templates.location_probable
        elif level == EpistemicVerbalizationLevel.PLAUSIBLE:
            tmpl = self.profile.templates.location_plausible
        else:
            tmpl = self.profile.templates.location_known
        return tmpl.format(theme=subj_str, adposition=adp_word, location=obj_str)

    def realize_frame(self, frame: SemanticFrame) -> str:
        """Verbalize a SemanticFrame directly (e.g. for interlingual translation)."""
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        loc_ref = frame.get_role(ThematicRole.LOCATION) or frame.get_role(ThematicRole.DESTINATION)
        theme_name: str = (
            (theme_ref.concept_name or "entity")
            if theme_ref and theme_ref.concept_name
            else "entity"
        )
        loc_name: str = (
            (loc_ref.concept_name or "location") if loc_ref and loc_ref.concept_name else "location"
        )

        concept_map = self.profile.closed_class.concept_map
        color_map = self.profile.closed_class.color_map

        theme_surface = concept_map.get(theme_name, theme_name)
        loc_surface = concept_map.get(loc_name, loc_name)

        if theme_ref and theme_ref.properties.get("color"):
            color_val = str(theme_ref.properties["color"])
            color_surface = color_map.get(color_val, color_val)
            theme_with_color = f"{color_surface} {theme_surface}".strip()
        else:
            theme_with_color = theme_surface

        if self.profile.typology.has_articles:
            spec_word = next(iter(self.profile.closed_class.specifiers.keys()), "the")
            loc_lower = self.profile.templates.location_known.lower()
            if not any(
                loc_lower.startswith(art)
                for art in ("the ", "le ", "la ", "l'", "der ", "die ", "das ")
            ):
                theme_with_color = f"{spec_word} {theme_with_color}".strip()
                loc_surface = f"{spec_word} {loc_surface}".strip()

        if frame.frame_type == FrameType.QUERY:
            if frame.query_target == "location":
                # Find query marker for location
                marker = "Where"
                for m, spec in self.profile.closed_class.question_markers.items():
                    if spec.get("target") == "location":
                        marker = m
                        break
                if self.profile.typology.wh_movement == "front":
                    cop = (
                        self.profile.closed_class.copula_forms[0]
                        if self.profile.closed_class.copula_forms
                        else "is"
                    )
                    marker_cap = marker.capitalize()
                    return f"{marker_cap} {cop} {theme_with_color}?"
                else:
                    return f"{theme_with_color} {marker}?"

        # Default spatial assertion realization
        rel_pred = frame.predicate or "located_on"
        adp_word = self._get_adposition_for_predicate(rel_pred, location=loc_surface)
        tmpl = self.profile.templates.location_known
        return tmpl.format(theme=theme_with_color, adposition=adp_word, location=loc_surface)

    def _get_adposition_for_predicate(self, predicate: str, location: str = "") -> str:
        effective_pred = predicate
        if predicate in ("located_in", "in") and location:
            loc_low = location.lower().strip()
            # Physical support surface vs container: flat support surfaces use "on"
            surface_markers = (
                "table",
                "shelf",
                "floor",
                "desk",
                "chair",
                "counter",
                "මේසය",
                "මේසේ",
                "රාක්කය",
                "පුටුව",
                "බිම",
                "மேசை",
                "மேசையின்",
                "நாற்காலி",
                "தரை",
            )
            if any(surf in loc_low for surf in surface_markers):
                effective_pred = "located_on"

        for adp, pred in self.profile.closed_class.adpositions.items():
            if pred == effective_pred:
                return adp
        for adp, pred in self.profile.closed_class.adpositions.items():
            if pred == predicate:
                return adp
        # Fallback to first available adposition or predicate
        if self.profile.closed_class.adpositions:
            return next(iter(self.profile.closed_class.adpositions.keys()))
        return effective_pred
