"""English Semantic Parser for A16.

Converts English parse tree AST nodes into language-neutral SemanticFrames.
Implements the LanguageParser protocol.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from hbllm.hcir.language.core.discourse import DiscourseSegmenter
from hbllm.hcir.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    LanguageErrorType,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.language.english.lexicon import EnglishLexicon, EnglishPOS
from hbllm.hcir.language.english.morphology import EnglishMorphology, MorphToken
from hbllm.hcir.language.english.normalizer import (
    COMMAND_VERB_VOCAB,
    strip_conversational_wrappers,
)
from hbllm.hcir.language.english.syntax import (
    ConstructionType,
    EnglishSyntaxParser,
    NounPhrase,
    SentenceNode,
)

logger = logging.getLogger(__name__)


_PRIORITY_IMPERATIVE_VERBS: frozenset[str] = frozenset(
    {
        "why",
        "how",
        "tell",
        "remember",
        "recall",
        "forget",
        "explain",
        "describe",
        "define",
        "summarize",
        "search",
        "websearch",
        "google",
        "fetch",
        "browse",
    }
)


class EnglishParser:
    """Parses English utterances into language-neutral SemanticFrames."""

    def __init__(
        self,
        lexicon: EnglishLexicon | None = None,
        morphology: EnglishMorphology | None = None,
        syntax: EnglishSyntaxParser | None = None,
    ) -> None:
        self._lexicon = lexicon or EnglishLexicon()
        self._morphology = morphology or EnglishMorphology(self._lexicon)
        self._syntax = syntax or EnglishSyntaxParser()

    def set_dynamic_vocabulary(self, words: Iterable[str]) -> None:
        """Update active world/persistent entity & concept vocabulary for typo recovery."""
        self._morphology._normalizer.set_dynamic_vocabulary(words)

    def parse(self, text: str) -> SemanticFrame:
        """Parse raw English text into a language-neutral SemanticFrame."""
        meta = LanguageMetadata(language="en", raw_text=text)
        cleaned_text, _ = strip_conversational_wrappers(text)

        # 1. Morphological analysis, tokenization & typo normalization
        tokens = self._morphology.lemmatize_sequence(cleaned_text or text)
        if not tokens:
            return SemanticFrame(
                frame_type=FrameType.ERROR,
                error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
                error_detail="Empty utterance.",
                metadata=meta,
            )

        # Priority check for specialized info/memory/web imperative commands
        # (prevents generic syntax parser from truncating "tell me about", "remember that", etc.)
        content = [t for t in tokens if t.pos != EnglishPOS.PUNCT]
        if content:
            first_lemma = content[0].lemma
            is_copular_noun = len(content) >= 2 and content[1].pos == EnglishPOS.AUX
            if first_lemma in _PRIORITY_IMPERATIVE_VERBS and (
                not is_copular_noun or first_lemma in ("why", "how")
            ):
                imp_priority = self._try_parse_imperative_command(tokens, meta)
                if imp_priority is not None:
                    return imp_priority

        # 2. Strict syntactic parsing
        ast = self._syntax.parse(tokens)
        if not ast:
            imp_frame = self._try_parse_imperative_command(tokens, meta)
            if imp_frame is not None:
                return imp_frame
            rec_frame = self._try_recover_ungrammatical(tokens, meta)
            if rec_frame is not None:
                return rec_frame
            return SemanticFrame(
                frame_type=FrameType.ERROR,
                error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
                error_detail=f"Syntactic parsing failed for: '{text}'",
                metadata=meta,
            )

        # 3. Semantic translation into language-neutral frame
        try:
            return self._ast_to_semantic_frame(ast, meta)
        except Exception as e:
            logger.debug("EnglishParser error in semantic mapping: %s", e)
            return SemanticFrame(
                frame_type=FrameType.ERROR,
                error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
                error_detail=str(e),
                metadata=meta,
            )

    def parse_discourse(self, text: str) -> list[SemanticFrame]:
        """Parse multi-sentence / multi-clause English discourse into ordered SemanticFrames."""
        clauses = DiscourseSegmenter.segment(text)
        return [self.parse(clause) for clause in clauses]

    def _try_parse_imperative_command(
        self, tokens: list[Any], meta: LanguageMetadata
    ) -> SemanticFrame | None:
        """Parse concise imperative tool/action commands (e.g. 'Read the file', 'Verify code')."""
        content = [t for t in tokens if t.pos != EnglishPOS.PUNCT]
        if len(content) < 2:
            return None
        first = content[0]
        # Reject immediately if followed by an auxiliary/verb ("Focus is ...", "Plants absorb ...")
        # or if first token is inflected (plural/3s/past/progressive) or typo-guessed in long prose.
        if first.lemma not in ("why", "how"):
            if (
                first.features.get("number") == "plural"
                or first.features.get("person") == "3s"
                or first.features.get("tense") == "past"
                or first.features.get("aspect") == "progressive"
            ):
                return None
            if first.features.get("corrected_from") and len(content) > 6:
                return None
            if content[1].pos in (EnglishPOS.AUX, EnglishPOS.VERB):
                return None
            if (
                first.lemma not in _PRIORITY_IMPERATIVE_VERBS
                and "noun" in first.features.get("alt_pos", "")
                and any(t.pos in (EnglishPOS.AUX, EnglishPOS.VERB) for t in content[1:])
            ):
                return None

        _KNOWN_COMMAND_VERBS = (
            frozenset(
                {
                    "explain",
                    "describe",
                    "define",
                    "tell",
                    "detail",
                    "summarize",
                    "why",
                    "how",
                    "read",
                    "show",
                    "display",
                    "list",
                    "search",
                    "find",
                    "verify",
                    "check",
                    "analyze",
                    "remember",
                    "store",
                    "recall",
                    "forget",
                    "ingest",
                    "fetch",
                    "browse",
                    "google",
                    "websearch",
                }
            )
            | COMMAND_VERB_VOCAB
        )
        is_verb = first.pos == EnglishPOS.VERB or first.lemma in _KNOWN_COMMAND_VERBS
        if not is_verb:
            return None

        rest = [t for t in content[1:] if t.pos != EnglishPOS.DET]
        verb_pred = first.lemma

        # Skip beneficiary pronouns ("list me files", "read for me README.md")
        if verb_pred in {
            "list",
            "read",
            "show",
            "display",
            "find",
            "search",
            "explain",
            "describe",
            "define",
            "fetch",
            "get",
        }:
            if len(rest) > 2 and rest[0].lemma == "for" and rest[1].lemma in ("me", "us"):
                rest = rest[2:]
            elif len(rest) > 1 and rest[0].lemma in ("me", "us"):
                rest = rest[1:]

        # Normalize "define X" -> predicate="explain"
        if verb_pred == "define":
            verb_pred = "explain"

        # Normalize "explain why X" -> predicate="why", "explain how X" -> predicate="how"
        if verb_pred == "explain" and len(rest) > 1 and rest[0].lemma in ("why", "how"):
            verb_pred = rest[0].lemma
            rest = rest[1:]

        # Normalize "tell me about X" -> predicate="explain", target="X"
        if verb_pred == "tell":
            about_idx = -1
            for idx, t in enumerate(content):
                if t.lemma == "about":
                    about_idx = idx
                    break
            if about_idx != -1 and about_idx + 1 < len(content):
                verb_pred = "explain"
                rest = [t for t in content[about_idx + 1 :] if t.pos != EnglishPOS.DET]

        # Strip stray auxiliaries in "why does X happen?" / "why X is happen?" / "how do they work?"
        _STRAY_AUX = frozenset(
            {"be", "do", "does", "did", "is", "are", "was", "were", "can", "will", "to"}
        )
        if verb_pred in ("why", "how"):
            filtered_rest = [t for t in rest if t.lemma not in _STRAY_AUX]
            if filtered_rest:
                rest = filtered_rest

        # Strip trailing functional verbs in "how X works" or "why X occurs"
        _TRAILING_VERBS = frozenset(
            {
                "work",
                "works",
                "occur",
                "occurs",
                "happen",
                "happens",
                "operate",
                "operates",
                "function",
                "functions",
            }
        )
        if rest and rest[-1].lemma in _TRAILING_VERBS:
            rest = rest[:-1]

        if not rest:
            return None
        head = rest[-1]
        props: dict[str, Any] = {}
        if len(rest) > 1:
            props["modifier"] = " ".join(t.lemma for t in rest[:-1])
            props["head"] = head.lemma

        is_anaphoric = head.lemma in {"it", "this", "that", "they", "them", "these", "those"}
        if is_anaphoric:
            head_name = ""
        elif verb_pred in ("why", "how") and len(rest) > 1:
            head_name = " ".join(t.lemma for t in rest)
        elif head.features.get("is_path") == "true":
            head_name = head.surface
        else:
            head_name = head.lemma
        patient = EntityReference(
            concept_name=head_name,
            specifier="anaphoric" if is_anaphoric else "generic",
            raw_text=" ".join(t.surface for t in rest)
            if verb_pred in ("why", "how")
            else head.surface,
            properties=props,
        )
        frame = SemanticFrame(
            frame_type=FrameType.COMMAND,
            predicate=verb_pred,
            metadata=meta,
        )
        frame.set_role(ThematicRole.PATIENT, patient)
        return frame

    def _attach_pps(
        self,
        frame: SemanticFrame,
        pps: list[Any],
        *,
        verb_lemma: str = "",
    ) -> None:
        """Attach prepositional phrases using VerbNet/PropBank thematic roles when known."""
        if not pps:
            return
        instrument_preps = {"through", "with", "by", "via", "using"}
        source_preps = {"from", "out"}
        spatial_motion_verbs = {"move", "push", "put", "roll", "place", "drop"}

        verb_roles: set[str] = set()
        if verb_lemma:
            for entry in self._lexicon.lookup(verb_lemma):
                for r in entry.properties.get("thematic_roles", []) or ():
                    verb_roles.add(str(r).strip().lower())
            if not verb_roles:
                store = getattr(self._lexicon, "_ontology_store", None)
                if store is not None:
                    for stored in store.lookup(verb_lemma, language="en"):
                        for r in getattr(stored, "thematic_roles", ()) or ():
                            verb_roles.add(str(r).strip().lower())

        for pp in pps:
            ref = self._np_to_reference(pp.noun_phrase)
            if ref is None:
                continue
            prep_lem = pp.preposition.lemma
            enriched = EntityReference(
                concept_name=ref.concept_name,
                properties={**ref.properties, "prep": prep_lem},
                specifier=ref.specifier,
                raw_text=ref.raw_text,
                discourse_id=ref.discourse_id,
            )
            if prep_lem in instrument_preps:
                if frame.get_role(ThematicRole.INSTRUMENT) is None:
                    frame.set_role(ThematicRole.INSTRUMENT, enriched)
            elif prep_lem in source_preps and frame.get_role(ThematicRole.SOURCE) is None:
                frame.set_role(ThematicRole.SOURCE, enriched)
            elif (
                prep_lem in {"to", "for"}
                and (verb_roles & {"recipient", "beneficiary"})
                and verb_lemma not in spatial_motion_verbs
                and frame.get_role(ThematicRole.RECIPIENT) is None
            ):
                frame.set_role(ThematicRole.RECIPIENT, enriched)
            else:
                if frame.get_role(ThematicRole.LOCATION) is None:
                    frame.set_role(ThematicRole.LOCATION, enriched)

    # ── AST to SemanticFrame Translation ──────────────────────────────

    def _ast_to_semantic_frame(self, ast: SentenceNode, meta: LanguageMetadata) -> SemanticFrame:
        """Map typed AST node to language-neutral SemanticFrame."""
        if ast.verb_phrase and ast.verb_phrase.head_verb:
            tense = ast.verb_phrase.head_verb.features.get("tense")
            if not tense and ast.auxiliary is not None:
                tense = ast.auxiliary.features.get("tense")
            if tense:
                meta.tense = tense
        meta.polarity = not ast.is_negated

        # 1. WH-QUESTIONS ("Where is the ball?", "What do florbs eat?", "Who was tired?")
        if ast.construction == ConstructionType.WH_QUESTION:
            subject_ref = self._np_to_reference(ast.subject)
            wh_word = ast.wh_word.lemma if ast.wh_word else "what"

            if wh_word == "where":
                if (
                    ast.verb_phrase
                    and ast.verb_phrase.head_verb.pos == EnglishPOS.VERB
                    and ast.verb_phrase.head_verb.lemma != "be"
                ):
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate=ast.verb_phrase.head_verb.lemma,
                        query_target="svo_location",
                        metadata=meta,
                    )
                    if subject_ref:
                        frame.set_role(ThematicRole.THEME, subject_ref)
                    return frame
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="located_on",
                    query_target="location",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                return frame

            elif wh_word == "who":
                if subject_ref is None and ast.verb_phrase:
                    if ast.verb_phrase.adjective_complement is not None:
                        adj_tok = ast.verb_phrase.adjective_complement
                        adj_props = self._get_properties_for_adj(adj_tok.lemma)
                        frame = SemanticFrame(
                            frame_type=FrameType.QUERY,
                            predicate="has_property",
                            query_target="svo_subject",
                            metadata=meta,
                        )
                        frame.set_role(
                            ThematicRole.PATIENT,
                            EntityReference(
                                concept_name=adj_tok.lemma,
                                properties=adj_props,
                                specifier="generic",
                                raw_text=adj_tok.surface,
                            ),
                        )
                        return frame
                    if ast.verb_phrase.prepositional_phrases:
                        pp = ast.verb_phrase.prepositional_phrases[0]
                        target_ref = self._np_to_reference(pp.noun_phrase)
                        prep_pred = self._get_predicate_for_prep(pp.preposition.lemma)
                        frame = SemanticFrame(
                            frame_type=FrameType.QUERY,
                            predicate=prep_pred,
                            query_target="svo_subject",
                            metadata=meta,
                        )
                        if target_ref:
                            frame.set_role(ThematicRole.LOCATION, target_ref)
                        return frame
                    if ast.verb_phrase.head_verb.lemma == "be" and ast.verb_phrase.direct_object:
                        pred_ref = self._np_to_reference(ast.verb_phrase.direct_object)
                        frame = SemanticFrame(
                            frame_type=FrameType.QUERY,
                            predicate="is_a",
                            query_target="svo_subject",
                            metadata=meta,
                        )
                        if pred_ref:
                            frame.set_role(ThematicRole.PATIENT, pred_ref)
                        return frame
                    if (
                        ast.verb_phrase.head_verb.pos == EnglishPOS.VERB
                        and ast.verb_phrase.head_verb.lemma != "be"
                    ):
                        obj_ref = (
                            self._np_to_reference(ast.verb_phrase.direct_object)
                            if ast.verb_phrase.direct_object
                            else None
                        )
                        frame = SemanticFrame(
                            frame_type=FrameType.QUERY,
                            predicate=ast.verb_phrase.head_verb.lemma,
                            query_target="svo_subject",
                            metadata=meta,
                        )
                        if obj_ref:
                            frame.set_role(ThematicRole.PATIENT, obj_ref)
                        self._attach_pps(frame, ast.verb_phrase.prepositional_phrases)
                        return frame

                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="identity",
                    query_target="identity",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                return frame

            elif wh_word == "what":
                if ast.verb_phrase and ast.verb_phrase.head_verb.lemma in (
                    "cause",
                    "trigger",
                    "lead",
                ):
                    obj_ref = (
                        self._np_to_reference(ast.verb_phrase.direct_object)
                        if ast.verb_phrase.direct_object
                        else None
                    )
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate="causes",
                        query_target="cause",
                        metadata=meta,
                    )
                    if obj_ref:
                        frame.set_role(ThematicRole.PATIENT, obj_ref)
                    return frame

                if ast.verb_phrase and ast.verb_phrase.head_verb.lemma in ("require", "need"):
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate="requires",
                        query_target="procedure",
                        metadata=meta,
                    )
                    if subject_ref:
                        frame.set_role(ThematicRole.THEME, subject_ref)
                    return frame

                if ast.verb_phrase and ast.verb_phrase.head_verb.lemma in (
                    "enable",
                    "allow",
                    "permit",
                ):
                    obj_ref = (
                        self._np_to_reference(ast.verb_phrase.direct_object)
                        if ast.verb_phrase.direct_object
                        else None
                    )
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate="enables",
                        query_target="enables",
                        metadata=meta,
                    )
                    if subject_ref:
                        frame.set_role(ThematicRole.THEME, subject_ref)
                    if obj_ref:
                        frame.set_role(ThematicRole.PATIENT, obj_ref)
                    return frame

                if (
                    ast.verb_phrase
                    and ast.verb_phrase.head_verb.pos == EnglishPOS.VERB
                    and ast.verb_phrase.head_verb.lemma != "be"
                ):
                    verb_lem = ast.verb_phrase.head_verb.lemma
                    if subject_ref is not None:
                        frame = SemanticFrame(
                            frame_type=FrameType.QUERY,
                            predicate=verb_lem,
                            query_target="svo_object",
                            metadata=meta,
                        )
                        frame.set_role(ThematicRole.THEME, subject_ref)
                        self._attach_pps(frame, ast.verb_phrase.prepositional_phrases)
                        return frame
                    obj_ref = (
                        self._np_to_reference(ast.verb_phrase.direct_object)
                        if ast.verb_phrase.direct_object
                        else None
                    )
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate=verb_lem,
                        query_target="svo_subject",
                        metadata=meta,
                    )
                    if obj_ref:
                        frame.set_role(ThematicRole.PATIENT, obj_ref)
                    self._attach_pps(frame, ast.verb_phrase.prepositional_phrases)
                    return frame

                prop_name = ast.wh_property.lemma if ast.wh_property else "property"
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate=f"property_{prop_name}",
                    query_target="property",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                return frame

            elif wh_word == "why":
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="causes",
                    query_target="cause",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                return frame

            elif wh_word == "how":
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="requires",
                    query_target="procedure",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                return frame

        # 2. YES/NO QUESTIONS ("Is the ball on the table?", "Is a drell warm-blooded?", "Do florbs eat meat?")
        elif ast.construction == ConstructionType.YES_NO_QUESTION:
            subject_ref = self._np_to_reference(ast.subject)

            # Spatial query: "Is the ball on the table?"
            if (
                ast.verb_phrase
                and ast.verb_phrase.prepositional_phrases
                and (
                    ast.verb_phrase.head_verb.pos != EnglishPOS.VERB
                    or ast.verb_phrase.head_verb.lemma == "be"
                )
            ):
                pp = ast.verb_phrase.prepositional_phrases[0]
                target_ref = self._np_to_reference(pp.noun_phrase)
                prep_pred = self._get_predicate_for_prep(pp.preposition.lemma)

                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate=prep_pred,
                    query_target="verification",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                if target_ref:
                    frame.set_role(ThematicRole.LOCATION, target_ref)
                return frame

            # Property verification: "Is the ball red?", "Is a drell warm-blooded?"
            if ast.verb_phrase and ast.verb_phrase.adjective_complement:
                adj_tok = ast.verb_phrase.adjective_complement
                adj_props = self._get_properties_for_adj(adj_tok.lemma)
                prop_key = list(adj_props.keys())[0] if adj_props else "property"

                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate=f"property_{prop_key}",
                    query_target="verification",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                frame.set_role(
                    ThematicRole.PATIENT,
                    EntityReference(
                        concept_name=adj_tok.lemma,
                        properties=adj_props,
                        specifier="generic",
                        raw_text=adj_tok.surface,
                    ),
                )
                return frame

            # Causal verification: "Does overflow cause crash?"
            if ast.verb_phrase and ast.verb_phrase.head_verb.lemma in ("cause", "trigger", "lead"):
                obj_ref = (
                    self._np_to_reference(ast.verb_phrase.direct_object)
                    if ast.verb_phrase.direct_object
                    else None
                )
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="causes",
                    query_target="verification",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                if obj_ref:
                    frame.set_role(ThematicRole.PATIENT, obj_ref)
                return frame

            # Requirement verification: "Does encryption require a key?"
            if ast.verb_phrase and ast.verb_phrase.head_verb.lemma in (
                "require",
                "depend",
                "need",
            ):
                obj_ref = (
                    self._np_to_reference(ast.verb_phrase.direct_object)
                    if ast.verb_phrase.direct_object
                    else None
                )
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="requires",
                    query_target="verification",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                if obj_ref:
                    frame.set_role(ThematicRole.PATIENT, obj_ref)
                return frame

            # Enablement verification: "Does gravity enable orbit?"
            if ast.verb_phrase and ast.verb_phrase.head_verb.lemma in (
                "enable",
                "allow",
                "permit",
            ):
                obj_ref = (
                    self._np_to_reference(ast.verb_phrase.direct_object)
                    if ast.verb_phrase.direct_object
                    else None
                )
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="enables",
                    query_target="verification",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                if obj_ref:
                    frame.set_role(ThematicRole.PATIENT, obj_ref)
                return frame

            # General SVO verb verification: "Do florbs eat meat?", "Do florbs fly?", "Do florbs live in caves?"
            if (
                ast.verb_phrase
                and ast.verb_phrase.head_verb.pos == EnglishPOS.VERB
                and ast.verb_phrase.head_verb.lemma != "be"
            ):
                obj_ref = (
                    self._np_to_reference(ast.verb_phrase.direct_object)
                    if ast.verb_phrase.direct_object
                    else None
                )
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate=ast.verb_phrase.head_verb.lemma,
                    query_target="verification",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                if obj_ref:
                    frame.set_role(ThematicRole.PATIENT, obj_ref)
                self._attach_pps(frame, ast.verb_phrase.prepositional_phrases)
                return frame

            # Nominal verification: "Is Pluto a planet?"
            if ast.verb_phrase and ast.verb_phrase.direct_object:
                pred_ref = self._np_to_reference(ast.verb_phrase.direct_object)
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="is_a",
                    query_target="verification",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                if pred_ref:
                    frame.set_role(ThematicRole.PATIENT, pred_ref)
                return frame

        # 3. IMPERATIVE COMMANDS ("Move the red ball to the table", "Push the box")
        elif ast.construction == ConstructionType.IMPERATIVE:
            verb_pred = ast.verb_phrase.head_verb.lemma if ast.verb_phrase else "act"
            patient_ref = (
                self._np_to_reference(ast.verb_phrase.direct_object) if ast.verb_phrase else None
            )

            frame = SemanticFrame(
                frame_type=FrameType.COMMAND,
                predicate=verb_pred,
                metadata=meta,
            )
            if patient_ref:
                frame.set_role(ThematicRole.PATIENT, patient_ref)

            if ast.verb_phrase and ast.verb_phrase.prepositional_phrases:
                pp = ast.verb_phrase.prepositional_phrases[0]
                dest_ref = self._np_to_reference(pp.noun_phrase)
                if dest_ref:
                    frame.set_role(ThematicRole.DESTINATION, dest_ref)

            return frame

        # 4. DECLARATIVE / COPULAR ASSERTIONS ("The red ball is on the table", "The ball is red")
        elif ast.construction in (
            ConstructionType.DECLARATIVE,
            ConstructionType.COPULAR,
            ConstructionType.EXISTENTIAL,
        ):
            subject_ref = self._np_to_reference(ast.subject)

            # Spatial assertion: "The ball is on the table"
            if (
                ast.verb_phrase
                and ast.verb_phrase.prepositional_phrases
                and (
                    ast.verb_phrase.head_verb.pos != EnglishPOS.VERB
                    or ast.verb_phrase.head_verb.lemma == "be"
                )
            ):
                pp = ast.verb_phrase.prepositional_phrases[0]
                target_ref = self._np_to_reference(pp.noun_phrase)
                prep_pred = self._get_predicate_for_prep(pp.preposition.lemma)

                frame = SemanticFrame(
                    frame_type=FrameType.ASSERTION,
                    predicate=prep_pred,
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                if target_ref:
                    frame.set_role(ThematicRole.LOCATION, target_ref)
                return frame

            # Copular property assertion: "The box is red", "Mira was tired", "A drell is warm-blooded"
            if ast.verb_phrase and ast.verb_phrase.adjective_complement:
                adj_tok = ast.verb_phrase.adjective_complement
                adj_props = self._get_properties_for_adj(adj_tok.lemma)
                prop_ref = EntityReference(
                    concept_name=adj_tok.lemma,
                    properties=adj_props,
                    raw_text=adj_tok.surface,
                    specifier="generic",
                )

                frame = SemanticFrame(
                    frame_type=FrameType.ASSERTION,
                    predicate="is_property",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                frame.set_role(ThematicRole.LOCATION, prop_ref)
                return frame

            # Nominal copular assertion: "Focus is a cognitive process"
            if (
                ast.verb_phrase
                and ast.verb_phrase.direct_object
                and ast.verb_phrase.head_verb.lemma == "be"
            ):
                pred_ref = self._np_to_reference(ast.verb_phrase.direct_object)
                meta.polarity = not ast.is_negated
                frame = SemanticFrame(
                    frame_type=FrameType.ASSERTION,
                    predicate="is_a",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                if pred_ref:
                    frame.set_role(ThematicRole.PATIENT, pred_ref)
                return frame

            # SVO action assertion: "The robot pushed the box", "Florbs live in caves",
            # "Heavy rain caused the river to flood", "Heat does not cause rust"
            if ast.verb_phrase and ast.verb_phrase.head_verb:
                verb_lemma = ast.verb_phrase.head_verb.lemma
                meta.polarity = not ast.is_negated

                # Normalize canonical relation predicates
                if verb_lemma in ("cause", "trigger", "lead"):
                    verb_pred = "causes"
                elif verb_lemma in ("enable", "allow", "permit"):
                    verb_pred = "enables"
                elif verb_lemma in ("require", "need", "depend"):
                    verb_pred = "requires"
                else:
                    verb_pred = verb_lemma

                obj_ref = self._np_to_reference(ast.verb_phrase.direct_object)
                rec_ref = (
                    self._np_to_reference(ast.verb_phrase.indirect_object)
                    if ast.verb_phrase.indirect_object is not None
                    else None
                )
                if obj_ref is not None and ast.verb_phrase.infinitive_verb is not None:
                    obj_ref = EntityReference(
                        concept_name=obj_ref.concept_name,
                        properties={
                            **obj_ref.properties,
                            "infinitive_verb": ast.verb_phrase.infinitive_verb.lemma,
                        },
                        specifier=obj_ref.specifier,
                        raw_text=obj_ref.raw_text,
                        discourse_id=obj_ref.discourse_id,
                    )

                frame = SemanticFrame(
                    frame_type=FrameType.ASSERTION,
                    predicate=verb_pred,
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.AGENT, subject_ref)
                if rec_ref:
                    frame.set_role(ThematicRole.RECIPIENT, rec_ref)
                if obj_ref:
                    frame.set_role(ThematicRole.PATIENT, obj_ref)
                self._attach_pps(
                    frame,
                    ast.verb_phrase.prepositional_phrases,
                    verb_lemma=verb_lemma,
                )
                return frame

        # Fallback error
        return SemanticFrame(
            frame_type=FrameType.ERROR,
            error_type=LanguageErrorType.UNSUPPORTED_CONSTRUCTION,
            error_detail=f"Unsupported sentence construction: {ast.construction}",
            metadata=meta,
        )

    def _try_recover_ungrammatical(
        self,
        tokens: list[MorphToken],
        meta: LanguageMetadata,
    ) -> SemanticFrame | None:
        """Recover structured intent from telegraphic or mildly ungrammatical utterances.

        Strictly rejects malformed word-salad containing unexpected verbs, auxiliaries,
        or adverbs inside NP slots so ``UNRESOLVED_LANGUAGE`` remains intact.
        """
        clean = [t for t in tokens if t.pos != EnglishPOS.PUNCT]
        if len(clean) < 2:
            return None

        # Pattern 1: Uninverted Wh-Question ("where red ball is?", "what focus is?")
        if clean[0].lemma in ("where", "what"):
            wh_lemma = clean[0].lemma
            rest = list(clean[1:])
            if rest and rest[-1].pos == EnglishPOS.AUX:
                rest = rest[:-1]
            wh_prop: str | None = None
            if (
                wh_lemma == "what"
                and rest
                and rest[0].lemma in ("color", "location", "shape", "size")
            ):
                wh_prop = rest[0].lemma
                rest = rest[1:]
                if rest and rest[-1].pos == EnglishPOS.AUX:
                    rest = rest[:-1]
            subject_ref = self._parse_relaxed_np(rest)
            if subject_ref is not None:
                if wh_lemma == "where":
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate="located_on",
                        query_target="location",
                        metadata=meta,
                    )
                else:
                    frame = SemanticFrame(
                        frame_type=FrameType.QUERY,
                        predicate=f"property_{wh_prop or 'property'}",
                        query_target="property",
                        metadata=meta,
                    )
                frame.set_role(ThematicRole.THEME, subject_ref)
                return frame

        # Pattern 2: Missing-copula spatial assertion ("the red ball on the table", "ball red on table")
        spatial_prep_idxs = [
            i
            for i, tok in enumerate(clean)
            if tok.pos == EnglishPOS.PREP
            and tok.lemma
            in ("on", "in", "inside", "within", "under", "beneath", "above", "near", "beside")
        ]
        if len(spatial_prep_idxs) == 1:
            p_idx = spatial_prep_idxs[0]
            left = clean[:p_idx]
            right = clean[p_idx + 1 :]
            left_ref = self._parse_relaxed_np(left)
            right_ref = self._parse_relaxed_np(right)
            if left_ref is not None and right_ref is not None:
                prep_pred = self._get_predicate_for_prep(clean[p_idx].lemma)
                frame = SemanticFrame(
                    frame_type=FrameType.ASSERTION,
                    predicate=prep_pred,
                    metadata=meta,
                )
                frame.set_role(ThematicRole.THEME, left_ref)
                frame.set_role(ThematicRole.LOCATION, right_ref)
                return frame

        # Pattern 3: Introductory clause / adverbial lead-in before comma
        comma_indices = [i for i, tok in enumerate(tokens) if tok.lemma == ","]
        if comma_indices:
            c_idx = comma_indices[0]
            prefix = [t for t in tokens[:c_idx] if t.pos != EnglishPOS.PUNCT]
            suffix = [t for t in tokens[c_idx + 1 :] if t.pos != EnglishPOS.PUNCT]
            if prefix and len(suffix) >= 2 and prefix[0].pos in (EnglishPOS.PREP, EnglishPOS.ADV):
                sub_ast = self._syntax.parse(suffix)
                if sub_ast is not None and not (
                    sub_ast.construction == ConstructionType.IMPERATIVE and len(suffix) > 6
                ):
                    return self._ast_to_semantic_frame(sub_ast, meta)

        return None

    def _parse_relaxed_np(self, tokens: list[MorphToken]) -> EntityReference | None:
        """Parse (Det)? (Adj* Noun+ | Noun+ Adj+) strictly consuming all *tokens*."""
        if not tokens:
            return None
        for tok in tokens:
            if tok.pos in (
                EnglishPOS.AUX,
                EnglishPOS.VERB,
                EnglishPOS.ADV,
                EnglishPOS.WH,
                EnglishPOS.PREP,
                EnglishPOS.PUNCT,
            ):
                return None

        if len(tokens) == 1 and tokens[0].pos == EnglishPOS.PRON:
            return EntityReference(specifier="anaphoric", raw_text=tokens[0].surface)

        idx = 0
        det_tok: MorphToken | None = None
        if tokens[idx].pos == EnglishPOS.DET:
            det_tok = tokens[idx]
            idx += 1

        rem = tokens[idx:]
        if not rem:
            return None

        adjs: list[MorphToken] = []
        nouns: list[MorphToken] = []
        # Standard order: Adj* Noun+ OR reversed order: Noun+ Adj+ (only if no leading Adj)
        if rem[0].pos == EnglishPOS.ADJ:
            for tok in rem:
                if tok.pos == EnglishPOS.ADJ and not nouns:
                    adjs.append(tok)
                elif tok.pos == EnglishPOS.NOUN:
                    nouns.append(tok)
                else:
                    return None
        elif rem[0].pos == EnglishPOS.NOUN:
            seen_adj = False
            for tok in rem:
                if tok.pos == EnglishPOS.NOUN and not seen_adj:
                    nouns.append(tok)
                elif tok.pos == EnglishPOS.ADJ:
                    seen_adj = True
                    adjs.append(tok)
                else:
                    return None
        else:
            return None

        if not nouns or len(nouns) > 3:
            return None

        props: dict[str, Any] = {}
        for adj in adjs:
            props.update(self._get_properties_for_adj(adj.lemma))

        if nouns[-1].features.get("is_path") == "true":
            concept_name = nouns[-1].surface
        else:
            concept_name = " ".join(n.lemma for n in nouns)

        specifier = "generic"
        if det_tok:
            if det_tok.lemma in ("a", "an"):
                specifier = "indefinite"
            elif det_tok.lemma in ("this", "that", "these", "those"):
                specifier = "demonstrative"
            else:
                specifier = "definite"

        return EntityReference(
            concept_name=concept_name,
            properties=props,
            specifier=specifier,
            raw_text=" ".join(t.surface for t in tokens),
        )

    # ── Helper Mappers ────────────────────────────────────────────────

    def _np_to_reference(self, np: NounPhrase | None) -> EntityReference | None:
        """Convert a parsed NounPhrase AST into an EntityReference."""
        if not np:
            return None

        # Pronoun reference
        if np.pronoun:
            return EntityReference(
                specifier="anaphoric",
                raw_text=np.pronoun.surface,
            )

        # Specifier (definite vs indefinite vs demonstrative vs bare generic)
        specifier = "generic"
        if np.determiner:
            det_lemma = np.determiner.lemma
            if det_lemma in ("a", "an"):
                specifier = "indefinite"
            elif det_lemma in ("this", "that", "these", "those"):
                specifier = "demonstrative"
            else:
                specifier = "definite"

        # Separate true ADJ property modifiers from compound NOUN modifiers
        props: dict[str, Any] = {}
        compound_mods: list[str] = []
        for mod in np.adjectives:
            if mod.pos == EnglishPOS.NOUN:
                compound_mods.append(mod.lemma)
            else:
                props.update(self._get_properties_for_adj(mod.lemma))

        if np.head_noun:
            if np.head_noun.features.get("is_path") == "true":
                concept_name: str | None = np.head_noun.surface
            elif compound_mods:
                concept_name = " ".join([*compound_mods, np.head_noun.lemma])
            else:
                concept_name = np.head_noun.lemma
        else:
            concept_name = None
        raw_text = " ".join(t.surface for t in np.raw_tokens)

        return EntityReference(
            concept_name=concept_name,
            properties=props,
            specifier=specifier,
            raw_text=raw_text,
        )

    def _get_predicate_for_prep(self, prep_lemma: str) -> str:
        entries = self._lexicon.lookup(prep_lemma)
        for e in entries:
            if e.pos == EnglishPOS.PREP and e.semantic_predicate:
                return e.semantic_predicate
        return f"located_{prep_lemma}"

    def _get_properties_for_adj(self, adj_lemma: str) -> dict[str, Any]:
        entries = self._lexicon.lookup(adj_lemma)
        for e in entries:
            if e.pos == EnglishPOS.ADJ and e.properties:
                return dict(e.properties)
        return {"property": adj_lemma}
