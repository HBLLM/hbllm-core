"""English Semantic Parser for A16.

Converts English parse tree AST nodes into language-neutral SemanticFrames.
Implements the LanguageParser protocol.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    LanguageErrorType,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)
from hbllm.brain.language.english.lexicon import EnglishLexicon, EnglishPOS
from hbllm.brain.language.english.morphology import EnglishMorphology
from hbllm.brain.language.english.syntax import (
    ConstructionType,
    EnglishSyntaxParser,
    NounPhrase,
    SentenceNode,
)

logger = logging.getLogger(__name__)


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

    def parse(self, text: str) -> SemanticFrame:
        """Parse raw English text into a language-neutral SemanticFrame."""
        meta = LanguageMetadata(language="en", raw_text=text)

        # 1. Morphological analysis & tokenization
        tokens = self._morphology.lemmatize_sequence(text)
        if not tokens:
            return SemanticFrame(
                frame_type=FrameType.ERROR,
                error_type=LanguageErrorType.UNRESOLVED_LANGUAGE,
                error_detail="Empty utterance.",
                metadata=meta,
            )

        # 2. Syntactic parsing
        ast = self._syntax.parse(tokens)
        if not ast:
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

    # ── AST to SemanticFrame Translation ──────────────────────────────

    def _ast_to_semantic_frame(self, ast: SentenceNode, meta: LanguageMetadata) -> SemanticFrame:
        """Map typed AST node to language-neutral SemanticFrame."""

        # 1. WH-QUESTIONS ("Where is the ball?", "What color is the box?")
        if ast.construction == ConstructionType.WH_QUESTION:
            subject_ref = self._np_to_reference(ast.subject)
            wh_word = ast.wh_word.lemma if ast.wh_word else "what"

            if wh_word == "where":
                frame = SemanticFrame(
                    frame_type=FrameType.QUERY,
                    predicate="located_on",
                    query_target="location",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                return frame

            elif wh_word == "what":
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

        # 2. YES/NO QUESTIONS ("Is the ball on the table?", "Is the ball red?")
        elif ast.construction == ConstructionType.YES_NO_QUESTION:
            subject_ref = self._np_to_reference(ast.subject)

            # Spatial query: "Is the ball on the table?"
            if ast.verb_phrase and ast.verb_phrase.prepositional_phrases:
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

            # Property verification: "Is the ball red?"
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
                return frame

        # 3. IMPERATIVE COMMANDS ("Move the red ball to the table", "Push the box")
        elif ast.construction == ConstructionType.IMPERATIVE:
            verb_pred = ast.verb_phrase.head_verb.lemma if ast.verb_phrase else "act"
            patient_ref = self._np_to_reference(ast.verb_phrase.direct_object) if ast.verb_phrase else None

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
        elif ast.construction in (ConstructionType.DECLARATIVE, ConstructionType.COPULAR, ConstructionType.EXISTENTIAL):
            subject_ref = self._np_to_reference(ast.subject)

            # Spatial assertion: "The ball is on the table"
            if ast.verb_phrase and ast.verb_phrase.prepositional_phrases:
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

            # Copular property assertion: "The ball is red"
            if ast.verb_phrase and ast.verb_phrase.adjective_complement:
                adj_tok = ast.verb_phrase.adjective_complement
                adj_props = self._get_properties_for_adj(adj_tok.lemma)

                # Merge adjective property into subject reference
                if subject_ref:
                    subject_ref = EntityReference(
                        concept_name=subject_ref.concept_name,
                        properties={**subject_ref.properties, **adj_props},
                        specifier=subject_ref.specifier,
                        raw_text=subject_ref.raw_text,
                    )

                frame = SemanticFrame(
                    frame_type=FrameType.ASSERTION,
                    predicate="is_property",
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.THEME, subject_ref)
                return frame

            # SVO action assertion: "The robot pushed the box"
            if ast.verb_phrase and ast.verb_phrase.head_verb:
                verb_pred = ast.verb_phrase.head_verb.lemma
                obj_ref = self._np_to_reference(ast.verb_phrase.direct_object)

                frame = SemanticFrame(
                    frame_type=FrameType.ASSERTION,
                    predicate=verb_pred,
                    metadata=meta,
                )
                if subject_ref:
                    frame.set_role(ThematicRole.AGENT, subject_ref)
                if obj_ref:
                    frame.set_role(ThematicRole.PATIENT, obj_ref)
                return frame

        # Fallback error
        return SemanticFrame(
            frame_type=FrameType.ERROR,
            error_type=LanguageErrorType.UNSUPPORTED_CONSTRUCTION,
            error_detail=f"Unsupported sentence construction: {ast.construction}",
            metadata=meta,
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

        # Specifier (definite vs indefinite vs demonstrative)
        specifier = "definite"
        if np.determiner:
            det_lemma = np.determiner.lemma
            if det_lemma in ("a", "an"):
                specifier = "indefinite"
            elif det_lemma in ("this", "that"):
                specifier = "demonstrative"

        # Adjective properties
        props: dict[str, Any] = {}
        for adj in np.adjectives:
            props.update(self._get_properties_for_adj(adj.lemma))

        concept_name = np.head_noun.lemma if np.head_noun else None
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
