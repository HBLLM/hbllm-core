"""Universal Declarative Cognitive Grammar & Language Independence Tests.

Verifies:
1. Declarative `LanguageProfile` serialization and validation.
2. `UniversalLanguageParser` parsing queries, commands, assertions across typologies (SVO, SOV, Preposition, Postposition).
3. `UniversalLanguageRealizer` generating calibrated verbalizations in target languages.
4. Auto-discovery of language profiles in `MultilingualLanguageRuntime`.
5. Zero-code addition of new languages (French, German, Spanish).
6. Multi-way interlingual semantic transfer across languages via language-neutral SemanticFrames.
7. Epistemic safety invariants (rejection of ungroundable or empty utterances).
"""

from __future__ import annotations

from pathlib import Path

from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)
from hbllm.hcir.language.core.profile import (
    ClosedClassLexicon,
    LanguageProfile,
    RealizationTemplates,
    TypologyConfig,
)
from hbllm.hcir.language.core.semantic_frame import (
    FrameType,
    LanguageErrorType,
    ThematicRole,
)
from hbllm.hcir.language.runtime import MultilingualLanguageRuntime


def _build_test_world() -> tuple[CognitiveGraph, PhysicalEntityNode, PhysicalEntityNode]:
    graph = CognitiveGraph()
    table = PhysicalEntityNode(entity_type="table")
    ball = PhysicalEntityNode(entity_type="ball", properties={"color": "red"})
    graph.add_node(table)
    graph.add_node(ball)
    graph.add_edge(
        HCIREdge(
            edge_type=HCIREdgeType.LOCATED_IN,
            sources=[ball.id],
            targets=[table.id],
        )
    )
    return graph, ball, table


class TestDeclarativeLanguageProfiles:
    """Tests declarative profile schema serialization and profile discovery."""

    def test_profile_schema_roundtrip(self, tmp_path: Path) -> None:
        profile = LanguageProfile(
            iso_code="es",
            name="Spanish",
            typology=TypologyConfig(
                word_order="SVO",
                adposition_type="preposition",
                wh_movement="front",
            ),
            closed_class=ClosedClassLexicon(
                specifiers={"el": "definite", "la": "definite", "un": "indefinite"},
                adpositions={"en": "located_on", "sobre": "located_on", "debajo": "located_under"},
                question_markers={"dónde": {"target": "location", "predicate": "located_on"}},
                command_verbs={"listar": "fs.list", "leer": "fs.read", "empujar": "push"},
                polite_wrappers=["por favor"],
            ),
            templates=RealizationTemplates(
                location_known="El {theme} está {adposition} la {location}.",
            ),
        )
        profile_file = tmp_path / "es.json"
        profile.to_file(profile_file)

        loaded = LanguageProfile.from_file(profile_file)
        assert loaded.iso_code == "es"
        assert loaded.name == "Spanish"
        assert loaded.typology.word_order == "SVO"
        assert loaded.closed_class.specifiers["el"] == "definite"
        assert loaded.closed_class.adpositions["sobre"] == "located_on"

    def test_multilingual_runtime_autodiscovery(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)

        # Profiles auto-discovered from profiles/ directory
        assert "en" in rt.supported_languages
        assert "si" in rt.supported_languages
        assert "ta" in rt.supported_languages
        assert "fr" in rt.supported_languages
        assert "de" in rt.supported_languages

        assert rt.get_profile("fr") is not None
        assert rt.get_profile("fr").name == "French"
        assert rt.get_profile("de") is not None
        assert rt.get_profile("de").name == "German"


class TestUniversalEngineFrench:
    """Verifies that French operates cleanly with zero Python code."""

    def test_french_wh_query_parsing(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)
        fr_parser = rt._parsers["fr"]

        frame = fr_parser.parse("Où est le ballon rouge?")
        assert frame.frame_type == FrameType.QUERY
        assert frame.query_target == "location"
        assert frame.predicate == "located_on"

        theme = frame.get_role(ThematicRole.THEME)
        assert theme is not None
        assert theme.concept_name == "ballon"
        assert theme.properties.get("color") == "rouge"

    def test_french_command_parsing(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)
        fr_parser = rt._parsers["fr"]

        frame = fr_parser.parse("Lister les fichiers")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate in ("lister", "liste")

    def test_french_spatial_assertion_parsing(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)
        fr_parser = rt._parsers["fr"]

        frame = fr_parser.parse("Le ballon rouge est sur la table.")
        assert frame.frame_type == FrameType.ASSERTION
        assert frame.predicate == "located_on"
        theme = frame.get_role(ThematicRole.THEME)
        loc = frame.get_role(ThematicRole.LOCATION)
        assert theme is not None and theme.concept_name == "ballon"
        assert loc is not None and loc.concept_name == "table"

    def test_french_dialogue_execution(self) -> None:
        graph, ball, table = _build_test_world()
        # Add french entity aliases
        ball.entity_type = "ballon"
        ball.properties["color"] = "rouge"
        table.entity_type = "table"

        rt = MultilingualLanguageRuntime(graph)
        res = rt.process_utterance("Où est le ballon rouge?", language="fr")
        assert res.is_success is True
        assert "ballon" in res.response_text
        assert "table" in res.response_text


class TestUniversalEngineGerman:
    """Verifies that German operates cleanly with zero Python code."""

    def test_german_wh_query_parsing(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)
        de_parser = rt._parsers["de"]

        frame = de_parser.parse("Wo ist der rote Ball?")
        assert frame.frame_type == FrameType.QUERY
        assert frame.query_target == "location"
        assert frame.predicate == "located_on"
        theme = frame.get_role(ThematicRole.THEME)
        assert theme is not None
        assert theme.concept_name == "ball"
        assert theme.properties.get("color") == "rot"

    def test_german_dialogue_execution(self) -> None:
        graph, ball, table = _build_test_world()
        ball.entity_type = "ball"
        ball.properties["color"] = "rot"
        table.entity_type = "tisch"

        rt = MultilingualLanguageRuntime(graph)
        res = rt.process_utterance("Wo ist der rote Ball?", language="de")
        assert res.is_success is True
        assert "ball" in res.response_text
        assert "tisch" in res.response_text


class TestZeroCodeLanguageTraining:
    """Tests dynamic, programmatic addition of a new language (Spanish) with zero code changes."""

    def test_dynamic_spanish_registration(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)

        spanish_profile = LanguageProfile(
            iso_code="es",
            name="Spanish",
            typology=TypologyConfig(
                word_order="SVO",
                adposition_type="preposition",
                wh_movement="front",
            ),
            closed_class=ClosedClassLexicon(
                specifiers={"el": "definite", "la": "definite", "un": "indefinite"},
                adpositions={"sobre": "located_on", "en": "located_on", "debajo": "located_under"},
                question_markers={"dónde": {"target": "location", "predicate": "located_on"}},
                copula_forms=["está", "están", "es", "son"],
                command_verbs={"listar": "fs.list", "explicar": "knowledge.explain"},
                polite_wrappers=["por favor"],
            ),
            templates=RealizationTemplates(
                location_known="El {theme} está {adposition} la {location}.",
                location_unknown="No sé dónde está el {theme}.",
            ),
        )

        rt.register_profile(spanish_profile)
        assert "es" in rt.supported_languages

        frame = rt._parsers["es"].parse("Por favor, ¿dónde está la pelota?")
        assert frame.frame_type == FrameType.QUERY
        assert frame.query_target == "location"
        theme = frame.get_role(ThematicRole.THEME)
        assert theme is not None
        assert "pelota" in theme.concept_name


class TestInterlingualSemanticTransfer:
    """Verifies deterministic multi-way translation between languages via language-neutral SemanticFrames."""

    def test_translation_english_to_french(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)

        res = rt.translate("Where is the ball?", source_lang="en", target_lang="fr")
        assert "où" in res.lower()
        assert "ball" in res.lower()

    def test_translation_french_to_german(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)

        res = rt.translate("Où est le ballon?", source_lang="fr", target_lang="de")
        assert "wo" in res.lower()
        assert "ballon" in res.lower()

    def test_epistemic_safety_empty_and_unknown(self) -> None:
        graph = CognitiveGraph()
        rt = MultilingualLanguageRuntime(graph)

        fr_parser = rt._parsers["fr"]
        err_frame = fr_parser.parse("")
        assert err_frame.frame_type == FrameType.ERROR
        assert err_frame.error_type == LanguageErrorType.UNRESOLVED_LANGUAGE
