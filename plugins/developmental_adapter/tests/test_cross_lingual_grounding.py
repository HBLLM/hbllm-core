"""Unit tests for Cross-Lingual Grounding & Multilingual Ingestion (Milestone A24)."""

from __future__ import annotations

from plugins.developmental_adapter.multilingual import (
    MultilingualLexiconRegistry,
    SupportedLanguage,
)
from plugins.developmental_adapter.types import (
    BabyActionType,
    BabyObjectType,
    BabyRelationType,
)


def test_language_detection() -> None:
    """Verify fast identification of English, French, Spanish, and German text."""
    en_sample = "The Project Gutenberg eBook of Classical Mechanics and Gravitational Force."
    fr_sample = "Traité élémentaire de physique. Les corps solides et la pression dans un fluide."
    es_sample = (
        "Tratado elemental de física. Las fuerzas y el movimiento de los cuerpos en el espacio."
    )
    de_sample = "Lehrbuch der Physik. Die Bewegung der Körper und das Gesetz der Trägheit."

    assert MultilingualLexiconRegistry.detect_language(en_sample) == SupportedLanguage.ENGLISH
    assert MultilingualLexiconRegistry.detect_language(fr_sample) == SupportedLanguage.FRENCH
    assert MultilingualLexiconRegistry.detect_language(es_sample) == SupportedLanguage.SPANISH
    assert MultilingualLexiconRegistry.detect_language(de_sample) == SupportedLanguage.GERMAN


def test_multilingual_lexicon_grounding_into_physical_invariants() -> None:
    """Verify non-English surface words map into canonical physical simulation types."""
    # French Grounding
    fr_box = MultilingualLexiconRegistry.ground_multilingual_term("boîte")
    assert fr_box is not None
    assert fr_box["entity_type"] == BabyObjectType.BOX

    fr_stick = MultilingualLexiconRegistry.ground_multilingual_term("bâton")
    assert fr_stick is not None
    assert fr_stick["entity_type"] == BabyObjectType.TOOL

    fr_ball = MultilingualLexiconRegistry.ground_multilingual_term("balle")
    assert fr_ball is not None
    assert fr_ball["entity_type"] == BabyObjectType.BALL

    fr_pull = MultilingualLexiconRegistry.ground_multilingual_term("tirer")
    assert fr_pull is not None
    assert fr_pull["action"] == BabyActionType.PULL

    fr_inside = MultilingualLexiconRegistry.ground_multilingual_term("dans")
    assert fr_inside is not None
    assert fr_inside["relation"] == BabyRelationType.INSIDE

    # Spanish Grounding
    es_box = MultilingualLexiconRegistry.ground_multilingual_term("caja")
    assert es_box is not None
    assert es_box["entity_type"] == BabyObjectType.BOX

    es_tool = MultilingualLexiconRegistry.ground_multilingual_term("palanca")
    assert es_tool is not None
    assert es_tool["entity_type"] == BabyObjectType.TOOL

    es_pull = MultilingualLexiconRegistry.ground_multilingual_term("jalar")
    assert es_pull is not None
    assert es_pull["action"] == BabyActionType.PULL

    # German Grounding
    de_box = MultilingualLexiconRegistry.ground_multilingual_term("kasten")
    assert de_box is not None
    assert de_box["entity_type"] == BabyObjectType.BOX

    de_tool = MultilingualLexiconRegistry.ground_multilingual_term("hebel")
    assert de_tool is not None
    assert de_tool["entity_type"] == BabyObjectType.TOOL

    de_pull = MultilingualLexiconRegistry.ground_multilingual_term("ziehen")
    assert de_pull is not None
    assert de_pull["action"] == BabyActionType.PULL


def test_multilingual_worked_problem_parsing() -> None:
    """Verify natural language instructions across languages compile into BabyWorld configs."""
    # French instruction
    fr_prob = MultilingualLexiconRegistry.parse_multilingual_worked_problem(
        "utiliser le bâton pour tirer la balle verte à l'intérieur de la boîte"
    )
    assert fr_prob["target_shape"] == BabyObjectType.BALL
    assert fr_prob["target_color"] == "green"
    assert fr_prob["target_action"] == BabyActionType.PULL
    assert fr_prob["target_relation"] == BabyRelationType.INSIDE
    assert fr_prob["needs_tool"] is True

    # Spanish instruction
    es_prob = MultilingualLexiconRegistry.parse_multilingual_worked_problem(
        "empujar el bloque rojo dentro de la caja"
    )
    assert es_prob["target_shape"] == BabyObjectType.BLOCK
    assert es_prob["target_color"] == "red"
    assert es_prob["target_action"] == BabyActionType.PUSH
    assert es_prob["target_relation"] == BabyRelationType.INSIDE
    assert es_prob["needs_tool"] is False

    # German instruction
    de_prob = MultilingualLexiconRegistry.parse_multilingual_worked_problem(
        "den grünen ball mit dem stock in den kasten ziehen"
    )
    assert de_prob["target_shape"] == BabyObjectType.BALL
    assert de_prob["target_color"] == "green"
    assert de_prob["target_action"] == BabyActionType.PULL
    assert de_prob["target_relation"] == BabyRelationType.INSIDE
    assert de_prob["needs_tool"] is True


def test_multilingual_simulation_execution() -> None:
    """Verify student executes and solves a puzzle formulated in French."""
    from plugins.developmental_adapter.school import CognitiveSchool
    from plugins.developmental_adapter.textbook_curriculum import (
        TextbookSection,
        TextbookSectionType,
        TextbookSimulationCompiler,
    )

    school = CognitiveSchool(seed=42)
    student = school.student

    # Create a French worked problem section
    sec = TextbookSection(
        section_id="sec_fr_1",
        title="Problème mécanique",
        section_type=TextbookSectionType.WORKED_PROBLEM,
        raw_text="utiliser le bâton pour tirer la balle verte à l'intérieur de la boîte",
        structured_payload={
            "instruction": "utiliser le bâton pour tirer la balle verte dans la boîte",
            "worked_solution_steps": ["MOVE", "GRASP", "MOVE", "PLACE"],
            "has_tool_mechanic": True,
        },
    )

    puzzle = TextbookSimulationCompiler.compile_puzzle(sec)
    eval_res = TextbookSimulationCompiler.verify_student_solution(student, puzzle)

    assert eval_res["is_success"] is True
    assert eval_res["plan_length"] >= 2
    assert eval_res["confidence"] > 0.0


def test_cross_lingual_curriculum_teaching() -> None:
    """Verify teaching a chapter written with French terminology updates student knowledge."""
    from plugins.developmental_adapter.school import CognitiveSchool
    from plugins.developmental_adapter.textbook_curriculum import (
        TextbookChapter,
        TextbookCurriculumCurator,
        TextbookSection,
        TextbookSectionType,
    )

    school = CognitiveSchool(seed=42)
    student = school.student
    curator = TextbookCurriculumCurator()

    # Teach French chapter
    chapter = TextbookChapter(
        chapter_id="physique_fr_ch1",
        title="Mécanique et Récipients",
        grade_level=3,
        sections=[
            TextbookSection(
                section_id="def_fr",
                title="Définitions Françaises",
                section_type=TextbookSectionType.DEFINITIONS,
                raw_text="Une boîte est un récipient utilisé pour contenir des objets.",
                structured_payload={
                    "glossary": {
                        "boîte": "Un récipient rigide utilisé pour stocker et contenir des charges.",
                        "bâton": "Une tige solide servant de levier pour déplacer des objets.",
                    }
                },
            )
        ],
    )

    res = curator.teach_chapter(student, chapter)
    assert res["sections_processed"] >= 1
    assert "boîte" in res["grounded_concepts"]
    assert "bâton" in res["grounded_concepts"]

    # Student now possesses grounded representations for these multilingual tokens
    assert "boîte" in student.grounding_engine.lexicon
    assert "bâton" in student.grounding_engine.lexicon
