"""Unit tests verifying externalized data loading, dynamic problem synthesis, and semantic induction.

Ensures that:
1. Foundational lexicon, multilingual mappings, and kindergarten primer load from TSV data files.
2. Semantic roles and parent concepts are correctly linked from TSV data.
3. Multilingual registry operates dynamically from external translation matrices.
4. Raw book pipeline extracts dynamic worked problems and analogies from literary texts.
5. Entity resolution resolves arbitrary taxonomic container concepts without hardcoding.
"""

from __future__ import annotations

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.compositional_language import CompositionalLanguageEngine
from plugins.developmental_adapter.dictionary_store import LanguageDictionary, LexicalCategory
from plugins.developmental_adapter.environment import (
    BabyObjectState,
    BabyObjectType,
    BabyWorldEnvironment,
    Vector2D,
)
from plugins.developmental_adapter.goal_planning import GoalDirectedPlanningEngine
from plugins.developmental_adapter.language_grounding import LanguageGroundingEngine
from plugins.developmental_adapter.multilingual import (
    MultilingualLexiconRegistry,
    SupportedLanguage,
)
from plugins.developmental_adapter.raw_book_pipeline import (
    AutomatedCurriculumCompiler,
    BookCurriculumItem,
    BookSourceType,
)
from plugins.developmental_adapter.teacher import PedagogicalTeacher


def test_externalized_tsv_lexicon_loading() -> None:
    """Verify that LanguageDictionary loads its foundational lexicon from TSV with high coverage."""
    dictionary = LanguageDictionary.get_instance()
    assert len(dictionary.entries) >= 200

    # Test that core container words loaded from TSV retain CONTAINER semantic role
    box_entry = dictionary.lookup("box")
    assert box_entry is not None
    assert box_entry.is_container is True
    assert box_entry.category == LexicalCategory.NOUN

    crucible_entry = dictionary.lookup("crucible")
    assert crucible_entry is not None
    assert crucible_entry.is_container is True

    # Test tools loaded from TSV
    lever_entry = dictionary.lookup("lever")
    assert lever_entry is not None
    assert lever_entry.is_tool is True

    # Test verbs loaded from TSV
    pull_entry = dictionary.lookup("pull")
    assert pull_entry is not None
    assert pull_entry.semantic_role == "PULL"
    assert pull_entry.category == LexicalCategory.VERB


def test_externalized_multilingual_lexicon_loading() -> None:
    """Verify that MultilingualLexiconRegistry dynamically populates from multilingual_lexicon.tsv."""
    registry = MultilingualLexiconRegistry

    # Ensure token mappings exist across all 4 supported languages
    assert len(registry.CONTAINER_TOKENS) >= 20
    assert registry.CONTAINER_TOKENS["boîte"] == SupportedLanguage.FRENCH
    assert registry.CONTAINER_TOKENS["caja"] == SupportedLanguage.SPANISH
    assert registry.CONTAINER_TOKENS["kasten"] == SupportedLanguage.GERMAN

    # Test tools across languages
    assert registry.TOOL_TOKENS["bâton"] == SupportedLanguage.FRENCH
    assert registry.TOOL_TOKENS["palanca"] == SupportedLanguage.SPANISH
    assert registry.TOOL_TOKENS["stock"] == SupportedLanguage.GERMAN

    # Test actions across languages
    act_fr, lang_fr = registry.ACTION_TOKENS["tirer"]
    assert lang_fr == SupportedLanguage.FRENCH
    assert act_fr.value == "PULL"

    act_de, lang_de = registry.ACTION_TOKENS["drücken"]
    assert lang_de == SupportedLanguage.GERMAN
    assert act_de.value == "PUSH"


def test_kindergarten_primer_tsv_loading() -> None:
    """Verify that PedagogicalTeacher loads its ostensive demonstration pairs from primer TSV."""
    teacher = PedagogicalTeacher()
    lessons = teacher.load_kindergarten_primer()

    assert len(lessons) >= 11
    utterances = [utt for utt, _ in lessons]
    assert "red" in utterances
    assert "ball" in utterances
    assert "box" in utterances
    assert "stick" in utterances
    assert "push" in utterances
    assert "pull" in utterances
    assert "inside" in utterances


def test_dynamic_worked_problem_and_analogy_from_novel_text() -> None:
    """Verify that raw book pipeline extracts worked problems and analogies from text."""
    faraday_excerpt = """
    A candle is a cylinder of tallow or wax with a central wick.
    When heat is applied, the wax melts into a liquid fluid that rises through capillary action.
    The flame of a candle is like a chemical furnace where carbon and hydrogen combine with air.
    An analogy between a candle flame and an industrial combustion engine demonstrates chemical energy transfer.
    A lever or rod can be used to lift and position the fuel container.
    """
    item = BookCurriculumItem(
        item_id="faraday_test_novel",
        source_type=BookSourceType.RAW_TEXT,
        identifier=faraday_excerpt,
        title="Faraday's Chemical History of a Candle",
        domain="chemistry",
        grade_level=3,
    )

    chapter = AutomatedCurriculumCompiler.compile_chapter(faraday_excerpt, item)

    # 1. Worked problem check: instruction dynamically formulated
    prob_sec = chapter.sections[1]
    assert "instruction" in prob_sec.structured_payload
    inst = prob_sec.structured_payload["instruction"]
    assert "inside box" in inst or "inside" in inst

    # 2. Analogy check: extracted from text patterns ("analogy between ... and ...")
    analogy_sec = chapter.sections[2]
    source_dom = analogy_sec.structured_payload["source_domain"]
    target_dom = analogy_sec.structured_payload["target_domain"]
    assert source_dom != ""
    assert target_dom != ""


def test_dynamic_container_resolution_in_compositional_language() -> None:
    """Verify entity resolution matches arbitrary container terms from LanguageDictionary."""
    substrate = create_blank_brain_substrate()
    env = BabyWorldEnvironment(seed=42)
    planner = GoalDirectedPlanningEngine(substrate=substrate, env=env)
    grounding = LanguageGroundingEngine(substrate=substrate, env=env)
    comp_engine = CompositionalLanguageEngine(
        substrate=substrate, env=env, grounding_engine=grounding, planning_engine=planner
    )

    # Place a custom receptacle in the environment
    custom_crate = BabyObjectState(
        id="custom_storage_crate",
        object_type=BabyObjectType.BOX,
        color="yellow",
        mass=3.0,
        position=Vector2D(0.2, 0.4),
        size=Vector2D(0.4, 0.4),
        is_container=True,
    )
    env.objects = {"custom_storage_crate": custom_crate}

    # Test resolving "crate" - even though object_type is "box", dictionary knows crate is a container!
    from plugins.developmental_adapter.types import LexicalCategory, LexicalEntry

    resolved_id = comp_engine._resolve_entity_reference(
        nouns=[
            LexicalEntry(
                token="crate",
                category=LexicalCategory.NOUN,
                grounded_symbol="crate",
                co_occurrence_count=1,
                confidence=1.0,
            )
        ],
        adjectives=[],
    )
    assert resolved_id == "custom_storage_crate"
