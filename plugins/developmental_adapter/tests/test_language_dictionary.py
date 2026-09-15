"""Unit tests for Autonomous Language Dictionary Store and Dynamic Lexicon Ingestion."""

from __future__ import annotations

import tempfile
from pathlib import Path

from plugins.developmental_adapter.dictionary_store import (
    LanguageDictionary,
    LexicalCategory,
    SemanticRole,
)
from plugins.developmental_adapter.textbook_curriculum import (
    TextbookChapter,
    TextbookCurriculumCurator,
    TextbookSection,
    TextbookSectionType,
)


def test_dictionary_lookup_lexical_categories() -> None:
    """Verify built-in dictionary accurately resolves nouns, verbs, prepositions, and adjectives."""
    dict_store = LanguageDictionary()

    # Nouns & semantic roles
    box_entry = dict_store.lookup("box")
    assert box_entry is not None
    assert box_entry.category == LexicalCategory.NOUN
    assert box_entry.semantic_role == SemanticRole.CONTAINER

    stick_entry = dict_store.lookup("stick")
    assert stick_entry is not None
    assert stick_entry.category == LexicalCategory.NOUN
    assert stick_entry.semantic_role == SemanticRole.TOOL

    ball_entry = dict_store.lookup("ball")
    assert ball_entry is not None
    assert ball_entry.category == LexicalCategory.NOUN
    assert ball_entry.semantic_role == SemanticRole.BALL

    block_entry = dict_store.lookup("cube")
    assert block_entry is not None
    assert block_entry.category == LexicalCategory.NOUN
    assert block_entry.semantic_role == SemanticRole.BLOCK

    # Verbs & semantic roles
    pull_entry = dict_store.lookup("pull")
    assert pull_entry is not None
    assert pull_entry.category == LexicalCategory.VERB
    assert pull_entry.semantic_role == SemanticRole.PULL

    push_entry = dict_store.lookup("push")
    assert push_entry is not None
    assert push_entry.category == LexicalCategory.VERB
    assert push_entry.semantic_role == SemanticRole.PUSH

    roll_entry = dict_store.lookup("roll")
    assert roll_entry is not None
    assert roll_entry.category == LexicalCategory.VERB
    assert roll_entry.semantic_role == SemanticRole.ROLL

    grasp_entry = dict_store.lookup("grasp")
    assert grasp_entry is not None
    assert grasp_entry.category == LexicalCategory.VERB
    assert grasp_entry.semantic_role == SemanticRole.GRASP

    # Prepositions & spatial roles
    in_entry = dict_store.lookup("inside")
    assert in_entry is not None
    assert in_entry.category == LexicalCategory.PREPOSITION
    assert in_entry.semantic_role == SemanticRole.INSIDE

    near_entry = dict_store.lookup("near")
    assert near_entry is not None
    assert near_entry.category == LexicalCategory.PREPOSITION
    assert near_entry.semantic_role == SemanticRole.NEAR

    # Adjectives
    red_entry = dict_store.lookup("red")
    assert red_entry is not None
    assert red_entry.category == LexicalCategory.ADJECTIVE


def test_morphological_lemmatization() -> None:
    """Verify stemming/lemmatization fallback for plurals, past tense, and continuous gerunds."""
    dict_store = LanguageDictionary()

    # Plural nouns
    crates_entry = dict_store.lookup("crates")
    assert crates_entry is not None
    assert crates_entry.word == "crate"
    assert crates_entry.semantic_role == SemanticRole.CONTAINER

    balls_entry = dict_store.lookup("balls")
    assert balls_entry is not None
    assert balls_entry.word == "ball"
    assert balls_entry.semantic_role == SemanticRole.BALL

    sticks_entry = dict_store.lookup("sticks")
    assert sticks_entry is not None
    assert sticks_entry.word == "stick"
    assert sticks_entry.semantic_role == SemanticRole.TOOL

    # Past tense verbs
    pulled_entry = dict_store.lookup("pulled")
    assert pulled_entry is not None
    assert pulled_entry.word == "pull"
    assert pulled_entry.semantic_role == SemanticRole.PULL

    pushed_entry = dict_store.lookup("pushed")
    assert pushed_entry is not None
    assert pushed_entry.word == "push"
    assert pushed_entry.semantic_role == SemanticRole.PUSH

    # Continuous participles / gerunds
    rolling_entry = dict_store.lookup("rolling")
    assert rolling_entry is not None
    assert rolling_entry.word == "roll"
    assert rolling_entry.semantic_role == SemanticRole.ROLL

    grasping_entry = dict_store.lookup("grasping")
    assert grasping_entry is not None
    assert grasping_entry.word == "grasp"
    assert grasping_entry.semantic_role == SemanticRole.GRASP


def test_dynamic_registration_and_semantic_inference() -> None:
    """Verify autonomous learning of novel terms and automated semantic role deduction."""
    dict_store = LanguageDictionary()

    # Register explicit role
    entry1 = dict_store.register_word(
        word="containment_vessel",
        category=LexicalCategory.NOUN,
        definition="A rigid chamber designed to enclose, hold, and store physical specimens.",
        semantic_role=SemanticRole.CONTAINER,
    )
    assert entry1.word == "containment_vessel"
    assert entry1.semantic_role == SemanticRole.CONTAINER
    assert dict_store.lookup("containment_vessel") == entry1

    # Register without role, rely on definition inference
    entry2 = dict_store.register_word(
        word="manipulator_arm",
        category=LexicalCategory.NOUN,
        definition="An elongated rigid lever instrument used to extend reach and pull distant objects.",
    )
    assert entry2.semantic_role == SemanticRole.TOOL

    entry3 = dict_store.register_word(
        word="vault",
        category=LexicalCategory.NOUN,
        definition="A secure room or compartment used to store and hold valuables inside.",
    )
    assert entry3.semantic_role == SemanticRole.CONTAINER


def test_translation_dictionary_support() -> None:
    """Verify dictionary entry maintains translation mappings for cross-lingual learning."""
    dict_store = LanguageDictionary()

    dict_store.register_word(
        word="box",
        category=LexicalCategory.NOUN,
        definition="A rigid container.",
        semantic_role=SemanticRole.CONTAINER,
        translations={"fr": "boîte", "es": "caja", "de": "Kasten"},
    )

    box = dict_store.lookup("box")
    assert box is not None
    assert box.translations["fr"] == "boîte"
    assert box.translations["es"] == "caja"
    assert box.translations["de"] == "Kasten"


def test_tsv_import_export() -> None:
    """Verify dictionary entries can be saved to and loaded from standardized TSV files."""
    dict_store = LanguageDictionary()
    dict_store.register_word(
        word="test_device",
        category=LexicalCategory.NOUN,
        definition="A mechanical tool apparatus.",
        semantic_role=SemanticRole.TOOL,
        translations={"es": "aparato"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tsv_path = Path(tmpdir) / "dict.tsv"
        dict_store.export_to_tsv(tsv_path)
        assert tsv_path.exists()

        new_dict = LanguageDictionary()
        new_dict.load_from_tsv(tsv_path)
        loaded = new_dict.lookup("test_device")
        assert loaded is not None
        assert loaded.category == LexicalCategory.NOUN
        assert loaded.semantic_role == SemanticRole.TOOL
        assert loaded.translations.get("es") == "aparato"


def test_textbook_curriculum_uses_language_dictionary() -> None:
    """Verify TextbookCurriculumCurator dynamically leverages the dictionary to ground concepts."""
    from plugins.developmental_adapter.school import CognitiveSchool

    school = CognitiveSchool(seed=42)
    student = school.student
    curator = TextbookCurriculumCurator(dictionary=school.teacher.dictionary)

    # Create a chapter introducing both known and novel terms
    chapter = TextbookChapter(
        chapter_id="novel_tools_ch1",
        title="Novel Tools and Vessels",
        grade_level=3,
        sections=[
            TextbookSection(
                section_id="novel_tools_def",
                title="Definitions and Tools",
                section_type=TextbookSectionType.DEFINITIONS,
                raw_text="A repository is a container used to hold and enclose items.",
                structured_payload={
                    "glossary": {
                        "repository": "A container used to hold and enclose items.",
                        "prying_bar": "A long rigid bar tool used to drag and pull objects.",
                    }
                },
            )
        ],
    )

    res = curator.teach_chapter(student, chapter)
    assert res["sections_processed"] >= 1
    assert "repository" in res["grounded_concepts"]
    assert "prying_bar" in res["grounded_concepts"]

    # Verify student learned grounded concepts
    assert "repository" in student.grounding_engine.lexicon
    assert "prying_bar" in student.grounding_engine.lexicon
