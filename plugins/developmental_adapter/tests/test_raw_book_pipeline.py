"""Unit tests for the Automated Raw Book & Article Ingestion Pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

from plugins.developmental_adapter.raw_book_pipeline import (
    AutomatedCurriculumCompiler,
    BookCurriculumItem,
    BookSourceType,
    StandardBookCatalog,
    download_and_compile_books,
)
from plugins.developmental_adapter.textbook_curriculum import TextbookSectionType
from plugins.developmental_adapter.trainer import (
    ContinuousTextbookSchoolTrainer,
    TextbookSchoolConfig,
)


def test_automated_curriculum_compiler_from_raw_text() -> None:
    """Verify raw natural language text compiles into a structured TextbookChapter."""
    sample_text = """
    A treatise on mechanical forces and levers.
    In physics, force is an applied influence that causes a body to accelerate.
    Friction is a resistance force opposing relative surface motion.
    A lever is a rigid bar used to lift or displace mechanical resistance.
    When an operator applies force, the lever transfers work to the load.
    A box provides rigid boundary constraints to store loads.
    A stick is an elongated tool arm used to pull objects.
    """
    item = BookCurriculumItem(
        item_id="test_raw_mechanics",
        source_type=BookSourceType.RAW_TEXT,
        identifier=sample_text,
        title="Treatise on Mechanical Forces",
        domain="physics",
        grade_level=3,
    )

    chapter = AutomatedCurriculumCompiler.compile_chapter(sample_text, item)

    assert chapter.chapter_id == "test_raw_mechanics"
    assert chapter.title == "Treatise on Mechanical Forces"
    assert len(chapter.sections) == 4

    def_sec = chapter.get_section(TextbookSectionType.DEFINITIONS)
    assert def_sec is not None
    assert "box" in def_sec.structured_payload["glossary"]
    assert "stick" in def_sec.structured_payload["glossary"]

    prob_sec = chapter.get_section(TextbookSectionType.WORKED_PROBLEM)
    assert prob_sec is not None
    assert "instruction" in prob_sec.structured_payload

    analogy_sec = chapter.get_section(TextbookSectionType.ANALOGY_SCHEMA)
    assert analogy_sec is not None

    exam_sec = chapter.get_section(TextbookSectionType.EXAM_CHALLENGE)
    assert exam_sec is not None
    assert exam_sec.structured_payload["has_trick_question"] is True


def test_download_and_compile_catalog_item() -> None:
    """Verify downloading real text and compiling it directly into an in-memory chapter."""
    items = StandardBookCatalog.get_comprehensive_curriculum()
    assert len(items) >= 8

    # Process first 2 items (Faraday Candle & Classical mechanics)
    chapters = download_and_compile_books(catalog=items[:2])
    assert len(chapters) == 2

    for ch in chapters:
        assert len(ch.sections) == 4
        assert ch.chapter_id in ["book_faraday_candle", "wiki_classical_mechanics"]


def test_feed_in_memory_compiled_books_to_trainer() -> None:
    """Verify feeding in-memory downloaded chapters directly to ContinuousTextbookSchoolTrainer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 2 in-memory compiled chapters from raw text
        item1 = BookCurriculumItem(
            item_id="book_optics",
            source_type=BookSourceType.RAW_TEXT,
            identifier="Optics text: Refraction is the bending of light in glass lenses and prisms.",
            title="Newton's Opticks & Wave Light",
            domain="optics",
            grade_level=3,
        )
        item2 = BookCurriculumItem(
            item_id="book_thermo",
            source_type=BookSourceType.RAW_TEXT,
            identifier="Thermodynamics text: Heat is spontaneous thermal energy transfer between bodies.",
            title="Treatise on Thermodynamics",
            domain="thermodynamics",
            grade_level=3,
        )

        ch1 = AutomatedCurriculumCompiler.compile_chapter(item1.identifier, item1)
        ch2 = AutomatedCurriculumCompiler.compile_chapter(item2.identifier, item2)

        cfg = TextbookSchoolConfig(
            student_name="Live In-Memory Scholar",
            in_memory_chapters=[ch1, ch2],
            episodes_per_chapter=1,
            checkpoint_dir=Path(tmpdir) / "checkpoints",
            enable_sleep_consolidation=True,
            seed=42,
        )

        trainer = ContinuousTextbookSchoolTrainer(config=cfg)
        summary = trainer.train()

        assert summary.total_chapters_trained == 2
        assert summary.final_accuracy >= 0.80
        assert summary.final_gpa >= 3.0
        assert len(summary.chapter_reports) == 2
        assert Path(summary.chapter_reports[0].checkpoint_path).exists()
        assert Path(summary.chapter_reports[1].checkpoint_path).exists()
