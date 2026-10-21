"""Unit tests for the CurriculumFetcher module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from plugins.developmental_adapter.curriculum_fetcher import CurriculumFetcher
from plugins.developmental_adapter.textbook_curriculum import TextbookSectionType


def test_list_and_load_available_chapters() -> None:
    fetcher = CurriculumFetcher()
    chapters = fetcher.list_available_chapters()

    assert len(chapters) >= 5
    file_names = [p.name for p in chapters]
    assert "openstax_simple_machines_ch9.md" in file_names
    assert "openstax_physics_ch4_newton_laws.md" in file_names
    assert "openstax_physics_ch11_fluid_statics.md" in file_names
    assert "openstax_chemistry_ch2_atomic_theory.md" in file_names
    assert "gutenberg_science_faraday_candle.md" in file_names

    # Verify each chapter parses completely into 4 sections
    all_loaded = fetcher.load_all_chapters()
    assert len(all_loaded) >= 5

    for ch in all_loaded:
        assert len(ch.sections) == 4
        def_sec = ch.get_section(TextbookSectionType.DEFINITIONS)
        assert def_sec is not None
        assert len(def_sec.structured_payload.get("glossary", {})) >= 3

        prob_sec = ch.get_section(TextbookSectionType.WORKED_PROBLEM)
        assert prob_sec is not None
        assert "instruction" in prob_sec.structured_payload

        analogy_sec = ch.get_section(TextbookSectionType.ANALOGY_SCHEMA)
        assert analogy_sec is not None

        exam_sec = ch.get_section(TextbookSectionType.EXAM_CHALLENGE)
        assert exam_sec is not None


def test_synthesize_custom_chapter() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = CurriculumFetcher(data_dir=tmpdir)
        ch = fetcher.synthesize_chapter(
            chapter_id="custom_biology_ch1",
            title="Cellular Transport Mechanisms",
            grade_level=5,
            source_attribution="OpenStax Biology 2e (CC-BY 4.0)",
            glossary={
                "membrane": "A selective barrier enclosing cell cytoplasm.",
                "diffusion": "Passive transport of particles down concentration gradient.",
                "box": "An impermeable cell chamber containing internal fluids.",
            },
            worked_instruction="pull green ball inside box",
            worked_scenario="A glucose vesicle is floating in extracellular matrix.",
            analogy_source="Cell Membrane Transport",
            analogy_target="Industrial Ultrafiltration",
            analogy_mapping="Membrane pores map to synthetic nanofilters.",
            review_questions=["How does active transport differ from passive diffusion?"],
            save_to_disk=True,
        )

        assert ch.chapter_id == "custom_biology_ch1"
        assert "Cellular Transport Mechanisms" in ch.title
        assert ch.grade_level == 5

        # Verify persisted to disk
        persisted_file = Path(tmpdir) / "custom_biology_ch1.md"
        assert persisted_file.exists()

        reloaded = fetcher.load_chapter(persisted_file)
        assert reloaded.chapter_id == "custom_biology_ch1"
        def_sec = reloaded.get_section(TextbookSectionType.DEFINITIONS)
        assert def_sec is not None
        assert "membrane" in def_sec.structured_payload["glossary"]
