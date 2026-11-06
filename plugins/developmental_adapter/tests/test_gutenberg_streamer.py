"""Unit tests for Project Gutenberg Streaming Curriculum & Knowledge Gain Engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

from plugins.developmental_adapter.gutenberg_streamer import (
    GutenbergCorpusStreamer,
    GutenbergIndexManager,
    KnowledgeGainTracker,
)
from plugins.developmental_adapter.trainer import (
    ContinuousTextbookSchoolTrainer,
    TextbookSchoolConfig,
)


def test_parse_gutenberg_index_and_domain_classification() -> None:
    sample_index = """
The Chemical History of a Candle, by Michael Faraday 14474
Opticks: or, a Treatise of the Reflections, by Isaac Newton 284
On the Origin of Species, by Charles Darwin 1228
The Principles of Psychology, by William James 1059
A Treatise on Electricity and Magnetism, by James Clerk Maxwell 48817
    """
    books = GutenbergIndexManager.parse_index(sample_index, max_books=10)

    assert len(books) == 5
    ids = [b.book_id for b in books]
    assert 14474 in ids
    assert 284 in ids
    assert 1228 in ids

    # Check domain classifications
    faraday = next(b for b in books if b.book_id == 14474)
    assert faraday.domain == "chemistry"

    newton = next(b for b in books if b.book_id == 284)
    assert newton.domain == "optics"

    darwin = next(b for b in books if b.book_id == 1228)
    assert darwin.domain == "biology"


def test_streaming_curriculum_and_knowledge_tracking() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        ckpt_dir = Path(tmpdir) / "ckpts"

        # Pre-seed cache with 2 offline books
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "book_14474.txt").write_text(
            "Candle combustion and capillary action in wax wicks. "
            "A force accelerates matter. A lever pulls loads into containers.",
            encoding="utf-8",
        )
        (cache_dir / "book_284.txt").write_text(
            "Refraction of light through prisms and lenses. "
            "Light rays undergo dispersion and focus into apertures.",
            encoding="utf-8",
        )

        sample_index = """
The Chemical History of a Candle, by Michael Faraday 14474
Opticks, by Isaac Newton 284
        """
        books = GutenbergIndexManager.parse_index(sample_index, max_books=2)
        streamer = GutenbergCorpusStreamer(cache_dir=cache_dir)
        tracker = KnowledgeGainTracker()

        cfg = TextbookSchoolConfig(
            student_name="Stream Test Student",
            episodes_per_chapter=1,
            checkpoint_dir=ckpt_dir,
            enable_sleep_consolidation=True,
            seed=42,
        )
        trainer = ContinuousTextbookSchoolTrainer(config=cfg)

        for b in books:
            raw_text = streamer.fetch_book_text(b)
            chapter = next(streamer.stream_curriculum_chapters([b]))
            trainer.train(chapters=[chapter])
            tracker.record_progress(trainer, b, len(raw_text))

        assert len(tracker.snapshots) == 2
        snap1 = tracker.snapshots[0]
        snap2 = tracker.snapshots[1]

        # Vocabulary must expand across books
        assert snap2.vocabulary_size >= snap1.vocabulary_size
        assert snap2.mean_examination_accuracy >= 0.75
        assert snap2.mean_brier_score <= 0.20

        # Verify Knowledge Report generation
        report = tracker.render_knowledge_report()
        assert "Project Gutenberg Cognitive Knowledge Acquisition Report" in report
        assert "Opticks" in report

        # Verify state persistence (save and load)
        state_file = ckpt_dir / "tracker_state.json"
        tracker.save_state(state_file)
        assert state_file.exists()

        restored_tracker = KnowledgeGainTracker.load_state(state_file)
        assert len(restored_tracker.snapshots) == 2
        assert restored_tracker.total_chars_processed == tracker.total_chars_processed
        assert (
            restored_tracker.snapshots[-1].vocabulary_size == tracker.snapshots[-1].vocabulary_size
        )
        streamer.close()


def test_background_prefetcher_lifecycle() -> None:
    """Verify BackgroundGutenbergPrefetcher queueing, duplicate avoidance, and clean shutdown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Pre-seed one book in cache
        (cache_dir / "book_99999.txt").write_text("Offline book content " * 100, encoding="utf-8")

        from plugins.developmental_adapter.gutenberg_streamer import (
            BackgroundGutenbergPrefetcher,
            GutenbergBookMetadata,
        )

        prefetcher = BackgroundGutenbergPrefetcher(cache_dir=cache_dir)
        prefetcher.start()

        meta_cached = GutenbergBookMetadata(
            book_id=99999,
            title="Pre-cached Book",
            author="Author",
        )
        # Should detect already cached and skip queue
        prefetcher.submit(meta_cached)
        assert meta_cached.book_id in prefetcher._submitted_ids

        # Duplicate submit should be no-op
        prefetcher.submit(meta_cached)

        # Stop prefetcher
        prefetcher.stop()
        assert not prefetcher._worker_thread.is_alive()


def test_causal_rule_deduplication_and_empirical_reinforcement() -> None:
    """Verify that confirming identical causal invariants increments support count instead of creating duplicates."""
    from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
    from plugins.developmental_adapter.causal_discovery import (
        CausalHypothesis,
        InterventionalCausalDiscoveryEngine,
    )
    from plugins.developmental_adapter.environment import BabyWorldEnvironment
    from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter
    from plugins.developmental_adapter.types import BabyActionType

    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    env = BabyWorldEnvironment(seed=42)
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)

    hyp1 = CausalHypothesis(
        action=BabyActionType.PUSH,
        variable="mass_sensation",
        operator="<",
        value=3.0,
        consequence="MOVES",
        confidence=0.95,
        supporting_episodes=["obj_1", "obj_2"],
    )
    engine._induce_causal_rule_into_substrate(hyp1)

    assert len(substrate.causal_rules) == 1
    assert substrate.causal_rules[0]["empirical_support_count"] == 2
    assert substrate.causal_rules[0]["rule_id"] == "causal_rule_1"

    # Induce identical invariant again
    hyp2 = CausalHypothesis(
        action=BabyActionType.PUSH,
        variable="mass_sensation",
        operator="<",
        value=3.0,
        consequence="MOVES",
        confidence=1.0,
        supporting_episodes=["obj_3"],
    )
    engine._induce_causal_rule_into_substrate(hyp2)

    # Substrate should still have exactly 1 rule, but reinforced with increased support
    assert len(substrate.causal_rules) == 1
    assert substrate.causal_rules[0]["empirical_support_count"] == 3
    assert substrate.causal_rules[0]["confidence"] == 1.0


def test_ing_nouns_classification_as_blocks() -> None:
    """Verify that -ing nouns (morning, thing, spring, building) are classified as NOUN blocks, not verbs."""
    from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
    from plugins.developmental_adapter.textbook_curriculum import (
        TextbookChapter,
        TextbookCurriculumCurator,
        TextbookSection,
        TextbookSectionType,
    )
    from plugins.developmental_adapter.types import LexicalCategory

    student_substrate = create_blank_brain_substrate()
    curator = TextbookCurriculumCurator()

    # Mock student with grounding engine
    from plugins.developmental_adapter.environment import BabyWorldEnvironment
    from plugins.developmental_adapter.language_grounding import LanguageGroundingEngine

    class DummyStudent:
        def __init__(self) -> None:
            self.substrate = student_substrate
            self.grounding_engine = LanguageGroundingEngine(
                student_substrate, BabyWorldEnvironment()
            )

    student = DummyStudent()
    sec = TextbookSection(
        section_id="def_1",
        title="Definitions",
        section_type=TextbookSectionType.DEFINITIONS,
        raw_text="",
        structured_payload={
            "glossary": {
                "morning": "The early part of the day.",
                "spring": "An elastic mechanical coil storing potential energy.",
                "building": "A stationary physical architectural structure.",
                "accelerate": "To increase velocity over unit time.",
            }
        },
    )
    ch = TextbookChapter(chapter_id="ch_nouns", title="Nouns Test", grade_level=3, sections=[sec])
    curator.teach_chapter(student, ch)

    lex = student.grounding_engine.lexicon
    assert "morning" in lex
    assert lex["morning"].category == LexicalCategory.NOUN
    assert "spring" in lex
    assert lex["spring"].category == LexicalCategory.NOUN
    assert "building" in lex
    assert lex["building"].category == LexicalCategory.NOUN
    assert "accelerate" in lex
    assert lex["accelerate"].category == LexicalCategory.VERB


def test_multi_shape_puzzle_compilation() -> None:
    """Verify that simulation puzzles compile both active and ambient shapes."""
    from plugins.developmental_adapter.textbook_curriculum import (
        TextbookSection,
        TextbookSectionType,
        TextbookSimulationCompiler,
    )
    from plugins.developmental_adapter.types import BabyObjectType

    sec_block = TextbookSection(
        section_id="sec_block",
        title="Block manipulation",
        section_type=TextbookSectionType.WORKED_PROBLEM,
        raw_text="",
        structured_payload={"instruction": "push red block inside box", "subject_shape": "block"},
    )
    puzzle = TextbookSimulationCompiler.compile_puzzle(sec_block)
    shapes = {obj.object_type for obj in puzzle.initial_objects.values()}
    assert BabyObjectType.BLOCK in shapes
    assert BabyObjectType.BALL in shapes  # Ambient ball included
    assert BabyObjectType.TOOL in shapes
    assert BabyObjectType.BOX in shapes


def test_multi_term_socratic_probing() -> None:
    """Verify that vocab_probes parameter enables multi-term Socratic probing across glossary."""
    from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
    from plugins.developmental_adapter.environment import BabyWorldEnvironment
    from plugins.developmental_adapter.language_grounding import LanguageGroundingEngine
    from plugins.developmental_adapter.textbook_curriculum import (
        TextbookChapter,
        TextbookCurriculumCurator,
        TextbookSection,
        TextbookSectionType,
    )

    substrate = create_blank_brain_substrate()
    engine = LanguageGroundingEngine(substrate, BabyWorldEnvironment())

    class DummyStudent:
        def __init__(self) -> None:
            self.substrate = substrate
            self.grounding_engine = engine

    student = DummyStudent()
    curator = TextbookCurriculumCurator()

    sec = TextbookSection(
        section_id="sec_def",
        title="Glossary",
        section_type=TextbookSectionType.DEFINITIONS,
        raw_text="",
        structured_payload={
            "glossary": {
                "force": "Interaction causing acceleration.",
                "inertia": "Resistance to change in motion.",
                "velocity": "Vector rate of displacement.",
            }
        },
    )
    ch = TextbookChapter(chapter_id="ch_multi", title="Multi Term", grade_level=3, sections=[sec])
    curator.teach_chapter(student, ch)

    # 1 vocab probe (default) -> exactly 1 question
    q_single = curator.conduct_chapter_examination(student, ch, vocab_probes=1)
    assert len(q_single) == 1
    assert "force" in q_single[0].question_text

    # 3 vocab probes -> exactly 3 questions
    q_multi = curator.conduct_chapter_examination(student, ch, vocab_probes=3)
    assert len(q_multi) == 3
    q_texts = [q.question_text for q in q_multi]
    assert any("force" in t for t in q_texts)
    assert any("inertia" in t for t in q_texts)
    assert any("velocity" in t for t in q_texts)
