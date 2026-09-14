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
