#!/usr/bin/env python3
"""CLI Daemon for Continuous Streaming Training across Project Gutenberg Library.

Discovers books from the master Gutenberg catalog, downloads and caches them,
compiles them on-the-fly into structured developmental chapters, and trains
the HBLLM cognitive student while tracking knowledge gain dynamics.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hbllm.gutenberg_stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and train developmental core on Project Gutenberg library."
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=10,
        help="Maximum number of Gutenberg books to process in this training run.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Filter by specific domains (physics, chemistry, optics, thermodynamics, astronomy, biology, philosophy).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/gutenberg_stream"),
        help="Directory to store periodic cognitive checkpoints.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/gutenberg_cache"),
        help="Local cache directory for raw book texts.",
    )
    parser.add_argument(
        "--episodes-per-book",
        type=int,
        default=1,
        help="Number of simulation puzzle trials executed per book.",
    )
    parser.add_argument(
        "--export-report",
        type=Path,
        default=None,
        help="Path to export the final Knowledge Acquisition Report (defaults to <checkpoint-dir>/knowledge_report.md).",
    )
    parser.add_argument(
        "--student-name",
        type=str,
        default="Baby HBLLM (Gutenberg Scholar)",
        help="Name of the developmental student agent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Fetch / Load Master Gutenberg Index
    logger.info("Accessing Project Gutenberg master catalog...")
    index_text = GutenbergIndexManager.fetch_or_load_index(cache_dir=args.cache_dir)
    books = GutenbergIndexManager.parse_index(
        index_text=index_text,
        filter_domains=args.domains,
        max_books=args.max_books,
    )

    if not books:
        logger.error("No books matched the filter criteria in the Gutenberg index.")
        return 1

    logger.info(f"Discovered and scheduled {len(books)} Gutenberg books for streaming training:")
    for b in books:
        logger.info(f"  [#{b.book_id}] {b.title} (Domain: {b.domain})")

    # 2. Check for existing checkpoints to support automatic resumption
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_ckpts = sorted(
        [d for d in args.checkpoint_dir.iterdir() if d.is_dir() and "gutenberg_" in d.name],
        key=lambda p: p.stat().st_mtime,
    )
    processed_bids = set()
    for d in existing_ckpts:
        m = re.search(r"gutenberg_(\d+)", d.name)
        if m:
            processed_bids.add(int(m.group(1)))

    cfg = TextbookSchoolConfig(
        student_name=args.student_name,
        episodes_per_chapter=args.episodes_per_book,
        checkpoint_dir=args.checkpoint_dir,
        enable_sleep_consolidation=True,
        seed=args.seed,
    )

    if existing_ckpts and processed_bids:
        latest_ckpt = existing_ckpts[-1]
        logger.info(
            f"Detected {len(processed_bids)} previously completed books. "
            f"Resuming student state from latest checkpoint: {latest_ckpt.name}..."
        )
        trainer = ContinuousTextbookSchoolTrainer.resume_from_checkpoint(
            checkpoint_dir=latest_ckpt,
            config=cfg,
        )
        # Filter out already processed books
        remaining_books = [b for b in books if b.book_id not in processed_bids]
        logger.info(
            f"Remaining books to stream: {len(remaining_books)} (skipped {len(processed_bids)} already trained)."
        )
        books = remaining_books
    else:
        trainer = ContinuousTextbookSchoolTrainer(config=cfg)

    # 3. Stream & Train Book-by-Book
    print("\n" + "=" * 70)
    print("STARTING STREAMING DEVELOPMENTAL TRAINING ON PROJECT GUTENBERG")
    print("=" * 70 + "\n")

    streamer = GutenbergCorpusStreamer(cache_dir=args.cache_dir)
    tracker_state_file = args.checkpoint_dir / "tracker_state.json"
    if tracker_state_file.exists() and existing_ckpts:
        tracker = KnowledgeGainTracker.load_state(tracker_state_file)
        logger.info(
            f"Loaded existing knowledge tracker state with {len(tracker.snapshots)} historical snapshots."
        )
    else:
        tracker = KnowledgeGainTracker()

    for idx, book_meta in enumerate(books, start=len(processed_bids) + 1):
        logger.info(f"\n>>> STREAMING BOOK {idx}: #{book_meta.book_id} — {book_meta.title} <<<")
        try:
            raw_text = streamer.fetch_book_text(book_meta)
            chapter = next(streamer.stream_curriculum_chapters([book_meta]))
            trainer.train(chapters=[chapter])

            snap = tracker.record_progress(trainer, book_meta, len(raw_text))
            tracker.save_state(tracker_state_file)
            logger.info(
                f"Knowledge State after Book #{idx}: "
                f"Vocab: {snap.vocabulary_size} words | "
                f"Accuracy: {snap.mean_examination_accuracy * 100:.1f}% | "
                f"Brier: {snap.mean_brier_score:.4f} | "
                f"BWT: {snap.mean_backward_transfer:.4f}"
            )
        except Exception as e:
            logger.error(
                f"Failed to process Book #{idx} (#{book_meta.book_id} — {book_meta.title}): {e}. "
                f"Skipping to next book..."
            )

    # 4. Generate Knowledge Acquisition Report
    report = tracker.render_knowledge_report()
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70 + "\n")

    export_path = args.export_report or (args.checkpoint_dir / "knowledge_report.md")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Exported Knowledge Gain Report to {export_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
