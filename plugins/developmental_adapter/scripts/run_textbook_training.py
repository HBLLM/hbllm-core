#!/usr/bin/env python3
"""CLI Entry Point for Long-Running Textbook Curriculum Training.

Executes sequential educational textbook training for HBLLM Developmental Core,
evaluating grounded language acquisition, BabyWorld simulation puzzles,
relational structure mapping, Socratic examinations, and dual-store sleep
consolidation with checkpoint persistence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from plugins.developmental_adapter.curriculum_fetcher import CurriculumFetcher
from plugins.developmental_adapter.trainer import (
    ContinuousTextbookSchoolTrainer,
    TextbookSchoolConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hbllm.textbook_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continuous long-running textbook curriculum training for HBLLM core."
    )
    parser.add_argument(
        "--curriculum-dir",
        type=Path,
        default=Path(__file__).parent.parent / "curriculum_data",
        help="Directory containing curriculum Markdown files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/textbook_school"),
        help="Directory where chapter checkpoints and transcripts will be stored.",
    )
    parser.add_argument(
        "--episodes-per-chapter",
        type=int,
        default=3,
        help="Number of active simulation puzzle episodes to execute per chapter.",
    )
    parser.add_argument(
        "--max-chapters",
        type=int,
        default=None,
        help="Maximum number of chapters to process (default: all available).",
    )
    parser.add_argument(
        "--disable-sleep",
        action="store_true",
        help="Disable dual-store sleep memory consolidation cycles.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to an existing chapter checkpoint to resume cognitive state from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproduction.",
    )
    parser.add_argument(
        "--student-name",
        type=str,
        default="Baby HBLLM (Textbook Scholar #001)",
        help="Identifier name for the developmental student agent.",
    )
    parser.add_argument(
        "--download-online",
        action="store_true",
        help="Dynamically download real open-access books (Gutenberg) and Wikipedia STEM articles.",
    )
    parser.add_argument(
        "--gutenberg-ids",
        nargs="*",
        default=None,
        help="Specific Project Gutenberg Book IDs to download and train on (e.g. 14474 284).",
    )
    parser.add_argument(
        "--wiki-topics",
        nargs="*",
        default=None,
        help="Specific Wikipedia topics to download and train on (e.g. 'Classical mechanics' 'Thermodynamics').",
    )
    parser.add_argument(
        "--export-summary",
        type=Path,
        default=None,
        help="Path to export final JSON summary transcript.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_memory_chapters = []

    if args.download_online or args.gutenberg_ids or args.wiki_topics:
        from plugins.developmental_adapter.raw_book_pipeline import (
            BookCurriculumItem,
            BookSourceType,
            StandardBookCatalog,
            download_and_compile_books,
        )

        catalog_items = []
        if args.gutenberg_ids:
            for gid in args.gutenberg_ids:
                catalog_items.append(
                    BookCurriculumItem(
                        item_id=f"gutenberg_{gid}",
                        source_type=BookSourceType.GUTENBERG,
                        identifier=gid,
                        title=f"Project Gutenberg Title #{gid}",
                    )
                )
        if args.wiki_topics:
            for top in args.wiki_topics:
                catalog_items.append(
                    BookCurriculumItem(
                        item_id=f"wiki_{top.lower().replace(' ', '_')}",
                        source_type=BookSourceType.WIKIPEDIA,
                        identifier=top,
                        title=f"Wikipedia: {top}",
                    )
                )
        if not catalog_items:
            catalog_items = StandardBookCatalog.get_comprehensive_curriculum()

        logger.info(f"Downloading and compiling {len(catalog_items)} online educational texts...")
        in_memory_chapters = download_and_compile_books(
            catalog=catalog_items, max_books=args.max_chapters
        )
        logger.info(f"Successfully compiled {len(in_memory_chapters)} in-memory chapters.")

        cfg = TextbookSchoolConfig(
            student_name=args.student_name,
            in_memory_chapters=in_memory_chapters,
            episodes_per_chapter=args.episodes_per_chapter,
            checkpoint_dir=args.checkpoint_dir,
            enable_sleep_consolidation=not args.disable_sleep,
            seed=args.seed,
        )
    else:
        fetcher = CurriculumFetcher(data_dir=args.curriculum_dir)
        chapter_paths = fetcher.list_available_chapters()

        if not chapter_paths:
            logger.error(f"No curriculum files (.md) found in {args.curriculum_dir}")
            return 1

        logger.info(
            f"Discovered {len(chapter_paths)} curriculum chapters in {args.curriculum_dir}: "
            f"{', '.join(p.name for p in chapter_paths)}"
        )

        cfg = TextbookSchoolConfig(
            student_name=args.student_name,
            curriculum_paths=chapter_paths,
            episodes_per_chapter=args.episodes_per_chapter,
            checkpoint_dir=args.checkpoint_dir,
            enable_sleep_consolidation=not args.disable_sleep,
            seed=args.seed,
        )

    if args.resume_from:
        logger.info(f"Resuming training from checkpoint: {args.resume_from}")
        trainer = ContinuousTextbookSchoolTrainer.resume_from_checkpoint(
            checkpoint_dir=args.resume_from,
            config=cfg,
        )
    else:
        trainer = ContinuousTextbookSchoolTrainer(config=cfg)

    summary = trainer.train(max_chapters=args.max_chapters)

    print("\n" + "=" * 70)
    print(summary.diploma_text)
    print("=" * 70)
    print(f"Total Chapters Processed: {summary.total_chapters_trained}")
    print(f"Final GPA:                {summary.final_gpa} / 4.0")
    print(f"Final Accuracy:           {summary.final_accuracy * 100:.2f}%")
    print(f"Final Mean Brier Score:   {summary.final_brier_score:.4f}")
    print(f"Backward Transfer (BWT):  {summary.mean_backward_transfer:.4f}")
    print(f"Graduated With Honors:    {summary.graduated_with_honors}")
    print(f"Final Checkpoint:         {summary.final_checkpoint_dir}")
    print("=" * 70 + "\n")

    if args.export_summary:
        summary_dict = {
            "student_name": summary.student_name,
            "total_chapters": summary.total_chapters_trained,
            "final_gpa": summary.final_gpa,
            "final_accuracy": summary.final_accuracy,
            "final_brier_score": summary.final_brier_score,
            "mean_backward_transfer": summary.mean_backward_transfer,
            "graduated_with_honors": summary.graduated_with_honors,
            "final_checkpoint": summary.final_checkpoint_dir,
            "chapter_breakdown": [
                {
                    "chapter_index": r.chapter_index,
                    "chapter_id": r.chapter_id,
                    "title": r.chapter_title,
                    "accuracy": r.assessment.accuracy,
                    "brier_score": r.assessment.mean_brier_score,
                    "grade": r.assessment.letter_grade,
                    "vocabulary_count": r.vocabulary_count,
                    "causal_rules": r.causal_rules_count,
                    "affordances": r.affordances_count,
                    "backward_transfer": r.backward_transfer,
                }
                for r in summary.chapter_reports
            ],
        }
        args.export_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_summary, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        logger.info(f"Exported final training summary to {args.export_summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
