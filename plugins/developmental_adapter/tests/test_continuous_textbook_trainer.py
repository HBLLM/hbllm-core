"""Unit and Integration tests for ContinuousTextbookSchoolTrainer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from plugins.developmental_adapter.curriculum_fetcher import CurriculumFetcher
from plugins.developmental_adapter.trainer import (
    ContinuousTextbookSchoolTrainer,
    TextbookSchoolConfig,
)


def test_continuous_textbook_training_multi_chapter() -> None:
    """Verify multi-chapter textbook training loop, assessments, sleep consolidation, and checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        fetcher = CurriculumFetcher()
        all_chapters = fetcher.list_available_chapters()
        assert len(all_chapters) >= 3

        # Select first 3 chapters for test
        selected_paths = all_chapters[:3]

        cfg = TextbookSchoolConfig(
            student_name="Test Student",
            curriculum_paths=selected_paths,
            episodes_per_chapter=2,
            checkpoint_dir=ckpt_dir,
            enable_sleep_consolidation=True,
            seed=42,
        )

        trainer = ContinuousTextbookSchoolTrainer(config=cfg)
        summary = trainer.train()

        assert summary.total_chapters_trained == 3
        assert summary.final_accuracy >= 0.80
        assert summary.final_gpa >= 3.0
        assert summary.final_brier_score <= 0.20
        assert summary.mean_backward_transfer >= -0.01
        assert "HBLLM DEVELOPMENTAL COGNITIVE ACADEMY" in summary.diploma_text

        # Verify each chapter produced report and checkpoint
        assert len(summary.chapter_reports) == 3
        for rep in summary.chapter_reports:
            assert rep.vocabulary_count >= 6
            assert rep.assessment.total_questions == 4
            assert rep.assessment.accuracy >= 0.75
            assert Path(rep.checkpoint_path).exists()

            # Verify checkpoint contents
            p = Path(rep.checkpoint_path)
            assert (p / "checkpoint_meta.json").exists()
            assert (p / "lexicon.json").exists()
            assert (p / "causal_rules.json").exists()
            assert (p / "affordances.json").exists()
            assert (p / "relational_schemas.json").exists()
            assert (p / "curriculum_transcript.json").exists()


def test_checkpoint_resumption_and_continual_learning() -> None:
    """Verify cognitive state can be saved at Chapter 2 and resumed to train Chapter 3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        fetcher = CurriculumFetcher()
        chapters = fetcher.list_available_chapters()[:3]

        # Phase 1: Train first 2 chapters
        cfg1 = TextbookSchoolConfig(
            student_name="Resumable Student",
            curriculum_paths=chapters[:2],
            episodes_per_chapter=1,
            checkpoint_dir=ckpt_dir,
            enable_sleep_consolidation=True,
            seed=42,
        )
        trainer1 = ContinuousTextbookSchoolTrainer(config=cfg1)
        summary1 = trainer1.train()
        assert summary1.total_chapters_trained == 2

        ch2_dir = ckpt_dir / f"chapter_2_{chapters[1].stem}"
        assert ch2_dir.exists()

        # Phase 2: Resume student from Chapter 2 checkpoint and train Chapter 3
        cfg2 = TextbookSchoolConfig(
            student_name="Resumable Student",
            curriculum_paths=[chapters[2]],
            episodes_per_chapter=1,
            checkpoint_dir=ckpt_dir / "resumed",
            enable_sleep_consolidation=True,
            seed=42,
            initial_kindergarten_priming=False,  # Already primed
        )
        trainer2 = ContinuousTextbookSchoolTrainer.resume_from_checkpoint(
            checkpoint_dir=ch2_dir,
            config=cfg2,
        )

        # Restored student must retain previously learned vocabulary
        assert len(trainer2.student.grounding_engine.lexicon) >= 6

        summary2 = trainer2.train()
        assert summary2.total_chapters_trained == 1
        assert summary2.final_accuracy >= 0.75


def test_cli_runner_script() -> None:
    """Verify the CLI entry point runs and writes an exported summary transcript."""
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_out = Path(tmpdir) / "summary.json"
        ckpt_out = Path(tmpdir) / "ckpts"

        cmd = [
            sys.executable,
            "-m",
            "plugins.developmental_adapter.scripts.run_textbook_training",
            "--max-chapters",
            "2",
            "--episodes-per-chapter",
            "1",
            "--checkpoint-dir",
            str(ckpt_out),
            "--export-summary",
            str(summary_out),
        ]

        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "."},
        )
        assert res.returncode == 0, f"CLI runner failed: {res.stderr}"
        assert "HBLLM DEVELOPMENTAL COGNITIVE ACADEMY" in res.stdout
        assert summary_out.exists()

        with open(summary_out) as f:
            data = json.load(f)
            assert data["total_chapters"] == 2
            assert data["final_gpa"] >= 3.0
            assert len(data["chapter_breakdown"]) == 2
