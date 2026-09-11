"""Unit tests for Developmental Cognitive School Trainer and Checkpoint Manager."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from plugins.developmental_adapter.trainer import (
    CognitiveSchoolTrainer,
    SchoolTrainingConfig,
)


class TestSchoolTrainer:
    """Test suite for CognitiveSchoolTrainer lifecycle and persistence."""

    def test_school_training_config_defaults(self) -> None:
        cfg = SchoolTrainingConfig()
        assert cfg.num_semesters == 3
        assert cfg.episodes_per_semester == 5
        assert cfg.enable_sleep_consolidation is True
        assert cfg.student_name == "Baby HBLLM (Student #001)"

    def test_multi_semester_training_and_honors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SchoolTrainingConfig(
                num_semesters=3,
                episodes_per_semester=2,
                checkpoint_dir=Path(tmpdir) / "checkpoints",
                enable_sleep_consolidation=True,
                seed=42,
            )
            trainer = CognitiveSchoolTrainer(cfg)
            summary = trainer.train()

            assert summary.total_semesters == 3
            assert summary.final_gpa >= 3.8
            assert summary.final_brier_score <= 0.05
            assert summary.final_accuracy >= 0.95
            assert summary.mean_backward_transfer >= -0.01
            assert summary.graduated_with_honors is True
            assert "SUMMA CUM LAUDE" in summary.diploma_text

            # Verify reports
            assert len(summary.semester_reports) == 3
            rep1, rep2, rep3 = summary.semester_reports
            assert rep1.vocabulary_count >= 6
            assert rep2.vocabulary_count >= 6
            assert rep3.assessment.accuracy == 1.0

    def test_checkpoint_save_and_load_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            cfg = SchoolTrainingConfig(
                num_semesters=2,
                episodes_per_semester=2,
                checkpoint_dir=ckpt_dir,
                enable_sleep_consolidation=True,
                seed=42,
            )
            trainer = CognitiveSchoolTrainer(cfg)
            trainer.train()

            sem2_dir = ckpt_dir / "semester_2"
            assert sem2_dir.exists()

            # Check individual files
            meta_file = sem2_dir / "checkpoint_meta.json"
            lex_file = sem2_dir / "lexicon.json"
            rules_file = sem2_dir / "causal_rules.json"
            aff_file = sem2_dir / "affordances.json"
            history_file = sem2_dir / "training_history.json"

            assert meta_file.exists()
            assert lex_file.exists()
            assert rules_file.exists()
            assert aff_file.exists()
            assert history_file.exists()

            with open(meta_file) as f:
                meta = json.load(f)
                assert meta["semester"] == 2
                assert meta["student_name"] == "Baby HBLLM (Student #001)"

            with open(lex_file) as f:
                lexicon = json.load(f)
                assert "red" in lexicon or "blue" in lexicon

            # Restore student from checkpoint
            restored_student = CognitiveSchoolTrainer.load_checkpoint(sem2_dir)
            assert restored_student is not None
            assert len(restored_student.grounding_engine.lexicon) > 0
            assert len(restored_student.substrate.causal_rules) > 0

            # Verify restored student can parse instructions with restored lexicon
            goal = restored_student.compositional_engine.parse_instruction_to_goal(
                "put blue ball inside box"
            )
            assert goal is not None
