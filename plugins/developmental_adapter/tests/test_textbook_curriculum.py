"""Unit tests for the Textbook Curriculum Ingestion Pipeline (A23)."""

from __future__ import annotations

from plugins.developmental_adapter.textbook_curriculum import (
    TextbookCurriculumCurator,
    TextbookSectionType,
    TextbookSimulationCompiler,
)
from plugins.developmental_adapter.types import BabyRelationType


def test_textbook_markdown_parsing():
    """Verify structured educational Markdown is correctly parsed into typed sections."""
    chapter = TextbookCurriculumCurator.create_sample_physics_chapter()

    assert chapter.chapter_id == "physics_ch3"
    assert "Mechanics, Tools, and Relational Systems" in chapter.title
    assert len(chapter.sections) == 4

    def_sec = chapter.get_section(TextbookSectionType.DEFINITIONS)
    assert def_sec is not None
    assert "box" in def_sec.structured_payload["glossary"]
    assert "stick" in def_sec.structured_payload["glossary"]
    assert "pull" in def_sec.structured_payload["glossary"]

    prob_sec = chapter.get_section(TextbookSectionType.WORKED_PROBLEM)
    assert prob_sec is not None
    assert "pull green ball inside box" in prob_sec.structured_payload["instruction"]

    analogy_sec = chapter.get_section(TextbookSectionType.ANALOGY_SCHEMA)
    assert analogy_sec is not None
    assert analogy_sec.structured_payload["schema_type"] == "containment"

    exam_sec = chapter.get_section(TextbookSectionType.EXAM_CHALLENGE)
    assert exam_sec is not None
    assert exam_sec.structured_payload["has_trick_question"] is True


def test_textbook_simulation_compilation_and_execution():
    """Verify textbook worked problem compiles into an executable simulation state and is solved."""
    from plugins.developmental_adapter.school import CognitiveSchool

    student = CognitiveSchool(seed=42).student
    chapter = TextbookCurriculumCurator.create_sample_physics_chapter()
    prob_sec = chapter.get_section(TextbookSectionType.WORKED_PROBLEM)
    assert prob_sec is not None

    puzzle = TextbookSimulationCompiler.compile_puzzle(prob_sec)

    assert "target_green_ball" in puzzle.initial_objects
    assert "reach_stick" in puzzle.initial_objects
    assert "storage_box" in puzzle.initial_objects

    assert puzzle.target_goal.predicate == BabyRelationType.INSIDE
    assert puzzle.target_goal.subject_id == "target_green_ball"
    assert puzzle.target_goal.target_id == "storage_box"

    # Verify student plans and executes solution in BabyWorld
    eval_res = TextbookSimulationCompiler.verify_student_solution(student, puzzle)
    assert eval_res["is_success"] is True
    assert eval_res["plan_length"] >= 2
    assert eval_res["confidence"] > 0.0


def test_full_textbook_curriculum_and_socratic_examination():
    """Verify teaching directly from a textbook chapter and administering an un-mocked exam."""
    from plugins.developmental_adapter.school import CognitiveSchool

    school = CognitiveSchool(seed=42)
    student = school.student
    teacher = school.teacher
    chapter = TextbookCurriculumCurator.create_sample_physics_chapter()

    # Pre-prime student with foundational vocabulary (simulating earlier lessons)
    teacher.conduct_kindergarten(student)

    # Teach the textbook chapter
    teach_results = teacher.teach_from_textbook(student, chapter)
    assert teach_results["sections_processed"] >= 3

    # Administer the textbook Socratic exam
    assessment = teacher.conduct_textbook_exam(student, chapter)

    assert assessment.total_questions == 4
    assert assessment.correct_count >= 3
    assert assessment.accuracy >= 0.75
    assert assessment.mean_brier_score <= 0.20
    assert assessment.letter_grade in ["A+ (Summa Cum Laude)", "A", "B"]

    # Verify un-mocked question breakdown
    q_texts = [q.question_text for q in assessment.question_results]
    assert any("Glossary Recall" in q for q in q_texts)
    assert any("Simulation Challenge" in q for q in q_texts)
    assert any("Analogy" in q for q in q_texts)
    assert any("Conceptual Defense" in q for q in q_texts)
