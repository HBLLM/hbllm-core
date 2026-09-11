"""Unit Tests for Pedagogical Teacher and Cognitive School Curriculum (Milestone A23).

Tests Vygotskian scaffolding, Socratic counter-examples, grade examinations,
and full academic progression from blank-brain Kindergarten to High School graduation.
"""

from __future__ import annotations

import pytest

from plugins.developmental_adapter.school import CognitiveSchool, GraduationTranscript
from plugins.developmental_adapter.teacher import (
    GradeAssessment,
    GradeLevel,
)


class TestPedagogicalSchool:
    """Test suite for the Cognitive School and Pedagogical Teacher."""

    @pytest.fixture
    def school(self) -> CognitiveSchool:
        """Create a standard CognitiveSchool instance."""
        return CognitiveSchool(student_name="Test Student", teacher_name="Dr. Vygotsky", seed=42)

    def test_pedagogical_teacher_initialization(self, school: CognitiveSchool) -> None:
        """Verify teacher and blank-brain student are correctly instantiated."""
        assert school.teacher.name == "Dr. Vygotsky"
        assert school.student.substrate is not None
        assert len(school.student.substrate.causal_rules) == 0
        assert len(school.student.substrate.affordances) == 0
        assert len(school.student.substrate.lexical_mapping) == 0

    def test_kindergarten_lexical_grounding(self, school: CognitiveSchool) -> None:
        """Verify Grade 1: Ostensive naming fast-maps lexicon and passes exam."""
        assessment: GradeAssessment = school.teacher.conduct_kindergarten(school.student)

        assert assessment.grade_level == GradeLevel.KINDERGARTEN
        assert assessment.accuracy >= 0.90
        assert assessment.mean_brier_score <= 0.05
        assert "A" in assessment.letter_grade
        assert assessment.total_questions == 6
        assert assessment.correct_count >= 5

        # Verify acquired vocabulary in student
        assert "red" in school.student.grounding_engine.lexicon
        assert "ball" in school.student.grounding_engine.lexicon
        assert "inside" in school.student.grounding_engine.lexicon

    def test_elementary_physics_socratic_counter_example(self, school: CognitiveSchool) -> None:
        """Verify Grade 2: Socratic counter-example breaks confounder and passes exam."""
        # First enroll and ground vocabulary in Kindergarten
        school.teacher.conduct_kindergarten(school.student)

        # Conduct Elementary Physics
        assessment: GradeAssessment = school.teacher.conduct_elementary_physics(school.student)

        assert assessment.grade_level == GradeLevel.ELEMENTARY
        assert assessment.accuracy == 1.0  # All Level 2 unseen entities predicted correctly
        assert assessment.mean_brier_score <= 0.05
        assert "A+" in assessment.letter_grade
        assert assessment.total_questions == 3

        # Verify student learned true mass rule rather than color correlate
        discovered_rules = school.student.substrate.causal_rules
        assert any(
            "mass" in str(r).lower() or "resistance" in str(r).lower() for r in discovered_rules
        )

    def test_middle_school_compositional_tool_planning(self, school: CognitiveSchool) -> None:
        """Verify Grade 3: Compositional instruction parsing and tool execution."""
        school.teacher.conduct_kindergarten(school.student)
        school.teacher.conduct_elementary_physics(school.student)

        assessment: GradeAssessment = school.teacher.conduct_middle_school_planning(school.student)

        assert assessment.grade_level == GradeLevel.MIDDLE_SCHOOL
        assert assessment.accuracy == 1.0
        assert assessment.mean_brier_score <= 0.05
        assert "A+" in assessment.letter_grade
        assert assessment.total_questions == 3
        assert assessment.correct_count == 3

    def test_high_school_transfer_and_abstention(self, school: CognitiveSchool) -> None:
        """Verify Grade 4: A20 analogy transfer, negative rejection, and calibrated abstention."""
        assessment: GradeAssessment = school.teacher.conduct_high_school_transfer(school.student)

        assert assessment.grade_level == GradeLevel.HIGH_SCHOOL
        assert assessment.accuracy == 1.0
        assert assessment.mean_brier_score <= 0.05
        assert "A+" in assessment.letter_grade
        assert assessment.total_questions == 4
        assert assessment.correct_count == 4

        # Verify question 4 (trick question / unobserved variable) abstained without guessing
        trick_q = assessment.question_results[3]
        assert "ABSTAIN" in trick_q.student_response
        assert trick_q.is_correct is True

    def test_cognitive_school_full_curriculum_graduation(self, school: CognitiveSchool) -> None:
        """Verify end-to-end graduation through all 4 grades with honors."""
        transcript: GraduationTranscript = school.run_full_curriculum()

        assert len(transcript.grades) == 4
        assert transcript.cumulative_gpa >= 3.8
        assert transcript.cumulative_accuracy >= 0.95
        assert transcript.cumulative_brier_score <= 0.05
        assert transcript.graduated_with_honors is True
        assert "SUMMA CUM LAUDE" in transcript.diploma_text
        assert "HBLLM DEVELOPMENTAL COGNITIVE ACADEMY" in transcript.diploma_text
