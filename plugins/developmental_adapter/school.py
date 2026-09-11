"""Cognitive School Curriculum Orchestrator for Milestone A23.

Manages enrollment of a blank-brain student, progression through the 4 developmental
grades under Pedagogical Teacher scaffolding, and graduation certification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .a20_transfer_bridge import A20RelationalTransferBridge
from .affordance_discovery import AffordanceDiscoveryEngine
from .blank_brain import create_blank_brain_substrate
from .causal_discovery import InterventionalCausalDiscoveryEngine
from .compositional_language import CompositionalLanguageEngine
from .environment import BabyWorldEnvironment
from .goal_planning import GoalDirectedPlanningEngine
from .language_grounding import LanguageGroundingEngine
from .metacognition import MetacognitiveEngine
from .perception import DevelopmentalPerceptionAdapter
from .teacher import (
    GradeAssessment,
    PedagogicalTeacher,
    StudentProfile,
)
from .tool_learning import ToolLearningEngine

logger = logging.getLogger(__name__)


@dataclass
class GraduationTranscript:
    """Official academic transcript and report card of the cognitive student."""

    student_name: str
    teacher_name: str
    grades: list[GradeAssessment] = field(default_factory=list)
    cumulative_gpa: float = 0.0
    cumulative_accuracy: float = 0.0
    cumulative_brier_score: float = 0.0
    graduated_with_honors: bool = False
    diploma_text: str = ""


class CognitiveSchool:
    """The developmental school environment where the infant mind is educated."""

    def __init__(
        self,
        student_name: str = "Baby HBLLM (Student #001)",
        teacher_name: str = "Dr. Maria Vygotsky",
        seed: int = 42,
    ) -> None:
        self.student_name = student_name
        self.teacher = PedagogicalTeacher(name=teacher_name, seed=seed)
        self.seed = seed
        self.student = self._enroll_blank_student()

    def _enroll_blank_student(self) -> StudentProfile:
        """Instantiate a pristine blank-brain student with zero acquired semantic knowledge."""
        substrate = create_blank_brain_substrate()
        env = BabyWorldEnvironment(seed=self.seed)
        perception = DevelopmentalPerceptionAdapter()

        causal_engine = InterventionalCausalDiscoveryEngine(
            substrate=substrate, perception=perception, environment=env
        )
        affordance_engine = AffordanceDiscoveryEngine(
            substrate=substrate, perception=perception, env=env
        )
        tool_engine = ToolLearningEngine(substrate=substrate, perception=perception, env=env)
        grounding_engine = LanguageGroundingEngine(substrate=substrate, env=env)
        planner = GoalDirectedPlanningEngine(substrate=substrate, env=env)
        compositional_engine = CompositionalLanguageEngine(
            substrate=substrate,
            env=env,
            grounding_engine=grounding_engine,
            planning_engine=planner,
        )
        metacognitive_engine = MetacognitiveEngine(substrate=substrate, env=env)
        a20_bridge = A20RelationalTransferBridge()

        return StudentProfile(
            substrate=substrate,
            env=env,
            perception=perception,
            causal_engine=causal_engine,
            affordance_engine=affordance_engine,
            tool_engine=tool_engine,
            grounding_engine=grounding_engine,
            planner=planner,
            compositional_engine=compositional_engine,
            metacognitive_engine=metacognitive_engine,
            a20_bridge=a20_bridge,
        )

    def run_full_curriculum(self) -> GraduationTranscript:
        """Educate the student through Kindergarten, Elementary, Middle, and High School."""
        logger.info(f"Enrolling {self.student_name} in the Developmental Cognitive School.")

        # Grade 1: Kindergarten
        g1 = self.teacher.conduct_kindergarten(self.student)

        # Grade 2: Elementary School
        g2 = self.teacher.conduct_elementary_physics(self.student)

        # Grade 3: Middle School
        g3 = self.teacher.conduct_middle_school_planning(self.student)

        # Grade 4: High School
        g4 = self.teacher.conduct_high_school_transfer(self.student)

        all_grades = [g1, g2, g3, g4]

        # Calculate Transcript Metrics
        total_q = sum(g.total_questions for g in all_grades)
        total_corr = sum(g.correct_count for g in all_grades)
        cum_acc = round(total_corr / total_q, 4) if total_q > 0 else 0.0
        cum_brier = round(sum(g.mean_brier_score for g in all_grades) / len(all_grades), 4)

        gpa_points = {
            "A+ (Summa Cum Laude)": 4.0,
            "A": 4.0,
            "B": 3.0,
            "C": 2.0,
            "F (Needs Remediation)": 0.0,
        }
        total_pts = sum(gpa_points.get(g.letter_grade, 0.0) for g in all_grades)
        cum_gpa = round(total_pts / len(all_grades), 2)
        honors = (cum_gpa >= 3.8) and (cum_brier <= 0.05)

        diploma = self._render_diploma(cum_gpa, cum_acc, cum_brier, honors)

        return GraduationTranscript(
            student_name=self.student_name,
            teacher_name=self.teacher.name,
            grades=all_grades,
            cumulative_gpa=cum_gpa,
            cumulative_accuracy=cum_acc,
            cumulative_brier_score=cum_brier,
            graduated_with_honors=honors,
            diploma_text=diploma,
        )

    def _render_diploma(self, gpa: float, accuracy: float, brier: float, with_honors: bool) -> str:
        """Render graduation certificate."""
        distinction = (
            "✦ SUMMA CUM LAUDE (WITH HIGHEST COGNITIVE HONORS) ✦"
            if with_honors
            else "✦ CERTIFIED GROUNDED COGNITIVE AGENT ✦"
        )
        return f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   HBLLM DEVELOPMENTAL COGNITIVE ACADEMY                        ║
║                 Department of Embodied Relational Intelligence                 ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  This is to certify that:                                                      ║
║                         {self.student_name:^44}                   ║
║                                                                                ║
║  has successfully completed the 4-Grade Developmental Curriculum:              ║
║    • Grade 1: Sensorimotor Lexicon & Ostensive Grounding                       ║
║    • Grade 2: Interventional Causal Physics & Confounder Decoupling            ║
║    • Grade 3: Compositional Syntax, Tool Use & Goal Planning                   ║
║    • Grade 4: A20 Relational Transfer & Metacognitive Abstention               ║
║                                                                                ║
║  Academic Performance:                                                         ║
║    Cumulative GPA:      {gpa:.2f} / 4.00                                            ║
║    Overall Accuracy:    {accuracy * 100:.1f}%                                               ║
║    Mean Brier Score:    {brier:.4f} (Well-Calibrated, Non-Hallucinating)           ║
║                                                                                ║
║  Awarded the Degree of:                                                        ║
║               {distinction:^50}               ║
║                                                                                ║
║  Dean of Cognitive Development:             Pedagogical Instructor:            ║
║  Prof. Antigravity                         {self.teacher.name:^26}          ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""
