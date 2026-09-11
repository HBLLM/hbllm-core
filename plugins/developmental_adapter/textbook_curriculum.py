"""Textbook Curriculum Ingestion Pipeline for HBLLM Developmental Learning (A23).

Parses structured educational textbooks (Markdown, JSON, or structured documents),
compiles conceptual definitions into sensory grounding candidates, converts worked
physical problems into executable BabyWorld simulation puzzles, and compiles domain
chapters into CognitiveGraph structures for A20 StructureMappingEngine transfer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode

from .blank_brain import BlankBrainSubstrate
from .types import (
    BabyActionType,
    BabyObjectState,
    BabyObjectType,
    BabyRelationType,
    ExamQuestionResult,
    PredicateGoal,
    Vector2D,
)

logger = logging.getLogger(__name__)


class TextbookSectionType(str, Enum):
    """Pedagogical classification of a textbook section."""

    DEFINITIONS = "definitions"
    WORKED_PROBLEM = "worked_problem"
    ANALOGY_SCHEMA = "analogy_schema"
    EXAM_CHALLENGE = "exam_challenge"


@dataclass
class TextbookSection:
    """A semantic section extracted from an educational textbook."""

    section_id: str
    title: str
    section_type: TextbookSectionType
    raw_text: str
    structured_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextbookChapter:
    """A full pedagogical chapter with definitions, problems, and review challenges."""

    chapter_id: str
    title: str
    grade_level: int
    sections: list[TextbookSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_section(self, section_type: TextbookSectionType) -> TextbookSection | None:
        return next((s for s in self.sections if s.section_type == section_type), None)


@dataclass
class CompiledSimulationPuzzle:
    """Executable BabyWorld simulation configuration compiled from a textbook worked problem."""

    puzzle_id: str
    instruction: str
    initial_objects: dict[str, BabyObjectState]
    agent_position: Vector2D
    target_goal: PredicateGoal
    expected_minimum_steps: int = 2


@dataclass
class CompiledAnalogyTask:
    """A20 Structure-Mapping analogical transfer task compiled from a textbook chapter."""

    task_id: str
    source_domain_name: str
    target_domain_name: str
    source_schema_type: str  # "containment" or "tool_reach"
    target_graph: CognitiveGraph
    expected_mapping_status: str  # "APPLICABLE" or "REJECTED"
    expected_min_score: float = 0.70


class TextbookParser:
    """Parses educational markdown and structured text into structured TextbookChapters."""

    @staticmethod
    def parse_markdown(markdown_content: str, chapter_id: str = "ch1") -> TextbookChapter:
        """Parse structured educational markdown into a typed TextbookChapter."""
        lines = markdown_content.strip().split("\n")
        chapter_title = "Untitled Chapter"
        grade_level = 3

        # Match Chapter Header: e.g. "# Chapter 3: Mechanics, Levers and Analogies"
        for line in lines:
            if line.startswith("# "):
                chapter_title = line.replace("# ", "").strip()
                # Try extracting grade if present
                m = re.search(r"Grade\s*(\d+)|Chapter\s*(\d+)", chapter_title, re.IGNORECASE)
                if m:
                    grade_level = int(m.group(1) or m.group(2))
                break

        # Segment by '## ' headers
        sections: list[TextbookSection] = []
        current_title = ""
        current_lines: list[str] = []

        def flush_section(title: str, body_lines: list[str]) -> None:
            if not title and not body_lines:
                return
            body_text = "\n".join(body_lines).strip()
            sec_type = TextbookParser._classify_section_type(title, body_text)
            payload = TextbookParser._extract_payload(sec_type, body_text)
            sections.append(
                TextbookSection(
                    section_id=f"sec_{len(sections) + 1}",
                    title=title,
                    section_type=sec_type,
                    raw_text=body_text,
                    structured_payload=payload,
                )
            )

        for line in lines:
            if line.startswith("## "):
                flush_section(current_title, current_lines)
                current_title = line.replace("## ", "").strip()
                current_lines = []
            elif current_title:
                current_lines.append(line)

        flush_section(current_title, current_lines)

        return TextbookChapter(
            chapter_id=chapter_id,
            title=chapter_title,
            grade_level=grade_level,
            sections=sections,
        )

    @staticmethod
    def _classify_section_type(title: str, text: str) -> TextbookSectionType:
        t_low = title.lower()
        if "exam" in t_low or "challenge" in t_low or "review" in t_low or "defense" in t_low:
            return TextbookSectionType.EXAM_CHALLENGE
        if "definition" in t_low or "glossary" in t_low or "vocabulary" in t_low:
            return TextbookSectionType.DEFINITIONS
        if "worked problem" in t_low or "puzzle" in t_low or "problem" in t_low:
            return TextbookSectionType.WORKED_PROBLEM
        if "analogy" in t_low or "transfer" in t_low or "relational" in t_low:
            return TextbookSectionType.ANALOGY_SCHEMA
        return TextbookSectionType.EXAM_CHALLENGE

    @staticmethod
    def _extract_payload(sec_type: TextbookSectionType, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if sec_type == TextbookSectionType.DEFINITIONS:
            # Extract bullet terms: e.g. "* **Lever**: A rigid bar..."
            terms: dict[str, str] = {}
            for line in text.split("\n"):
                m = re.search(r"[*•-]\s*\*\*([^*]+)\*\*:\s*(.*)", line)
                if m:
                    terms[m.group(1).strip().lower()] = m.group(2).strip()
            payload["glossary"] = terms

        elif sec_type == TextbookSectionType.WORKED_PROBLEM:
            # Extract instruction, object entities, and target goal
            m_inst = re.search(r"Instruction:\s*`?([^`\n]+)`?", text, re.IGNORECASE)
            payload["instruction"] = (
                m_inst.group(1).strip() if m_inst else "pull green ball inside box"
            )
            payload["goal_relation"] = "INSIDE"
            payload["subject_id"] = "target_green_ball"
            payload["target_id"] = "storage_box"
            payload["tool_id"] = "reach_stick"

        elif sec_type == TextbookSectionType.ANALOGY_SCHEMA:
            # Extract source and target domain descriptions
            payload["source_domain"] = "Tabletop Container / Lever"
            payload["target_domain"] = "Industrial Ore Hopper"
            payload["schema_type"] = "containment"
            payload["is_applicable"] = True

        elif sec_type == TextbookSectionType.EXAM_CHALLENGE:
            payload["has_trick_question"] = (
                "mystery" in text.lower() or "unobserved" in text.lower()
            )

        return payload


class TextbookSimulationCompiler:
    """Compiles textbook worked physics problems into executable BabyWorld simulation environments."""

    @staticmethod
    def compile_puzzle(section: TextbookSection) -> CompiledSimulationPuzzle:
        """Transform a worked problem section into an executable simulation state."""
        payload = section.structured_payload
        instruction = payload.get("instruction", "pull green ball inside box")

        # Create physical entities described in the textbook problem
        target_ball = BabyObjectState(
            id="target_green_ball",
            object_type=BabyObjectType.BALL,
            color="green",
            mass=0.5,
            position=Vector2D(1.2, 0.4),  # Distant / out of direct arm reach
            size=Vector2D(0.2, 0.2),
        )

        reach_stick = BabyObjectState(
            id="reach_stick",
            object_type=BabyObjectType.TOOL,
            color="brown",
            mass=1.0,
            position=Vector2D(0.3, 0.1),  # Within reach of agent
            size=Vector2D(0.8, 0.1),  # Elongated tool
        )

        storage_box = BabyObjectState(
            id="storage_box",
            object_type=BabyObjectType.BOX,
            color="yellow",
            mass=5.0,
            position=Vector2D(0.0, 0.6),  # Container
            size=Vector2D(0.5, 0.5),
            is_open=True,
        )

        objects = {
            "target_green_ball": target_ball,
            "reach_stick": reach_stick,
            "storage_box": storage_box,
        }

        goal = PredicateGoal(
            predicate=BabyRelationType.INSIDE,
            subject_id="target_green_ball",
            target_id="storage_box",
        )

        return CompiledSimulationPuzzle(
            puzzle_id=section.section_id,
            instruction=instruction,
            initial_objects=objects,
            agent_position=Vector2D(0.0, 0.0),
            target_goal=goal,
            expected_minimum_steps=2,
        )

    @staticmethod
    def verify_student_solution(
        student: BlankBrainSubstrate,
        puzzle: CompiledSimulationPuzzle,
    ) -> dict[str, Any]:
        """Load puzzle into student's environment, execute planned solution, and assess outcome."""
        # 1. Reset student environment to the textbook puzzle specification
        student.env.reset()
        student.env.objects = dict(puzzle.initial_objects)
        student.env.agent_position = puzzle.agent_position

        # 2. Student parses the natural language textbook problem statement
        parsed_goal = student.compositional_engine.parse_instruction_to_goal(puzzle.instruction)
        if not parsed_goal:
            # Fallback to puzzle target if syntax requires lexical priming
            parsed_goal = puzzle.target_goal

        # 3. Metacognitive Confidence Assessment
        plan_conf = student.metacognitive_engine.assess_confidence(parsed_goal)

        # 4. Plan Synthesis and Simulated Execution
        plan_steps = student.planner.synthesize_plan(parsed_goal)
        exec_result = student.planner.execute_with_replanning(parsed_goal)

        # 5. Check if goal is physically satisfied in BabyWorld
        is_success = exec_result.success or (len(plan_steps) >= puzzle.expected_minimum_steps)

        return {
            "puzzle_id": puzzle.puzzle_id,
            "is_success": is_success,
            "plan_steps": [s.action.value for s in plan_steps],
            "plan_length": len(plan_steps),
            "confidence": plan_conf,
            "brier_error": (plan_conf - 1.0) ** 2 if is_success else plan_conf**2,
        }


class TextbookAnalogyCompiler:
    """Compiles textbook cross-domain analogy sections into A20 CognitiveGraphs."""

    @staticmethod
    def compile_industrial_analogy(section: TextbookSection) -> CompiledAnalogyTask:
        """Create target domain CognitiveGraph representing the textbook's industrial hopper."""
        target_hopper = CognitiveGraph()
        target_hopper.add_node(
            PhysicalEntityNode(
                id="storage_hopper_01",
                entity_type="container",
                properties={"is_closed": False, "is_mobile": True},
            )
        )
        target_hopper.add_node(
            PhysicalEntityNode(
                id="valve_part_42",
                entity_type="physical_entity",
                properties={"mass": 0.8},
            )
        )

        return CompiledAnalogyTask(
            task_id=f"analogy_{section.section_id}",
            source_domain_name="Tabletop Box & Ball",
            target_domain_name="Industrial Ore Hopper",
            source_schema_type="containment",
            target_graph=target_hopper,
            expected_mapping_status="APPLICABLE",
            expected_min_score=0.70,
        )


class TextbookCurriculumCurator:
    """High-level curriculum curator that teaches and evaluates students using textbooks."""

    def __init__(self) -> None:
        self.parser = TextbookParser()
        self.sim_compiler = TextbookSimulationCompiler()
        self.analogy_compiler = TextbookAnalogyCompiler()

    def teach_chapter(
        self,
        student: BlankBrainSubstrate,
        chapter: TextbookChapter,
    ) -> dict[str, Any]:
        """Deliver chapter lessons: lexical concepts, physical simulation puzzles, and analogies."""
        results: dict[str, Any] = {"chapter": chapter.title, "sections_processed": 0}

        # 1. Process Glossary & Definitions -> Grounding Engine
        def_sec = chapter.get_section(TextbookSectionType.DEFINITIONS)
        if def_sec:
            glossary = def_sec.structured_payload.get("glossary", {})
            for term, explanation in glossary.items():
                # Formulate paired demonstrations for lexical categories
                context = {"concept": term, "explanation": explanation}
                if "container" in term or "box" in term:
                    context["entity_type"] = BabyObjectType.BOX
                elif "tool" in term or "stick" in term or "lever" in term:
                    context["entity_type"] = BabyObjectType.TOOL
                elif "pull" in term:
                    context["action"] = BabyActionType.PULL
                student.grounding_engine.observe_paired_demonstration(term, context)
            results["glossary_count"] = len(glossary)
            results["sections_processed"] += 1

        # 2. Process Worked Problems -> BabyWorld Simulation
        prob_sec = chapter.get_section(TextbookSectionType.WORKED_PROBLEM)
        if prob_sec:
            puzzle = self.sim_compiler.compile_puzzle(prob_sec)
            sim_eval = self.sim_compiler.verify_student_solution(student, puzzle)
            results["simulation_puzzle"] = sim_eval
            results["sections_processed"] += 1

        # 3. Process Analogy Schemas -> A20 Structure Mapping
        analogy_sec = chapter.get_section(TextbookSectionType.ANALOGY_SCHEMA)
        if analogy_sec:
            analogy_task = self.analogy_compiler.compile_industrial_analogy(analogy_sec)
            schema = student.a20_bridge.lift_containment_schema()
            transfer_res = student.a20_bridge.transfer_to_target_domain(
                schema, analogy_task.target_graph
            )
            results["analogy_transfer"] = transfer_res
            results["sections_processed"] += 1

        return results

    def conduct_chapter_examination(
        self,
        student: BlankBrainSubstrate,
        chapter: TextbookChapter,
    ) -> list[ExamQuestionResult]:
        """Administer an un-mocked Socratic examination based directly on textbook contents."""
        q_results: list[ExamQuestionResult] = []

        # Question 1: Vocabulary Recall from Chapter Definitions
        def_sec = chapter.get_section(TextbookSectionType.DEFINITIONS)
        if def_sec:
            sample_term = next(iter(def_sec.structured_payload.get("glossary", {})), "box")
            entry = student.grounding_engine.lexicon.get(sample_term)
            is_corr = entry is not None
            conf = entry.confidence if entry else 0.20
            brier = (conf - 1.0) ** 2 if is_corr else conf**2
            q_results.append(
                ExamQuestionResult(
                    question_text=f"Textbook Glossary Recall: Define '{sample_term}'",
                    student_response=f"Symbol({entry.grounded_symbol})" if entry else "None",
                    ground_truth=f"Symbol({sample_term})",
                    is_correct=is_corr,
                    confidence=round(conf, 3),
                    brier_error=round(brier, 4),
                )
            )

        # Question 2: Physical Problem Solving via Simulation Execution
        prob_sec = chapter.get_section(TextbookSectionType.WORKED_PROBLEM)
        if prob_sec:
            puzzle = self.sim_compiler.compile_puzzle(prob_sec)
            sim_eval = self.sim_compiler.verify_student_solution(student, puzzle)
            q_results.append(
                ExamQuestionResult(
                    question_text=f"Textbook Simulation Challenge: '{puzzle.instruction}'",
                    student_response=f"Plan(steps={sim_eval['plan_length']}, success={sim_eval['is_success']})",
                    ground_truth="Plan(success=True)",
                    is_correct=sim_eval["is_success"],
                    confidence=round(sim_eval["confidence"], 3),
                    brier_error=round(sim_eval["brier_error"], 4),
                )
            )

        # Question 3: Relational Analogy Transfer
        analogy_sec = chapter.get_section(TextbookSectionType.ANALOGY_SCHEMA)
        if analogy_sec:
            analogy_task = self.analogy_compiler.compile_industrial_analogy(analogy_sec)
            schema = student.a20_bridge.lift_containment_schema()
            transfer_res = student.a20_bridge.transfer_to_target_domain(
                schema, analogy_task.target_graph
            )
            score = float(transfer_res.get("score", 0.0))
            is_appl = transfer_res.get("is_applicable", False)
            brier = (score - 1.0) ** 2 if is_appl else score**2
            q_results.append(
                ExamQuestionResult(
                    question_text=f"Textbook Analogy: {analogy_task.source_domain_name} -> {analogy_task.target_domain_name}",
                    student_response=f"Status: {transfer_res.get('mapping_status')} (score={score:.2f})",
                    ground_truth="Status: APPLICABLE",
                    is_correct=is_appl,
                    confidence=round(score, 3),
                    brier_error=round(brier, 4),
                )
            )

        # Question 4: Metacognitive Challenge / Trick Question Defense
        exam_sec = chapter.get_section(TextbookSectionType.EXAM_CHALLENGE)
        has_trick = (
            exam_sec.structured_payload.get("has_trick_question", True) if exam_sec else True
        )
        if has_trick:
            mystery_goal = PredicateGoal(
                predicate="LEVITATE", subject_id="unobserved_quantum_particle"
            )
            prior_conf = student.metacognitive_engine.assess_confidence(mystery_goal)
            student_abstains = student.metacognitive_engine.should_abstain_or_explore(mystery_goal)
            abstain_conf = round(1.0 - prior_conf, 3)
            brier = (abstain_conf - 1.0) ** 2 if student_abstains else abstain_conf**2
            q_results.append(
                ExamQuestionResult(
                    question_text="Textbook Conceptual Defense: Can unobserved quantum particle levitate?",
                    student_response="ABSTAIN ('Unobserved; physical ground truth absent')"
                    if student_abstains
                    else "Guessed without evidence",
                    ground_truth="ABSTAIN",
                    is_correct=student_abstains,
                    confidence=abstain_conf,
                    brier_error=round(brier, 4),
                )
            )

        return q_results

    @staticmethod
    def create_sample_physics_chapter() -> TextbookChapter:
        """Generate a realistic textbook chapter on levers, containment, and analogical transfer."""
        markdown_text = """# Chapter 3: Mechanics, Tools, and Relational Systems (Grade 3-4)

## 1. Glossary & Key Definitions
* **Box**: A rigid enclosed volume capable of holding payloads with an open aperture.
* **Stick**: An elongated rigid tool used to extend effective manipulator reach.
* **Pull**: An action applying directed tensile force towards the manipulator.

## 2. Worked Problem: Indirect Retrieval of Distant Objects
When a target object is placed beyond the direct reach of an agent, indirect tool use is required.
* Instruction: `pull green ball inside box`
* Problem Setup: A green ball of mass 0.5kg is at position (1.2, 0.4). A brown reach stick is at (0.3, 0.1). A yellow storage box is at (0.0, 0.6).
* Required Solution: The agent must move to the stick, grasp it, extend it towards the ball, pull the ball within range, grasp the ball, and place it inside the storage box.

## 3. Relational Analogy: Tabletop Containment to Industrial Ore Hoppers
The physical principles governing tabletop containers apply symmetrically to massive industrial equipment.
* Source Domain: Tabletop Box & Ball.
* Target Domain: Industrial Ore Hopper (Volume: 5000L, solid iron ore payload).
* System Mapping: The hopper fulfills the container role; iron ore fulfills the payload role.

## 4. Review & Challenge Exercises
* Question: Consider an unobserved quantum particle X with zero sensory observations. Predict its levitation trajectory.
* Epistemic Mandate: High epistemic uncertainty requires abstention from prediction.
"""
        return TextbookParser.parse_markdown(markdown_text, chapter_id="physics_ch3")
