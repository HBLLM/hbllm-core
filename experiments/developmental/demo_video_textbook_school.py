#!/usr/bin/env python3
"""Interactive Demonstration: Learning from Video Demonstrations & Textbooks in HBLLM.

Demonstrates the dual cultural-learning pipeline:
  Act I: Observational Learning from Multimodal Video (YouTube/Demonstration Stream)
         - Spoken narration time-aligned with visual keyframes & OpenCV detections
         - Fast-mapping words to physical attributes without pre-compiled lexicon
  Act II: Formal Socratic Learning from an Educational Textbook Chapter
         - Markdown chapter parsed into definitions, simulation puzzles, and analogies
         - Textbook worked problem compiled into an executable BabyWorld simulation
         - A20 Structure Mapping transfers tabletop containment to an industrial hopper
         - Epistemic defense against an unobserved trick question
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plugins.developmental_adapter.textbook_curriculum import (
    TextbookCurriculumCurator,
    TextbookSectionType,
)
from plugins.developmental_adapter.video_curriculum import VideoCurriculumCurator


def print_banner(text: str) -> None:
    line = "=" * 85
    print(f"\n{line}\n{text.center(85)}\n{line}")


def main() -> None:
    print_banner("HBLLM DEVELOPMENTAL COGNITIVE ACADEMY: VIDEO & TEXTBOOK CURRICULUM")
    print("Initializing Tabula Rasa (Blank-Brain) infant cognitive substrate...")
    from plugins.developmental_adapter.school import CognitiveSchool

    school = CognitiveSchool(
        student_name="Baby HBLLM (Student #001)",
        teacher_name="Prof. Maria Montessori",
        seed=42,
    )
    student = school.student
    teacher = school.teacher

    print(f"  • Student:          {school.student_name}")
    print(
        f"  • Initial Lexicon:  {len(student.grounding_engine.lexicon)} symbols (Completely Blank)"
    )
    print(
        f"  • Causal Graph:     {len(student.causal_engine.confirmed_causal_rules)} confirmed rules"
    )

    # =========================================================================
    # ACT I: MULTIMODAL VIDEO DEMONSTRATION
    # =========================================================================
    print_banner("ACT I: OBSERVATIONAL MULTIMODAL VIDEO DEMONSTRATION")
    print("Generating synthetic educational tabletop video and time-aligned audio narration...")

    video_curator = VideoCurriculumCurator(temporal_tolerance_sec=0.8)
    manifest, _ = video_curator.create_synthetic_educational_video(
        title="Lesson 1: Introduction to Objects, Colors, and Tabletop Dynamics",
        duration_sec=6.0,
        fps=30.0,
    )

    print(
        f"  • Video Stream:     '{manifest.title}' (Duration: {manifest.duration_sec}s, {manifest.fps} FPS)"
    )
    print(f"  • Narration Tracks: {len(manifest.audio_segments)} time-stamped spoken utterances:")
    for a in manifest.audio_segments:
        print(f'      [{a.start_sec:4.1f}s - {a.end_sec:4.1f}s] {a.speaker}: "{a.text}"')

    print("\nExecuting temporal cross-modal alignment between audio narration and visual frames...")
    aligned_pairs = video_curator.align_manifest(manifest)
    print(f"  • Synchronized {len(aligned_pairs)} multimodal demonstration events:")
    for p in aligned_pairs:
        print(
            f"      t={p.timestamp_sec:4.1f}s: Utterance '{p.utterance}' aligned with visual context: {p.visual_context}"
        )

    print("\nBaby student attends video lecture (running 3 observational passes)...")
    for _ in range(3):
        teacher.teach_from_video(student, manifest)

    print("\n[Lexicon Inspection after Video Observation]:")
    for word, entry in sorted(student.grounding_engine.lexicon.items()):
        print(
            f"  ✓ Grounded '{word}' -> Symbol: '{entry.grounded_symbol}' ({entry.category.value}) [Confidence: {entry.confidence:.2f}]"
        )

    # =========================================================================
    # ACT II: FORMAL SOCRATIC TEXTBOOK CHAPTER
    # =========================================================================
    print_banner("ACT II: FORMAL SOCRATIC TEXTBOOK INGESTION & SIMULATION")
    print("Ingesting Chapter 3 Markdown: 'Mechanics, Tools, and Relational Systems'...")

    chapter = TextbookCurriculumCurator.create_sample_physics_chapter()
    print(f"  • Title:            '{chapter.title}' (Grade Level: {chapter.grade_level})")
    print(f"  • Parsed Sections:  {len(chapter.sections)} sections detected:")
    for sec in chapter.sections:
        print(f"      - Section {sec.section_id}: '{sec.title}' [{sec.section_type.value.upper()}]")

    print("\nTeacher presents textbook curriculum to the student...")
    teach_report = teacher.teach_from_textbook(student, chapter)
    sim_puzzle = teach_report.get("simulation_puzzle", {})
    analogy_transfer = teach_report.get("analogy_transfer", {})

    print(
        f"  • Physical Puzzle Compiled: Goal instruction '{chapter.get_section(TextbookSectionType.WORKED_PROBLEM).structured_payload.get('instruction')}'"
    )
    print(f"      - Simulated Plan Steps:   {sim_puzzle.get('plan_steps')}")
    print(f"      - BabyWorld Execution:    Success={sim_puzzle.get('is_success')}")
    print(f"      - Metacognitive Conf:     {sim_puzzle.get('confidence', 0):.2f}")

    print("  • Relational Analogy Transfer: Tabletop Container -> Industrial Ore Hopper")
    print(f"      - Structure Mapping Status: {analogy_transfer.get('mapping_status')}")
    print(f"      - Relational Alignment:     {analogy_transfer.get('score', 0):.2f}")

    # =========================================================================
    # ACT III: UN-MOCKED TEXTBOOK SOCRATIC EXAMINATION
    # =========================================================================
    print_banner("ACT III: UN-MOCKED SOCRATIC CHAPTER EXAMINATION")
    print("Administering comprehensive chapter exam directly compiled from textbook material...")

    assessment = teacher.conduct_textbook_exam(student, chapter)

    print("\n" + "-" * 85)
    print(
        f"{'#':<3} {'Question / Concept':<40} {'Student Response':<22} {'Target':<10} {'Conf':<6} {'Result'}"
    )
    print("-" * 85)
    for idx, q in enumerate(assessment.question_results, 1):
        trunc_q = (q.question_text[:37] + "...") if len(q.question_text) > 40 else q.question_text
        trunc_resp = (
            (q.student_response[:20] + "..") if len(q.student_response) > 22 else q.student_response
        )
        trunc_gt = (q.ground_truth[:8] + "..") if len(q.ground_truth) > 10 else q.ground_truth
        res_str = "PASS ✓" if q.is_correct else "FAIL ✗"
        print(
            f"{idx:<3} {trunc_q:<40} {trunc_resp:<22} {trunc_gt:<10} {q.confidence:<6.2f} {res_str}"
        )
    print("-" * 85)

    print("\nFinal Academic Assessment:")
    print(
        f"  • Accuracy:           {assessment.accuracy * 100:.1f}% ({assessment.correct_count}/{assessment.total_questions})"
    )
    print(f"  • Mean Brier Score:   {assessment.mean_brier_score:.4f} (Epistemically Calibrated)")
    print(f"  • Letter Grade:       {assessment.letter_grade}")
    print(f'  • Teacher Feedback:   "{assessment.teacher_feedback}"')
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
