#!/usr/bin/env python3
"""Train and evaluate HBLLM infant agent using a real educational video and textbook.

Real Media Sources:
  1. Real Video: Scientific study on physical rolling dynamics
     - Source: Wikimedia Commons / Research demonstration (Bombus terrestris rolling ball)
     - Format: 1280x720 29.97 FPS MP4 (22 keyframes extracted via OpenCV)
  2. Real Textbook: OpenStax College Physics 2e (CC-BY 4.0, Paul Peter Urone & Roger Hinrichs)
     - Source: Chapter 9, Section 9.5: "Simple Machines: Levers, Pulleys, and Mechanical Advantage"
     - URL: openstax.org/books/college-physics-2e/pages/9-5-simple-machines
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plugins.developmental_adapter.school import CognitiveSchool
from plugins.developmental_adapter.textbook_curriculum import (
    TextbookParser,
    TextbookSectionType,
)
from plugins.developmental_adapter.video_curriculum import (
    VideoCurriculumCurator,
    VideoDemonstrationManifest,
)


def print_banner(text: str) -> None:
    line = "=" * 85
    print(f"\n{line}\n{text.center(85)}\n{line}")


def main() -> None:
    print_banner("HBLLM DEVELOPMENTAL TRAINING: REAL VIDEO & REAL OPENSTAX TEXTBOOK")

    # 1. Initialize Tabula Rasa Student
    school = CognitiveSchool(
        student_name="Baby HBLLM (Student #001)",
        teacher_name="Prof. Maria Montessori",
        seed=42,
    )
    student = school.student
    teacher = school.teacher

    print("Student Enrollment:")
    print(f"  • Student Name:       {school.student_name}")
    print(f"  • Starting Lexicon:   {len(student.grounding_engine.lexicon)} symbols (Tabula Rasa)")
    print(f"  • Starting Causal:    {len(student.causal_engine.confirmed_causal_rules)} rules")

    # =========================================================================
    # PHASE 1: REAL VIDEO TRAINING (Observational Learning)
    # =========================================================================
    print_banner("PHASE 1: TRAINING ON REAL EDUCATIONAL VIDEO")
    manifest_path = Path(
        "plugins/developmental_adapter/curriculum_data/rolling_ball_demo_manifest.json"
    )
    if not manifest_path.exists():
        manifest_path = Path("data/curriculum/real_video/rolling_ball_demo_manifest.json")
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    manifest = VideoDemonstrationManifest.from_dict(manifest_data)
    print(f"  • Video Title:        '{manifest.title}'")
    print(
        f"  • Video File:         '{manifest.video_path_or_url}' ({manifest.duration_sec}s, {manifest.fps} FPS)"
    )
    print(f"  • Processed Frames:   {len(manifest.frame_segments)} OpenCV keyframes")
    print(f"  • Spoken Utterances:  {len(manifest.audio_segments)} time-stamped narration tracks:")
    for a in manifest.audio_segments:
        print(f'      [{a.start_sec:4.1f}s - {a.end_sec:4.1f}s] {a.speaker}: "{a.text}"')

    video_curator = VideoCurriculumCurator(temporal_tolerance_sec=0.2)
    aligned_pairs = video_curator.align_manifest(manifest)
    print(
        f"\nTemporal Alignment: Synchronized {len(aligned_pairs)} cross-modal demonstration events:"
    )
    for p in aligned_pairs:
        print(
            f"  ✓ t={p.timestamp_sec:4.1f}s: Spoken '{p.utterance}' -> Visual Context: {p.visual_context}"
        )

    print("\nDelivering real video lecture to infant agent (3 observational passes)...")
    for _ in range(3):
        teacher.teach_from_video(student, manifest)

    print("\n[Student Lexicon State after Real Video Training]:")
    for word, entry in sorted(student.grounding_engine.lexicon.items()):
        print(
            f"  ✓ Word '{word}': Symbol='{entry.grounded_symbol}' ({entry.category.value}) [Confidence: {entry.confidence:.2f}]"
        )

    # =========================================================================
    # PHASE 2: REAL TEXTBOOK TRAINING (OpenStax College Physics Section 9.5)
    # =========================================================================
    print_banner("PHASE 2: TRAINING ON REAL OPENSTAX COLLEGE PHYSICS TEXTBOOK")
    textbook_path = Path(
        "plugins/developmental_adapter/curriculum_data/openstax_simple_machines_ch9.md"
    )
    if not textbook_path.exists():
        textbook_path = Path("data/curriculum/real_textbook/openstax_simple_machines_ch9.md")
    with open(textbook_path) as f:
        textbook_content = f.read()

    chapter = TextbookParser.parse_markdown(textbook_content, chapter_id="openstax_ch9_5")
    print(f"  • Textbook Chapter:   '{chapter.title}'")
    print("  • Source Citation:    OpenStax College Physics 2e (Urone & Hinrichs, CC-BY 4.0)")
    print(f"  • Sections Parsed:    {len(chapter.sections)} structural sections:")
    for sec in chapter.sections:
        print(f"      - [{sec.section_type.value.upper()}] {sec.title}")

    print("\nTeacher presents OpenStax physics chapter to student...")
    teach_report = teacher.teach_from_textbook(student, chapter)
    sim_puzzle = teach_report.get("simulation_puzzle", {})
    analogy_transfer = teach_report.get("analogy_transfer", {})

    prob_sec = chapter.get_section(TextbookSectionType.WORKED_PROBLEM)
    inst = prob_sec.structured_payload.get("instruction") if prob_sec else "N/A"
    print(f"\nOpenStax Mechanics Simulation Puzzle: '{inst}'")
    print("  • Initial State:      Ball out of direct reach; rigid stick tool available")
    print(f"  • Student Plan Steps: {sim_puzzle.get('plan_steps')}")
    print(f"  • Simulated Outcome:  Success={sim_puzzle.get('is_success')}")
    print(f"  • Metacognitive Conf: {sim_puzzle.get('confidence', 0):.2f}")

    print("\nOpenStax Cross-Domain Analogy: Tabletop Lever -> Industrial Ore Hopper")
    print(f"  • Mapping Status:     {analogy_transfer.get('mapping_status')}")
    print(f"  • Relational Score:   {analogy_transfer.get('score', 0):.2f}")

    # =========================================================================
    # PHASE 3: REAL SOCRATIC EXAMINATION
    # =========================================================================
    print_banner("PHASE 3: COMPREHENSIVE SOCRATIC CHAPTER EXAMINATION")
    print("Administering un-mocked exam compiled directly from OpenStax Section 9.5...")

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

    print("\nFinal Academic Assessment on Real OpenStax Material:")
    print(
        f"  • Overall Accuracy:   {assessment.accuracy * 100:.1f}% ({assessment.correct_count}/{assessment.total_questions})"
    )
    print(f"  • Mean Brier Score:   {assessment.mean_brier_score:.4f} (Calibrated Epistemics)")
    print(f"  • Letter Grade:       {assessment.letter_grade}")
    print(f'  • Teacher Feedback:   "{assessment.teacher_feedback}"')
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
