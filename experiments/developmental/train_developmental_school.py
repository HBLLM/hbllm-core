#!/usr/bin/env python3
"""Interactive Training CLI: Multi-Semester Developmental Cognitive School.

Executes a full multi-semester training session for Baby HBLLM under pedagogical scaffolding:
1. Semester 1 (Freshman): Grounded Lexical Fast-Mapping & Initial Sensory Exploration.
2. Semester 2 (Sophomore): Causal Mechanics, Socratic Counter-Examples, and Tool Planning.
3. Semester 3 (Senior): A20 Relational Analogy Transfer & Metacognitive Abstention Defense.
4. Sleep Consolidation (Stage D12): Dual-store consolidation between semesters guaranteeing BWT >= 0.
5. Checkpoint Persistence: Full cognitive state saved to disk at every semester.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from plugins.developmental_adapter.trainer import (
    CognitiveSchoolTrainer,
    SchoolTrainingConfig,
)

# Terminal ANSI Styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_school_banner() -> None:
    print(f"""{CYAN}{BOLD}
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                   ║
║         🎓  HBLLM DEVELOPMENTAL COGNITIVE TRAINING SESSION (MULTI-EPOCH) 🎓        ║
║                                                                                   ║
║         "From Sensorimotor Exploration to General Relational Intelligence"        ║
║             Dual-Store Sleep Consolidation  •  Causal Socratic Induction          ║
║                                                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
{RESET}""")


def semester_header(semester_idx: int, total_semesters: int, name: str) -> None:
    print("\n" + "=" * 83)
    print(f"{BOLD}{BLUE}ACADEMIC SEMESTER {semester_idx}/{total_semesters}: {name.upper()}{RESET}")
    print("=" * 83)


def print_semester_report(r, delay: float = 0.2) -> None:
    time.sleep(delay)
    print(f"\n{BOLD}{YELLOW}📊 SEMESTER {r.semester_index} SUMMARY REPORT{RESET}")
    print("-" * 83)
    print(f"  • {BOLD}Assessment:{RESET}       {r.assessment.subject_title}")
    print(
        f"  • {BOLD}Exam Accuracy:{RESET}    {r.assessment.accuracy * 100:.1f}% ({r.assessment.correct_count}/{r.assessment.total_questions} questions passed)"
    )
    print(f"  • {BOLD}Letter Grade:{RESET}     {BOLD}{GREEN}{r.assessment.letter_grade}{RESET}")
    print(
        f"  • {BOLD}Brier Score:{RESET}      {r.assessment.mean_brier_score:.4f} (Epistemic Calibration)"
    )
    print(f"  • {BOLD}Grounded Vocab:{RESET}   {r.vocabulary_count} words in cognitive lexicon")
    print(
        f"  • {BOLD}Causal Rules:{RESET}     {r.causal_rules_count} invariant physical laws discovered"
    )
    print(f"  • {BOLD}Affordances:{RESET}      {r.affordances_count} geometric action capabilities")
    if r.sleep_cycle_info:
        print(
            f"  • {BOLD}Sleep Phase:{RESET}      Cycle {r.sleep_cycle_info.get('cycle')} completed "
            f"({r.sleep_cycle_info.get('retained_rules')} rules consolidated, BWT={r.backward_transfer:+.4f})"
        )
    print(f"  • {BOLD}Checkpoint:{RESET}       {GREEN}{r.checkpoint_path}{RESET}")
    print("-" * 83)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real multi-semester training session for HBLLM Cognitive School."
    )
    parser.add_argument(
        "--semesters", type=int, default=3, help="Number of academic semesters (default: 3)"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Episodes per semester (default: 3)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/cognitive_school",
        help="Path to save checkpoints",
    )
    parser.add_argument("--student-name", type=str, default="Baby HBLLM (Student #001)")
    parser.add_argument("--teacher-name", type=str, default="Dr. Maria Vygotsky")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-sleep", action="store_true", help="Disable sleep consolidation cycles"
    )
    args = parser.parse_args()

    print_school_banner()
    print(f"{BOLD}Student:{RESET}          {args.student_name}")
    print(f"{BOLD}Lead Teacher:{RESET}     {args.teacher_name}")
    print(
        f"{BOLD}Curriculum Scope:{RESET} {args.semesters} Semesters ({args.episodes} training episodes / semester)"
    )
    print(f"{BOLD}Checkpoint Dir:{RESET}   {args.checkpoint_dir}")
    print(
        f"{BOLD}Sleep Consolidation:{RESET} {'Disabled' if args.no_sleep else 'Enabled (Stage D12 BWT >= 0)'}\n"
    )

    config = SchoolTrainingConfig(
        student_name=args.student_name,
        teacher_name=args.teacher_name,
        num_semesters=args.semesters,
        episodes_per_semester=args.episodes,
        checkpoint_dir=Path(args.checkpoint_dir),
        enable_sleep_consolidation=not args.no_sleep,
        seed=args.seed,
    )

    trainer = CognitiveSchoolTrainer(config)

    print(f"{CYAN}▶ Initializing training session... Starting blank brain substrate.{RESET}")
    summary = trainer.train()

    for r in summary.semester_reports:
        print_semester_report(r)

    # Commencement & Final Diploma
    print("\n" + "=" * 83)
    print(f"{BOLD}{CYAN}COMMENCEMENT CEREMONY & GRADUATION DIPLOMA{RESET}")
    print("=" * 83)
    print(summary.diploma_text)

    # Checkpoint Artifacts Overview
    final_dir = Path(summary.final_checkpoint_dir)
    print(f"{BOLD}Saved Cognitive Artifacts ({final_dir}):{RESET}")
    if final_dir.exists():
        for p in sorted(final_dir.glob("*.json")):
            size_kb = p.stat().st_size / 1024
            print(f"  • {GREEN}{p.name:<25}{RESET} ({size_kb:.2f} KB)")

    print(f"\n{BOLD}{GREEN}✅ Training session completed successfully!{RESET}\n")


if __name__ == "__main__":
    main()
