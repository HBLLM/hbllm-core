"""Track T3 CLI Runner: Parallel Shadow Distillation (LLM Scaffold -> Pure-HCIR).

Runs multi-epoch dialogue turns through ``CognitiveSystem.process_dialogue()``:
  1. Pure-HCIR attempts every turn first (using Track T1 110K-entry ontology).
  2. On HCIR parse/grounding gaps, ``LLMTeacherScaffold`` (strictly outside
     ``hbllm/hcir/``) distills verified entities, lexical entries, and beliefs
     into HCIR via ``TransactionManager``.
  3. On subsequent epochs, Pure-HCIR handles those turns natively without the LLM,
     tracking ``hcir_parse_autonomy_rate`` and ``hcir_realize_autonomy_rate``
     toward the ``>= 98%`` retirement gate.

Examples:
    # Fast curriculum shadow run (verifies distillation & autonomy convergence on CPU)
    .venv/bin/python scripts/run_parallel_shadow_training.py --mode curriculum --epochs 5

    # Live local HuggingFace teacher run (uses checkpoints/Llama-3.2-1B offline)
    .venv/bin/python scripts/run_parallel_shadow_training.py --mode live --checkpoint checkpoints/Llama-3.2-1B --epochs 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hbllm.hcir.language.ontology_store import DEFAULT_ONTOLOGY_DB, LexicalOntologyStore
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
from plugins.llm_scaffold import (
    DEFAULT_LOCAL_CHECKPOINT,
    AutonomyTracker,
    LLMTeacherScaffold,
)

# Grounded open-domain + engineering dialogue curriculum for T3 shadow distillation
CURRICULUM_EPISODES: list[dict[str, Any]] = [
    {
        "utterance": "The server is in the cluster.",
        "lesson": {
            "response_text": "Noted: the server is located in the cluster.",
            "claim": "server located_in cluster",
            "entities": [
                {"name": "server", "type": "server"},
                {"name": "cluster", "type": "cluster"},
            ],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The database is on the server.",
        "lesson": {
            "response_text": "Noted: the database is located on the server.",
            "claim": "database located_on server",
            "entities": [
                {"name": "database", "type": "database"},
                {"name": "server", "type": "server"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The sensor is near the robot.",
        "lesson": {
            "response_text": "Noted: the sensor is near the robot.",
            "claim": "sensor near robot",
            "entities": [{"name": "sensor", "type": "sensor"}, {"name": "robot", "type": "robot"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "The microscope is on the table.",
        "lesson": {
            "response_text": "Noted: the microscope is on the table.",
            "claim": "microscope located_on table",
            "entities": [
                {"name": "microscope", "type": "microscope"},
                {"name": "table", "type": "table"},
            ],
            "lexicon": [],
            "confidence": 0.93,
            "verified": True,
        },
    },
    {
        "utterance": "The telescope is in the room.",
        "lesson": {
            "response_text": "Noted: the telescope is in the room.",
            "claim": "telescope located_in room",
            "entities": [
                {"name": "telescope", "type": "telescope"},
                {"name": "room", "type": "room"},
            ],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The key is under the book.",
        "lesson": {
            "response_text": "Noted: the key is under the book.",
            "claim": "key located_under book",
            "entities": [{"name": "key", "type": "key"}, {"name": "book", "type": "book"}],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the server?",
        "lesson": {
            "response_text": "The server is in the cluster.",
            "claim": "server located_in cluster",
            "entities": [{"name": "server", "type": "server"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the microscope?",
        "lesson": {
            "response_text": "The microscope is on the table.",
            "claim": "microscope located_on table",
            "entities": [{"name": "microscope", "type": "microscope"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
]


def _make_curriculum_teacher() -> Any:
    lookup = {ep["utterance"]: ep["lesson"] for ep in CURRICULUM_EPISODES}

    def _teacher(utterance: str) -> dict[str, Any]:
        return lookup.get(
            utterance,
            {
                "response_text": "",
                "claim": "",
                "entities": [],
                "lexicon": [],
                "confidence": 0.0,
                "verified": False,
            },
        )

    return _teacher


async def run_training(args: argparse.Namespace) -> dict[str, Any]:
    mem_dir = Path(args.memory_dir) if args.memory_dir else None
    db_path = Path(args.ontology_db) if args.ontology_db else DEFAULT_ONTOLOGY_DB
    store = LexicalOntologyStore(db_path)

    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id=args.world_id,
            memory_dir=mem_dir,
            enable_grounding=False,
            enable_brain_llm=True,
        )
    )
    tracker = AutonomyTracker(
        min_turns_for_retirement=args.min_turns,
        retirement_threshold=args.retirement_threshold,
    )
    system.services.extensions["autonomy_metrics"] = tracker

    if args.mode == "nvidia":
        scaffold = LLMTeacherScaffold.from_nvidia_endpoint(
            model=args.model,
            base_url=args.base_url,
            tracker=tracker,
            ontology_store=store,
        )
    elif args.mode == "live":
        scaffold = LLMTeacherScaffold.from_local_checkpoint(
            checkpoint_path=args.checkpoint,
            tracker=tracker,
            ontology_store=store,
        )
    else:
        scaffold = LLMTeacherScaffold(
            checkpoint_path=args.checkpoint,
            tracker=tracker,
            teacher_fn=_make_curriculum_teacher(),
            ontology_store=store,
        )
    system.services.extensions["llm_scaffold"] = scaffold

    epoch_reports: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_hcir_hits = 0
        epoch_shadow_hits = 0
        for ep in CURRICULUM_EPISODES:
            res = system.process_dialogue(ep["utterance"])
            if isinstance(res, dict):
                epoch_shadow_hits += 1
            elif getattr(res, "is_success", False):
                epoch_hcir_hits += 1

        if system.runtime is not None:
            await system.runtime.run_offline_consolidation()

        epoch_reports.append(
            {
                "epoch": epoch,
                "turns": len(CURRICULUM_EPISODES),
                "hcir_native_successes": epoch_hcir_hits,
                "shadow_fallbacks": epoch_shadow_hits,
                "epoch_hcir_autonomy": round(epoch_hcir_hits / len(CURRICULUM_EPISODES), 4),
                "cumulative": tracker.to_dict(),
                "scaffold_retired": scaffold.retired,
            }
        )

    report = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "world_id": args.world_id,
        "epochs": epoch_reports,
        "final_metrics": tracker.to_dict(),
        "scaffold_retired": scaffold.retired,
        "ontology_store_stats": store.stats(),
    }
    store.close()

    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Track T3: Parallel Shadow Distillation into HCIR")
    parser.add_argument("--mode", choices=["curriculum", "nvidia", "live"], default="curriculum")
    parser.add_argument("--checkpoint", default=str(DEFAULT_LOCAL_CHECKPOINT))
    parser.add_argument(
        "--model", default=None, help="NVIDIA NIM model (default: meta/llama-3.1-8b-instruct)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="NVIDIA NIM base URL (default: https://integrate.api.nvidia.com/v1)",
    )
    parser.add_argument("--world-id", default="pure_hcir_t3")
    parser.add_argument("--memory-dir", default="checkpoints/t3_shadow")
    parser.add_argument("--ontology-db", default=str(DEFAULT_ONTOLOGY_DB))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-turns", type=int, default=32)
    parser.add_argument("--retirement-threshold", type=float, default=0.80)
    parser.add_argument("--report-path", default="reports/training/t3_shadow_report.json")
    args = parser.parse_args()

    report = asyncio.run(run_training(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
