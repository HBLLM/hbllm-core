"""Track T3 CLI Runner: Parallel Shadow Distillation (LLM Scaffold -> Pure-HCIR).

Runs multi-epoch multi-domain dialogue + governed tool-use tasks through
``CognitiveSystem.process_dialogue()`` and ``KernelServices.capability_resolver``:
  1. Pure-HCIR attempts every turn first (using Track T1 110K-entry ontology).
  2. On HCIR parse/grounding gaps, ``LLMTeacherScaffold`` (strictly outside
     ``hbllm/hcir/``) distills verified entities, lexical entries, and beliefs
     into HCIR via ``TransactionManager``.
  3. On subsequent epochs, Pure-HCIR handles those turns natively without the LLM,
     tracking ``hcir_parse_autonomy_rate``, ``hcir_plan_autonomy_rate``, and
     ``hcir_realize_autonomy_rate`` toward the ``>= 98%`` retirement gate.

Examples:
    # Fast curriculum shadow run (CPU-only, offline)
    .venv/bin/python scripts/run_parallel_shadow_training.py --mode curriculum --epochs 5

    # Live NVIDIA NIM teacher run (uses NVIDIA_API_KEY from .env)
    .venv/bin/python scripts/run_parallel_shadow_training.py --mode nvidia --epochs 5
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

# Multi-domain dialogue curriculum (software/cloud, science/lab, everyday + queries)
CURRICULUM_EPISODES: list[dict[str, Any]] = [
    # Software / cloud infrastructure
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
        "utterance": "The cache is near the database.",
        "lesson": {
            "response_text": "Noted: the cache is near the database.",
            "claim": "cache near database",
            "entities": [
                {"name": "cache", "type": "cache"},
                {"name": "database", "type": "database"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The container is in the cluster.",
        "lesson": {
            "response_text": "Noted: the container is in the cluster.",
            "claim": "container located_in cluster",
            "entities": [
                {"name": "container", "type": "container"},
                {"name": "cluster", "type": "cluster"},
            ],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The router is near the firewall.",
        "lesson": {
            "response_text": "Noted: the router is near the firewall.",
            "claim": "router near firewall",
            "entities": [
                {"name": "router", "type": "router"},
                {"name": "firewall", "type": "firewall"},
            ],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "The file is in the folder.",
        "lesson": {
            "response_text": "Noted: the file is in the folder.",
            "claim": "file located_in folder",
            "entities": [{"name": "file", "type": "file"}, {"name": "folder", "type": "folder"}],
            "lexicon": [],
            "confidence": 0.93,
            "verified": True,
        },
    },
    # Science / robotics / lab
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
        "utterance": "The battery is inside the robot.",
        "lesson": {
            "response_text": "Noted: the battery is inside the robot.",
            "claim": "battery located_in robot",
            "entities": [
                {"name": "battery", "type": "battery"},
                {"name": "robot", "type": "robot"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    {
        "utterance": "The lens is on the microscope.",
        "lesson": {
            "response_text": "Noted: the lens is on the microscope.",
            "claim": "lens located_on microscope",
            "entities": [
                {"name": "lens", "type": "lens"},
                {"name": "microscope", "type": "microscope"},
            ],
            "lexicon": [],
            "confidence": 0.91,
            "verified": True,
        },
    },
    # Everyday / spatial
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
        "utterance": "The document is on the desk.",
        "lesson": {
            "response_text": "Noted: the document is on the desk.",
            "claim": "document located_on desk",
            "entities": [
                {"name": "document", "type": "document"},
                {"name": "desk", "type": "desk"},
            ],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    {
        "utterance": "The lamp is above the desk.",
        "lesson": {
            "response_text": "Noted: the lamp is above the desk.",
            "claim": "lamp above desk",
            "entities": [{"name": "lamp", "type": "lamp"}, {"name": "desk", "type": "desk"}],
            "lexicon": [],
            "confidence": 0.92,
            "verified": True,
        },
    },
    # Grounded Queries (tests recall & realization)
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
        "utterance": "Where is the database?",
        "lesson": {
            "response_text": "The database is on the server.",
            "claim": "database located_on server",
            "entities": [{"name": "database", "type": "database"}],
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
    {
        "utterance": "Where is the file?",
        "lesson": {
            "response_text": "The file is in the folder.",
            "claim": "file located_in folder",
            "entities": [{"name": "file", "type": "file"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the key?",
        "lesson": {
            "response_text": "The key is under the book.",
            "claim": "key located_under book",
            "entities": [{"name": "key", "type": "key"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
    {
        "utterance": "Where is the document?",
        "lesson": {
            "response_text": "The document is on the desk.",
            "claim": "document located_on desk",
            "entities": [{"name": "document", "type": "document"}],
            "lexicon": [],
            "confidence": 0.90,
            "verified": True,
        },
    },
]

# Governed Kernel Tool-Use Gym Tasks (executed 100% natively via CapabilityResolver)
TOOL_GYM_TASKS: list[tuple[str, dict[str, Any]]] = [
    ("fs.list", {"path": "hbllm/hcir/language"}),
    ("fs.read", {"path": "hbllm/hcir/language/ontology_store.py", "max_bytes": 512}),
    ("code.python_ast", {"path": "hbllm/hcir/language/ontology_store.py"}),
    ("git.status", {}),
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

        # Governed Kernel Tool-Use Gym (native Pure-HCIR capability dispatch)
        tool_ok = 0
        for cap_name, params in TOOL_GYM_TASKS:
            executor = await system.services.capability_resolver.resolve(cap_name)
            ok = False
            if executor is not None:
                out = await executor.execute(params)
                ok = bool(out.get("ok", False))
            tracker.record_plan(ok)
            if ok:
                tool_ok += 1

        if tracker.can_retire_llm:
            scaffold.retired = True

        if system.runtime is not None:
            await system.runtime.run_offline_consolidation()

        epoch_reports.append(
            {
                "epoch": epoch,
                "turns": len(CURRICULUM_EPISODES),
                "hcir_native_successes": epoch_hcir_hits,
                "shadow_fallbacks": epoch_shadow_hits,
                "epoch_hcir_autonomy": round(epoch_hcir_hits / len(CURRICULUM_EPISODES), 4),
                "tool_gym_passed": f"{tool_ok}/{len(TOOL_GYM_TASKS)}",
                "cumulative": tracker.to_dict(),
                "scaffold_retired": scaffold.retired,
            }
        )

    report = {
        "mode": args.mode,
        "teacher": str(scaffold.checkpoint_path),
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
        "--model",
        default=None,
        help="NVIDIA NIM model (default: nvidia/nemotron-3.5-lightning-30b-a3b)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="NVIDIA NIM base URL (default: https://integrate.api.nvidia.com/v1)",
    )
    parser.add_argument("--world-id", default="pure_hcir_t3")
    parser.add_argument("--memory-dir", default="checkpoints/t3_shadow")
    parser.add_argument("--ontology-db", default=str(DEFAULT_ONTOLOGY_DB))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--min-turns", type=int, default=50)
    parser.add_argument("--retirement-threshold", type=float, default=0.80)
    parser.add_argument("--report-path", default="reports/training/t3_shadow_report.json")
    args = parser.parse_args()

    report = asyncio.run(run_training(args))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
