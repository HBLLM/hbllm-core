"""Track T4 CLI Runner: Open-Ontology Reading Curriculum for HCIR.

Reads local architecture docs and books, extracts taxonomic (``is_a``),
causal (``causes``), and dependency (``requires``) knowledge, commits into
``TieredWorkspace.persistent`` + ``LexicalOntologyStore``, and runs
``ContradictionOperator`` checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hbllm.hcir.language.ontology_store import DEFAULT_ONTOLOGY_DB, LexicalOntologyStore
from hbllm.hcir.language.reading_curriculum import ReadingCurriculumIngestor
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Track T4: Open-Ontology Reading Curriculum")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[
            "plugins/developmental_adapter/curriculum_data",
            "data/curriculum",
            "docs",
            "data/gutenberg_cache",
        ],
        help="Directories of .md/.txt files to read.",
    )
    parser.add_argument("--world-id", default="pure_hcir_t4")
    parser.add_argument("--memory-dir", default="checkpoints/t4_reading")
    parser.add_argument("--ontology-db", default=str(DEFAULT_ONTOLOGY_DB))
    parser.add_argument("--max-files", type=int, default=550)
    parser.add_argument("--max-propositions", type=int, default=3000)
    parser.add_argument("--report-path", default="reports/training/t4_reading_report.json")
    args = parser.parse_args()

    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id=args.world_id,
            memory_dir=Path(args.memory_dir),
            enable_grounding=False,
        )
    )
    target_ws = system.tiered_workspace.persistent
    grant = getattr(system.tiered_workspace, "_system_grant", None)
    store = LexicalOntologyStore(args.ontology_db)

    ingestor = ReadingCurriculumIngestor()
    stats = ingestor.ingest_directories(
        args.dirs,
        target_ws,
        ontology_store=store,
        grant=grant,
        max_files=args.max_files,
        max_propositions=args.max_propositions,
    )
    system.tiered_workspace.notify_commit()

    report = {
        "world_id": args.world_id,
        "directories": args.dirs,
        "files_read": stats.files_read,
        "sentences_scanned": stats.sentences_scanned,
        "taxonomic_extracted": stats.taxonomic_extracted,
        "causal_extracted": stats.causal_extracted,
        "dependency_extracted": stats.dependency_extracted,
        "nodes_committed": stats.nodes_committed,
        "edges_committed": stats.edges_committed,
        "store_hypernyms_added": stats.store_hypernyms_added,
        "elapsed_ms": stats.elapsed_ms,
        "persistent_nodes": target_ws.graph.node_count,
        "persistent_edges": target_ws.graph.edge_count,
        "sample_propositions": stats.sample_propositions,
        "ontology_store_stats": store.stats(),
    }
    store.close()

    out_path = Path(args.report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
