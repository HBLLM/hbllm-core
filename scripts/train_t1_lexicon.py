"""Track T1 CLI Runner: Open-Vocabulary Lexicon & Ontology Bootstrap for HCIR.

Examples:
    # 1. Bootstrap built-in core ontology + local Gutenberg corpus into persistent TieredWorkspace
    .venv/bin/python scripts/train_t1_lexicon.py --source all --corpus-dir data/gutenberg_cache --limit 5000

    # 2. Ingest a Kaikki/Wiktionary JSONL dump (en/si/ta)
    .venv/bin/python scripts/train_t1_lexicon.py --source kaikki --kaikki-path data/kaikki-en.jsonl --language en --limit 50000

    # 3. Ingest Open English WordNet (after `pip install wn && python -m wn download oewn:2024`)
    .venv/bin/python scripts/train_t1_lexicon.py --source wordnet --limit 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hbllm.hcir.language.english.lexicon import EnglishLexicon
from hbllm.hcir.language.ontology_bootstrap import OntologyBootstrapper
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Track T1: HCIR Lexicon & Ontology Bootstrap")
    parser.add_argument(
        "--source",
        choices=["core", "corpus", "kaikki", "wordnet", "all"],
        default="core",
        help="Ingestion source ('all' runs core + corpus if --corpus-dir exists).",
    )
    parser.add_argument("--world-id", default="pure_hcir_t1", help="Target HCIR world_id.")
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Override persistent SQLite directory (defaults to ~/.hbllm/worlds/<world_id>).",
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/gutenberg_cache",
        help="Local text/markdown corpus directory for --source corpus|all.",
    )
    parser.add_argument("--kaikki-path", default=None, help="Path to Kaikki/Wiktionary JSONL file.")
    parser.add_argument("--language", default="en", help="Language code (en/si/ta).")
    parser.add_argument("--limit", type=int, default=5000, help="Max vocabulary records to ingest.")
    parser.add_argument("--min-freq", type=int, default=5, help="Min frequency for corpus words.")
    parser.add_argument("--max-files", type=int, default=50, help="Max corpus files to scan.")
    args = parser.parse_args()

    cfg = CognitiveSystemConfig(
        world_id=args.world_id,
        memory_dir=Path(args.memory_dir) if args.memory_dir else None,
        enable_grounding=False,
        max_events_per_store=None,
    )
    system = build_cognitive_system(cfg)
    target_ws = system.tiered_workspace.persistent
    grant = getattr(system.tiered_workspace, "_system_grant", None)
    lexicon = EnglishLexicon()
    before_words = len(lexicon._entries)
    bootstrapper = OntologyBootstrapper()

    results = []
    if args.source in ("core", "all"):
        results.append(bootstrapper.ingest_core_ontology(target_ws, lexicon=lexicon, grant=grant))

    if args.source in ("corpus", "all") and Path(args.corpus_dir).exists():
        results.append(
            bootstrapper.ingest_corpus_directory(
                args.corpus_dir,
                target_ws,
                limit=args.limit,
                min_freq=args.min_freq,
                max_files=args.max_files,
                lexicon=lexicon,
                grant=grant,
            )
        )

    if args.source == "kaikki":
        if not args.kaikki_path:
            raise SystemExit("--kaikki-path is required for --source kaikki")
        results.append(
            bootstrapper.ingest_kaikki_jsonl(
                args.kaikki_path,
                target_ws,
                language=args.language,
                limit=args.limit,
                lexicon=lexicon,
                grant=grant,
            )
        )

    if args.source == "wordnet":
        results.append(
            bootstrapper.ingest_wordnet(target_ws, limit=args.limit, lexicon=lexicon, grant=grant)
        )

    system.tiered_workspace.notify_commit()

    summary = {
        "world_id": args.world_id,
        "lexicon_words_before": before_words,
        "lexicon_words_after": len(lexicon._entries),
        "persistent_nodes": target_ws.graph.node_count,
        "persistent_edges": target_ws.graph.edge_count,
        "runs": [
            {
                "source": r.source,
                "records_processed": r.records_processed,
                "concepts_committed": r.concepts_committed,
                "edges_committed": r.edges_committed,
                "lexicon_entries_added": r.lexicon_entries_added,
                "elapsed_ms": r.elapsed_ms,
                "extra": r.extra,
            }
            for r in results
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
