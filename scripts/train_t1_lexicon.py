"""Track T1 CLI Runner: Open-Vocabulary Lexicon & Ontology Bootstrap for HCIR.

Examples:
    # 1. Full T1 bootstrap (core ontology + Open English WordNet 100K+ + Gutenberg corpus)
    .venv/bin/python scripts/train_t1_lexicon.py --source full --memory-dir checkpoints/t1_ontology

    # 2. Core + Gutenberg corpus only (offline, no WordNet needed)
    .venv/bin/python scripts/train_t1_lexicon.py --source all --corpus-dir data/gutenberg_cache --memory-dir checkpoints/t1_ontology

    # 3. Kaikki/Wiktionary JSONL dump (en/si/ta)
    .venv/bin/python scripts/train_t1_lexicon.py --source kaikki --kaikki-path data/kaikki-si.jsonl --language si --limit 50000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hbllm.hcir.language.english.lexicon import EnglishLexicon
from hbllm.hcir.language.ontology_bootstrap import OntologyBootstrapper
from hbllm.hcir.language.ontology_store import DEFAULT_ONTOLOGY_DB, LexicalOntologyStore
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Track T1: HCIR Lexicon & Ontology Bootstrap")
    parser.add_argument(
        "--source",
        choices=["core", "corpus", "kaikki", "wordnet", "all", "full"],
        default="full",
        help="Ingestion source ('full' runs core + wordnet + corpus; 'all' runs core + corpus).",
    )
    parser.add_argument("--world-id", default="pure_hcir_t1", help="Target HCIR world_id.")
    parser.add_argument(
        "--memory-dir",
        default="checkpoints/t1_ontology",
        help="Persistent SQLite directory for TieredWorkspace and LexicalOntologyStore.",
    )
    parser.add_argument(
        "--ontology-db",
        default=None,
        help="Path to LexicalOntologyStore SQLite DB (defaults to <memory-dir>/lexical_ontology.db).",
    )
    parser.add_argument(
        "--corpus-dir",
        default="data/gutenberg_cache",
        help="Local text/markdown corpus directory for --source corpus|all|full.",
    )
    parser.add_argument("--kaikki-path", default=None, help="Path to Kaikki/Wiktionary JSONL file.")
    parser.add_argument("--language", default="en", help="Language code (en/si/ta).")
    parser.add_argument(
        "--limit", type=int, default=120_000, help="Max records for WordNet/Kaikki."
    )
    parser.add_argument("--corpus-limit", type=int, default=15_000, help="Max corpus vocabulary.")
    parser.add_argument("--min-freq", type=int, default=5, help="Min frequency for corpus words.")
    parser.add_argument("--max-files", type=int, default=150, help="Max corpus files to scan.")
    args = parser.parse_args()

    mem_dir = Path(args.memory_dir) if args.memory_dir else None
    db_path = (
        Path(args.ontology_db)
        if args.ontology_db
        else (mem_dir / "lexical_ontology.db" if mem_dir else DEFAULT_ONTOLOGY_DB)
    )

    cfg = CognitiveSystemConfig(
        world_id=args.world_id,
        memory_dir=mem_dir,
        enable_grounding=False,
        max_events_per_store=None,
    )
    system = build_cognitive_system(cfg)
    target_ws = system.tiered_workspace.persistent
    grant = getattr(system.tiered_workspace, "_system_grant", None)
    store = LexicalOntologyStore(db_path)
    lexicon = EnglishLexicon(ontology_store=store)
    before_words = len(lexicon._entries)
    bootstrapper = OntologyBootstrapper()

    results = []
    if args.source in ("core", "all", "full"):
        results.append(
            bootstrapper.ingest_core_ontology(
                target_ws, lexicon=lexicon, grant=grant, ontology_store=store
            )
        )

    if args.source in ("wordnet", "full"):
        results.append(
            bootstrapper.ingest_wordnet(
                target_ws,
                limit=args.limit,
                grant=grant,
                ontology_store=store,
                workspace_limit=0,
            )
        )

    if args.source in ("corpus", "all", "full") and Path(args.corpus_dir).exists():
        results.append(
            bootstrapper.ingest_corpus_directory(
                args.corpus_dir,
                target_ws,
                limit=args.corpus_limit,
                min_freq=args.min_freq,
                max_files=args.max_files,
                grant=grant,
                ontology_store=store,
                workspace_limit=0,
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
                grant=grant,
                ontology_store=store,
                workspace_limit=0,
            )
        )

    system.tiered_workspace.notify_commit()

    # Probe O(1) lookup latency & transitive taxonomy reasoning
    probe_words = ["algorithm", "server", "quantum", "microscope", "telescope", "verify", "dog"]
    t0 = time.perf_counter()
    probe_hits = {w: lexicon.has_word(w) for w in probe_words}
    probe_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    taxonomy_checks = {
        "bug_is_a_error": store.is_a("bug", "error"),
        "server_is_a_entity": store.is_a("server", "entity"),
        "dog_is_a_animal": store.is_a("dog", "animal"),
        "microscope_is_a_instrument": store.is_a("microscope", "instrument"),
    }

    summary = {
        "world_id": args.world_id,
        "in_memory_core_words_before": before_words,
        "persistent_active_nodes": target_ws.graph.node_count,
        "persistent_active_edges": target_ws.graph.edge_count,
        "ontology_store": store.stats(),
        "probe_hits": probe_hits,
        "probe_total_ms": probe_ms,
        "taxonomy_checks": taxonomy_checks,
        "runs": [
            {
                "source": r.source,
                "records_processed": r.records_processed,
                "concepts_committed": r.concepts_committed,
                "edges_committed": r.edges_committed,
                "elapsed_ms": r.elapsed_ms,
                "extra": r.extra,
            }
            for r in results
        ],
    }
    store.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
