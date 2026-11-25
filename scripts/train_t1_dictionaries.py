"""Track T1.5 CLI Runner: Multilingual Dictionary, Commonsense & Frame Training.

Ingests downloaded dictionaries in ``data/dictionaries/`` directly into
``LexicalOntologyStore`` with zero working-graph RAM bloat and evaluates
held-out Dakshina transliteration recall, cross-lingual alignments, verb
thematic frames, ConceptNet commonsense/antonym links, and O(1) lookup latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hbllm.hcir.language.dictionary_ingestors import (
    IngestionReport,
    evaluate_dakshina_transliteration,
    ingest_bilingual_dictionaries,
    ingest_conceptnet_gz,
    ingest_dakshina_transliterations,
    ingest_frequency_list,
    ingest_gutenberg_enrichment,
    ingest_kaikki_multilingual,
    ingest_propbank_zip,
    ingest_verbnet_zip,
)
from hbllm.hcir.stores.ontology_store import DEFAULT_ONTOLOGY_DB, LexicalOntologyStore

_ALL_STAGES = (
    "freq",
    "kaikki",
    "verbnet",
    "propbank",
    "conceptnet",
    "bilingual",
    "dakshina",
    "gutenberg",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track T1.5: Multilingual Dictionary Training")
    parser.add_argument("--dict-dir", default="data/dictionaries")
    parser.add_argument("--ontology-db", default=str(DEFAULT_ONTOLOGY_DB))
    parser.add_argument("--stages", default="all", help=f"all or CSV of {_ALL_STAGES}")
    parser.add_argument("--kaikki-en-limit", type=int, default=150_000)
    parser.add_argument("--kaikki-ta-limit", type=int, default=100_000)
    parser.add_argument("--conceptnet-min-weight", type=float, default=2.0)
    parser.add_argument("--conceptnet-max-edges", type=int, default=250_000)
    parser.add_argument("--report-path", default="reports/training/t1_dictionaries_report.json")
    args = parser.parse_args()

    selected = (
        set(_ALL_STAGES)
        if args.stages.strip().lower() == "all"
        else {s.strip().lower() for s in args.stages.split(",") if s.strip()}
    )

    dict_root = Path(args.dict_dir)
    store = LexicalOntologyStore(args.ontology_db)
    before_stats = store.stats()
    reports: list[IngestionReport] = []
    freq_maps: dict[str, dict[str, int]] = {}

    if "freq" in selected:
        for lang, fname in (("en", "en_full.txt"), ("si", "si_full.txt"), ("ta", "ta_full.txt")):
            fpath = dict_root / "frequency" / fname
            if fpath.exists():
                fmap, rep = ingest_frequency_list(fpath, store, language=lang)
                freq_maps[lang] = fmap
                reports.append(rep)

    if "kaikki" in selected:
        for lang, fname, lim in (
            ("si", "kaikki-si.jsonl", 100_000),
            ("ta", "kaikki-ta.jsonl", args.kaikki_ta_limit),
            ("en", "kaikki-en.jsonl", args.kaikki_en_limit),
        ):
            kpath = dict_root / "kaikki" / fname
            if kpath.exists():
                reports.append(
                    ingest_kaikki_multilingual(
                        kpath,
                        store,
                        language=lang,
                        limit=lim,
                        freq_map=freq_maps.get(lang),
                    )
                )

    if "verbnet" in selected and (dict_root / "verbnet" / "verbnet-master.zip").exists():
        reports.append(ingest_verbnet_zip(dict_root / "verbnet" / "verbnet-master.zip", store))

    if "propbank" in selected and (dict_root / "english" / "propbank-frames-main.zip").exists():
        reports.append(
            ingest_propbank_zip(dict_root / "english" / "propbank-frames-main.zip", store)
        )

    if (
        "conceptnet" in selected
        and (dict_root / "conceptnet" / "conceptnet-assertions-5.7.0.csv.gz").exists()
    ):
        reports.append(
            ingest_conceptnet_gz(
                dict_root / "conceptnet" / "conceptnet-assertions-5.7.0.csv.gz",
                store,
                min_weight=args.conceptnet_min_weight,
                max_edges=args.conceptnet_max_edges,
            )
        )

    if "bilingual" in selected:
        reports.extend(ingest_bilingual_dictionaries(dict_root, store))

    dakshina_eval: dict[str, dict[str, object]] = {}
    if "dakshina" in selected and (dict_root / "dakshina").exists():
        reports.append(ingest_dakshina_transliterations(dict_root / "dakshina", store))
        for lang in ("si", "ta"):
            dakshina_eval[lang] = evaluate_dakshina_transliteration(
                dict_root / "dakshina", store, language=lang, split="test"
            )

    if "gutenberg" in selected and (dict_root / "gutenberg").exists():
        reports.extend(ingest_gutenberg_enrichment(dict_root / "gutenberg", store))

    t0 = time.perf_counter()
    probes = {
        "si_water_en": store.align_to_english("වතුර", language="si"),
        "ta_water_en": store.align_to_english("தண்ணீர்", language="ta"),
        "verb_give_roles": [
            r.thematic_roles for r in store.lookup("give", language="en") if r.thematic_roles
        ],
        "fire_relations": store.get_relations("fire", language="en", limit=6),
        "antonym_hot_cold": store.are_antonyms("hot", "cold", language="en"),
        "lemmatize_went": store.lemmatize_surface("went", language="en"),
    }
    probe_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    after_stats = store.stats()
    store.optimize()
    store.close()

    summary = {
        "before_stats": before_stats,
        "after_stats": after_stats,
        "dakshina_heldout_eval": dakshina_eval,
        "probes": probes,
        "probe_total_ms": probe_ms,
        "runs": [asdict(r) for r in reports],
    }
    out = Path(args.report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
