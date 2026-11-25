"""Hermetic Held-Out Pure-HCIR Language Evaluation Runner (M-6)."""

from __future__ import annotations

import hashlib
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from hbllm.hcir.graph import ConceptNode, HCIREdge, HCIREdgeType
from hbllm.hcir.language.ontology_bootstrap import OntologyBootstrapper
from hbllm.hcir.stores.ontology_store import LexicalOntologyStore, StoredLexicalEntry
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "suites" / "language_heldout.yaml"
)


def _seed_heldout_ontology(db_path: Path) -> None:
    """Seed an isolated lexical ontology store matching the held-out vocabulary."""
    store = LexicalOntologyStore(db_path)
    try:
        OntologyBootstrapper().ingest_core_ontology(None, ontology_store=store)
        entries = [
            StoredLexicalEntry(
                lemma="prism",
                pos="noun",
                language="en",
                definition="A transparent optical glass instrument that refracts white light into a spectrum.",
                hypernyms=("instrument", "device"),
                domain="optics",
                frequency=420,
            ),
            StoredLexicalEntry(
                lemma="compass",
                pos="noun",
                language="en",
                definition="A magnetic navigation instrument used to determine cardinal direction.",
                hypernyms=("instrument", "device"),
                domain="navigation",
                frequency=510,
            ),
            StoredLexicalEntry(
                lemma="turbine",
                pos="noun",
                language="en",
                definition="A rotary engine that extracts mechanical energy from flowing fluid or steam.",
                hypernyms=("engine", "machine"),
                domain="engineering",
                frequency=440,
            ),
            StoredLexicalEntry(
                lemma="glacier",
                pos="noun",
                language="en",
                definition="A persistent slow-moving mass of dense frozen ice on land.",
                hypernyms=("formation", "ice"),
                domain="earth",
                frequency=400,
            ),
            StoredLexicalEntry(
                lemma="torque",
                pos="noun",
                language="en",
                definition="A rotational mechanical force produced by an energized rotor.",
                hypernyms=("force",),
                domain="physics",
                frequency=480,
            ),
            StoredLexicalEntry(
                lemma="rotor",
                pos="noun",
                language="en",
                definition="The rotating component of an electrical machine driven by applied voltage.",
                hypernyms=("component",),
                domain="engineering",
                frequency=410,
            ),
            StoredLexicalEntry(
                lemma="voltage",
                pos="noun",
                language="en",
                definition="An electrical potential difference that drives current across a circuit.",
                hypernyms=("potential", "quantity"),
                domain="physics",
                frequency=560,
            ),
            StoredLexicalEntry(
                lemma="ecosystem",
                pos="noun",
                language="en",
                definition="A biological community of interacting organisms sustained by biodiversity.",
                hypernyms=("system",),
                domain="ecology",
                frequency=470,
            ),
            StoredLexicalEntry(
                lemma="biodiversity",
                pos="noun",
                language="en",
                definition="The variety of living species within a habitat supported by pollination.",
                hypernyms=("variety", "property"),
                domain="ecology",
                frequency=430,
            ),
            StoredLexicalEntry(
                lemma="pollination",
                pos="noun",
                language="en",
                definition="The ecological transfer of pollen enabling plant reproduction.",
                hypernyms=("process",),
                domain="ecology",
                frequency=390,
            ),
        ]
        store.upsert_many(entries, source="language_heldout_eval")
        for unknown in ("xenolith", "magnetar", "quasarite", "chronoton"):
            store._conn.execute("DELETE FROM lexical_entries WHERE lemma = ?", (unknown,))
            store._conn.execute(
                "DELETE FROM hypernym_edges WHERE child = ? OR parent = ?", (unknown, unknown)
            )
        store._conn.commit()
        store._lookup_cache.clear()
        store._hypernyms_cache.clear()
        store._is_a_cache.clear()
    finally:
        store.close()


def _seed_heldout_graph_chains(system: Any) -> None:
    p_graph = system.tiered_workspace.persistent.graph
    for cid, label in (
        ("lo_voltage", "voltage"),
        ("lo_rotor", "rotor"),
        ("lo_torque", "torque"),
        ("lo_ecosystem", "ecosystem"),
        ("lo_biodiversity", "biodiversity"),
        ("lo_pollination", "pollination"),
    ):
        p_graph.add_node(ConceptNode(id=cid, label=label))
    p_graph.add_edge(
        HCIREdge(
            id="lo_causes_1",
            edge_type=HCIREdgeType.CAUSES,
            sources=["lo_voltage"],
            targets=["lo_rotor"],
        )
    )
    p_graph.add_edge(
        HCIREdge(
            id="lo_causes_2",
            edge_type=HCIREdgeType.CAUSES,
            sources=["lo_rotor"],
            targets=["lo_torque"],
        )
    )
    p_graph.add_edge(
        HCIREdge(
            id="lo_req_1",
            edge_type=HCIREdgeType.REQUIRES,
            sources=["lo_ecosystem"],
            targets=["lo_biodiversity"],
        )
    )
    p_graph.add_edge(
        HCIREdge(
            id="lo_req_2",
            edge_type=HCIREdgeType.REQUIRES,
            sources=["lo_biodiversity"],
            targets=["lo_pollination"],
        )
    )


def run_language_heldout_eval(
    manifest_path: Path | str = _DEFAULT_MANIFEST,
    *,
    enable_brain_llm: bool = False,
) -> dict[str, Any]:
    """Run the sealed held-out language evaluation suite hermetically."""
    mpath = Path(manifest_path)
    raw_bytes = mpath.read_bytes()
    manifest_sha = hashlib.sha256(raw_bytes).hexdigest()
    manifest = yaml.safe_load(raw_bytes.decode("utf-8"))
    cases = list(manifest.get("cases") or [])

    cat_tot: dict[str, int] = defaultdict(int)
    cat_ok: dict[str, int] = defaultdict(int)
    failures: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="hbllm_lang_heldout_") as tmp_dir:
        ont_path = Path(tmp_dir) / "heldout_ontology.db"
        _seed_heldout_ontology(ont_path)
        system = build_cognitive_system(
            CognitiveSystemConfig(
                world_id="language_heldout_eval",
                memory_dir=None,
                seed_ontology_path=ont_path,
                enable_grounding=False,
                enable_brain_llm=enable_brain_llm,
                enable_autonomous_curiosity=False,
            )
        )
        try:
            _seed_heldout_graph_chains(system)
            for case in cases:
                cid = str(case.get("id", "unknown"))
                cat = str(case.get("category", "uncategorized"))
                utt = str(case.get("utterance", ""))
                expect_subs = [str(s) for s in (case.get("expect_substrings") or ())]
                expect_known = case.get("expect_known")

                cat_tot[cat] += 1
                res = system.process_dialogue(utt)
                resp = getattr(res, "response_text", "") or ""
                ok = bool(getattr(res, "is_success", False)) and bool(resp.strip())
                if ok and expect_subs:
                    low = resp.lower()
                    ok = any(sub.lower() in low for sub in expect_subs)
                if ok and expect_known is not None:
                    ep = getattr(res, "epistemic_state", None)
                    if ep is None or bool(getattr(ep, "is_known", None)) != bool(expect_known):
                        ok = False

                if ok:
                    cat_ok[cat] += 1
                else:
                    failures.append(
                        {"id": cid, "category": cat, "utterance": utt, "response": resp[:160]}
                    )
        finally:
            ont_store = system.services.extensions.get("ontology_store")
            if ont_store is not None and hasattr(ont_store, "close"):
                ont_store.close()

    total = len(cases)
    passed = sum(cat_ok.values())
    return {
        "split": str(manifest.get("split", "language_heldout")),
        "manifest_sha256": manifest_sha,
        "total": total,
        "passed": passed,
        "accuracy": round(passed / max(1, total), 4),
        "category_scores": {
            cat: round(cat_ok[cat] / max(1, n), 4) for cat, n in sorted(cat_tot.items())
        },
        "failures": failures,
    }
