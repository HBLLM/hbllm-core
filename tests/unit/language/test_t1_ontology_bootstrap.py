"""Tests for Track T1 open-vocabulary lexicon & ontology bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

from hbllm.hcir.language.english.lexicon import EnglishLexicon
from hbllm.hcir.language.english.parser import EnglishParser
from hbllm.hcir.language.ontology_bootstrap import OntologyBootstrapper
from hbllm.hcir.nodes import PhysicalEntityNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_t1_core_ontology_commits_transactionally_and_hydrates_parser() -> None:
    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_t1_core",
            memory_dir=None,
            enable_grounding=False,
        )
    )
    lexicon = EnglishLexicon()
    assert not lexicon.has_word("server")
    assert not lexicon.has_word("cluster")

    bootstrapper = OntologyBootstrapper()
    stats = bootstrapper.ingest_core_ontology(system.workspace, lexicon=lexicon)
    assert stats.concepts_committed >= 70
    assert stats.edges_committed >= 30
    assert lexicon.has_word("server")
    assert lexicon.has_word("cluster")
    assert lexicon.has_word("verify")

    # Parser can now parse open-domain sentences using T1-bootstrapped words
    parser = EnglishParser(lexicon=lexicon)
    frame = parser.parse("The server is in the cluster.")
    assert frame.predicate == "located_in"


def test_t1_kaikki_jsonl_and_corpus_ingestion(tmp_path: Path) -> None:
    kaikki = tmp_path / "sample.jsonl"
    kaikki.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "word": "compiler",
                        "pos": "noun",
                        "senses": [
                            {
                                "glosses": ["A program that translates source code."],
                                "hypernyms": [{"word": "program"}],
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "word": "refactor",
                        "pos": "verb",
                        "senses": [{"glosses": ["Restructure code without changing behavior."]}],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc.txt").write_text(
        "telemetry telemetry telemetry telemetry latency latency latency latency\n",
        encoding="utf-8",
    )

    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_t1_sources",
            memory_dir=None,
            enable_grounding=False,
        )
    )
    lexicon = EnglishLexicon()
    bootstrapper = OntologyBootstrapper()

    k_stats = bootstrapper.ingest_kaikki_jsonl(kaikki, system.workspace, lexicon=lexicon)
    assert k_stats.concepts_committed == 2
    assert lexicon.has_word("compiler")
    assert lexicon.has_word("refactor")

    c_stats = bootstrapper.ingest_corpus_directory(
        corpus_dir, system.workspace, min_freq=3, lexicon=lexicon
    )
    assert c_stats.concepts_committed == 2
    assert lexicon.has_word("telemetry")
    assert lexicon.has_word("latency")


def test_t1_runtime_auto_hydrates_from_workspace() -> None:
    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_t1_auto_hydrate",
            memory_dir=None,
            enable_grounding=False,
        )
    )
    OntologyBootstrapper().ingest_core_ontology(system.workspace)

    server = PhysicalEntityNode(id="srv_1", entity_name="server", entity_type="server")
    cluster = PhysicalEntityNode(id="cls_1", entity_name="cluster", entity_type="cluster")
    system.workspace.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=server.id, node_data=server.model_dump()
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=cluster.id, node_data=cluster.model_dump()
            ),
        ],
        author="test_setup",
    )

    turn = system.process_dialogue("The server is in the cluster.")
    assert turn.is_success is True
    assert turn.hcir_node_id is not None


def test_t1_ontology_store_o1_lookup_and_transitive_taxonomy(tmp_path: Path) -> None:
    from hbllm.hcir.language.ontology_store import LexicalOntologyStore

    store = LexicalOntologyStore(tmp_path / "lexical_ontology.db")
    bootstrapper = OntologyBootstrapper()
    stats = bootstrapper.ingest_core_ontology(None, ontology_store=store)
    assert stats.extra["store_entries_upserted"] >= 70
    assert store.is_a("bug", "error") is True
    assert store.is_a("server", "entity") is True
    assert store.is_a("server", "agent") is False

    lexicon = EnglishLexicon(ontology_store=store)
    assert lexicon.has_word("server") is True
    assert lexicon.has_word("cluster") is True
    parser = EnglishParser(lexicon=lexicon)
    frame = parser.parse("The server is in the cluster.")
    assert frame.predicate == "located_in"
    store.close()
