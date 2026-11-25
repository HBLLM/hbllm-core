"""Unit tests for Track T4/T1.5 book and chat learning upgrades."""

from __future__ import annotations

import asyncio
from pathlib import Path

from hbllm.hcir.capabilities.memory_capabilities import MemoryCapabilityExecutor
from hbllm.hcir.language.core.semantic_frame import FrameType, ThematicRole
from hbllm.hcir.language.english.lexicon import EnglishLexicon
from hbllm.hcir.language.english.parser import EnglishParser
from hbllm.hcir.language.ontology_bootstrap import LexicalOntologyRecord
from hbllm.hcir.language.reading_curriculum import (
    ReadingCurriculumIngestor,
    commit_propositions,
    extract_propositions_from_text,
)
from hbllm.hcir.nodes import HCIREdgeType
from hbllm.hcir.stores.ontology_store import LexicalOntologyStore
from hbllm.hcir.workspace_tiers import TieredWorkspace


def test_live_memory_store_detects_conceptnet_antonym_contradiction(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_semantic_relations([("en", "dry", "Antonym", "wet", 4.5, "conceptnet")])

    tw = TieredWorkspace()
    executor = MemoryCapabilityExecutor(
        "memory.store",
        tiered_workspace=tw,
        ontology_store=store,
    )

    r1 = asyncio.run(executor.execute({"content": "The river is dry"}))
    assert r1["ok"] is True
    assert r1["contradictions_detected"] == 0

    r2 = asyncio.run(executor.execute({"content": "The river is wet"}))
    assert r2["ok"] is True
    assert r2["contradictions_detected"] >= 1
    assert "contradiction" in r2["content"].lower()

    has_contra = any(
        e.edge_type == HCIREdgeType.CONTRADICTS for e in tw.persistent.graph.edges.values()
    )
    assert has_contra is True
    store.close()


def test_ditransitive_and_verbnet_recipient_thematic_roles(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [
            LexicalOntologyRecord(
                "give",
                "verb",
                "en",
                "Transfer possession.",
                thematic_roles=("Agent", "Theme", "Recipient"),
            )
        ],
        source="verbnet:test",
    )
    parser = EnglishParser(lexicon=EnglishLexicon(ontology_store=store))

    f1 = parser.parse("Alice gave Bob the brass key.")
    assert f1.frame_type == FrameType.ASSERTION
    assert f1.predicate == "give"
    assert f1.get_role(ThematicRole.AGENT) is not None
    assert f1.get_role(ThematicRole.RECIPIENT) is not None
    assert f1.get_role(ThematicRole.RECIPIENT).concept_name.lower() == "bob"  # type: ignore[union-attr]
    assert f1.get_role(ThematicRole.PATIENT) is not None
    assert "key" in (f1.get_role(ThematicRole.PATIENT).concept_name or "").lower()  # type: ignore[union-attr]

    f2 = parser.parse("Alice gave the brass key to Bob in the garden.")
    assert f2.frame_type == FrameType.ASSERTION
    assert f2.get_role(ThematicRole.RECIPIENT) is not None
    assert f2.get_role(ThematicRole.RECIPIENT).concept_name.lower() == "bob"  # type: ignore[union-attr]
    assert f2.get_role(ThematicRole.LOCATION) is not None
    assert f2.get_role(ThematicRole.LOCATION).concept_name.lower() == "garden"  # type: ignore[union-attr]

    props = extract_propositions_from_text(
        "Alice gave the brass key to Bob in the garden.",
        source_file="story.txt",
        include_extended=True,
        ontology_store=store,
    )
    give_prop = next(p for p in props if p.relation == "give")
    assert ("recipient", "bob") in give_prop.modifiers
    assert ("in", "garden") in give_prop.modifiers

    tw = TieredWorkspace()
    commit_propositions(props, tw.persistent, ontology_store=store)
    assert any(
        e.edge_type == HCIREdgeType.SUPPORTS
        and e.properties.get("relation") == "give_recipient"
        and "t4_concept_bob" in e.targets
        for e in tw.persistent.graph.edges.values()
    )
    store.close()


def test_coreference_paragraph_reset_and_narrative_before_chaining(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [
            LexicalOntologyRecord(
                "give",
                "verb",
                "en",
                "Transfer possession.",
                thematic_roles=("Agent", "Theme", "Recipient"),
            )
        ],
        source="verbnet:test",
    )

    story = (
        "Alice found a brass key. She gave it to Bob. She unlocked the oak door.\n\n"
        "Bob walked into the room. Gravity is a force. Rain causes flood. Combustion requires oxygen."
    )
    props = extract_propositions_from_text(
        story,
        source_file="chapter1.txt",
        include_extended=True,
        ontology_store=store,
    )
    triples = {(p.subject, p.relation, p.obj) for p in props}
    assert ("alice", "find", "key") in triples
    assert ("alice", "give", "key") in triples
    assert ("alice", "unlock", "door") in triples
    assert ("bob", "walk", "room") in triples

    give_prop = next(p for p in props if p.subject == "alice" and p.relation == "give")
    assert ("recipient", "bob") in give_prop.modifiers

    tw = TieredWorkspace()
    commit_propositions(props, tw.persistent, ontology_store=store)
    before_edges = [
        e for e in tw.persistent.graph.edges.values() if e.edge_type == HCIREdgeType.BEFORE
    ]
    assert len(before_edges) == 2
    before_pairs = {(e.sources[0], e.targets[0]) for e in before_edges}
    assert ("t4_belief_alice_find_key", "t4_belief_alice_give_key") in before_pairs
    assert ("t4_belief_alice_give_key", "t4_belief_alice_unlock_door") in before_pairs
    # Cross-paragraph static assertions must NOT get BEFORE chains
    assert not any("gravity" in e.sources[0] or "rain" in e.sources[0] for e in before_edges)
    store.close()


def test_ingest_book_salience_consolidation_across_chapters(tmp_path: Path) -> None:
    book_lines = [
        "Chapter One. Gravity causes attraction. Gravity keeps planets in orbit.",
        "",
        "A pebble sits silently. An ant crawls away. Dust settles slowly.",
        "",
        "Chapter Two. Gravity requires mass. Orbit requires gravity.",
        "",
        "Wind blows gently. Cloud drifts lazily. Leaf falls quietly.",
    ]
    book_path = tmp_path / "physics_book.txt"
    book_path.write_text("\n".join(book_lines), encoding="utf-8")

    tw = TieredWorkspace()
    stats = ReadingCurriculumIngestor().ingest_book(
        book_path,
        tw.persistent,
        max_propositions=3,
    )
    assert stats.files_read == 1
    assert stats.causal_extracted >= 1 or stats.dependency_extracted >= 1
    statements = " ".join(
        getattr(n, "statement", "") for n in tw.persistent.graph.nodes.values()
    ).lower()
    assert "gravity" in statements


def test_multilingual_si_ta_book_reading_aligns_to_hcir_concepts(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [
            LexicalOntologyRecord("tree", "noun", "en", "Perennial woody plant."),
            LexicalOntologyRecord("plant", "noun", "en", "Living photosynthetic organism."),
            LexicalOntologyRecord("rain", "noun", "en", "Condensed atmospheric water."),
            LexicalOntologyRecord("flood", "noun", "en", "Overflow of water."),
        ],
        source="core_ontology",
    )
    store.upsert_concept_alignments(
        [
            ("si", "ගස", "tree", 0.9, "osdb"),
            ("si", "ශාකයක්", "plant", 0.9, "osdb"),
            ("si", "වැස්ස", "rain", 0.9, "osdb"),
            ("si", "ගංවතුර", "flood", 0.9, "osdb"),
            ("ta", "மரம்", "tree", 0.9, "eng2tamildict"),
            ("ta", "தாவரம்", "plant", 0.9, "eng2tamildict"),
            ("ta", "மழை", "rain", 0.9, "eng2tamildict"),
            ("ta", "வெள்ளம்", "flood", 0.9, "eng2tamildict"),
        ]
    )

    si_props = extract_propositions_from_text(
        "ගස ශාකයක් වේ. වැස්ස ගංවතුර හේතු වේ.",
        source_file="si_book.txt",
        include_extended=True,
        ontology_store=store,
    )
    si_triples = {(p.subject, p.relation, p.obj) for p in si_props}
    assert ("tree", "is_a", "plant") in si_triples
    assert ("rain", "causes", "flood") in si_triples

    ta_props = extract_propositions_from_text(
        "மரம் தாவரம் ஆகும். மழை வெள்ளம் காரணமாகிறது.",
        source_file="ta_book.txt",
        include_extended=True,
        ontology_store=store,
    )
    ta_triples = {(p.subject, p.relation, p.obj) for p in ta_props}
    assert ("tree", "is_a", "plant") in ta_triples
    assert ("rain", "causes", "flood") in ta_triples
    store.close()
