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


def test_ed_participle_verb_and_proper_noun_recipient_under_noisy_lexicon(
    tmp_path: Path,
) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [
            LexicalOntologyRecord("unlock", "verb", "en", "Release a lock."),
            LexicalOntologyRecord(
                "unlocked",
                "adj",
                "en",
                "Not secured or protected by an engaged lock.",
            ),
            LexicalOntologyRecord(
                "give",
                "verb",
                "en",
                "Transfer possession.",
                thematic_roles=("Agent", "Theme", "Recipient"),
            ),
            LexicalOntologyRecord("bob", "verb", "en", "Move up and down."),
        ],
        source="kaikki:test",
    )

    story = "Alice found a brass key. She gave it to Bob. She unlocked the oak door."
    props = extract_propositions_from_text(
        story,
        source_file="story.txt",
        include_extended=True,
        ontology_store=store,
    )
    triples = {(p.subject, p.relation, p.obj) for p in props}
    assert ("alice", "find", "key") in triples
    assert ("alice", "give", "key") in triples
    assert ("alice", "unlock", "door") in triples
    give_prop = next(p for p in props if p.relation == "give")
    assert ("recipient", "bob") in give_prop.modifiers

    tw = TieredWorkspace()
    commit_propositions(props, tw.persistent, ontology_store=store)
    before_edges = [
        e for e in tw.persistent.graph.edges.values() if e.edge_type == HCIREdgeType.BEFORE
    ]
    assert len(before_edges) == 2
    assert any(
        str(e.properties.get("relation", "")) == "give_recipient"
        for e in tw.persistent.graph.edges.values()
    )
    store.close()


def test_noise_filter_edge_dedupe_grant_fallback_and_explainer_sense_ranking(
    tmp_path: Path,
) -> None:
    from hbllm.hcir.capabilities.concept_explainer import ConceptExplainer
    from hbllm.hcir.language.english.lexicon import EnglishLexicon
    from hbllm.hcir.language.english.parser import EnglishParser
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    noisy_book = (
        "[Illustration: Dear Mother of my hart]\n"
        "Copyright, 1912\n"
        "Where you are know is good.\n"
        "Giovanna 53 found the lamp 105.\n"
        "Alice watched the bright museum.\n"
    )
    props = extract_propositions_from_text(
        noisy_book, source_file="noisy.txt", include_extended=True
    )
    subjects = {p.subject for p in props}
    objs = {p.obj for p in props}
    assert "illustration" not in subjects
    assert "copyright" not in subjects
    assert "where_you" not in subjects
    assert not any(o.endswith("_53") or o.endswith("_105") for o in objs)
    assert ("alice", "watch", "museum") in {(p.subject, p.relation, p.obj) for p in props}

    # Repeated propositions emitting identical edge IDs must not roll back batch commit,
    # and commit_propositions must fall back to workspace._system_grant when grant=None.
    sys = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_grant_dedupe",
            memory_dir=None,
            enable_grounding=False,
            enable_brain_llm=False,
            enable_autonomous_curiosity=False,
        )
    )
    parser = EnglishParser(lexicon=EnglishLexicon())
    dup_props = [
        *extract_propositions_from_text("Rain causes flood.", source_file="b1", parser=parser),
        *extract_propositions_from_text("Rain causes flood.", source_file="b2", parser=parser),
        *extract_propositions_from_text(
            "Arts exposition is in Boston.", source_file="b1", parser=parser
        ),
    ]
    _, edges_committed, _ = commit_propositions(
        dup_props,
        sys.tiered_workspace,
        grant=None,
    )
    assert edges_committed >= 2

    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [
            LexicalOntologyRecord(
                "tree",
                "n",
                "en",
                "English actor and theatrical producer (1853-1917)",
            ),
            LexicalOntologyRecord("tree", "noun", "en", "a tall perennial woody plant"),
        ],
        source="open_english_wordnet",
    )
    explainer = ConceptExplainer(
        tiered_workspace=sys.tiered_workspace,
        ontology_store=store,
    )
    tree_snap = explainer.gather_knowledge("tree")
    assert "woody plant" in tree_snap.definition
    assert "1853-1917" not in tree_snap.definition

    expo_snap = explainer.gather_knowledge("arts_exposition")
    assert expo_snap.has_knowledge is True
    assert any("located in boston" in b for b in expo_snap.beliefs)
    store.close()


def test_universal_engine_prefers_higher_frequency_stemmed_alignment(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "ont.db")
    store.upsert_many(
        [
            LexicalOntologyRecord("peony", "noun", "en", "Flowering plant."),
            LexicalOntologyRecord("plant", "noun", "en", "Living photosynthetic organism."),
        ],
        source="core_ontology",
    )
    store.record_frequencies({"peony": 2, "plant": 15000}, language="en")
    store.upsert_concept_alignments(
        [
            ("si", "ශාකයක්", "peony", 0.9, "osdb"),
            ("si", "ශාකය", "plant", 0.9, "osdb"),
            ("si", "ගස", "tree", 0.9, "osdb"),
        ]
    )
    props = extract_propositions_from_text(
        "ගස ශාකයක් වේ.",
        include_extended=True,
        ontology_store=store,
    )
    assert ("tree", "is_a", "plant") in {(p.subject, p.relation, p.obj) for p in props}
    store.close()
