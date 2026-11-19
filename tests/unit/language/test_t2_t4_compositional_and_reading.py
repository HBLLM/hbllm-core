"""Tests for Track T2 compositional/imperative parser and Track T4 reading curriculum."""

from __future__ import annotations

from pathlib import Path

from hbllm.hcir.language.core.semantic_frame import FrameType, ThematicRole
from hbllm.hcir.language.english.lexicon import EnglishLexicon
from hbllm.hcir.language.english.parser import EnglishParser
from hbllm.hcir.language.ontology_bootstrap import OntologyBootstrapper
from hbllm.hcir.language.ontology_store import LexicalOntologyStore
from hbllm.hcir.language.reading_curriculum import (
    ReadingCurriculumIngestor,
    extract_propositions_from_text,
)
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_t2_compositional_discourse_and_imperative_commands(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "t2_lex.db")
    OntologyBootstrapper().ingest_core_ontology(None, ontology_store=store)
    parser = EnglishParser(lexicon=EnglishLexicon(ontology_store=store))

    frames = parser.parse_discourse("Read the file and verify the code.")
    assert len(frames) == 2
    assert frames[0].frame_type == FrameType.COMMAND
    assert frames[0].predicate == "read"
    p0 = frames[0].get_role(ThematicRole.PATIENT)
    assert p0 is not None and p0.concept_name == "file"
    assert frames[1].frame_type == FrameType.COMMAND
    assert frames[1].predicate == "verify"
    p1 = frames[1].get_role(ThematicRole.PATIENT)
    assert p1 is not None and p1.concept_name == "code"

    inside = parser.parse("The battery is inside the robot.")
    assert inside.frame_type == FrameType.ASSERTION
    assert inside.predicate == "located_in"
    store.close()


def test_t4_reading_curriculum_extracts_and_commits_causality(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "A compiler is a program. Overflow causes crash. Encryption requires key.\n",
        encoding="utf-8",
    )

    props = extract_propositions_from_text((docs / "guide.md").read_text(), source_file="guide.md")
    rels = {(p.subject, p.relation, p.obj) for p in props}
    assert ("compiler", "is_a", "program") in rels
    assert ("overflow", "causes", "crash") in rels
    assert ("encryption", "requires", "key") in rels

    store = LexicalOntologyStore(tmp_path / "t4_lex.db")
    system = build_cognitive_system(
        CognitiveSystemConfig(world_id="test_t4", memory_dir=None, enable_grounding=False)
    )
    grant = getattr(system.tiered_workspace, "_system_grant", None)
    stats = ReadingCurriculumIngestor().ingest_directories(
        [docs],
        system.tiered_workspace.persistent,
        ontology_store=store,
        grant=grant,
    )
    assert stats.taxonomic_extracted == 1
    assert stats.causal_extracted == 1
    assert stats.dependency_extracted == 1
    assert stats.edges_committed == 3
    assert store.is_a("compiler", "program") is True
    store.close()
