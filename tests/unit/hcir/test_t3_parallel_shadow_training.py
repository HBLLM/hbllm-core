"""Tests for Track T3 Parallel Shadow Distillation (LLM Scaffold -> Pure-HCIR)."""

from __future__ import annotations

from pathlib import Path

from hbllm.hcir.language.ontology_store import LexicalOntologyStore
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
from plugins.llm_scaffold import (
    AutonomyTracker,
    LLMTeacherScaffold,
    parse_teacher_json,
)


def test_parse_teacher_json_extracts_and_validates() -> None:
    raw = (
        "Here is the JSON:\n"
        '{"response_text": "ok", "claim": "drone near tower", '
        '"entities": [{"name": "drone", "type": "drone"}], '
        '"lexicon": [{"lemma": "drone", "pos": "noun", "definition": "unmanned aerial vehicle", "hypernyms": ["aircraft"]}], '
        '"confidence": 0.9}'
    )
    parsed = parse_teacher_json(raw)
    assert parsed["verified"] is True
    assert parsed["claim"] == "drone near tower"
    assert parsed["entities"][0]["name"] == "drone"
    assert parsed["lexicon"][0]["lemma"] == "drone"


def test_t3_shadow_distillation_enables_pure_hcir_native_autonomy(tmp_path: Path) -> None:
    store = LexicalOntologyStore(tmp_path / "t3_lexicon.db")
    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_t3_distillation",
            memory_dir=None,
            enable_grounding=False,
            enable_brain_llm=True,
        )
    )
    tracker = AutonomyTracker(min_turns_for_retirement=4, retirement_threshold=0.75)
    system.services.extensions["autonomy_metrics"] = tracker

    def teacher(utterance: str) -> dict[str, object]:
        return {
            "response_text": "The drone is on the pad.",
            "claim": "drone located_on pad",
            "entities": [
                {"name": "drone", "type": "drone"},
                {"name": "pad", "type": "pad"},
            ],
            "lexicon": [
                {
                    "lemma": "drone",
                    "pos": "noun",
                    "definition": "aerial vehicle",
                    "hypernyms": ["vehicle"],
                },
                {
                    "lemma": "pad",
                    "pos": "noun",
                    "definition": "landing surface",
                    "hypernyms": ["surface"],
                },
            ],
            "confidence": 0.92,
            "verified": True,
        }

    scaffold = LLMTeacherScaffold(tracker=tracker, teacher_fn=teacher, ontology_store=store)
    system.services.extensions["llm_scaffold"] = scaffold

    # Ensure MultilingualLanguageRuntime uses our temporary store
    from hbllm.hcir.language.runtime import MultilingualLanguageRuntime

    system.services.extensions["language_runtime"] = MultilingualLanguageRuntime(
        system.workspace.graph,
        workspace=system.workspace,
        ontology_store=store,
    )

    # Turn 1: HCIR lacks 'drone' and 'pad' -> shadow scaffold distills lexicon + entities + belief
    turn1 = system.process_dialogue("The drone is on the pad.")
    assert isinstance(turn1, dict)
    assert turn1["distilled"] is True
    assert tracker.shadow_invocations == 1
    assert tracker.distilled_lexicon_entries == 2
    assert tracker.distilled_entities == 2
    assert store.has_word("drone") is True
    assert store.has_word("pad") is True

    # Turns 2-5: Pure-HCIR now parses, grounds, and commits 100% natively (ZERO new shadow calls!)
    for _ in range(4):
        turn_native = system.process_dialogue("The drone is on the pad.")
        assert not isinstance(turn_native, dict)
        assert turn_native.is_success is True

    assert tracker.shadow_invocations == 1
    assert tracker.hcir_parse_autonomy_rate >= 0.8
    assert tracker.can_retire_llm is True
    store.close()
