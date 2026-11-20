"""Tests for Track T3 Parallel Shadow Distillation (LLM Scaffold -> Pure-HCIR)."""

from __future__ import annotations

from pathlib import Path

from hbllm.hcir.kernel.autonomy_tracker import AutonomyTracker
from hbllm.hcir.language.ontology_store import LexicalOntologyStore
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
from plugins.llm_scaffold import (
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

    # Turn 1: With open-world auto-creation, HCIR now handles "drone" and "pad" natively
    # by auto-creating PhysicalEntityNodes from assertion context (no scaffold needed!)
    turn1 = system.process_dialogue("The drone is on the pad.")
    # Auto-creation means HCIR succeeds natively — LanguageTurnResult, not dict
    if isinstance(turn1, dict):
        # Scaffold fallback path (if ontology lacks the words)
        assert turn1["distilled"] is True
        assert tracker.shadow_invocations == 1
    else:
        # Pure HCIR path (open-world auto-creation succeeded)
        assert turn1.is_success is True
        assert turn1.response_text == "Understood."

    # Turns 2-5: Pure-HCIR now parses, grounds, and commits 100% natively (ZERO new shadow calls!)
    initial_shadow = tracker.shadow_invocations
    for _ in range(4):
        turn_native = system.process_dialogue("The drone is on the pad.")
        assert not isinstance(turn_native, dict)
        assert turn_native.is_success is True

    assert tracker.shadow_invocations == initial_shadow  # No new shadow calls
    assert tracker.hcir_parse_autonomy_rate >= 0.8
    assert tracker.can_retire_llm is True
    store.close()


def test_nvidia_nim_teacher_parses_chat_completion_response(monkeypatch: object) -> None:
    import io
    import json
    import urllib.request

    from plugins.llm_scaffold import NvidiaNIMTeacher

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    captured: dict[str, object] = {}

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> _FakeResp:
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "response_text": "The satellite is in orbit.",
                                "claim": "satellite located_in orbit",
                                "entities": [{"name": "satellite", "type": "satellite"}],
                                "lexicon": [],
                                "confidence": 0.91,
                            }
                        )
                    }
                }
            ]
        }
        return _FakeResp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)  # type: ignore[attr-defined]
    teacher = NvidiaNIMTeacher(api_key="nvapi-test-key", model="meta/llama-3.1-8b-instruct")
    result = teacher("The satellite is in orbit.")

    assert captured["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured["auth"] == "Bearer nvapi-test-key"
    assert result["verified"] is True
    assert result["claim"] == "satellite located_in orbit"
    assert result["entities"] == [{"name": "satellite", "type": "satellite"}]
