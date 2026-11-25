"""Verify DialogueService extraction and typed KernelServices wiring (M-4)."""

from __future__ import annotations

from pathlib import Path

from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
from hbllm.runtime.dialogue_service import DialogueService

ROOT_DIR = Path(__file__).resolve().parents[3]


def test_bootstrap_concise_and_wires_typed_services() -> None:
    loc = sum(1 for _ in (ROOT_DIR / "hbllm/runtime/bootstrap.py").open(encoding="utf-8"))
    assert loc <= 380, (
        f"bootstrap.py expected <= 380 LOC after DialogueService extraction, got {loc}"
    )

    sys_obj = build_cognitive_system(
        CognitiveSystemConfig(world_id="test_m4_dialogue", memory_dir=None, enable_grounding=False)
    )
    assert isinstance(sys_obj.dialogue_service, DialogueService)
    assert sys_obj.services.reasoning_runtime is sys_obj.services.extensions["reasoning_runtime"]
    assert sys_obj.services.neuromodulation is sys_obj.services.extensions["neuromodulation"]
    assert (
        sys_obj.services.lexicon_acquisition is sys_obj.services.extensions["lexicon_acquisition"]
    )

    turn = sys_obj.process_dialogue("The red ball is on the table.")
    assert turn.is_success
    assert sys_obj.services.language_runtime is not None
    assert sys_obj.services.ontology_store is not None
