from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import pytest

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortical_pipeline import CorticalPipeline
from hbllm.hcir.world.cortical_stage import StageResult, WorkingMemoryDelta
from hbllm.hcir.world.engine_state import CognitiveContext, FacultyWriteJournal, JournaledEngineView
from hbllm.hcir.world.percept import EpistemicPhase
from hbllm.hcir.world.prefrontal_working_memory import PrefrontalWorkingMemory


def test_no_direct_engine_in_world_modules() -> None:
    """Invariant 2 & 8: Verify no hardcoded string literal '_engine' exists in hbllm/hcir/world."""
    world_dir = Path("hbllm/hcir/world")
    assert world_dir.is_dir()
    violating_files = []
    pattern = re.compile(r'"_engine"')
    for py_file in world_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if pattern.search(content):
            violating_files.append(py_file.name)
    assert not violating_files, f"Found literal '_engine' in: {violating_files}"


def test_no_direct_view_writes_in_cortex_modules() -> None:
    """Invariant 2: Verify zero direct assignments to view.<attr> in cortex modules."""
    world_dir = Path("hbllm/hcir/world")
    write_pattern = re.compile(r"view\.[a-z_]+\s*=[^=]")
    violating_sites: list[str] = []
    for cortex_file in world_dir.glob("cortex_*.py"):
        lines = cortex_file.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            if write_pattern.search(line):
                violating_sites.append(f"{cortex_file.name}:{idx}: {line.strip()}")
    assert not violating_sites, "Direct view assignments detected:\n" + "\n".join(violating_sites)


def test_working_memory_mirrored_fields() -> None:
    """Invariant 8: Mirrored fields are dynamically obtained from PrefrontalWorkingMemory."""
    mirrored = PrefrontalWorkingMemory.get_mirrored_fields()
    assert isinstance(mirrored, frozenset)
    assert "phase" in mirrored
    assert "step_counter" in mirrored
    assert "avatar_pos" in mirrored
    assert "simulation_cooldown" in mirrored
    assert "consecutive_simulation_failures" in mirrored
    assert "consecutive_plan_failures" in mirrored


def test_working_memory_delta_journaling() -> None:
    """Invariant 2: JournaledEngineView records WorkingMemoryDelta and before-images."""
    engine = AutonomousEpistemicEngine()
    journal = FacultyWriteJournal(author="test_faculty")
    view = JournaledEngineView(engine, journal)

    # Initially cooldown is 0
    assert engine.working_memory.simulation_cooldown == 0

    delta = WorkingMemoryDelta(field="simulation_cooldown", value=4)
    view.apply_wm_delta(delta)

    # Value updated on working_memory
    assert engine.working_memory.simulation_cooldown == 4
    # Recorded in journal deltas
    assert len(journal.deltas) == 1
    assert journal.deltas[0] == delta
    # Before-image recorded for rollback (property on engine recorded as simulation_cooldown)
    assert "simulation_cooldown" in journal.before
    assert journal.before["simulation_cooldown"] == 0

    # Also test field solely on working_memory
    delta_wm = WorkingMemoryDelta(field="focal_attention_point", value=(3, 4))
    view.apply_wm_delta(delta_wm)
    assert engine.working_memory.focal_attention_point == (3, 4)
    assert "working_memory.focal_attention_point" in journal.before
    assert journal.before["working_memory.focal_attention_point"] is None


def test_cortical_stage_exception_rollback_byte_identity() -> None:
    """Invariant 1 & 2: Stage exception triggers journal rollback preserving byte-identical working memory."""
    engine = AutonomousEpistemicEngine()
    engine.working_memory.simulation_cooldown = 1
    engine.working_memory.consecutive_simulation_failures = 2
    engine.working_memory.phase = EpistemicPhase.MOTOR_GROUNDING

    # Take byte-level before snapshot
    before_bytes = pickle.dumps(engine.working_memory)

    class FailingStage:
        name = "failing_stage"
        engine: Any = None
        _current_journal: Any = None

        def bind_engine(self, eng: Any) -> None:
            self.engine = eng

        def observe(self, ctx: CognitiveContext) -> StageResult:
            journal = FacultyWriteJournal(author=self.name)
            self._current_journal = journal
            view = JournaledEngineView(self.engine, journal)
            try:
                view.apply_wm_delta(WorkingMemoryDelta("simulation_cooldown", 99))
                view.apply_wm_delta(WorkingMemoryDelta("consecutive_simulation_failures", 42))
                view.apply_wm_delta(WorkingMemoryDelta("phase", EpistemicPhase.EXPLOITATION))
                # Trigger an unexpected exception
                raise RuntimeError("Simulated failure mid-stage")
            except Exception:
                self.engine.rollback_faculty_journal(journal)
                raise
            finally:
                self._current_journal = None

    pipeline = CorticalPipeline(
        stages=[FailingStage()],
        engine=engine,
    )

    ctx = CognitiveContext(step_counter=1)

    with pytest.raises(RuntimeError, match="Simulated failure mid-stage"):
        pipeline.run(ctx)

    # State was reverted to initial values
    assert engine.working_memory.simulation_cooldown == 1
    assert engine.working_memory.consecutive_simulation_failures == 2
    assert engine.working_memory.phase == EpistemicPhase.MOTOR_GROUNDING

    # Working memory is byte-identical
    after_bytes = pickle.dumps(engine.working_memory)
    assert after_bytes == before_bytes, "Working memory bytes diverged after rollback!"


def test_engine_delegated_properties_invariant_8() -> None:
    """Invariant 8: Engine attributes route to formal internal memory stores without ad-hoc sprawl."""
    engine = AutonomousEpistemicEngine()

    # relational_confidence_map delegates to symbolic_theory
    engine.relational_confidence_map["rel_1"] = 0.85
    assert engine.symbolic_theory.relational_confidence_map["rel_1"] == 0.85

    # evidence_provenance_log delegates to working_memory
    rec = {"claim": "c1", "source": "s1"}
    engine.evidence_provenance_log.append(rec)
    assert engine.working_memory.evidence_provenance_log[-1] == rec

    # pipeline_runs delegates to working_memory
    engine.pipeline_runs = 5
    assert engine.working_memory.pipeline_runs == 5
    assert engine._pipeline_runs == 5

    # last_decision_branch delegates to working_memory
    engine.last_decision_branch = "macro_plan"
    assert engine.working_memory.last_decision_branch == "macro_plan"
    assert engine._last_decision_branch == "macro_plan"
