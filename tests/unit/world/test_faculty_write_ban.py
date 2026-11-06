from __future__ import annotations

import ast
from pathlib import Path

import pytest

WORLD = Path("hbllm/hcir/world")


def _writes_to_engine(tree: ast.AST) -> list[str]:
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id in ("engine", "state", "staged")
                ):
                    bad.append(tgt.attr)
    return bad


@pytest.mark.parametrize(
    "path", sorted(WORLD.glob("cortex_*.py")) + [WORLD / "macro_actions.py"], ids=lambda p: p.name
)
def test_no_direct_engine_field_writes(path):
    tree = ast.parse(path.read_text())
    bad = _writes_to_engine(tree)
    assert bad == [], f"{path}: faculty writes engine fields directly: {bad}"


def test_read_only_view_blocks_writes():
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.engine_state import ReadOnlyEngineView

    eng = AutonomousEpistemicEngine(enable_logging=False, seed=0)
    view = ReadOnlyEngineView(eng)
    with pytest.raises(AttributeError):
        view.exploration_budget = 5
    assert view.exploration_budget == eng.exploration_budget


def test_state_delta_semantics():
    from hbllm.hcir.world.engine_state import StateDelta

    d = StateDelta()
    assert d.is_empty()
    d.set("phase", "x")
    assert d.get("phase") == "x"
    assert not d.is_empty()
    lst = d.mutate("items", [1])
    lst.append(2)
    assert d.items()["items"] == [1, 2]


def test_apply_delta_assigns_engine_property():
    """apply_delta must reach engine properties (working_memory), not shadow state."""
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.engine_state import StateDelta

    eng = AutonomousEpistemicEngine(enable_logging=False, seed=0)
    d = StateDelta()
    d.set("level_epistemic_probes", 7)
    eng.apply_delta(d, author="test")
    assert eng.level_epistemic_probes == 7


def test_journaled_view_write_through():
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.engine_state import FacultyWriteJournal, JournaledEngineView

    eng = AutonomousEpistemicEngine(enable_logging=False, seed=0)
    j = FacultyWriteJournal(author="t")
    view = JournaledEngineView(eng, j)

    # write-through: engine sees it immediately, journal records before-image
    prev = eng.avatar_pos
    view.avatar_pos = (3, 4)
    assert eng.avatar_pos == (3, 4)
    assert "avatar_pos" in j.touched
    assert j.before["avatar_pos"] == prev

    # in-place container edit lands live; field journaled
    view.tested_actions.add(9)
    assert 9 in eng.tested_actions
    assert "tested_actions" in j.touched

    # nested state proxy: write-through + journaled as "state.X"
    view.state.prev_grid = None
    assert "state.prev_grid" in j.touched

    # owned subsystem returned as identity
    assert view.symbolic_theory is eng.symbolic_theory
    assert view.working_memory is eng.working_memory

    # journal records no change -> commit_faculty_journal is a no-op True
    assert eng.commit_faculty_journal(FacultyWriteJournal(author="t")) is True


def test_faculty_journal_commits_to_tm_workspace():
    """Journaled faculty writes must land on the TM-governed workspace and not
    be reverted when a CommitAuthorityRegistry is active (WP1 fix)."""
    from hbllm.hcir.kernel.services import KernelServices
    from hbllm.hcir.workspace import HCIRWorkspaceState
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.engine_state import FacultyWriteJournal, JournaledEngineView

    ws = HCIRWorkspaceState()
    services = KernelServices.create_default(ws)
    eng = AutonomousEpistemicEngine(enable_logging=False, seed=0, services=services)

    j = FacultyWriteJournal(author="test")
    view = JournaledEngineView(eng, j)
    view.tested_actions.add(3)
    view.consecutive_plan_failures = 5  # a _MIRRORED field -> emits ops
    assert eng.commit_faculty_journal(j) is True
    # committed, not reverted
    assert eng.consecutive_plan_failures == 5
    assert 3 in eng.tested_actions
    assert getattr(eng, "rejected_faculty_commits", 0) == 0
    # ops actually reached the TM-governed workspace (get_node — .nodes is
    # not exposed on the native-adapter facade)
    assert ws.graph.get_node("autonomous_epistemic_engine") is not None
    assert len(getattr(services.transaction_manager, "committed_log", [])) >= 1
