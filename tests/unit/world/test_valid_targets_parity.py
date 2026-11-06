"""Valid-target probing and perf-parity regression tests.

Task A (behavior): when a point actuator's affordance declares a discrete
``valid_targets`` lattice, every probe payload the engine emits must be drawn
from that lattice — arbitrary coordinates are physical no-ops.

Task B (parity): the optimized paths (JournaledEngineView attribute routing,
extract_entities component lifting, IoU, Euler characteristic) must produce
outputs identical to the pre-change implementations.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortex_causal import ActionAffordance
from hbllm.hcir.world.cortex_perception import PerceptionEngine
from hbllm.hcir.world.engine_state import FacultyWriteJournal, JournaledEngineView
from hbllm.hcir.world.macro_actions import InteractWith
from hbllm.hcir.world.spatiotemporal_tracker import MorphologicalDeformationTracker

# ─────────────────────────────────────────────────────────────────────────────
# Task A: valid-target probing
# ─────────────────────────────────────────────────────────────────────────────

_VT = [{"x": 1, "y": 1}, {"x": 2, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 2}]


def _engine_with_point_actuator(
    valid_targets: list[dict[str, int]] | None,
) -> AutonomousEpistemicEngine:
    """Engine with a single declared point actuator (action 6)."""
    engine = AutonomousEpistemicEngine()
    meta = {"valid_targets": list(valid_targets)} if valid_targets else {}
    engine.action_affordances[6] = ActionAffordance(
        action_id=6,
        name="ACTION6",
        requires_spatial_target=True,
        target_param_keys=("x", "y"),
        metadata=meta,
    )
    return engine


def test_ground_effector_action_restricts_to_valid_targets_featureless() -> None:
    """Featureless lattice: probe must still land on a declared valid target."""
    engine = _engine_with_point_actuator(_VT)
    grid = np.zeros((8, 8), dtype=np.int32)
    for _ in range(len(_VT) * 2):
        data = engine.ground_effector_action(grid, 6)
        assert data is not None
        assert data in _VT, f"probe payload {data} left the valid-target lattice"


def test_ground_effector_action_snaps_salient_site_to_nearest_valid() -> None:
    """A salient entity off-lattice snaps to the nearest declared target."""
    engine = _engine_with_point_actuator(_VT)
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[3, 1] = 3  # interior salient entity, off the declared lattice
    data = engine.ground_effector_action(grid, 6)
    assert data is not None
    assert data in _VT
    # Nearest declared target to (r=3, c=1) is (vy=2, vx=1).
    assert data == {"x": 1, "y": 2}


def test_ground_effector_action_unchanged_without_valid_targets() -> None:
    """Absent/empty valid_targets: legacy behavior is preserved."""
    engine = _engine_with_point_actuator(None)
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[3, 3] = 3
    data = engine.ground_effector_action(grid, 6)
    assert data is None or data.get("x") is not None
    # No valid_targets: payload must be a coordinate dict, not a declared payload.
    if data is not None:
        assert set(data) <= {"x", "y"}


def test_ground_effector_action_enumerates_valid_targets() -> None:
    """Quiescent declared targets are skipped; the lattice is enumerated."""
    engine = _engine_with_point_actuator(_VT)
    engine.quiescent_actuation_targets.add((1, 1))  # (r=1,c=1) proven inert
    grid = np.zeros((8, 8), dtype=np.int32)
    seen: set[tuple[int, int]] = set()
    for _ in range(3):
        data = engine.ground_effector_action(grid, 6)
        assert data is not None
        seen.add((data["y"], data["x"]))
    assert (1, 1) not in seen
    assert seen <= {(vt["y"], vt["x"]) for vt in _VT}


def test_interact_with_snaps_target_to_valid_lattice() -> None:
    """InteractWith on an off-lattice object emits a declared target payload."""
    engine = _engine_with_point_actuator(_VT)
    grid = np.zeros((8, 8), dtype=np.int32)
    op = InteractWith(target_coord=(3, 1), effector_action=6)
    result = op.step(engine, grid, [6])
    assert result is not None
    action, data = result
    assert action == 6
    assert data in _VT
    assert data == {"x": 1, "y": 2}


def test_interact_with_unaffected_without_valid_targets() -> None:
    engine = _engine_with_point_actuator(None)
    grid = np.zeros((8, 8), dtype=np.int32)
    op = InteractWith(target_coord=(3, 5), effector_action=6)
    action, data = op.step(engine, grid, [6])  # type: ignore[misc]
    assert action == 6
    assert data == {"x": 5, "y": 3}


# ─────────────────────────────────────────────────────────────────────────────
# Task B parity: JournaledEngineView journaling semantics
# ─────────────────────────────────────────────────────────────────────────────


def test_journaled_view_before_images_and_touch_semantics() -> None:
    """First-touch before-images + touched set identical to reference semantics."""
    engine = AutonomousEpistemicEngine()
    engine.entity_visit_counts["probe_key"] = 7
    journal = FacultyWriteJournal(author="parity")
    view = JournaledEngineView(engine, journal)

    # Repeat reads of a mutable field: before-image from first touch only.
    assert view.entity_visit_counts["probe_key"] == 7
    assert journal.before["entity_visit_counts"] == {"probe_key": 7}
    assert "entity_visit_counts" in journal.touched
    assert view.entity_visit_counts["probe_key"] == 7
    assert journal.before["entity_visit_counts"] == {"probe_key": 7}

    # In-place mutation through the view lands on the engine (write-through).
    view.entity_visit_counts["probe_key"] = 9
    assert engine.entity_visit_counts["probe_key"] == 9
    assert journal.before["entity_visit_counts"] == {"probe_key": 7}

    # Immutable field reads return live values without journaling.
    n_before = len(journal.touched)
    _ = view.step_counter
    assert len(journal.touched) == n_before

    # state-proxy fields journal under "state.<name>" once.
    _ = view.state.step_counter
    _ = view.state.step_counter
    assert "state.step_counter" in journal.touched
    assert journal.before["state.step_counter"] == engine.state.step_counter

    # Owned subsystems pass through untouched.
    assert view.symbolic_theory is engine.symbolic_theory

    # Writes are recorded against pre-write values.
    view.step_counter = 42
    assert engine.step_counter == 42
    assert "step_counter" in journal.touched


def test_journaled_view_field_type_flip_is_journaled() -> None:
    """A field that becomes a container later is journaled on first mutable read."""
    engine = AutonomousEpistemicEngine()
    engine.last_action_data = None
    journal = FacultyWriteJournal(author="parity")
    view = JournaledEngineView(engine, journal)
    _ = view.last_action_data  # resolves immutable -> passthrough
    engine.last_action_data = {"x": 1}
    _ = view.last_action_data  # now mutable -> must be journaled
    assert journal.before["last_action_data"] == {"x": 1}
    assert "last_action_data" in journal.touched


# ─────────────────────────────────────────────────────────────────────────────
# Task B parity: extract_entities component lifting
# ─────────────────────────────────────────────────────────────────────────────


def _reference_component_coords(
    grid: np.ndarray, bg: int
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Pre-change per-component np.argwhere lifting (reference impl)."""
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    out: list[tuple[int, list[tuple[int, int]]]] = []
    for val in [int(v) for v in np.unique(grid) if int(v) != bg]:
        labeled_mask, num_components = ndimage.label(grid == val, structure=struct)
        for comp_idx in range(1, num_components + 1):
            coords = np.argwhere(labeled_mask == comp_idx)
            out.append((val, [tuple(int(x) for x in c) for c in coords]))
    return out


def test_extract_entities_matches_reference_lifting() -> None:
    rng = np.random.default_rng(0)
    for _ in range(6):
        h, w = int(rng.integers(8, 24)), int(rng.integers(8, 24))
        grid = rng.integers(0, 5, size=(h, w)).astype(np.int32)
        bg = int(np.bincount(grid.ravel()).argmax())
        entities = PerceptionEngine.extract_entities(grid, bg)
        got = [(e.feature_id, sorted(tuple(c) for c in e.properties["cells"])) for e in entities]
        ref = [(v, sorted(cells)) for v, cells in _reference_component_coords(grid, bg)]
        assert got == ref


# ─────────────────────────────────────────────────────────────────────────────
# Task B parity: spatiotemporal tracker primitives
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_iou_matches_reference() -> None:
    rng = np.random.default_rng(1)
    for _ in range(30):
        a = {(int(r), int(c)) for r, c in rng.integers(0, 10, size=(20, 2))}
        b = {(int(r), int(c)) for r, c in rng.integers(0, 10, size=(15, 2))}
        inter = len(a & b)
        union = len(a | b)
        ref = float(inter / union) if union > 0 else 0.0
        assert MorphologicalDeformationTracker.compute_iou(a, b) == ref


def test_euler_characteristic_large_complex_parity() -> None:
    """Both regimes (set-count and raster) agree with brute-force V - E + F."""
    rng = np.random.default_rng(2)
    mask = (rng.random((48, 48)) < 0.95).astype(bool)
    cells = {(int(rc[0]), int(rc[1])) for rc in np.argwhere(mask)}
    assert len(cells) >= 2048  # exercises the raster path
    verts: set[tuple[int, int]] = set()
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in cells:
        verts.update(((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)))
        for a, b in (
            ((r, c), (r, c + 1)),
            ((r + 1, c), (r + 1, c + 1)),
            ((r, c), (r + 1, c)),
            ((r, c + 1), (r + 1, c + 1)),
        ):
            edges.add((a, b) if a <= b else (b, a))
    ref = len(verts) - len(edges) + len(cells)
    assert MorphologicalDeformationTracker.compute_euler_characteristic(cells) == ref


# ─────────────────────────────────────────────────────────────────────────────
# Micro-timing sanity (wall-clock noise tolerated at this scale)
# ─────────────────────────────────────────────────────────────────────────────


def test_journaled_view_repeat_read_throughput() -> None:
    """Repeat mutable-field reads must not re-run journal bookkeeping."""
    engine = AutonomousEpistemicEngine()
    journal = FacultyWriteJournal(author="bench")
    view = JournaledEngineView(engine, journal)
    _ = view.entity_visit_counts  # first touch (journaled)
    t0 = time.perf_counter()
    for _ in range(20000):
        _ = view.entity_visit_counts
    repeat_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20000):
        _ = engine.entity_visit_counts
    direct_s = time.perf_counter() - t0
    # Repeat reads should cost no more than ~6x a bare attribute access.
    assert repeat_s < max(0.5, direct_s * 6 + 0.02)
