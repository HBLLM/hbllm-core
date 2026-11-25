"""Tests for domain-neutral ObservationProtocol and dynamic PhysicsPredictor bounds (M-1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.world.percept import EmbodiedObservation, ObservationProtocol, TerminalSignal
from hbllm.hcir.world.predictors.physics import PhysicsPredictor


@dataclass(frozen=True)
class _NonLatticeSymbolicObservation:
    available_actions: tuple[int, ...] = (1, 2)
    entities: tuple[str, ...] = ("door_a", "key_b")
    context_id: str = "symbolic_ctx"
    episode_id: str = "ep_1"
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal: TerminalSignal | None = None


def test_observation_protocol_supports_lattice_and_non_lattice() -> None:
    embodied = EmbodiedObservation(available_actions=[1, 2, 3, 4], context_id="c1")
    symbolic = _NonLatticeSymbolicObservation()
    assert isinstance(embodied, ObservationProtocol)
    assert isinstance(symbolic, ObservationProtocol)


def test_physics_predictor_infers_bounds_when_grid_shape_none() -> None:
    assert PhysicsPredictor.is_corner_deadlock(
        entity_pos=(0, 0),
        barrier_cells={(0, 1), (1, 0)},
        target_positions=set(),
    )
    assert not PhysicsPredictor.is_2x2_deadlock(
        entity_pos=(5, 5),
        barrier_cells=set(),
        block_cells=set(),
        target_positions=set(),
    )
    path = PhysicsPredictor.compute_geodesic_path(
        start=(70, 70),
        goal=(85, 85),
        barrier_cells=set(),
    )
    assert path[0] == (70, 70)
    assert path[-1] == (85, 85)
    assert all(r >= 64 and c >= 64 for r, c in path)
