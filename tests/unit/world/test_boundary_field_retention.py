"""WP0.5/F1: per-field retention policies across episode boundaries."""

from collections import deque

from hbllm.hcir.world.percept import (
    EpisodeBoundary,
    EpisodeBoundaryKind,
    RetentionPolicy,
    apply_boundary_retention,
)
from hbllm.hcir.world.spatiotemporal_tracker import SpatiotemporalHazardTracker

FACULTY_CLASSES = [SpatiotemporalHazardTracker]


def _populate(owner, field: str) -> None:
    cur = getattr(owner, field)
    if isinstance(cur, set):
        cur.add("x")
    elif isinstance(cur, dict):
        cur[("k", 0)] = "v"
    elif isinstance(cur, (list, deque)):
        cur.append("x")
    elif isinstance(cur, int):
        setattr(owner, field, 5)


def _is_empty(owner, field: str) -> bool:
    cur = getattr(owner, field)
    if isinstance(cur, (set, dict, list, deque)):
        return len(cur) == 0
    return cur in (0, None)


def test_all_faculties_field_retention_level_local_cleared() -> None:
    boundary = EpisodeBoundary(kind=EpisodeBoundaryKind.LEVEL_COMPLETE)
    for cls in FACULTY_CLASSES:
        inst = cls()
        declared = getattr(cls, "FIELD_RETENTION", None) or inst.FIELD_RETENTION
        assert declared, f"{cls.__name__} declares no FIELD_RETENTION"
        for field, policy in declared.items():
            inst2 = cls()
            _populate(inst2, field)
            apply_boundary_retention(inst2, boundary)
            if policy == RetentionPolicy.LEVEL_LOCAL:
                assert _is_empty(inst2, field), f"{cls.__name__}.{field} not cleared"
            else:
                assert not _is_empty(inst2, field), f"{cls.__name__}.{field} wrongly cleared"


def test_context_local_survives_death_boundary() -> None:
    inst = SpatiotemporalHazardTracker()
    inst.known_lethal_features.add(7)
    apply_boundary_retention(inst, EpisodeBoundary(kind=EpisodeBoundaryKind.DEATH))
    assert inst.known_lethal_features == {7}
