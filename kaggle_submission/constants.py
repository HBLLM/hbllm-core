"""Peripheral action schema definitions for ARC-AGI-3 environments."""

from __future__ import annotations

from typing import Any

ARC_DEFAULT_ACTION_SCHEMAS: list[dict[str, Any]] = [
    {"name": "MOVE_UP", "action_id": 1, "delta_r": -1, "delta_c": 0},
    {"name": "MOVE_DOWN", "action_id": 2, "delta_r": 1, "delta_c": 0},
    {"name": "MOVE_LEFT", "action_id": 3, "delta_r": 0, "delta_c": -1},
    {"name": "MOVE_RIGHT", "action_id": 4, "delta_r": 0, "delta_c": 1},
    {"name": "INTERACT", "action_id": 5},
    {
        "name": "CLICK_CELL",
        "action_id": 6,
        "parameters": {"x": "int", "y": "int"},
        "is_spatial_effector": True,
    },
    {"name": "ACTION7", "action_id": 7},
]
