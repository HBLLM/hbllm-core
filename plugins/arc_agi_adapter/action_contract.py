"""Action contract validation + effect reconciliation (WP0.4).

Legal action ids, ACTION6 point data bounds, frame-vs-lattice scaling, and
post-step observation of whether the environment applied the action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EffectKind(Enum):
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    ENV_REJECTED = "env_rejected"


@dataclass(frozen=True)
class ContractResult:
    ok: bool
    action_id: int
    data: dict[str, Any] | None = None
    violation: str | None = None
    repaired_action_id: int | None = None


@dataclass(frozen=True)
class EffectObservation:
    kind: EffectKind
    changed_cells: int = 0
    changed_bbox: tuple[int, int, int, int] | None = None
    avatar_delta: tuple[int, int] | None = None


class ActionContract:
    """Validates action payloads against frame bounds and availability."""

    VALID_IDS = frozenset(range(0, 8))

    @classmethod
    def validate(
        cls,
        action_id: int,
        action_data: dict[str, Any] | None,
        available_actions: list[int],
        grid_shape: tuple[int, int],
    ) -> ContractResult:
        if not isinstance(action_id, int) or action_id not in cls.VALID_IDS:
            return ContractResult(False, action_id, violation=f"illegal_action_id:{action_id}")
        if action_id not in available_actions and action_id != 0:
            return ContractResult(False, action_id, violation=f"action_unavailable:{action_id}")
        if action_id == 6:
            if (
                not isinstance(action_data, dict)
                or "x" not in action_data
                or "y" not in action_data
            ):
                return ContractResult(
                    False, action_id, data=action_data, violation="action6_missing_xy"
                )
            x, y = action_data["x"], action_data["y"]
            if not isinstance(x, int) or not isinstance(y, int):
                return ContractResult(
                    False, action_id, data=action_data, violation="action6_non_int_xy"
                )
            H, W = grid_shape
            if not (0 <= y < H and 0 <= x < W):
                return ContractResult(
                    False, action_id, data=action_data, violation="action6_out_of_bounds"
                )
        return ContractResult(True, action_id, data=action_data)

    @staticmethod
    def _nearest_legal(action_id: int, available_actions: list[int]) -> int | None:
        """Deterministic fallback: nearest available discrete action id."""
        cands = sorted(a for a in available_actions if a != 6) or list(available_actions)
        if not cands:
            return None
        return min(cands, key=lambda a: (abs(a - action_id), a))

    @classmethod
    def validate_or_repair(
        cls,
        action_id: int,
        action_data: dict[str, Any] | None,
        available_actions: list[int],
        grid_shape: tuple[int, int],
    ) -> ContractResult:
        """Validate; on violation, deterministically pick the nearest legal action."""
        res = cls.validate(action_id, action_data, available_actions, grid_shape)
        if res.ok:
            return res
        repaired = cls._nearest_legal(action_id, available_actions)
        if repaired is not None:
            logger.warning(
                "Action contract violation (%s) for action %s — repaired to %s",
                res.violation,
                action_id,
                repaired,
            )
        return ContractResult(
            False,
            action_id,
            data=action_data,
            violation=res.violation,
            repaired_action_id=repaired,
        )

    @staticmethod
    def reconcile(
        prev_grid: Any,
        new_grid: Any,
        action_id: int,
        action_was_available: bool = True,
    ) -> EffectObservation:
        """Classify post-step diff: applied, unchanged, or env-rejected.

        Zero diff is UNCHANGED (blocked moves are real motor evidence).
        ENV_REJECTED only when the env itself rejected the actuation:
        the sent action was not in available_actions, or the new frame is
        malformed (shape changed / missing).
        """
        import numpy as np

        if not action_was_available:
            return EffectObservation(EffectKind.ENV_REJECTED)
        prev, new = np.asarray(prev_grid), np.asarray(new_grid)
        if prev.shape != new.shape:
            return EffectObservation(EffectKind.ENV_REJECTED)
        diff = prev != new
        changed = int(diff.sum())
        if changed == 0:
            return EffectObservation(EffectKind.UNCHANGED)
        ys, xs = np.nonzero(diff)
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
        return EffectObservation(EffectKind.CHANGED, changed_cells=changed, changed_bbox=bbox)
