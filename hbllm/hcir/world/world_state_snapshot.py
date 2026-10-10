"""
World State Snapshot — Immutable State Container for Deterministic Simulation & Replay.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorldStateSnapshot:
    """Immutable snapshot of environmental variables and physical entity states."""

    world_id: str
    timestamp: float = field(default_factory=time.time)
    variables: dict[str, Any] = field(default_factory=dict)
    entity_states: dict[str, str] = field(default_factory=dict)
    state_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute deterministic SHA256 state hash for simulation and replay matching."""

        def _default_serializer(obj: Any) -> Any:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, (set, frozenset)):
                return sorted(list(obj))
            return str(obj)

        raw_payload = {
            "world_id": self.world_id,
            "variables": sorted(self.variables.items()),
            "entity_states": sorted(self.entity_states.items()),
        }
        encoded = json.dumps(raw_payload, sort_keys=True, default=_default_serializer).encode(
            "utf-8"
        )
        computed_hash = hashlib.sha256(encoded).hexdigest()[:16]
        object.__setattr__(self, "state_hash", computed_hash)


class TemporalStateBuffer:
    """W041, W042: Timeline buffer tracking current and previous world states."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._history: list[WorldStateSnapshot] = []

    def push(self, snapshot: WorldStateSnapshot) -> None:
        """Append a new state snapshot to the chronological history."""
        self._history.append(snapshot)
        if len(self._history) > self.capacity:
            self._history.pop(0)

    @property
    def current_state(self) -> WorldStateSnapshot | None:
        """W041: Active current-state representation."""
        return self._history[-1] if self._history else None

    @property
    def previous_state(self) -> WorldStateSnapshot | None:
        """W042: Immediately preceding previous-state representation."""
        return self._history[-2] if len(self._history) >= 2 else None

    def get_history(self) -> list[WorldStateSnapshot]:
        """Return shallow copy of chronological state sequence."""
        return list(self._history)

    def rollback(self, steps: int = 1) -> WorldStateSnapshot | None:
        """Roll back buffer state by N steps, returning the restored state."""
        if steps <= 0:
            return self.current_state
        for _ in range(min(steps, len(self._history))):
            self._history.pop()
        return self.current_state


@dataclass
class StateDifferenceResult:
    """W043-W047: Quantified delta between two world states."""

    is_identical: bool
    changed_variables: dict[str, tuple[Any, Any]] = field(default_factory=dict)  # W045
    changed_entities: dict[str, tuple[str, str]] = field(default_factory=dict)
    changed_positions: dict[str, tuple[tuple[int, int], tuple[int, int]]] = field(
        default_factory=dict
    )  # W046
    changed_cells: list[tuple[int, int]] = field(default_factory=list)  # W044
    bounding_box_of_change: tuple[int, int, int, int] | None = None  # W044
    structural_changes: list[str] = field(default_factory=list)  # W047


class StateDifferenceComputer:
    """W043-W047: Computes multidimensional state differences, localized changes, and structural mutations."""

    @classmethod
    def compute_difference(
        cls, pre: WorldStateSnapshot, post: WorldStateSnapshot
    ) -> StateDifferenceResult:
        """W043: State difference computation between two state snapshots."""
        if pre.state_hash == post.state_hash:
            return StateDifferenceResult(is_identical=True)

        def _differs(a: Any, b: Any) -> bool:
            if a is None or b is None:
                return a is not b
            if hasattr(a, "shape") or hasattr(b, "shape"):
                import numpy as np

                if hasattr(a, "shape") and hasattr(b, "shape"):
                    return not np.array_equal(a, b)
                return True
            try:
                return bool(a != b)
            except Exception:
                return True

        changed_vars: dict[str, tuple[Any, Any]] = {}
        all_var_keys = set(pre.variables.keys()) | set(post.variables.keys())
        for k in all_var_keys:
            v_pre = pre.variables.get(k)
            v_post = post.variables.get(k)
            if _differs(v_pre, v_post):
                changed_vars[k] = (v_pre, v_post)

        changed_ents: dict[str, tuple[str, str]] = {}
        all_ent_keys = set(pre.entity_states.keys()) | set(post.entity_states.keys())
        for e in all_ent_keys:
            s_pre = pre.entity_states.get(e, "<absent>")
            s_post = post.entity_states.get(e, "<absent>")
            if s_pre != s_post:
                changed_ents[e] = (s_pre, s_post)

        # Position changes (W046) from variables or entity states
        changed_positions: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        for k, (v_pre, v_post) in changed_vars.items():
            if (
                isinstance(v_pre, (tuple, list))
                and isinstance(v_post, (tuple, list))
                and len(v_pre) == 2
                and len(v_post) == 2
                and all(isinstance(x, (int, float)) for x in (*v_pre, *v_post))
            ):
                changed_positions[k] = (
                    (int(v_pre[0]), int(v_pre[1])),
                    (int(v_post[0]), int(v_post[1])),
                )

        # Structural changes (W047)
        structural_changes: list[str] = []
        added_entities = set(post.entity_states.keys()) - set(pre.entity_states.keys())
        removed_entities = set(pre.entity_states.keys()) - set(post.entity_states.keys())
        for a in added_entities:
            structural_changes.append(f"ENTITY_ADDED:{a}")
        for r in removed_entities:
            structural_changes.append(f"ENTITY_REMOVED:{r}")

        # Localized cell changes if grids are present
        changed_cells: list[tuple[int, int]] = []
        bbox: tuple[int, int, int, int] | None = None
        if "grid" in pre.variables and "grid" in post.variables:
            g_pre = pre.variables["grid"]
            g_post = post.variables["grid"]
            if hasattr(g_pre, "shape") and hasattr(g_post, "shape") and g_pre.shape == g_post.shape:
                import numpy as np

                diff_mask = g_pre != g_post
                coords = np.argwhere(diff_mask)
                changed_cells = [(int(r), int(c)) for r, c in coords]
                if changed_cells:
                    r_min = int(coords[:, 0].min())
                    r_max = int(coords[:, 0].max())
                    c_min = int(coords[:, 1].min())
                    c_max = int(coords[:, 1].max())
                    bbox = (r_min, c_min, r_max, c_max)

        return StateDifferenceResult(
            is_identical=False,
            changed_variables=changed_vars,
            changed_entities=changed_ents,
            changed_positions=changed_positions,
            changed_cells=changed_cells,
            bounding_box_of_change=bbox,
            structural_changes=structural_changes,
        )


@dataclass
class StateTransitionRecord:
    """W048: Formal representation of an action-conditioned state transition."""

    transition_id: str
    pre_state: WorldStateSnapshot
    post_state: WorldStateSnapshot
    action: str | None
    delta: StateDifferenceResult
    is_reversible: bool
    timestamp: float = field(default_factory=time.time)


class TransitionSequenceModel:
    """W049: Sequence modeling of transitions, trajectories, and forward reachability."""

    def __init__(self) -> None:
        self._transitions: list[StateTransitionRecord] = []
        self._graph: dict[str, dict[str, str]] = {}  # pre_hash -> action -> post_hash

    def record_transition(
        self,
        pre_state: WorldStateSnapshot,
        action: str | None,
        post_state: WorldStateSnapshot,
        is_reversible: bool = True,
    ) -> StateTransitionRecord:
        """Record and index a state transition in the sequential model."""
        import uuid

        delta = StateDifferenceComputer.compute_difference(pre_state, post_state)
        rec = StateTransitionRecord(
            transition_id=f"trans_{uuid.uuid4().hex[:8]}",
            pre_state=pre_state,
            post_state=post_state,
            action=action,
            delta=delta,
            is_reversible=is_reversible,
        )
        self._transitions.append(rec)
        action_key = action or "NOOP"
        if pre_state.state_hash not in self._graph:
            self._graph[pre_state.state_hash] = {}
        self._graph[pre_state.state_hash][action_key] = post_state.state_hash
        return rec

    def predict_next_state_hash(self, pre_state_hash: str, action: str) -> str | None:
        """Predict post-state hash under given action from empirical transition model."""
        return self._graph.get(pre_state_hash, {}).get(action)

    def get_trajectory(self) -> list[StateTransitionRecord]:
        """Return ordered list of recorded transitions."""
        return list(self._transitions)


@dataclass
class ReversibilityAnalysis:
    """W050: Rigorous reversibility analysis and information-loss quantification."""

    is_invertible: bool
    information_loss_bits: float
    inverse_action: str | None
    restoration_possible: bool
    reason: str


class ReversibilityEngine:
    """W050: Computes transition reversibility, entropy loss, and executes restoration trajectories."""

    DEFAULT_INVERSES: dict[str, str] = {
        "MOVE_UP": "MOVE_DOWN",
        "MOVE_DOWN": "MOVE_UP",
        "MOVE_LEFT": "MOVE_RIGHT",
        "MOVE_RIGHT": "MOVE_LEFT",
        "ROTATE_90_CW": "ROTATE_90_CCW",
        "ROTATE_90_CCW": "ROTATE_90_CW",
        "ROTATE_180": "ROTATE_180",
        "FLIP_H": "FLIP_H",
        "FLIP_V": "FLIP_V",
    }

    @classmethod
    def analyze_transition(
        cls,
        transition: StateTransitionRecord,
        custom_inverses: dict[str, str] | None = None,
    ) -> ReversibilityAnalysis:
        """W050: Analyzes whether an action/transition is mathematically and physically invertible."""
        inverses = dict(cls.DEFAULT_INVERSES)
        if custom_inverses:
            inverses.update(custom_inverses)

        action = transition.action or "NOOP"
        inv_action = inverses.get(action)

        # Check for destructive/irreversible structural changes
        has_destructive_loss = False
        info_loss_bits = 0.0

        for struct_chg in transition.delta.structural_changes:
            if struct_chg.startswith("ENTITY_REMOVED"):
                has_destructive_loss = True
                info_loss_bits += 8.0  # Lost entity identity & state entropy

        if transition.delta.changed_cells:
            # Overwriting cells: calculate Shannon entropy loss of overwriting
            n_cells = len(transition.delta.changed_cells)
            import math

            info_loss_bits += n_cells * math.log2(10.0)  # ARC 10-color discrete entropy

        if has_destructive_loss:
            return ReversibilityAnalysis(
                is_invertible=False,
                information_loss_bits=info_loss_bits,
                inverse_action=None,
                restoration_possible=False,
                reason="DESTRUCTIVE_ENTITY_REMOVAL",
            )

        if inv_action is not None and not has_destructive_loss:
            return ReversibilityAnalysis(
                is_invertible=True,
                information_loss_bits=0.0,
                inverse_action=inv_action,
                restoration_possible=True,
                reason="EXACT_INVERSE_ACTION_AVAILABLE",
            )

        return ReversibilityAnalysis(
            is_invertible=transition.is_reversible,
            information_loss_bits=info_loss_bits,
            inverse_action=None,
            restoration_possible=transition.is_reversible,
            reason="STATE_RESTORE_VIA_MEMORY_ROLLBACK"
            if transition.is_reversible
            else "IRREVERSIBLE",
        )

    @classmethod
    def restore_state(
        cls,
        target_state: WorldStateSnapshot,
        buffer: TemporalStateBuffer,
    ) -> bool:
        """W050: Executes state restoration rollback to a verified target state."""
        history = buffer.get_history()
        for idx, snap in enumerate(reversed(history)):
            if snap.state_hash == target_state.state_hash:
                buffer.rollback(steps=idx)
                return True
        return False
