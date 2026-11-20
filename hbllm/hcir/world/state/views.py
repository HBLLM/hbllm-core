"""ReadOnlyEngineView, FacultyWriteJournal, and JournaledEngineView."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from hbllm.hcir.world.cortical_stage import WorkingMemoryDelta
from hbllm.hcir.world.state.models import apply_wm_field_delta

logger = logging.getLogger(__name__)


class ReadOnlyEngineView:
    """Read-through facade handed to faculties instead of the engine (WP1).

    Attribute reads and method calls pass through to the engine; writes raise.
    Kernel-service mutation methods are whitelisted via ``_ALLOWED_MUTATORS``
    (kept minimal: engine-owned bookkeeping the faculty can't express as delta).
    """

    _ALLOWED_MUTATORS: frozenset[str] = frozenset(
        {
            "transition_to_phase",  # engine-owned phase ledger append
            "register_action_space",  # action-space schema registry (idempotent)
        }
    )

    def __init__(self, engine: Any) -> None:
        object.__setattr__(self, "_wrapped_engine", engine)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_engine, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._ALLOWED_MUTATORS or name.startswith("_"):
            object.__setattr__(self._wrapped_engine, name, value)
            return
        raise AttributeError("faculties emit intent via StateDelta")


# Faculty-owned sub-models: pass-through as-is. They own their internal state
# and their methods self-mutate — they are NOT engine fields to capture.
_OWNED_SUBSYSTEMS: frozenset[str] = frozenset(
    {
        "symbolic_theory",
        "hazard_tracker",
        "episodic_cortex",
        "working_memory",
        "object_planner",
        "deliberation_engine",
        "surprise_engine",
        "budget_governor",
        "services",
        "state",
        "prev_grid_cache",
        "habenular_ior",
    }
)


class FacultyWriteJournal:
    """Write-through journal: before-images for undo on rejected commit (WP1).

    Faculties write through to the engine immediately (behaviour-exact); the
    journal records first-touch before-images so a rejected commit can revert
    the engine to its pre-faculty state.
    """

    def __init__(self, author: str) -> None:
        self.author = author
        self.before: dict[str, Any] = {}
        self.touched: set[str] = set()
        self.deltas: list[WorkingMemoryDelta] = []


_ABSENT = object()


class _JournaledStateProxy:
    """Write-through proxy for EngineState fields."""

    def __init__(self, view: JournaledEngineView) -> None:
        object.__setattr__(self, "_view", view)
        # Names whose first-touch before-image is already in the journal.
        # ``journal.before``/``journal.touched`` writes are paired at every
        # site, so once a key is journaled repeat reads need no bookkeeping.
        object.__setattr__(self, "_seen", set())

    def __getattr__(self, name: str) -> Any:
        view = object.__getattribute__(self, "_view")
        wrapped = object.__getattribute__(view, "_wrapped_engine")
        val = getattr(wrapped.state, name)
        seen = object.__getattribute__(self, "_seen")
        if name not in seen:
            seen.add(name)
            journal = object.__getattribute__(view, "_journal")
            key = f"state.{name}"
            if key not in journal.before:
                try:
                    journal.before[key] = (
                        val.copy() if isinstance(val, (dict, set, list, deque)) else val
                    )
                except Exception:
                    journal.before[key] = _ABSENT
            journal.touched.add(key)
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        view = object.__getattribute__(self, "_view")
        wrapped = object.__getattribute__(view, "_wrapped_engine")
        view._record_touch(f"state.{name}", getattr(wrapped.state, name, _ABSENT))
        setattr(wrapped.state, name, value)


class JournaledEngineView:
    """Write-through faculty view: writes land on the engine immediately,
    journaled for undo on rejected commit (WP1 design v2)."""

    def __init__(self, engine: Any, journal: FacultyWriteJournal) -> None:
        object.__setattr__(self, "_wrapped_engine", engine)
        object.__setattr__(self, "_journal", journal)
        object.__setattr__(self, "_state_proxy", _JournaledStateProxy(self))
        # name -> routing memo: 0=state proxy, 1=owned subsystem, 2=plain field.
        # The route depends only on the attribute name, so it is safe to cache
        # for the lifetime of this view (values themselves are never cached).
        object.__setattr__(self, "_routes", {})

    def _record_touch(self, key: str, current: Any) -> None:
        journal = object.__getattribute__(self, "_journal")
        if key not in journal.before:
            try:
                journal.before[key] = (
                    current.copy()
                    if isinstance(current, (dict, set, list)) or type(current).__name__ == "deque"
                    else current
                )
            except Exception:
                journal.before[key] = _ABSENT
        journal.touched.add(key)

    def apply_wm_delta(self, delta: WorkingMemoryDelta) -> None:
        """Apply a declarative WorkingMemoryDelta, recording it in journal for undo and verification."""
        journal = object.__getattribute__(self, "_journal")
        journal.deltas.append(delta)
        eng = object.__getattribute__(self, "_wrapped_engine")
        field = delta.field
        val = delta.value
        op = delta.op

        if field == "inhibit_action":
            action, duration = val if isinstance(val, (tuple, list)) else (val, 4)
            self._record_touch(
                "working_memory.inhibited_actions",
                dict(getattr(eng.working_memory, "inhibited_actions", {})),
            )
            eng.working_memory.inhibit_action(action, duration)
            return

        apply_wm_field_delta(
            eng,
            eng.working_memory,
            eng.state,
            field,
            val,
            op,
            before_recorder=lambda key, before_val: self._record_touch(key, before_val),
        )

    def __getattr__(self, name: str) -> Any:
        routes = object.__getattribute__(self, "_routes")
        route = routes.get(name)
        if route is None:
            if name == "state":
                route = 0
            elif name in _OWNED_SUBSYSTEMS:
                route = 1
            else:
                route = 2
            routes[name] = route
        if route == 0:
            return object.__getattribute__(self, "_state_proxy")
        val = getattr(object.__getattribute__(self, "_wrapped_engine"), name)
        if route == 1:
            if name in _OWNED_SUBSYSTEMS and name != "prev_grid_cache":
                object.__setattr__(self, name, val)
            return val
        if not isinstance(val, (dict, set, list, deque)):
            return val
        journal = object.__getattribute__(self, "_journal")
        if name in journal.before:
            # First touch already captured the before-image; ``before``/``touched``
            # are written in pairs at every site, so this key is also already in
            # ``touched``. Demote to passthrough: repeat reads need no journal work.
            routes[name] = 1
            return val
        try:
            journal.before[name] = val.copy()
        except Exception:
            journal.before[name] = _ABSENT
        journal.touched.add(name)
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        eng = object.__getattribute__(self, "_wrapped_engine")
        self._record_touch(name, getattr(eng, name, _ABSENT))
        setattr(eng, name, value)
