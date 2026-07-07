"""Cognitive State Machine — the brainstem of HBLLM.

Implements a hierarchical finite state machine that controls the system's
operating mode, adaptive tick rates, and transition guards. Every other
autonomy subsystem reads this state to decide how aggressively to process.

State Hierarchy
───────────────
Operational
├── ACTIVE
│   ├── OBSERVING   — passively monitoring environment
│   ├── FOCUSED     — deep single-task concentration
│   ├── PLANNING    — constructing multi-step plans
│   └── EXECUTING   — running tool / skill actions
│
├── PASSIVE
│   ├── IDLE        — waiting for stimuli
│   ├── REFLECTING  — background memory consolidation
│   └── LOW_POWER   — battery / thermal conservation
│
└── TRANSITIONAL
    ├── INTERRUPTED  — paused current task for urgent event
    ├── RECOVERING   — repairing after failure / reconnect
    └── SLEEPING     — deep maintenance cycle
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ── State Enumerations ───────────────────────────────────────────────────────


class CognitiveStateCategory(StrEnum):
    """Top-level state grouping."""

    ACTIVE = "active"
    PASSIVE = "passive"
    TRANSITIONAL = "transitional"


class CognitiveState(StrEnum):
    """Concrete cognitive states the system can occupy."""

    # Active states
    OBSERVING = "observing"
    FOCUSED = "focused"
    PLANNING = "planning"
    EXECUTING = "executing"

    # Passive states
    IDLE = "idle"
    REFLECTING = "reflecting"
    LOW_POWER = "low_power"

    # Transitional states
    INTERRUPTED = "interrupted"
    RECOVERING = "recovering"
    SLEEPING = "sleeping"


# Category lookup
_STATE_CATEGORY: dict[CognitiveState, CognitiveStateCategory] = {
    CognitiveState.OBSERVING: CognitiveStateCategory.ACTIVE,
    CognitiveState.FOCUSED: CognitiveStateCategory.ACTIVE,
    CognitiveState.PLANNING: CognitiveStateCategory.ACTIVE,
    CognitiveState.EXECUTING: CognitiveStateCategory.ACTIVE,
    CognitiveState.IDLE: CognitiveStateCategory.PASSIVE,
    CognitiveState.REFLECTING: CognitiveStateCategory.PASSIVE,
    CognitiveState.LOW_POWER: CognitiveStateCategory.PASSIVE,
    CognitiveState.INTERRUPTED: CognitiveStateCategory.TRANSITIONAL,
    CognitiveState.RECOVERING: CognitiveStateCategory.TRANSITIONAL,
    CognitiveState.SLEEPING: CognitiveStateCategory.TRANSITIONAL,
}


# ── Adaptive Tick Configuration ──────────────────────────────────────────────


@dataclass(frozen=True)
class TickProfile:
    """Tick rate and resource limits for a cognitive state."""

    tick_interval_s: float
    allow_heavy_llm: bool = True
    allow_fast_router: bool = True
    max_concurrent_thoughts: int = 5
    interruption_threshold: float = 0.5  # 0.0 = interrupt always, 1.0 = never


# Default tick profiles — tuned for sovereign hardware (server / edge / mobile)
DEFAULT_TICK_PROFILES: dict[CognitiveState, TickProfile] = {
    # Active — tight loops
    CognitiveState.OBSERVING: TickProfile(
        tick_interval_s=5.0,
        allow_heavy_llm=False,
        allow_fast_router=True,
        max_concurrent_thoughts=3,
        interruption_threshold=0.3,
    ),
    CognitiveState.FOCUSED: TickProfile(
        tick_interval_s=1.0,
        allow_heavy_llm=True,
        allow_fast_router=True,
        max_concurrent_thoughts=1,
        interruption_threshold=0.8,  # Hard to interrupt
    ),
    CognitiveState.PLANNING: TickProfile(
        tick_interval_s=2.0,
        allow_heavy_llm=True,
        allow_fast_router=True,
        max_concurrent_thoughts=3,
        interruption_threshold=0.6,
    ),
    CognitiveState.EXECUTING: TickProfile(
        tick_interval_s=0.5,
        allow_heavy_llm=False,
        allow_fast_router=True,
        max_concurrent_thoughts=2,
        interruption_threshold=0.7,
    ),
    # Passive — conserve resources
    CognitiveState.IDLE: TickProfile(
        tick_interval_s=15.0,
        allow_heavy_llm=False,
        allow_fast_router=False,
        max_concurrent_thoughts=1,
        interruption_threshold=0.1,  # Easy to wake
    ),
    CognitiveState.REFLECTING: TickProfile(
        tick_interval_s=10.0,
        allow_heavy_llm=False,
        allow_fast_router=True,
        max_concurrent_thoughts=2,
        interruption_threshold=0.3,
    ),
    CognitiveState.LOW_POWER: TickProfile(
        tick_interval_s=30.0,
        allow_heavy_llm=False,
        allow_fast_router=False,
        max_concurrent_thoughts=1,
        interruption_threshold=0.2,
    ),
    # Transitional
    CognitiveState.INTERRUPTED: TickProfile(
        tick_interval_s=1.0,
        allow_heavy_llm=True,
        allow_fast_router=True,
        max_concurrent_thoughts=2,
        interruption_threshold=0.9,  # Already interrupted; don't cascade
    ),
    CognitiveState.RECOVERING: TickProfile(
        tick_interval_s=5.0,
        allow_heavy_llm=False,
        allow_fast_router=True,
        max_concurrent_thoughts=1,
        interruption_threshold=0.4,
    ),
    CognitiveState.SLEEPING: TickProfile(
        tick_interval_s=60.0,
        allow_heavy_llm=False,
        allow_fast_router=False,
        max_concurrent_thoughts=0,
        interruption_threshold=0.05,  # Wake on almost anything
    ),
}


# ── Transition Record ────────────────────────────────────────────────────────


@dataclass
class StateTransition:
    """Immutable record of a state transition for audit logging."""

    from_state: CognitiveState
    to_state: CognitiveState
    reason: str
    timestamp: float = field(default_factory=time.monotonic)
    metadata: dict[str, Any] = field(default_factory=dict)


# Type for transition guard callbacks
TransitionGuard = Callable[[CognitiveState, CognitiveState], bool]
TransitionHook = Callable[[StateTransition], None]


# ── Cognitive State Machine ──────────────────────────────────────────────────


class CognitiveStateMachine:
    """Hierarchical cognitive state machine.

    Controls:
      - Current operating mode of the cognitive system
      - Adaptive tick rate (via ``TickProfile``)
      - Transition guards (prevent invalid state jumps)
      - Transition hooks (notify listeners on state change)
      - Cognitive load tracking (affects attention scoring)

    Usage::

        csm = CognitiveStateMachine()
        csm.transition_to(CognitiveState.OBSERVING, reason="boot_complete")

        profile = csm.current_profile
        asyncio.sleep(profile.tick_interval_s)

        if csm.should_allow_interruption(priority_score=0.6):
            csm.transition_to(CognitiveState.INTERRUPTED, reason="user_input")
    """

    def __init__(
        self,
        initial_state: CognitiveState = CognitiveState.IDLE,
        tick_profiles: dict[CognitiveState, TickProfile] | None = None,
        history_limit: int = 100,
    ) -> None:
        self._state = initial_state
        self._profiles = tick_profiles or dict(DEFAULT_TICK_PROFILES)
        self._history_limit = history_limit

        # Timing
        self._state_entered_at: float = time.monotonic()
        self._boot_time: float = time.monotonic()

        # Cognitive load (0.0 = idle, 1.0 = saturated)
        self._cognitive_load: float = 0.0

        # Transition history (ring buffer)
        self._history: deque[StateTransition] = deque(maxlen=history_limit)

        # Guards: if any guard returns False, transition is blocked
        self._guards: list[TransitionGuard] = []

        # Hooks: called after every successful transition
        self._hooks: list[TransitionHook] = []

        # Saved state for interruption recovery
        self._saved_state: CognitiveState | None = None
        self._saved_reason: str = ""

        logger.info(
            "CognitiveStateMachine initialized in state=%s tick=%.1fs",
            self._state,
            self.current_profile.tick_interval_s,
        )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def state(self) -> CognitiveState:
        """Current cognitive state."""
        return self._state

    @property
    def category(self) -> CognitiveStateCategory:
        """Category of the current state (ACTIVE / PASSIVE / TRANSITIONAL)."""
        return _STATE_CATEGORY[self._state]

    @property
    def is_active(self) -> bool:
        return self.category == CognitiveStateCategory.ACTIVE

    @property
    def is_passive(self) -> bool:
        return self.category == CognitiveStateCategory.PASSIVE

    @property
    def is_transitional(self) -> bool:
        return self.category == CognitiveStateCategory.TRANSITIONAL

    @property
    def current_profile(self) -> TickProfile:
        """Tick profile for the current state."""
        return self._profiles.get(self._state, DEFAULT_TICK_PROFILES[CognitiveState.IDLE])

    @property
    def tick_interval(self) -> float:
        """Current adaptive tick interval in seconds."""
        return self.current_profile.tick_interval_s

    @property
    def state_duration(self) -> float:
        """Seconds spent in the current state."""
        return time.monotonic() - self._state_entered_at

    @property
    def uptime(self) -> float:
        """Total seconds since boot."""
        return time.monotonic() - self._boot_time

    @property
    def cognitive_load(self) -> float:
        """Current cognitive load (0.0–1.0)."""
        return self._cognitive_load

    @property
    def has_saved_state(self) -> bool:
        """Whether there is a saved state to resume from after interruption."""
        return self._saved_state is not None

    # ── Cognitive Load ────────────────────────────────────────────────

    def update_cognitive_load(self, load: float) -> None:
        """Update the current cognitive load (0.0–1.0).

        This value is used by the AttentionSystem to apply
        ``cognitive_load_penalty`` when scoring incoming events.
        """
        self._cognitive_load = max(0.0, min(1.0, load))

    # ── Transition Logic ──────────────────────────────────────────────

    def transition_to(
        self,
        new_state: CognitiveState,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> bool:
        """Attempt a state transition.

        Args:
            new_state: Target state.
            reason: Human-readable reason for the transition.
            metadata: Optional context to attach to the transition record.
            force: If True, bypass transition guards.

        Returns:
            True if the transition was accepted, False if blocked.
        """
        if new_state == self._state:
            return True  # No-op

        # Run guards (unless forced)
        if not force:
            for guard in self._guards:
                if not guard(self._state, new_state):
                    logger.debug(
                        "Transition %s → %s blocked by guard (reason=%s)",
                        self._state,
                        new_state,
                        reason,
                    )
                    return False

        # Save state for interruption recovery
        if new_state == CognitiveState.INTERRUPTED and self._saved_state is None:
            self._saved_state = self._state
            self._saved_reason = reason

        old_state = self._state
        self._state = new_state
        self._state_entered_at = time.monotonic()

        # Record transition
        record = StateTransition(
            from_state=old_state,
            to_state=new_state,
            reason=reason,
            metadata=metadata or {},
        )
        self._history.append(record)

        logger.info(
            "Cognitive state: %s → %s (reason=%s, tick=%.1fs)",
            old_state,
            new_state,
            reason,
            self.current_profile.tick_interval_s,
        )

        # Fire hooks
        for hook in self._hooks:
            try:
                hook(record)
            except Exception as exc:
                logger.warning("Error in transition hook: %s", exc, exc_info=True)

        return True

    def resume_from_interruption(self, reason: str = "interruption_resolved") -> bool:
        """Resume the state that was active before the interruption.

        Returns True if a saved state was restored, False otherwise.
        """
        if self._saved_state is None:
            logger.warning("No saved state to resume from")
            return False

        target = self._saved_state
        self._saved_state = None
        self._saved_reason = ""
        return self.transition_to(target, reason=reason)

    # ── Interruption Decisions ────────────────────────────────────────

    def should_allow_interruption(self, priority_score: float) -> bool:
        """Decide whether an incoming event should interrupt the current state.

        Args:
            priority_score: The attention priority score (0.0–1.0) of the
                incoming event, as computed by the AttentionSystem.

        Returns:
            True if the event's priority exceeds the current state's
            interruption threshold (adjusted by cognitive load).
        """
        profile = self.current_profile
        # Cognitive load raises the bar for interruption
        adjusted_threshold = profile.interruption_threshold + (self._cognitive_load * 0.15)
        adjusted_threshold = min(adjusted_threshold, 1.0)
        return priority_score > adjusted_threshold

    # ── Guards & Hooks ────────────────────────────────────────────────

    def add_guard(self, guard: TransitionGuard) -> None:
        """Register a transition guard.

        The guard receives ``(from_state, to_state)`` and returns True to
        allow the transition or False to block it.
        """
        self._guards.append(guard)

    def remove_guard(self, guard: TransitionGuard) -> None:
        """Remove a previously registered guard."""
        self._guards = [g for g in self._guards if g is not guard]

    def add_hook(self, hook: TransitionHook) -> None:
        """Register a post-transition hook.

        Called with a ``StateTransition`` record after each successful
        transition.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: TransitionHook) -> None:
        """Remove a previously registered hook."""
        self._hooks = [h for h in self._hooks if h is not hook]

    # ── Introspection ─────────────────────────────────────────────────

    def get_history(self, limit: int = 20) -> list[StateTransition]:
        """Return the most recent state transitions."""
        return list(self._history)[-limit:]

    def snapshot(self) -> dict[str, Any]:
        """Serializable snapshot of the state machine for telemetry."""
        profile = self.current_profile
        return {
            "state": self._state.value,
            "category": self.category.value,
            "tick_interval_s": profile.tick_interval_s,
            "allow_heavy_llm": profile.allow_heavy_llm,
            "allow_fast_router": profile.allow_fast_router,
            "interruption_threshold": profile.interruption_threshold,
            "max_concurrent_thoughts": profile.max_concurrent_thoughts,
            "cognitive_load": round(self._cognitive_load, 3),
            "state_duration_s": round(self.state_duration, 2),
            "uptime_s": round(self.uptime, 2),
            "has_saved_state": self.has_saved_state,
            "transition_count": len(self._history),
        }
