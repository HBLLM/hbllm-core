"""Per-attempt telemetry for the ARC-AGI benchmark runner (benchmark-side only).

Measurement-only instrumentation: the runner records what happened during an
attempt (latency, no-ops, contract violations, why it stopped) so the scorecard
and trace analysis report real data instead of post-hoc inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Stop reason vocabulary (why the runner stopped an attempt)
STOP_WIN = "win"
STOP_GAME_OVER = "game_over"
STOP_TIME_LIMIT = "time_limit"
STOP_LEVEL_STEP_LIMIT = "level_step_limit"
STOP_STEP_BUDGET = "step_budget"
STOP_RESET_FAILED = "reset_failed"
STOP_RUNNER_EXCEPTION = "runner_exception"

STOP_REASONS = (
    STOP_WIN,
    STOP_GAME_OVER,
    STOP_TIME_LIMIT,
    STOP_LEVEL_STEP_LIMIT,
    STOP_STEP_BUDGET,
    STOP_RESET_FAILED,
    STOP_RUNNER_EXCEPTION,
)


@dataclass
class AttemptTelemetry:
    """Measured record of a single game attempt.

    Attributes:
        attempt: Attempt index within the game (0 = first try).
        cycle_latencies_ms: Wall-clock ms of each full
            ``manager.execute_cognitive_step()`` (observe -> decide ->
            env.step -> update).
        decide_latencies_ms: ``agent._decide_latency_ms`` per step (engine
            ``step()`` only); only recorded when the agent refreshed it.
        noop_actions: Steps where the post-step grid equals the pre-step grid
            AND ``levels_completed`` did not change.
        max_stagnation_streak: Longest consecutive run of no-op steps.
        contract_violations: Delta of ``agent._contract_violations`` over this
            attempt.
        decision_exceptions: Delta of ``agent.decision_exceptions`` over this
            attempt.
        action_counts: Histogram of chosen action ids over the attempt.
        stop_reason: First stop reason recorded (see module constants).
        stop_step: Step index where the stop reason fired.
        stop_detail: Optional detail string about the stop.
        agent_yield_step: First step where ``agent.is_done()`` returned True
            while the runner was still running (measurement only).
        agent_yield_reason: Short label of which agent guard yielded.
    """

    attempt: int
    cycle_latencies_ms: list[float] = field(default_factory=list)
    decide_latencies_ms: list[float] = field(default_factory=list)
    noop_actions: int = 0
    max_stagnation_streak: int = 0
    contract_violations: int = 0
    decision_exceptions: int = 0
    action_counts: dict[int, int] = field(default_factory=dict)
    stop_reason: str | None = None
    stop_step: int | None = None
    stop_detail: str | None = None
    agent_yield_step: int | None = None
    agent_yield_reason: str | None = None
    _stagnation_streak: int = field(default=0, repr=False)

    def record_step(
        self,
        *,
        step: int,
        cycle_ms: float,
        decide_ms: float | None = None,
        action_id: int | None = None,
        noop: bool = False,
    ) -> None:
        """Record one executed env step.

        Args:
            step: Step index within the attempt loop.
            cycle_ms: Full ``execute_cognitive_step()`` wall time in ms.
            decide_ms: Engine decide latency in ms, if refreshed this step.
            action_id: Action id dispatched to the env, if known.
            noop: True when the step produced no grid change and no level
                progress.
        """
        self.cycle_latencies_ms.append(cycle_ms)
        if decide_ms is not None:
            self.decide_latencies_ms.append(decide_ms)
        if action_id is not None:
            self.action_counts[action_id] = self.action_counts.get(action_id, 0) + 1
        if noop:
            self.noop_actions += 1
            self._stagnation_streak += 1
            if self._stagnation_streak > self.max_stagnation_streak:
                self.max_stagnation_streak = self._stagnation_streak
        else:
            self._stagnation_streak = 0

    def stop(self, reason: str, step: int, detail: str | None = None) -> None:
        """Record the attempt stop reason. First write wins.

        Args:
            reason: One of the module ``STOP_*`` constants.
            step: Step index at which the stop fired.
            detail: Optional human-readable detail.
        """
        if self.stop_reason is None:
            self.stop_reason = reason
            self.stop_step = step
            self.stop_detail = detail
