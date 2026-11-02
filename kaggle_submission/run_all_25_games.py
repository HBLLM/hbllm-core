"""Run HBLLM Cognitive USB Agent against ALL 25 ARC-AGI-3 games across ALL levels.

Evaluates against official Arcade environments installed in .venv.
Writes continuous live progress to `all_25_games_scorecard.json`.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure core repository is on sys.path
_core_dir = Path(__file__).resolve().parent.parent
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))

from arc_agi import Arcade

try:
    import plugins.arc_agi_adapter  # noqa: F401
except Exception:
    pass

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverCapability,
    DriverFeedback,
    DriverInput,
    DriverModality,
    DriverStreamType,
    SynapticDeviceDescriptor,
)
from hbllm.drivers.manager import DriverManager
from plugins.arc_agi_adapter.telemetry import (
    STOP_GAME_OVER,
    STOP_LEVEL_STEP_LIMIT,
    STOP_RESET_FAILED,
    STOP_RUNNER_EXCEPTION,
    STOP_STEP_BUDGET,
    STOP_TIME_LIMIT,
    STOP_WIN,
    AttemptTelemetry,
)

try:
    from kaggle_submission.constants import ARC_DEFAULT_ACTION_SCHEMAS
except ImportError:
    from constants import ARC_DEFAULT_ACTION_SCHEMAS  # type: ignore[no-redef]

# Import competitive agent
try:
    from kaggle_submission.my_agent import MyAgent
except ImportError:
    from my_agent import MyAgent  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("all_25_benchmark")


import numpy as np


def _extract_grid(raw: Any) -> np.ndarray | None:
    """Safely extract 2D integer grid from bare arrays, temporal lists, or 3D buffers."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        raw = raw[-1]
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3 and len(raw) > 0:
            raw = raw[-1]
        return np.asarray(raw, dtype=int)
    try:
        arr = np.asarray(raw, dtype=int)
        if arr.ndim == 3 and len(arr) > 0:
            arr = arr[-1]
        return arr
    except Exception:
        return None


# Width of the HUD border to exclude from effective-change no-op detection.
# Some games render a step-counter bar on the outermost grid edge; those cells
# change every step and would defeat naive grid-equality no-op detection.
_HUD_BORDER = 1

# Sentinel for changed-cell counts when grids cannot be compared
# (missing grid, or pre/post shape mismatch — treated as changed, never a no-op).
_CHANGED_UNKNOWN = -1


def _p95_ms(latencies_ms: list[float]) -> float | None:
    if not latencies_ms:
        return None
    if len(latencies_ms) == 1:
        return round(latencies_ms[0], 2)
    return round(statistics.quantiles(latencies_ms, n=20)[18], 2)


def _safe_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _write_runner_event(fp: Any, rec: dict[str, Any]) -> None:
    fp.write(json.dumps(rec, default=str) + "\n")
    fp.flush()


# ── 1. Cognitive USB Game Driver ──────────────────────────────────────────────
class ArcAgiConsoleDriver(BaseDriver):
    """Cognitive USB Peripheral Driver wrapping ARC-AGI-3 Arcade environments."""

    def __init__(self, name: str = "arc_console", env: Any = None, game_id: str = "") -> None:
        desc = SynapticDeviceDescriptor(
            device_id=name,
            vendor_id="arc_prize",
            device_type="bidirectional",
            modalities=[DriverModality.GRID_2D, DriverModality.STRUCTURED],
            stream_type=DriverStreamType.POLL,
            action_schema=[dict(s) for s in ARC_DEFAULT_ACTION_SCHEMAS],
        )
        super().__init__(
            name=name,
            capabilities={
                DriverCapability.DISCRETE_ACTIONS,
                DriverCapability.STEP_BASED_EXECUTION,
                DriverCapability.SPATIAL_2D,
            },
            descriptor=desc,
        )
        self.env = env
        self.game_id = game_id
        self.current_frame = None

    def connect(self, target: Any) -> bool:
        self.env = target
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False
        self.env = None

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        avail = inputs.metadata.get("available_actions", [1, 2, 3, 4, 5, 6])
        schema_map = {item["action_id"]: item for item in self.descriptor.action_schema}
        actions = []
        for aid in avail:
            item = schema_map.get(aid, {})
            name = item.get("name", str(aid))
            params = item.get("parameters", {})
            actions.append(DriverAction(action_id=aid, semantic_intent=name, parameters=params))
        return actions

    def set_frame(self, frame: Any) -> None:
        if (
            frame is not None
            and self.game_id
            and (not hasattr(frame, "game_id") or getattr(frame, "game_id", None) is None)
        ):
            try:
                frame.game_id = self.game_id
            except Exception:
                pass
        self.current_frame = frame

    def get_inputs(self) -> DriverInput:
        grid = getattr(self.current_frame, "frame", None)
        avail = getattr(self.current_frame, "available_actions", [1, 2, 3, 4, 5, 6])
        metadata: dict[str, Any] = {
            "frame_obj": self.current_frame,
            "game_id": self.game_id,
            "available_actions": list(avail),
            "levels_completed": getattr(self.current_frame, "levels_completed", 0),
            "win_levels": getattr(self.current_frame, "win_levels", 1),
            "state": getattr(
                getattr(self.current_frame, "state", None),
                "name",
                str(getattr(self.current_frame, "state", "")),
            ),
        }
        game_obj = getattr(self.env, "_game", None) or getattr(self.env, "game", None)
        cur_lvl = (
            getattr(game_obj, "current_level", None)
            if game_obj
            else getattr(self.env, "current_level", None)
        )
        if cur_lvl is not None and hasattr(cur_lvl, "get_data"):
            try:
                sc = cur_lvl.get_data("StepCounter")
                if sc is not None:
                    metadata["StepCounter"] = int(sc)
            except Exception:
                pass

        game_actions = getattr(game_obj, "actions", None)
        valid_targets: list[dict[str, Any]] = []
        if game_actions and isinstance(game_actions, (list, tuple)):
            for act_in in game_actions:
                d = getattr(act_in, "data", None)
                if isinstance(d, dict) and d:
                    valid_targets.append(d)

        schemas = [dict(s) for s in self.descriptor.action_schema]
        if valid_targets:
            for s in schemas:
                if s.get("action_id") == 6:
                    s["valid_targets"] = valid_targets
        metadata["action_schemas"] = schemas

        return DriverInput(
            raw_data=grid,
            source_id=self.name,
            modality=DriverModality.GRID_2D,
            metadata=metadata,
        )

    def send_output(self, action: DriverAction) -> Any:
        act_id = action.action_id
        params = action.parameters or {}
        data = params if params else None
        next_frame = self.env.step(act_id, data=data)
        if (
            next_frame is not None
            and self.game_id
            and (not hasattr(next_frame, "game_id") or getattr(next_frame, "game_id", None) is None)
        ):
            try:
                next_frame.game_id = self.game_id
            except Exception:
                pass
        self.current_frame = next_frame
        return next_frame

    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        if raw_result is None:
            return DriverFeedback(success=False, terminated=True)

        state_str = getattr(
            getattr(raw_result, "state", None), "name", str(getattr(raw_result, "state", ""))
        )
        lvl = getattr(raw_result, "levels_completed", 0)
        win_levels = getattr(raw_result, "win_levels", 1)

        is_win = state_str == "WIN" or lvl >= win_levels
        is_done = (
            is_win or state_str == "GAME_OVER" or bool(getattr(raw_result, "terminated", False))
        )
        reward = 1.0 if is_win else (0.5 if lvl > 0 else 0.0)

        avail_actions = list(getattr(raw_result, "available_actions", [1, 2, 3, 4, 5, 6]))
        info_dict: dict[str, Any] = {
            "levels_completed": lvl,
            "state": state_str,
            "win_levels": win_levels,
            "available_actions": avail_actions,
            "terminated": is_done,
        }
        game_obj = getattr(self.env, "_game", None) or getattr(self.env, "game", None)
        cur_lvl = (
            getattr(game_obj, "current_level", None)
            if game_obj
            else getattr(self.env, "current_level", None)
        )
        if cur_lvl is not None and hasattr(cur_lvl, "get_data"):
            try:
                sc = cur_lvl.get_data("StepCounter")
                if sc is not None:
                    info_dict["StepCounter"] = int(sc)
            except Exception:
                pass

        return DriverFeedback(
            success=True,
            reward=reward,
            terminated=is_done,
            causal_delta=_extract_grid(getattr(raw_result, "frame", None)),
            info=info_dict,
        )


# ── 2. Cognitive Agent Adapter ────────────────────────────────────────────────
class CognitiveAgentAdapter:
    """Wraps MyAgent into the canonical CognitiveBlackbox interface."""

    def __init__(self, agent: MyAgent) -> None:
        self.agent = agent
        self.history: list[Any] = []
        self.last_frame: Any = None
        self.last_action: DriverAction | None = None

    def reset_episode(
        self,
        retain_dynamics: bool = False,
        is_new_level: bool = False,
        level: int | None = None,
    ) -> None:
        self.history.clear()
        self.last_frame = None
        try:
            self.agent.reset_episode(
                retain_dynamics=retain_dynamics, is_new_level=is_new_level, level=level
            )
        except TypeError:
            try:
                self.agent.reset_episode(retain_dynamics=retain_dynamics, is_new_level=is_new_level)
            except TypeError:
                self.agent.reset_episode(retain_dynamics=retain_dynamics)

    def observe(self, driver_input: DriverInput, perception_data: Any = None) -> None:
        self.last_frame = driver_input.metadata.get("frame_obj")
        sc = driver_input.metadata.get("StepCounter")
        if sc is not None and hasattr(self.agent, "internal_engine") and self.agent.internal_engine:
            self.agent.internal_engine.set_environmental_step_limit(int(sc))
        schemas = driver_input.metadata.get("action_schemas")
        if schemas:
            self.agent.action_schemas = schemas

    def decide(self, available_actions: list[DriverAction], source_id: str = "") -> DriverAction:
        game_act = self.agent.choose_action(self.history, self.last_frame)
        self.history.append(self.last_frame)

        # Extract integer action id
        if hasattr(game_act, "value"):
            act_id = int(game_act.value)
        else:
            act_id = int(game_act)

        # Extract click parameters if applicable
        params: dict[str, Any] = {}
        data = getattr(game_act, "data", None)
        if data is None and hasattr(game_act, "is_complex") and game_act.is_complex():
            if hasattr(game_act, "action_data") and game_act.action_data is not None:
                ad = game_act.action_data
                data = {"x": getattr(ad, "x", 0), "y": getattr(ad, "y", 0)}
            else:
                data = {"x": 0, "y": 0}

        if data is not None and isinstance(data, dict):
            params = data

        self.last_action = DriverAction(action_id=act_id, parameters=params)
        return self.last_action

    def update(self, action: DriverAction, feedback: DriverFeedback, source_id: str = "") -> None:
        if hasattr(self.agent, "internal_engine") and self.agent.internal_engine is not None:
            engine = self.agent.internal_engine
            if "StepCounter" in feedback.info:
                engine.set_environmental_step_limit(int(feedback.info["StepCounter"]))
            lvl = int(feedback.info.get("levels_completed", 0))
            if lvl > getattr(self, "_last_lvl", 0):
                self.history.clear()
                self.last_frame = None
            self._last_lvl = lvl


DEFAULT_EXECUTIVE_DIRECTIVES = [
    "1. check if exit is open.",
    "2. if open go through it.",
    "3. if not explore other objects.",
    "4. if you know what they already do and those can unlock the door use it.",
    "5. if not play with it and see what is the outcome.",
]


# ── 3. Evaluation Harness for All 25 Games & All Levels ───────────────────────
def run_all_25_games(
    selected_games: list[str] | None = None,
    disable_archetypes: bool = True,
    scorecard_file: str = "all_25_games_scorecard.json",
    instructions: list[str] | str | None = None,
    max_steps_per_game: int = 1000,
    max_level_steps: int | None = None,
    retries: int = 0,
    resume: bool = True,
    per_game_time_limit: float = 600.0,
    trace_dir: str | None = None,
    seed: int | None = None,
) -> None:
    if instructions is None:
        instructions = DEFAULT_EXECUTIVE_DIRECTIVES

    arcade = Arcade()
    raw_environments = arcade.get_environments()

    # Deduplicate environments by unique game_id preserving order
    seen_ids = set()
    environments = []
    for e in raw_environments:
        if e.game_id not in seen_ids:
            seen_ids.add(e.game_id)
            environments.append(e)

    if selected_games:
        selected_lower = [g.strip().lower() for g in selected_games if g.strip()]
        environments = [
            e
            for e in environments
            if e.game_id.lower() in selected_lower
            or getattr(e, "title", "").lower() in selected_lower
        ]
        if not environments:
            print(f"⚠️ No environments matched filters: {selected_games}")
            return

    print(f"\n{'=' * 75}")
    print(f"🚀 HBLLM ARC-AGI 3 BENCHMARK: {len(environments)} GAMES x ALL LEVELS")
    print("   Architecture: Cognitive USB Driver + Autonomic Epistemic Brain")
    print("   Mode: Pure AutonomousEpistemicEngine (Domain-General Neuro-Symbolic Core)")
    limit_desc = (
        f"No Per-Level Limit (Full Runway: {max_steps_per_game} steps/game)"
        if max_level_steps is None
        else f"Per-Level Limit: {max_level_steps} steps (Max Game Steps: {max_steps_per_game})"
    )
    print(f"   Step Limit: {limit_desc}")
    print(f"   Scorecard Target: {scorecard_file}")
    if instructions:
        inst_list = [instructions] if isinstance(instructions, str) else list(instructions)
        print(f"   Loaded Executive Directives ({len(inst_list)}):")
        for d in inst_list:
            print(f"     • {d}")
    print(f"{'=' * 75}\n")

    scorecard_path = Path(scorecard_file)
    results: dict[str, Any] = {}
    run_best_telemetries: dict[str, AttemptTelemetry] = {}

    total_levels_passed = 0
    total_levels_possible = 0
    total_steps_taken = 0
    total_time_start = time.time()

    if resume and scorecard_path.exists():
        try:
            saved_data = json.loads(scorecard_path.read_text())
            results = saved_data.get("games", {})
            total_levels_passed = int(saved_data.get("total_levels_passed", 0))
            total_levels_possible = int(saved_data.get("total_levels_possible", 0))
            print(
                f"   ⏩ Resuming benchmark: {len(results)} games already completed in existing scorecard.\n"
            )
        except Exception:
            pass
    elif not resume and scorecard_path.exists():
        backup_path = scorecard_path.with_name(
            f"{scorecard_path.stem}_backup_{int(time.time())}.json"
        )
        try:
            backup_path.write_text(scorecard_path.read_text())
            print(f"   💾 Backed up previous scorecard to: {backup_path}\n")
        except Exception:
            pass

    for idx, env_meta in enumerate(environments, 1):
        gid = env_meta.game_id
        title = getattr(env_meta, "title", gid)
        baseline = getattr(env_meta, "baseline_actions", None) or [100]

        if gid in results:
            print(
                f"[{idx:2d}/{len(environments)}] ⏩ Skipping already completed game {title} ({gid}) ({results[gid].get('levels_passed', 0)}/{results[gid].get('target_levels', len(baseline))} levels)"
            )
            continue

        print(f"\n[{idx:2d}/{len(environments)}] 🎮 Game: {title} ({gid})")
        print(f"    Baseline Actions: {baseline} | Tags: {getattr(env_meta, 'tags', [])}")

        best_completed = 0
        best_target = len(baseline) or 1
        best_is_win = False
        best_game_steps = 0
        best_elapsed = 0.0
        best_telemetry: AttemptTelemetry | None = None
        attempts_taken = 0
        total_game_steps_all = 0

        # Per-game trace isolation: each game gets <trace_dir>/<gid>/ containing
        # the agent's trace.jsonl plus the runner's runner.jsonl.
        game_trace_dir: Path | None = None
        runner_fp = None
        if trace_dir:
            game_trace_dir = Path(trace_dir) / _safe_dirname(gid)
            game_trace_dir.mkdir(parents=True, exist_ok=True)
            runner_fp = open(game_trace_dir / "runner.jsonl", "a")

        # Instantiate fresh agent & Cognitive USB Driver for clean inter-game isolation
        agent = MyAgent(
            instructions=instructions,
            seed=seed,
            trace_dir=str(game_trace_dir) if game_trace_dir is not None else None,
        )
        cognitive_engine = CognitiveAgentAdapter(agent)

        # Phase C: Seed engine with cross-game prior knowledge
        try:
            epistemic_engine = getattr(agent, "internal_engine", getattr(agent, "engine", None))
            if epistemic_engine is not None and hasattr(epistemic_engine, "knowledge_transfer"):
                epistemic_engine.knowledge_transfer.on_game_start(
                    epistemic_engine,
                    set(range(8)),  # Standard 8-action ARC space
                )
                # Phase C: Clear per-game episodic macros for tenant isolation
                if hasattr(epistemic_engine, "episodic_cortex"):
                    epistemic_engine.episodic_cortex.reset_consolidated_macros()
        except Exception as e:
            logger.debug("Phase C on_game_start error: %s", e)

        for attempt in range(retries + 1):
            attempts_taken += 1
            if attempt > 0:
                print(f"\n    🔁 [Retry {attempt}/{retries}] for failed game {title} ({gid})...")

            env = arcade.make(gid)
            if env is None:
                print(f"    ⚠️ Could not make game environment for {gid}")
                continue

            agent.arc_env = env

            driver = ArcAgiConsoleDriver(name="arc_console", env=env, game_id=gid)
            manager = DriverManager()
            manager.set_cognitive_engine(cognitive_engine)

            manager.register(driver)
            manager.bind("arc_console", target=env)

            frame = env.reset()
            if not hasattr(frame, "game_id") or getattr(frame, "game_id", None) is None:
                try:
                    frame.game_id = gid
                except Exception:
                    pass

            base_gid = gid.split("-")[0].strip()
            agent.game_id = base_gid
            agent.current_game_id = base_gid
            driver.set_frame(frame)
            cognitive_engine.reset_episode(
                retain_dynamics=(attempt > 0), is_new_level=False, level=0
            )

            # Evaluate ALL levels (no capping)
            target_levels = int(getattr(frame, "win_levels", None) or len(baseline) or 1)
            best_target = target_levels

            # Step budget for this attempt
            total_game_budget = max_steps_per_game if max_steps_per_game > 0 else 10000

            start_time = time.time()
            completed_prev = 0
            game_steps = 0
            current_level_steps = 0
            telemetry = AttemptTelemetry(attempt=attempt)
            violations_prev = int(getattr(agent, "_contract_violations", 0) or 0)
            exceptions_prev = int(getattr(agent, "decision_exceptions", 0) or 0)

            for step in range(1, total_game_budget + 1):
                lvl_before = int(getattr(driver.current_frame, "levels_completed", 0) or 0)
                pre_grid = _extract_grid(getattr(driver.current_frame, "frame", None))
                agent._decide_latency_ms = None
                _t_step = time.perf_counter()
                try:
                    feedback = manager.execute_cognitive_step()
                except Exception as e:
                    logger.exception(
                        "execute_cognitive_step failed (game=%s attempt=%d step=%d)",
                        gid,
                        attempt,
                        step,
                    )
                    telemetry.stop(STOP_RUNNER_EXCEPTION, step, repr(e))
                    break
                cycle_ms = (time.perf_counter() - _t_step) * 1000.0
                game_steps += 1
                current_level_steps += 1
                info = feedback.info
                lvl = int(info.get("levels_completed", 0))
                state_str = str(info.get("state", "RUNNING"))

                # Measurement: grid-diff no-op detection + decide latency + counters.
                # A step is a no-op when the grid interior (excluding the HUD
                # border) is unchanged AND levels_completed did not advance.
                post_grid = _extract_grid(feedback.causal_delta)
                grid_changed: bool | None = None
                changed_cells = _CHANGED_UNKNOWN
                effective_changed_cells = _CHANGED_UNKNOWN
                if pre_grid is not None and post_grid is not None:
                    if pre_grid.shape == post_grid.shape:
                        diff = np.asarray(pre_grid != post_grid)
                        changed_cells = int(diff.sum())
                        grid_changed = changed_cells > 0
                        interior = diff[_HUD_BORDER:-_HUD_BORDER, _HUD_BORDER:-_HUD_BORDER]
                        # Degenerate grids (nothing but border) fall back to the
                        # raw count so changes are never masked.
                        effective_changed_cells = (
                            int(interior.sum()) if interior.size > 0 else changed_cells
                        )
                    else:
                        grid_changed = True  # shape mismatch — treat as changed
                is_noop = (effective_changed_cells == 0) and (lvl == lvl_before)
                decide_ms = getattr(agent, "_decide_latency_ms", None)
                violations_now = int(getattr(agent, "_contract_violations", 0) or 0)
                exceptions_now = int(getattr(agent, "decision_exceptions", 0) or 0)
                d_violations = violations_now - violations_prev
                d_exceptions = exceptions_now - exceptions_prev
                violations_prev, exceptions_prev = violations_now, exceptions_now
                telemetry.contract_violations += d_violations
                telemetry.decision_exceptions += d_exceptions
                last_act = getattr(cognitive_engine, "last_action", None)
                action_id = int(last_act.action_id) if last_act is not None else None
                action_data = (
                    dict(last_act.parameters)
                    if last_act is not None and last_act.parameters
                    else None
                )
                telemetry.record_step(
                    step=step,
                    cycle_ms=cycle_ms,
                    decide_ms=decide_ms,
                    action_id=action_id,
                    noop=is_noop,
                )
                if runner_fp is not None:
                    _write_runner_event(
                        runner_fp,
                        {
                            "event": "step",
                            "game": gid,
                            "attempt": attempt,
                            "step": step,
                            "level_before": lvl_before,
                            "level_after": lvl,
                            "action_id": action_id,
                            "action_data": action_data,
                            "cycle_ms": round(cycle_ms, 3),
                            "decide_ms": round(decide_ms, 3)
                            if isinstance(decide_ms, (int, float))
                            else None,
                            "grid_changed": grid_changed,
                            "changed_cells": changed_cells,
                            "effective_changed_cells": effective_changed_cells,
                            "noop": is_noop,
                            "state": state_str,
                            "violation": d_violations > 0,
                            "exception": d_exceptions > 0,
                        },
                    )

                # Measurement: observe when the agent's own guards would yield.
                # is_done() has side effects (_is_marked_done) but the runner
                # deliberately ignores them for stopping.
                if telemetry.agent_yield_step is None:
                    try:
                        if agent.is_done([], driver.current_frame):
                            telemetry.agent_yield_step = step
                            telemetry.agent_yield_reason = getattr(agent, "_yield_reason", None)
                    except Exception:
                        pass

                if lvl > completed_prev:
                    print(
                        f"    🎉 Level {lvl}/{target_levels} PASSED at step {step} (in {current_level_steps} steps)!"
                    )
                    completed_prev = lvl
                    current_level_steps = 0
                    # Note: cognitive reset is handled internally by choose_action
                    # on level transition detection. Only clear adapter history.
                    if hasattr(cognitive_engine, "history"):
                        cognitive_engine.history.clear()
                elif lvl < completed_prev:
                    completed_prev = lvl
                    current_level_steps = 0

                if state_str == "WIN" or lvl >= target_levels:
                    telemetry.stop(
                        STOP_WIN, step, f"state={state_str} levels={lvl}/{target_levels}"
                    )
                    break

                # Per-game time guard in benchmark: prevent expensive games from starving the rest
                game_elapsed = time.time() - start_time
                if game_elapsed >= per_game_time_limit:
                    print(
                        f"    ⏱️ Per-game time limit ({per_game_time_limit:.0f}s) exceeded at step {step} "
                        f"({completed_prev}/{target_levels} levels). Yielding game."
                    )
                    telemetry.stop(
                        STOP_TIME_LIMIT,
                        step,
                        f"elapsed_s={game_elapsed:.2f} limit_s={per_game_time_limit}",
                    )
                    break

                # Optional per-level step limit if explicitly configured by user
                if max_level_steps is not None and current_level_steps >= max_level_steps:
                    print(
                        f"    ⏱️ Level {lvl + 1}/{target_levels} reached per-level limit "
                        f"({current_level_steps} >= {max_level_steps} steps). Concluding attempt."
                    )
                    telemetry.stop(
                        STOP_LEVEL_STEP_LIMIT,
                        step,
                        f"level_steps={current_level_steps} limit={max_level_steps}",
                    )
                    break
                elif feedback.terminated or state_str == "GAME_OVER":
                    print(f"    ⚠️ Level retry / death at step {step}, retaining dynamics...")
                    cognitive_engine.reset_episode(
                        retain_dynamics=True, is_new_level=False, level=lvl
                    )
                    try:
                        reset_frame = env.step(0)
                    except Exception:
                        try:
                            reset_frame = env.reset()
                        except Exception:
                            telemetry.stop(
                                STOP_RESET_FAILED,
                                step,
                                "env.step(0) and env.reset() both raised",
                            )
                            break
                    reset_state = getattr(
                        getattr(reset_frame, "state", None),
                        "name",
                        str(getattr(reset_frame, "state", "")),
                    )
                    if reset_state == "GAME_OVER":
                        print(f"    ❌ GAME_OVER terminal at step {step}")
                        telemetry.stop(STOP_GAME_OVER, step, f"reset_state={reset_state}")
                        break
                    driver.set_frame(reset_frame)
            else:
                telemetry.stop(STOP_STEP_BUDGET, game_steps, f"budget={total_game_budget}")

            elapsed = time.time() - start_time
            total_game_steps_all += game_steps

            if runner_fp is not None:
                _write_runner_event(
                    runner_fp,
                    {
                        "event": "stop",
                        "game": gid,
                        "attempt": attempt,
                        "stop_reason": telemetry.stop_reason,
                        "stop_step": telemetry.stop_step,
                        "stop_detail": telemetry.stop_detail,
                        "agent_yield_step": telemetry.agent_yield_step,
                        "agent_yield_reason": telemetry.agent_yield_reason,
                        "levels_completed": completed_prev,
                        "target_levels": target_levels,
                        "elapsed_s": round(elapsed, 3),
                    },
                )

            is_win = completed_prev >= target_levels
            if (
                completed_prev > best_completed
                or (completed_prev == best_completed and attempts_taken == 1)
                or (completed_prev == best_completed and game_steps < best_game_steps)
            ):
                best_completed = completed_prev
                best_is_win = is_win
                best_game_steps = game_steps
                best_elapsed = round(elapsed, 2)
                best_telemetry = telemetry

            if is_win:
                break
            elif attempt < retries:
                print(
                    f"    ⚠️ Game {gid} incomplete ({completed_prev}/{target_levels} levels, {game_steps} steps). Scheduling retry."
                )

        # Game finished (win or attempts exhausted): close per-game trace files.
        if runner_fp is not None:
            runner_fp.close()
            runner_fp = None
        try:
            if getattr(agent, "_trace", None) is not None:
                agent._trace.close()
        except Exception:
            pass

        total_steps_taken += total_game_steps_all
        total_levels_passed += best_completed
        total_levels_possible += best_target

        # Phase C & D: Post-game autonomous learning hooks
        try:
            epistemic_engine = getattr(agent, "internal_engine", getattr(agent, "engine", None))
            if epistemic_engine is not None:
                # Phase C: Extract and promote learned invariants
                if hasattr(epistemic_engine, "knowledge_transfer"):
                    epistemic_engine.knowledge_transfer.on_game_complete(
                        epistemic_engine,
                        game_id=gid,
                        levels_won=best_completed,
                    )
                # Phase D: Record strategy outcome for meta-learning governor
                if hasattr(epistemic_engine, "meta_governor"):
                    from hbllm.hcir.world.autonomous_learning import (
                        GameFeatureSignature,
                        StrategyOutcome,
                    )

                    features = GameFeatureSignature(
                        action_space=frozenset(range(8)),
                        grid_shape=(0, 0),
                        has_movement_actions=True,
                    )
                    epistemic_engine.meta_governor.record_outcome(
                        StrategyOutcome(
                            game_id=gid,
                            features=features,
                            strategy_name="epistemic_brain",
                            levels_won=best_completed,
                            total_levels=best_target,
                            time_seconds=best_elapsed,
                        )
                    )
        except Exception as e:
            logger.debug("Phase C/D on_game_complete error: %s", e)

        status_label = (
            "✅ WON ALL LEVELS"
            if best_is_win
            else (
                f"⚠️ {best_completed}/{best_target} levels"
                if best_completed > 0
                else "❌ INCOMPLETE"
            )
        )

        print(
            f"    🏁 Result: {status_label} in {best_elapsed:.2f}s "
            f"({best_game_steps} steps in best run, {total_game_steps_all} total steps across {attempts_taken} attempt(s))"
        )

        from plugins.arc_agi_adapter.scorecard import compute_game_metrics

        results[gid] = {
            "title": title,
            "levels_passed": best_completed,
            "target_levels": best_target,
            "won": best_is_win,
            "steps": best_game_steps,
            "total_steps": total_game_steps_all,
            "attempts": attempts_taken,
            "elapsed_seconds": best_elapsed,
            "baseline": baseline,
            **compute_game_metrics(
                steps=best_game_steps,
                latencies_ms=(best_telemetry.cycle_latencies_ms if best_telemetry else []),
                stagnant_steps=(best_telemetry.max_stagnation_streak if best_telemetry else 0),
                noop_actions=(best_telemetry.noop_actions if best_telemetry else 0),
                contract_violations=(best_telemetry.contract_violations if best_telemetry else 0),
                decision_exceptions=(best_telemetry.decision_exceptions if best_telemetry else 0),
                decide_latencies_ms=(
                    best_telemetry.decide_latencies_ms if best_telemetry else None
                ),
                max_stagnation_streak=(
                    best_telemetry.max_stagnation_streak if best_telemetry else 0
                ),
                agent_yield_step=(best_telemetry.agent_yield_step if best_telemetry else None),
                agent_yield_reason=(best_telemetry.agent_yield_reason if best_telemetry else None),
                stop_detail=(best_telemetry.stop_detail if best_telemetry else None),
                levels_completed=best_completed,
                win_levels=best_target,
                state="WIN" if best_is_win else "NOT_FINISHED",
                stop_reason=(
                    best_telemetry.stop_reason
                    if best_telemetry and best_telemetry.stop_reason
                    else "unknown"
                ),
                baseline_actions=len(baseline) if baseline else None,
            ),
        }
        if best_telemetry is not None:
            run_best_telemetries[gid] = best_telemetry

        # Save continuous scorecard checkpoint
        _all_cycle_ms = [ms for t in run_best_telemetries.values() for ms in t.cycle_latencies_ms]
        _all_decide_ms = [ms for t in run_best_telemetries.values() for ms in t.decide_latencies_ms]
        checkpoint = {
            "timestamp": time.time(),
            "total_games_tested": idx,
            "total_levels_passed": total_levels_passed,
            "total_levels_possible": total_levels_possible,
            "pass_rate_pct": round((total_levels_passed / max(1, total_levels_possible)) * 100, 2),
            "latency_summary": {
                "mean_cycle_ms": round(statistics.fmean(_all_cycle_ms), 2)
                if _all_cycle_ms
                else None,
                "p95_cycle_ms": _p95_ms(_all_cycle_ms),
                "mean_decide_ms": round(statistics.fmean(_all_decide_ms), 2)
                if _all_decide_ms
                else None,
            },
            "games": results,
        }
        scorecard_path.write_text(json.dumps(checkpoint, indent=2))

    total_time_elapsed = time.time() - total_time_start
    overall_pct = (total_levels_passed / max(1, total_levels_possible)) * 100.0

    print(f"\n{'=' * 75}")
    print("🏁 FINAL BENCHMARK SUMMARY")
    print(f"{'=' * 75}")
    print(
        f"  Total Levels Passed:   {total_levels_passed} / {total_levels_possible} ({overall_pct:.1f}%)"
    )
    print(f"  Total Steps Executed:  {total_steps_taken}")
    print(
        f"  Total Benchmark Time:  {total_time_elapsed:.2f}s ({total_time_elapsed / 60:.1f} minutes)"
    )
    print(f"{'=' * 75}\n")

    for gid, res in results.items():
        title = res.get("title", gid)
        passed = res.get("levels_passed", 0)
        target = res.get("target_levels", 1)
        st = "✅" if res.get("won") else ("⚠️" if passed > 0 else "❌")
        print(
            f"  {st} {title:<10} ({gid:<15}): {passed}/{target} levels in {res.get('steps')} steps ({res.get('elapsed_seconds')}s)"
        )

    print(f"\n📄 Saved full scorecard to: {scorecard_path.resolve()}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HBLLM Cognitive Architecture Benchmark across ARC-AGI-3 games."
    )
    parser.add_argument(
        "--game",
        "-g",
        type=str,
        default=None,
        help="Comma-separated game IDs or titles to run (e.g. 'AR25', 'TR87', 'ar25,tr87'). Default: all games.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="all_25_games_scorecard.json",
        help="Path for saving benchmark results JSON scorecard.",
    )
    parser.add_argument(
        "--disable-archetypes",
        action="store_true",
        default=True,
        help="No-op flag for backward compatibility (archetypes removed).",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Custom executive directives separated by semicolon ';' or newline (default: canonical 5-step cognitive directive).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all 25 ARC-AGI-3 games and exit.",
    )
    parser.add_argument(
        "--max-steps",
        "-s",
        type=int,
        default=1000,
        help="Maximum total actions allowed per game (competition standard: 1000). Set to 0 for unlimited (10000).",
    )
    parser.add_argument(
        "--max-level-steps",
        type=int,
        default=None,
        help="Optional maximum steps per individual level (default: None, completely unlimited per level).",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=0,
        help="Number of retries for failed games (default: 0).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Restart entire benchmark from Game 1, backing up any prior scorecard.",
    )

    parser.add_argument("--per-game-time-limit", type=float, default=600.0)
    parser.add_argument("--trace-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env", choices=["arc"], default="arc")

    args = parser.parse_args()

    if args.list:
        arcade = Arcade()
        envs = arcade.get_environments()
        seen = set()
        deduped_envs = []
        for e in envs:
            if e.game_id not in seen:
                seen.add(e.game_id)
                deduped_envs.append(e)
        print(f"Available ARC-AGI-3 games ({len(deduped_envs)} total):")
        for i, e in enumerate(deduped_envs, 1):
            print(f"  {i:2d}. {e.game_id:<12} | Title: {getattr(e, 'title', e.game_id)}")
        sys.exit(0)

    selected = [x.strip() for x in args.game.split(",")] if args.game else None
    custom_inst = None
    if args.instructions:
        custom_inst = [
            x.strip() for x in args.instructions.replace("\n", ";").split(";") if x.strip()
        ]

    run_all_25_games(
        selected_games=selected,
        scorecard_file=args.output,
        instructions=custom_inst,
        max_steps_per_game=args.max_steps,
        max_level_steps=args.max_level_steps,
        retries=args.retries,
        resume=not args.fresh,
        per_game_time_limit=args.per_game_time_limit,
        trace_dir=args.trace_dir,
        seed=args.seed,
    )
