"""Run HBLLM Cognitive USB Agent against ALL 25 ARC-AGI-3 games across ALL levels.

Evaluates against official Arcade environments installed in .venv.
Writes continuous live progress to `all_25_games_scorecard.json`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure core repository is on sys.path
_core_dir = Path(__file__).resolve().parent.parent
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))

from arc_agi import Arcade

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
            action_schema=[
                {"name": "MOVE_UP", "action_id": 1},
                {"name": "MOVE_DOWN", "action_id": 2},
                {"name": "MOVE_LEFT", "action_id": 3},
                {"name": "MOVE_RIGHT", "action_id": 4},
                {"name": "INTERACT", "action_id": 5},
                {"name": "CLICK_CELL", "action_id": 6, "parameters": {"x": "int", "y": "int"}},
            ],
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
        return DriverInput(
            raw_data=grid,
            source_id=self.name,
            modality=DriverModality.GRID_2D,
            metadata={
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
            },
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
        is_done = is_win or state_str == "GAME_OVER"
        reward = 1.0 if is_win else (0.5 if lvl > 0 else 0.0)

        return DriverFeedback(
            success=True,
            reward=reward,
            terminated=is_done,
            causal_delta=_extract_grid(getattr(raw_result, "frame", None)),
            info={"levels_completed": lvl, "state": state_str, "win_levels": win_levels},
        )


# ── 2. Cognitive Agent Adapter ────────────────────────────────────────────────
class CognitiveAgentAdapter:
    """Wraps MyAgent into the canonical CognitiveBlackbox interface."""

    def __init__(self, agent: MyAgent) -> None:
        self.agent = agent
        self.history: list[Any] = []
        self.last_frame: Any = None

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

        return DriverAction(action_id=act_id, parameters=params)

    def update(self, action: DriverAction, feedback: DriverFeedback, source_id: str = "") -> None:
        if hasattr(self.agent, "internal_engine") and self.agent.internal_engine is not None:
            engine = self.agent.internal_engine
            curr_grid = feedback.causal_delta
            avail = feedback.info.get("available_actions", [1, 2, 3, 4, 5, 6])
            st = feedback.info.get("state", "")
            lvl = int(feedback.info.get("levels_completed", 0))
            is_win = st == "WIN" or lvl > getattr(self, "_last_lvl", 0)
            is_lost = st == "GAME_OVER"
            if curr_grid is not None:
                engine.assimilate_feedback(
                    curr_grid,
                    avail,
                    is_win=is_win,
                    is_lost=is_lost,
                    action=action.action_id,
                    action_data=action.parameters,
                )
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
) -> None:
    if instructions is None:
        instructions = DEFAULT_EXECUTIVE_DIRECTIVES

    arcade = Arcade()
    environments = arcade.get_environments()

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
    print(
        f"   Mode: {'Pure AutonomousEpistemicEngine (No Archetypes)' if disable_archetypes else 'Hybrid Competitive (Inductive HCIR + Epistemic Brain)'}"
    )
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

    total_levels_passed = 0
    total_levels_possible = 0
    total_steps_taken = 0
    total_time_start = time.time()

    for idx, env_meta in enumerate(environments, 1):
        gid = env_meta.game_id
        title = getattr(env_meta, "title", gid)
        baseline = getattr(env_meta, "baseline_actions", None) or [100]

        print(f"\n[{idx:2d}/{len(environments)}] 🎮 Game: {title} ({gid})")
        print(f"    Baseline Actions: {baseline} | Tags: {getattr(env_meta, 'tags', [])}")

        best_completed = 0
        best_target = len(baseline) or 1
        best_is_win = False
        best_game_steps = 0
        best_elapsed = 0.0
        attempts_taken = 0
        total_game_steps_all = 0

        # Instantiate fresh agent & Cognitive USB Driver for clean inter-game isolation
        agent = MyAgent(disable_archetypes=disable_archetypes, instructions=instructions)
        cognitive_engine = CognitiveAgentAdapter(agent)

        for attempt in range(retries + 1):
            attempts_taken += 1
            if attempt > 0:
                print(f"\n    🔁 [Retry {attempt}/{retries}] for failed game {title} ({gid})...")

            env = arcade.make(gid)
            if env is None:
                print(f"    ⚠️ Could not make game environment for {gid}")
                continue

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

            for step in range(1, total_game_budget + 1):
                feedback = manager.execute_cognitive_step()
                game_steps += 1
                current_level_steps += 1
                info = feedback.info
                lvl = int(info.get("levels_completed", 0))

                if lvl > completed_prev:
                    print(
                        f"    🎉 Level {lvl}/{target_levels} PASSED at step {step} (in {current_level_steps} steps)!"
                    )
                    completed_prev = lvl
                    current_level_steps = 0
                    cognitive_engine.reset_episode(
                        retain_dynamics=True, is_new_level=True, level=lvl
                    )
                elif lvl < completed_prev:
                    completed_prev = lvl
                    current_level_steps = 0
                    cognitive_engine.reset_episode(
                        retain_dynamics=True, is_new_level=True, level=lvl
                    )

                state_str = str(info.get("state", "RUNNING"))
                if state_str == "WIN" or lvl >= target_levels:
                    break

                # Optional per-level step limit if explicitly configured by user
                if max_level_steps is not None and current_level_steps >= max_level_steps:
                    print(
                        f"    ⏱️ Level {lvl + 1}/{target_levels} reached per-level limit "
                        f"({current_level_steps} >= {max_level_steps} steps). Concluding attempt."
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
                            break
                    reset_state = getattr(
                        getattr(reset_frame, "state", None),
                        "name",
                        str(getattr(reset_frame, "state", "")),
                    )
                    if reset_state == "GAME_OVER":
                        print(f"    ❌ GAME_OVER terminal at step {step}")
                        break
                    driver.set_frame(reset_frame)

            elapsed = time.time() - start_time
            total_game_steps_all += game_steps

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

            if is_win:
                break
            elif attempt < retries:
                print(
                    f"    ⚠️ Game {gid} incomplete ({completed_prev}/{target_levels} levels, {game_steps} steps). Scheduling retry."
                )

        total_steps_taken += total_game_steps_all
        total_levels_passed += best_completed
        total_levels_possible += best_target

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
        }

        # Save continuous scorecard checkpoint
        checkpoint = {
            "timestamp": time.time(),
            "total_games_tested": idx,
            "total_levels_passed": total_levels_passed,
            "total_levels_possible": total_levels_possible,
            "pass_rate_pct": round((total_levels_passed / max(1, total_levels_possible)) * 100, 2),
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
        "--enable-archetypes",
        action="store_true",
        help="Enable legacy heuristic archetypes (default: disabled, 100 percent pure AutonomousEpistemicEngine).",
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

    args = parser.parse_args()

    if args.list:
        arcade = Arcade()
        envs = arcade.get_environments()
        print(f"Available ARC-AGI-3 games ({len(envs)} total):")
        for i, e in enumerate(envs, 1):
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
        disable_archetypes=not args.enable_archetypes,
        scorecard_file=args.output,
        instructions=custom_inst,
        max_steps_per_game=args.max_steps,
        max_level_steps=args.max_level_steps,
        retries=args.retries,
    )
