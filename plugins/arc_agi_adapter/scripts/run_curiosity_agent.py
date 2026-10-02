"""Standalone Benchmark Runner for Universal Curiosity & Exit AGI Agent.

Runs the 4-pillar curiosity-and-exit AGI agent independently on ARC-AGI-3 environments
without modifying any existing files.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from arcengine import GameAction as ARCGameAction

from plugins.arc_agi_adapter.universal_agi_agent import UniversalAGIAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("curiosity_runner")


@dataclass
class CuriosityLevelResult:
    level_index: int
    completed: bool
    actions_taken: int
    baseline_actions: int
    efficiency_ratio: float


@dataclass
class CuriosityGameResult:
    game_id: str
    levels_completed: int
    total_levels: int
    total_actions: int
    level_results: list[CuriosityLevelResult] = field(default_factory=list)


class CuriosityBenchmarkRunner:
    """Benchmark runner for UniversalCuriosityAgent."""

    def __init__(self, agent: UniversalAGIAgent | None = None) -> None:
        self.agent = agent or UniversalAGIAgent()

    def run_environment(
        self,
        arcade_client: Any,
        game_id: str,
        max_levels: int = 2,
        max_retries_per_level: int = 1,
    ) -> CuriosityGameResult:
        logger.info(f"Starting Universal Curiosity evaluation on game: {game_id}...")
        env = arcade_client.make(game_id, render_mode=None)
        frame_data = env.reset()

        total_levels = max_levels
        if hasattr(env, "info") and hasattr(env.info, "total_levels") and env.info.total_levels:
            total_levels = min(max_levels, env.info.total_levels)
        elif hasattr(env, "_game") and hasattr(env._game, "levels"):
            total_levels = min(max_levels, len(env._game.levels))

        baseline_list: list[int] = [30] * total_levels
        if hasattr(env, "_game") and hasattr(env._game, "baseline_actions"):
            baseline_list = list(env._game.baseline_actions)[:total_levels]
        elif (
            hasattr(env, "info")
            and hasattr(env.info, "baseline_actions")
            and env.info.baseline_actions
        ):
            baseline_list = list(env.info.baseline_actions)[:total_levels]

        level_results: list[CuriosityLevelResult] = []
        levels_completed = 0
        total_actions = 0

        for lvl_idx in range(total_levels):
            baseline = baseline_list[lvl_idx] if lvl_idx < len(baseline_list) else 30
            effective_max_steps = max(int(baseline * 3.5), 80)
            max_attempts = 1 + max_retries_per_level
            lvl_completed = False
            lvl_actions = 0

            for attempt in range(max_attempts):
                is_retry = attempt > 0
                if is_retry:
                    logger.info(
                        f"Retrying level {lvl_idx + 1} (attempt {attempt + 1}/{max_attempts}) on {game_id}..."
                    )
                    self.agent.reset_episode(retain_dynamics=True, is_retry=True)
                    try:
                        frame_data = env.step(ARCGameAction.RESET)
                    except Exception as e:
                        logger.warning(f"Reset failed: {e}")
                        break
                else:
                    self.agent.reset_episode(retain_dynamics=(lvl_idx > 0), is_retry=False)

                curr_grid = (
                    frame_data.frame[-1] if frame_data and frame_data.frame else np.zeros((16, 16))
                )

                for _ in range(effective_max_steps):
                    available_actions = getattr(frame_data, "available_actions", [1, 2, 3, 4])
                    if not available_actions:
                        available_actions = [1, 2, 3, 4]
                    available_ints = [
                        a.value if hasattr(a, "value") else int(a) for a in available_actions
                    ]

                    action_int, _ = self.agent.plan_next_action(
                        curr_grid, available_ints, level=lvl_idx
                    )
                    game_act = getattr(ARCGameAction, f"ACTION{action_int}", ARCGameAction.ACTION1)

                    action_data = self.agent.last_action_data
                    if action_int == 6:
                        if not isinstance(action_data, dict) or "x" not in action_data:
                            H, W = curr_grid.shape
                            action_data = {"x": W // 2, "y": H // 2}

                    if action_data:
                        try:
                            frame_data = env.step(game_act, data=action_data)
                        except TypeError:
                            frame_data = env.step(game_act)
                    else:
                        frame_data = env.step(game_act)

                    lvl_actions += 1
                    total_actions += 1

                    curr_grid = (
                        frame_data.frame[-1]
                        if frame_data and frame_data.frame
                        else np.zeros((16, 16))
                    )

                    state_name = getattr(frame_data.state, "name", str(frame_data.state))
                    current_env_level = getattr(
                        frame_data,
                        "levels_completed",
                        getattr(env, "_current_level_index", lvl_idx),
                    )

                    if state_name == "WIN" or current_env_level > lvl_idx:
                        lvl_completed = True
                        break

                    if state_name in ("GAME_OVER", "NOT_PLAYED", "LOSE"):
                        break

                if lvl_completed:
                    break

            eff = (baseline / lvl_actions) if lvl_completed and lvl_actions > 0 else 0.0
            level_results.append(
                CuriosityLevelResult(
                    level_index=lvl_idx,
                    completed=lvl_completed,
                    actions_taken=lvl_actions,
                    baseline_actions=baseline,
                    efficiency_ratio=eff,
                )
            )

            if lvl_completed:
                levels_completed += 1
                logger.info(
                    f"Level {lvl_idx + 1}: PASSED in {lvl_actions} actions (Baseline={baseline}, Eff={eff * 100:.1f}%)"
                )
            else:
                logger.info(f"Level {lvl_idx + 1}: FAILED after {lvl_actions} actions")
                break  # Stop evaluating higher levels if this level didn't pass

        return CuriosityGameResult(
            game_id=game_id,
            levels_completed=levels_completed,
            total_levels=total_levels,
            total_actions=total_actions,
            level_results=level_results,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute Universal Curiosity & Exit Agent on ARC-AGI-3."
    )
    parser.add_argument(
        "--games",
        nargs="+",
        default=["sp80"],
        help="List of ARC-AGI-3 game IDs to evaluate.",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=2,
        help="Maximum levels per game to evaluate (default: 2).",
    )
    parser.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        default=1,
        help="Number of retries per level if attempt fails (default: 1).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ARC_API_KEY", ""),
        help="ARC-AGI API key.",
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="hcir",
        choices=["hcir", "standalone"],
        help="Agent architecture: 'hcir' (grounded in core HCIR language) or 'standalone'.",
    )

    args = parser.parse_args()

    game_list: list[str] = []
    for g in args.games:
        game_list.extend([x.strip() for x in g.split(",") if x.strip()])

    if "all" in game_list:
        game_list = [
            "ar25",
            "bp35",
            "cd82",
            "cn04",
            "dc22",
            "ft09",
            "g50t",
            "ka59",
            "lf52",
            "lp85",
            "ls20",
            "m0r0",
            "r11l",
            "re86",
            "s5i5",
            "sb26",
            "sc25",
            "sk48",
            "sp80",
            "su15",
            "tn36",
            "tr87",
            "tu93",
            "vc33",
            "wa30",
        ]

    try:
        from arc_agi import Arcade

        arcade_client = Arcade(arc_api_key=args.api_key) if args.api_key else Arcade()
    except Exception as e:
        logger.error(f"Failed to initialize ARC Arcade client: {e}")
        sys.exit(1)

    if args.agent == "hcir":
        from plugins.arc_agi_adapter.hcir_curiosity_agent import HCIRCuriosityAgent

        selected_agent = HCIRCuriosityAgent()
    else:
        from plugins.arc_agi_adapter.universal_agi_agent import UniversalAGIAgent

        selected_agent = UniversalAGIAgent()

    runner = CuriosityBenchmarkRunner(agent=selected_agent)

    logger.info("=" * 60)
    logger.info(f"Starting Universal Curiosity & Exit AGI Benchmark [{args.agent.upper()}]")
    logger.info(f"Games to test: {game_list}")
    logger.info(f"Max levels: {args.max_levels} | Retries: {args.max_retries}")
    logger.info("=" * 60)

    start_time = time.time()
    results: list[CuriosityGameResult] = []
    for idx, gid in enumerate(game_list, 1):
        try:
            res = runner.run_environment(
                arcade_client,
                gid,
                max_levels=args.max_levels,
                max_retries_per_level=args.max_retries,
            )
            results.append(res)
            logger.info(
                f"[{idx}/{len(game_list)}] Completed {gid}: {res.levels_completed}/{res.total_levels} levels passed ({res.total_actions} actions)"
            )
        except Exception as e:
            logger.error(f"Error evaluating game {gid}: {e}", exc_info=True)

    total_time = time.time() - start_time
    total_levels_completed = sum(r.levels_completed for r in results)
    total_levels_attempted = sum(r.total_levels for r in results)
    total_actions = sum(r.total_actions for r in results)

    logger.info("=" * 60)
    logger.info(f"Benchmark Complete in {total_time:.2f}s")
    logger.info(f"Total Score: {total_levels_completed}/{total_levels_attempted}")
    logger.info(f"Total Actions: {total_actions}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
