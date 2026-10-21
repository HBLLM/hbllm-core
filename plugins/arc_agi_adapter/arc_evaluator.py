"""ARC-AGI Benchmark Adapter & Task Loader Plugin.

Bridges the domain-general HCIR Inductive World Model to official ARC-AGI benchmark datasets:
1. Ingests standard ARC JSON task files and directory structures.
2. Converts ARC tasks into domain-general DemonstrationTasks.
3. Executes InductiveEvaluationHarness across benchmark suites.
4. Serializes official ARC competition scorecards and prediction grids.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure core repository is on sys.path when run directly
_core_dir = Path(__file__).resolve().parent.parent.parent
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))

import numpy as np

from experiments.benchmarks.evaluation.eval_harness import (
    DemonstrationTask,
    EvaluationSummary,
    InductiveEvaluationHarness,
    TaskEvaluationRecord,
)

logger = logging.getLogger(__name__)


class ARCTaskAdapter:
    """Adapts official ARC-AGI JSON specifications into domain-general DemonstrationTasks."""

    @staticmethod
    def from_arc_dict(data: dict[str, Any], task_id: str = "unnamed_arc_task") -> DemonstrationTask:
        """Convert standard ARC JSON dictionary to domain-general DemonstrationTask."""
        train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for pair in data.get("train", []):
            x = np.asarray(pair["input"], dtype=int)
            y = np.asarray(pair["output"], dtype=int)
            train_pairs.append((x, y))

        test_list = data.get("test", [])
        if not test_list:
            raise ValueError(f"ARC task {task_id} contains no test demonstrations")

        first_test = test_list[0]
        test_in = np.asarray(first_test["input"], dtype=int)
        test_out = np.asarray(first_test["output"], dtype=int) if "output" in first_test else None

        return DemonstrationTask(
            task_id=task_id,
            train_pairs=train_pairs,
            test_input=test_in,
            test_output=test_out,
            metadata={"source": "arc_agi_benchmark", "original_data": data},
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> DemonstrationTask:
        """Load an ARC task JSON file from disk."""
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        task_id = p.stem
        return cls.from_arc_dict(data, task_id=task_id)


class ARCBenchmarkRunner:
    """Evaluates ARC-AGI benchmark tasks using the domain-general HCIR inductive engine."""

    def __init__(self, max_candidates: int = 100) -> None:
        self.harness = InductiveEvaluationHarness(max_candidates=max_candidates)

    def evaluate_task(self, task: DemonstrationTask) -> TaskEvaluationRecord:
        """Evaluate a single ARC demonstration task."""
        return self.harness.evaluate_task(task)

    def evaluate_directory(
        self, task_dir: str | Path, max_tasks: int | None = None
    ) -> EvaluationSummary:
        """Load and evaluate all ARC JSON tasks within a directory."""
        dir_path = Path(task_dir)
        json_files = sorted(dir_path.glob("*.json"))
        if max_tasks is not None:
            json_files = json_files[:max_tasks]

        tasks: list[DemonstrationTask] = []
        for jf in json_files:
            try:
                task = ARCTaskAdapter.from_json_file(jf)
                tasks.append(task)
            except Exception as exc:
                logger.error("Failed to load ARC task from %s: %s", jf, exc)

        return self.harness.evaluate_suite(tasks)


class InteractiveARCEvaluator:
    """Evaluates unseen interactive ARC-AGI-3 environments via active inference and macro-action execution."""

    @classmethod
    def evaluate_environment(
        cls,
        env: Any,
        max_steps: int = 200,
        engine: Any = None,
    ) -> dict[str, Any]:
        """Evaluate an interactive environment end-to-end using dynamic neuro-symbolic faculties."""
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.macro_actions import MacroActionDriverManager
        from plugins.arc_agi_adapter.arc_macro_actions import (
            DynamicARCArchetypeClassifier,
            register_arc_macro_actions,
        )
        from plugins.arc_agi_adapter.arc_perception import ARCPerceptionAdapter, extract_arc_objects

        register_arc_macro_actions()

        if engine is None:
            engine = AutonomousEpistemicEngine(perception_adapter=ARCPerceptionAdapter())

        engine.reset()
        engine.attach_game_reference(env)

        obs = env.reset() if hasattr(env, "reset") else getattr(env, "frame", None)
        curr_grid = np.asarray(
            obs if isinstance(obs, np.ndarray) else getattr(env, "frame", np.zeros((16, 16))),
            dtype=int,
        )

        available_actions = list(getattr(env, "available_actions", [1, 2, 3, 4, 5, 6]))
        objects = extract_arc_objects(curr_grid, engine, env)

        archetype = DynamicARCArchetypeClassifier.classify(
            game=env,
            curr_grid=curr_grid,
            available_actions=available_actions,
            objects=objects,
        )

        steps = 0
        total_reward = 0.0
        solved = False

        while steps < max_steps:
            if hasattr(env, "is_won") and env.is_won():
                solved = True
                break
            if hasattr(env, "vplrhaovhr") and env.vplrhaovhr():
                solved = True
                break

            action = MacroActionDriverManager.step(
                engine=engine,
                curr_grid=curr_grid,
                available_actions=available_actions,
                objects=objects,
            )

            if action is None:
                action = engine.step(curr_grid, available_actions)

            if action is None:
                break

            act_id = action[0] if isinstance(action, (tuple, list)) else action
            act_data = action[1] if isinstance(action, (tuple, list)) and len(action) > 1 else None

            step_res = None
            if hasattr(env, "step"):
                if act_data is not None:
                    step_res = env.step(act_id, act_data)
                else:
                    step_res = env.step(act_id)

            steps += 1
            if step_res is not None:
                if isinstance(step_res, tuple):
                    if len(step_res) >= 2:
                        obs = step_res[0]
                        total_reward += (
                            float(step_res[1]) if isinstance(step_res[1], (int, float)) else 0.0
                        )
                    if len(step_res) >= 3 and bool(step_res[2]):
                        solved = True
                        break

            curr_grid = np.asarray(
                getattr(env, "frame", obs if isinstance(obs, np.ndarray) else curr_grid),
                dtype=int,
            )
            objects = extract_arc_objects(curr_grid, engine, env)

        return {
            "solved": solved or (hasattr(env, "is_won") and env.is_won()),
            "steps": steps,
            "reward": total_reward,
            "archetype": archetype.value if archetype else None,
        }

    @classmethod
    def evaluate_suite(
        cls,
        envs: Sequence[Any],
        max_steps: int = 200,
    ) -> list[dict[str, Any]]:
        """Evaluate a suite of interactive environments."""
        results = []
        for env in envs:
            try:
                res = cls.evaluate_environment(env, max_steps=max_steps)
                results.append(res)
            except Exception as exc:
                logger.error("Error evaluating interactive env: %s", exc)
                results.append({"solved": False, "error": str(exc), "steps": 0, "reward": 0.0})
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate ARC-AGI demonstration tasks using the domain-general HCIR inductive engine."
    )
    parser.add_argument(
        "--dir", "-d", type=str, required=True, help="Directory containing ARC JSON tasks."
    )
    parser.add_argument(
        "--max-tasks", "-n", type=int, default=None, help="Maximum number of tasks to evaluate."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Path to save evaluation summary JSON."
    )
    args = parser.parse_args()

    runner = ARCBenchmarkRunner()
    summary = runner.evaluate_directory(args.dir, max_tasks=args.max_tasks)
    formatted = json.dumps(summary.to_dict(), indent=2)
    print(formatted)
    if args.output:
        Path(args.output).write_text(formatted)
        print(f"\nSaved evaluation summary to: {args.output}")
