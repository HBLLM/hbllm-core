"""Brain — cognitive router, task planner, and intent classification."""

from __future__ import annotations

import importlib
from typing import Any

# PEP 562 Lazy Loading: Map public exports to (module_path, attribute_name).
# Eliminates monolithic dependency loading when importing submodules (e.g. edge/robotics plugins).
_EXPORTS: dict[str, tuple[str, str]] = {
    "RouterNode": ("hbllm.brain.control.router_node", "RouterNode"),
    "Brain": ("hbllm.brain.core.factory", "Brain"),
    "BrainConfig": ("hbllm.brain.core.factory", "BrainConfig"),
    "BrainFactory": ("hbllm.brain.core.factory", "BrainFactory"),
    "GoalManager": ("hbllm.brain.emotion.goal_manager", "GoalManager"),
    "GoalPriority": ("hbllm.brain.emotion.goal_manager", "GoalPriority"),
    "SleepCycleNode": ("hbllm.brain.emotion.sleep_node", "SleepCycleNode"),
    "SpawnerNode": ("hbllm.brain.emotion.spawner_node", "SpawnerNode"),
    "RevisionNode": ("hbllm.brain.evaluation.revision_node", "RevisionNode"),
    "CalibrationTrace": ("hbllm.brain.evaluation.utility_calibrator", "CalibrationTrace"),
    "UtilityCalibrator": ("hbllm.brain.evaluation.utility_calibrator", "UtilityCalibrator"),
    "CognitiveUtilityEngine": ("hbllm.brain.evaluation.utility_engine", "CognitiveUtilityEngine"),
    "ThoughtBudget": ("hbllm.brain.evaluation.utility_engine", "ThoughtBudget"),
    "UtilityBreakdown": ("hbllm.brain.evaluation.utility_engine", "UtilityBreakdown"),
    "ProcessRewardNode": ("hbllm.brain.evaluation.process_reward_node", "ProcessRewardNode"),
    "PolicyEngine": ("hbllm.brain.governance.policy_engine", "PolicyEngine"),
    "SentinelNode": ("hbllm.brain.governance.sentinel_node", "SentinelNode"),
    "WorkspaceNode": ("hbllm.brain.planning.workspace_node", "WorkspaceNode"),
    "CognitiveMetrics": ("hbllm.brain.self_model.cognitive_metrics", "CognitiveMetrics"),
    "ConfidenceEstimator": ("hbllm.brain.self_model.confidence_estimator", "ConfidenceEstimator"),
    "SelfModel": ("hbllm.brain.self_model.self_model", "SelfModel"),
    "SkillRegistry": ("hbllm.brain.skills.skill_registry", "SkillRegistry"),
    "WorldModelNode": ("hbllm.brain.world.world_model_node", "WorldModelNode"),
    "WorldStateEngine": ("hbllm.brain.world.world_state", "WorldStateEngine"),
    "SimulationInterface": ("hbllm.brain.world.world_state", "SimulationInterface"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_path, attr = _EXPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = [
    "BrainFactory",
    "BrainConfig",
    "Brain",
    "CognitiveMetrics",
    "ConfidenceEstimator",
    "GoalManager",
    "GoalPriority",
    "PolicyEngine",
    "ProcessRewardNode",
    "RevisionNode",
    "RouterNode",
    "SelfModel",
    "SentinelNode",
    "SkillRegistry",
    "SleepCycleNode",
    "SpawnerNode",
    "ThoughtBudget",
    "CognitiveUtilityEngine",
    "UtilityBreakdown",
    "CalibrationTrace",
    "UtilityCalibrator",
    "WorkspaceNode",
    "WorldModelNode",
    "WorldStateEngine",
    "SimulationInterface",
]
