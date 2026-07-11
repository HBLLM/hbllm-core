"""Brain — cognitive router, task planner, and intent classification."""

from hbllm.brain.control.router_node import RouterNode
from hbllm.brain.core.factory import Brain, BrainConfig, BrainFactory
from hbllm.brain.emotion.goal_manager import GoalManager, GoalPriority
from hbllm.brain.emotion.sleep_node import SleepCycleNode
from hbllm.brain.evaluation.revision_node import RevisionNode
from hbllm.brain.evaluation.utility_calibrator import CalibrationTrace, UtilityCalibrator
from hbllm.brain.evaluation.utility_engine import (
    CognitiveUtilityEngine,
    ThoughtBudget,
    UtilityBreakdown,
)
from hbllm.brain.governance.policy_engine import PolicyEngine
from hbllm.brain.governance.sentinel_node import SentinelNode
from hbllm.brain.planning.workspace_node import WorkspaceNode
from hbllm.brain.self_model.cognitive_metrics import CognitiveMetrics
from hbllm.brain.self_model.confidence_estimator import ConfidenceEstimator
from hbllm.brain.self_model.self_model import SelfModel
from hbllm.brain.skills.skill_registry import SkillRegistry
from hbllm.brain.world.world_model_node import WorldModelNode
from hbllm.brain.world.world_state import SimulationInterface, WorldStateEngine

# Lazy imports for torch-dependent modules to avoid hard dependency
# on PyTorch at package-import time (e.g. in PyInstaller sidecar builds).


def __getattr__(name: str):
    if name == "ProcessRewardNode":
        from hbllm.brain.evaluation.process_reward_node import ProcessRewardNode

        return ProcessRewardNode
    if name == "SpawnerNode":
        from hbllm.brain.emotion.spawner_node import SpawnerNode

        return SpawnerNode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
