"""Brain — cognitive router, task planner, and intent classification."""

from hbllm.brain.cognitive_metrics import CognitiveMetrics
from hbllm.brain.confidence_estimator import ConfidenceEstimator
from hbllm.brain.factory import Brain, BrainConfig, BrainFactory
from hbllm.brain.goal_manager import GoalManager, GoalPriority
from hbllm.brain.policy_engine import PolicyEngine
from hbllm.brain.revision_node import RevisionNode
from hbllm.brain.router_node import RouterNode
from hbllm.brain.self_model import SelfModel
from hbllm.brain.sentinel_node import SentinelNode
from hbllm.brain.skill_registry import SkillRegistry
from hbllm.brain.sleep_node import SleepCycleNode
from hbllm.brain.spawner_node import SpawnerNode
from hbllm.brain.workspace_node import WorkspaceNode
from hbllm.brain.world_model_node import WorldModelNode
from hbllm.brain.world_simulator import WorldSimulator

# Lazy imports for torch-dependent modules to avoid hard dependency
# on PyTorch at package-import time (e.g. in PyInstaller sidecar builds).


def __getattr__(name: str):
    if name == "ProcessRewardNode":
        from hbllm.brain.process_reward_node import ProcessRewardNode
        return ProcessRewardNode
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
    "WorkspaceNode",
    "WorldModelNode",
    "WorldSimulator",
]
