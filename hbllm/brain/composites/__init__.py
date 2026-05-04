"""
Composite brain nodes — consolidate related cognitive functions.

Each composite wraps multiple underlying nodes into a single bus-registered
entity, reducing bus overhead and simplifying the factory wiring.
"""

from hbllm.brain.composites.governance_guard import GovernanceGuard
from hbllm.brain.composites.learning_loop import LearningLoop
from hbllm.brain.composites.memory_system import MemorySystem
from hbllm.brain.composites.meta_cognition import MetaCognition
from hbllm.brain.composites.reasoning_core import ReasoningCore
from hbllm.brain.composites.resource_manager import ResourceManager
from hbllm.brain.composites.skill_engine import SkillEngine
from hbllm.brain.composites.social_layer import SocialLayer

__all__ = [
    "GovernanceGuard",
    "LearningLoop",
    "MemorySystem",
    "MetaCognition",
    "ReasoningCore",
    "ResourceManager",
    "SkillEngine",
    "SocialLayer",
]
