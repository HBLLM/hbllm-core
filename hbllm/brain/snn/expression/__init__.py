"""
Expression-side Cognitive Stream (Layer 5).

Decomposes ``UnderstandingState`` into a thought outline, gates LLM
generation per thought unit via an SNN controller, and evaluates each
generated fragment with a reward evaluator.

Pipeline::

    UnderstandingState → ThoughtPlanner → thought goals[]
        → for each goal:
            ThoughtController (SNN gate) → LLM generation
            → RewardEvaluator / TrainedPRM → accept / revise
        → assembled response

Public API
----------
.. autosummary::
    ThoughtPlanner
    ThoughtController
    RewardEvaluator
    TrainedPRM
    ShallowRenderer
    ContentPlanner
    BrocaEncoder
    PRMTrainer
    ExpressionStream
    ThoughtGoal
    ThoughtFragment
    ExpressionResult
"""

from __future__ import annotations

from hbllm.brain.snn.expression.expression_stream import ExpressionStream
from hbllm.brain.snn.expression.models import (
    ExpressionResult,
    ThoughtFragment,
    ThoughtGoal,
)
from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator
from hbllm.brain.snn.expression.thought_controller import ThoughtController
from hbllm.brain.snn.expression.thought_planner import ThoughtPlanner
from hbllm.brain.snn.expression.trained_prm import (
    RewardNetwork,
    TrainedPRM,
    TrainingCollector,
)
from hbllm.brain.snn.expression.shallow_renderer import (
    RenderingContext,
    RenderPromptBuilder,
    ShallowRenderer,
)
from hbllm.brain.snn.expression.content_planner import (
    ContentNode,
    ContentPlanner,
    ContentPlanNetwork,
)
from hbllm.brain.snn.expression.broca_encoder import BrocaEncoder, BrocaPrompt
from hbllm.brain.snn.expression.prm_trainer import PRMTrainer, TrainingMetrics

__all__ = [
    "ThoughtPlanner",
    "ThoughtController",
    "RewardEvaluator",
    "TrainedPRM",
    "RewardNetwork",
    "TrainingCollector",
    "ShallowRenderer",
    "RenderingContext",
    "RenderPromptBuilder",
    "ContentPlanner",
    "ContentNode",
    "ContentPlanNetwork",
    "BrocaEncoder",
    "BrocaPrompt",
    "PRMTrainer",
    "TrainingMetrics",
    "ExpressionStream",
    "ThoughtGoal",
    "ThoughtFragment",
    "ExpressionResult",
]
