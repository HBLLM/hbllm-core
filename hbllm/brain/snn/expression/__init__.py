"""
Expression-side Cognitive Stream (Layer 5).

Decomposes ``UnderstandingState`` into a thought outline, gates LLM
generation per thought unit via an SNN controller, and evaluates each
generated fragment with a reward evaluator.

Pipeline::

    UnderstandingState → ThoughtPlanner → thought goals[]
        → for each goal:
            ThoughtController (SNN gate) → LLM generation
            → RewardEvaluator → accept / revise
        → assembled response

Public API
----------
.. autosummary::
    ThoughtPlanner
    ThoughtController
    RewardEvaluator
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

__all__ = [
    "ThoughtPlanner",
    "ThoughtController",
    "RewardEvaluator",
    "ExpressionStream",
    "ThoughtGoal",
    "ThoughtFragment",
    "ExpressionResult",
]
