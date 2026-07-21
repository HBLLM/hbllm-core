"""
Attention Graph & Attention Manager — dynamic resource allocation of cognitive focus.

Attention is represented as a first-class resource allocation graph:

    Goals / Surprise Signals (PredictionError)
               │
      AttentionManager
               │
    AttentionGraph (allocates salience across subtrees)
               │
    CognitiveScheduler Dispatch Priority
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hbllm.hcir.graph import HCIRNodeType, PredictionErrorNode
from hbllm.hcir.types import Attention
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class AttentionNode:
    """A focus element in the attention graph."""

    target_node_id: str
    salience: float = 0.5  # 0.0 to 1.0
    focus_reason: str = "goal_driven"


@dataclass
class AttentionGraph:
    """Represents the active attention distribution across workspace nodes."""

    focus_nodes: dict[str, AttentionNode] = field(default_factory=dict)
    total_salience: float = 0.0

    def add_focus(self, node_id: str, salience: float, reason: str = "goal_driven") -> None:
        self.focus_nodes[node_id] = AttentionNode(
            target_node_id=node_id,
            salience=salience,
            focus_reason=reason,
        )
        self.total_salience = sum(f.salience for f in self.focus_nodes.values())


class AttentionManager:
    """Manages cognitive attention distribution across goals and surprise signals.

    Usage::

        attn_mgr = AttentionManager(workspace)
        attn_graph = attn_mgr.recompute_attention()
    """

    def __init__(self, workspace: HCIRWorkspaceState) -> None:
        self._workspace = workspace

    def recompute_attention(self) -> AttentionGraph:
        """Recompute attention distribution based on active goals and prediction error surprise signals."""
        attn_graph = AttentionGraph()

        # 1. Allocate attention to active goals
        active_goals = self._workspace.active_goals()
        for goal in active_goals:
            salience = min(1.0, goal.priority * 0.8)
            attn_graph.add_focus(goal.id, salience, reason="active_goal")

        # 2. Allocate attention to surprise signals (PredictionErrorNode)
        error_nodes = self._workspace.graph.nodes_by_type(HCIRNodeType.PREDICTION_ERROR)
        for err in error_nodes:
            if isinstance(err, PredictionErrorNode):
                # Larger prediction error magnitude = higher surprise salience
                surprise_salience = min(1.0, err.error_magnitude * 1.5)
                attn_graph.add_focus(err.id, surprise_salience, reason="prediction_error_surprise")

        # Update global workspace attention
        avg_salience = (
            attn_graph.total_salience / len(attn_graph.focus_nodes)
            if attn_graph.focus_nodes
            else 0.5
        )
        self._workspace.attention = Attention(salience=min(1.0, avg_salience))

        logger.debug(
            "Recomputed attention: %d focus nodes, avg salience %.2f",
            len(attn_graph.focus_nodes),
            avg_salience,
        )

        return attn_graph
