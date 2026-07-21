"""
Unified Cognitive State — consolidated snapshot of cognitive workspace state.

Aggregates all cognitive dimensions into a single inspectable state structure:

    UnifiedCognitiveState
        ├── world_state (WorldVariables, EnvironmentStates, PhysicalEntities)
        ├── attention_state (Focus nodes, dynamic salience allocation)
        ├── active_goals (GoalNodes, priorities)
        ├── beliefs (BeliefNodes, consensus claims, uncertainty)
        ├── predictions (PredictionNodes, expected outcomes)
        ├── resources (Budget constraints, consumption rates)
        ├── active_branch (Branch name, BranchMode)
        └── learning_state (SkillNodes, induction count)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import HCIRNodeType
from hbllm.hcir.kernel.attention_graph import AttentionGraph
from hbllm.hcir.types import BranchMode
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class UnifiedCognitiveState:
    """Consolidated snapshot of the cognitive runtime state."""

    timestamp: float = field(default_factory=time.time)
    branch_name: str = "main"
    branch_mode: BranchMode = BranchMode.LIVE
    node_count: int = 0
    active_goals: list[dict[str, Any]] = field(default_factory=list)
    world_variables: dict[str, Any] = field(default_factory=dict)
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    prediction_errors: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    attention_salience: float = 0.5
    attention_focus_count: int = 0

    @classmethod
    def from_workspace(
        cls,
        workspace: HCIRWorkspaceState,
        attention_graph: AttentionGraph | None = None,
    ) -> UnifiedCognitiveState:
        """Construct a unified cognitive state snapshot from a workspace state."""
        g = workspace.graph

        goals = [n.model_dump() for n in workspace.active_goals()]
        beliefs = [n.model_dump() for n in g.nodes_by_type(HCIRNodeType.BELIEF)]
        predictions = [n.model_dump() for n in g.nodes_by_type(HCIRNodeType.PREDICTION)]
        pe_nodes = [n.model_dump() for n in g.nodes_by_type(HCIRNodeType.PREDICTION_ERROR)]
        skills = [n.model_dump() for n in g.nodes_by_type(HCIRNodeType.SKILL)]

        world_vars: dict[str, Any] = {}
        for wv in g.nodes_by_type(HCIRNodeType.WORLD_VARIABLE):
            if hasattr(wv, "variable_name"):
                world_vars[wv.variable_name] = getattr(wv, "value", None)

        focus_count = len(attention_graph.focus_nodes) if attention_graph else 0
        salience = workspace.attention.salience

        return cls(
            timestamp=time.time(),
            branch_name=workspace.branch_name,
            branch_mode=workspace.branch_mode,
            node_count=g.node_count,
            active_goals=goals,
            world_variables=world_vars,
            beliefs=beliefs,
            predictions=predictions,
            prediction_errors=pe_nodes,
            skills=skills,
            attention_salience=salience,
            attention_focus_count=focus_count,
        )
