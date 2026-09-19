"""Counterfactual Planner — predictive simulation & candidate branch evaluation.

Implements the counterfactual planning loop:

    Goal
      ↓
    Generate Candidate Plans / Actions
      ↓
    FORK Simulation Branch per candidate (or MCTS Tree)
      ↓
    Simulate Candidate Execution & Forward Prediction
      ↓
    Evaluate Execution Receipts & Outcome Utility
      ↓
    MERGE Best Branch to main workspace
      ↓
    ROLLBACK / Drop discarded simulation branches

This transforms planning from static search into executable predictive simulation.
Supports both greedy multi-step beam search and Monte Carlo Tree Search (MCTS) with UCT selection.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import (
    ActionModality,
    ActionNode,
    GoalNode,
    HCIRNodeType,
    PhysicalEntityNode,
)
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world_kernel import WorldKernel

logger = logging.getLogger(__name__)


@dataclass
class CandidatePlanResult:
    """Evaluation result for a single candidate simulation branch."""

    candidate_id: str
    action: ActionNode
    branch_name: str
    receipt: ExecutionReceipt
    utility_score: float


@dataclass
class MCTSConfig:
    """Configuration options for Monte Carlo Tree Search in CounterfactualPlanner."""

    exploration_constant: float = 1.414  # UCT exploration coefficient (c)
    max_simulations: int = 16  # Total MCTS expansion and rollout iterations
    max_depth: int = 3  # Maximum tree depth per rollout trajectory
    discount_factor: float = 0.85  # Discount factor for future expected utilities
    enable_subgoals: bool = True  # Prioritize goal-directed candidate actions
    causal_pruning: bool = True  # Prioritize actions advancing unsatisfied causal conditions


@dataclass
class MCTSNode:
    """A search tree node in the HCIR counterfactual simulation tree."""

    node_id: str
    state_branch: str
    action: ActionNode | None = None
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    unexpanded_actions: list[ActionNode] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    depth: int = 0
    is_terminal: bool = False
    step_utility: float = 0.0
    receipt: ExecutionReceipt | None = None

    def average_value(self) -> float:
        """Q(s, a): expected utility of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def uct_score(self, parent_visits: int, exploration_constant: float) -> float:
        """Compute Upper Confidence Bound for Trees (UCT)."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.average_value()
        exploration = exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / self.visit_count
        )
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        return len(self.unexpanded_actions) == 0


class CounterfactualPlanner:
    """Predictive counterfactual planner using simulation branch forking.

    Usage::

        planner = CounterfactualPlanner(workspace, services)
        best_candidate = await planner.evaluate_and_select(
            goal=GoalNode(description="Optimize solar dehydrator efficiency"),
            candidate_actions=[
                ActionNode(intent="use_copper_tubing"),
                ActionNode(intent="use_aluminum_fins"),
            ],
        )
    """

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        services: KernelServices,
    ) -> None:
        self._workspace = workspace
        self._services = services
        self._world_kernel = WorldKernel(workspace)
        self._interpreter = HCIRInterpreter(workspace, services)

    def _should_auto_use_mcts(
        self,
        goal: GoalNode,
        candidate_actions: list[ActionNode],
        horizon: int,
    ) -> bool:
        """Inspect environmental complexity and autonomously decide whether to activate MCTS."""
        # 1. Movable hazards / pushable entities with non-passable obstacles (e.g. Sokoban/box deadlock risk)
        # Exclude the avatar itself; check for external movable objects (boxes, boulders) and obstacles
        entities = self._workspace.graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)
        has_movable_objects = any(
            isinstance(e, PhysicalEntityNode)
            and not e.properties.get("is_avatar", False)
            and (
                e.properties.get("movable", False)
                or "PUSHABLE" in e.properties.get("affordances", [])
            )
            for e in entities
        )
        has_obstacles = any(
            isinstance(e, PhysicalEntityNode)
            and not e.properties.get("is_avatar", False)
            and not e.properties.get("passable", True)
            for e in entities
        )
        if has_movable_objects and has_obstacles:
            return True

        # 2. Active POMDP latent variables (requires belief distribution rollouts)
        if self._world_kernel.belief_graph.get_latent_beliefs():
            return True

        # 3. High combinatorial branching depth
        if horizon >= 3 or (horizon >= 2 and len(candidate_actions) >= 4):
            return True

        return False

    async def evaluate_and_select(
        self,
        goal: GoalNode,
        candidate_actions: list[ActionNode],
        horizon: int = 1,
        beam_width: int = 2,
        author: str = "counterfactual_planner",
        use_mcts: bool | None = None,
        mcts_config: MCTSConfig | None = None,
    ) -> CandidatePlanResult:
        """Run counterfactual simulation for candidate actions and merge the best branch.

        Supports single-step, multi-step (horizon > 1) beam search, or Monte Carlo Tree Search.
        If use_mcts is None (default), autonomously selects MCTS when environmental complexity warrants it.
        """
        if not candidate_actions:
            raise ValueError("Candidate actions list cannot be empty")

        should_mcts = (
            self._should_auto_use_mcts(goal, candidate_actions, horizon)
            if use_mcts is None
            else use_mcts
        )

        if should_mcts:
            cfg = mcts_config or MCTSConfig(max_depth=max(1, horizon))
            return await self.mcts_evaluate_and_select(
                goal=goal,
                candidate_actions=candidate_actions,
                config=cfg,
                author=author,
            )

        results: list[CandidatePlanResult] = []

        for action in candidate_actions:
            branch_name = f"sim_{uuid.uuid4().hex[:6]}"
            # 1. Create simulation branch fork
            self._workspace.fork_branch(branch_name)

            # 2. Build simulation instruction stream (FORK -> ASSERT -> EXECUTE -> MERGE)
            stream = InstructionStream(
                author=author,
                description=f"Simulate candidate: {action.intent}",
                instructions=[
                    Instruction(
                        opcode=Opcode.ASSERT,
                        params={"node_data": action.model_dump(), "author": author},
                    ),
                    Instruction(
                        opcode=Opcode.EXECUTE,
                        params={
                            "capability": "world_prediction",
                            "params": {"action": action.intent, "branch": branch_name},
                        },
                    ),
                ],
            )

            # 3. Execute stream within simulation branch and get receipt
            res, receipt = await self._interpreter.execute_with_receipt(
                stream, process_id=f"proc_{branch_name}", thread_id=f"thr_{branch_name}"
            )

            # 4. Predict forward outcome using WorldKernel
            prediction = self._world_kernel.predict(
                action=action,
                confidence=0.85 if res.success else 0.2,
                author=author,
            )

            # 5. Compute utility score for step 1
            step1_utility = self._compute_action_utility(action, prediction, res.success)

            # 6. Multi-Step Mental Lookahead Rollout (if horizon > 1)
            total_utility = step1_utility
            if horizon > 1 and step1_utility > 0.01:
                discount = 0.85
                curr_discount = discount

                for step_idx in range(2, horizon + 1):
                    # Fork a sub-branch for the next step rollout
                    sub_branch = f"{branch_name}_s{step_idx}_{uuid.uuid4().hex[:4]}"
                    self._workspace.fork_branch(sub_branch)

                    # Predict next action outcome from candidate set
                    best_next_utility = -999.0
                    for next_act in candidate_actions:
                        next_pred = self._world_kernel.predict(
                            action=next_act,
                            confidence=0.85,
                            author=author,
                        )
                        next_u = self._compute_action_utility(next_act, next_pred, True)
                        if next_u > best_next_utility:
                            best_next_utility = next_u

                    self._workspace.drop_branch(sub_branch)

                    if best_next_utility < 0.01:
                        # Trajectory encounters fatal deadlock
                        total_utility = 0.001
                        break

                    total_utility += curr_discount * best_next_utility
                    curr_discount *= discount

            results.append(
                CandidatePlanResult(
                    candidate_id=action.id,
                    action=action,
                    branch_name=branch_name,
                    receipt=receipt,
                    utility_score=total_utility,
                )
            )

        # 7. Rank candidates by utility score
        best_candidate = max(results, key=lambda r: r.utility_score)

        logger.info(
            "Counterfactual planner selected candidate '%s' (branch '%s', horizon=%d) with utility score %.3f",
            best_candidate.action.intent,
            best_candidate.branch_name,
            horizon,
            best_candidate.utility_score,
        )

        # 8. Merge best branch to main workspace state and cleanup unused branches
        self._workspace.merge_branch(best_candidate.branch_name)
        for r in results:
            if r.branch_name != best_candidate.branch_name:
                self._workspace.drop_branch(r.branch_name)

        return best_candidate

    def evaluate_and_select_sync(
        self,
        goal: GoalNode,
        candidate_actions: list[ActionNode],
        horizon: int = 1,
        beam_width: int = 2,
        author: str = "counterfactual_planner",
        use_mcts: bool | None = None,
        mcts_config: MCTSConfig | None = None,
    ) -> CandidatePlanResult:
        """Synchronous wrapper for evaluate_and_select.

        Safely handles existing event loops (e.g. within async tests or runners)
        by offloading to an isolated thread executor if an active loop is running.
        """
        coro = self.evaluate_and_select(
            goal=goal,
            candidate_actions=candidate_actions,
            horizon=horizon,
            beam_width=beam_width,
            author=author,
            use_mcts=use_mcts,
            mcts_config=mcts_config,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()
        else:
            return asyncio.run(coro)

    async def mcts_evaluate_and_select(
        self,
        goal: GoalNode,
        candidate_actions: list[ActionNode],
        config: MCTSConfig | None = None,
        author: str = "counterfactual_planner",
    ) -> CandidatePlanResult:
        """Evaluate candidate actions using Monte Carlo Tree Search over simulation branches."""
        if not candidate_actions:
            raise ValueError("Candidate actions list cannot be empty")

        cfg = config or MCTSConfig()
        root_branch = f"mcts_root_{uuid.uuid4().hex[:6]}"
        self._workspace.fork_branch(root_branch)
        all_created_branches: list[str] = [root_branch]

        root_actions = self._rank_and_filter_causal_actions(
            candidate_actions, goal, cfg.causal_pruning
        )

        root = MCTSNode(
            node_id="root",
            state_branch=root_branch,
            unexpanded_actions=root_actions,
            depth=0,
        )

        for _ in range(cfg.max_simulations):
            # 1. Selection
            curr = root
            while curr.is_fully_expanded() and curr.children and not curr.is_terminal:
                curr = max(
                    curr.children,
                    key=lambda c: c.uct_score(curr.visit_count, cfg.exploration_constant),
                )

            # 2. Expansion
            if not curr.is_terminal and curr.unexpanded_actions:
                action = curr.unexpanded_actions.pop(0)
                child_branch = f"{curr.state_branch}_c{len(curr.children)}_{uuid.uuid4().hex[:4]}"
                self._workspace.fork_branch(child_branch)
                all_created_branches.append(child_branch)

                sim_node_data = {
                    **action.model_dump(),
                    "id": f"{action.id}_{uuid.uuid4().hex[:6]}",
                }
                stream = InstructionStream(
                    author=author,
                    description=f"MCTS simulate: {action.intent}",
                    instructions=[
                        Instruction(
                            opcode=Opcode.ASSERT,
                            params={"node_data": sim_node_data, "author": author},
                        ),
                        Instruction(
                            opcode=Opcode.EXECUTE,
                            params={
                                "capability": "world_prediction",
                                "params": {"action": action.intent, "branch": child_branch},
                            },
                        ),
                    ],
                )

                res, receipt = await self._interpreter.execute_with_receipt(
                    stream, process_id=f"proc_{child_branch}", thread_id=f"thr_{child_branch}"
                )

                prediction = self._world_kernel.predict(
                    action=action,
                    confidence=0.85 if res.success else 0.2,
                    author=author,
                )
                step_utility = self._compute_action_utility(
                    action, prediction, res.success, goal=goal
                )

                # Check if outcome is terminal
                pred_props = getattr(prediction, "properties", {}) or {}
                predicted_state = pred_props.get("predicted_state") or {}
                spatial_outcome = (
                    predicted_state.get("spatial_outcome")
                    if isinstance(predicted_state, dict)
                    else None
                )
                if getattr(action, "properties", None):
                    act_pred = action.properties.get("predicted_state", {})
                    if isinstance(act_pred, dict) and "spatial_outcome" in act_pred:
                        spatial_outcome = act_pred["spatial_outcome"]
                    elif "spatial_outcome" in action.properties:
                        spatial_outcome = action.properties["spatial_outcome"]

                is_deadlock = step_utility <= 0.005 or (
                    isinstance(spatial_outcome, dict) and spatial_outcome.get("deadlock", False)
                )
                goal_reached = isinstance(spatial_outcome, dict) and spatial_outcome.get(
                    "goal_reached", False
                )
                is_terminal = is_deadlock or goal_reached or ((curr.depth + 1) >= cfg.max_depth)

                child_actions = (
                    self._rank_and_filter_causal_actions(
                        candidate_actions, goal, cfg.causal_pruning
                    )
                    if not is_terminal
                    else []
                )
                child_node = MCTSNode(
                    node_id=f"mcts_{uuid.uuid4().hex[:6]}",
                    state_branch=child_branch,
                    action=action,
                    parent=curr,
                    unexpanded_actions=child_actions,
                    depth=curr.depth + 1,
                    is_terminal=is_terminal,
                    step_utility=step_utility,
                    receipt=receipt,
                )
                curr.children.append(child_node)
                leaf = child_node
            else:
                leaf = curr

            # 3. Rollout / Evaluation
            if leaf.is_terminal or leaf.depth >= cfg.max_depth:
                rollout_val = leaf.step_utility
            else:
                rollout_val = leaf.step_utility
                curr_discount = cfg.discount_factor
                for _ in range(leaf.depth + 1, cfg.max_depth + 1):
                    best_next = -999.0
                    for cand in candidate_actions:
                        next_p = self._world_kernel.predict(
                            action=cand, confidence=0.85, author=author
                        )
                        u = self._compute_action_utility(cand, next_p, True)
                        if u > best_next:
                            best_next = u
                    if best_next <= 0.005:
                        rollout_val = 0.001
                        break
                    rollout_val += curr_discount * best_next
                    curr_discount *= cfg.discount_factor

            # 4. Backpropagation
            b_node: MCTSNode | None = leaf
            val = rollout_val
            while b_node is not None:
                b_node.visit_count += 1
                b_node.total_value += val
                val *= cfg.discount_factor
                b_node = b_node.parent

        if not root.children:
            raise ValueError("MCTS could not expand any candidate action")

        best_child = max(root.children, key=lambda c: (c.average_value(), c.visit_count))
        logger.info(
            "MCTS planner selected candidate '%s' (visits=%d, avg_q=%.3f, branch '%s')",
            best_child.action.intent if best_child.action else "unknown",
            best_child.visit_count,
            best_child.average_value(),
            best_child.state_branch,
        )

        # Merge best branch and clean up all other simulation branches
        self._workspace.merge_branch(best_child.state_branch)
        for b in all_created_branches:
            if b != best_child.state_branch:
                self._workspace.drop_branch(b)

        assert best_child.action is not None
        return CandidatePlanResult(
            candidate_id=best_child.action.id,
            action=best_child.action,
            branch_name=best_child.state_branch,
            receipt=best_child.receipt or ExecutionReceipt(author=author, success=True),
            utility_score=best_child.average_value(),
        )

    async def evaluate_multimodal_plan(
        self,
        goal: GoalNode,
        candidate_actions: list[ActionNode],
        active_affordances: list[str] | None = None,
        horizon: int = 1,
        beam_width: int = 2,
        author: str = "counterfactual_planner",
        use_mcts: bool | None = None,
        mcts_config: MCTSConfig | None = None,
    ) -> CandidatePlanResult:
        """Evaluate actions across multiple effector modalities (LOCOMOTION, MANIPULATION, TARGETING).

        Filters and prioritizes actions according to environmental affordance context:
        - If manipulation affordance is present ('CAN_MANIPULATE', 'ADJACENT_OBJECT'), allows manipulation actions.
        - If targeting affordance is present ('CAN_TARGET', 'CLICKABLE'), allows targeting actions.
        - Otherwise, prioritizes locomotion to reach intermediate waypoints.
        """
        active_affordances = active_affordances or []
        filtered_candidates: list[ActionNode] = []

        can_manipulate = any(
            a in active_affordances for a in ("CAN_MANIPULATE", "ADJACENT_OBJECT", "HOLDING_ITEM")
        )
        can_target = any(
            a in active_affordances for a in ("CAN_TARGET", "CLICKABLE", "REMOTE_TRIGGER")
        )

        for act in candidate_actions:
            modality = getattr(act, "modality", ActionModality.COGNITIVE)
            if modality == ActionModality.MANIPULATION and not can_manipulate:
                continue
            if modality == ActionModality.TARGETING and not can_target:
                continue
            filtered_candidates.append(act)

        if not filtered_candidates:
            filtered_candidates = candidate_actions

        return await self.evaluate_and_select(
            goal=goal,
            candidate_actions=filtered_candidates,
            horizon=horizon,
            beam_width=beam_width,
            author=author,
            use_mcts=use_mcts,
            mcts_config=mcts_config,
        )

    @staticmethod
    def _rank_and_filter_causal_actions(
        candidate_actions: list[ActionNode],
        goal: GoalNode | None,
        causal_pruning: bool = True,
    ) -> list[ActionNode]:
        """Prioritize actions advancing unsatisfied causal conditions in MCTS tree nodes.

        Ranks actions matching target conditions or active subgoals first,
        suppressing motor noise and pruning the branching factor.
        """
        if not causal_pruning or not candidate_actions or goal is None:
            return list(candidate_actions)

        target_conditions: list[str] = []
        if hasattr(goal, "properties") and goal.properties:
            tc = goal.properties.get("target_conditions", [])
            if isinstance(tc, list):
                target_conditions.extend([str(c).strip() for c in tc])
            elif isinstance(tc, str):
                target_conditions.append(tc.strip())
        if hasattr(goal, "description") and goal.description:
            target_conditions.append(goal.description.strip())

        causal_actions: list[ActionNode] = []
        regular_actions: list[ActionNode] = []

        for act in candidate_actions:
            is_causal = False
            for p in getattr(act, "produces", []):
                p_clean = p.strip()
                if any(p_clean in tc or tc in p_clean for tc in target_conditions):
                    is_causal = True
                    break
            if not is_causal and any(act.intent in tc for tc in target_conditions):
                is_causal = True

            if is_causal:
                causal_actions.append(act)
            else:
                regular_actions.append(act)

        return causal_actions + regular_actions

    @staticmethod
    def _compute_action_utility(
        action: ActionNode,
        prediction: Any,
        success: bool,
        goal: GoalNode | None = None,
    ) -> float:
        """Compute utility score from action cost, prediction confidence, spatial outcomes, modality, and causal progress."""
        base_utility = (
            prediction.uncertainty.confidence
            * (1.0 if success else 0.0)
            * (1.0 / (1.0 + getattr(action, "estimated_cost", 0) * 0.01))
        )

        # Modality prioritization bonus
        modality = getattr(action, "modality", ActionModality.COGNITIVE)
        if modality == ActionModality.MANIPULATION and action.properties.get("can_interact"):
            base_utility += 1.5
        elif modality == ActionModality.TARGETING and action.properties.get("target_in_range"):
            base_utility += 1.2

        # Macro-action efficiency bonus
        if getattr(action, "properties", {}).get("is_macro"):
            base_utility += 1.0

        # Causal condition alignment bonus
        if goal is not None and getattr(action, "produces", None):
            goal_desc = getattr(goal, "description", "")
            goal_props = getattr(goal, "properties", {}) or {}
            target_conds = goal_props.get("target_conditions", [])
            if isinstance(target_conds, str):
                target_conds = [target_conds]
            for p in action.produces:
                if (p and p in goal_desc) or any(p in tc for tc in target_conds):
                    base_utility += 2.0
                    break

        # Check action properties for explicit mock / override first
        act_spatial = None
        if getattr(action, "properties", None):
            act_pred = action.properties.get("predicted_state", {})
            if isinstance(act_pred, dict) and "spatial_outcome" in act_pred:
                act_spatial = act_pred["spatial_outcome"]
            elif "spatial_outcome" in action.properties:
                act_spatial = action.properties["spatial_outcome"]

        pred_props = getattr(prediction, "properties", {}) or {}
        predicted_state = pred_props.get("predicted_state") or {}
        spatial_outcome = None
        if isinstance(predicted_state, dict):
            spatial_outcome = predicted_state.get("spatial_outcome")
            if (
                spatial_outcome is None
                and "predicted_state" in predicted_state
                and isinstance(predicted_state["predicted_state"], dict)
            ):
                spatial_outcome = predicted_state["predicted_state"].get("spatial_outcome")

        # Prefer explicit action-specified spatial outcome if present
        if act_spatial is not None:
            spatial_outcome = act_spatial

        if spatial_outcome and isinstance(spatial_outcome, dict):
            deadlock = spatial_outcome.get("deadlock", False)
            collision = spatial_outcome.get("collision", False)
            progress = spatial_outcome.get("progress", 0.0)
            goal_reached = spatial_outcome.get("goal_reached", False)

            if deadlock:
                return 0.001
            elif collision:
                return base_utility * 0.25
            elif goal_reached:
                return base_utility + 10.0
            else:
                progress_boost = max(-0.5, min(2.0, progress * 0.5))
                return base_utility + progress_boost

        return base_utility
