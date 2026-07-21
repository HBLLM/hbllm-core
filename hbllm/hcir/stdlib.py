"""
HCIR Standard Library — high-level cognitive macros built on bytecodes.

Like libc for C or the Python standard library, ``hcir.stdlib``
provides composable cognitive operations implemented as sequences
of the 8 primitive opcodes.

    hcir.stdlib.memory.search()     → QUERY
    hcir.stdlib.reasoning.compare() → QUERY + QUERY + ASSERT
    hcir.stdlib.planning.decompose() → QUERY + ASSERT + ASSERT + ASSERT
    hcir.stdlib.learning.extract()  → QUERY + ASSERT + EXECUTE + ASSERT
    hcir.stdlib.simulation.hypothesize() → FORK + ASSERT + EXECUTE + MERGE

Each macro returns an ``InstructionStream`` that can be composed,
optimized, or passed directly to the interpreter.
"""

from __future__ import annotations

import uuid
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import GoalNode

# ═══════════════════════════════════════════════════════════════════════════
# Memory Operations
# ═══════════════════════════════════════════════════════════════════════════


class MemoryOps:
    """Standard memory operations: search, store, recall, forget."""

    @staticmethod
    def search(
        text: str = "",
        node_type: str | None = None,
        category: str | None = None,
        limit: int = 100,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Search the cognitive graph for matching nodes."""
        params: dict[str, Any] = {"limit": limit}
        if text:
            params["text_contains"] = text
        if node_type:
            params["node_type"] = node_type
        if category:
            params["category"] = category

        return InstructionStream(
            author=author,
            description=f"stdlib.memory.search({text!r})",
            instructions=[
                Instruction(opcode=Opcode.QUERY, params=params, cost_estimate=10),
            ],
        )

    @staticmethod
    def store(
        node_data: dict[str, Any],
        author: str = "stdlib",
    ) -> InstructionStream:
        """Store a new node in the cognitive graph."""
        return InstructionStream(
            author=author,
            description="stdlib.memory.store",
            instructions=[
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={"node_data": node_data, "author": author},
                    cost_estimate=5,
                ),
            ],
        )

    @staticmethod
    def forget(
        node_id: str,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Retract a node from the cognitive graph."""
        return InstructionStream(
            author=author,
            description=f"stdlib.memory.forget({node_id})",
            instructions=[
                Instruction(
                    opcode=Opcode.RETRACT,
                    params={"node_id": node_id, "author": author},
                    cost_estimate=5,
                ),
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Reasoning Operations
# ═══════════════════════════════════════════════════════════════════════════


class ReasoningOps:
    """Standard reasoning operations: compare, infer, evaluate."""

    @staticmethod
    def compare(
        subject_a: str,
        subject_b: str,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Compare two subjects by querying both, then asserting a comparison result.

        Macro: QUERY(A) + QUERY(B) + EXECUTE(comparison)
        """
        return InstructionStream(
            author=author,
            description=f"stdlib.reasoning.compare({subject_a}, {subject_b})",
            instructions=[
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"text_contains": subject_a},
                    cost_estimate=10,
                ),
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"text_contains": subject_b},
                    cost_estimate=10,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "reasoning.compare",
                        "params": {"a": subject_a, "b": subject_b},
                    },
                    cost_estimate=50,
                ),
            ],
        )

    @staticmethod
    def evaluate(
        node_id: str,
        criteria: list[str] | None = None,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Evaluate a node against criteria.

        Macro: QUERY(node) + QUERY(criteria) + EXECUTE(evaluation)
        """
        return InstructionStream(
            author=author,
            description=f"stdlib.reasoning.evaluate({node_id})",
            instructions=[
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"text_contains": node_id},
                    cost_estimate=10,
                ),
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"node_type": "constraint"},
                    cost_estimate=10,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "reasoning.evaluate",
                        "params": {"node_id": node_id, "criteria": criteria or []},
                    },
                    cost_estimate=50,
                ),
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Planning Operations
# ═══════════════════════════════════════════════════════════════════════════


class PlanningOps:
    """Standard planning operations: decompose, prioritize, schedule."""

    @staticmethod
    def decompose(
        goal_description: str,
        max_subtasks: int = 5,
        priority: float = 0.5,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Decompose a goal into subtasks.

        Macro: ASSERT(goal) + QUERY(constraints) + EXECUTE(planner)
        """
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        return InstructionStream(
            author=author,
            description=f"stdlib.planning.decompose({goal_description!r})",
            instructions=[
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={
                        "node_data": GoalNode(
                            id=goal_id,
                            description=goal_description,
                            priority=priority,
                        ).model_dump(),
                        "author": author,
                    },
                    cost_estimate=5,
                ),
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"node_type": "constraint"},
                    cost_estimate=10,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "planning.decompose",
                        "params": {
                            "goal_id": goal_id,
                            "max_subtasks": max_subtasks,
                        },
                    },
                    cost_estimate=100,
                ),
            ],
        )

    @staticmethod
    def prioritize(
        goal_ids: list[str],
        author: str = "stdlib",
    ) -> InstructionStream:
        """Re-prioritize a set of goals.

        Macro: QUERY(goals) + EXECUTE(prioritizer)
        """
        return InstructionStream(
            author=author,
            description="stdlib.planning.prioritize",
            instructions=[
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"node_type": "goal"},
                    cost_estimate=10,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "planning.prioritize",
                        "params": {"goal_ids": goal_ids},
                    },
                    cost_estimate=50,
                ),
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Learning Operations
# ═══════════════════════════════════════════════════════════════════════════


class LearningOps:
    """Standard learning operations: extract_skill, consolidate, reinforce."""

    @staticmethod
    def extract_skill(
        episode_id: str,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Extract a reusable skill from an episode.

        Macro: QUERY(episode) + EXECUTE(skill_extraction) + ASSERT(skill)
        """
        return InstructionStream(
            author=author,
            description=f"stdlib.learning.extract_skill({episode_id})",
            instructions=[
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"text_contains": episode_id, "node_type": "episode"},
                    cost_estimate=10,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "learning.extract_skill",
                        "params": {"episode_id": episode_id},
                    },
                    cost_estimate=100,
                ),
            ],
        )

    @staticmethod
    def consolidate(
        memory_type: str = "episodic",
        max_age_hours: int = 24,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Consolidate recent memories into long-term knowledge.

        Macro: QUERY(recent_episodes) + EXECUTE(consolidation) + ASSERT(knowledge)
        """
        return InstructionStream(
            author=author,
            description=f"stdlib.learning.consolidate({memory_type})",
            instructions=[
                Instruction(
                    opcode=Opcode.QUERY,
                    params={"node_type": memory_type, "category": "memory"},
                    cost_estimate=20,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "learning.consolidate",
                        "params": {"memory_type": memory_type, "max_age_hours": max_age_hours},
                    },
                    cost_estimate=150,
                ),
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Operations
# ═══════════════════════════════════════════════════════════════════════════


class SimulationOps:
    """Standard simulation operations: hypothesize, counterfactual."""

    @staticmethod
    def hypothesize(
        hypothesis: str,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Test a hypothesis via simulation.

        Macro: FORK → ASSERT(hypothesis) → EXECUTE(test) → MERGE
        """
        branch = f"hyp_{uuid.uuid4().hex[:6]}"
        return InstructionStream(
            author=author,
            description=f"stdlib.simulation.hypothesize({hypothesis!r})",
            instructions=[
                Instruction(
                    opcode=Opcode.FORK,
                    params={"branch_name": branch},
                    cost_estimate=20,
                ),
                Instruction(
                    opcode=Opcode.ASSERT,
                    params={
                        "node_data": {
                            "id": f"hyp_{uuid.uuid4().hex[:8]}",
                            "node_type": "hypothesis",
                            "claim": hypothesis,
                        },
                        "author": author,
                    },
                    cost_estimate=5,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "simulation.test_hypothesis",
                        "params": {"hypothesis": hypothesis, "branch": branch},
                    },
                    cost_estimate=200,
                ),
                Instruction(
                    opcode=Opcode.MERGE,
                    params={"branch_name": branch},
                    cost_estimate=20,
                ),
            ],
        )

    @staticmethod
    def counterfactual(
        scenario: str,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Run a "what if" counterfactual scenario.

        Macro: FORK → EXECUTE(scenario) → ROLLBACK (no merge — just observe)
        """
        branch = f"cf_{uuid.uuid4().hex[:6]}"
        return InstructionStream(
            author=author,
            description=f"stdlib.simulation.counterfactual({scenario!r})",
            instructions=[
                Instruction(
                    opcode=Opcode.FORK,
                    params={"branch_name": branch},
                    cost_estimate=20,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "simulation.counterfactual",
                        "params": {"scenario": scenario, "branch": branch},
                    },
                    cost_estimate=200,
                ),
                # Don't merge — just observe what would happen
                Instruction(
                    opcode=Opcode.ROLLBACK,
                    params={"target_version": 0},  # Reset simulation
                    cost_estimate=10,
                ),
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: namespace container
# ═══════════════════════════════════════════════════════════════════════════


class stdlib:  # noqa: N801
    """HCIR Standard Library — ``hcir.stdlib.*`` namespace.

    Usage::

        from hbllm.hcir.stdlib import stdlib

        stream = stdlib.memory.search("battery temperature")
        stream = stdlib.planning.decompose("Build solar dehydrator")
        stream = stdlib.simulation.hypothesize("Copper tubing is more efficient")
    """

    memory = MemoryOps
    reasoning = ReasoningOps
    planning = PlanningOps
    learning = LearningOps
    simulation = SimulationOps
