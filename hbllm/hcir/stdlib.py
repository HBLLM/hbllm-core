"""HCIR Standard Library — pure multi-opcode ISA v2 cognitive streams."""

from __future__ import annotations

import uuid
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import BeliefNode, ConceptNode, GoalNode, SkillNode


def _ins(op: Opcode, cost: int = 0, /, **params: Any) -> Instruction:
    return Instruction(opcode=op, params=params, cost_estimate=cost)


def _ret(*regs: str, label: str | None = None) -> Instruction:
    return Instruction(opcode=Opcode.RETURN, params={"yield": list(regs)}, label=label)


def _uid(prefix: str, n: int = 8) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:n]}"


class MemoryOps:
    """Standard memory operations: search, store, forget."""

    @staticmethod
    def search(
        text: str = "",
        node_type: str | None = None,
        category: str | None = None,
        limit: int = 100,
        author: str = "stdlib",
    ) -> InstructionStream:
        """Search the cognitive graph and bind matches to `$r0`."""
        params: dict[str, Any] = {"limit": limit, "bind": "$r0"}
        if text:
            params["text_contains"] = text
        if node_type:
            params["node_type"] = node_type
        if category:
            params["category"] = category
        return InstructionStream(
            author=author,
            description=f"stdlib.memory.search({text!r})",
            instructions=[Instruction(opcode=Opcode.QUERY, params=params, cost_estimate=10)],
        )

    @staticmethod
    def store(node_data: dict[str, Any], author: str = "stdlib") -> InstructionStream:
        """Store a new node in the cognitive graph and bind its ID to `$r0`."""
        return InstructionStream(
            author=author,
            description="stdlib.memory.store",
            instructions=[_ins(Opcode.ASSERT, 5, node_data=node_data, author=author, bind="$r0")],
        )

    @staticmethod
    def forget(node_id: str, author: str = "stdlib") -> InstructionStream:
        """Retract a node from the cognitive graph."""
        return InstructionStream(
            author=author,
            description=f"stdlib.memory.forget({node_id})",
            instructions=[_ins(Opcode.RETRACT, 5, node_id=node_id, author=author)],
        )


class ReasoningOps:
    """Standard reasoning operations compiled into `QUERY -> UNIFY -> BRANCH -> ASSERT -> RETURN`."""

    @staticmethod
    def compare(subject_a: str, subject_b: str, author: str = "stdlib") -> InstructionStream:
        """Compare two subjects across working memory and assert a comparison BeliefNode."""
        cmp_node = BeliefNode(
            id=_uid("cmp"),
            claim=f"Comparison recorded between {subject_a} and {subject_b}",
            properties={"subject_a": subject_a, "subject_b": subject_b},
        ).model_dump()
        return InstructionStream(
            author=author,
            description=f"stdlib.reasoning.compare({subject_a}, {subject_b})",
            instructions=[
                _ins(Opcode.QUERY, 10, text_contains=subject_a, bind="$r0"),
                _ins(Opcode.QUERY, 10, text_contains=subject_b, bind="$r1"),
                _ins(Opcode.UNIFY, 10, pattern={"var": "?a"}, source="$r0", bind="$r2"),
                _ins(Opcode.ASSERT, 5, node_data=cmp_node, author=author, bind="$r3"),
                _ret("$r3"),
            ],
        )

    @staticmethod
    def evaluate(
        node_id: str, criteria: list[str] | None = None, author: str = "stdlib"
    ) -> InstructionStream:
        """Unify target node against active constraints and assert an evaluation BeliefNode."""
        eval_node = BeliefNode(
            id=_uid("eval"),
            claim=f"Evaluation of {node_id} against constraints",
            properties={"target_id": node_id, "criteria": criteria or []},
        ).model_dump()
        return InstructionStream(
            author=author,
            description=f"stdlib.reasoning.evaluate({node_id})",
            instructions=[
                _ins(Opcode.UNIFY, 10, pattern={"id": node_id}, bind="$r0"),
                _ins(Opcode.QUERY, 10, node_type="constraint", bind="$r1"),
                _ins(
                    Opcode.BRANCH, 0, condition="if_empty", register="$r0", target_label="eval_done"
                ),
                _ins(Opcode.ASSERT, 5, node_data=eval_node, author=author, bind="$r2"),
                _ret("$r2", label="eval_done"),
            ],
        )


class PlanningOps:
    """Standard planning operations compiled into `ASSERT -> QUERY -> UNIFY -> BRANCH -> CALL`."""

    @staticmethod
    def decompose(
        goal_description: str, max_subtasks: int = 5, priority: float = 0.5, author: str = "stdlib"
    ) -> InstructionStream:
        """Assert goal, inspect constraints, and invoke matching procedural skill via `Opcode.CALL`."""
        goal_data = GoalNode(
            id=_uid("goal"),
            description=goal_description,
            priority=priority,
            properties={"max_subtasks": max_subtasks},
        ).model_dump()
        return InstructionStream(
            author=author,
            description=f"stdlib.planning.decompose({goal_description!r})",
            instructions=[
                _ins(Opcode.ASSERT, 5, node_data=goal_data, author=author, bind="$r0"),
                _ins(Opcode.QUERY, 10, node_type="constraint", bind="$r1"),
                _ins(Opcode.UNIFY, 10, pattern={"node_type": "skill"}, bind="$r2"),
                _ins(
                    Opcode.BRANCH,
                    0,
                    condition="if_empty",
                    register="$r2",
                    target_label="plan_ready",
                ),
                _ins(Opcode.CALL, 15, skill_id="$r2", args={"target": "$r0"}, bind="$r3"),
                _ret("$r0", label="plan_ready"),
            ],
        )

    @staticmethod
    def prioritize(goal_ids: list[str], author: str = "stdlib") -> InstructionStream:
        """Query active goal nodes, unify highest-confidence goal into `$r1`, and return list."""
        return InstructionStream(
            author=author,
            description="stdlib.planning.prioritize",
            instructions=[
                _ins(Opcode.QUERY, 10, node_type="goal", bind="$r0"),
                _ins(Opcode.UNIFY, 10, pattern={"node_type": "goal"}, source="$r0", bind="$r1"),
                _ret("$r0"),
            ],
        )


class LearningOps:
    """Standard learning operations producing persistent homoiconic Skill/Concept nodes."""

    @staticmethod
    def extract_skill(episode_id: str, author: str = "stdlib") -> InstructionStream:
        """Query episode and assert a parameterized homoiconic `SkillNode`."""
        inner = InstructionStream(
            author=author,
            instructions=[
                _ins(Opcode.UNIFY, 0, pattern={"id": "?target"}, bind="$r0"),
                _ret("$r0"),
            ],
        ).model_dump()
        skill_data = SkillNode(
            id=_uid("skill"),
            skill_name=f"extracted_{episode_id}",
            description=f"Extracted from episode {episode_id}",
            parameters=[{"name": "target", "required": False, "default": episode_id}],
            postcondition_pattern={"episode_id": episode_id, "target": "?target"},
            bytecode_stream=inner,
        ).model_dump()
        return InstructionStream(
            author=author,
            description=f"stdlib.learning.extract_skill({episode_id})",
            instructions=[
                _ins(Opcode.QUERY, 10, text_contains=episode_id, node_type="episode", bind="$r0"),
                _ins(Opcode.BRANCH, 0, condition="if_empty", register="$r0", target_label="no_ep"),
                _ins(Opcode.ASSERT, 10, node_data=skill_data, author=author, bind="$r1"),
                _ret("$r1", label="no_ep"),
            ],
        )

    @staticmethod
    def consolidate(
        memory_type: str = "episodic", max_age_hours: int = 24, author: str = "stdlib"
    ) -> InstructionStream:
        """Query candidate memories and assert a consolidated ConceptNode."""
        query_node_type = "episode" if memory_type == "episodic" else memory_type
        concept_data = ConceptNode(
            id=_uid("consolidated"),
            label=f"consolidated_{memory_type}",
            definition=f"Consolidated summary of {memory_type} nodes",
            properties={"memory_type": memory_type, "max_age_hours": max_age_hours},
        ).model_dump()
        return InstructionStream(
            author=author,
            description=f"stdlib.learning.consolidate({memory_type})",
            instructions=[
                _ins(Opcode.QUERY, 20, node_type=query_node_type, category="memory", bind="$r0"),
                _ins(Opcode.BRANCH, 0, condition="if_empty", register="$r0", target_label="done"),
                _ins(Opcode.ASSERT, 15, node_data=concept_data, author=author, bind="$r1"),
                _ret("$r1", label="done"),
            ],
        )


def _sim_stream(claim: str, *, key: str, commit: bool, desc: str, author: str) -> InstructionStream:
    branch = _uid("hyp" if commit else "cf", 6)
    node = {"id": _uid("hyp" if commit else "cf"), "node_type": "hypothesis", "claim": claim}
    end_op = Opcode.MERGE if commit else Opcode.ROLLBACK
    return InstructionStream(
        author=author,
        description=desc,
        instructions=[
            _ins(Opcode.FORK, 20, branch_name=branch),
            _ins(Opcode.ASSERT, 5, node_data=node, author=author, branch_name=branch, bind="$r0"),
            _ins(
                Opcode.EXECUTE,
                50,
                capability="world.predict_outcome",
                params={key: claim, "target": "$r0", "branch": branch},
                bind="$r1",
            ),
            _ins(end_op, 20 if commit else 10, branch_name=branch),
            _ret("$r0"),
        ],
    )


class SimulationOps:
    """Standard counterfactual simulation macros over copy-on-write branches."""

    @staticmethod
    def hypothesize(hypothesis: str, author: str = "stdlib") -> InstructionStream:
        """FORK -> ASSERT(hypothesis) -> EXECUTE(world.predict_outcome) -> MERGE -> RETURN."""
        return _sim_stream(
            hypothesis,
            key="hypothesis",
            commit=True,
            desc=f"stdlib.simulation.hypothesize({hypothesis!r})",
            author=author,
        )

    @staticmethod
    def counterfactual(scenario: str, author: str = "stdlib") -> InstructionStream:
        """FORK -> ASSERT(scenario) -> EXECUTE(world.predict_outcome) -> ROLLBACK -> RETURN."""
        return _sim_stream(
            scenario,
            key="scenario",
            commit=False,
            desc=f"stdlib.simulation.counterfactual({scenario!r})",
            author=author,
        )


class stdlib:  # noqa: N801
    """HCIR Standard Library — ``hcir.stdlib.*`` namespace."""

    memory = MemoryOps
    reasoning = ReasoningOps
    planning = PlanningOps
    learning = LearningOps
    simulation = SimulationOps
