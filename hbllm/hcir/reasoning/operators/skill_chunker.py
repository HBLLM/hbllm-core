"""Skill Macro-Chunking and Subroutine Compilation Engine for HCIR.

Detects repeated, successful action sub-paths and compiles them into composite
MacroActionNodes with analytically synthesized composite preconditions and net effects:

    Sub-Action Sequence:
        a1: navigate_to(key_gold)  [Pre: visible(key_gold), Prod: adjacent_to(agent, key_gold)]
        a2: pickup(key_gold)       [Pre: adjacent_to(agent, key_gold), Prod: holds(agent, key_gold)]
            ↓
    Compiled Macro-Action:
        Macro: AcquireObject(key_gold)
        Composite Pre:  [visible(key_gold)]  (adjacent_to resolved internally by a1)
        Composite Prod: [holds(agent, key_gold)]
        Sub-actions:    [a1, a2]

Reduces Counterfactual MCTS search branching depth from exponential to logarithmic.
Independence Level: L1 (100% deterministic, 0 LLM tokens).
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import (
    ActionModality,
    ActionNode,
    CognitiveCategory,
    HCIRNodeType,
    SkillNode,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope

logger = logging.getLogger(__name__)


class MacroActionNode(ActionNode):
    """Composite action encapsulating an ordered sequence of primitive ActionNodes."""

    node_type: HCIRNodeType = HCIRNodeType.ACTION
    category: CognitiveCategory = CognitiveCategory.EXECUTION
    intent: str = "macro_action"
    modality: ActionModality = ActionModality.MANIPULATION

    # Sub-actions in execution order
    sub_actions: list[ActionNode] = Field(default_factory=list)
    macro_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_pointer: int = 0
    composite_negates: list[str] = Field(default_factory=list)

    @property
    def is_completed(self) -> bool:
        return self.step_pointer >= len(self.sub_actions)

    @property
    def current_action(self) -> ActionNode | None:
        if self.step_pointer < len(self.sub_actions):
            return self.sub_actions[self.step_pointer]
        return None

    def advance(self) -> ActionNode | None:
        """Advance the macro execution pointer and return next action."""
        if self.step_pointer < len(self.sub_actions):
            act = self.sub_actions[self.step_pointer]
            self.step_pointer += 1
            return act
        return None

    def reset(self) -> None:
        self.step_pointer = 0


@dataclass
class SkillTrace:
    """Historical execution trace for skill discovery."""

    trace_id: str
    actions: list[ActionNode]
    goal_achieved: str
    total_cost: float
    success: bool


_ARG_RE = re.compile(r"^([a-zA-Z0-9_]+)\(([^)]+)\)$")


def _parse_intent(intent: str) -> tuple[str, str]:
    m = _ARG_RE.match(intent.strip())
    return (m.group(1), m.group(2).strip()) if m else (intent.strip(), "")


class SkillMacroChunker:
    """Discovers, composes, and registers MacroActionNodes and homoiconic SkillNodes."""

    def __init__(self, min_trace_occurrences: int = 2) -> None:
        self.min_trace_occurrences = min_trace_occurrences
        self.recorded_traces: list[SkillTrace] = []
        self.compiled_macros: dict[str, MacroActionNode] = {}
        self.compiled_skill_nodes: dict[str, SkillNode] = {}

    def record_trace(
        self,
        actions: list[ActionNode],
        goal_achieved: str,
        success: bool = True,
        total_cost: float = 0.0,
    ) -> None:
        """Record an executed action trace toward an achieved goal."""
        trace = SkillTrace(
            trace_id=str(uuid.uuid4())[:8],
            actions=list(actions),
            goal_achieved=goal_achieved,
            total_cost=total_cost,
            success=success,
        )
        self.recorded_traces.append(trace)

    def compile_sequence(
        self,
        actions: list[ActionNode],
        macro_name: str = "",
        goal_target: str = "",
    ) -> MacroActionNode:
        """Analytically synthesize composite preconditions and net effects for an action sequence."""
        if not actions:
            raise ValueError("Cannot compile empty action sequence into a MacroActionNode")

        # 1. Synthesize Composite Requirements:
        # Pre(M) = Pre(a1) U (Pre(a2) \ Prod(a1)) U (Pre(a3) \ (Prod(a1) U Prod(a2))) ...
        composite_reqs: list[str] = []
        produced_so_far: set[str] = set()
        negated_so_far: set[str] = set()
        total_cost: float = 0.0

        for act in actions:
            # Check requirements not yet produced by prior actions in sequence
            for req in act.requirements:
                if req not in produced_so_far and req not in composite_reqs:
                    composite_reqs.append(req)

            # Update produced set
            for prod in act.produces:
                produced_so_far.add(prod)
                negated_so_far.discard(prod)

            # Check explicit negations or properties
            drops = act.properties.get("drops")
            if drops:
                negated_so_far.add(f"holds({drops})")
            for neg in act.properties.get("negates", []):
                negated_so_far.add(neg)

            total_cost += float(act.estimated_cost or 1.0)

        # 2. Net produced effects: produced_so_far \ negated_so_far
        net_produces = [p for p in produced_so_far if p not in negated_so_far]

        name = macro_name or f"Macro_{actions[0].intent}_to_{actions[-1].intent}"
        macro_id = f"macro_{str(uuid.uuid4())[:8]}"

        macro = MacroActionNode(
            id=macro_id,
            macro_id=macro_id,
            intent=name,
            sub_actions=list(actions),
            requirements=composite_reqs,
            produces=net_produces,
            composite_negates=list(negated_so_far),
            estimated_cost=int(total_cost),
            properties={
                "goal_target": goal_target,
                "step_count": len(actions),
                "is_macro": True,
            },
        )
        return macro

    def discover_and_compile_skills(self) -> list[MacroActionNode]:
        """Scan successful traces, find repeated sequences (n >= min_occurrences), and compile."""
        success_traces = [t for t in self.recorded_traces if t.success and len(t.actions) >= 2]
        sequence_counts: dict[tuple[str, ...], list[list[ActionNode]]] = {}

        for t in success_traces:
            intents = tuple(a.intent for a in t.actions)
            if intents not in sequence_counts:
                sequence_counts[intents] = []
            sequence_counts[intents].append(t.actions)

        newly_compiled: list[MacroActionNode] = []
        for intents, action_lists in sequence_counts.items():
            if len(action_lists) >= self.min_trace_occurrences:
                key = " -> ".join(intents)
                if key not in self.compiled_macros:
                    macro = self.compile_sequence(
                        actions=action_lists[0],
                        macro_name=f"Skill({intents[0]}..{intents[-1]})",
                    )
                    self.compiled_macros[key] = macro
                    newly_compiled.append(macro)
                    logger.info(
                        "Compiled recurring skill: %s (occurred %d times)", key, len(action_lists)
                    )

        return newly_compiled

    def compile_to_skill_node(
        self,
        macro: MacroActionNode,
        *,
        tenant_id: str = "default",
        support_count: int = 2,
    ) -> SkillNode:
        """Compile a `MacroActionNode` into a parameterized homoiconic `SkillNode`."""
        arg_memo: dict[str, str] = {}
        params_meta: list[dict[str, Any]] = []
        instructions: list[Instruction] = []
        for act in macro.sub_actions:
            verb, raw_arg = _parse_intent(act.intent)
            if raw_arg:
                var = arg_memo.setdefault(raw_arg, f"?target_{len(arg_memo)}")
                if len(params_meta) < len(arg_memo):
                    params_meta.append({"name": var[1:], "required": False, "default": None})
            else:
                var = "?target_0"
            reg = f"$r{min(7, max(0, len(params_meta) - 1))}"
            cap = (
                "motor.navigate" if act.modality == ActionModality.LOCOMOTION else "motor.interact"
            )
            if not instructions:
                instructions.append(
                    Instruction(
                        opcode=Opcode.UNIFY,
                        params={"pattern": {"id": var}, "bind": reg},
                        cost_estimate=5,
                    )
                )
            instructions.append(
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={"capability": cap, "params": {"target": var, "verb": verb}},
                    cost_estimate=15,
                )
            )
        instructions.append(Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}))
        stream = InstructionStream(
            author="skill_chunker", description=macro.intent, instructions=instructions
        )
        node = SkillNode(
            id=f"skill_{macro.macro_id}",
            skill_name=macro.intent,
            description=f"Parameterized macro skill '{macro.intent}'",
            parameters=params_meta,
            precondition_pattern={"requirements": list(macro.requirements)},
            postcondition_pattern={"produces": list(macro.produces), "skill_name": macro.intent},
            bytecode_stream=stream.model_dump(),
            alpha_success=1.0 + float(support_count),
            beta_failure=1.0,
            invocation_count=support_count,
            provenance=Provenance(created_by="skill_chunker"),
            scope=Scope(tenant_id=tenant_id),
            tags=["macro_chunker", macro.intent],
        )
        self.compiled_skill_nodes[node.id] = node
        return node

    def discover_homoiconic_skills(self, tenant_id: str = "default") -> list[SkillNode]:
        """Anti-unify repeated traces ignoring concrete entity args and return homoiconic SkillNodes."""
        grouped_traces: dict[tuple[str, ...], list[list[ActionNode]]] = {}
        for trace in self.recorded_traces:
            if trace.success and len(trace.actions) >= 2:
                verbs = tuple(_parse_intent(act.intent)[0] for act in trace.actions)
                grouped_traces.setdefault(verbs, []).append(trace.actions)
        discovered: list[SkillNode] = []
        for verbs, group in grouped_traces.items():
            key = f"skill_chunk_{' -> '.join(verbs)}"
            if len(group) < self.min_trace_occurrences or key in self.compiled_skill_nodes:
                continue
            macro = self.compile_sequence(group[0], macro_name=f"Skill({verbs[0]}..{verbs[-1]})")
            node = self.compile_to_skill_node(macro, tenant_id=tenant_id, support_count=len(group))
            self.compiled_skill_nodes[key] = node
            discovered.append(node)
        return discovered

    def propose_skill_transactions(
        self, skills: Sequence[SkillNode] | None = None, *, tenant_id: str = "default"
    ) -> list[TransactionOperation]:
        """Emit atomic HCIR transactions for discovered homoiconic SkillNodes."""
        nodes = (
            list(skills)
            if skills is not None
            else self.discover_homoiconic_skills(tenant_id=tenant_id)
        )
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node.id,
                node_data=node.model_dump(),
            )
            for node in nodes
        ]
