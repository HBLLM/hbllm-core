"""
HCIR Compiler Frontend — Natural Language → HCIR Bytecode.

Pipeline::

    Natural Language
          ↓
    Semantic AST          (intent extraction)
          ↓
    Intent Graph          (goal/constraint decomposition)
          ↓
    Constraint Injection  (policy/scope/resource bounds)
          ↓
    HCIR IR               (graph operations)
          ↓
    Bytecode              (instruction stream)
          ↓
    Scheduler             (dispatch)

The compiler is itself a cognitive node — it receives user
intent and produces executable HCIR bytecode.  LLM-backed
compilation is one strategy; rule-based compilation is another.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import (
    GoalNode,
)
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.dialogue import (
    ActionIntentProposal,
    DialogueInterpretationResult,
)
from hbllm.hcir.grounding.contracts.enums import (
    DecisionOutcome,
    DialogueAct,
)
from hbllm.hcir.grounding.contracts.references import (
    GroundedEntityRef,
    UnresolvedRef,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Semantic AST — intermediate representation of user intent
# ═══════════════════════════════════════════════════════════════════════════


class IntentType(StrEnum):
    """High-level intent categories extracted from natural language."""

    INVESTIGATE = "investigate"  # "Find out why...", "Analyze..."
    CREATE = "create"  # "Build...", "Generate...", "Write..."
    MODIFY = "modify"  # "Change...", "Update...", "Fix..."
    DELETE = "delete"  # "Remove...", "Delete...", "Clean up..."
    QUERY = "query"  # "What is...", "Show me...", "List..."
    PLAN = "plan"  # "Plan...", "Design...", "How should I..."
    LEARN = "learn"  # "Remember...", "Learn...", "Note that..."
    SIMULATE = "simulate"  # "What if...", "Imagine...", "Try..."
    MONITOR = "monitor"  # "Watch...", "Alert me...", "Track..."
    ACT = "act"  # "Push...", "Move...", "Grab...", "Interact..."


@dataclass
class SemanticSlot:
    """A named semantic slot extracted from user intent."""

    name: str
    value: Any
    confidence: float = 0.8


@dataclass
class SemanticAST:
    """Abstract syntax tree of user intent.

    Represents the structured meaning of a natural language input,
    ready for compilation into HCIR bytecode.

    Example::

        "Find why battery temperature increased"

        SemanticAST(
            intent=IntentType.INVESTIGATE,
            subject="battery_temperature",
            action="find_cause",
            slots=[
                SemanticSlot("metric", "temperature"),
                SemanticSlot("direction", "increased"),
            ],
        )
    """

    intent: IntentType
    subject: str = ""
    action: str = ""
    slots: list[SemanticSlot] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    priority: float = 0.5
    raw_text: str = ""
    grounded_entities: dict[str, GroundedEntityRef | UnresolvedRef] = field(default_factory=dict)

    def get_slot(self, name: str) -> Any | None:
        for slot in self.slots:
            if slot.name == name:
                return slot.value
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Compiler — AST → HCIR Bytecode
# ═══════════════════════════════════════════════════════════════════════════


class HCIRCompiler:
    """Compiles a SemanticAST into an HCIR InstructionStream.

    The compiler maps high-level intents to sequences of
    primitive bytecode instructions.  Each intent type has
    a compilation strategy.

    Usage::

        compiler = HCIRCompiler()
        ast = SemanticAST(
            intent=IntentType.INVESTIGATE,
            subject="battery_temperature",
            action="find_cause",
        )
        stream = compiler.compile(ast, author="user_session")
    """

    def __init__(
        self,
        tenant_id: str = "default",
        optimizer: Any | None = None,
        concept_store: WorkingConceptStore | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._optimizer = optimizer
        self._concept_store = concept_store
        # Intent → compilation strategy mapping
        self._strategies: dict[IntentType, Any] = {
            IntentType.INVESTIGATE: self._compile_investigate,
            IntentType.CREATE: self._compile_create,
            IntentType.MODIFY: self._compile_modify,
            IntentType.DELETE: self._compile_delete,
            IntentType.QUERY: self._compile_query,
            IntentType.PLAN: self._compile_plan,
            IntentType.LEARN: self._compile_learn,
            IntentType.SIMULATE: self._compile_simulate,
            IntentType.MONITOR: self._compile_monitor,
            IntentType.ACT: self._compile_act,
        }

    @property
    def concept_store(self) -> WorkingConceptStore | None:
        return self._concept_store

    @concept_store.setter
    def concept_store(self, store: WorkingConceptStore | None) -> None:
        self._concept_store = store

    def resolve_grounded_entities(
        self,
        ast: SemanticAST,
        concept_store: WorkingConceptStore | None = None,
    ) -> dict[str, GroundedEntityRef | UnresolvedRef]:
        """Resolves noun phrases in SemanticAST against WorkingConceptStore linguistic aliases.

        - If matched to a concept with exemplar entities, resolves to GroundedEntityRef.
        - If matched to a concept without exemplars, creates a canonical GroundedEntityRef.
        - If unmatched (unseen noun), produces an UnresolvedRef triggering active inquiry.
        """
        store = concept_store or self._concept_store
        resolved: dict[str, GroundedEntityRef | UnresolvedRef] = {}

        # Collect candidate phrases from slots and subject
        phrases_to_resolve: list[str] = []
        for slot in ast.slots:
            if slot.name in (
                "target",
                "target_entity",
                "entity",
                "object",
                "noun",
                "item",
            ) and isinstance(slot.value, str):
                if slot.value not in phrases_to_resolve:
                    phrases_to_resolve.append(slot.value)
        if ast.subject and ast.subject not in phrases_to_resolve:
            phrases_to_resolve.append(ast.subject)

        for raw_phrase in phrases_to_resolve:
            clean = raw_phrase.strip().strip(".,!?:;\"'()[]{}")
            norm = clean.lower()
            for prefix in ("the ", "a ", "an ", "this ", "that "):
                if norm.startswith(prefix):
                    norm = norm[len(prefix) :].strip()
                    break

            if not norm:
                continue

            concept = None
            if store is not None:
                concept = store.find_concept_by_alias(norm)
                if concept is None:
                    concept = store.find_concept_by_alias(clean.lower())

            if concept is not None:
                # Resolved grounded concept
                if concept.exemplar_entity_ids:
                    eid = sorted(concept.exemplar_entity_ids)[0]
                    if not eid.startswith("e_"):
                        eid = f"e_{eid}"
                    ref: GroundedEntityRef | UnresolvedRef = GroundedEntityRef(eid)
                else:
                    cid_sanitized = concept.concept_ref.id.lower().replace("-", "_")
                    ref = GroundedEntityRef(f"e_{cid_sanitized}")
            else:
                placeholder = f"unresolved_{uuid.uuid4().hex[:6]}"
                ref = UnresolvedRef(
                    placeholder_id=placeholder,
                    constraints={
                        "noun_phrase": norm,
                        "raw_phrase": raw_phrase,
                        "source": "dialogue",
                    },
                )

            resolved[raw_phrase] = ref
            if norm != raw_phrase:
                resolved[norm] = ref

        ast.grounded_entities.update(resolved)
        return resolved

    def compile(
        self,
        ast: SemanticAST,
        author: str = "compiler",
        optimize: bool = False,
        concept_store: WorkingConceptStore | None = None,
    ) -> InstructionStream:
        """Compile a SemanticAST into bytecode, with optional optimization pass.

        If a WorkingConceptStore is present, resolves linguistic entities. Unresolved
        references emit active inquiry instructions (REQUEST_EVIDENCE) rather than
        failing silently.
        """
        effective_store = concept_store or self._concept_store
        if effective_store is not None:
            self.resolve_grounded_entities(ast, concept_store=effective_store)

        # Active inquiry gating for unseen / unresolved referenced nouns
        unresolved_entries = [
            (phrase, ref)
            for phrase, ref in ast.grounded_entities.items()
            if isinstance(ref, UnresolvedRef)
        ]
        # Only gate on unresolved entities that correspond to core target slots or subject
        critical_unresolved = [
            (p, r)
            for p, r in unresolved_entries
            if p == ast.subject
            or any(s.value == p for s in ast.slots if s.name in ("target", "target_entity"))
        ]

        if critical_unresolved:
            stream = InstructionStream(
                author=author,
                description=f"Active Inquiry / Request Evidence: {ast.subject or ast.raw_text}",
            )
            for phrase, unres in critical_unresolved:
                stream.append(
                    Instruction(
                        opcode=Opcode.QUERY,
                        params={
                            "query_type": "request_evidence",
                            "outcome": DecisionOutcome.REQUEST_EVIDENCE.value,
                            "unresolved_ref": unres.placeholder_id,
                            "target_noun": phrase,
                            "constraints": unres.constraints,
                            "rationale": f"Unseen noun '{phrase}' requires sensory grounding evidence",
                        },
                        cost_estimate=20,
                    )
                )
                stream.append(
                    Instruction(
                        opcode=Opcode.WAIT,
                        params={
                            "condition": f"evidence_resolved:{unres.placeholder_id}",
                            "timeout_ms": 5000,
                        },
                        cost_estimate=5,
                    )
                )
            return stream

        strategy = self._strategies.get(ast.intent)
        if strategy is None:
            logger.warning("No compilation strategy for intent: %s", ast.intent)
            return InstructionStream(author=author, description=f"Unknown intent: {ast.intent}")

        stream = strategy(ast, author)
        if optimize:
            opt = self._optimizer
            if opt is None:
                from hbllm.hcir.optimizer import HCIROptimizer

                opt = HCIROptimizer()
            stream = opt.optimize(stream)

        logger.debug(
            "Compiled '%s' intent → %d instructions (optimized=%s)",
            ast.intent,
            stream.length,
            optimize,
        )
        return stream

    def parse_utterance(self, text: str) -> SemanticAST:
        """Parses natural language prompt into a SemanticAST with slots."""
        text_lower = text.lower().strip()

        # Check action command prefixes
        action_match: str | None = None
        for act_word in (
            "push",
            "move",
            "grab",
            "pick up",
            "pull",
            "open",
            "close",
            "press",
            "navigate to",
            "go to",
            "reach",
            "touch",
        ):
            if text_lower.startswith(act_word):
                action_match = act_word
                break
            elif f" {act_word} " in f" {text_lower} ":
                action_match = act_word
                break

        if action_match:
            intent = IntentType.ACT
            target = text_lower
            if target.startswith(action_match):
                target = target[len(action_match) :].strip()
            # Strip leading articles
            for prefix in ("the ", "a ", "an ", "this ", "that "):
                if target.startswith(prefix):
                    target = target[len(prefix) :].strip()
                    break
            return SemanticAST(
                intent=intent,
                subject=target,
                action=action_match,
                slots=[
                    SemanticSlot(name="action", value=action_match),
                    SemanticSlot(name="target", value=target),
                ],
                raw_text=text,
            )
        elif any(
            w in text_lower for w in ["find why", "investigate", "analyze", "why did", "diagnose"]
        ):
            intent = IntentType.INVESTIGATE
            action = "find_cause"
        elif any(w in text_lower for w in ["plan", "how to", "design", "steps for"]):
            intent = IntentType.PLAN
            action = "plan_steps"
        elif any(w in text_lower for w in ["create", "build", "generate", "write", "make"]):
            intent = IntentType.CREATE
            action = "create_item"
        elif any(w in text_lower for w in ["modify", "change", "update", "fix"]):
            intent = IntentType.MODIFY
            action = "modify_item"
        elif any(w in text_lower for w in ["simulate", "what if", "imagine", "try"]):
            intent = IntentType.SIMULATE
            action = "simulate_scenario"
        elif any(w in text_lower for w in ["remember", "learn", "note"]):
            intent = IntentType.LEARN
            action = "learn_fact"
        else:
            intent = IntentType.QUERY
            action = "evaluate_prompt"

        return SemanticAST(
            intent=intent,
            subject=text[:100],
            action=action,
            raw_text=text,
        )

    def compile_text(
        self,
        text: str,
        author: str = "user",
        optimize: bool = False,
        concept_store: WorkingConceptStore | None = None,
    ) -> InstructionStream:
        """Parse natural language prompt into a SemanticAST and compile to InstructionStream."""
        ast = self.parse_utterance(text)
        return self.compile(ast, author=author, optimize=optimize, concept_store=concept_store)

    def interpret_dialogue(
        self,
        text: str,
        speaker_id: str = "user",
        concept_store: WorkingConceptStore | None = None,
    ) -> DialogueInterpretationResult:
        """Parses natural language dialogue into a structured DialogueInterpretationResult
        with entities grounded via WorkingConceptStore.
        """
        ast = self.parse_utterance(text)
        effective_store = concept_store or self._concept_store
        if effective_store is not None:
            self.resolve_grounded_entities(ast, concept_store=effective_store)

        dialogue_act = DialogueAct.COMMAND if ast.intent == IntentType.ACT else DialogueAct.REQUEST
        if ast.intent == IntentType.QUERY:
            dialogue_act = DialogueAct.QUESTION
        elif ast.intent == IntentType.LEARN:
            dialogue_act = DialogueAct.INFORM

        action_proposals: list[ActionIntentProposal] = []
        if ast.action or ast.intent == IntentType.ACT:
            target_ref = None
            for ref in ast.grounded_entities.values():
                target_ref = ref
                break
            action_proposals.append(
                ActionIntentProposal(
                    intent_id=f"act_prop_{uuid.uuid4().hex[:8]}",
                    action_type=ast.action or "execute",
                    target_ref=target_ref,
                    parameters={"subject": ast.subject, "raw_text": ast.raw_text},
                )
            )

        has_unresolved = any(isinstance(r, UnresolvedRef) for r in ast.grounded_entities.values())
        return DialogueInterpretationResult(
            utterance_id=f"utt_{uuid.uuid4().hex[:8]}",
            dialogue_act=dialogue_act,
            action_proposals=tuple(action_proposals),
            confidence=0.5 if has_unresolved else 1.0,
        )

    def evaluate_grounded_expression(
        self,
        ast: SemanticAST,
        theory: Any,
        grid: Any | None = None,
    ) -> list[Any]:
        """Evaluate compositional natural language expression against HCIRSymbolicWorldTheory and scene grid (G8).

        Resolves expressions such as:
        - "the block left of the key"
        - "the barrier above the switch"
        - "the green key"
        """
        import numpy as np

        text = (ast.raw_text or ast.subject).lower()

        rel_map = {
            "left of": "LEFT_OF",
            "right of": "RIGHT_OF",
            "above": "ABOVE",
            "below": "BELOW",
            "inside": "INSIDE",
            "contains": "CONTAINS",
            "adjacent to": "ADJACENT",
        }
        detected_rel: str | None = None
        subj_name = ""
        ref_name = ""

        for phrase, rel in rel_map.items():
            if phrase in text:
                detected_rel = rel
                parts = text.split(phrase, 1)
                subj_name = parts[0].strip()
                ref_name = parts[1].strip()
                break

        if not detected_rel:
            subj_name = text
            ref_name = ""

        for prefix in ("the ", "a ", "an ", "this ", "that "):
            if subj_name.startswith(prefix):
                subj_name = subj_name[len(prefix) :].strip()
            if ref_name.startswith(prefix):
                ref_name = ref_name[len(prefix) :].strip()

        # If grid is provided, evaluate spatial coordinates of matching entities
        if grid is not None and isinstance(grid, np.ndarray) and detected_rel and ref_name:
            subj_feat = self._resolve_feature_for_label(subj_name, theory)
            ref_feat = self._resolve_feature_for_label(ref_name, theory)

            if subj_feat is not None and ref_feat is not None:
                subj_coords = list(zip(*((grid == subj_feat).nonzero())))
                ref_coords = list(zip(*((grid == ref_feat).nonzero())))

                matching_coords: list[tuple[int, int]] = []
                for sr, sc in subj_coords:
                    for rr, rc in ref_coords:
                        if detected_rel == "LEFT_OF" and sc < rc and abs(sr - rr) <= 2:
                            matching_coords.append((int(sr), int(sc)))
                        elif detected_rel == "RIGHT_OF" and sc > rc and abs(sr - rr) <= 2:
                            matching_coords.append((int(sr), int(sc)))
                        elif detected_rel == "ABOVE" and sr < rr and abs(sc - rc) <= 2:
                            matching_coords.append((int(sr), int(sc)))
                        elif detected_rel == "BELOW" and sr > rr and abs(sc - rc) <= 2:
                            matching_coords.append((int(sr), int(sc)))
                        elif detected_rel == "ADJACENT" and abs(sr - rr) + abs(sc - rc) <= 1:
                            matching_coords.append((int(sr), int(sc)))

                if matching_coords:
                    return matching_coords

        return [subj_name] if subj_name else []

    def _resolve_feature_for_label(self, label: str, theory: Any) -> Any | None:
        """Resolve a textual entity label to a physical feature ID via theory or concept store."""
        lbl = label.lower().strip()
        color_map = {
            "black": 0,
            "blue": 1,
            "red": 2,
            "green": 3,
            "yellow": 4,
            "grey": 5,
            "gray": 5,
            "magenta": 6,
            "orange": 7,
            "cyan": 8,
            "brown": 9,
        }
        for cname, ccode in color_map.items():
            if cname in lbl:
                return ccode

        if theory is not None:
            if "key" in lbl and hasattr(theory, "tool_barrier_affinities"):
                for tool in theory.tool_barrier_affinities:
                    return tool
            if "door" in lbl or "barrier" in lbl:
                if hasattr(theory, "barrier_features") and theory.barrier_features:
                    return next(iter(theory.barrier_features))
            if "goal" in lbl and hasattr(theory, "goal_features") and theory.goal_features:
                return next(iter(theory.goal_features))

        return None

    def compile_utterance_to_causal_hypothesis(
        self,
        utterance: str,
        theory: Any | None = None,
    ) -> Any:
        """Parse natural language rule into a testable CausalPredicate hypothesis (G8).

        Examples:
        - "red doors need red keys"
        - "switches unlock gates"
        - "touching fire causes reset"
        - "pushing block clears path"
        """
        from hbllm.hcir.world.causal_discovery import CausalPredicate

        text = utterance.lower().strip()
        premise = ""
        relation = "causes"
        effect = ""

        keywords = [
            (" need ", "requires"),
            (" needs ", "requires"),
            (" require ", "requires"),
            (" requires ", "requires"),
            (" unlock ", "unlocks"),
            (" unlocks ", "unlocks"),
            (" clear ", "clears"),
            (" clears ", "clears"),
            (" cause ", "causes"),
            (" causes ", "causes"),
            (" enable ", "enables"),
            (" enables ", "enables"),
        ]

        for kw, rel in keywords:
            if kw in text:
                parts = text.split(kw, 1)
                premise = parts[0].strip()
                relation = rel
                effect = parts[1].strip()
                break

        if not premise and "if " in text and " then " in text:
            after_if = text.split("if ", 1)[1]
            parts = after_if.split(" then ", 1)
            premise = parts[0].strip()
            relation = "implies"
            effect = parts[1].strip()

        if not premise:
            premise = text
            relation = "asserted_rule"
            effect = "true"

        for prefix in ("the ", "a ", "an ", "all "):
            if premise.startswith(prefix):
                premise = premise[len(prefix) :].strip()
            if effect.startswith(prefix):
                effect = effect[len(prefix) :].strip()

        var_name = f"rule:{premise.replace(' ', '_')}->{effect.replace(' ', '_')}"
        pred = CausalPredicate(
            variable=var_name,
            operator="==",
            value=effect,
            relation=relation,
            target_entity=premise,
        )
        pred.causal_status = "HYPOTHESISED"

        if theory is not None and hasattr(theory, "record_causal_predicate"):
            theory.record_causal_predicate(pred)
            if relation in ("requires", "unlocks", "clears"):
                prem_feat = self._resolve_feature_for_label(premise, theory)
                eff_feat = self._resolve_feature_for_label(effect, theory)
                if prem_feat is not None and eff_feat is not None:
                    theory.record_tool_barrier_affinity(eff_feat, prem_feat)

        return pred

    # ── Compilation Strategies ───────────────────────────────────────

    def _compile_act(self, ast: SemanticAST, author: str) -> InstructionStream:
        """ACT: ASSERT action proposal → EXECUTE motor capability on target entity."""
        target_ref: str = ast.subject
        for ref in ast.grounded_entities.values():
            if isinstance(ref, GroundedEntityRef):
                target_ref = ref.id
                break

        action_name = ast.action or "act"
        act_id = f"act_{uuid.uuid4().hex[:8]}"
        stream = InstructionStream(
            author=author,
            description=f"Act: {action_name} on {target_ref}",
        )
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": {
                        "id": act_id,
                        "node_type": "action_intent",
                        "action": action_name,
                        "target_ref": target_ref,
                        "priority": ast.priority,
                    },
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": f"motor_{action_name}",
                    "target_entity_id": target_ref,
                    "target_ref": target_ref,
                    "params": {"action": action_name, "target": target_ref},
                },
                cost_estimate=50,
            )
        )
        return stream

    def _compile_investigate(self, ast: SemanticAST, author: str) -> InstructionStream:
        """INVESTIGATE: ASSERT goal → QUERY observations → QUERY procedures → EXECUTE analysis."""
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        stream = InstructionStream(
            author=author,
            description=f"Investigate: {ast.subject}",
        )
        # 1. Assert investigation goal
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id=goal_id,
                        description=f"Investigate: {ast.subject} ({ast.action})",
                        priority=ast.priority,
                    ).model_dump(),
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        # 2. Query relevant observations
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={
                    "node_type": "observation",
                    "text_contains": ast.subject,
                },
                cost_estimate=10,
            )
        )
        # 3. Query relevant procedures
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={
                    "node_type": "procedure",
                    "text_contains": ast.action or ast.subject,
                },
                cost_estimate=10,
            )
        )
        # 4. Execute analysis capability
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "causal_analysis",
                    "params": {
                        "subject": ast.subject,
                        "action": ast.action,
                        "goal_id": goal_id,
                    },
                },
                cost_estimate=100,
            )
        )
        return stream

    def _compile_create(self, ast: SemanticAST, author: str) -> InstructionStream:
        """CREATE: ASSERT goal → EXECUTE creation → ASSERT result."""
        stream = InstructionStream(author=author, description=f"Create: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id=f"goal_{uuid.uuid4().hex[:8]}",
                        description=f"Create: {ast.subject}",
                        priority=ast.priority,
                    ).model_dump(),
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "content_generation",
                    "params": {"subject": ast.subject, "action": ast.action},
                },
                cost_estimate=200,
            )
        )
        return stream

    def _compile_modify(self, ast: SemanticAST, author: str) -> InstructionStream:
        """MODIFY: QUERY target → EXECUTE modification."""
        stream = InstructionStream(author=author, description=f"Modify: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"text_contains": ast.subject},
                cost_estimate=10,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "content_modification",
                    "params": {"subject": ast.subject, "action": ast.action},
                },
                cost_estimate=100,
            )
        )
        return stream

    def _compile_delete(self, ast: SemanticAST, author: str) -> InstructionStream:
        """DELETE: QUERY target → RETRACT."""
        stream = InstructionStream(author=author, description=f"Delete: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"text_contains": ast.subject},
                cost_estimate=10,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.RETRACT,
                params={"node_id": ast.subject},
                cost_estimate=5,
            )
        )
        return stream

    def _compile_query(self, ast: SemanticAST, author: str) -> InstructionStream:
        """QUERY: direct graph query."""
        stream = InstructionStream(author=author, description=f"Query: {ast.subject}")
        params: dict[str, Any] = {}
        if ast.subject:
            params["text_contains"] = ast.subject
        node_type = ast.get_slot("node_type")
        if node_type:
            params["node_type"] = node_type
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params=params,
                cost_estimate=10,
            )
        )
        return stream

    def _compile_plan(self, ast: SemanticAST, author: str) -> InstructionStream:
        """PLAN: ASSERT goal → QUERY constraints → EXECUTE planner."""
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        stream = InstructionStream(author=author, description=f"Plan: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id=goal_id,
                        description=f"Plan: {ast.subject}",
                        priority=ast.priority,
                    ).model_dump(),
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"node_type": "constraint"},
                cost_estimate=10,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "planning",
                    "params": {"goal_id": goal_id, "subject": ast.subject},
                },
                cost_estimate=150,
            )
        )
        return stream

    def _compile_learn(self, ast: SemanticAST, author: str) -> InstructionStream:
        """LEARN: QUERY existing → ASSERT new knowledge → EXECUTE skill extraction."""
        stream = InstructionStream(author=author, description=f"Learn: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"text_contains": ast.subject},
                cost_estimate=10,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": {
                        "id": f"belief_{uuid.uuid4().hex[:8]}",
                        "node_type": "belief",
                        "claim": ast.subject,
                        "belief_type": "learned",
                    },
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "skill_extraction",
                    "params": {"subject": ast.subject},
                },
                cost_estimate=50,
            )
        )
        return stream

    def _compile_simulate(self, ast: SemanticAST, author: str) -> InstructionStream:
        """SIMULATE: FORK → EXECUTE scenario → MERGE or ROLLBACK."""
        branch_name = f"sim_{uuid.uuid4().hex[:6]}"
        stream = InstructionStream(author=author, description=f"Simulate: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.FORK,
                params={"branch_name": branch_name},
                cost_estimate=20,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "simulation",
                    "params": {"scenario": ast.subject, "branch": branch_name},
                },
                cost_estimate=200,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.MERGE,
                params={"branch_name": branch_name},
                cost_estimate=20,
            )
        )
        return stream

    def _compile_monitor(self, ast: SemanticAST, author: str) -> InstructionStream:
        """MONITOR: ASSERT goal → WAIT for condition."""
        stream = InstructionStream(author=author, description=f"Monitor: {ast.subject}")
        stream.append(
            Instruction(
                opcode=Opcode.ASSERT,
                params={
                    "node_data": GoalNode(
                        id=f"goal_{uuid.uuid4().hex[:8]}",
                        description=f"Monitor: {ast.subject}",
                        priority=ast.priority,
                    ).model_dump(),
                    "author": author,
                },
                cost_estimate=5,
            )
        )
        stream.append(
            Instruction(
                opcode=Opcode.WAIT,
                params={"condition": ast.subject, "timeout_ms": 60000},
                cost_estimate=5,
            )
        )
        return stream
