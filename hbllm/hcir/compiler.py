"""HCIR Compiler Frontend — Construction Unification & Unknown-Instruction Resolution."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import (
    BeliefNode,
    ConceptNode,
    ConstructionNode,
    HCIRNodeType,
    NodeLifecycle,
    SkillNode,
    UnknownNode,
)
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.dialogue import (
    ActionIntentProposal,
    DialogueInterpretationResult,
    GoalProposal,
)
from hbllm.hcir.grounding.contracts.enums import DecisionOutcome, DialogueAct
from hbllm.hcir.grounding.contracts.references import GroundedEntityRef, UnresolvedRef
from hbllm.hcir.seed.loader import build_seeded_constructions, load_constructions_spec
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope

logger = logging.getLogger(__name__)
_LEADING_VERB_RE = re.compile(
    r"^([a-zA-Z\u0D80-\u0DFF\u0B80-\u0BFF][a-zA-Z0-9_\u0D80-\u0DFF\u0B80-\u0BFF-]{1,24})\s+(.+)$"
)
_NUMERIC_LABEL_RE = re.compile(r"\b(?:feature|target|symbol|object|tile|value|id|color)\s+(\d+)\b")


def _ins(op: Opcode, cost: int = 0, /, **params: Any) -> Instruction:
    return Instruction(opcode=op, params=params, cost_estimate=cost)


class IntentType(StrEnum):
    """High-level intent categories extracted from natural language."""

    INVESTIGATE = "investigate"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    QUERY = "query"
    PLAN = "plan"
    LEARN = "learn"
    SIMULATE = "simulate"
    MONITOR = "monitor"
    ACT = "act"


@dataclass
class SemanticSlot:
    """A named semantic slot extracted from user intent."""

    name: str
    value: Any
    confidence: float = 0.8


@dataclass
class SemanticAST:
    """Abstract syntax tree of user intent compiled into HCIR bytecode."""

    intent: IntentType
    subject: str = ""
    action: str = ""
    slots: list[SemanticSlot] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    priority: float = 0.5
    raw_text: str = ""
    grounded_entities: dict[str, GroundedEntityRef | UnresolvedRef] = field(default_factory=dict)

    def get_slot(self, name: str) -> Any | None:
        return next((s.value for s in self.slots if s.name == name), None)


def _strip_prefix(text: str, prefixes: list[str]) -> str:
    for pfx in prefixes:
        if text.startswith(pfx):
            return text[len(pfx) :].strip()
    return text


def _substitute_template(value: Any, bindings: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        if value in bindings:
            return bindings[value]
        if value.startswith("{") and value.endswith("?}"):
            return bindings.get(f"{{{value[1:-2]}}}")
        out = value
        for k, v in bindings.items():
            if k in out:
                out = out.replace(k, "" if v is None else str(v))
        return out
    if isinstance(value, list):
        return [_substitute_template(item, bindings) for item in value]
    if isinstance(value, dict):
        return {
            k: _substitute_template(v, bindings)
            for k, v in value.items()
            if not (
                isinstance(v, str) and v.endswith("?}") and not _substitute_template(v, bindings)
            )
        }
    return value


class HCIRCompiler:
    """Graph-driven Embodied Construction Grammar compiler & 4-step unknown resolver."""

    def __init__(
        self,
        tenant_id: str = "default",
        optimizer: Any | None = None,
        concept_store: WorkingConceptStore | None = None,
        *,
        workspace: Any | None = None,
        tiered_workspace: Any | None = None,
        ontology_store: Any | None = None,
        language: str = "en",
    ) -> None:
        self._tenant_id = tenant_id
        self._optimizer = optimizer
        self._concept_store = concept_store
        self._workspace = workspace
        self._tiered_workspace = tiered_workspace
        self._ontology_store = ontology_store
        self._language = language

    @property
    def concept_store(self) -> WorkingConceptStore | None:
        return self._concept_store

    @concept_store.setter
    def concept_store(self, store: WorkingConceptStore | None) -> None:
        self._concept_store = store

    def _workspaces(self) -> list[Any]:
        out: list[Any] = [self._workspace] if self._workspace is not None else []
        if self._tiered_workspace is not None:
            for tier_ws in (
                self._tiered_workspace.working_state,
                self._tiered_workspace.brain,
                self._tiered_workspace.persistent,
            ):
                if tier_ws not in out:
                    out.append(tier_ws)
        return out

    def get_constructions(self, language: str | None = None) -> list[ConstructionNode]:
        """Return active `ConstructionNode`s from workspace graph + declarative seed YAML."""
        lang = language or self._language
        by_id: dict[str, ConstructionNode] = {
            node.id: node
            for ws in self._workspaces()
            for node in ws.graph.nodes_by_type(HCIRNodeType.CONSTRUCTION)
            if isinstance(node, ConstructionNode) and node.language == lang
        }
        for seed_cxn in build_seeded_constructions(lang, tenant_id=self._tenant_id):
            by_id.setdefault(seed_cxn.id, seed_cxn)
        return sorted(
            by_id.values(),
            key=lambda c: (
                0 if "construction_seed" not in c.tags else 1,
                -c.reliability_mean,
                c.id,
            ),
        )

    def _collect_skill_nodes(self) -> list[SkillNode]:
        seen = {
            n.id: n
            for ws in self._workspaces()
            for n in ws.graph.nodes_by_type(HCIRNodeType.SKILL)
            if isinstance(n, SkillNode)
        }
        return list(seen.values())

    def _commit(self, ops: list[TransactionOperation], *, author: str = "compiler") -> None:
        if not ops:
            return
        if self._tiered_workspace is not None:
            self._tiered_workspace.brain.commit_operations(
                ops,
                author=author,
                grant=self._tiered_workspace._system_grant,
                tenant_id=self._tenant_id,
            )
        elif self._workspace is not None:
            self._workspace.commit_operations(ops, author=author, tenant_id=self._tenant_id)

    def resolve_grounded_entities(
        self, ast: SemanticAST, concept_store: WorkingConceptStore | None = None
    ) -> dict[str, GroundedEntityRef | UnresolvedRef]:
        """Resolve entity phrases in `SemanticAST` against `WorkingConceptStore` or graph."""
        store = concept_store or self._concept_store
        rules = load_constructions_spec(self._language).get("lexical_rules", {})
        target_slots = set(rules.get("target_slot_names", ()))
        articles = list(rules.get("articles", ()))

        phrases = [
            s.value for s in ast.slots if s.name in target_slots and isinstance(s.value, str)
        ]
        if ast.subject:
            phrases.append(ast.subject)
        resolved: dict[str, GroundedEntityRef | UnresolvedRef] = {}
        for raw_phrase in dict.fromkeys(phrases):
            clean = raw_phrase.strip().strip(".,!?:;\"'()[]{}")
            norm = _strip_prefix(clean.lower(), articles)
            if not norm:
                continue
            concept = None
            if store is not None:
                concept = store.find_concept_by_alias(norm) or store.find_concept_by_alias(
                    clean.lower()
                )
                if (
                    concept is None
                    and (cc := self._resolve_feature_for_label(norm, None)) is not None
                ):
                    for c in store._concepts.values():
                        if any(
                            p.get("raw_color") == cc or p.get("color") == cc for p in c.prototypes
                        ):
                            concept = c
                            break
            if concept is None:
                ref: GroundedEntityRef | UnresolvedRef = UnresolvedRef(
                    placeholder_id=f"unresolved_{uuid.uuid4().hex[:6]}",
                    constraints={
                        "noun_phrase": norm,
                        "raw_phrase": raw_phrase,
                        "source": "dialogue",
                    },
                )
            elif concept.exemplar_entity_ids:
                eid = sorted(concept.exemplar_entity_ids)[0]
                ref = GroundedEntityRef(eid if eid.startswith("e_") else f"e_{eid}")
            else:
                ref = GroundedEntityRef(f"e_{concept.concept_ref.id.lower().replace('-', '_')}")
            resolved[raw_phrase] = ref
            if norm != raw_phrase:
                resolved[norm] = ref
        ast.grounded_entities.update(resolved)
        return resolved

    def _resolve_target_ref(self, ast: SemanticAST) -> str:
        for ref in ast.grounded_entities.values():
            if isinstance(ref, GroundedEntityRef):
                return ref.id
        return ast.subject

    def _find_skill_for_verb(self, verb: str) -> SkillNode | None:
        v = verb.lower()
        for sk in self._collect_skill_nodes():
            if sk.skill_name.lower() == v or (
                isinstance(sk.postcondition_pattern, Mapping)
                and str(sk.postcondition_pattern.get("intent", "")).lower() == v
            ):
                return sk
        return None

    def _find_definition_text(self, verb: str) -> str | None:
        v = verb.lower()
        for ws in self._workspaces():
            for b in ws.graph.nodes_by_type(HCIRNodeType.BELIEF):
                if isinstance(b, BeliefNode):
                    text = f"{b.claim} {b.statement}".strip().lower()
                    if v in text and ("means" in text or "is to" in text):
                        return b.claim or b.statement
            for c in ws.graph.nodes_by_type(HCIRNodeType.CONCEPT):
                if isinstance(c, ConceptNode) and c.label.lower() == v and c.definition:
                    return f"to {v} means {c.definition}"
        if self._ontology_store is not None and hasattr(self._ontology_store, "get_record"):
            rec = self._ontology_store.get_record(v)
            if rec is not None:
                if defn := (getattr(rec, "definition", "") or ""):
                    return f"to {v} means {defn}"
                if hyps := (getattr(rec, "hypernyms", ()) or ()):
                    return f"to {v} an object means to {hyps[0]} the object"
        return None

    def _analogy_skill_match(self, verb: str, ast: SemanticAST) -> SkillNode | None:
        skills = [s for s in self._collect_skill_nodes() if s.bytecode_stream]
        if not skills:
            return None
        stop = {
            a.strip()
            for a in load_constructions_spec(self._language)
            .get("lexical_rules", {})
            .get("articles", ())
        }
        target_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9_]+", f"{ast.subject} {ast.raw_text}".lower())
            if tok and tok != verb.lower() and tok not in stop
        }
        best_skill: SkillNode | None = None
        best_score, tie = 0.0, False
        pat_edge = {"rel_type": "acts_on", "source_var": "?s", "target_var": "?t"}
        for sk in skills:
            pat_vals = f"{sk.description} {sk.precondition_pattern or {}} {sk.postcondition_pattern or {}}".lower()
            pat_tokens = set(re.findall(r"[a-z0-9_]+", pat_vals))
            if not pat_tokens:
                continue
            overlap = len(target_tokens & pat_tokens) / max(1, len(target_tokens))
            sys_score = 1.0
            try:
                import hbllm_structure_matcher

                tgt_edge = {
                    "rel_type": "acts_on",
                    "source": sk.id,
                    "target": ast.subject or "target",
                }
                alns = hbllm_structure_matcher.match_relational_schema(
                    {"variables": ["?s", "?t"], "edges": [pat_edge]},
                    {"nodes": [sk.id, ast.subject or "target"], "edges": [tgt_edge]},
                    0.3,
                )
                sys_score = float(alns[0]["systematicity_score"]) if alns else 0.0
            except ImportError:
                sys_score = 1.0
            score = (0.7 * overlap) + (0.3 * sys_score)
            if score > best_score:
                best_score, best_skill, tie = score, sk, False
            elif score == best_score and score > 0.0:
                tie = True
        return best_skill if best_skill is not None and not tie and best_score >= 0.6 else None

    def _call_stream(
        self, skill_id: str, target_ref: str, subject: str, desc: str, author: str
    ) -> InstructionStream:
        return InstructionStream(
            author=author,
            description=desc,
            instructions=[
                _ins(Opcode.UNIFY, 5, pattern={"id": target_ref}, bind="$r0"),
                _ins(
                    Opcode.CALL,
                    15,
                    skill_id=skill_id,
                    args={"target": "$r0", "target_ref": target_ref, "subject": subject},
                    bind="$r1",
                ),
                _ins(Opcode.RETURN, 0, **{"yield": ["$r1"]}),
            ],
        )

    def resolve_unknown_instruction(
        self, ast: SemanticAST, author: str = "compiler"
    ) -> InstructionStream:
        """Execute the 4-Step Unknown-Instruction Resolution Pipeline."""
        verb = str(ast.get_slot("unknown_verb") or ast.action or "unknown").lower()
        target_ref = self._resolve_target_ref(ast)
        unk_id = f"unk_{verb}_{uuid.uuid4().hex[:6]}"
        unk_node = UnknownNode(
            id=unk_id,
            question=f"Resolve unknown action '{verb}' on '{ast.subject}'",
            context=ast.raw_text,
            domain="construction_grammar",
            provenance=Provenance(created_by=author),
            scope=Scope(tenant_id=self._tenant_id),
        )
        matched_skill = self._find_skill_for_verb(verb)
        ops: list[TransactionOperation] = []
        if matched_skill is None and (defn_text := self._find_definition_text(verb)) is not None:
            from hbllm.hcir.language.reading_curriculum import (
                compile_procedural_definitions_to_ops,
                extract_procedural_definitions,
            )

            if defs := extract_procedural_definitions(defn_text, source_file="memory_recall"):
                ops.extend(compile_procedural_definitions_to_ops(defs, tenant_id=self._tenant_id))
                matched_skill = SkillNode.model_validate(ops[0].node_data)

        if matched_skill is not None:
            existing = next(
                (
                    c
                    for c in self.get_constructions()
                    if verb in [t.lower() for t in c.trigger_tokens]
                ),
                None,
            )
            if existing is None and self._workspaces():
                tpl: list[dict[str, Any]] = [
                    {
                        "opcode": "UNIFY",
                        "cost_estimate": 5,
                        "params": {"pattern": {"id": "{target_ref}"}, "bind": "$r0"},
                    },
                    {
                        "opcode": "CALL",
                        "cost_estimate": 15,
                        "params": {
                            "skill_id": matched_skill.id,
                            "args": {"target": "$r0", "target_ref": "{target_ref}"},
                            "bind": "$r1",
                        },
                    },
                    {"opcode": "RETURN", "params": {"yield": ["$r1"]}},
                ]
                cxn = ConstructionNode(
                    id=f"cxn_{self._language}_{verb}",
                    construction_name=f"learned_{verb}",
                    language=self._language,
                    intent_type="act",
                    syntactic_template="{action} {target}",
                    trigger_tokens=[verb],
                    compiled_hcir_template=tpl,
                    lifecycle=NodeLifecycle.ACTIVE,
                    provenance=Provenance(created_by="construction_resolver"),
                    scope=Scope(tenant_id=self._tenant_id),
                    tags=["learned_construction", verb],
                )
                ops.append(
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE, node_id=cxn.id, node_data=cxn.model_dump()
                    )
                )
            self._commit(ops, author=author)
            return self._call_stream(
                matched_skill.id,
                target_ref,
                ast.subject,
                f"Resolved '{verb}' -> CALL({matched_skill.skill_name})",
                author,
            )

        if (analogy_skill := self._analogy_skill_match(verb, ast)) is not None:
            return self._call_stream(
                analogy_skill.id,
                target_ref,
                ast.subject,
                f"Analogy '{verb}' -> CALL({analogy_skill.skill_name})",
                author,
            )

        if self._workspaces():
            self._commit(
                [
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=unk_node.id,
                        node_data=unk_node.model_dump(),
                    )
                ],
                author=author,
            )
        return InstructionStream(
            author=author,
            description=f"Active Inquiry for unknown verb '{verb}'",
            instructions=[
                _ins(Opcode.ASSERT, 5, node_data=unk_node.model_dump(), author=author, bind="$r0"),
                _ins(Opcode.FORK, 10, branch_name="probe_unknown"),
                _ins(
                    Opcode.QUERY,
                    20,
                    query_type="request_evidence",
                    outcome=DecisionOutcome.REQUEST_EVIDENCE.value,
                    unknown_verb=verb,
                    target_noun=ast.subject,
                    unresolved_ref=unk_id,
                ),
                _ins(Opcode.ROLLBACK, 5, branch_name="probe_unknown"),
                _ins(Opcode.WAIT, 5, condition=f"evidence_resolved:{unk_id}", timeout_ms=5000),
            ],
        )

    def _compile_from_construction(
        self, cxn: ConstructionNode, ast: SemanticAST, author: str
    ) -> InstructionStream:
        target_ref = self._resolve_target_ref(ast)
        bindings: dict[str, Any] = {
            "{subject}": ast.subject,
            "{action}": ast.action or cxn.intent_type,
            "{action_or_subject}": ast.action or ast.subject,
            "{target_ref}": target_ref,
            "{priority}": ast.priority,
            "{author}": author,
            "{act_id}": f"act_{uuid.uuid4().hex[:8]}",
            "{goal_id}": f"goal_{uuid.uuid4().hex[:8]}",
            "{belief_id}": f"belief_{uuid.uuid4().hex[:8]}",
            "{branch}": f"sim_{uuid.uuid4().hex[:6]}",
            "{slot.node_type}": ast.get_slot("node_type"),
        }
        desc_tpl = str(cxn.semantic_mapping.get("description_template", "{action}: {subject}"))
        ins_list = [
            Instruction(
                opcode=Opcode(str(raw["opcode"]).upper()),
                params=_substitute_template(dict(raw.get("params", {})), bindings),
                cost_estimate=int(raw.get("cost_estimate", 10)),
                label=raw.get("label"),
            )
            for raw in cxn.compiled_hcir_template
        ]
        if not ins_list and cxn.bytecode_stream is not None:
            ins_list = InstructionStream.model_validate(cxn.bytecode_stream).instructions
        return InstructionStream(
            author=author,
            description=str(_substitute_template(desc_tpl, bindings)),
            instructions=ins_list,
        )

    def _select_construction(self, ast: SemanticAST) -> ConstructionNode | None:
        constructions, act_l = self.get_constructions(), ast.action.lower()
        for cxn in constructions:
            if (
                cxn.intent_type == ast.intent.value
                and act_l
                and act_l in [t.lower() for t in cxn.trigger_tokens]
            ):
                return cxn
        return next((c for c in constructions if c.intent_type == ast.intent.value), None)

    def compile(
        self,
        ast: SemanticAST,
        author: str = "compiler",
        optimize: bool = False,
        concept_store: WorkingConceptStore | None = None,
    ) -> InstructionStream:
        """Compile a `SemanticAST` via Construction Unification or 4-step Unknown Resolution."""
        effective_store = concept_store or self._concept_store
        if effective_store is not None:
            self.resolve_grounded_entities(ast, concept_store=effective_store)
        rules = load_constructions_spec(self._language).get("lexical_rules", {})
        target_slots = tuple(rules.get("target_slot_names", ("target", "target_entity")))
        critical_unres = [
            (p, r)
            for p, r in ast.grounded_entities.items()
            if isinstance(r, UnresolvedRef)
            and (p == ast.subject or any(s.value == p for s in ast.slots if s.name in target_slots))
        ]
        if critical_unres:
            stream = InstructionStream(
                author=author,
                description=f"Active Inquiry / Request Evidence: {ast.subject or ast.raw_text}",
            )
            for phrase, unres in critical_unres:
                stream.append(
                    _ins(
                        Opcode.QUERY,
                        20,
                        query_type="request_evidence",
                        outcome=DecisionOutcome.REQUEST_EVIDENCE.value,
                        unresolved_ref=unres.placeholder_id,
                        target_noun=phrase,
                        constraints=unres.constraints,
                        rationale=f"Unseen noun '{phrase}' requires sensory grounding evidence",
                    )
                )
                stream.append(
                    _ins(
                        Opcode.WAIT,
                        5,
                        condition=f"evidence_resolved:{unres.placeholder_id}",
                        timeout_ms=5000,
                    )
                )
            return stream

        cxn = self._select_construction(ast)
        if ast.get_slot("unknown_verb") is not None:
            if cxn is not None and (
                "learned_construction" in cxn.tags or "t4_construction" in cxn.tags
            ):
                stream = self._compile_from_construction(cxn, ast, author)
            else:
                stream = self.resolve_unknown_instruction(ast, author=author)
        elif cxn is None:
            return InstructionStream(author=author, description=f"Unknown intent: {ast.intent}")
        else:
            stream = self._compile_from_construction(cxn, ast, author)

        if optimize:
            opt = self._optimizer
            if opt is None:
                from hbllm.hcir.optimizer import HCIROptimizer

                opt = HCIROptimizer()
            stream = opt.optimize(stream)
        return stream

    def _is_known_target(self, noun_phrase: str) -> bool:
        norm = noun_phrase.lower().strip()
        if (
            self._concept_store is not None
            and self._concept_store.find_concept_by_alias(norm) is not None
        ):
            return True
        return any(
            getattr(ent, "entity_name", "").lower() == norm
            or getattr(ent, "name", "").lower() == norm
            for ws in self._workspaces()
            for ent in ws.graph.nodes_by_type(HCIRNodeType.PHYSICAL_ENTITY)
        )

    def parse_utterance(self, text: str, language: str | None = None) -> SemanticAST:
        """Parse natural language using graph `ConstructionNode`s and declarative seed YAML."""
        lang = language or self._language
        rules = load_constructions_spec(lang).get("lexical_rules", {})
        articles, directions = list(rules.get("articles", ())), list(rules.get("directions", ()))
        q_starters = set(rules.get("question_starters", ()))
        text_lower = text.strip().rstrip(".!").lower()
        constructions = self.get_constructions(lang)
        act_triggers = [
            tl
            for cxn in constructions
            if cxn.intent_type == "act"
            for tl in (t.lower() for t in cxn.trigger_tokens)
        ]

        def _make_act_ast(tl: str) -> SemanticAST:
            tgt = (
                text_lower[len(tl) :].strip()
                if text_lower.startswith(tl)
                else text_lower[: -len(tl)].strip()
                if text_lower.endswith(tl)
                else text_lower
            )
            tgt = _strip_prefix(tgt, articles)
            d_word = next((d for d in directions if tgt.endswith(f" {d}")), None)
            if d_word is not None:
                tgt = tgt[: -len(f" {d_word}")].strip()
            slots = [SemanticSlot("action", tl), SemanticSlot("target", tgt)]
            if d_word:
                slots.append(SemanticSlot("direction", d_word))
            return SemanticAST(
                intent=IntentType.ACT, subject=tgt, action=tl, slots=slots, raw_text=text
            )

        if matched_tl := next(
            (tl for tl in act_triggers if text_lower == tl or text_lower.startswith(f"{tl} ")),
            None,
        ):
            return _make_act_ast(matched_tl)
        if lang != "en":
            if matched_tl := next(
                (tl for tl in act_triggers if text_lower.endswith(f" {tl}")), None
            ):
                return _make_act_ast(matched_tl)
        if matched_tl := next(
            (tl for tl in act_triggers if f" {tl} " in f" {text_lower} "),
            None,
        ):
            return _make_act_ast(matched_tl)

        for cxn in constructions:
            if cxn.intent_type not in ("act", "query") and cxn.trigger_tokens:
                if any(tok.lower() in text_lower for tok in cxn.trigger_tokens):
                    return SemanticAST(
                        intent=IntentType(cxn.intent_type),
                        subject=text[:100],
                        action=str(cxn.semantic_mapping.get("default_action", cxn.intent_type)),
                        raw_text=text,
                    )

        if "?" not in text and (m := _LEADING_VERB_RE.match(text_lower)):
            verb, rest = m.group(1), m.group(2).strip()
            has_art, target = (
                any(rest.startswith(a) for a in articles),
                _strip_prefix(rest, articles),
            )
            known_verb = (
                self._find_skill_for_verb(verb) is not None
                or self._find_definition_text(verb) is not None
            )
            if verb not in q_starters and (has_art or known_verb or self._is_known_target(target)):
                return SemanticAST(
                    intent=IntentType.ACT,
                    subject=target,
                    action=verb,
                    slots=[
                        SemanticSlot("action", verb),
                        SemanticSlot("target", target),
                        SemanticSlot("unknown_verb", verb),
                    ],
                    raw_text=text,
                )

        return SemanticAST(
            intent=IntentType.QUERY, subject=text[:100], action="evaluate_prompt", raw_text=text
        )

    def compile_text(
        self,
        text: str,
        author: str = "user",
        optimize: bool = False,
        concept_store: WorkingConceptStore | None = None,
        language: str | None = None,
    ) -> InstructionStream:
        """Compile single or multi-clause instructions into an executable `InstructionStream`."""
        lang = language or self._language
        clauses = [text]
        if lang == "en":
            from hbllm.hcir.language.core.discourse import DiscourseSegmenter

            if len(segmented := DiscourseSegmenter.segment(text)) >= 2:
                clauses = segmented
        if len(clauses) == 1:
            return self.compile(
                self.parse_utterance(clauses[0], language=lang),
                author=author,
                optimize=optimize,
                concept_store=concept_store,
            )
        combined: list[Instruction] = []
        has_return = False
        for clause in clauses:
            for ins in self.compile(
                self.parse_utterance(clause, language=lang),
                author=author,
                optimize=optimize,
                concept_store=concept_store,
            ).instructions:
                if ins.opcode == Opcode.RETURN:
                    has_return = True
                else:
                    combined.append(ins)
        if has_return:
            combined.append(_ins(Opcode.RETURN, 0, **{"yield": ["$r1"]}))
        return InstructionStream(
            author=author, description=f"Composed: {text[:80]}", instructions=combined
        )

    def interpret_dialogue(
        self, text: str, speaker_id: str = "user", concept_store: WorkingConceptStore | None = None
    ) -> DialogueInterpretationResult:
        """Parse natural language dialogue into a structured `DialogueInterpretationResult`."""
        ast = self.parse_utterance(text)
        if (store := concept_store or self._concept_store) is not None:
            self.resolve_grounded_entities(ast, concept_store=store)
        d_act = {
            IntentType.ACT: DialogueAct.COMMAND,
            IntentType.QUERY: DialogueAct.QUESTION,
            IntentType.LEARN: DialogueAct.INFORM,
        }.get(ast.intent, DialogueAct.REQUEST)
        act_props: list[ActionIntentProposal] = []
        goal_props: list[GoalProposal] = []
        if ast.action or ast.intent == IntentType.ACT:
            t_ref = next(iter(ast.grounded_entities.values()), None)
            params: dict[str, Any] = {"subject": ast.subject, "raw_text": ast.raw_text}
            if (d := ast.get_slot("direction")) is not None:
                params["direction"] = d
            act_props.append(
                ActionIntentProposal(
                    intent_id=f"act_prop_{uuid.uuid4().hex[:8]}",
                    action_type=ast.action or "execute",
                    target_ref=t_ref,
                    parameters=params,
                )
            )
            goal_props.append(
                GoalProposal(
                    proposal_id=f"goal_prop_{uuid.uuid4().hex[:8]}",
                    goal_intent=ast.action or "act",
                    target_entity=t_ref,
                    parameters=params,
                    priority=0.8,
                )
            )
        return DialogueInterpretationResult(
            utterance_id=f"utt_{uuid.uuid4().hex[:8]}",
            dialogue_act=d_act,
            goal_proposals=tuple(goal_props),
            action_proposals=tuple(act_props),
            confidence=0.5
            if any(isinstance(r, UnresolvedRef) for r in ast.grounded_entities.values())
            else 1.0,
        )

    def evaluate_grounded_expression(
        self, ast: SemanticAST, theory: Any, grid: Any | None = None
    ) -> list[Any]:
        """Evaluate compositional spatial expression using `lexical_rules` from construction seed."""
        import numpy as np

        rules = load_constructions_spec(self._language).get("lexical_rules", {})
        articles = list(rules.get("articles", ()))
        text = (ast.raw_text or ast.subject).lower()
        detected_rel, subj_name, ref_name = None, text, ""
        for phrase, rel in rules.get("spatial_relations", {}).items():
            if phrase in text:
                detected_rel = rel
                subj_name, ref_name = (p.strip() for p in text.split(phrase, 1))
                break
        subj_name, ref_name = _strip_prefix(subj_name, articles), _strip_prefix(ref_name, articles)
        if grid is not None and isinstance(grid, np.ndarray) and detected_rel and ref_name:
            sf, rf = (
                self._resolve_feature_for_label(subj_name, theory),
                self._resolve_feature_for_label(ref_name, theory),
            )
            if sf is not None and rf is not None:
                sc_list = list(zip(*((grid == sf).nonzero())))
                rc_list = list(zip(*((grid == rf).nonzero())))
                matches = [
                    (int(sr), int(sc))
                    for sr, sc in sc_list
                    for rr, rc in rc_list
                    if (
                        (detected_rel == "LEFT_OF" and sc < rc and abs(sr - rr) <= 2)
                        or (detected_rel == "RIGHT_OF" and sc > rc and abs(sr - rr) <= 2)
                        or (detected_rel == "ABOVE" and sr < rr and abs(sc - rc) <= 2)
                        or (detected_rel == "BELOW" and sr > rr and abs(sc - rc) <= 2)
                        or (detected_rel == "ADJACENT" and abs(sr - rr) + abs(sc - rc) <= 1)
                    )
                ]
                if matches:
                    return matches
        return [subj_name] if subj_name else []

    def _resolve_feature_for_label(self, label: str, theory: Any) -> Any | None:
        lbl = label.lower().strip()
        if self._concept_store is not None:
            for concept in self._concept_store.get_all():
                for cl in (k.lower() for k in concept.linguistic_aliases):
                    if cl and cl in lbl and (m := _NUMERIC_LABEL_RE.search(cl)):
                        return int(m.group(1))
        if m_sym := _NUMERIC_LABEL_RE.search(lbl):
            return int(m_sym.group(1))
        rules = load_constructions_spec(self._language).get("lexical_rules", {})
        for cname, ccode in rules.get("perceptual_color_defaults", {}).items():
            if str(cname) in lbl:
                return int(ccode)
        if theory is not None:
            rh = rules.get("role_hints", {})
            if any(w in lbl for w in rh.get("tool", ())) and getattr(
                theory, "tool_barrier_affinities", None
            ):
                return next(iter(theory.tool_barrier_affinities))
            if any(w in lbl for w in rh.get("barrier", ())) and getattr(
                theory, "barrier_features", None
            ):
                return next(iter(theory.barrier_features))
            if any(w in lbl for w in rh.get("goal", ())) and getattr(theory, "goal_features", None):
                return next(iter(theory.goal_features))
        return None

    def compile_utterance_to_causal_hypothesis(
        self, utterance: str, theory: Any | None = None
    ) -> Any:
        """Compile natural language causal utterance into a `CausalPredicate` via seed connectives."""
        from hbllm.hcir.world.causal_discovery import CausalPredicate

        rules = load_constructions_spec(self._language).get("lexical_rules", {})
        text, premise, relation, effect = utterance.lower().strip(), "", "causes", ""
        for kw, rel in rules.get("causal_connectives", ()):
            if str(kw) in text:
                premise, effect = (p.strip() for p in text.split(str(kw), 1))
                relation = str(rel)
                break
        if_then = list(rules.get("if_then", ("if ", " then ", "implies")))
        if not premise and if_then[0] in text and if_then[1] in text:
            premise, effect = (p.strip() for p in text.split(if_then[0], 1)[1].split(if_then[1], 1))
            relation = str(if_then[2])
        if not premise:
            premise, relation, effect = text, "asserted_rule", "true"
        articles = list(rules.get("articles", ()))
        premise, effect = _strip_prefix(premise, articles), _strip_prefix(effect, articles)
        pred = CausalPredicate(
            variable=f"rule:{premise.replace(' ', '_')}->{effect.replace(' ', '_')}",
            operator="==",
            value=effect,
            relation=relation,
            target_entity=premise,
        )
        pred.causal_status = "HYPOTHESISED"
        if theory is not None and callable(getattr(theory, "record_causal_predicate", None)):
            theory.record_causal_predicate(pred)
            if relation in ("requires", "unlocks", "clears"):
                pf, ef = (
                    self._resolve_feature_for_label(premise, theory),
                    self._resolve_feature_for_label(effect, theory),
                )
                if pf is not None and ef is not None:
                    theory.record_tool_barrier_affinity(ef, pf)
        return pred
