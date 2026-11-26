"""Wake–Sleep Anti-Unification & MDL Library Learner (Phase M4).

Synthesizes reusable parameterized homoiconic `SkillNode`s from recurring wake-phase
`InstructionStream` execution traces via deterministic Plotkin Least General
Generalization (LGG) anti-unification, gates promotion by Minimum Description Length
(MDL) compression gain, strips episode-local surface identifiers across seed
boundaries, and rewrites future bytecode streams into `Opcode.CALL` sites.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import NodeLifecycle, SkillNode
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope

logger = logging.getLogger(__name__)

_SURFACE_KEYS = frozenset(
    {
        "pos",
        "position",
        "coord",
        "target_pos",
        "target_positions",
        "coordinates",
        "bottleneck_state",
        "feature",
        "color",
        "required_feature",
        "initial_grid_hash",
    }
)
_ENTITY_ID_RE = re.compile(
    r"^(entity|obj|object|key|door|item|portal|box|beacon|target)_[A-Za-z0-9_]+$"
)


def _make_instruction(opcode: Opcode, cost_estimate: int = 0, /, **params: Any) -> Instruction:
    return Instruction(opcode=opcode, params=params, cost_estimate=cost_estimate)


def _is_symbolic_var(value: Any) -> bool:
    return isinstance(value, str) and (
        value.startswith("?") or (value.startswith("$r") and len(value) == 3)
    )


def _is_coordinate_pair(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(coord, (int, float)) and not isinstance(coord, bool) for coord in value)
    )


def _count_param_leaves(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_count_param_leaves(v) for v in value.values()) if value else 1
    if isinstance(value, (list, tuple)):
        return sum(_count_param_leaves(item) for item in value) if value else 1
    return 1


def instruction_dl(instruction: Instruction) -> float:
    """Compute description length of a single HCIR bytecode instruction."""
    return 1.0 + 0.5 * float(_count_param_leaves(instruction.params))


def stream_dl(stream: InstructionStream) -> float:
    """Compute total description length of non-RETURN instructions in an `InstructionStream`."""
    return sum(instruction_dl(ins) for ins in stream.instructions if ins.opcode != Opcode.RETURN)


def compute_mdl_gain(
    body: Sequence[Instruction],
    param_count: int,
    support_count: int,
    header_cost: float = 1.5,
) -> tuple[float, float, float]:
    """Compute `(raw_cost, compressed_cost, mdl_gain)` for promoting a macro candidate."""
    motif_cost = sum(instruction_dl(ins) for ins in body if ins.opcode != Opcode.RETURN)
    support = float(max(1, support_count))
    raw_cost = support * motif_cost
    compressed_cost = header_cost + motif_cost + support * (1.0 + 0.25 * param_count)
    return raw_cost, compressed_cost, raw_cost - compressed_cost


def sanitize_stream_for_cross_seed(
    stream: InstructionStream,
) -> tuple[InstructionStream, list[dict[str, Any]]]:
    """Strip episode-local surface IDs, colors, and coordinates into typed `?var` parameters."""
    parameters: list[dict[str, Any]] = []
    seen_vars: dict[tuple[str, str], str] = {}

    def _alloc_var(key: str, role: str) -> str:
        cache_key = (key, role)
        if cache_key not in seen_vars:
            entity_match = _ENTITY_ID_RE.match(key)
            raw_base = entity_match.group(1) if entity_match else (role or key or "arg")
            clean_base = re.sub(r"[^a-z0-9_]", "", raw_base.lower()) or "arg"
            param_name = f"{clean_base}_{len(parameters)}"
            parameters.append(
                {"name": param_name, "role": role, "required": False, "default": None}
            )
            seen_vars[cache_key] = f"?{param_name}"
        return seen_vars[cache_key]

    def _clean(value: Any, key: str = "", role: str = "") -> Any:
        if _is_symbolic_var(value):
            return value
        if isinstance(value, Mapping):
            local_role = str(value.get("role", role) or "")
            cleaned: dict[str, Any] = {}
            for k, v in value.items():
                key_str = str(k)
                if key_str in _SURFACE_KEYS or _is_coordinate_pair(v):
                    if local_role or key_str != "id":
                        cleaned[key_str] = _alloc_var(key_str, local_role)
                else:
                    cleaned[key_str] = _clean(v, key_str, local_role)
            return cleaned
        if isinstance(value, (list, tuple)):
            if _is_coordinate_pair(value):
                return _alloc_var(key or "coord", role)
            return type(value)(_clean(item, key, role) for item in value)
        if isinstance(value, str) and _ENTITY_ID_RE.match(value):
            return _alloc_var(value, role)
        return value

    sanitized_instructions = [
        ins.model_copy(update={"params": _clean(ins.params)}) for ins in stream.instructions
    ]
    return (
        stream.model_copy(
            update={"instructions": sanitized_instructions, "description": ""},
            deep=True,
        ),
        parameters,
    )


def _longest_common_subsequence(
    seq_a: Sequence[Instruction], seq_b: Sequence[Instruction]
) -> tuple[list[Instruction], list[Instruction]]:
    """Find longest contiguous opcode+capability alignment between two instruction slices."""

    def _signatures(seq: Sequence[Instruction]) -> list[tuple[Opcode, str]]:
        return [(ins.opcode, str(ins.params.get("capability", ""))) for ins in seq]

    sig_a, sig_b = _signatures(seq_a), _signatures(seq_b)
    if sig_a == sig_b:
        return list(seq_a), list(seq_b)

    best_len, best_i, best_j = 0, 0, 0
    for i in range(len(sig_a)):
        for j in range(len(sig_b)):
            k = 0
            while i + k < len(sig_a) and j + k < len(sig_b) and sig_a[i + k] == sig_b[j + k]:
                k += 1
            if k > best_len:
                best_len, best_i, best_j = k, i, j
    return list(seq_a[best_i : best_i + best_len]), list(seq_b[best_j : best_j + best_len])


@dataclass
class WakeTraceRecord:
    """Recorded wake-phase bytecode execution trace for offline consolidation."""

    trace_id: str
    stream: InstructionStream
    success: bool = True
    episode_epoch: int = 0
    tenant_id: str = "default"
    capabilities_used: list[str] = field(default_factory=list)
    sanitized_for_transfer: bool = False


@dataclass
class AntiUnifiedCandidate:
    """Parameterized macro candidate synthesized by Plotkin LGG anti-unification."""

    candidate_id: str
    skill_name: str
    stream: InstructionStream
    parameters: list[dict[str, Any]]
    precondition_pattern: dict[str, Any]
    postcondition_pattern: dict[str, Any]
    support_count: int
    raw_cost: float
    compressed_cost: float
    mdl_gain: float

    def to_skill_node(self, tenant_id: str = "default") -> SkillNode:
        """Materialize candidate as a homoiconic executable `SkillNode`."""
        description = (
            f"Wake-sleep macro '{self.skill_name}' "
            f"(support={self.support_count}, mdl={self.mdl_gain:.2f})"
        )
        return SkillNode(
            id=f"skill_ws_{self.candidate_id}",
            skill_name=self.skill_name,
            description=description,
            parameters=list(self.parameters),
            precondition_pattern=dict(self.precondition_pattern),
            postcondition_pattern=dict(self.postcondition_pattern),
            bytecode_stream=self.stream.model_dump(),
            alpha_success=1.0 + float(self.support_count),
            beta_failure=1.0,
            success_rate=1.0,
            invocation_count=self.support_count,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by="library_learner"),
            scope=Scope(tenant_id=tenant_id),
            tags=["wake_sleep_macro", "anti_unified", self.skill_name],
            properties={
                "candidate_id": self.candidate_id,
                "mdl_gain": self.mdl_gain,
                "support_count": self.support_count,
            },
        )


def anti_unify_streams(
    stream_a: InstructionStream,
    stream_b: InstructionStream,
    *,
    skill_name: str = "",
    strip_surface_ids: bool = True,
    support_count: int = 2,
) -> AntiUnifiedCandidate | None:
    """Anti-unify two successful instruction streams via memoized Plotkin LGG."""
    seq_a, seq_b = _longest_common_subsequence(
        [ins for ins in stream_a.instructions if ins.opcode != Opcode.RETURN],
        [ins for ins in stream_b.instructions if ins.opcode != Opcode.RETURN],
    )
    if len(seq_a) < 2 or len(seq_a) != len(seq_b):
        return None

    memo: dict[tuple[str, str, str], str] = {}
    parameters: list[dict[str, Any]] = []

    def _lgg(val_a: Any, val_b: Any, key: str = "", role: str = "") -> Any:
        if (
            isinstance(val_a, Mapping)
            and isinstance(val_b, Mapping)
            and val_a.keys() == val_b.keys()
        ):
            local_role = str(val_a.get("role", val_b.get("role", role)) or "")
            return {str(k): _lgg(val_a[k], val_b[k], str(k), local_role) for k in val_a}
        if isinstance(val_a, list) and isinstance(val_b, list) and len(val_a) == len(val_b):
            return [_lgg(item_a, item_b, key, role) for item_a, item_b in zip(val_a, val_b)]

        is_surface = (
            strip_surface_ids
            and not _is_symbolic_var(val_a)
            and (
                key in _SURFACE_KEYS
                or _is_coordinate_pair(val_a)
                or (isinstance(val_a, str) and bool(_ENTITY_ID_RE.match(val_a)))
            )
        )
        if val_a == val_b and (not is_surface or _is_symbolic_var(val_a)):
            return val_a

        # Memoizing on (repr(val_a), repr(val_b), role) ensures recurring concrete IDs
        # across UNIFY and EXECUTE operands bind consistently to the same ?var.
        memo_key = (repr(val_a), repr(val_b), role)
        if memo_key not in memo:
            param_name = f"target_{len(parameters)}"
            memo[memo_key] = f"?{param_name}"
            parameters.append(
                {"name": param_name, "role": role, "required": False, "default": None}
            )
        return memo[memo_key]

    body = [
        ins_a.model_copy(update={"params": _lgg(ins_a.params, ins_b.params)})
        for ins_a, ins_b in zip(seq_a, seq_b)
    ]
    capabilities = [
        str(ins.params["capability"])
        for ins in seq_a
        if ins.opcode == Opcode.EXECUTE and "capability" in ins.params
    ]
    raw_cost, compressed_cost, mdl_gain = compute_mdl_gain(body, len(parameters), support_count)

    # Guarantee every synthesized macro grounds its primary parameter register before EXECUTE.
    if not any(ins.opcode == Opcode.UNIFY for ins in body) and parameters:
        primary = parameters[0]
        unify_pattern = {
            "id": f"?{primary['name']}",
            **({"role": primary["role"]} if primary["role"] else {}),
        }
        body.insert(0, _make_instruction(Opcode.UNIFY, 5, pattern=unify_pattern, bind="$r0"))

    macro_name = (
        skill_name or f"macro_{'_'.join(capabilities[:2]) or 'routine'}_{uuid.uuid4().hex[:6]}"
    )
    stream = InstructionStream(
        author="library_learner",
        description=f"Anti-unified macro {macro_name}",
        instructions=[*body, _make_instruction(Opcode.RETURN, 0, **{"yield": ["$r0"]})],
    )
    candidate_id = stream.compute_hash()[:12]
    precondition = {
        "capabilities": list(dict.fromkeys(capabilities)),
        "roles": [p["role"] for p in parameters if p["role"]],
    }
    return AntiUnifiedCandidate(
        candidate_id=candidate_id,
        skill_name=macro_name,
        stream=stream,
        parameters=parameters,
        precondition_pattern=precondition,
        postcondition_pattern={"skill_name": macro_name, "macro_hash": candidate_id},
        support_count=support_count,
        raw_cost=raw_cost,
        compressed_cost=compressed_cost,
        mdl_gain=mdl_gain,
    )


def _match_subsequence(
    subsequence: Sequence[Instruction],
    template: Sequence[Instruction],
    param_names: set[str],
) -> dict[str, Any] | None:
    """Attempt consistent positional+structural unification of `subsequence` against `template`."""
    if len(subsequence) != len(template):
        return None
    bindings: dict[str, Any] = {}

    def _unify_value(concrete: Any, pattern: Any) -> bool:
        if isinstance(pattern, str) and pattern.startswith("?") and pattern[1:] in param_names:
            return bindings.setdefault(pattern[1:], concrete) == concrete
        if isinstance(concrete, Mapping) and isinstance(pattern, Mapping):
            return concrete.keys() == pattern.keys() and all(
                _unify_value(concrete[k], pattern[k]) for k in pattern
            )
        if isinstance(concrete, list) and isinstance(pattern, list):
            return len(concrete) == len(pattern) and all(
                _unify_value(c_item, p_item) for c_item, p_item in zip(concrete, pattern)
            )
        return concrete == pattern

    matched = all(
        src.opcode == tmpl.opcode and _unify_value(src.params, tmpl.params)
        for src, tmpl in zip(subsequence, template)
    )
    return bindings if matched else None


def rewrite_stream_with_library(
    stream: InstructionStream, skills: Sequence[SkillNode]
) -> InstructionStream:
    """Replace recurring instruction subsequences with `Opcode.CALL` to promoted `SkillNode`s."""
    parsed_templates: list[tuple[SkillNode, list[Instruction], set[str]]] = []
    for skill in skills:
        if skill.bytecode_stream is None:
            continue
        param_names = {
            str(param["name"])
            for param in (skill.parameters or [])
            if isinstance(param, Mapping) and "name" in param
        }
        core = [
            ins
            for ins in InstructionStream.model_validate(skill.bytecode_stream).instructions
            if ins.opcode != Opcode.RETURN
        ]
        if len(core) >= 2:
            parsed_templates.append((skill, core, param_names))
        if len(core) >= 3 and core[0].opcode == Opcode.UNIFY:
            parsed_templates.append((skill, core[1:], param_names))

    parsed_templates.sort(key=lambda entry: len(entry[1]), reverse=True)
    instructions = list(stream.instructions)
    rewritten: list[Instruction] = []
    idx = 0
    while idx < len(instructions):
        matched_call: tuple[SkillNode, int, dict[str, Any]] | None = None
        for skill, core, param_names in parsed_templates:
            if idx + len(core) <= len(instructions):
                args = _match_subsequence(instructions[idx : idx + len(core)], core, param_names)
                if args is not None:
                    matched_call = (skill, len(core), args)
                    break
        if matched_call is not None:
            skill, span_len, args = matched_call
            rewritten.append(
                _make_instruction(Opcode.CALL, 10, skill_id=skill.id, args=args, bind="$r0")
            )
            idx += span_len
        else:
            rewritten.append(instructions[idx])
            idx += 1

    return stream.model_copy(update={"instructions": rewritten}, deep=True)


class LibraryLearner:
    """Deterministic Wake–Sleep Anti-Unification & MDL Skill Consolidation Engine."""

    def __init__(
        self,
        *,
        tiered_workspace: Any | None = None,
        tenant_id: str = "default",
        min_mdl_gain: float = 0.01,
    ) -> None:
        self._tiered_workspace = tiered_workspace
        self._tenant_id = tenant_id
        self.min_mdl_gain = min_mdl_gain
        self._episodic_cortex: Any | None = None
        self._ingested_trajectory_count: int = 0
        self.wake_traces: list[WakeTraceRecord] = []
        self.promoted_skills: dict[str, SkillNode] = {}
        self.consolidation_runs: int = 0

    def attach_workspace(self, workspace: Any, tenant_id: str | None = None) -> None:
        self._tiered_workspace = workspace
        if tenant_id:
            self._tenant_id = tenant_id

    def attach_episodic_cortex(self, cortex: Any) -> None:
        self._episodic_cortex = cortex

    def record_stream(
        self,
        stream: InstructionStream,
        *,
        success: bool = True,
        episode_epoch: int = 0,
        tenant_id: str = "default",
        capabilities_used: Sequence[str] = (),
    ) -> None:
        """Buffer a wake-phase instruction stream for offline anti-unification."""
        if not success or len(stream.instructions) < 2:
            return
        caps = list(capabilities_used) or [
            str(ins.params["capability"])
            for ins in stream.instructions
            if ins.opcode == Opcode.EXECUTE and "capability" in ins.params
        ]
        trace_id = f"wt_{len(self.wake_traces)}_{uuid.uuid4().hex[:6]}"
        self.wake_traces.append(
            WakeTraceRecord(
                trace_id=trace_id,
                stream=stream.model_copy(deep=True),
                success=True,
                episode_epoch=episode_epoch,
                tenant_id=tenant_id,
                capabilities_used=caps,
            )
        )

    def record_receipt(
        self,
        receipt: ExecutionReceipt,
        source_stream: InstructionStream | None = None,
        *,
        episode_epoch: int = 0,
        tenant_id: str = "default",
    ) -> None:
        """Buffer a successful ExecutionReceipt + its InstructionStream from wake phase."""
        if receipt.success and source_stream is not None:
            self.record_stream(
                source_stream,
                success=True,
                episode_epoch=episode_epoch,
                tenant_id=tenant_id,
                capabilities_used=receipt.capabilities_used,
            )

    @staticmethod
    def trajectory_to_stream(
        trajectory: Sequence[Any], *, episode_epoch: int = 0, abstract_surface: bool = True
    ) -> InstructionStream | None:
        """Convert a winning episodic trajectory into an allowlisted ISA v2 InstructionStream."""
        from hbllm.hcir.world.episodic.models import compress_action_trajectory

        compressed, _ = compress_action_trajectory(trajectory) if trajectory else ([], [])
        actions = compressed or [
            getattr(step, "action_id", None)
            for step in trajectory
            if getattr(step, "action_id", None) is not None
        ]
        if len(actions) < 2:
            return None

        head = actions[:6]
        instructions = [
            _make_instruction(
                Opcode.UNIFY, 5, pattern={"role": "goal", "id": "?goal_0"}, bind="$r0"
            )
        ]
        for idx, action in enumerate(head):
            capability = "motor.interact" if idx == len(head) - 1 else "motor.navigate"
            target = (
                "?goal_0" if (abstract_surface or idx == len(head) - 1) else f"step_{idx}_{action}"
            )
            instructions.append(
                _make_instruction(
                    Opcode.EXECUTE,
                    10,
                    capability=capability,
                    params={"target": target, "action": str(action)},
                )
            )
        instructions.append(_make_instruction(Opcode.RETURN, 0, **{"yield": ["$r0"]}))
        return InstructionStream(
            author="episodic_wake",
            description=f"Winning episodic trajectory epoch {episode_epoch}",
            instructions=instructions,
        )

    def ingest_episodic_trajectories(
        self,
        trajectories: Sequence[Sequence[Any]],
        *,
        episode_epoch: int = 0,
        abstract_surface: bool = True,
    ) -> int:
        """Ingest newly completed winning trajectories from HippocampalEpisodicCortex."""
        added = 0
        while self._ingested_trajectory_count < len(trajectories):
            trajectory = trajectories[self._ingested_trajectory_count]
            self._ingested_trajectory_count += 1
            stream = self.trajectory_to_stream(
                trajectory, episode_epoch=episode_epoch, abstract_surface=abstract_surface
            )
            if stream is not None:
                self.record_stream(
                    stream, success=True, episode_epoch=episode_epoch, tenant_id=self._tenant_id
                )
                added += 1
        return added

    def _sync_cortex(self) -> None:
        if self._episodic_cortex is not None:
            trajectories = getattr(self._episodic_cortex, "successful_trajectories", None) or []
            self.ingest_episodic_trajectories(trajectories, abstract_surface=True)

    def on_boundary(self, boundary: Any) -> None:
        """Sanitize buffered traces across seeds so surface IDs/colors never poison transfer."""
        from hbllm.hcir.world.percept import resolve_boundary_flags

        if resolve_boundary_flags(boundary)[2]:
            return
        for record in self.wake_traces:
            if not record.sanitized_for_transfer:
                record.stream = sanitize_stream_for_cross_seed(record.stream)[0]
                record.sanitized_for_transfer = True
        self._sync_cortex()
        if self._tiered_workspace is not None:
            self.sleep_consolidation(self._tiered_workspace, tenant_id=self._tenant_id)

    def sleep_consolidation(
        self, tiered_workspace: Any | None, tenant_id: str = "default"
    ) -> dict[str, Any]:
        """Anti-unify wake traces, gate by MDL compression, and commit SkillNodes transactionally."""
        self._sync_cortex()
        traces = [record for record in self.wake_traces if record.success]
        candidates: dict[str, AntiUnifiedCandidate] = {}

        for idx, rec_a in enumerate(traces):
            for rec_b in traces[idx + 1 :]:
                cand = anti_unify_streams(
                    rec_a.stream,
                    rec_b.stream,
                    strip_surface_ids=True,
                    support_count=max(2, len(traces)),
                )
                if cand is None:
                    continue
                if cand.candidate_id in candidates:
                    prev = candidates[cand.candidate_id]
                    prev.support_count += 1
                    prev.raw_cost, prev.compressed_cost, prev.mdl_gain = compute_mdl_gain(
                        prev.stream.instructions, len(prev.parameters), prev.support_count
                    )
                else:
                    candidates[cand.candidate_id] = cand

        newly_promoted: list[SkillNode] = []
        total_mdl_gain = 0.0
        ops: list[TransactionOperation] = []

        for cid, cand in candidates.items():
            if cand.mdl_gain < self.min_mdl_gain:
                continue
            skill_id = f"skill_ws_{cid}"
            if skill_id in self.promoted_skills:
                existing = self.promoted_skills[skill_id]
                node = existing.model_copy(
                    update={
                        "alpha_success": existing.alpha_success + 1.0,
                        "invocation_count": existing.invocation_count + 1,
                    }
                )
            else:
                node = cand.to_skill_node(tenant_id=tenant_id)
                newly_promoted.append(node)
                total_mdl_gain += cand.mdl_gain
            self.promoted_skills[node.id] = node
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node.id,
                    node_data=node.model_dump(),
                )
            )

        if ops and tiered_workspace is not None:
            grant = getattr(tiered_workspace, "_system_grant", None)
            brain = getattr(tiered_workspace, "brain", None)
            persistent = getattr(tiered_workspace, "persistent", None)
            targets = (
                (brain, persistent)
                if (brain is not None and persistent is not None)
                else (tiered_workspace,)
            )
            for ws in targets:
                if hasattr(ws, "commit_operations"):
                    ws.commit_operations(
                        ops, author="library_learner", grant=grant, tenant_id=tenant_id
                    )
            if (
                brain is not None
                and persistent is not None
                and hasattr(tiered_workspace, "notify_commit")
            ):
                tiered_workspace.notify_commit()

        self.consolidation_runs += 1
        return {
            "promoted_count": len(newly_promoted),
            "total_skills": len(self.promoted_skills),
            "skill_ids": [skill.id for skill in newly_promoted],
            "total_mdl_gain": round(total_mdl_gain, 4),
            "traces_processed": len(traces),
        }


WakeSleepLibraryLearner = LibraryLearner

__all__ = [
    "AntiUnifiedCandidate",
    "LibraryLearner",
    "WakeSleepLibraryLearner",
    "WakeTraceRecord",
    "anti_unify_streams",
    "compute_mdl_gain",
    "instruction_dl",
    "rewrite_stream_with_library",
    "sanitize_stream_for_cross_seed",
    "stream_dl",
]
