"""HCIR Interpreter — ISA v2 register-unification bytecode virtual machine."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import replace as dc_replace
from typing import Any, Literal

from hbllm.hcir.abi import ExecutionMetrics, ExecutionResult
from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import CognitiveCategory, HCIRNodeType, NodeLifecycle
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.query import GraphQuery
from hbllm.hcir.registers import (
    REGISTER_NAMES,
    CallFrame,
    CallStack,
    RegisterFile,
    VMRuntimeError,
    is_symbolic_var,
    normalize_var_name,
    resolve_operands,
)
from hbllm.hcir.stores import EventType
from hbllm.hcir.transactions import (
    HCIRTransaction,
    TransactionAnnotation,
    TransactionOp,
    TransactionOperation,
)
from hbllm.hcir.unification import unify_dicts, unify_pattern
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)
SyscallHandler = Callable[
    [Instruction, HCIRWorkspaceState, KernelServices], Coroutine[Any, Any, dict[str, Any]]
]
StepHook = Callable[[int, Instruction, dict[str, Any], CallFrame, HCIRWorkspaceState], bool | None]


class SyscallAllowlist:
    """Enforces Layer-0 hardware/sandboxed boundaries on `Opcode.EXECUTE`."""

    ALLOWED_PREFIXES: tuple[str, ...] = tuple(
        "motor. sensor. math. tool.sandboxed. web.sandboxed. memory. world. fs. web. knowledge.".split()
    )
    ALLOWED_EXACT: frozenset[str] = frozenset(
        "code.python_ast git.status shell.exec execution.sandbox_run routing.resolve_route".split()
    )
    DEPRECATED_PREFIXES: tuple[str, ...] = tuple(
        "planning. reasoning. learning. simulation. spatial. motor_".split()
    )
    DEPRECATED_EXACT: frozenset[str] = frozenset(
        "causal_analysis content_generation content_modification planning simulation skill_extraction world_prediction".split()
    )

    def __init__(self, *, strict: bool = False, extra_allowed: set[str] | None = None) -> None:
        self.strict = strict
        self.extra_allowed = set(extra_allowed or ())

    def classify(self, capability: str) -> Literal["allowed", "deprecated", "rejected"]:
        """Classify *capability* into allowed Layer-0 syscall, deprecated macro, or rejected."""
        if not capability:
            return "rejected"
        if (
            capability in self.extra_allowed
            or capability in self.ALLOWED_EXACT
            or capability.startswith(self.ALLOWED_PREFIXES)
        ):
            return "allowed"
        if capability in self.DEPRECATED_EXACT or capability.startswith(self.DEPRECATED_PREFIXES):
            return "rejected" if self.strict else "deprecated"
        return "rejected"


class SyscallDispatcher:
    """Maps the 12 ISA v2 opcodes to async syscall handlers."""

    def __init__(self) -> None:
        self._handlers: dict[Opcode, SyscallHandler] = {}
        self._register_defaults()

    def register(self, opcode: Opcode, handler: SyscallHandler) -> None:
        """Register a syscall handler for an opcode."""
        self._handlers[opcode] = handler

    async def dispatch(
        self, instruction: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
    ) -> dict[str, Any]:
        """Dispatch an instruction to its syscall handler."""
        handler = self._handlers.get(instruction.opcode)
        if handler is None:
            raise ValueError(f"No syscall handler for opcode: {instruction.opcode}")
        return await handler(instruction, workspace, services)

    def _register_defaults(self) -> None:
        """Register all 12 ISA v2 opcode handlers."""
        defaults = {
            Opcode.ASSERT: sys_assert,
            Opcode.RETRACT: sys_retract,
            Opcode.QUERY: sys_query,
            Opcode.EXECUTE: sys_execute,
            Opcode.WAIT: sys_wait,
            Opcode.FORK: sys_fork,
            Opcode.MERGE: sys_merge,
            Opcode.ROLLBACK: sys_rollback,
            Opcode.UNIFY: sys_unify,
            Opcode.BRANCH: sys_branch,
            Opcode.CALL: sys_call,
            Opcode.RETURN: sys_return,
        }
        for op, fn in defaults.items():
            self.register(op, fn)


async def sys_assert(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """ASSERT: inject a node or edge into the workspace."""
    node_data, edge_data = ins.params.get("node_data"), ins.params.get("edge_data")
    author = str(ins.params.get("author", "interpreter"))
    if node_data:
        node_id = node_data.get("id")
        op = (
            TransactionOp.UPSERT_NODE
            if (node_id and workspace.graph.has_node(node_id))
            else TransactionOp.ADD_NODE
        )
        tx = HCIRTransaction(
            author=author,
            operations=[TransactionOperation(op=op, node_id=node_id, node_data=node_data)],
        )
        result = services.transaction_manager.commit(tx)
        cid = node_id or (tx.operations[0].node_id if tx.operations else None)
        return {"committed": result.is_committed, "tx_id": result.id, "node_id": cid}
    if edge_data:
        edge_id = edge_data.get("id")
        tx = HCIRTransaction(
            author=author,
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_EDGE, edge_id=edge_id, edge_data=edge_data
                )
            ],
        )
        result = services.transaction_manager.commit(tx)
        return {"committed": result.is_committed, "tx_id": result.id, "edge_id": edge_id}
    return {"error": "ASSERT requires 'node_data' or 'edge_data'"}


async def sys_retract(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """RETRACT: remove a node or edge from the workspace."""
    node_id, edge_id = ins.params.get("node_id"), ins.params.get("edge_id")
    ops = []
    if node_id:
        ops.append(TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id=node_id))
    if edge_id:
        ops.append(TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id=edge_id))
    if not ops:
        return {"error": "RETRACT requires 'node_id' or 'edge_id'"}
    result = services.transaction_manager.commit(
        HCIRTransaction(author=str(ins.params.get("author", "interpreter")), operations=ops)
    )
    return {"committed": result.is_committed, "tx_id": result.id}


async def sys_query(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """QUERY: retrieve nodes matching structural constraints."""
    query = GraphQuery(
        node_type=HCIRNodeType(ins.params["node_type"]) if ins.params.get("node_type") else None,
        category=CognitiveCategory(ins.params["category"]) if ins.params.get("category") else None,
        lifecycle=NodeLifecycle(ins.params["lifecycle"]) if ins.params.get("lifecycle") else None,
        scope_tenant=ins.params.get("scope_tenant"),
        text_contains=ins.params.get("text_contains"),
        min_confidence=ins.params.get("min_confidence"),
        limit=ins.params.get("limit", 100),
    )
    res = workspace.query(query)
    return {
        "matches": [n.model_dump() for n in res.nodes],
        "total": res.total_matches,
        "truncated": res.truncated,
    }


async def sys_execute(
    ins: Instruction,
    workspace: HCIRWorkspaceState,
    services: KernelServices,
    *,
    allowlist: SyscallAllowlist | None = None,
) -> dict[str, Any]:
    """EXECUTE: dispatch a capability through CapabilityResolver gated by SyscallAllowlist."""
    capability, exec_params = str(ins.params.get("capability", "")), ins.params.get("params", {})
    policy = allowlist or SyscallAllowlist(strict=False)
    verdict = policy.classify(capability)
    if verdict == "rejected" and policy.strict:
        return {
            "error": f"Capability '{capability}' rejected by SyscallAllowlist",
            "blocked_by_allowlist": True,
        }

    executor = await services.capability_resolver.resolve(capability)
    if executor is None:
        out: dict[str, Any] = {
            "error": (
                f"Capability '{capability}' rejected by SyscallAllowlist"
                if verdict == "rejected"
                else f"No executor for capability: {capability}"
            )
        }
        if verdict == "rejected":
            out["blocked_by_allowlist"] = True
        return out
    if verdict != "allowed":
        logger.warning(
            "Deprecated/non-Layer-0 EXECUTE capability '%s'; migrate to ISA v2 bytecode", capability
        )
    try:
        result = await executor.execute(exec_params)
        payload: dict[str, Any] = {"result": result, "capability": capability}
        if verdict != "allowed":
            payload["deprecated_capability"] = True
        return payload
    except Exception as exc:
        return {"error": str(exc)}


async def sys_wait(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """WAIT: block execution until a condition resolves or event fires."""
    event_type = ins.params.get("event_type") or ins.params.get("event")
    timeout_ms = ins.params.get("timeout_ms", 1000)
    if event_type and hasattr(services, "event_bus") and services.event_bus is not None:
        try:
            evt = await services.event_bus.wait_for(event_type=event_type, timeout_ms=timeout_ms)
            return {
                "waited": True,
                "event": evt.to_dict() if hasattr(evt, "to_dict") else str(evt),
                "resolved": True,
            }
        except TimeoutError:
            return {"waited": True, "event_type": event_type, "resolved": False, "timeout": True}
        except Exception as exc:
            return {"waited": True, "event_type": event_type, "resolved": False, "error": str(exc)}

    condition = ins.params.get("condition", "")
    if condition:
        try:
            node_type = HCIRNodeType(condition)
        except ValueError:
            node_type = None
        if node_type is not None:
            resolved = workspace.query(GraphQuery(node_type=node_type, limit=1)).total_matches > 0
            return {"waited": True, "condition": condition, "resolved": resolved}
    return {
        "waited": False,
        "condition": condition,
        "resolved": False,
        "error": "WAIT requires 'event_type' or a resolvable 'condition'",
    }


async def sys_fork(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """FORK: create a simulation branch."""
    branch_name = ins.params.get("branch_name", "simulation")
    try:
        workspace.fork(branch_name)
        workspace.snapshot_manager.record_kernel_event(
            EventType.SIMULATION_STARTED, {"branch": branch_name}
        )
        return {"forked": True, "branch": branch_name}
    except ValueError as exc:
        return {"error": str(exc)}


async def sys_merge(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """MERGE: commit simulated modifications back to the parent branch transactionally."""
    branch_name = ins.params.get("branch_name", "simulation")
    if workspace.get_branch(branch_name) is None:
        return {"error": f"Branch '{branch_name}' not found"}
    ops = workspace.diff_branch(branch_name)
    merged = workspace.merge_branch(
        branch_name,
        transaction_manager=services.transaction_manager,
        author=ins.params.get("author", "merge"),
    )
    if not merged:
        return {
            "error": f"Merge of branch '{branch_name}' rejected by transaction pipeline",
            "branch": branch_name,
            "merged_count": 0,
        }
    workspace.snapshot_manager.record_kernel_event(
        EventType.SIMULATION_FINISHED, {"branch": branch_name, "merged_count": len(ops)}
    )
    return {"merged": True, "branch": branch_name, "merged_count": len(ops)}


async def sys_rollback(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """ROLLBACK: drop a simulation branch or fail-closed restore to a prior snapshot."""
    target_version, branch_name = ins.params.get("target_version"), ins.params.get("branch_name")
    if target_version is None and branch_name:
        if workspace.get_branch(str(branch_name)) is None:
            return {"rolled_back": False, "error": f"Branch '{branch_name}' not found"}
        workspace.drop_branch(str(branch_name))
        return {
            "rolled_back": True,
            "branch_dropped": str(branch_name),
            "state_matches_snapshot": True,
        }
    if target_version is None:
        return {
            "rolled_back": False,
            "error": "ROLLBACK requires 'target_version' or 'branch_name'",
        }
    snapshot = workspace.snapshot_manager.get_snapshot(target_version)
    if snapshot is None:
        return {"rolled_back": False, "error": f"Snapshot version {target_version} not found"}

    reverted = services.transaction_manager.rollback_to_version(
        target_version, reason=ins.params.get("reason", "isa_rollback")
    )
    restored_hash = workspace.graph.content_hash()
    matches_snapshot = restored_hash == snapshot.content_hash
    out = {
        "rolled_back": matches_snapshot,
        "target_version": target_version,
        "transactions_reverted": len(reverted),
        "rollback_tx_ids": [tx.id for tx in reverted],
        "state_matches_snapshot": matches_snapshot,
    }
    if not matches_snapshot:
        err = (
            f"Rollback integrity failure at version {target_version}: "
            f"restored hash {restored_hash} != snapshot hash {snapshot.content_hash}"
        )
        logger.error(err)
        out["error"] = err
    return out


async def sys_unify(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """UNIFY: match a GraphPattern against workspace graph or explicit source."""
    pattern = ins.params.get("pattern")
    if pattern is None:
        return {"error": "UNIFY requires 'pattern'"}
    target = workspace.graph if ins.params.get("source") is None else ins.params.get("source")
    matches = unify_pattern(target, pattern, limit=int(ins.params.get("limit", 32)))
    return {
        "matched": bool(matches),
        "matches": matches,
        "count": len(matches),
        "bindings": matches[0] if matches else {},
    }


async def sys_branch(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """BRANCH: control-flow handler for SyscallDispatcher parity."""
    return {
        "branched": False,
        "target_label": ins.params.get("target_label"),
        "target_pc": ins.params.get("target_pc"),
    }


async def sys_call(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """CALL: control-flow handler for SyscallDispatcher parity."""
    return {
        "called": False,
        "skill_id": ins.params.get("skill_id"),
        "goal_pattern": ins.params.get("goal_pattern"),
    }


async def sys_return(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """RETURN: control-flow handler for SyscallDispatcher parity."""
    return {"returned": True, "yield": ins.params.get("yield", ins.params.get("value"))}


def _is_empty_val(val: Any) -> bool:
    if val in (None, False, [], {}, ""):
        return True
    if isinstance(val, Mapping) and "matched" in val:
        return not bool(val["matched"])
    return isinstance(val, Mapping) and "total" in val and int(val["total"]) == 0


def _eval_branch_condition(params: Mapping[str, Any], frame: CallFrame, last_result: Any) -> bool:
    cond = str(params.get("condition", "unconditional"))
    if cond == "unconditional":
        return True
    reg, var = params.get("register"), params.get("var")
    if isinstance(reg, str) and reg in REGISTER_NAMES:
        is_bound = frame.registers.is_bound(reg)
        probe = frame.registers.get(reg) if is_bound else None
    elif isinstance(var, str) and var:
        vname = normalize_var_name(var)
        is_bound, probe = vname in frame.bindings, frame.bindings.get(vname)
    else:
        is_bound, probe = last_result is not None, last_result

    if cond == "if_empty":
        return (not is_bound) or _is_empty_val(probe)
    if cond in ("if_not_empty", "if_true"):
        return is_bound and not _is_empty_val(probe)
    if cond == "if_false":
        return (not is_bound) or not bool(probe)
    if cond == "if_equals":
        return probe == params.get("value")
    if cond == "if_not_equals":
        return probe != params.get("value")

    num = probe
    if isinstance(probe, Mapping):
        if cond == "if_confidence_lt":
            num = probe.get("confidence", probe.get("uncertainty", {}).get("confidence"))
        elif cond == "if_surprise_gt":
            num = probe.get("surprise", probe.get("prediction_error"))
    threshold = float(params.get("threshold", params.get("value", 0.0)))
    if num is None or isinstance(num, bool):
        return False
    try:
        fval = float(num)
    except (TypeError, ValueError):
        return False
    ops = {
        "if_lt": fval < threshold,
        "if_confidence_lt": fval < threshold,
        "if_lte": fval <= threshold,
        "if_gt": fval > threshold,
        "if_surprise_gt": fval > threshold,
        "if_gte": fval >= threshold,
    }
    return ops.get(cond, False)


def _bind_output(frame: CallFrame, bind_spec: Any, value: Any) -> None:
    if not bind_spec:
        return
    if isinstance(bind_spec, str):
        if bind_spec in REGISTER_NAMES:
            frame.registers.set(bind_spec, value)
        elif is_symbolic_var(bind_spec):
            frame.bindings[normalize_var_name(bind_spec)] = value
    elif isinstance(bind_spec, Mapping) and isinstance(value, Mapping):
        for src_key, dst_slot in bind_spec.items():
            if src_key in value and isinstance(dst_slot, str) and dst_slot in REGISTER_NAMES:
                frame.registers.set(dst_slot, value[src_key])


class HCIRInterpreter:
    """Register-unification virtual machine for HCIR ISA v2 instruction streams."""

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        services: KernelServices,
        dispatcher: SyscallDispatcher | None = None,
        *,
        max_call_depth: int = 16,
        max_fuel: int = 1024,
        syscall_allowlist: SyscallAllowlist | None = None,
    ) -> None:
        self._workspace = workspace
        self._services = services
        self._dispatcher = dispatcher or SyscallDispatcher()
        self._max_call_depth = max_call_depth
        self._max_fuel = max_fuel
        self._allowlist = syscall_allowlist or SyscallAllowlist(strict=False)
        self._branch_tx_managers: dict[int, TransactionManager] = {}

    def _target_context(
        self, ins: Instruction, frame: CallFrame
    ) -> tuple[HCIRWorkspaceState, KernelServices, str | None]:
        if ins.opcode in (Opcode.FORK, Opcode.MERGE, Opcode.ROLLBACK):
            return self._workspace, self._services, None
        branch_name = ins.params.get("branch_name") or frame.branch_name
        if not branch_name:
            return self._workspace, self._services, None
        branch_ws = self._workspace.get_branch(str(branch_name))
        if branch_ws is None:
            return self._workspace, self._services, f"Branch '{branch_name}' not found"
        key = id(branch_ws)
        tx_mgr = self._branch_tx_managers.get(key)
        if tx_mgr is None:
            stages = list(getattr(self._services.transaction_manager, "_verification_stages", []))
            tx_mgr = TransactionManager(
                branch_ws, verification_stages=stages, event_bus=self._services.event_bus
            )
            self._branch_tx_managers[key] = tx_mgr
        return (
            branch_ws,
            dc_replace(self._services, workspace=branch_ws, transaction_manager=tx_mgr),
            None,
        )

    def _candidate_workspaces(self, active_ws: HCIRWorkspaceState) -> list[HCIRWorkspaceState]:
        candidates: list[HCIRWorkspaceState] = [active_ws]
        tiered = self._services.tiered_workspace
        if tiered is not None:
            candidates.append(tiered.working_state)
            candidates.extend(frame.workspace for frame in tiered.working.active_frames)
            candidates.extend([tiered.brain, tiered.persistent, tiered.meta])
        seen: set[int] = set()
        ordered: list[HCIRWorkspaceState] = []
        for ws in candidates:
            if id(ws) not in seen:
                seen.add(id(ws))
                ordered.append(ws)
        return ordered

    @staticmethod
    def _iter_executable_nodes(workspaces: list[HCIRWorkspaceState]) -> Iterator[Any]:
        for ws in workspaces:
            for node in ws.graph.nodes.values():
                if getattr(node, "bytecode_stream", None) is not None:
                    yield node

    @staticmethod
    def _skill_rank_key(node: Any) -> tuple[float, int, str]:
        return (
            -float(getattr(node, "reliability_mean", 0.5)),
            -int(getattr(node, "invocation_count", 0)),
            str(node.id),
        )

    def _resolve_callable(
        self, params: Mapping[str, Any], active_ws: HCIRWorkspaceState, frame: CallFrame
    ) -> tuple[Any | None, dict[str, Any]]:
        workspaces = self._candidate_workspaces(active_ws)
        if skill_id := params.get("skill_id"):
            for ws in workspaces:
                node = ws.get_node(str(skill_id))
                if node is not None and getattr(node, "bytecode_stream", None) is not None:
                    return node, {}
            return None, {}
        if skill_name := params.get("skill_name"):
            matches = [
                n
                for n in self._iter_executable_nodes(workspaces)
                if str(skill_name)
                in {
                    getattr(n, "name", ""),
                    getattr(n, "skill_name", ""),
                    getattr(n, "procedure_name", ""),
                }
            ]
            if matches:
                matches.sort(key=self._skill_rank_key)
                return matches[0], {}
            return None, {}
        goal_pattern = params.get("goal_pattern")
        if isinstance(goal_pattern, Mapping):
            ranked: list[tuple[tuple[float, int, str], Any, dict[str, Any]]] = []
            for node in self._iter_executable_nodes(workspaces):
                post = getattr(node, "postcondition_pattern", None)
                if not isinstance(post, Mapping):
                    continue
                goal_b = unify_dicts(post, goal_pattern, registers=frame.registers)
                if goal_b is None:
                    continue
                if isinstance(pre := getattr(node, "precondition_pattern", None), Mapping):
                    if pre and not (
                        set(pre.keys()) <= {"symbolic_expr", "capabilities", "signature"}
                    ):
                        pre_matches = unify_pattern(
                            active_ws.graph,
                            pre,
                            initial_bindings=goal_b,
                            registers=frame.registers,
                            limit=1,
                        )
                        if not pre_matches:
                            continue
                        goal_b = pre_matches[0]
                ranked.append((self._skill_rank_key(node), node, goal_b))
            if ranked:
                ranked.sort(key=lambda item: item[0])
                return ranked[0][1], ranked[0][2]
        return None, {}

    def _build_child_frame(
        self,
        node: Any,
        params: Mapping[str, Any],
        goal_bindings: Mapping[str, Any],
        caller_branch: str | None,
        active_ws: HCIRWorkspaceState,
    ) -> CallFrame:
        child_regs, child_bindings, args = RegisterFile(), dict(goal_bindings), params.get("args")
        if isinstance(args, Mapping):
            for k, v in args.items():
                ks = str(k)
                if ks in REGISTER_NAMES:
                    child_regs.set(ks, v)
                else:
                    child_bindings[normalize_var_name(ks)] = v
        elif isinstance(args, list):
            for idx, val in enumerate(args[:8]):
                child_regs.set(f"$r{idx}", val)
        for idx, pspec in enumerate(list(getattr(node, "parameters", []) or [])[:8]):
            if isinstance(pspec, Mapping) and "name" in pspec:
                vname = normalize_var_name(str(pspec["name"]))
                val = child_bindings.get(vname, pspec.get("default"))
                if val is not None or vname in child_bindings:
                    child_bindings[vname] = val
                    if not child_regs.is_bound(f"$r{idx}"):
                        child_regs.set(f"$r{idx}", val)
        sig = json.dumps(
            {"args": child_bindings, "regs": child_regs.snapshot()}, sort_keys=True, default=str
        )
        return CallFrame(
            callee_id=str(node.id),
            stream=InstructionStream.model_validate(node.bytecode_stream),
            registers=child_regs,
            bindings=child_bindings,
            return_bind=params.get("bind"),
            branch_name=params.get("branch_name") or caller_branch,
            call_signature=(str(node.id), sig, active_ws.current_version),
        )

    @staticmethod
    def _fail_result(
        error: str,
        start_time: float,
        total_tokens: int,
        annotations: list[TransactionAnnotation],
        events: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            success=False,
            error=error,
            annotations=annotations,
            events=events or [],
            metrics=ExecutionMetrics(
                elapsed_ms=int((time.monotonic() - start_time) * 1000), tokens_consumed=total_tokens
            ),
        )

    async def execute(
        self,
        stream: InstructionStream,
        *,
        initial_registers: Mapping[str, Any] | None = None,
        initial_bindings: Mapping[str, Any] | None = None,
        step_hook: StepHook | None = None,
    ) -> ExecutionResult:
        """Execute an entire ISA v2 instruction stream cleanly within fuel and depth bounds."""
        start_time, total_tokens, step_counter = time.monotonic(), 0, 0
        results, annotations, executed_capabilities = [], [], []
        stack = CallStack(max_depth=self._max_call_depth, max_fuel=self._max_fuel)
        root_frame = CallFrame(
            callee_id=stream.id,
            stream=stream,
            registers=RegisterFile(initial_registers),
            bindings=dict(initial_bindings or {}),
        )
        stack.push(root_frame)
        last_result: Any = None
        return_value: Any = None

        try:
            while not stack.is_empty:
                frame = stack.current
                if frame.pc >= len(frame.stream.instructions):
                    stack.pop()
                    continue
                raw_ins = frame.stream.instructions[frame.pc]
                stack.consume_fuel(1)
                resolved_params = resolve_operands(
                    raw_ins.params,
                    frame.registers,
                    frame.bindings,
                    preserve_unbound_vars=raw_ins.opcode in (Opcode.UNIFY, Opcode.CALL),
                )
                eff_ins = raw_ins.model_copy(update={"params": resolved_params})
                active_ws, active_services, branch_err = self._target_context(eff_ins, frame)
                if branch_err is not None:
                    result: dict[str, Any] = {"error": branch_err}
                    frame.pc += 1
                elif eff_ins.opcode == Opcode.UNIFY:
                    pattern = eff_ins.params.get("pattern")
                    if pattern is None:
                        result = {"error": "UNIFY requires 'pattern'"}
                    else:
                        target = (
                            active_ws.graph
                            if eff_ins.params.get("source") is None
                            else eff_ins.params.get("source")
                        )
                        matches = unify_pattern(
                            target,
                            pattern,
                            initial_bindings=frame.bindings,
                            registers=frame.registers,
                            limit=int(eff_ins.params.get("limit", 32)),
                        )
                        first = matches[0] if matches else {}
                        if matches:
                            frame.bindings.update(
                                {bk: bv for bk, bv in first.items() if bk.startswith("?")}
                            )
                            if "bind" in eff_ins.params:
                                bspec = eff_ins.params["bind"]
                                _bind_output(
                                    frame,
                                    bspec,
                                    first.get("__primary_id__", first)
                                    if isinstance(bspec, str)
                                    else first,
                                )
                        elif (
                            isinstance(eff_ins.params.get("bind"), str)
                            and eff_ins.params["bind"] in REGISTER_NAMES
                        ):
                            frame.registers.unset(eff_ins.params["bind"])
                        if (
                            isinstance(eff_ins.params.get("bind_all"), str)
                            and eff_ins.params["bind_all"] in REGISTER_NAMES
                        ):
                            frame.registers.set(eff_ins.params["bind_all"], matches)
                        result = {
                            "matched": bool(matches),
                            "matches": matches,
                            "count": len(matches),
                            "bindings": first,
                        }
                    frame.pc += 1
                elif eff_ins.opcode == Opcode.BRANCH:
                    taken, tlabel, tpc = (
                        _eval_branch_condition(eff_ins.params, frame, last_result),
                        eff_ins.params.get("target_label"),
                        eff_ins.params.get("target_pc"),
                    )
                    if not taken:
                        frame.pc += 1
                        result = {"branched": False, "target_label": tlabel, "target_pc": tpc}
                    elif isinstance(tlabel, str) and tlabel in frame.label_index:
                        frame.pc = frame.label_index[tlabel]
                        result = {"branched": True, "target_label": tlabel, "target_pc": frame.pc}
                    elif tpc is not None and 0 <= int(tpc) <= len(frame.stream.instructions):
                        frame.pc = int(tpc)
                        result = {"branched": True, "target_pc": frame.pc}
                    else:
                        result = {"error": f"Invalid BRANCH target (label={tlabel!r}, pc={tpc!r})"}
                        frame.pc += 1
                elif eff_ins.opcode == Opcode.CALL:
                    callee_node, goal_b = self._resolve_callable(eff_ins.params, active_ws, frame)
                    if callee_node is None:
                        result = {"error": "CALL could not resolve target SkillNode/ProcedureNode"}
                        frame.pc += 1
                    else:
                        child = self._build_child_frame(
                            callee_node, eff_ins.params, goal_b, frame.branch_name, active_ws
                        )
                        frame.pc += 1
                        stack.push(child)
                        result = {
                            "called": True,
                            "callee_id": str(callee_node.id),
                            "depth": stack.depth,
                        }
                elif eff_ins.opcode == Opcode.RETURN:
                    raw_yield = eff_ins.params.get("yield", eff_ins.params.get("value"))
                    yield_val = (
                        raw_yield[0]
                        if isinstance(raw_yield, list)
                        and len(raw_yield) == 1
                        and (isinstance(frame.return_bind, str) or frame.return_bind is None)
                        else raw_yield
                    )
                    popped = stack.pop()
                    return_value = yield_val
                    if not stack.is_empty and popped.return_bind:
                        _bind_output(stack.current, popped.return_bind, yield_val)
                    result = {"returned": True, "yield": raw_yield, "value": yield_val}
                else:
                    if (
                        eff_ins.opcode == Opcode.EXECUTE
                        and self._dispatcher._handlers.get(Opcode.EXECUTE) is sys_execute
                    ):
                        result = await sys_execute(
                            eff_ins, active_ws, active_services, allowlist=self._allowlist
                        )
                    else:
                        result = await self._dispatcher.dispatch(
                            eff_ins, active_ws, active_services
                        )
                    if eff_ins.opcode == Opcode.EXECUTE and "capability" in eff_ins.params:
                        executed_capabilities.append(str(eff_ins.params["capability"]))
                    if "error" not in result and "bind" in eff_ins.params:
                        bind_val = (
                            result.get("matches")
                            if eff_ins.opcode == Opcode.QUERY
                            else result.get("result")
                            if eff_ins.opcode == Opcode.EXECUTE
                            else (result.get("node_id") or result.get("edge_id") or result)
                            if eff_ins.opcode == Opcode.ASSERT
                            else result
                        )
                        _bind_output(frame, eff_ins.params["bind"], bind_val)
                    frame.pc += 1

                results.append(result)
                last_result = result
                total_tokens += raw_ins.cost_estimate
                if result.get("deprecated_capability"):
                    annotations.append(
                        TransactionAnnotation(
                            author="interpreter",
                            assertion=f"Instruction {step_counter} executed deprecated non-Layer-0 capability '{result.get('capability', '')}'",
                            severity="warning",
                        )
                    )
                if "error" in result:
                    annotations.append(
                        TransactionAnnotation(
                            author="interpreter",
                            assertion=f"Instruction {step_counter} ({raw_ins.opcode}) error: {result['error']}",
                            severity="error",
                        )
                    )
                if (
                    eff_ins.opcode == Opcode.ROLLBACK
                    and result.get("state_matches_snapshot") is False
                ):
                    return self._fail_result(
                        str(result.get("error", "Rollback hash mismatch")),
                        start_time,
                        total_tokens,
                        annotations,
                        [{"type": "stream_aborted", "results": results}],
                    )
                if (
                    step_hook is not None
                    and step_hook(step_counter, eff_ins, result, frame, active_ws) is False
                ):
                    break
                step_counter += 1
        except (VMRuntimeError, Exception) as exc:
            logger.error("VM step %d failed: %s", step_counter, exc)
            annotations.append(
                TransactionAnnotation(
                    author="interpreter",
                    assertion=f"Instruction {step_counter} exception: {exc}",
                    severity="error",
                )
            )
            return self._fail_result(str(exc), start_time, total_tokens, annotations)

        return ExecutionResult(
            success=True,
            annotations=annotations,
            events=[
                {
                    "type": "stream_completed",
                    "results": results,
                    "return_value": return_value,
                    "registers": root_frame.registers.snapshot(),
                    "capabilities_used": executed_capabilities,
                }
            ],
            metrics=ExecutionMetrics(
                elapsed_ms=int((time.monotonic() - start_time) * 1000), tokens_consumed=total_tokens
            ),
        )

    async def execute_with_receipt(
        self, stream: InstructionStream, process_id: str = "", thread_id: str = ""
    ) -> tuple[ExecutionResult, Any]:
        """Execute an instruction stream and return an ExecutionReceipt."""
        from hbllm.hcir.receipt import ExecutionReceipt

        input_ver = self._workspace.snapshot_manager.current_version
        committed_before = len(self._services.transaction_manager.committed_log)
        res = await self.execute(stream)
        tx_committed_ids = [
            tx.id for tx in self._services.transaction_manager.committed_log[committed_before:]
        ]
        rejected_txs = [
            {"id": tx.id, "reason": getattr(tx, "error_message", "rejected")}
            for tx in getattr(self._services.transaction_manager, "_rejected_log", [])
        ]
        evt0 = res.events[0] if res.events else {}
        caps = list(evt0.get("capabilities_used", [])) or [
            str(ins.params["capability"])
            for ins in stream.instructions
            if ins.opcode == Opcode.EXECUTE and "capability" in ins.params
        ]
        receipt = ExecutionReceipt(
            process_id=process_id,
            thread_id=thread_id,
            author=stream.author,
            instruction_stream_hash=stream.compute_hash(),
            input_snapshot_version=input_ver,
            final_snapshot_version=self._workspace.snapshot_manager.current_version,
            transactions_committed=tx_committed_ids,
            transactions_rejected=rejected_txs,
            capabilities_used=caps,
            metrics=res.metrics,
            outputs={
                "results": evt0.get("results", []),
                "return_value": evt0.get("return_value"),
                "registers": evt0.get("registers", {}),
            }
            if evt0
            else {},
            success=res.success,
            error=res.error,
        )
        return res, receipt
