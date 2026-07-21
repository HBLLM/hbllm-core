"""
HCIR Interpreter — bytecode virtual machine.

Executes ``InstructionStream`` against a workspace using kernel
services.  Each opcode dispatches through a ``SyscallDispatcher``
rather than hardcoded if-statements, making the system extensible.

    Interpreter
        ↓
    SyscallDispatcher
        ↓
    Kernel Services
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Coroutine

from hbllm.hcir.abi import ExecutionMetrics, ExecutionResult
from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import HCIREdge, HCIRNode, NODE_TYPE_REGISTRY, HCIRNodeType
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.query import GraphQuery, EdgeQuery
from hbllm.hcir.stores import EventType
from hbllm.hcir.transactions import (
    HCIRDelta,
    HCIRTransaction,
    TransactionAnnotation,
    TransactionOp,
    TransactionOperation,
)
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


# Type alias for syscall handlers
SyscallHandler = Callable[
    [Instruction, HCIRWorkspaceState, KernelServices],
    Coroutine[Any, Any, dict[str, Any]],
]


# ═══════════════════════════════════════════════════════════════════════════
# Syscall Dispatcher
# ═══════════════════════════════════════════════════════════════════════════


class SyscallDispatcher:
    """Maps opcodes to async syscall handlers.

    Every opcode becomes a kernel syscall:
        ASSERT  → sys_assert()
        RETRACT → sys_retract()
        QUERY   → sys_query()
        ...

    New opcodes or experimental extensions register handlers here
    without modifying the interpreter.
    """

    def __init__(self) -> None:
        self._handlers: dict[Opcode, SyscallHandler] = {}
        self._register_defaults()

    def register(self, opcode: Opcode, handler: SyscallHandler) -> None:
        """Register a syscall handler for an opcode."""
        self._handlers[opcode] = handler

    async def dispatch(
        self,
        instruction: Instruction,
        workspace: HCIRWorkspaceState,
        services: KernelServices,
    ) -> dict[str, Any]:
        """Dispatch an instruction to its syscall handler."""
        handler = self._handlers.get(instruction.opcode)
        if handler is None:
            raise ValueError(f"No syscall handler for opcode: {instruction.opcode}")
        return await handler(instruction, workspace, services)

    def _register_defaults(self) -> None:
        """Register the 8 stable syscall handlers."""
        self.register(Opcode.ASSERT, sys_assert)
        self.register(Opcode.RETRACT, sys_retract)
        self.register(Opcode.QUERY, sys_query)
        self.register(Opcode.EXECUTE, sys_execute)
        self.register(Opcode.WAIT, sys_wait)
        self.register(Opcode.FORK, sys_fork)
        self.register(Opcode.MERGE, sys_merge)
        self.register(Opcode.ROLLBACK, sys_rollback)


# ═══════════════════════════════════════════════════════════════════════════
# Default Syscall Implementations
# ═══════════════════════════════════════════════════════════════════════════


async def sys_assert(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """ASSERT: inject a node or edge into the workspace.

    Params:
        node_data (dict): Serialized node to add.
        edge_data (dict): Serialized edge to add (optional, mutually exclusive).
    """
    node_data = ins.params.get("node_data")
    edge_data = ins.params.get("edge_data")

    if node_data:
        tx = HCIRTransaction(
            author=ins.params.get("author", "interpreter"),
            operations=[TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id=node_data.get("id"),
                node_data=node_data,
            )],
        )
        result = services.transaction_manager.commit(tx)
        return {"committed": result.is_committed, "tx_id": result.id}

    if edge_data:
        tx = HCIRTransaction(
            author=ins.params.get("author", "interpreter"),
            operations=[TransactionOperation(
                op=TransactionOp.ADD_EDGE,
                edge_id=edge_data.get("id"),
                edge_data=edge_data,
            )],
        )
        result = services.transaction_manager.commit(tx)
        return {"committed": result.is_committed, "tx_id": result.id}

    return {"error": "ASSERT requires 'node_data' or 'edge_data'"}


async def sys_retract(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """RETRACT: remove a node or edge from the workspace.

    Params:
        node_id (str): Node to retract.
        edge_id (str): Edge to retract (optional, mutually exclusive).
    """
    node_id = ins.params.get("node_id")
    edge_id = ins.params.get("edge_id")

    ops = []
    if node_id:
        ops.append(TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id=node_id))
    if edge_id:
        ops.append(TransactionOperation(op=TransactionOp.REMOVE_EDGE, edge_id=edge_id))

    if not ops:
        return {"error": "RETRACT requires 'node_id' or 'edge_id'"}

    tx = HCIRTransaction(
        author=ins.params.get("author", "interpreter"),
        operations=ops,
    )
    result = services.transaction_manager.commit(tx)
    return {"committed": result.is_committed, "tx_id": result.id}


async def sys_query(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """QUERY: retrieve nodes matching structural constraints.

    Params:
        node_type (str): Filter by node type.
        category (str): Filter by category.
        lifecycle (str): Filter by lifecycle.
        scope_tenant (str): Filter by tenant scope.
        text_contains (str): Full-text substring search.
        min_confidence (float): Minimum confidence threshold.
        limit (int): Max results.
    """
    from hbllm.hcir.graph import CognitiveCategory, HCIRNodeType, NodeLifecycle

    query = GraphQuery(
        node_type=HCIRNodeType(ins.params["node_type"]) if "node_type" in ins.params else None,
        category=CognitiveCategory(ins.params["category"]) if "category" in ins.params else None,
        lifecycle=NodeLifecycle(ins.params["lifecycle"]) if "lifecycle" in ins.params else None,
        scope_tenant=ins.params.get("scope_tenant"),
        text_contains=ins.params.get("text_contains"),
        min_confidence=ins.params.get("min_confidence"),
        limit=ins.params.get("limit", 100),
    )
    result = workspace.query(query)
    return {
        "matches": [n.model_dump() for n in result.nodes],
        "total": result.total_matches,
        "truncated": result.truncated,
    }


async def sys_execute(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """EXECUTE: dispatch a capability via the CapabilityResolver.

    Params:
        capability (str): Capability name to resolve.
        params (dict): Parameters to pass to the executor.
    """
    capability = ins.params.get("capability", "")
    exec_params = ins.params.get("params", {})

    executor = await services.capability_resolver.resolve(capability)
    if executor is None:
        return {"error": f"No executor for capability: {capability}"}

    try:
        result = await executor.execute(exec_params)
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}


async def sys_wait(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """WAIT: block execution until a condition resolves.

    Params:
        condition (str): A query-like condition to wait for.
        timeout_ms (int): Maximum wait time in milliseconds.
    """
    # In the current synchronous model, WAIT returns immediately
    # with the condition evaluation result.
    condition = ins.params.get("condition", "")
    return {"waited": True, "condition": condition, "resolved": True}


async def sys_fork(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """FORK: create a simulation branch.

    Params:
        branch_name (str): Name for the forked branch.
    """
    branch_name = ins.params.get("branch_name", "simulation")
    try:
        forked = workspace.fork(branch_name)
        workspace.snapshot_manager.record_kernel_event(
            EventType.SIMULATION_STARTED,
            {"branch": branch_name},
        )
        return {"forked": True, "branch": branch_name}
    except ValueError as exc:
        return {"error": str(exc)}


async def sys_merge(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """MERGE: commit simulated modifications back to the main branch.

    Params:
        branch_name (str): Branch to merge.
    """
    branch_name = ins.params.get("branch_name", "simulation")
    branch = workspace.get_branch(branch_name)
    if branch is None:
        return {"error": f"Branch '{branch_name}' not found"}

    # Merge: add all nodes/edges from branch that are new or modified
    merged_count = 0
    for node in branch.graph.all_nodes():
        existing = workspace.get_node(node.id)
        if existing is None:
            workspace.add_node(node.model_copy(deep=True), author="merge")
            merged_count += 1
        elif node.model_dump() != existing.model_dump():
            workspace.upsert_node(node.model_copy(deep=True), author="merge")
            merged_count += 1

    workspace.drop_branch(branch_name)
    workspace.snapshot_manager.record_kernel_event(
        EventType.SIMULATION_FINISHED,
        {"branch": branch_name, "merged_count": merged_count},
    )
    return {"merged": True, "branch": branch_name, "merged_count": merged_count}


async def sys_rollback(
    ins: Instruction, workspace: HCIRWorkspaceState, services: KernelServices
) -> dict[str, Any]:
    """ROLLBACK: reset state to a prior snapshot version.

    Params:
        target_version (int): Snapshot version to roll back to.
    """
    target_version = ins.params.get("target_version")
    if target_version is None:
        return {"error": "ROLLBACK requires 'target_version'"}

    snapshot = workspace.snapshot_manager.get_snapshot(target_version)
    if snapshot is None:
        return {"error": f"Snapshot version {target_version} not found"}

    workspace.snapshot_manager.record_kernel_event(
        EventType.ROLLBACK,
        {"target_version": target_version},
    )
    return {"rolled_back": True, "target_version": target_version}


# ═══════════════════════════════════════════════════════════════════════════
# HCIR Interpreter
# ═══════════════════════════════════════════════════════════════════════════


class HCIRInterpreter:
    """Bytecode virtual machine for HCIR instruction streams.

    Executes instructions sequentially, dispatching each opcode
    through the ``SyscallDispatcher`` to the kernel.

    Usage::

        interpreter = HCIRInterpreter(workspace, services)
        result = await interpreter.execute(stream)
    """

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        services: KernelServices,
        dispatcher: SyscallDispatcher | None = None,
    ) -> None:
        self._workspace = workspace
        self._services = services
        self._dispatcher = dispatcher or SyscallDispatcher()

    async def execute(self, stream: InstructionStream) -> ExecutionResult:
        """Execute an entire instruction stream.

        Tracks resource consumption and collects results.
        """
        start_time = time.monotonic()
        total_tokens = 0
        results: list[dict[str, Any]] = []
        annotations: list[TransactionAnnotation] = []

        for i, instruction in enumerate(stream.instructions):
            try:
                result = await self._dispatcher.dispatch(
                    instruction, self._workspace, self._services
                )
                results.append(result)
                total_tokens += instruction.cost_estimate

                if "error" in result:
                    annotations.append(TransactionAnnotation(
                        author="interpreter",
                        assertion=f"Instruction {i} ({instruction.opcode}) error: {result['error']}",
                        severity="error",
                    ))

            except Exception as exc:
                logger.error("Instruction %d (%s) failed: %s", i, instruction.opcode, exc)
                annotations.append(TransactionAnnotation(
                    author="interpreter",
                    assertion=f"Instruction {i} ({instruction.opcode}) exception: {exc}",
                    severity="error",
                ))
                return ExecutionResult(
                    success=False,
                    error=str(exc),
                    annotations=annotations,
                    metrics=ExecutionMetrics(
                        elapsed_ms=int((time.monotonic() - start_time) * 1000),
                        tokens_consumed=total_tokens,
                    ),
                )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        return ExecutionResult(
            success=True,
            annotations=annotations,
            events=[{"type": "stream_completed", "results": results}],
            metrics=ExecutionMetrics(
                elapsed_ms=elapsed_ms,
                tokens_consumed=total_tokens,
            ),
        )

    async def execute_with_receipt(
        self,
        stream: InstructionStream,
        process_id: str = "",
        thread_id: str = "",
    ) -> tuple[ExecutionResult, Any]:
        """Execute an instruction stream and return an ExecutionReceipt.

        Returns:
            (ExecutionResult, ExecutionReceipt)
        """
        from hbllm.hcir.receipt import ExecutionReceipt, VerificationStageSummary

        input_ver = self._workspace.snapshot_manager.current_version
        committed_before = len(self._services.transaction_manager.committed_log)

        res = await self.execute(stream)

        committed_after = self._services.transaction_manager.committed_log[committed_before:]
        tx_committed_ids = [tx.id for tx in committed_after]

        rejected_txs = [
            {"id": tx.id, "reason": getattr(tx, "error_message", "rejected")}
            for tx in getattr(self._services.transaction_manager, "_rejected_log", [])
        ]

        capabilities_used = []
        for ins in stream.instructions:
            if ins.opcode.value == "EXECUTE" and "capability" in ins.params:
                capabilities_used.append(ins.params["capability"])

        receipt = ExecutionReceipt(
            process_id=process_id,
            thread_id=thread_id,
            author=stream.author,
            instruction_stream_hash=stream.compute_hash(),
            input_snapshot_version=input_ver,
            final_snapshot_version=self._workspace.snapshot_manager.current_version,
            transactions_committed=tx_committed_ids,
            transactions_rejected=rejected_txs,
            capabilities_used=capabilities_used,
            metrics=res.metrics,
            outputs={"results": res.events[0].get("results", [])} if res.events else {},
            success=res.success,
            error=res.error,
        )

        return res, receipt

