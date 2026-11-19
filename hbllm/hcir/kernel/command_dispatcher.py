"""Command Dispatcher — routes COMMAND SemanticFrames to governed kernel capabilities.

Bridges the A16 Language Pipeline to the Kernel's CapabilityResolver:
  COMMAND SemanticFrame → verb mapping → CapabilityResolver.resolve_and_execute() → execution receipt

Invariants:
  1. All execution goes through CapabilityResolver (sandbox/policy enforcement).
  2. Execution receipts are committed as EvidenceNodes via TransactionManager.
  3. Multi-clause discourse (from parse_discourse()) is executed sequentially.
  4. No LLM dependency — pure rule-based verb→capability mapping.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import EvidenceNode
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.language.core.semantic_frame import (
    FrameType,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Verb → Capability Mapping (extensible, domain-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

# Maps imperative verbs (lemmas) to governed kernel capabilities.
# Each verb can map to a primary capability + optional parameter extraction.
VERB_CAPABILITY_MAP: dict[str, str] = {
    # Filesystem operations
    "read": "fs.read",
    "open": "fs.read",
    "cat": "fs.read",
    "show": "fs.read",
    "display": "fs.read",
    "print": "fs.read",
    "view": "fs.read",
    "list": "fs.list",
    "ls": "fs.list",
    "dir": "fs.list",
    # Code analysis
    "verify": "code.python_ast",
    "check": "code.python_ast",
    "analyze": "code.python_ast",
    "parse": "code.python_ast",
    "inspect": "code.python_ast",
    "lint": "code.python_ast",
    # Git
    "status": "git.status",
    # Search (glob/regex pattern matching)
    "search": "fs.search",
    "find": "fs.search",
    "locate": "fs.search",
    "grep": "fs.search",
    # Shell execution (sandboxed)
    "run": "shell.exec",
    "execute": "shell.exec",
    "start": "shell.exec",
    "launch": "shell.exec",
    # World model queries
    "predict": "world.predict_outcome",
    "evaluate": "world.evaluate_prediction_error",
    "simulate": "world.predict_outcome",
    # Planning
    "plan": "planning.generate_candidates",
    "expand": "planning.expand",
    "validate": "planning.validate",
}


@dataclass
class CommandExecutionReceipt:
    """Result of dispatching a COMMAND SemanticFrame through the kernel."""

    command_id: str
    verb: str
    capability: str
    target: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    evidence_node_id: str | None = None


@dataclass
class DiscourseExecutionResult:
    """Result of dispatching a multi-clause discourse (sequential execution)."""

    receipts: list[CommandExecutionReceipt] = field(default_factory=list)
    total_commands: int = 0
    successful: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0

    @property
    def all_succeeded(self) -> bool:
        return self.failed == 0 and self.successful > 0

    @property
    def summary(self) -> str:
        return (
            f"{self.successful}/{self.total_commands} commands executed successfully "
            f"({self.total_latency_ms:.1f}ms total)"
        )


class CommandDispatcher:
    """Dispatches COMMAND SemanticFrames to governed kernel capabilities.

    Usage::

        dispatcher = CommandDispatcher(capability_resolver, workspace)

        # Single command
        receipt = await dispatcher.dispatch(command_frame)

        # Multi-clause discourse (from parse_discourse)
        result = await dispatcher.dispatch_discourse([frame1, frame2, frame3])
    """

    def __init__(
        self,
        capability_resolver: CapabilityResolver,
        workspace: Any | None = None,
        verb_map: dict[str, str] | None = None,
    ) -> None:
        self._resolver = capability_resolver
        self._workspace = workspace
        self._verb_map = dict(VERB_CAPABILITY_MAP)
        if verb_map:
            self._verb_map.update(verb_map)

    # ── Core Dispatch ─────────────────────────────────────────────────

    async def dispatch(self, frame: SemanticFrame) -> CommandExecutionReceipt:
        """Dispatch a single COMMAND SemanticFrame to the appropriate capability.

        1. Extract verb (predicate) and target (PATIENT/THEME concept_name).
        2. Resolve verb → capability via VERB_CAPABILITY_MAP.
        3. Build capability params from frame arguments.
        4. Execute via CapabilityResolver.resolve_and_execute().
        5. Commit execution receipt as EvidenceNode.
        """
        command_id = f"cmd_{uuid.uuid4().hex[:8]}"
        verb = frame.predicate or "act"
        t0 = time.monotonic()

        # 1. Resolve capability
        capability = self._resolve_capability(verb)
        if capability is None:
            latency = (time.monotonic() - t0) * 1000
            return CommandExecutionReceipt(
                command_id=command_id,
                verb=verb,
                capability="",
                target=self._extract_target(frame),
                success=False,
                error=f"No capability mapping for verb '{verb}'. "
                f"Known verbs: {sorted(self._verb_map.keys())}",
                latency_ms=latency,
            )

        # 2. Build execution params from frame arguments
        target = self._extract_target(frame)
        params = self._build_params(frame, capability, target)

        # 3. Execute through governed capability resolver
        try:
            result = await self._resolver.resolve_and_execute(capability, params)
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            logger.warning(
                "CommandDispatcher: execution error for verb=%s cap=%s: %s",
                verb,
                capability,
                exc,
            )
            return CommandExecutionReceipt(
                command_id=command_id,
                verb=verb,
                capability=capability,
                target=target,
                success=False,
                error=str(exc),
                latency_ms=latency,
            )

        latency = (time.monotonic() - t0) * 1000
        success = result.get("ok", False) if "ok" in result else "error" not in result

        receipt = CommandExecutionReceipt(
            command_id=command_id,
            verb=verb,
            capability=capability,
            target=target,
            success=bool(success),
            result=result,
            error=str(result.get("error", "")),
            latency_ms=latency,
        )

        # 4. Commit execution receipt as EvidenceNode
        evidence_id = self._commit_receipt(receipt, frame)
        receipt.evidence_node_id = evidence_id

        logger.info(
            "CommandDispatcher: dispatched verb=%s → cap=%s target=%s success=%s (%.1fms)",
            verb,
            capability,
            target,
            success,
            latency,
        )
        return receipt

    async def dispatch_discourse(
        self,
        frames: list[SemanticFrame],
    ) -> DiscourseExecutionResult:
        """Dispatch a multi-clause discourse sequentially.

        Only COMMAND frames are dispatched; ASSERTION/QUERY/ERROR frames are skipped.
        Execution is sequential to preserve command ordering semantics.
        """
        result = DiscourseExecutionResult()
        command_frames = [f for f in frames if f.frame_type == FrameType.COMMAND]
        result.total_commands = len(command_frames)

        t0 = time.monotonic()
        for frame in command_frames:
            receipt = await self.dispatch(frame)
            result.receipts.append(receipt)
            if receipt.success:
                result.successful += 1
            else:
                result.failed += 1

        result.total_latency_ms = (time.monotonic() - t0) * 1000
        return result

    # ── Verb → Capability Resolution ──────────────────────────────────

    def _resolve_capability(self, verb: str) -> str | None:
        """Resolve a verb lemma to a registered capability name."""
        # Direct lookup
        cap = self._verb_map.get(verb.lower())
        if cap is not None:
            # Verify capability actually exists in the resolver
            if self._resolver.has_capability(cap):
                return cap
            logger.debug(
                "CommandDispatcher: verb '%s' maps to '%s' but capability not registered",
                verb,
                cap,
            )
            return cap  # Still return it; resolve_and_execute will give clear error

        # Fallback: check if verb itself matches a capability name pattern
        for pattern in [verb, f"fs.{verb}", f"code.{verb}", f"world.{verb}"]:
            if self._resolver.has_capability(pattern):
                return pattern

        return None

    def register_verb(self, verb: str, capability: str) -> None:
        """Register a custom verb → capability mapping at runtime."""
        self._verb_map[verb.lower()] = capability

    def known_verbs(self) -> list[str]:
        """Return all known verb lemmas."""
        return sorted(self._verb_map.keys())

    # ── Parameter Building ────────────────────────────────────────────

    def _extract_target(self, frame: SemanticFrame) -> str:
        """Extract the primary target string from a command frame."""
        patient = frame.get_role(ThematicRole.PATIENT)
        if patient and patient.concept_name:
            return patient.concept_name

        theme = frame.get_role(ThematicRole.THEME)
        if theme and theme.concept_name:
            return theme.concept_name

        return ""

    def _build_params(
        self,
        frame: SemanticFrame,
        capability: str,
        target: str,
    ) -> dict[str, Any]:
        """Build capability execution params from frame arguments."""
        params: dict[str, Any] = {
            "capability": capability,
            "source_command": frame.predicate,
        }

        # Map target to the right parameter based on capability
        if capability in ("fs.read", "code.python_ast"):
            params["path"] = target
        elif capability == "fs.list":
            # Map generic words like "directory", "folder", "files" to current directory "."
            if target.lower() in ("directory", "dir", "folder", "files", ""):
                params["path"] = "."
            else:
                params["path"] = target
        elif capability == "fs.search":
            params["pattern"] = target or "*"
        elif capability == "shell.exec":
            # Extract shell command: strip leading imperative verb if present
            raw = getattr(frame.metadata, "raw_text", "").strip()
            words = raw.split(maxsplit=1)
            if len(words) > 1 and words[0].lower() == (frame.predicate or "").lower():
                params["command"] = words[1]
            else:
                params["command"] = target or raw
        elif capability == "git.status":
            pass  # No specific target needed
        else:
            # Generic: pass target as-is
            if target:
                params["target"] = target

        # Add destination if present (e.g., "move X to Y")
        dest = frame.get_role(ThematicRole.DESTINATION)
        if dest and dest.concept_name:
            params["destination"] = dest.concept_name

        # Add instrument if present (e.g., "open X with Y")
        instrument = frame.get_role(ThematicRole.INSTRUMENT)
        if instrument and instrument.concept_name:
            params["instrument"] = instrument.concept_name

        return params

    # ── Receipt Commit ────────────────────────────────────────────────

    def _commit_receipt(
        self,
        receipt: CommandExecutionReceipt,
        frame: SemanticFrame,
    ) -> str | None:
        """Commit an execution receipt as an EvidenceNode to the workspace."""
        if self._workspace is None:
            return None

        claim = (
            f"command_executed:{receipt.verb}:{receipt.capability}:"
            f"{'success' if receipt.success else 'failure'}"
        )

        evidence = EvidenceNode(
            claim_id=claim,
            methodology=f"command_dispatch:{receipt.capability}",
            confidence=0.99 if receipt.success else 0.3,
            source_uri=f"command://{receipt.command_id}",
            tags=[
                "command_receipt",
                receipt.verb,
                receipt.capability,
                "success" if receipt.success else "failure",
            ],
            properties={
                "command_id": receipt.command_id,
                "verb": receipt.verb,
                "capability": receipt.capability,
                "target": receipt.target,
                "latency_ms": receipt.latency_ms,
                "success": receipt.success,
                "error": receipt.error,
            },
        )

        try:
            if hasattr(self._workspace, "commit_operations"):
                self._workspace.commit_operations(
                    [
                        TransactionOperation(
                            op=TransactionOp.UPSERT_NODE,
                            node_id=evidence.id,
                            node_data=evidence.model_dump(),
                        )
                    ],
                    author="command_dispatcher",
                )
            else:
                graph = getattr(self._workspace, "graph", None)
                if graph is not None:
                    graph.add_node(evidence)
            return evidence.id
        except Exception as exc:
            logger.warning("CommandDispatcher: failed to commit receipt: %s", exc)
            return None


def dispatch_command_sync(
    frame: SemanticFrame,
    capability_resolver: CapabilityResolver,
    workspace: Any | None = None,
) -> CommandExecutionReceipt:
    """Synchronous convenience wrapper for single command dispatch."""
    dispatcher = CommandDispatcher(capability_resolver, workspace)
    loop = asyncio.get_event_loop()
    if loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, dispatcher.dispatch(frame))
            return future.result(timeout=30.0)
    return asyncio.run(dispatcher.dispatch(frame))
