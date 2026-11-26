"""HCIR-backed Procedural Skill Store.

Stores procedural skills and training-discovered skills as homoiconic
``SkillNode`` entries inside ``TieredWorkspace`` with parameterized
ISA v2 ``InstructionStream`` programs committed through
``ConstitutionalVerifier``-backed workspace transactions.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.nodes import SkillNode
from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope
from hbllm.hcir.workspace_tiers import TieredWorkspace

logger = logging.getLogger(__name__)


@dataclass
class HCIRSkillRecord:
    """Serializable view of a procedural skill managed by ``HCIRSkillStore``."""

    skill_id: str
    name: str
    description: str = ""
    category: str = "general"
    steps: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    success_criteria: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)
    invocations: int = 0
    success_rate: float = 1.0
    confidence_score: float = 0.8
    avg_latency_ms: float = 0.0
    tenant_id: str = "default"
    source: str = ""


def _build_skill_bytecode(name: str, steps: list[str], tools_used: list[str]) -> dict[str, Any]:
    """Construct a valid parameterized ISA v2 InstructionStream for a SkillNode."""
    instructions: list[Instruction] = []
    if tools_used:
        for tool in tools_used:
            instructions.append(
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={"capability": tool, "params": {"skill": name}},
                )
            )
    else:
        instructions.append(
            Instruction(
                opcode=Opcode.QUERY,
                params={"node_type": "concept", "label_contains": name},
            )
        )
    instructions.append(
        Instruction(
            opcode=Opcode.RETURN,
            params={"yield": ["$r0"], "steps": list(steps)},
        )
    )
    return InstructionStream(author="hcir_skill_store", instructions=instructions).model_dump()


class HCIRSkillStore:
    """HCIR workspace-backed repository for reusable procedural skills."""

    def __init__(
        self,
        data_dir: str | None = None,
        tiered_workspace: TieredWorkspace | None = None,
        tenant_id: str = "default",
    ) -> None:
        self._tenant_id = tenant_id
        if tiered_workspace is not None:
            self._tiered = tiered_workspace
        else:
            persistent_store = None
            if data_dir is not None:
                db_path = Path(data_dir) / "hcir_skills.db"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                persistent_store = SQLiteEventStore(db_path, tenant_id=tenant_id)
            self._tiered = TieredWorkspace(persistent_store=persistent_store, tenant_id=tenant_id)
        authority = CommitAuthorityRegistry()
        verifier = ConstitutionalVerifier(authority_registry=authority)
        pipeline = create_default_pipeline(
            default_scope=Scope(tenant_id=tenant_id),
            constitutional_verifier=verifier,
        )
        self._ws = self._tiered.brain
        if getattr(self._ws, "transaction_manager", None) is None:
            from hbllm.hcir.kernel.transaction_manager import TransactionManager

            self._ws.attach_transaction_manager(
                TransactionManager(workspace=self._ws, verification_stages=pipeline)
            )
        self._records: dict[str, HCIRSkillRecord] = {}

    def _commit_skill_node(self, record: HCIRSkillRecord) -> None:
        stream_dict = _build_skill_bytecode(record.name, record.steps, record.tools_used)
        node = SkillNode(
            id=record.skill_id,
            name=record.name,
            skill_name=record.name,
            description=record.description,
            preconditions=list(record.tools_used),
            postconditions=[record.success_criteria] if record.success_criteria else [],
            success_rate=max(0.0, min(1.0, record.success_rate)),
            invocation_count=record.invocations,
            bytecode_stream=stream_dict,
            provenance=Provenance(created_by=record.source or "hcir_skill_store"),
            scope=Scope(tenant_id=record.tenant_id or self._tenant_id),
            tags=[record.category, "hcir_skill_store"],
            properties={
                "category": record.category,
                "steps": list(record.steps),
                "tools_used": list(record.tools_used),
                "success_criteria": record.success_criteria,
                "examples": list(record.examples),
                "source": record.source,
                "confidence_score": record.confidence_score,
            },
        )
        self._ws.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node.id,
                    node_data=node.model_dump(),
                )
            ],
            author=record.source or "hcir_skill_store",
        )

    def _store(self, skill: HCIRSkillRecord) -> None:
        self._records[skill.skill_id] = skill
        self._commit_skill_node(skill)

    def store_skill(
        self,
        name: str,
        description: str,
        category: str = "general",
        steps: list[str] | None = None,
        tools_used: list[str] | None = None,
        success_criteria: str = "",
        examples: list[dict[str, Any]] | None = None,
        tenant_id: str | None = None,
        source: str = "installer",
    ) -> HCIRSkillRecord:
        """Persist a skill declaratively as a homoiconic SkillNode."""
        skill_id = f"skill_{name.lower().replace(' ', '_')}"
        record = HCIRSkillRecord(
            skill_id=skill_id,
            name=name,
            description=description,
            category=category,
            steps=list(steps or []),
            tools_used=list(tools_used or []),
            success_criteria=success_criteria,
            examples=list(examples or []),
            tenant_id=tenant_id or self._tenant_id,
            source=source,
        )
        self._store(record)
        return record

    def extract_and_store(
        self,
        task_description: str,
        execution_trace: list[dict[str, Any]],
        tools_used: list[str],
        success: bool,
        category: str = "general",
        tenant_id: str | None = None,
    ) -> HCIRSkillRecord | None:
        """Extract a reusable SkillNode from a successful execution trace."""
        if not success:
            return None
        steps = [str(step.get("action", "")) for step in execution_trace if step.get("action")]
        if not steps:
            return None
        skill_id = f"skill_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        name = "_".join(task_description.lower().split()[:4]) or skill_id
        record = HCIRSkillRecord(
            skill_id=skill_id,
            name=name,
            description=task_description,
            category=category,
            steps=steps,
            tools_used=list(tools_used),
            success_criteria=f"Successfully completed: {task_description}",
            examples=[{"input": task_description, "steps": steps}],
            tenant_id=tenant_id or self._tenant_id,
            source="cognitive_trainer",
        )
        self._store(record)
        return record

    def get_skill(self, skill_id: str) -> HCIRSkillRecord | None:
        return self._records.get(skill_id)

    def delete_skill(self, skill_name: str) -> bool:
        target_ids = [
            sid for sid, rec in self._records.items() if rec.name == skill_name or sid == skill_name
        ]
        if not target_ids:
            return False
        ops = []
        for sid in target_ids:
            self._records.pop(sid, None)
            if self._ws.get_node(sid) is not None:
                ops.append(TransactionOperation(op=TransactionOp.REMOVE_NODE, node_id=sid))
        if ops:
            self._ws.commit_operations(ops, author="hcir_skill_store")
        return True

    def list_skills(self) -> list[HCIRSkillRecord]:
        return list(self._records.values())

    def stats(self) -> dict[str, Any]:
        records = list(self._records.values())
        categories = len({r.category for r in records})
        avg_sr = sum(r.success_rate for r in records) / len(records) if records else 0.0
        by_source: dict[str, int] = {}
        for r in records:
            by_source[r.source] = by_source.get(r.source, 0) + 1
        return {
            "total_skills": len(records),
            "categories": categories,
            "avg_success_rate": round(avg_sr, 3),
            "by_source": by_source,
        }
