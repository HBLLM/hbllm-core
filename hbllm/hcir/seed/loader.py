"""Transactional loader for `core_priors.hcir.yaml` bounded by `docs/architecture/priors.md`."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hbllm.hcir.bytecode import InstructionStream
from hbllm.hcir.graph import (
    ConceptNode,
    HCIREdge,
    HCIREdgeType,
    LearnedRuleNode,
    NodeLifecycle,
    SkillNode,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope, SecurityLevel

DEFAULT_CORE_PRIORS_PATH: Path = Path(__file__).resolve().parent / "core_priors.hcir.yaml"
_FORBIDDEN_RE = re.compile(
    r"\b(?:lava|key|door)\b|color\s+\d+\s+is|action\s+\d+\s*=", re.IGNORECASE
)


def _assert_domain_purity(raw_text: str) -> None:
    if match := _FORBIDDEN_RE.search(raw_text):
        raise ValueError(
            f"core_priors.hcir.yaml violates docs/architecture/priors.md §3: {match.group(0)!r}"
        )


def load_core_priors_spec(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the declarative `.hcir` Spelke/active-inference seed YAML."""
    seed_path = Path(path) if path is not None else DEFAULT_CORE_PRIORS_PATH
    raw_text = seed_path.read_text(encoding="utf-8")
    _assert_domain_purity(raw_text)
    loaded = yaml.safe_load(raw_text)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Invalid seed format in {seed_path}: expected mapping")
    return dict(loaded)


def _upsert(node: Any) -> TransactionOperation:
    return TransactionOperation(
        op=TransactionOp.UPSERT_NODE, node_id=node.id, node_data=node.model_dump()
    )


def build_core_prior_operations(
    spec: Mapping[str, Any] | None = None, *, tenant_id: str | None = None
) -> list[TransactionOperation]:
    """Compile validated `core_priors.hcir.yaml` entries into atomic `TransactionOperation`s."""
    data = spec if spec is not None else load_core_priors_spec()
    prov = Provenance(created_by="core_priors_seed")
    scope = Scope(tenant_id=tenant_id or "default", security_level=SecurityLevel.SYSTEM)
    ops: list[TransactionOperation] = [
        _upsert(
            ConceptNode(
                id=str(c["id"]),
                label=str(c["label"]),
                definition=str(c.get("definition", "")),
                domain=str(c.get("domain", "core_priors")),
                confidence=float(c.get("confidence", 0.99)),
                lifecycle=NodeLifecycle.VALIDATED,
                provenance=prov,
                scope=scope,
                tags=["core_prior", str(c.get("domain", "core_priors"))],
            )
        )
        for c in data.get("concepts", ())
    ]
    ops.extend(
        _upsert(
            LearnedRuleNode(
                id=str(r["id"]),
                condition=str(r["condition"]),
                prediction=str(r["prediction"]),
                domain=str(r.get("domain", "core_priors")),
                confidence=float(r.get("confidence", 0.9)),
                lifecycle=NodeLifecycle.VALIDATED,
                provenance=prov,
                scope=scope,
                tags=["core_prior", "innate_rule"],
            )
        )
        for r in data.get("rules", ())
    )
    ops.extend(
        _upsert(
            SkillNode(
                id=str(s["id"]),
                skill_name=str(s["skill_name"]),
                description=str(s.get("description", "")),
                parameters=list(s.get("parameters", ())),
                precondition_pattern=s.get("precondition_pattern"),
                postcondition_pattern=s.get("postcondition_pattern"),
                bytecode_stream=InstructionStream.model_validate(s["bytecode_stream"]).model_dump(),
                alpha_success=5.0,
                beta_failure=1.0,
                lifecycle=NodeLifecycle.VALIDATED,
                provenance=prov,
                scope=scope,
                tags=["core_prior", "innate_skill"],
            )
        )
        for s in data.get("skills", ())
    )
    for e in data.get("edges", ()):
        edge = HCIREdge(
            id=str(e["id"]),
            edge_type=HCIREdgeType(str(e["edge_type"])),
            sources=[str(src) for src in e["sources"]],
            targets=[str(tgt) for tgt in e["targets"]],
            provenance=prov,
        )
        ops.append(
            TransactionOperation(
                op=TransactionOp.ADD_EDGE, edge_id=edge.id, edge_data=edge.model_dump()
            )
        )
    return ops


def hydrate_core_priors(
    target: Any, *, grant: Any = None, tenant_id: str | None = None, path: Path | None = None
) -> int:
    """Hydrate Layer-1 innate Spelke priors transactionally into `TieredWorkspace.brain` or workspace."""
    from hbllm.hcir.workspace_tiers import TieredWorkspace

    ops = build_core_prior_operations(load_core_priors_spec(path), tenant_id=tenant_id)
    ws = target.brain if isinstance(target, TieredWorkspace) else target
    resolved_grant = grant or (
        getattr(target, "_system_grant", None) if isinstance(target, TieredWorkspace) else None
    )
    filtered = [
        op
        for op in ops
        if not (op.op == TransactionOp.ADD_EDGE and op.edge_id and ws.graph.has_edge(op.edge_id))
    ]
    if not filtered:
        return 0
    res = ws.commit_operations(
        filtered, author="core_priors_loader", grant=resolved_grant, tenant_id=tenant_id
    )
    if not res.is_committed:
        raise RuntimeError(f"Failed to hydrate core_priors.hcir.yaml: {res.status}")
    return sum(1 for op in filtered if op.op == TransactionOp.UPSERT_NODE)
