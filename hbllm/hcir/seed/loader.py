"""Transactional loader for `.hcir` and YAML genesis priors, skills, and constructions."""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from hbllm.hcir.bytecode import InstructionStream
from hbllm.hcir.fuzzy import RelationCompositionPolicy
from hbllm.hcir.graph import (
    ConceptNode,
    ConstructionNode,
    HCIREdge,
    HCIREdgeType,
    LearnedRuleNode,
    NodeLifecycle,
    SkillNode,
)
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope, SecurityLevel

_SEED_DIR: Path = Path(__file__).resolve().parent
DEFAULT_CORE_PRIORS_HCIR_PATH: Path = _SEED_DIR / "core_priors.hcir"
DEFAULT_CORE_PRIORS_YAML_PATH: Path = _SEED_DIR / "core_priors.hcir.yaml"
DEFAULT_CORE_PRIORS_PATH: Path = (
    DEFAULT_CORE_PRIORS_HCIR_PATH
    if DEFAULT_CORE_PRIORS_HCIR_PATH.exists()
    else DEFAULT_CORE_PRIORS_YAML_PATH
)
CONSTRUCTION_HCIR_SEED_PATHS: dict[str, Path] = {
    lang: _SEED_DIR / f"constructions_{lang}.hcir" for lang in ("en", "si", "ta")
}
CONSTRUCTION_YAML_SEED_PATHS: dict[str, Path] = {
    lang: _SEED_DIR / f"constructions_{lang}.yaml" for lang in ("en", "si", "ta")
}
CONSTRUCTION_SEED_PATHS: dict[str, Path] = {
    lang: (
        CONSTRUCTION_HCIR_SEED_PATHS[lang]
        if CONSTRUCTION_HCIR_SEED_PATHS[lang].exists()
        else CONSTRUCTION_YAML_SEED_PATHS[lang]
    )
    for lang in ("en", "si", "ta")
}
_FORBIDDEN_RE = re.compile(
    r"(?<!:)\b(?:lava|key|door)\b|color\s+\d+\s+is|action\s+\d+\s*=", re.IGNORECASE
)


def _assert_domain_purity(raw_text: str) -> None:
    if match := _FORBIDDEN_RE.search(raw_text):
        raise ValueError(
            f"core_priors seed violates docs/architecture/priors.md §3: {match.group(0)!r}"
        )


def _normalize_seed_bundle_dict(parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Convert parsed `.hcir` seed-bundle Pydantic items into plain dicts."""
    out: dict[str, Any] = {}
    for k, v in parsed.items():
        if k in ("concepts", "rules", "skills", "constructions", "edges") and isinstance(
            v, Sequence
        ):
            out[k] = [
                item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in v
            ]
        elif hasattr(v, "model_dump"):
            out[k] = v.model_dump()
        else:
            out[k] = v
    return out


def _load_seed_file(seed_path: Path, *, check_purity: bool = False) -> dict[str, Any]:
    raw_text = seed_path.read_text(encoding="utf-8")
    if check_purity:
        _assert_domain_purity(raw_text)
    if seed_path.suffix == ".hcir":
        from hbllm.hcir.sexpr import parse_hcir

        parsed = parse_hcir(raw_text, expected_type="seed-bundle")
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Invalid .hcir seed bundle in {seed_path}")
        return _normalize_seed_bundle_dict(parsed)
    loaded = yaml.safe_load(raw_text)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Invalid seed format in {seed_path}: expected mapping")
    return dict(loaded)


def load_core_priors_spec(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the declarative `.hcir` / `.hcir.yaml` Spelke/active-inference seed."""
    seed_path = (
        Path(path)
        if path is not None
        else (
            DEFAULT_CORE_PRIORS_HCIR_PATH
            if DEFAULT_CORE_PRIORS_HCIR_PATH.exists()
            else DEFAULT_CORE_PRIORS_YAML_PATH
        )
    )
    return _load_seed_file(seed_path, check_purity=True)


@functools.lru_cache(maxsize=8)
def _load_cached_constructions_spec(language: str) -> dict[str, Any]:
    hcir_path = CONSTRUCTION_HCIR_SEED_PATHS.get(language, CONSTRUCTION_HCIR_SEED_PATHS["en"])
    yaml_path = CONSTRUCTION_YAML_SEED_PATHS.get(language, CONSTRUCTION_YAML_SEED_PATHS["en"])
    seed_path = hcir_path if hcir_path.exists() else yaml_path
    return _load_seed_file(seed_path, check_purity=False)


def load_constructions_spec(language: str = "en", path: Path | None = None) -> dict[str, Any]:
    """Load multilingual Embodied Construction Grammar seed (`.hcir` or `.yaml`)."""
    if path is None:
        return _load_cached_constructions_spec(language)
    return _load_seed_file(Path(path), check_purity=False)


def _upsert(node: Any) -> TransactionOperation:
    return TransactionOperation(
        op=TransactionOp.UPSERT_NODE, node_id=node.id, node_data=node.model_dump()
    )


def _commit_seed_ops(
    target: Any,
    ops: Sequence[TransactionOperation],
    *,
    author: str,
    grant: Any,
    tenant_id: str | None,
) -> int:
    from hbllm.hcir.workspace_tiers import TieredWorkspace

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
    res = ws.commit_operations(filtered, author=author, grant=resolved_grant, tenant_id=tenant_id)
    if not res.is_committed:
        raise RuntimeError(f"Failed to hydrate {author}: {res.status}")
    return sum(1 for op in filtered if op.op == TransactionOp.UPSERT_NODE)


def build_seeded_constructions(
    language: str = "en", *, tenant_id: str | None = None
) -> list[ConstructionNode]:
    """Instantiate typed `ConstructionNode`s from the seed YAML for *language*."""
    data = load_constructions_spec(language)
    prov = Provenance(created_by="constructions_seed")
    scope = Scope(tenant_id=tenant_id or "default", security_level=SecurityLevel.SYSTEM)
    return [
        ConstructionNode(
            id=str(c["id"]),
            construction_name=str(c["construction_name"]),
            language=str(c.get("language", language)),
            intent_type=str(c.get("intent_type", "query")),
            syntactic_template=str(c.get("syntactic_template", "")),
            trigger_tokens=[str(t) for t in c.get("trigger_tokens", ())],
            slot_constraints=dict(c.get("slot_constraints", {})),
            semantic_mapping=dict(c.get("semantic_mapping", {})),
            compiled_hcir_template=[dict(i) for i in c.get("compiled_hcir_template", ())],
            alpha_success=float(c.get("alpha_success", 5.0)),
            beta_failure=float(c.get("beta_failure", 1.0)),
            lifecycle=NodeLifecycle.VALIDATED,
            provenance=prov,
            scope=scope,
            tags=["core_prior", "construction_seed", str(c.get("language", language))],
        )
        for c in data.get("constructions", ())
    ]


def build_construction_operations(
    languages: Sequence[str] = ("en", "si", "ta"), *, tenant_id: str | None = None
) -> list[TransactionOperation]:
    """Build atomic `UPSERT_NODE` transactions for seeded multilingual `ConstructionNode`s."""
    return [
        _upsert(cxn)
        for lang in languages
        for cxn in build_seeded_constructions(lang, tenant_id=tenant_id)
    ]


def hydrate_constructions(
    target: Any,
    *,
    languages: Sequence[str] = ("en", "si", "ta"),
    grant: Any = None,
    tenant_id: str | None = None,
) -> int:
    """Hydrate multilingual seeded `ConstructionNode`s transactionally into workspace."""
    ops = build_construction_operations(languages, tenant_id=tenant_id)
    return _commit_seed_ops(
        target, ops, author="constructions_loader", grant=grant, tenant_id=tenant_id
    )


def load_seeded_composition_policy(path: Path | None = None) -> RelationCompositionPolicy:
    """Load the declarative `RelationCompositionPolicy` from `core_priors.hcir.yaml`."""
    data = load_core_priors_spec(path)
    raw_policy = data.get("relation_composition_policy")
    if isinstance(raw_policy, Mapping):
        return RelationCompositionPolicy.model_validate(dict(raw_policy))
    return RelationCompositionPolicy.default_spatial_policy()


def build_core_prior_operations(
    spec: Mapping[str, Any] | None = None, *, tenant_id: str | None = None
) -> list[TransactionOperation]:
    """Compile validated `core_priors.hcir.yaml` entries into atomic `TransactionOperation`s."""
    data = spec if spec is not None else load_core_priors_spec()
    prov = Provenance(created_by="core_priors_seed")
    kw: dict[str, Any] = {
        "lifecycle": NodeLifecycle.VALIDATED,
        "provenance": prov,
        "scope": Scope(tenant_id=tenant_id or "default", security_level=SecurityLevel.SYSTEM),
    }
    ops: list[TransactionOperation] = [
        _upsert(
            ConceptNode(
                id=str(c["id"]),
                label=str(c["label"]),
                definition=str(c.get("definition", "")),
                domain=str(c.get("domain", "core_priors")),
                confidence=float(c.get("confidence", 0.99)),
                tags=["core_prior", str(c.get("domain", "core_priors"))],
                **kw,
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
                tags=["core_prior", "innate_rule"],
                **kw,
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
                alpha_success=float(s.get("alpha_success", 5.0)),
                beta_failure=float(s.get("beta_failure", 1.0)),
                tags=["core_prior", "innate_skill"],
                **kw,
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
    if isinstance(raw_policy := data.get("relation_composition_policy"), Mapping):
        policy = RelationCompositionPolicy.model_validate(dict(raw_policy))
        ops.append(
            _upsert(
                ConceptNode(
                    id="prior_policy_relation_composition",
                    label="declarative_relation_composition_policy",
                    definition="Declarative scale- and geometry-guarded spatial relation composition rules.",
                    domain="spatial_topology",
                    confidence=0.99,
                    properties={"policy": policy.model_dump()},
                    tags=["core_prior", "composition_policy"],
                    **kw,
                )
            )
        )
    return ops


def hydrate_core_priors(
    target: Any, *, grant: Any = None, tenant_id: str | None = None, path: Path | None = None
) -> int:
    """Hydrate Layer-1 innate Spelke priors and seeded constructions into `TieredWorkspace.brain`."""
    ops = build_core_prior_operations(load_core_priors_spec(path), tenant_id=tenant_id)
    count = _commit_seed_ops(
        target, ops, author="core_priors_loader", grant=grant, tenant_id=tenant_id
    )
    if path is None:
        count += hydrate_constructions(target, grant=grant, tenant_id=tenant_id)
    return count


def load_skills_seed(path: Path) -> list[SkillNode]:
    """Load executable homoiconic `SkillNode`s from a `.hcir` or `.yaml` seed file."""
    seed_path = Path(path)
    raw_text = seed_path.read_text(encoding="utf-8")
    if seed_path.suffix == ".hcir":
        from hbllm.hcir.sexpr import parse_hcir

        parsed = parse_hcir(raw_text)
        if isinstance(parsed, SkillNode):
            skills = [parsed]
        elif isinstance(parsed, Mapping) and "skills" in parsed:
            skills = [
                s if isinstance(s, SkillNode) else SkillNode.model_validate(dict(s))
                for s in parsed["skills"]
            ]
        else:
            raise ValueError(f"Expected skill or seed-bundle with skills in {seed_path}")
    else:
        loaded = yaml.safe_load(raw_text)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Invalid YAML skill seed in {seed_path}")
        skills = [SkillNode.model_validate(dict(s)) for s in loaded.get("skills", ())]
    for sk in skills:
        if sk.bytecode_stream is None:
            raise ValueError(
                f"Skill '{sk.id}' in {seed_path} is missing an executable bytecode_stream"
            )
    return skills


def hydrate_skills_seed(
    target: Any,
    path: Path,
    *,
    grant: Any = None,
    tenant_id: str | None = None,
) -> int:
    """Hydrate homoiconic `SkillNode`s from a `.hcir` or `.yaml` skill seed transactionally."""
    ops = [_upsert(sk) for sk in load_skills_seed(path)]
    return _commit_seed_ops(
        target, ops, author="skills_seed_loader", grant=grant, tenant_id=tenant_id
    )
