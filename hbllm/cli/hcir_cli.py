"""CLI subcommands for inspecting, printing, validating, and round-tripping `.hcir` files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import typer
import yaml

from hbllm.hcir.bytecode import InstructionStream
from hbllm.hcir.nodes import HCIRNodeType
from hbllm.hcir.sexpr import (
    HCIRParseError,
    blake3_hex,
    exact_digest,
    format_hcir,
    parse_hcir,
    semantic_digest,
)

hcir_app = typer.Typer(
    name="hcir",
    help="Inspect, print, validate, and canonicalize `.hcir` symbolic programs and nodes",
    no_args_is_help=True,
)

_KIND_TO_NODE_TYPE: dict[str, HCIRNodeType | None] = {
    "skill": HCIRNodeType.SKILL,
    "procedure": HCIRNodeType.PROCEDURE,
    "rule": HCIRNodeType.LEARNED_RULE,
    "construction": HCIRNodeType.CONSTRUCTION,
    "concept": HCIRNodeType.CONCEPT,
    "adaptive-membership": HCIRNodeType.ADAPTIVE_MEMBERSHIP,
    "node": None,
}


def _resolve_mode(mode: str) -> Literal["canonical", "exact"]:
    if mode == "canonical":
        return "canonical"
    if mode == "exact":
        return "exact"
    typer.echo(f"Invalid --mode {mode!r}; expected 'canonical' or 'exact'", err=True)
    raise typer.Exit(code=1)


def _find_node_in_system(system: Any, kind: str, target_id: str) -> Any | None:
    expected_nt = _KIND_TO_NODE_TYPE.get(kind)
    tw = system.tiered_workspace
    tiers = (tw.working_state, tw.brain, tw.persistent, tw.meta)
    for ws in tiers:
        exact = ws.get_node(target_id)
        if exact is not None and (expected_nt is None or exact.node_type == expected_nt):
            return exact
    for ws in tiers:
        for node in ws.graph.all_nodes():
            if expected_nt is not None and node.node_type != expected_nt:
                continue
            for attr in ("skill_name", "procedure_name", "construction_name", "label", "name"):
                if getattr(node, attr, None) == target_id:
                    return node
    return None


def _bundle_or_object_digest(obj: Any) -> str:
    if isinstance(obj, Mapping):
        return blake3_hex(format_hcir(obj, mode="canonical").encode("utf-8"))
    return semantic_digest(obj)


@hcir_app.command("inspect")
def hcir_inspect(
    kind: str = typer.Argument(
        ...,
        help="Node kind: skill, procedure, rule, construction, concept, adaptive-membership, node",
    ),
    target_id: str = typer.Argument(..., help="Node ID or symbolic name (e.g. skill_name)"),
    world_id: str = typer.Option("default", "--world-id", help="World/Context identifier"),
    tenant_id: str = typer.Option("default", "--tenant-id", help="Tenant scope identifier"),
    memory_dir: Path | None = typer.Option(
        None, "--memory-dir", help="Optional persistent memory directory (contains persistent.db)"
    ),
    mode: str = typer.Option("canonical", "--mode", help="Formatting mode: canonical or exact"),
    no_envelope: bool = typer.Option(
        False, "--no-envelope", help="Omit top-level (hcir :version 1 :type ...) wrapper"
    ),
    digest: bool = typer.Option(
        False, "--digest", help="Print BLAKE3 semantic_digest and exact_digest comments"
    ),
) -> None:
    """Inspect a cognitive node from the HCIR workspace in canonical `.hcir` format."""
    norm_kind = kind.strip().lower()
    if norm_kind not in _KIND_TO_NODE_TYPE:
        typer.echo(
            f"Unsupported kind {kind!r}; expected one of: {', '.join(_KIND_TO_NODE_TYPE)}",
            err=True,
        )
        raise typer.Exit(code=1)
    fmt_mode = _resolve_mode(mode)

    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    cfg_kwargs: dict[str, Any] = {
        "world_id": world_id,
        "tenant_id": tenant_id,
        "enable_grounding": False,
    }
    if memory_dir is not None:
        cfg_kwargs["memory_dir"] = memory_dir
    system = build_cognitive_system(CognitiveSystemConfig(**cfg_kwargs))

    node = _find_node_in_system(system, norm_kind, target_id)
    if node is None:
        typer.echo(f"No {norm_kind} matching {target_id!r} found in workspace.", err=True)
        raise typer.Exit(code=1)

    if digest:
        typer.echo(f"; semantic_digest: {node.semantic_digest()}")
        typer.echo(f"; exact_digest: {node.exact_digest()}")
    typer.echo(format_hcir(node, mode=fmt_mode, envelope=not no_envelope), nl=False)


def _extract_stream_from_mapping(data: Mapping[str, Any]) -> InstructionStream:
    if "instructions" in data:
        return InstructionStream.model_validate(dict(data))
    if isinstance(data.get("bytecode_stream"), Mapping):
        return InstructionStream.model_validate(dict(data["bytecode_stream"]))
    if "capabilities_used" in data or "execution_id" in data:
        from hbllm.hcir.abi import ExecutionMetrics
        from hbllm.hcir.learning_loop import SkillCompiler
        from hbllm.hcir.receipt import ExecutionReceipt

        raw_m = data.get("metrics") or {}
        receipt = ExecutionReceipt(
            execution_id=str(data.get("execution_id", "rcpt_cli")),
            author=str(data.get("author", "cli")),
            capabilities_used=[str(c) for c in data.get("capabilities_used", ())],
            metrics=ExecutionMetrics(
                elapsed_ms=int(raw_m.get("elapsed_ms", 0)),
                tokens_consumed=int(raw_m.get("tokens_consumed", 0)),
            ),
            success=bool(data.get("success", True)),
        )
        skill = SkillCompiler().compile_receipt_to_skill(
            receipt, skill_name=str(data.get("skill_name") or receipt.execution_id)
        )
        assert skill.bytecode_stream is not None
        return InstructionStream.model_validate(skill.bytecode_stream)
    raise ValueError("Mapping does not contain 'instructions', 'bytecode_stream', or receipt data")


@hcir_app.command("print")
def hcir_print(
    kind: str = typer.Argument(..., help="Target kind: trace, seed, or file"),
    target: str = typer.Argument(..., help="Node/skill ID or path to .hcir/.json/.yaml file"),
    world_id: str = typer.Option("default", "--world-id", help="World/Context identifier"),
    tenant_id: str = typer.Option("default", "--tenant-id", help="Tenant scope identifier"),
    memory_dir: Path | None = typer.Option(
        None, "--memory-dir", help="Optional persistent memory directory (contains persistent.db)"
    ),
    mode: str = typer.Option("canonical", "--mode", help="Formatting mode: canonical or exact"),
    no_envelope: bool = typer.Option(
        False, "--no-envelope", help="Omit top-level (hcir :version 1 :type ...) wrapper"
    ),
) -> None:
    """Print an execution trace, skill stream, or `.hcir` file in canonical `.hcir` format."""
    norm_kind = kind.strip().lower()
    if norm_kind not in ("trace", "seed", "file"):
        typer.echo(f"Unsupported print kind {kind!r}; expected trace, seed, or file", err=True)
        raise typer.Exit(code=1)
    fmt_mode = _resolve_mode(mode)

    target_path = Path(target)
    if norm_kind in ("seed", "file"):
        if not target_path.exists():
            typer.echo(f"File not found: {target_path}", err=True)
            raise typer.Exit(code=1)
        if target_path.suffix == ".hcir":
            obj = parse_hcir(target_path.read_text(encoding="utf-8"))
        else:
            loaded = yaml.safe_load(target_path.read_text(encoding="utf-8"))
            obj = loaded
        typer.echo(format_hcir(obj, mode=fmt_mode, envelope=not no_envelope), nl=False)
        return

    stream: InstructionStream | None = None
    if target_path.exists():
        raw = target_path.read_text(encoding="utf-8")
        try:
            if target_path.suffix == ".hcir":
                parsed = parse_hcir(raw)
                if isinstance(parsed, InstructionStream):
                    stream = parsed
                elif getattr(parsed, "bytecode_stream", None) is not None:
                    stream = InstructionStream.model_validate(parsed.bytecode_stream)
                else:
                    raise ValueError(f"{target_path} does not hold an InstructionStream")
            else:
                data = json.loads(raw) if target_path.suffix == ".json" else yaml.safe_load(raw)
                if not isinstance(data, Mapping):
                    raise ValueError(f"Expected mapping in {target_path}")
                stream = _extract_stream_from_mapping(data)
        except (HCIRParseError, ValueError, json.JSONDecodeError) as exc:
            typer.echo(f"Failed to load trace from {target_path}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    else:
        from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

        cfg_kwargs: dict[str, Any] = {
            "world_id": world_id,
            "tenant_id": tenant_id,
            "enable_grounding": False,
        }
        if memory_dir is not None:
            cfg_kwargs["memory_dir"] = memory_dir
        system = build_cognitive_system(CognitiveSystemConfig(**cfg_kwargs))
        node = _find_node_in_system(system, "node", target)
        if node is not None:
            raw_bs = getattr(node, "bytecode_stream", None) or (node.properties or {}).get(
                "bytecode_stream"
            )
            if isinstance(raw_bs, Mapping):
                stream = InstructionStream.model_validate(dict(raw_bs))

    if stream is None:
        typer.echo(f"No instruction stream or trace found for {target!r}.", err=True)
        raise typer.Exit(code=1)
    typer.echo(format_hcir(stream, mode=fmt_mode, envelope=not no_envelope), nl=False)


@hcir_app.command("validate")
def hcir_validate(
    file_path: Path = typer.Argument(..., help="Path to `.hcir` file to validate"),
    core_priors: bool | None = typer.Option(
        None,
        "--core-priors/--no-core-priors",
        help="Enforce priors.md domain purity check (auto-detected by filename/bundle kind if omitted)",
    ),
) -> None:
    """Validate syntax, Rule-0 compliance, model schema, and domain purity of a `.hcir` file."""
    if not file_path.exists():
        typer.echo(f"File not found: {file_path}", err=True)
        raise typer.Exit(code=1)

    from hbllm.hcir.seed.loader import _assert_domain_purity

    raw_text = file_path.read_text(encoding="utf-8")
    try:
        if core_priors is True or (
            core_priors is None and file_path.name.startswith("core_priors")
        ):
            _assert_domain_purity(raw_text)
        parsed = parse_hcir(raw_text)
        if (
            core_priors is None
            and isinstance(parsed, Mapping)
            and parsed.get("kind") == "core-priors"
        ):
            _assert_domain_purity(raw_text)
        digest = _bundle_or_object_digest(parsed)
    except (HCIRParseError, ValueError, TypeError) as exc:
        typer.echo(f"INVALID {file_path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    kind_name = (
        f"seed-bundle:{parsed.get('kind', 'unknown')}"
        if isinstance(parsed, Mapping)
        else type(parsed).__name__
    )
    typer.echo(f"VALID: {file_path} (type={kind_name}, digest={digest[:16]})")


@hcir_app.command("roundtrip")
def hcir_roundtrip(
    file_path: Path = typer.Argument(..., help="Path to `.hcir` file to round-trip"),
    mode: str = typer.Option("canonical", "--mode", help="Formatting mode: canonical or exact"),
    in_place: bool = typer.Option(
        False, "--in-place", help="Rewrite the file with its canonicalized `.hcir` text"
    ),
) -> None:
    """Verify parse -> format -> parse -> format idempotence and digest stability."""
    if not file_path.exists():
        typer.echo(f"File not found: {file_path}", err=True)
        raise typer.Exit(code=1)
    fmt_mode = _resolve_mode(mode)

    raw_text = file_path.read_text(encoding="utf-8")
    try:
        obj1 = parse_hcir(raw_text)
        txt1 = format_hcir(obj1, mode=fmt_mode)
        obj2 = parse_hcir(txt1)
        txt2 = format_hcir(obj2, mode=fmt_mode)
        if txt1 != txt2:
            raise ValueError("Canonicalization idempotence check failed (txt1 != txt2)")
        if _bundle_or_object_digest(obj1) != _bundle_or_object_digest(obj2):
            raise ValueError("Semantic digest mismatch across round-trip")
        if fmt_mode == "exact" and not isinstance(obj1, Mapping):
            if obj1 != obj2 or exact_digest(obj1) != exact_digest(obj2):
                raise ValueError("Exact model/digest mismatch across round-trip")
    except (HCIRParseError, ValueError, TypeError) as exc:
        typer.echo(f"ROUNDTRIP FAILED {file_path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if in_place:
        file_path.write_text(txt1, encoding="utf-8")
        typer.echo(f"ROUNDTRIP OK (updated): {file_path}")
    else:
        typer.echo(txt1, nl=False)
