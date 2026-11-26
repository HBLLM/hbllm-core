"""HCIR ISA v2 structural pattern unification over CognitiveGraph and mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field, field_validator

from hbllm.hcir.registers import (
    REGISTER_NAMES,
    RegisterFile,
    RegisterRef,
    SymbolicVar,
    is_register_ref,
    is_symbolic_var,
    normalize_var_name,
)


class NodePattern(BaseModel):
    """Pattern matching a single HCIR node and capturing its attributes into `?var`."""

    var: str = "?node"
    node_type: str | None = None
    id: str | None = None
    min_confidence: float | None = None
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("var")
    @classmethod
    def _norm_var(cls, value: str) -> str:
        return normalize_var_name(value)


class EdgePattern(BaseModel):
    """Pattern matching a hyperedge connecting `?source` / literal to `?target` / literal."""

    edge_type: str | None = None
    source: str
    target: str
    min_weight: float | None = None


class GraphPattern(BaseModel):
    """Serializable conjunctive / disjunctive / negated subgraph pattern."""

    nodes: list[NodePattern] = Field(default_factory=list)
    edges: list[EdgePattern] = Field(default_factory=list)
    any_of: list[GraphPattern] = Field(default_factory=list)
    none_of: list[GraphPattern] = Field(default_factory=list)
    distinct_nodes: bool = True


_GRAPH_PATTERN_KEYS = frozenset({"nodes", "edges", "any_of", "none_of", "distinct_nodes"})
_NODE_RESERVED_KEYS = frozenset({"var", "node_type", "id", "min_confidence", "tags", "attributes"})


def coerce_graph_pattern(raw: GraphPattern | Mapping[str, Any]) -> tuple[GraphPattern, bool]:
    """Convert a `GraphPattern` or shorthand dict into `(GraphPattern, is_shorthand)`."""
    if isinstance(raw, GraphPattern):
        return raw, False
    if any(k in _GRAPH_PATTERN_KEYS for k in raw):
        return GraphPattern.model_validate(dict(raw)), False
    attrs = dict(raw.get("attributes", {})) if isinstance(raw.get("attributes"), Mapping) else {}
    attrs.update({k: v for k, v in raw.items() if k not in _NODE_RESERVED_KEYS})
    min_c = float(raw["min_confidence"]) if raw.get("min_confidence") is not None else None
    np = NodePattern(
        var=str(raw.get("var", "?node")),
        node_type=str(raw["node_type"]) if raw.get("node_type") is not None else None,
        id=str(raw["id"]) if raw.get("id") is not None else None,
        min_confidence=min_c,
        tags=list(raw.get("tags", [])),
        attributes=attrs,
    )
    return GraphPattern(nodes=[np]), True


def _node_as_mapping(node: Any) -> dict[str, Any]:
    if isinstance(node, Mapping):
        return dict(node)
    return (
        dict(node.model_dump())
        if hasattr(node, "model_dump")
        else dict(getattr(node, "__dict__", {}))
    )


def _lookup_node_attr(node_map: Mapping[str, Any], key: str) -> tuple[bool, Any]:
    if key in node_map:
        return True, node_map[key]
    props = node_map.get("properties")
    if isinstance(props, Mapping) and key in props:
        return True, props[key]
    if key == "tag" and isinstance(node_map.get("tags"), list) and node_map["tags"]:
        return True, node_map["tags"]
    return False, None


def _enum_str(val: Any) -> Any:
    return val.value if hasattr(val, "value") and isinstance(getattr(val, "value"), str) else val


def _node_conf(nm: Mapping[str, Any]) -> float:
    conf = nm.get("confidence")
    if conf is None:
        unc = nm.get("uncertainty")
        conf = unc.get("confidence", 0.5) if isinstance(unc, Mapping) else 0.5
    return float(conf)


def _bind_var(bindings: dict[str, Any], vname: str, val: Any) -> dict[str, Any] | None:
    if vname in bindings:
        return bindings if bindings[vname] == val else None
    nxt = dict(bindings)
    nxt[vname] = val
    return nxt


def unify_value(
    pat: Any,
    val: Any,
    bindings: dict[str, Any],
    registers: RegisterFile | None = None,
) -> dict[str, Any] | None:
    """Unify pattern term *pat* against concrete value *val* under *bindings*."""
    next_b = dict(bindings)
    if isinstance(pat, RegisterRef) or (isinstance(pat, str) and pat in REGISTER_NAMES):
        slot = pat.slot if isinstance(pat, RegisterRef) else pat
        if registers is None or not registers.is_bound(slot):
            return None
        pat = registers.get(slot)

    if isinstance(pat, SymbolicVar):
        if pat.node_type is not None:
            actual_type = _enum_str(val.get("node_type") if isinstance(val, Mapping) else None)
            if actual_type != pat.node_type:
                return None
        return _bind_var(next_b, pat.name, val)
    if is_symbolic_var(pat):
        return _bind_var(next_b, normalize_var_name(str(pat)), val)
    if is_symbolic_var(val):
        return _bind_var(next_b, normalize_var_name(str(val)), pat)
    if isinstance(pat, Mapping) and isinstance(val, Mapping):
        return unify_dicts(pat, val, next_b, registers=registers)
    if isinstance(pat, (list, tuple)) and isinstance(val, (list, tuple)):
        if len(pat) != len(val):
            return None
        cur = next_b
        for p_item, v_item in zip(pat, val):
            nxt = unify_value(p_item, v_item, cur, registers=registers)
            if nxt is None:
                return None
            cur = nxt
        return cur
    return next_b if _enum_str(pat) == _enum_str(val) else None


def unify_dicts(
    pattern: Mapping[str, Any],
    candidate: Mapping[str, Any],
    bindings: Mapping[str, Any] | None = None,
    registers: RegisterFile | None = None,
) -> dict[str, Any] | None:
    """Unify all keys of *pattern* against *candidate* (supporting nested dicts & `?var`)."""
    cur = dict(bindings or {})
    for key, pat_val in pattern.items():
        if key not in candidate:
            return None
        nxt = unify_value(pat_val, candidate[key], cur, registers=registers)
        if nxt is None:
            return None
        cur = nxt
    return cur


def _match_node(
    np: NodePattern,
    node_map: Mapping[str, Any],
    bindings: dict[str, Any],
    registers: RegisterFile | None,
) -> dict[str, Any] | None:
    node_id = str(node_map.get("id", ""))
    cur = bindings
    if np.id is not None:
        nxt = unify_value(np.id, node_id, cur, registers)
        if nxt is None:
            return None
        cur = nxt
    if np.node_type is not None:
        ntype = str(_enum_str(node_map.get("node_type", "")))
        if is_symbolic_var(np.node_type) or is_register_ref(np.node_type):
            nxt = unify_value(np.node_type, ntype, cur, registers)
            if nxt is None:
                return None
            cur = nxt
        elif ntype != np.node_type:
            return None
    if np.min_confidence is not None and _node_conf(node_map) < np.min_confidence:
        return None
    node_tags = set(node_map.get("tags") or [])
    if any(tag not in node_tags for tag in np.tags):
        return None
    bound = _bind_var(cur, np.var, node_id)
    if bound is None:
        return None
    cur = bound
    cur.setdefault("__primary_id__", node_id)
    for attr_key, attr_pat in np.attributes.items():
        found, attr_val = _lookup_node_attr(node_map, attr_key)
        if not found:
            return None
        nxt = unify_value(attr_pat, attr_val, cur, registers)
        if nxt is None:
            return None
        cur = nxt
    return cur


def _resolve_term(
    term: str, bindings: Mapping[str, Any], registers: RegisterFile | None
) -> str | None:
    if term in REGISTER_NAMES:
        val = registers.get(term) if registers and registers.is_bound(term) else None
        return str(val) if val is not None else None
    if is_symbolic_var(term):
        val = bindings.get(normalize_var_name(term))
        return str(val) if val is not None else None
    return term


def _match_edges(
    edge_patterns: list[EdgePattern],
    edge_maps: list[dict[str, Any]],
    bindings: dict[str, Any],
    registers: RegisterFile | None,
) -> list[dict[str, Any]]:
    if not edge_patterns:
        return [bindings]
    results: list[dict[str, Any]] = []

    def _bt(idx: int, cur_b: dict[str, Any]) -> None:
        if idx == len(edge_patterns):
            results.append(cur_b)
            return
        ep = edge_patterns[idx]
        src_b = _resolve_term(ep.source, cur_b, registers)
        tgt_b = _resolve_term(ep.target, cur_b, registers)
        for edge in edge_maps:
            if ep.edge_type is not None and str(_enum_str(edge.get("edge_type"))) != ep.edge_type:
                continue
            if ep.min_weight is not None and float(edge.get("weight", 1.0)) < ep.min_weight:
                continue
            for s in (str(x) for x in edge.get("sources", [])):
                if src_b is not None and s != src_b:
                    continue
                for t in (str(y) for y in edge.get("targets", [])):
                    if tgt_b is not None and t != tgt_b:
                        continue
                    nxt = dict(cur_b)
                    if is_symbolic_var(ep.source) and src_b is None:
                        nxt[normalize_var_name(ep.source)] = s
                    if is_symbolic_var(ep.target) and tgt_b is None:
                        nxt[normalize_var_name(ep.target)] = t
                    _bt(idx + 1, nxt)

    _bt(0, bindings)
    return results


def _extract_graph_items(source: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if hasattr(source, "nodes") and hasattr(source, "edges"):
        raw_nodes, raw_edges = list(source.nodes.values()), list(source.edges.values())
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        raw_nodes, raw_edges = list(source), []
    elif isinstance(source, Mapping):
        raw_nodes, raw_edges = [source], []
    else:
        return [], []
    node_maps = sorted(
        (_node_as_mapping(n) for n in raw_nodes),
        key=lambda nm: (-_node_conf(nm), str(nm.get("id", ""))),
    )
    return node_maps, [_node_as_mapping(e) for e in raw_edges]


def unify_pattern(
    source: Any,
    raw_pattern: GraphPattern | Mapping[str, Any],
    *,
    initial_bindings: Mapping[str, Any] | None = None,
    registers: RegisterFile | None = None,
    limit: int = 32,
) -> list[dict[str, Any]]:
    """Deterministically unify *raw_pattern* against a graph or collection of mappings."""
    pattern, _ = coerce_graph_pattern(raw_pattern)
    base_b = dict(initial_bindings or {})
    if pattern.any_of:
        merged: list[dict[str, Any]] = []
        for sub in pattern.any_of:
            for match in unify_pattern(
                source, sub, initial_bindings=base_b, registers=registers, limit=limit
            ):
                merged.append(match)
                if len(merged) >= limit:
                    return merged
        return merged

    node_maps, edge_maps = _extract_graph_items(source)
    solutions: list[dict[str, Any]] = []

    def _bt_nodes(idx: int, used_ids: set[str], cur_b: dict[str, Any]) -> None:
        if len(solutions) >= limit:
            return
        if idx == len(pattern.nodes):
            for eb in _match_edges(pattern.edges, edge_maps, cur_b, registers):
                if any(
                    unify_pattern(source, neg, initial_bindings=eb, registers=registers, limit=1)
                    for neg in pattern.none_of
                ):
                    continue
                solutions.append(eb)
                if len(solutions) >= limit:
                    return
            return
        np = pattern.nodes[idx]
        for nm in node_maps:
            nid = str(nm.get("id", ""))
            if pattern.distinct_nodes and nid and nid in used_ids:
                continue
            matched_b = _match_node(np, nm, cur_b, registers)
            if matched_b is None:
                continue
            _bt_nodes(idx + 1, used_ids | ({nid} if nid else set()), matched_b)
            if len(solutions) >= limit:
                return

    _bt_nodes(0, set(), base_b)
    return solutions
