"""Canonical `.hcir` S-expression codec and BLAKE3 semantic digest (ADR 007).

This module is a lossless serialization/inspection boundary over existing HCIR
structures (`InstructionStream`, `GraphPattern`, `SymbolicExpr`, `FuzzyExpr`,
`HCIRNode` subclasses, `RelationCompositionPolicy`, `HCIREdge`, and seed
bundles).  It does not introduce a second runtime or Lisp evaluator.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.fuzzy import (
    FuzzyExpr,
    RelationCompositionPolicy,
    RelationCompositionRule,
)
from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType
from hbllm.hcir.nodes import (
    NODE_TYPE_REGISTRY,
    AdaptiveMembershipNode,
    ConceptNode,
    ConstructionNode,
    HCIRNode,
    HCIRNodeType,
    LearnedRuleNode,
    ProcedureNode,
    SkillNode,
)
from hbllm.hcir.registers import REGISTER_NAMES
from hbllm.hcir.skills.declarative import SymbolicExpr
from hbllm.hcir.unification import EdgePattern, GraphPattern, NodePattern

try:
    from hbllm_hcir_graph import blake3_hash as _native_blake3_hash
except ImportError:
    _native_blake3_hash = None

__all__ = [
    "HCIRParseError",
    "blake3_hex",
    "canonical_digest",
    "exact_digest",
    "format_hcir",
    "parse_hcir",
    "semantic_digest",
]

# ═══════════════════════════════════════════════════════════════════════════
# BLAKE3 Substrate (Native PyO3 + Pure-Python Spec Parity Fallback)
# ═══════════════════════════════════════════════════════════════════════════

_BLAKE3_IV: tuple[int, ...] = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
_BLAKE3_MSG_PERM: tuple[int, ...] = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_B3_CHUNK_START = 1 << 0
_B3_CHUNK_END = 1 << 1
_B3_PARENT = 1 << 2
_B3_ROOT = 1 << 3
_B3_BLOCK_LEN = 64
_B3_CHUNK_LEN = 1024
_MASK32 = 0xFFFFFFFF


def _rotr32(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK32


def _b3_g(st: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    st[a] = (st[a] + st[b] + mx) & _MASK32
    st[d] = _rotr32(st[d] ^ st[a], 16)
    st[c] = (st[c] + st[d]) & _MASK32
    st[b] = _rotr32(st[b] ^ st[c], 12)
    st[a] = (st[a] + st[b] + my) & _MASK32
    st[d] = _rotr32(st[d] ^ st[a], 8)
    st[c] = (st[c] + st[d]) & _MASK32
    st[b] = _rotr32(st[b] ^ st[c], 7)


def _b3_round(st: list[int], m: Sequence[int]) -> None:
    _b3_g(st, 0, 4, 8, 12, m[0], m[1])
    _b3_g(st, 1, 5, 9, 13, m[2], m[3])
    _b3_g(st, 2, 6, 10, 14, m[4], m[5])
    _b3_g(st, 3, 7, 11, 15, m[6], m[7])
    _b3_g(st, 0, 5, 10, 15, m[8], m[9])
    _b3_g(st, 1, 6, 11, 12, m[10], m[11])
    _b3_g(st, 2, 7, 8, 13, m[12], m[13])
    _b3_g(st, 3, 4, 9, 14, m[14], m[15])


def _b3_compress(
    cv: tuple[int, ...], block_words: list[int], counter: int, block_len: int, flags: int
) -> list[int]:
    st = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        _BLAKE3_IV[0],
        _BLAKE3_IV[1],
        _BLAKE3_IV[2],
        _BLAKE3_IV[3],
        counter & _MASK32,
        (counter >> 32) & _MASK32,
        block_len,
        flags,
    ]
    m = list(block_words)
    for round_idx in range(7):
        _b3_round(st, m)
        if round_idx < 6:
            m = [m[i] for i in _BLAKE3_MSG_PERM]
    for i in range(8):
        st[i] ^= st[i + 8]
        st[i + 8] ^= cv[i]
    return st


def _b3_words_from_bytes(block: bytes) -> list[int]:
    padded = block + b"\x00" * (_B3_BLOCK_LEN - len(block))
    return list(struct.unpack("<16I", padded))


def _b3_parent_output(
    left_cv: tuple[int, ...], right_cv: tuple[int, ...]
) -> tuple[tuple[int, ...], list[int], int, int, int]:
    return _BLAKE3_IV, list(left_cv) + list(right_cv), 0, _B3_BLOCK_LEN, _B3_PARENT


def _b3_chunk_compress(chunk: bytes, chunk_counter: int, is_root: bool) -> tuple[int, ...]:
    cv = _BLAKE3_IV
    n = len(chunk)
    offset = 0
    blocks_compressed = 0
    while True:
        block = chunk[offset : offset + _B3_BLOCK_LEN]
        is_first = blocks_compressed == 0
        is_last = offset + _B3_BLOCK_LEN >= n
        flags = (
            (_B3_CHUNK_START if is_first else 0)
            | (_B3_CHUNK_END if is_last else 0)
            | (_B3_ROOT if (is_root and is_last) else 0)
        )
        words = _b3_words_from_bytes(block)
        out = _b3_compress(cv, words, chunk_counter, len(block), flags)
        if is_last:
            return tuple(out[:8])
        cv = tuple(out[:8])
        blocks_compressed += 1
        offset += _B3_BLOCK_LEN


def _py_blake3_hex(data: bytes) -> str:
    """Pure-Python BLAKE3 (32-byte digest) matching `hbllm_hcir_graph.blake3_hash`."""
    total_len = len(data)
    if total_len <= _B3_CHUNK_LEN:
        cv = _b3_chunk_compress(data, 0, is_root=True)
        return struct.pack("<8I", *cv).hex()

    cv_stack: list[tuple[int, ...]] = []
    chunk_counter = 0
    pos = 0
    while pos + _B3_CHUNK_LEN < total_len:
        chunk_cv = _b3_chunk_compress(data[pos : pos + _B3_CHUNK_LEN], chunk_counter, is_root=False)
        chunk_counter += 1
        total_chunks = chunk_counter
        while total_chunks & 1 == 0:
            left_cv = cv_stack.pop()
            p_cv, p_words, p_ctr, p_len, p_flags = _b3_parent_output(left_cv, chunk_cv)
            chunk_cv = tuple(_b3_compress(p_cv, p_words, p_ctr, p_len, p_flags)[:8])
            total_chunks >>= 1
        cv_stack.append(chunk_cv)
        pos += _B3_CHUNK_LEN

    last_cv = _b3_chunk_compress(data[pos:], chunk_counter, is_root=False)
    while len(cv_stack) > 1:
        left_cv = cv_stack.pop()
        p_cv, p_words, p_ctr, p_len, p_flags = _b3_parent_output(left_cv, last_cv)
        last_cv = tuple(_b3_compress(p_cv, p_words, p_ctr, p_len, p_flags)[:8])
    left_cv = cv_stack.pop()
    p_cv, p_words, p_ctr, p_len, p_flags = _b3_parent_output(left_cv, last_cv)
    root_cv = tuple(_b3_compress(p_cv, p_words, p_ctr, p_len, p_flags | _B3_ROOT)[:8])
    return struct.pack("<8I", *root_cv).hex()


def blake3_hex(data: bytes) -> str:
    """Compute a 64-character BLAKE3 hex digest using the Rust substrate or Python fallback."""
    if _native_blake3_hash is not None:
        return str(_native_blake3_hash(data))
    return _py_blake3_hex(data)


# ═══════════════════════════════════════════════════════════════════════════
# Errors & Concrete S-Expression AST
# ═══════════════════════════════════════════════════════════════════════════


class HCIRParseError(ValueError):
    """Raised when `.hcir` text fails lexical, syntactic, or Rule-0 validation."""

    def __init__(self, message: str, *, line: int = 1, col: int = 1) -> None:
        self.line = line
        self.col = col
        super().__init__(f"line {line}, col {col}: {message}")


@dataclass(frozen=True)
class SAtomNil:
    line: int
    col: int


@dataclass(frozen=True)
class SAtomBool:
    value: bool
    line: int
    col: int


@dataclass(frozen=True)
class SAtomInt:
    value: int
    line: int
    col: int


@dataclass(frozen=True)
class SAtomFloat:
    value: float
    line: int
    col: int


@dataclass(frozen=True)
class SAtomStr:
    value: str
    line: int
    col: int


@dataclass(frozen=True)
class SAtomReg:
    slot: str
    line: int
    col: int


@dataclass(frozen=True)
class SAtomVar:
    name: str
    line: int
    col: int


@dataclass(frozen=True)
class SAtomKw:
    name: str
    line: int
    col: int


@dataclass(frozen=True)
class SAtomSym:
    name: str
    line: int
    col: int


@dataclass(frozen=True)
class SList:
    items: tuple[SNode, ...]
    line: int
    col: int


SNode = (
    SAtomNil
    | SAtomBool
    | SAtomInt
    | SAtomFloat
    | SAtomStr
    | SAtomReg
    | SAtomVar
    | SAtomKw
    | SAtomSym
    | SList
)

_FORBIDDEN_LISP_HEADS: frozenset[str] = frozenset({"defmacro", "eval", "lambda", "quote", "if"})
_VAR_RE = re.compile(r"^\?[A-Za-z0-9_.-]+$")
_KW_RE = re.compile(r"^:[A-Za-z_][A-Za-z0-9_-]*$")
_SAFE_SYM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./*+\-?]*$")
_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_FLOAT_RE = re.compile(r"^[+-]?[0-9]+(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)$")

# ═══════════════════════════════════════════════════════════════════════════
# Lexer & Recursive-Descent S-Expression Parser
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    line: int
    col: int


def _tokenize(src: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(src)
    line = 1
    col = 1

    while i < n:
        ch = src[i]
        if ch in (" ", "\t", "\r"):
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if ch == ";":
            while i < n and src[i] != "\n":
                i += 1
                col += 1
            continue
        if ch == "(":
            tokens.append(_Token("LPAREN", "(", line, col))
            i += 1
            col += 1
            continue
        if ch == ")":
            tokens.append(_Token("RPAREN", ")", line, col))
            i += 1
            col += 1
            continue
        if ch == '"':
            start_line, start_col = line, col
            i += 1
            col += 1
            buf: list[str] = []
            closed = False
            while i < n:
                c = src[i]
                if c == '"':
                    i += 1
                    col += 1
                    closed = True
                    break
                if c == "\\":
                    i += 1
                    col += 1
                    if i >= n:
                        raise HCIRParseError("Unterminated escape in string", line=line, col=col)
                    esc = src[i]
                    if esc in ('"', "\\", "/"):
                        buf.append(esc)
                        i += 1
                        col += 1
                    elif esc == "n":
                        buf.append("\n")
                        i += 1
                        col += 1
                    elif esc == "r":
                        buf.append("\r")
                        i += 1
                        col += 1
                    elif esc == "t":
                        buf.append("\t")
                        i += 1
                        col += 1
                    elif esc == "u":
                        hex_part = src[i + 1 : i + 5]
                        if len(hex_part) < 4 or not all(
                            hc in "0123456789abcdefABCDEF" for hc in hex_part
                        ):
                            raise HCIRParseError(
                                f"Invalid \\u escape: {hex_part!r}", line=line, col=col
                            )
                        buf.append(chr(int(hex_part, 16)))
                        i += 5
                        col += 5
                    else:
                        raise HCIRParseError(
                            f"Unsupported string escape \\{esc}", line=line, col=col
                        )
                elif c == "\n":
                    buf.append(c)
                    i += 1
                    line += 1
                    col = 1
                else:
                    buf.append(c)
                    i += 1
                    col += 1
            if not closed:
                raise HCIRParseError("Unterminated string literal", line=start_line, col=start_col)
            tokens.append(_Token("STRING", "".join(buf), start_line, start_col))
            continue

        start_i = i
        start_col = col
        while i < n and src[i] not in (" ", "\t", "\r", "\n", "(", ")", '"', ";"):
            i += 1
            col += 1
        tokens.append(_Token("ATOM", src[start_i:i], line, start_col))

    return tokens


def _classify_atom(tok: _Token) -> SNode:
    txt, line, col = tok.text, tok.line, tok.col
    if tok.kind == "STRING":
        return SAtomStr(txt, line, col)
    if txt == "nil":
        return SAtomNil(line, col)
    if txt == "true":
        return SAtomBool(True, line, col)
    if txt == "false":
        return SAtomBool(False, line, col)
    if txt.lower() in ("nan", "+nan", "-nan"):
        raise HCIRParseError("NaN float literal is not allowed in .hcir", line=line, col=col)
    if txt in ("+inf", "-inf"):
        return SAtomFloat(float("inf") if txt == "+inf" else float("-inf"), line, col)
    if _INT_RE.fullmatch(txt):
        return SAtomInt(int(txt), line, col)
    if _FLOAT_RE.fullmatch(txt):
        val = float(txt)
        if val == 0.0:
            val = 0.0
        return SAtomFloat(val, line, col)
    if txt.startswith("$"):
        if txt in REGISTER_NAMES:
            return SAtomReg(txt, line, col)
        raise HCIRParseError(
            f"Invalid register {txt!r}; expected one of $r0..$r7", line=line, col=col
        )
    if txt.startswith("?"):
        if not _VAR_RE.fullmatch(txt):
            raise HCIRParseError(f"Invalid symbolic variable {txt!r}", line=line, col=col)
        return SAtomVar(txt, line, col)
    if txt.startswith(":"):
        if not _KW_RE.fullmatch(txt):
            raise HCIRParseError(f"Invalid keyword {txt!r}", line=line, col=col)
        return SAtomKw(txt[1:], line, col)
    if txt == "->" or _SAFE_SYM_RE.fullmatch(txt):
        return SAtomSym(txt, line, col)
    raise HCIRParseError(f"Invalid atom token {txt!r}", line=line, col=col)


def _parse_sexpr_nodes(src: str) -> list[SNode]:
    tokens = _tokenize(src)
    pos = 0
    total = len(tokens)

    def parse_one() -> SNode:
        nonlocal pos
        tok = tokens[pos]
        if tok.kind == "LPAREN":
            start_line, start_col = tok.line, tok.col
            pos += 1
            items: list[SNode] = []
            while pos < total and tokens[pos].kind != "RPAREN":
                items.append(parse_one())
            if pos >= total:
                raise HCIRParseError("Unclosed '('", line=start_line, col=start_col)
            pos += 1
            return SList(tuple(items), start_line, start_col)
        if tok.kind == "RPAREN":
            raise HCIRParseError("Unexpected ')'", line=tok.line, col=tok.col)
        pos += 1
        return _classify_atom(tok)

    forms: list[SNode] = []
    while pos < total:
        forms.append(parse_one())
    return forms


# ═══════════════════════════════════════════════════════════════════════════
# Lossless Value Bijection (Untyped Payloads)
# ═══════════════════════════════════════════════════════════════════════════


def _format_float(val: float) -> str:
    fv = float(val)
    if math.isnan(fv):
        raise ValueError("NaN is not permitted in canonical .hcir")
    if math.isinf(fv):
        return "+inf" if fv > 0.0 else "-inf"
    if fv == 0.0:
        fv = 0.0
    s = repr(fv)
    if "." not in s and "e" not in s and "E" not in s:
        s = f"{s}.0"
    return s


def _quote_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _is_bare_ident(value: str) -> bool:
    return (
        value not in ("nil", "true", "false")
        and bool(_SAFE_SYM_RE.fullmatch(value))
        and not value.startswith(("?", ":", "$"))
    )


def _encode_ident(value: str) -> SAtomSym | SAtomStr:
    if _is_bare_ident(value):
        return SAtomSym(value, 0, 0)
    return SAtomStr(value, 0, 0)


def _encode_str_token(value: str) -> SAtomReg | SAtomVar | SAtomStr:
    if value in REGISTER_NAMES:
        return SAtomReg(value, 0, 0)
    if _VAR_RE.fullmatch(value):
        return SAtomVar(value, 0, 0)
    return SAtomStr(value, 0, 0)


def _encode_value(value: Any) -> SNode:
    if value is None:
        return SAtomNil(0, 0)
    if isinstance(value, bool):
        return SAtomBool(value, 0, 0)
    if isinstance(value, int):
        return SAtomInt(value, 0, 0)
    if isinstance(value, float):
        return SAtomFloat(float(value), 0, 0)
    if isinstance(value, str):
        return _encode_str_token(value)
    if isinstance(value, tuple):
        return SList((SAtomSym("tuple", 0, 0), *(_encode_value(v) for v in value)), 0, 0)
    if isinstance(value, list):
        return SList((SAtomSym("list", 0, 0), *(_encode_value(v) for v in value)), 0, 0)
    if isinstance(value, Mapping):
        entries = sorted(value.items(), key=lambda kv: str(kv[0]))
        pairs = tuple(SList((SAtomStr(str(k), 0, 0), _encode_value(v)), 0, 0) for k, v in entries)
        return SList((SAtomSym("map", 0, 0), *pairs), 0, 0)
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return _encode_str_token(str(value.value))
    if hasattr(value, "model_dump"):
        return _encode_value(value.model_dump())
    raise TypeError(f"Unsupported value type for .hcir encoding: {type(value).__name__}")


def _decode_str(node: SNode, *, allow_sym: bool = True, allow_reg_var: bool = True) -> str:
    if isinstance(node, SAtomStr):
        return node.value
    if allow_reg_var and isinstance(node, SAtomReg):
        return node.slot
    if allow_reg_var and isinstance(node, SAtomVar):
        return node.name
    if allow_sym and isinstance(node, SAtomSym):
        return node.name
    raise HCIRParseError(f"Expected string/symbol atom, got {node!r}", line=node.line, col=node.col)


def _decode_kv_pairs(nodes: Sequence[SNode]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in nodes:
        if not isinstance(item, SList) or len(item.items) != 2:
            raise HCIRParseError(
                "Expected (key value) pair in map/params", line=item.line, col=item.col
            )
        k = _decode_str(item.items[0], allow_sym=True, allow_reg_var=False)
        out[k] = _decode_value(item.items[1])
    return out


def _decode_value(node: SNode) -> Any:
    if isinstance(node, SAtomNil):
        return None
    if isinstance(node, SAtomBool):
        return node.value
    if isinstance(node, SAtomInt):
        return node.value
    if isinstance(node, SAtomFloat):
        return node.value
    if isinstance(node, SAtomStr):
        return node.value
    if isinstance(node, SAtomReg):
        return node.slot
    if isinstance(node, SAtomVar):
        return node.name
    if isinstance(node, SAtomSym):
        return node.name
    if isinstance(node, SAtomKw):
        raise HCIRParseError(
            f"Unexpected keyword :{node.name} in value position", line=node.line, col=node.col
        )
    assert isinstance(node, SList)
    if not node.items:
        raise HCIRParseError(
            "Empty '()' is not valid in untyped value position", line=node.line, col=node.col
        )
    head = node.items[0]
    if not isinstance(head, SAtomSym):
        raise HCIRParseError(
            "Expected symbolic head in value container", line=head.line, col=head.col
        )
    if head.name in _FORBIDDEN_LISP_HEADS:
        raise HCIRParseError(
            f"Lisp form '{head.name}' is forbidden by ADR 007 Rule 0", line=head.line, col=head.col
        )
    if head.name == "list":
        return [_decode_value(item) for item in node.items[1:]]
    if head.name == "tuple":
        return tuple(_decode_value(item) for item in node.items[1:])
    if head.name == "map":
        return _decode_kv_pairs(node.items[1:])
    if head.name == "pattern":
        return _decode_graph_pattern(node).model_dump()
    if head.name == "pattern-map":
        return _decode_kv_pairs(node.items[1:])
    if head.name == "expr-guard":
        return _decode_expr_guard(node)
    if head.name == "fuzzy-ast":
        if len(node.items) != 2:
            raise HCIRParseError(
                "(fuzzy-ast ...) requires 1 child form", line=node.line, col=node.col
            )
        return _decode_fuzzy_expr(node.items[1]).model_dump()
    raise HCIRParseError(
        f"Unsupported form '{head.name}' in untyped value position", line=head.line, col=head.col
    )


def _decode_bool(node: SNode) -> bool:
    if isinstance(node, SAtomBool):
        return node.value
    raise HCIRParseError(f"Expected true/false boolean, got {node!r}", line=node.line, col=node.col)


def _decode_int(node: SNode) -> int:
    if isinstance(node, SAtomInt):
        return node.value
    raise HCIRParseError(f"Expected integer, got {node!r}", line=node.line, col=node.col)


def _decode_float(node: SNode) -> float:
    if isinstance(node, SAtomFloat):
        return node.value
    if isinstance(node, SAtomInt):
        return float(node.value)
    raise HCIRParseError(f"Expected float/int number, got {node!r}", line=node.line, col=node.col)


def _head_sym(lst: SList) -> str:
    if not lst.items or not isinstance(lst.items[0], SAtomSym):
        raise HCIRParseError("Expected list with symbolic head", line=lst.line, col=lst.col)
    h = lst.items[0].name
    if h in _FORBIDDEN_LISP_HEADS:
        raise HCIRParseError(
            f"Form '({h} ...)' is forbidden in .hcir v1 (ADR 007 Rule 0)",
            line=lst.items[0].line,
            col=lst.items[0].col,
        )
    return h


def _split_kw_and_pos(items: Sequence[SNode]) -> tuple[dict[str, SNode], list[SNode]]:
    kws: dict[str, SNode] = {}
    pos: list[SNode] = []
    i = 0
    n = len(items)
    while i < n:
        cur = items[i]
        if isinstance(cur, SAtomKw):
            if i + 1 >= n:
                raise HCIRParseError(
                    f"Keyword :{cur.name} missing value", line=cur.line, col=cur.col
                )
            if cur.name in kws:
                raise HCIRParseError(f"Duplicate keyword :{cur.name}", line=cur.line, col=cur.col)
            kws[cur.name] = items[i + 1]
            i += 2
        else:
            pos.append(cur)
            i += 1
    return kws, pos


# ═══════════════════════════════════════════════════════════════════════════
# Family 2 — GraphPattern Codec
# ═══════════════════════════════════════════════════════════════════════════


def _encode_edge_type_kw(edge_type: str | None) -> SAtomKw:
    if edge_type is None:
        return SAtomKw("any", 0, 0)
    return SAtomKw(edge_type, 0, 0)


def _encode_node_pattern(np: NodePattern) -> SList:
    items: list[SNode] = [SAtomSym("node", 0, 0), _encode_str_token(np.var)]
    if np.node_type is not None:
        items.extend((SAtomKw("type", 0, 0), _encode_ident(np.node_type)))
    if np.id is not None:
        items.extend((SAtomKw("id", 0, 0), SAtomStr(np.id, 0, 0)))
    if np.min_confidence is not None:
        items.extend((SAtomKw("min-confidence", 0, 0), SAtomFloat(float(np.min_confidence), 0, 0)))
    if np.tags:
        items.extend(
            (SAtomKw("tags", 0, 0), SList(tuple(SAtomStr(t, 0, 0) for t in np.tags), 0, 0))
        )
    if np.attributes:
        pairs = tuple(
            SList((SAtomStr(str(k), 0, 0), _encode_value(v)), 0, 0)
            for k, v in sorted(np.attributes.items(), key=lambda kv: str(kv[0]))
        )
        items.append(SList((SAtomSym("attrs", 0, 0), *pairs), 0, 0))
    if np.fuzzy_memberships:
        pairs = tuple(
            SList((SAtomStr(str(k), 0, 0), SAtomFloat(float(v), 0, 0)), 0, 0)
            for k, v in sorted(np.fuzzy_memberships.items(), key=lambda kv: str(kv[0]))
        )
        items.append(SList((SAtomSym("fuzzy", 0, 0), *pairs), 0, 0))
    if np.numeric_tolerances:
        t_items: list[SNode] = [SAtomSym("tolerances", 0, 0)]
        for attr, spec in sorted(np.numeric_tolerances.items(), key=lambda kv: str(kv[0])):
            sp_pairs = tuple(
                SList((SAtomStr(str(sk), 0, 0), SAtomFloat(float(sv), 0, 0)), 0, 0)
                for sk, sv in sorted(spec.items(), key=lambda kv: str(kv[0]))
            )
            t_items.append(SList((SAtomStr(str(attr), 0, 0), *sp_pairs), 0, 0))
        items.append(SList(tuple(t_items), 0, 0))
    return SList(tuple(items), 0, 0)


def _decode_node_pattern(node: SNode) -> NodePattern:
    if not isinstance(node, SList) or _head_sym(node) != "node":
        raise HCIRParseError("Expected (node ...)", line=node.line, col=node.col)
    kws, pos = _split_kw_and_pos(node.items[1:])
    if not pos:
        raise HCIRParseError("(node ...) requires <var> argument", line=node.line, col=node.col)
    var_name = _decode_str(pos[0])
    attrs: dict[str, Any] = {}
    fuzzy_mem: dict[str, float] = {}
    num_tols: dict[str, dict[str, float]] = {}
    for sub in pos[1:]:
        if not isinstance(sub, SList):
            raise HCIRParseError(
                "Expected (attrs ...)/(fuzzy ...)/(tolerances ...)", line=sub.line, col=sub.col
            )
        sh = _head_sym(sub)
        if sh == "attrs":
            attrs.update(_decode_kv_pairs(sub.items[1:]))
        elif sh == "fuzzy":
            for p in sub.items[1:]:
                if not isinstance(p, SList) or len(p.items) != 2:
                    raise HCIRParseError(
                        "Expected (name degree) in (fuzzy ...)", line=p.line, col=p.col
                    )
                fuzzy_mem[_decode_str(p.items[0], allow_reg_var=False)] = _decode_float(p.items[1])
        elif sh == "tolerances":
            for t_entry in sub.items[1:]:
                if not isinstance(t_entry, SList) or len(t_entry.items) < 2:
                    raise HCIRParseError(
                        "Expected (attr (k v) ...) in (tolerances ...)",
                        line=t_entry.line,
                        col=t_entry.col,
                    )
                attr_k = _decode_str(t_entry.items[0], allow_reg_var=False)
                spec_map: dict[str, float] = {}
                for sp in t_entry.items[1:]:
                    if not isinstance(sp, SList) or len(sp.items) != 2:
                        raise HCIRParseError(
                            "Expected (key float) tolerance spec", line=sp.line, col=sp.col
                        )
                    spec_map[_decode_str(sp.items[0], allow_reg_var=False)] = _decode_float(
                        sp.items[1]
                    )
                num_tols[attr_k] = spec_map
        else:
            raise HCIRParseError(f"Unknown node pattern clause '{sh}'", line=sub.line, col=sub.col)
    tags: list[str] = []
    if "tags" in kws:
        tnode = kws["tags"]
        if not isinstance(tnode, SList):
            raise HCIRParseError(":tags expects a list", line=tnode.line, col=tnode.col)
        tags = [_decode_str(t) for t in tnode.items]
    return NodePattern(
        var=var_name,
        node_type=_decode_str(kws["type"]) if "type" in kws else None,
        id=_decode_str(kws["id"]) if "id" in kws else None,
        min_confidence=_decode_float(kws["min-confidence"]) if "min-confidence" in kws else None,
        tags=tags,
        attributes=attrs,
        fuzzy_memberships=fuzzy_mem,
        numeric_tolerances=num_tols,
    )


def _encode_edge_pattern(ep: EdgePattern) -> SList:
    if ep.edge_type is None:
        et_tok: SNode = SAtomKw("any", 0, 0)
    elif ep.edge_type != "any" and re.fullmatch(r"^[A-Za-z_][A-Za-z0-9_-]*$", ep.edge_type):
        et_tok = _encode_edge_type_kw(ep.edge_type)
    else:
        et_tok = SAtomStr(ep.edge_type, 0, 0)
    items: list[SNode] = [
        SAtomSym("edge", 0, 0),
        _encode_str_token(ep.source),
        et_tok,
        _encode_str_token(ep.target),
    ]
    if ep.min_weight is not None:
        items.extend((SAtomKw("min-weight", 0, 0), SAtomFloat(float(ep.min_weight), 0, 0)))
    if ep.min_semantic_degree is not None:
        items.extend(
            (SAtomKw("min-semantic-degree", 0, 0), SAtomFloat(float(ep.min_semantic_degree), 0, 0))
        )
    return SList(tuple(items), 0, 0)


def _decode_edge_pattern(node: SNode) -> EdgePattern:
    if not isinstance(node, SList) or _head_sym(node) != "edge":
        raise HCIRParseError("Expected (edge ...)", line=node.line, col=node.col)
    if len(node.items) < 4:
        raise HCIRParseError(
            "(edge ...) requires <source> <edge-type> <target>", line=node.line, col=node.col
        )
    src = _decode_str(node.items[1])
    et_node = node.items[2]
    if isinstance(et_node, SAtomKw):
        edge_type = None if et_node.name == "any" else et_node.name
    elif isinstance(et_node, SAtomNil):
        edge_type = None
    else:
        edge_type = _decode_str(et_node)
    tgt = _decode_str(node.items[3])
    kws, pos = _split_kw_and_pos(node.items[4:])
    if pos:
        raise HCIRParseError(
            "Unexpected positional argument in (edge ...)", line=pos[0].line, col=pos[0].col
        )
    return EdgePattern(
        source=src,
        edge_type=edge_type,
        target=tgt,
        min_weight=_decode_float(kws["min-weight"]) if "min-weight" in kws else None,
        min_semantic_degree=(
            _decode_float(kws["min-semantic-degree"]) if "min-semantic-degree" in kws else None
        ),
    )


def _encode_graph_pattern(gp: GraphPattern) -> SList:
    items: list[SNode] = [SAtomSym("pattern", 0, 0)]
    if not gp.distinct_nodes:
        items.extend((SAtomKw("distinct-nodes", 0, 0), SAtomBool(False, 0, 0)))
    if gp.nodes:
        items.append(
            SList((SAtomSym("nodes", 0, 0), *(_encode_node_pattern(n) for n in gp.nodes)), 0, 0)
        )
    if gp.edges:
        items.append(
            SList((SAtomSym("edges", 0, 0), *(_encode_edge_pattern(e) for e in gp.edges)), 0, 0)
        )
    if gp.any_of:
        items.append(
            SList(
                (SAtomSym("any-of", 0, 0), *(_encode_graph_pattern(sub) for sub in gp.any_of)),
                0,
                0,
            )
        )
    if gp.none_of:
        items.append(
            SList(
                (SAtomSym("none-of", 0, 0), *(_encode_graph_pattern(sub) for sub in gp.none_of)),
                0,
                0,
            )
        )
    return SList(tuple(items), 0, 0)


def _decode_graph_pattern(node: SNode) -> GraphPattern:
    if not isinstance(node, SList) or _head_sym(node) != "pattern":
        raise HCIRParseError("Expected (pattern ...)", line=node.line, col=node.col)
    kws, pos = _split_kw_and_pos(node.items[1:])
    distinct = _decode_bool(kws["distinct-nodes"]) if "distinct-nodes" in kws else True
    nodes: list[NodePattern] = []
    edges: list[EdgePattern] = []
    any_of: list[GraphPattern] = []
    none_of: list[GraphPattern] = []
    for sub in pos:
        if not isinstance(sub, SList):
            raise HCIRParseError("Expected clause inside (pattern ...)", line=sub.line, col=sub.col)
        sh = _head_sym(sub)
        if sh == "nodes":
            nodes.extend(_decode_node_pattern(c) for c in sub.items[1:])
        elif sh == "edges":
            edges.extend(_decode_edge_pattern(c) for c in sub.items[1:])
        elif sh == "any-of":
            any_of.extend(_decode_graph_pattern(c) for c in sub.items[1:])
        elif sh == "none-of":
            none_of.extend(_decode_graph_pattern(c) for c in sub.items[1:])
        else:
            raise HCIRParseError(f"Unknown pattern clause '{sh}'", line=sub.line, col=sub.col)
    return GraphPattern(
        nodes=nodes,
        edges=edges,
        any_of=any_of,
        none_of=none_of,
        distinct_nodes=distinct,
    )


def _encode_expr_guard(d: Mapping[str, Any], *, mode: Literal["canonical", "exact"]) -> SList:
    raw_se = d["symbolic_expr"]
    se = SymbolicExpr.model_validate(dict(raw_se))
    if mode == "exact" and se.model_dump() != dict(raw_se):
        return SList((SAtomSym("pattern-map", 0, 0), *_encode_map_pairs(d)), 0, 0)
    items: list[SNode] = [SAtomSym("expr-guard", 0, 0), _encode_symbolic_expr(se)]
    if "capabilities" in d:
        caps = d["capabilities"]
        if isinstance(caps, list):
            items.extend(
                (
                    SAtomKw("capabilities", 0, 0),
                    SList(tuple(_encode_value(c) for c in caps), 0, 0),
                )
            )
        else:
            items.extend((SAtomKw("capabilities-val", 0, 0), _encode_value(caps)))
    if "signature" in d:
        items.extend((SAtomKw("signature", 0, 0), _encode_value(d["signature"])))
    extra = {k: v for k, v in d.items() if k not in ("symbolic_expr", "capabilities", "signature")}
    if extra:
        items.extend((SAtomKw("extra", 0, 0), SList(_encode_map_pairs(extra), 0, 0)))
    return SList(tuple(items), 0, 0)


def _decode_expr_guard(node: SList) -> dict[str, Any]:
    kws, pos = _split_kw_and_pos(node.items[1:])
    if len(pos) != 1:
        raise HCIRParseError(
            "(expr-guard ...) requires 1 SymbolicExpr", line=node.line, col=node.col
        )
    out: dict[str, Any] = {"symbolic_expr": _decode_symbolic_expr(pos[0]).model_dump()}
    if "capabilities" in kws:
        cap_node = kws["capabilities"]
        if not isinstance(cap_node, SList):
            raise HCIRParseError(
                ":capabilities expects a list", line=cap_node.line, col=cap_node.col
            )
        out["capabilities"] = [_decode_value(c) for c in cap_node.items]
    elif "capabilities-val" in kws:
        out["capabilities"] = _decode_value(kws["capabilities-val"])
    if "signature" in kws:
        out["signature"] = _decode_value(kws["signature"])
    if "extra" in kws:
        ex = kws["extra"]
        if not isinstance(ex, SList):
            raise HCIRParseError(":extra expects pair list", line=ex.line, col=ex.col)
        out.update(_decode_kv_pairs(ex.items))
    return out


def _encode_map_pairs(mapping: Mapping[str, Any]) -> tuple[SList, ...]:
    return tuple(
        SList((SAtomStr(str(k), 0, 0), _encode_value(v)), 0, 0)
        for k, v in sorted(mapping.items(), key=lambda kv: str(kv[0]))
    )


def _is_clean_graph_pattern_dict(d: Mapping[str, Any]) -> bool:
    if not any(k in {"nodes", "edges", "any_of", "none_of", "distinct_nodes"} for k in d):
        return False
    try:
        gp = GraphPattern.model_validate(dict(d))
    except (ValueError, TypeError):
        return False
    return gp.model_dump() == dict(d)


def _encode_pattern_slot(val: Any, *, mode: Literal["canonical", "exact"]) -> SNode:
    if isinstance(val, GraphPattern):
        return _encode_graph_pattern(val)
    if isinstance(val, Mapping):
        if _is_clean_graph_pattern_dict(val):
            return _encode_graph_pattern(GraphPattern.model_validate(dict(val)))
        if mode == "canonical" and any(
            k in {"nodes", "edges", "any_of", "none_of", "distinct_nodes"} for k in val
        ):
            try:
                return _encode_graph_pattern(GraphPattern.model_validate(dict(val)))
            except (ValueError, TypeError):
                pass
        if (
            "symbolic_expr" in val
            and isinstance(val["symbolic_expr"], Mapping)
            and set(val.keys()) <= {"symbolic_expr", "capabilities", "signature"}
        ):
            try:
                return _encode_expr_guard(val, mode=mode)
            except (ValueError, TypeError):
                pass
        if not any(k in {"nodes", "edges", "any_of", "none_of", "distinct_nodes"} for k in val):
            return SList((SAtomSym("pattern-map", 0, 0), *_encode_map_pairs(val)), 0, 0)
        return SList((SAtomSym("raw-pattern", 0, 0), _encode_value(dict(val))), 0, 0)
    return SList((SAtomSym("raw-pattern", 0, 0), _encode_value(val)), 0, 0)


def _decode_pattern_slot(node: SNode) -> Any:
    if isinstance(node, SList) and node.items and isinstance(node.items[0], SAtomSym):
        h = node.items[0].name
        if h == "pattern":
            return _decode_graph_pattern(node).model_dump()
        if h == "pattern-map":
            return _decode_kv_pairs(node.items[1:])
        if h == "expr-guard":
            return _decode_expr_guard(node)
        if h == "raw-pattern":
            if len(node.items) != 2:
                raise HCIRParseError(
                    "(raw-pattern ...) requires 1 value", line=node.line, col=node.col
                )
            return _decode_value(node.items[1])
    return _decode_value(node)


# ═══════════════════════════════════════════════════════════════════════════
# Family 3 — FuzzyExpr & SymbolicExpr Codecs
# ═══════════════════════════════════════════════════════════════════════════


def _encode_fuzzy_expr(fe: FuzzyExpr) -> SList:
    op = fe.op
    a = fe.args
    ch = fe.children

    def _fallback() -> SList:
        items: list[SNode] = [SAtomSym("fuzzy-op", 0, 0), SAtomStr(op, 0, 0)]
        if a:
            items.extend((SAtomKw("args", 0, 0), SList(_encode_map_pairs(a), 0, 0)))
        if ch:
            items.extend(
                (SAtomKw("children", 0, 0), SList(tuple(_encode_fuzzy_expr(c) for c in ch), 0, 0))
            )
        return SList(tuple(items), 0, 0)

    if (
        op == "const"
        and not ch
        and set(a.keys()) == {"value"}
        and isinstance(a["value"], (int, float))
        and not isinstance(a["value"], bool)
    ):
        return SList((SAtomSym("const", 0, 0), SAtomFloat(float(a["value"]), 0, 0)), 0, 0)
    if (
        op == "var"
        and not ch
        and "name" in a
        and isinstance(a["name"], str)
        and set(a.keys()) <= {"name", "default"}
        and (
            "default" not in a
            or (isinstance(a["default"], (int, float)) and not isinstance(a["default"], bool))
        )
    ):
        items: list[SNode] = [SAtomSym("var", 0, 0), SAtomStr(a["name"], 0, 0)]
        if "default" in a:
            items.extend((SAtomKw("default", 0, 0), SAtomFloat(float(a["default"]), 0, 0)))
        return SList(tuple(items), 0, 0)
    if op == "membership" and not ch:
        items = [SAtomSym("membership", 0, 0)]
        slot_map = (
            ("model_id", "model-id", "str"),
            ("var", "var", "str"),
            ("value", "value", "val"),
            ("kernel_family", "kernel", "ident"),
            ("reference_scale_kind", "scale", "ident"),
            ("scale_context", "scale-context", "val"),
        )
        used: set[str] = set()
        for pk, kw_name, kind in slot_map:
            if pk in a:
                used.add(pk)
                val = a[pk]
                if kind == "str" and isinstance(val, str):
                    items.extend((SAtomKw(kw_name, 0, 0), SAtomStr(val, 0, 0)))
                elif kind == "ident" and isinstance(val, str):
                    items.extend((SAtomKw(kw_name, 0, 0), _encode_ident(val)))
                else:
                    items.extend((SAtomKw(kw_name, 0, 0), _encode_value(val)))
        if "parameters" in a and isinstance(a["parameters"], Mapping):
            used.add("parameters")
            items.extend((SAtomKw("params", 0, 0), SList(_encode_map_pairs(a["parameters"]), 0, 0)))
        extra = {k: v for k, v in a.items() if k not in used}
        if extra:
            items.extend((SAtomKw("extra", 0, 0), SList(_encode_map_pairs(extra), 0, 0)))
        return SList(tuple(items), 0, 0)
    if op in ("and", "or") and not a:
        return SList((SAtomSym(op, 0, 0), *(_encode_fuzzy_expr(c) for c in ch)), 0, 0)
    if op == "not" and not a and len(ch) <= 1:
        return SList((SAtomSym("not", 0, 0), *(_encode_fuzzy_expr(c) for c in ch)), 0, 0)
    if op in ("implies", "equiv") and not a and len(ch) == 2:
        return SList(
            (SAtomSym(op, 0, 0), _encode_fuzzy_expr(ch[0]), _encode_fuzzy_expr(ch[1])), 0, 0
        )
    if (
        op == "hedge"
        and len(ch) == 1
        and set(a.keys()) == {"hedge"}
        and isinstance(a["hedge"], str)
    ):
        return SList(
            (SAtomSym("hedge", 0, 0), _encode_ident(a["hedge"]), _encode_fuzzy_expr(ch[0])),
            0,
            0,
        )
    if (
        op == "alpha_cut"
        and len(ch) == 1
        and "alpha" in a
        and isinstance(a["alpha"], (int, float))
        and not isinstance(a["alpha"], bool)
        and set(a.keys()) <= {"alpha", "strict"}
        and ("strict" not in a or isinstance(a["strict"], bool))
    ):
        items = [SAtomSym("alpha-cut", 0, 0), SAtomFloat(float(a["alpha"]), 0, 0)]
        if "strict" in a:
            items.extend((SAtomKw("strict", 0, 0), SAtomBool(bool(a["strict"]), 0, 0)))
        items.append(_encode_fuzzy_expr(ch[0]))
        return SList(tuple(items), 0, 0)
    return _fallback()


def _decode_fuzzy_expr(node: SNode) -> FuzzyExpr:
    if not isinstance(node, SList):
        raise HCIRParseError("Expected FuzzyExpr list form", line=node.line, col=node.col)
    h = _head_sym(node)
    if h == "fuzzy-op":
        kws, pos = _split_kw_and_pos(node.items[1:])
        if len(pos) != 1:
            raise HCIRParseError("(fuzzy-op ...) requires op string", line=node.line, col=node.col)
        op_name = _decode_str(pos[0])
        args: dict[str, Any] = {}
        children: list[FuzzyExpr] = []
        if "args" in kws:
            ak = kws["args"]
            if not isinstance(ak, SList):
                raise HCIRParseError(":args expects pair list", line=ak.line, col=ak.col)
            args = _decode_kv_pairs(ak.items)
        if "children" in kws:
            ck = kws["children"]
            if not isinstance(ck, SList):
                raise HCIRParseError(":children expects list", line=ck.line, col=ck.col)
            children = [_decode_fuzzy_expr(c) for c in ck.items]
        return FuzzyExpr.model_validate(
            {"op": op_name, "args": args, "children": [c.model_dump() for c in children]}
        )
    if h == "const":
        if len(node.items) != 2:
            raise HCIRParseError("(const ...) requires 1 float", line=node.line, col=node.col)
        return FuzzyExpr(op="const", args={"value": _decode_float(node.items[1])})
    if h == "var":
        kws, pos = _split_kw_and_pos(node.items[1:])
        if len(pos) != 1:
            raise HCIRParseError("(var ...) requires 1 name", line=node.line, col=node.col)
        args = {"name": _decode_str(pos[0], allow_sym=True)}
        if "default" in kws:
            args["default"] = _decode_float(kws["default"])
        return FuzzyExpr(op="var", args=args)
    if h == "membership":
        kws, pos = _split_kw_and_pos(node.items[1:])
        if pos:
            raise HCIRParseError(
                "Unexpected positional arg in (membership ...)", line=pos[0].line, col=pos[0].col
            )
        args = {}
        rev = {
            "model-id": "model_id",
            "var": "var",
            "value": "value",
            "kernel": "kernel_family",
            "scale": "reference_scale_kind",
            "scale-context": "scale_context",
        }
        for kw_k, arg_k in rev.items():
            if kw_k in kws:
                args[arg_k] = _decode_value(kws[kw_k])
        if "params" in kws:
            pnode = kws["params"]
            if not isinstance(pnode, SList):
                raise HCIRParseError(":params expects pair list", line=pnode.line, col=pnode.col)
            args["parameters"] = _decode_kv_pairs(pnode.items)
        if "extra" in kws:
            ex = kws["extra"]
            if not isinstance(ex, SList):
                raise HCIRParseError(":extra expects pair list", line=ex.line, col=ex.col)
            args.update(_decode_kv_pairs(ex.items))
        return FuzzyExpr(op="membership", args=args)
    if h == "and":
        return FuzzyExpr(op="and", children=[_decode_fuzzy_expr(c) for c in node.items[1:]])
    if h == "or":
        return FuzzyExpr(op="or", children=[_decode_fuzzy_expr(c) for c in node.items[1:]])
    if h == "not":
        return FuzzyExpr(op="not", children=[_decode_fuzzy_expr(c) for c in node.items[1:]])
    if h in ("implies", "equiv"):
        if len(node.items) != 3:
            raise HCIRParseError(f"({h} ...) requires 2 children", line=node.line, col=node.col)
        bin_children = [_decode_fuzzy_expr(node.items[1]), _decode_fuzzy_expr(node.items[2])]
        if h == "implies":
            return FuzzyExpr(op="implies", children=bin_children)
        return FuzzyExpr(op="equiv", children=bin_children)
    if h == "hedge":
        if len(node.items) != 3:
            raise HCIRParseError("(hedge ...) requires <kind> <expr>", line=node.line, col=node.col)
        return FuzzyExpr(
            op="hedge",
            args={"hedge": _decode_str(node.items[1])},
            children=[_decode_fuzzy_expr(node.items[2])],
        )
    if h == "alpha-cut":
        kws, pos = _split_kw_and_pos(node.items[1:])
        if len(pos) != 2:
            raise HCIRParseError(
                "(alpha-cut ...) requires <alpha> <expr>", line=node.line, col=node.col
            )
        args = {"alpha": _decode_float(pos[0])}
        if "strict" in kws:
            args["strict"] = _decode_bool(kws["strict"])
        return FuzzyExpr(op="alpha_cut", args=args, children=[_decode_fuzzy_expr(pos[1])])
    raise HCIRParseError(f"Unknown FuzzyExpr form '{h}'", line=node.line, col=node.col)


_SEXPR_COMPOUND_HEADS = {"all_of": "all-of", "any_of": "any-of", "not": "not"}
_SEXPR_LEAF_HEADS = {
    "affordance": "affordance",
    "grid_dim": "grid-dim",
    "entity_bbox": "entity-bbox",
    "entity_count": "entity-count",
    "color_dist": "color-dist",
    "cut_set": "cut-set",
    "homogeneous_region": "homogeneous-region",
    "pixel_density": "pixel-density",
    "symmetry": "symmetry",
    "metadata": "metadata",
    "causal_do": "causal-do",
    "fuzzy_expr": "fuzzy-expr",
    "predictive_confidence": "predictive-confidence",
}
_REV_SEXPR_COMPOUND = {v: k for k, v in _SEXPR_COMPOUND_HEADS.items()}
_REV_SEXPR_LEAF = {v: k for k, v in _SEXPR_LEAF_HEADS.items()}

_SEXPR_ARG_TO_KW: dict[str, str] = {
    "exact": "exact",
    "required": "required",
    "forbidden": "forbidden",
    "exact_shape": "exact-shape",
    "min_height": "min-height",
    "max_height": "max-height",
    "min_width": "min-width",
    "max_width": "max-width",
    "min_distinct_colors": "min-distinct-colors",
    "max_distinct_colors": "max-distinct-colors",
    "present_colors": "present-colors",
    "absent_colors": "absent-colors",
    "min_partitions": "min-partitions",
    "key": "key",
    "expected_value": "expected-value",
    "variable": "variable",
    "operator": "operator",
    "profile_id": "profile-id",
    "alpha": "alpha",
    "max_free_energy": "max-free-energy",
    "min_confidence": "min-confidence",
}
_KW_TO_SEXPR_ARG = {v: k for k, v in _SEXPR_ARG_TO_KW.items()}


def _encode_symbolic_expr(se: SymbolicExpr) -> SList:
    if se.op in _SEXPR_COMPOUND_HEADS and not se.args:
        items: list[SNode] = [SAtomSym(_SEXPR_COMPOUND_HEADS[se.op], 0, 0)]
        if se.alu_syscall is not None:
            items.extend((SAtomKw("alu-syscall", 0, 0), SAtomStr(se.alu_syscall, 0, 0)))
        items.extend(_encode_symbolic_expr(c) for c in se.children)
        return SList(tuple(items), 0, 0)

    if se.op in _SEXPR_LEAF_HEADS and not se.children:
        head = _SEXPR_LEAF_HEADS[se.op]
        items = [SAtomSym(head, 0, 0)]
        used: set[str] = set()
        if se.op == "fuzzy_expr":
            if "expr" in se.args and isinstance(se.args["expr"], Mapping):
                try:
                    fe = FuzzyExpr.model_validate(dict(se.args["expr"]))
                except (ValueError, TypeError):
                    fe = None
                if fe is not None and fe.model_dump() == dict(se.args["expr"]):
                    used.add("expr")
                    items.append(_encode_fuzzy_expr(fe))
                else:
                    return _encode_sexpr_fallback(se)
            else:
                return _encode_sexpr_fallback(se)
        if se.alu_syscall is not None:
            items.extend((SAtomKw("alu-syscall", 0, 0), SAtomStr(se.alu_syscall, 0, 0)))
        for k in sorted(se.args.keys()):
            if k in used:
                continue
            val = se.args[k]
            if k in _SEXPR_ARG_TO_KW:
                items.extend((SAtomKw(_SEXPR_ARG_TO_KW[k], 0, 0), _encode_value(val)))
                used.add(k)
        extra = {k: v for k, v in se.args.items() if k not in used}
        if extra:
            items.extend((SAtomKw("extra", 0, 0), SList(_encode_map_pairs(extra), 0, 0)))
        return SList(tuple(items), 0, 0)

    return _encode_sexpr_fallback(se)


def _encode_sexpr_fallback(se: SymbolicExpr) -> SList:
    items: list[SNode] = [
        SAtomSym("sexpr", 0, 0),
        SAtomKw("op", 0, 0),
        SAtomStr(se.op, 0, 0),
    ]
    if se.alu_syscall is not None:
        items.extend((SAtomKw("alu-syscall", 0, 0), SAtomStr(se.alu_syscall, 0, 0)))
    if se.args:
        items.extend((SAtomKw("args", 0, 0), SList(_encode_map_pairs(se.args), 0, 0)))
    if se.children:
        items.extend(
            (
                SAtomKw("children", 0, 0),
                SList(tuple(_encode_symbolic_expr(c) for c in se.children), 0, 0),
            )
        )
    return SList(tuple(items), 0, 0)


def _decode_symbolic_expr(node: SNode) -> SymbolicExpr:
    if not isinstance(node, SList):
        raise HCIRParseError("Expected SymbolicExpr list form", line=node.line, col=node.col)
    h = _head_sym(node)
    if h == "sexpr":
        kws, pos = _split_kw_and_pos(node.items[1:])
        if pos or "op" not in kws:
            raise HCIRParseError("(sexpr ...) requires :op", line=node.line, col=node.col)
        args = {}
        if "args" in kws:
            ak = kws["args"]
            if not isinstance(ak, SList):
                raise HCIRParseError(":args expects pair list", line=ak.line, col=ak.col)
            args = _decode_kv_pairs(ak.items)
        children = []
        if "children" in kws:
            ck = kws["children"]
            if not isinstance(ck, SList):
                raise HCIRParseError(":children expects list", line=ck.line, col=ck.col)
            children = [_decode_symbolic_expr(c) for c in ck.items]
        return SymbolicExpr(
            op=_decode_str(kws["op"]),
            args=args,
            children=children,
            alu_syscall=_decode_str(kws["alu-syscall"]) if "alu-syscall" in kws else None,
        )
    if h in _REV_SEXPR_COMPOUND:
        kws, pos = _split_kw_and_pos(node.items[1:])
        return SymbolicExpr(
            op=_REV_SEXPR_COMPOUND[h],
            children=[_decode_symbolic_expr(c) for c in pos],
            alu_syscall=_decode_str(kws["alu-syscall"]) if "alu-syscall" in kws else None,
        )
    if h in _REV_SEXPR_LEAF:
        op_name = _REV_SEXPR_LEAF[h]
        kws, pos = _split_kw_and_pos(node.items[1:])
        leaf_args: dict[str, Any] = {}
        if op_name == "fuzzy_expr":
            if len(pos) != 1:
                raise HCIRParseError(
                    "(fuzzy-expr ...) requires 1 FuzzyExpr", line=node.line, col=node.col
                )
            leaf_args["expr"] = _decode_fuzzy_expr(pos[0]).model_dump()
        elif pos:
            raise HCIRParseError(
                f"Unexpected positional arg in ({h} ...)", line=pos[0].line, col=pos[0].col
            )
        alu_sys = _decode_str(kws["alu-syscall"]) if "alu-syscall" in kws else None
        for kw_k, kw_v in kws.items():
            if kw_k in ("alu-syscall", "extra"):
                continue
            if kw_k in _KW_TO_SEXPR_ARG:
                leaf_args[_KW_TO_SEXPR_ARG[kw_k]] = _decode_value(kw_v)
            else:
                raise HCIRParseError(
                    f"Unknown keyword :{kw_k} in ({h} ...)", line=kw_v.line, col=kw_v.col
                )
        if "extra" in kws:
            ex = kws["extra"]
            if not isinstance(ex, SList):
                raise HCIRParseError(":extra expects pair list", line=ex.line, col=ex.col)
            leaf_args.update(_decode_kv_pairs(ex.items))
        return SymbolicExpr(op=op_name, args=leaf_args, alu_syscall=alu_sys)
    raise HCIRParseError(f"Unknown SymbolicExpr head '{h}'", line=node.line, col=node.col)


# ═══════════════════════════════════════════════════════════════════════════
# Family 1 — Instruction & InstructionStream Codec
# ═══════════════════════════════════════════════════════════════════════════

_BRANCH_OP_TO_SYM: dict[str, str] = {
    "if_empty": "empty?",
    "if_not_empty": "not-empty?",
    "if_true": "true?",
    "if_false": "false?",
    "if_equals": "eq?",
    "if_not_equals": "neq?",
    "if_lt": "lt?",
    "if_lte": "lte?",
    "if_gt": "gt?",
    "if_gte": "gte?",
    "if_confidence_lt": "confidence-lt?",
    "if_surprise_gt": "surprise-gt?",
    "if_degree_gte": "degree-gte?",
    "if_degree_lt": "degree-lt?",
}
_SYM_TO_BRANCH_OP = {v: k for k, v in _BRANCH_OP_TO_SYM.items()}
_BRANCH_EQ_OPS = frozenset({"if_equals", "if_not_equals"})
_BRANCH_NUM_OPS = frozenset(
    {
        "if_lt",
        "if_lte",
        "if_gt",
        "if_gte",
        "if_confidence_lt",
        "if_surprise_gt",
        "if_degree_gte",
        "if_degree_lt",
    }
)


def _encode_bind_spec(val: Any) -> SNode:
    if isinstance(val, str):
        return _encode_str_token(val)
    return _encode_value(val)


def _decode_bind_spec(node: SNode) -> Any:
    return _decode_value(node)


def _can_encode_clean_probe(params: Mapping[str, Any]) -> bool:
    has_reg = "register" in params
    has_var = "var" in params
    if has_reg and has_var:
        return False
    if has_reg:
        return isinstance(params["register"], str) and params["register"] in REGISTER_NAMES
    if has_var:
        return isinstance(params["var"], str) and bool(_VAR_RE.fullmatch(params["var"]))
    return True


def _encode_branch_when(params: Mapping[str, Any]) -> tuple[SNode | None, set[str]]:
    if "condition" not in params:
        return None, set()
    cond = params["condition"]
    if cond == "unconditional":
        return SAtomSym("always", 0, 0), {"condition"}
    if not isinstance(cond, str) or cond not in _BRANCH_OP_TO_SYM:
        return None, set()
    if not _can_encode_clean_probe(params):
        return None, set()

    sym = _BRANCH_OP_TO_SYM[cond]
    used: set[str] = {"condition"}
    items: list[SNode] = [SAtomSym(sym, 0, 0)]
    if "register" in params:
        items.append(SAtomReg(params["register"], 0, 0))
        used.add("register")
    elif "var" in params:
        items.append(SAtomVar(params["var"], 0, 0))
        used.add("var")

    if cond in _BRANCH_EQ_OPS:
        if "value" not in params:
            return None, set()
        items.append(_encode_value(params["value"]))
        used.add("value")
        return SList(tuple(items), 0, 0), used
    if cond in _BRANCH_NUM_OPS:
        if "threshold" in params and "value" not in params:
            num_key = "threshold"
        elif "value" in params and "threshold" not in params:
            num_key = "value"
        else:
            return None, set()
        num_val = params[num_key]
        if not isinstance(num_val, (int, float)) or isinstance(num_val, bool):
            return None, set()
        items.append(_encode_value(num_val))
        used.add(num_key)
        if num_key == "value":
            items.extend((SAtomKw("key", 0, 0), SAtomSym("value", 0, 0)))
        return SList(tuple(items), 0, 0), used
    return SList(tuple(items), 0, 0), used


def _decode_branch_when(node: SNode) -> dict[str, Any]:
    if isinstance(node, SAtomSym) and node.name == "always":
        return {"condition": "unconditional"}
    if not isinstance(node, SList):
        raise HCIRParseError("Invalid :when clause in (branch ...)", line=node.line, col=node.col)
    sym = _head_sym(node)
    if sym not in _SYM_TO_BRANCH_OP:
        raise HCIRParseError(f"Unknown branch condition '{sym}'", line=node.line, col=node.col)
    cond = _SYM_TO_BRANCH_OP[sym]
    kws, pos = _split_kw_and_pos(node.items[1:])
    out: dict[str, Any] = {"condition": cond}
    if cond in _BRANCH_EQ_OPS:
        if len(pos) == 2:
            _assign_probe(out, pos[0])
            out["value"] = _decode_value(pos[1])
        elif len(pos) == 1:
            out["value"] = _decode_value(pos[0])
        else:
            raise HCIRParseError(
                f"({sym} ...) expects [<probe>] <value>", line=node.line, col=node.col
            )
        return out
    if cond in _BRANCH_NUM_OPS:
        if len(pos) == 2:
            _assign_probe(out, pos[0])
            val = _decode_value(pos[1])
        elif len(pos) == 1:
            val = _decode_value(pos[0])
        else:
            raise HCIRParseError(
                f"({sym} ...) expects [<probe>] <number>", line=node.line, col=node.col
            )
        num_key = _decode_str(kws["key"]) if "key" in kws else "threshold"
        out[num_key] = val
        return out
    if len(pos) == 1:
        _assign_probe(out, pos[0])
    elif len(pos) > 1:
        raise HCIRParseError(f"({sym} ...) takes at most 1 probe", line=node.line, col=node.col)
    return out


def _assign_probe(out: dict[str, Any], probe: SNode) -> None:
    if isinstance(probe, SAtomReg):
        out["register"] = probe.slot
    elif isinstance(probe, SAtomVar):
        out["var"] = probe.name
    else:
        raise HCIRParseError(
            "Branch probe must be $r0..$r7 or ?var", line=probe.line, col=probe.col
        )


_SIMPLE_OPCODE_SLOTS: dict[Opcode, tuple[tuple[str, str], ...]] = {
    Opcode.ASSERT: (
        ("node_data", "node"),
        ("edge_data", "edge"),
        ("author", "author"),
        ("branch_name", "branch"),
        ("bind", "bind"),
    ),
    Opcode.RETRACT: (
        ("node_id", "node-id"),
        ("edge_id", "edge-id"),
        ("author", "author"),
        ("branch_name", "branch"),
        ("bind", "bind"),
    ),
    Opcode.QUERY: (
        ("node_type", "node-type"),
        ("category", "category"),
        ("lifecycle", "lifecycle"),
        ("scope_tenant", "scope-tenant"),
        ("text_contains", "text-contains"),
        ("min_confidence", "min-confidence"),
        ("limit", "limit"),
        ("branch_name", "branch"),
        ("bind", "bind"),
    ),
    Opcode.WAIT: (
        ("event_type", "event-type"),
        ("event", "event"),
        ("condition", "condition"),
        ("timeout_ms", "timeout-ms"),
        ("bind", "bind"),
    ),
    Opcode.FORK: (("branch_name", "branch"), ("bind", "bind")),
    Opcode.MERGE: (("branch_name", "branch"), ("author", "author"), ("bind", "bind")),
    Opcode.ROLLBACK: (
        ("branch_name", "branch"),
        ("target_version", "target-version"),
        ("reason", "reason"),
        ("bind", "bind"),
    ),
}
_REV_SIMPLE_OPCODE_SLOTS: dict[Opcode, dict[str, str]] = {
    op: {kw: pk for pk, kw in pairs} for op, pairs in _SIMPLE_OPCODE_SLOTS.items()
}


def _encode_instruction(ins: Instruction, *, mode: Literal["canonical", "exact"]) -> SList:
    op = ins.opcode
    p = ins.params
    used: set[str] = set()
    items: list[SNode] = [SAtomSym(op.value.lower(), 0, 0)]

    if op == Opcode.EXECUTE:
        if "capability" in p and isinstance(p["capability"], str):
            items.append(_encode_ident(p["capability"]))
            used.add("capability")
        else:
            items.append(SAtomKw("no-capability", 0, 0))
        if "params" in p and isinstance(p["params"], Mapping):
            used.add("params")
            items.extend((SAtomKw("params", 0, 0), SList(_encode_map_pairs(p["params"]), 0, 0)))
        if "branch_name" in p:
            used.add("branch_name")
            items.extend((SAtomKw("branch", 0, 0), _encode_value(p["branch_name"])))
        if "bind" in p:
            used.add("bind")
            items.extend((SAtomKw("bind", 0, 0), _encode_bind_spec(p["bind"])))
    elif op == Opcode.UNIFY:
        if "pattern" in p:
            used.add("pattern")
            items.append(_encode_pattern_slot(p["pattern"], mode=mode))
        else:
            items.append(SAtomKw("no-pattern", 0, 0))
        for pk, kw in (
            ("source", "source"),
            ("fuzzy", "fuzzy"),
            ("profile_id", "profile-id"),
            ("min_degree", "min-degree"),
            ("limit", "limit"),
            ("branch_name", "branch"),
            ("bind", "bind"),
            ("bind_all", "bind-all"),
        ):
            if pk in p:
                used.add(pk)
                val_node = (
                    _encode_bind_spec(p[pk]) if pk in ("bind", "bind_all") else _encode_value(p[pk])
                )
                items.extend((SAtomKw(kw, 0, 0), val_node))
    elif op == Opcode.BRANCH:
        when_node, when_used = _encode_branch_when(p)
        if when_node is not None:
            used.update(when_used)
            items.extend((SAtomKw("when", 0, 0), when_node))
        if "target_label" in p:
            used.add("target_label")
            items.extend((SAtomKw("goto", 0, 0), _encode_value(p["target_label"])))
        if "target_pc" in p:
            used.add("target_pc")
            items.extend((SAtomKw("goto-pc", 0, 0), _encode_value(p["target_pc"])))
    elif op == Opcode.CALL:
        has_id = "skill_id" in p
        has_name = "skill_name" in p
        has_goal = "goal_pattern" in p
        if has_id and not has_name and not has_goal:
            used.add("skill_id")
            items.append(SList((SAtomSym("id", 0, 0), _encode_value(p["skill_id"])), 0, 0))
        elif has_name and not has_id and not has_goal:
            used.add("skill_name")
            items.append(SList((SAtomSym("name", 0, 0), _encode_value(p["skill_name"])), 0, 0))
        elif has_goal and not has_id and not has_name:
            used.add("goal_pattern")
            items.append(
                SList(
                    (SAtomSym("goal", 0, 0), _encode_pattern_slot(p["goal_pattern"], mode=mode)),
                    0,
                    0,
                )
            )
        else:
            items.append(SList((SAtomSym("none", 0, 0),), 0, 0))
        for pk, kw in (
            ("args", "args"),
            ("fuzzy", "fuzzy"),
            ("profile_id", "profile-id"),
            ("min_degree", "min-degree"),
            ("eval_context", "eval-context"),
            ("branch_name", "branch"),
            ("bind", "bind"),
        ):
            if pk in p:
                used.add(pk)
                val_node = _encode_bind_spec(p[pk]) if pk == "bind" else _encode_value(p[pk])
                items.extend((SAtomKw(kw, 0, 0), val_node))
    elif op == Opcode.RETURN:
        if set(p.keys()) == {"yield"} and isinstance(p["yield"], list):
            used.add("yield")
            items.extend(_encode_value(v) for v in p["yield"])
        elif not p:
            items.append(SAtomKw("empty", 0, 0))
        elif set(p.keys()) == {"value"}:
            used.add("value")
            items.extend((SAtomKw("value", 0, 0), _encode_value(p["value"])))
    else:
        for pk, kw in _SIMPLE_OPCODE_SLOTS[op]:
            if pk in p:
                used.add(pk)
                val_node = _encode_bind_spec(p[pk]) if pk == "bind" else _encode_value(p[pk])
                items.extend((SAtomKw(kw, 0, 0), val_node))

    if mode == "exact":
        items.extend((SAtomKw("id", 0, 0), SAtomStr(ins.id, 0, 0)))
    if ins.cost_estimate != 0:
        items.extend((SAtomKw("cost", 0, 0), SAtomInt(ins.cost_estimate, 0, 0)))

    extra = {k: v for k, v in p.items() if k not in used}
    if extra:
        items.extend((SAtomKw("extra", 0, 0), SList(_encode_map_pairs(extra), 0, 0)))

    core = SList(tuple(items), 0, 0)
    if ins.label is not None:
        return SList((SAtomSym("label", 0, 0), _encode_ident(ins.label), core), 0, 0)
    return core


def _decode_instruction(node: SNode) -> Instruction:
    if not isinstance(node, SList):
        raise HCIRParseError("Expected instruction form", line=node.line, col=node.col)
    label: str | None = None
    cur = node
    if _head_sym(cur) == "label":
        if len(cur.items) != 3:
            raise HCIRParseError(
                "(label ...) requires <label> <instruction>", line=cur.line, col=cur.col
            )
        label = _decode_str(cur.items[1], allow_sym=True)
        inner = cur.items[2]
        if not isinstance(inner, SList):
            raise HCIRParseError(
                "Expected instruction inside (label ...)", line=inner.line, col=inner.col
            )
        cur = inner

    head = _head_sym(cur).upper()
    try:
        op = Opcode(head)
    except ValueError as exc:
        raise HCIRParseError(f"Unknown opcode '{head}'", line=cur.line, col=cur.col) from exc

    # Special handling for boolean-flag keywords (:no-capability, :no-pattern, :empty)
    raw_items = list(cur.items[1:])
    flag_kws: set[str] = set()
    cleaned_items: list[SNode] = []
    idx = 0
    while idx < len(raw_items):
        tok = raw_items[idx]
        if isinstance(tok, SAtomKw) and tok.name in ("no-capability", "no-pattern", "empty"):
            flag_kws.add(tok.name)
            idx += 1
        else:
            cleaned_items.append(tok)
            idx += 1

    kws, pos = _split_kw_and_pos(cleaned_items)
    params: dict[str, Any] = {}
    ins_id: str | None = _decode_str(kws["id"]) if "id" in kws else None
    cost = _decode_int(kws["cost"]) if "cost" in kws else 0

    if op == Opcode.EXECUTE:
        if "no-capability" not in flag_kws:
            if not pos:
                raise HCIRParseError(
                    "(execute ...) requires <capability>", line=cur.line, col=cur.col
                )
            params["capability"] = _decode_str(pos[0])
            pos = pos[1:]
        if pos:
            raise HCIRParseError(
                "Unexpected positional argument in (execute ...)", line=pos[0].line, col=pos[0].col
            )
        if "params" in kws:
            pnode = kws["params"]
            if not isinstance(pnode, SList):
                raise HCIRParseError(":params expects pair list", line=pnode.line, col=pnode.col)
            params["params"] = _decode_kv_pairs(pnode.items)
        if "branch" in kws:
            params["branch_name"] = _decode_value(kws["branch"])
        if "bind" in kws:
            params["bind"] = _decode_bind_spec(kws["bind"])
    elif op == Opcode.UNIFY:
        if "no-pattern" not in flag_kws:
            if not pos:
                raise HCIRParseError("(unify ...) requires <pattern>", line=cur.line, col=cur.col)
            params["pattern"] = _decode_pattern_slot(pos[0])
            pos = pos[1:]
        if pos:
            raise HCIRParseError(
                "Unexpected positional arg in (unify ...)", line=pos[0].line, col=pos[0].col
            )
        for kw, pk in (
            ("source", "source"),
            ("fuzzy", "fuzzy"),
            ("profile-id", "profile_id"),
            ("min-degree", "min_degree"),
            ("limit", "limit"),
            ("branch", "branch_name"),
            ("bind", "bind"),
            ("bind-all", "bind_all"),
        ):
            if kw in kws:
                params[pk] = _decode_value(kws[kw])
    elif op == Opcode.BRANCH:
        if pos:
            raise HCIRParseError(
                "Unexpected positional arg in (branch ...)", line=pos[0].line, col=pos[0].col
            )
        if "when" in kws:
            params.update(_decode_branch_when(kws["when"]))
        if "goto" in kws:
            params["target_label"] = _decode_value(kws["goto"])
        if "goto-pc" in kws:
            params["target_pc"] = _decode_value(kws["goto-pc"])
    elif op == Opcode.CALL:
        if not pos or not isinstance(pos[0], SList):
            raise HCIRParseError(
                "(call ...) requires (id ...)/(name ...)/(goal ...)/(none)",
                line=cur.line,
                col=cur.col,
            )
        spec = pos[0]
        if len(pos) > 1:
            raise HCIRParseError(
                "Unexpected positional arg in (call ...)", line=pos[1].line, col=pos[1].col
            )
        sh = _head_sym(spec)
        if sh == "id":
            if len(spec.items) != 2:
                raise HCIRParseError("(id ...) requires 1 arg", line=spec.line, col=spec.col)
            params["skill_id"] = _decode_value(spec.items[1])
        elif sh == "name":
            if len(spec.items) != 2:
                raise HCIRParseError("(name ...) requires 1 arg", line=spec.line, col=spec.col)
            params["skill_name"] = _decode_value(spec.items[1])
        elif sh == "goal":
            if len(spec.items) != 2:
                raise HCIRParseError("(goal ...) requires 1 pattern", line=spec.line, col=spec.col)
            params["goal_pattern"] = _decode_pattern_slot(spec.items[1])
        elif sh != "none":
            raise HCIRParseError(f"Unknown (call ...) target '{sh}'", line=spec.line, col=spec.col)
        for kw, pk in (
            ("args", "args"),
            ("fuzzy", "fuzzy"),
            ("profile-id", "profile_id"),
            ("min-degree", "min_degree"),
            ("eval-context", "eval_context"),
            ("branch", "branch_name"),
            ("bind", "bind"),
        ):
            if kw in kws:
                params[pk] = _decode_value(kws[kw])
    elif op == Opcode.RETURN:
        if "empty" in flag_kws:
            if pos or "value" in kws:
                raise HCIRParseError(
                    "(return :empty) cannot take positional or :value args",
                    line=cur.line,
                    col=cur.col,
                )
        elif "value" in kws:
            if pos:
                raise HCIRParseError(
                    "(return :value ...) cannot take positional args",
                    line=pos[0].line,
                    col=pos[0].col,
                )
            params["value"] = _decode_value(kws["value"])
        elif "extra" not in kws or pos:
            params["yield"] = [_decode_value(v) for v in pos]
    else:
        if pos:
            raise HCIRParseError(
                f"Unexpected positional arg in ({op.value.lower()} ...)",
                line=pos[0].line,
                col=pos[0].col,
            )
        rev = _REV_SIMPLE_OPCODE_SLOTS[op]
        for kw, pk in rev.items():
            if kw in kws:
                params[pk] = _decode_value(kws[kw])

    if "extra" in kws:
        ex = kws["extra"]
        if not isinstance(ex, SList):
            raise HCIRParseError(":extra expects pair list", line=ex.line, col=ex.col)
        params.update(_decode_kv_pairs(ex.items))

    kw_args: dict[str, Any] = {
        "opcode": op,
        "params": params,
        "label": label,
        "cost_estimate": cost,
    }
    if ins_id is not None:
        kw_args["id"] = ins_id
    return Instruction(**kw_args)


def _encode_stream_body(
    stream: InstructionStream, *, head: str = "stream", mode: Literal["canonical", "exact"]
) -> SList:
    items: list[SNode] = [SAtomSym(head, 0, 0)]
    if mode == "exact":
        items.extend((SAtomKw("id", 0, 0), SAtomStr(stream.id, 0, 0)))
    if stream.author:
        items.extend((SAtomKw("author", 0, 0), SAtomStr(stream.author, 0, 0)))
    if stream.description:
        items.extend((SAtomKw("description", 0, 0), SAtomStr(stream.description, 0, 0)))
    items.extend(_encode_instruction(ins, mode=mode) for ins in stream.instructions)
    return SList(tuple(items), 0, 0)


def _decode_stream(node: SList) -> InstructionStream:
    kws, pos = _split_kw_and_pos(node.items[1:])
    kw_args: dict[str, Any] = {
        "instructions": [_decode_instruction(c) for c in pos],
        "author": _decode_str(kws["author"]) if "author" in kws else "",
        "description": _decode_str(kws["description"]) if "description" in kws else "",
    }
    if "id" in kws:
        kw_args["id"] = _decode_str(kws["id"])
    return InstructionStream(**kw_args)


# ═══════════════════════════════════════════════════════════════════════════
# Family 4 — Cognitive Nodes, Policies, Edges & Seed Bundles
# ═══════════════════════════════════════════════════════════════════════════

_META_BASE_KEYS = frozenset(
    {
        "id",
        "node_type",
        "category",
        "lifecycle",
        "provenance",
        "uncertainty",
        "attention",
        "scope",
        "tags",
        "properties",
        "confidence",
    }
)


def _encode_meta(
    node: HCIRNode,
    *,
    mode: Literal["canonical", "exact"],
    extra_fields: Mapping[str, Any] | None = None,
) -> SList | None:
    if mode == "exact":
        dumped = node.model_dump(mode="json")
        clauses: list[SNode] = [
            SAtomSym("meta", 0, 0),
            SList((SAtomSym("id", 0, 0), SAtomStr(str(dumped["id"]), 0, 0)), 0, 0),
            SList((SAtomSym("category", 0, 0), _encode_ident(str(dumped["category"]))), 0, 0),
            SList((SAtomSym("lifecycle", 0, 0), _encode_ident(str(dumped["lifecycle"]))), 0, 0),
            SList((SAtomSym("provenance", 0, 0), _encode_value(dumped["provenance"])), 0, 0),
            SList((SAtomSym("uncertainty", 0, 0), _encode_value(dumped["uncertainty"])), 0, 0),
            SList((SAtomSym("attention", 0, 0), _encode_value(dumped["attention"])), 0, 0),
            SList((SAtomSym("scope", 0, 0), _encode_value(dumped["scope"])), 0, 0),
        ]
        if dumped.get("tags"):
            clauses.append(
                SList((SAtomSym("tags", 0, 0), *(SAtomStr(t, 0, 0) for t in dumped["tags"])), 0, 0)
            )
        if dumped.get("properties"):
            clauses.append(
                SList((SAtomSym("properties", 0, 0), _encode_value(dumped["properties"])), 0, 0)
            )
        if dumped.get("confidence") is not None:
            clauses.append(
                SList(
                    (SAtomSym("confidence", 0, 0), SAtomFloat(float(dumped["confidence"]), 0, 0)),
                    0,
                    0,
                )
            )
        if extra_fields:
            clauses.append(
                SList((SAtomSym("extra-fields", 0, 0), _encode_value(dict(extra_fields))), 0, 0)
            )
        return SList(tuple(clauses), 0, 0)

    clauses = [SAtomSym("meta", 0, 0)]
    if node.tags:
        clauses.append(
            SList((SAtomSym("tags", 0, 0), *(SAtomStr(t, 0, 0) for t in node.tags)), 0, 0)
        )
    if node.properties:
        clauses.append(SList((SAtomSym("properties", 0, 0), _encode_value(node.properties)), 0, 0))
    if node.confidence is not None and not isinstance(node, (LearnedRuleNode, ConceptNode)):
        clauses.append(
            SList((SAtomSym("confidence", 0, 0), SAtomFloat(float(node.confidence), 0, 0)), 0, 0)
        )
    return SList(tuple(clauses), 0, 0) if len(clauses) > 1 else None


def _decode_meta(node: SList) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in node.items[1:]:
        if not isinstance(item, SList) or not item.items:
            raise HCIRParseError("Invalid clause in (meta ...)", line=item.line, col=item.col)
        h = _head_sym(item)
        if h in ("id", "category", "lifecycle"):
            out[h] = _decode_str(item.items[1])
        elif h in ("provenance", "uncertainty", "attention", "scope", "properties"):
            out[h] = _decode_value(item.items[1])
        elif h == "tags":
            out["tags"] = [_decode_str(t) for t in item.items[1:]]
        elif h == "confidence":
            out["confidence"] = _decode_float(item.items[1])
        elif h == "extra-fields":
            ef = _decode_value(item.items[1])
            if isinstance(ef, Mapping):
                out.update(ef)
        else:
            raise HCIRParseError(f"Unknown (meta ...) clause '{h}'", line=item.line, col=item.col)
    return out


def _encode_params_clause(parameters: Sequence[Mapping[str, Any]]) -> SList:
    p_nodes: list[SNode] = []
    for p in parameters:
        pname = str(p.get("name", ""))
        p_items: list[SNode] = [SAtomSym("param", 0, 0), SAtomStr(pname, 0, 0)]
        used = {"name"}
        for pk, kw in (
            ("required", "required"),
            ("default", "default"),
            ("node_type", "node-type"),
            ("role", "role"),
        ):
            if pk in p:
                used.add(pk)
                p_items.extend((SAtomKw(kw, 0, 0), _encode_value(p[pk])))
        extra = {k: v for k, v in p.items() if k not in used}
        if extra:
            p_items.extend((SAtomKw("extra", 0, 0), SList(_encode_map_pairs(extra), 0, 0)))
        p_nodes.append(SList(tuple(p_items), 0, 0))
    return SList((SAtomSym("params", 0, 0), *p_nodes), 0, 0)


def _decode_params_clause(node: SList) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in node.items[1:]:
        if not isinstance(item, SList) or _head_sym(item) != "param":
            raise HCIRParseError(
                "Expected (param ...) in (params ...)", line=item.line, col=item.col
            )
        kws, pos = _split_kw_and_pos(item.items[1:])
        if len(pos) != 1:
            raise HCIRParseError("(param ...) requires <name>", line=item.line, col=item.col)
        entry: dict[str, Any] = {"name": _decode_str(pos[0])}
        for kw, pk in (
            ("required", "required"),
            ("default", "default"),
            ("node-type", "node_type"),
            ("role", "role"),
        ):
            if kw in kws:
                entry[pk] = _decode_value(kws[kw])
        if "extra" in kws:
            ex = kws["extra"]
            if not isinstance(ex, SList):
                raise HCIRParseError(":extra expects pair list", line=ex.line, col=ex.col)
            entry.update(_decode_kv_pairs(ex.items))
        out.append(entry)
    return out


def _encode_homoiconic_clauses(node: Any, *, mode: Literal["canonical", "exact"]) -> list[SNode]:
    clauses: list[SNode] = []
    if node.parameters:
        clauses.append(_encode_params_clause(node.parameters))
    if node.precondition_pattern is not None:
        clauses.append(
            SList(
                (
                    SAtomSym("precondition", 0, 0),
                    _encode_pattern_slot(node.precondition_pattern, mode=mode),
                ),
                0,
                0,
            )
        )
    if node.postcondition_pattern is not None:
        clauses.append(
            SList(
                (
                    SAtomSym("postcondition", 0, 0),
                    _encode_pattern_slot(node.postcondition_pattern, mode=mode),
                ),
                0,
                0,
            )
        )
    if mode == "exact" or node.alpha_success != 1.0 or node.beta_failure != 1.0:
        rel_items: list[SNode] = [
            SAtomSym("reliability", 0, 0),
            SAtomKw("alpha", 0, 0),
            SAtomFloat(float(node.alpha_success), 0, 0),
            SAtomKw("beta", 0, 0),
            SAtomFloat(float(node.beta_failure), 0, 0),
        ]
        if isinstance(node, SkillNode) and mode == "exact":
            rel_items.extend(
                (
                    SAtomKw("success-rate", 0, 0),
                    SAtomFloat(float(node.success_rate), 0, 0),
                    SAtomKw("invocations", 0, 0),
                    SAtomInt(int(node.invocation_count), 0, 0),
                )
            )
        clauses.append(SList(tuple(rel_items), 0, 0))
    if node.bytecode_stream is not None:
        raw_stream = node.bytecode_stream
        stream = InstructionStream.model_validate(raw_stream)
        if mode == "exact" and stream.model_dump() != dict(raw_stream):
            clauses.append(
                SList((SAtomSym("body-raw", 0, 0), _encode_value(dict(raw_stream))), 0, 0)
            )
        else:
            clauses.append(_encode_stream_body(stream, head="body", mode=mode))
    return clauses


def _apply_homoiconic_clause(sh: str, item: SList, fields: dict[str, Any]) -> bool:
    if sh == "params":
        fields["parameters"] = _decode_params_clause(item)
        return True
    if sh == "precondition":
        if len(item.items) != 2:
            raise HCIRParseError(
                "(precondition ...) requires 1 pattern", line=item.line, col=item.col
            )
        fields["precondition_pattern"] = _decode_pattern_slot(item.items[1])
        return True
    if sh == "postcondition":
        if len(item.items) != 2:
            raise HCIRParseError(
                "(postcondition ...) requires 1 pattern", line=item.line, col=item.col
            )
        fields["postcondition_pattern"] = _decode_pattern_slot(item.items[1])
        return True
    if sh == "reliability":
        kws, pos = _split_kw_and_pos(item.items[1:])
        if pos:
            raise HCIRParseError(
                "Unexpected positional arg in (reliability ...)", line=pos[0].line, col=pos[0].col
            )
        if "alpha" in kws:
            fields["alpha_success"] = _decode_float(kws["alpha"])
        if "beta" in kws:
            fields["beta_failure"] = _decode_float(kws["beta"])
        if "success-rate" in kws:
            fields["success_rate"] = _decode_float(kws["success-rate"])
        if "invocations" in kws:
            fields["invocation_count"] = _decode_int(kws["invocations"])
        return True
    if sh == "body":
        fields["bytecode_stream"] = _decode_stream(item).model_dump()
        return True
    if sh == "body-raw":
        if len(item.items) != 2:
            raise HCIRParseError("(body-raw ...) requires 1 map", line=item.line, col=item.col)
        fields["bytecode_stream"] = _decode_value(item.items[1])
        return True
    return False


def _encode_skill_node(node: SkillNode, *, mode: Literal["canonical", "exact"]) -> SList:
    items: list[SNode] = [SAtomSym("skill", 0, 0), SAtomStr(node.skill_name, 0, 0)]
    if mode == "canonical" and node.id:
        items.extend((SAtomKw("id", 0, 0), SAtomStr(node.id, 0, 0)))
    if node.description:
        items.extend((SAtomKw("description", 0, 0), SAtomStr(node.description, 0, 0)))
    if node.preconditions:
        items.append(
            SList(
                (SAtomSym("preconditions", 0, 0), *(SAtomStr(s, 0, 0) for s in node.preconditions)),
                0,
                0,
            )
        )
    if node.postconditions:
        items.append(
            SList(
                (
                    SAtomSym("postconditions", 0, 0),
                    *(SAtomStr(s, 0, 0) for s in node.postconditions),
                ),
                0,
                0,
            )
        )
    items.extend(_encode_homoiconic_clauses(node, mode=mode))
    extra = {"name": node.name} if mode == "exact" and node.name != node.skill_name else None
    if (meta := _encode_meta(node, mode=mode, extra_fields=extra)) is not None:
        items.append(meta)
    return SList(tuple(items), 0, 0)


def _decode_skill_node(node: SList) -> SkillNode:
    kws, pos = _split_kw_and_pos(node.items[1:])
    if not pos:
        raise HCIRParseError("(skill ...) requires <skill_name>", line=node.line, col=node.col)
    fields: dict[str, Any] = {"skill_name": _decode_str(pos[0])}
    if "id" in kws:
        fields["id"] = _decode_str(kws["id"])
    if "description" in kws:
        fields["description"] = _decode_str(kws["description"])
    for sub in pos[1:]:
        if not isinstance(sub, SList):
            raise HCIRParseError("Expected clause inside (skill ...)", line=sub.line, col=sub.col)
        sh = _head_sym(sub)
        if sh == "preconditions":
            fields["preconditions"] = [_decode_str(s) for s in sub.items[1:]]
        elif sh == "postconditions":
            fields["postconditions"] = [_decode_str(s) for s in sub.items[1:]]
        elif _apply_homoiconic_clause(sh, sub, fields):
            continue
        elif sh == "meta":
            fields.update(_decode_meta(sub))
        else:
            raise HCIRParseError(f"Unknown (skill ...) clause '{sh}'", line=sub.line, col=sub.col)
    return SkillNode.model_validate(fields)


def _encode_procedure_node(node: ProcedureNode, *, mode: Literal["canonical", "exact"]) -> SList:
    items: list[SNode] = [SAtomSym("procedure", 0, 0), SAtomStr(node.procedure_name, 0, 0)]
    if mode == "canonical" and node.id:
        items.extend((SAtomKw("id", 0, 0), SAtomStr(node.id, 0, 0)))
    if node.description:
        items.extend((SAtomKw("description", 0, 0), SAtomStr(node.description, 0, 0)))
    if node.is_atomic:
        items.extend((SAtomKw("atomic", 0, 0), SAtomBool(True, 0, 0)))
    if node.steps:
        items.append(
            SList((SAtomSym("steps", 0, 0), *(SAtomStr(s, 0, 0) for s in node.steps)), 0, 0)
        )
    items.extend(_encode_homoiconic_clauses(node, mode=mode))
    extra = {"name": node.name} if mode == "exact" and node.name != node.procedure_name else None
    if (meta := _encode_meta(node, mode=mode, extra_fields=extra)) is not None:
        items.append(meta)
    return SList(tuple(items), 0, 0)


def _decode_procedure_node(node: SList) -> ProcedureNode:
    kws, pos = _split_kw_and_pos(node.items[1:])
    if not pos:
        raise HCIRParseError(
            "(procedure ...) requires <procedure_name>", line=node.line, col=node.col
        )
    fields: dict[str, Any] = {"procedure_name": _decode_str(pos[0])}
    if "id" in kws:
        fields["id"] = _decode_str(kws["id"])
    if "description" in kws:
        fields["description"] = _decode_str(kws["description"])
    if "atomic" in kws:
        fields["is_atomic"] = _decode_bool(kws["atomic"])
    for sub in pos[1:]:
        if not isinstance(sub, SList):
            raise HCIRParseError(
                "Expected clause inside (procedure ...)", line=sub.line, col=sub.col
            )
        sh = _head_sym(sub)
        if sh == "steps":
            fields["steps"] = [_decode_str(s) for s in sub.items[1:]]
        elif _apply_homoiconic_clause(sh, sub, fields):
            continue
        elif sh == "meta":
            fields.update(_decode_meta(sub))
        else:
            raise HCIRParseError(
                f"Unknown (procedure ...) clause '{sh}'", line=sub.line, col=sub.col
            )
    return ProcedureNode.model_validate(fields)


def _encode_rule_node(node: LearnedRuleNode, *, mode: Literal["canonical", "exact"]) -> SList:
    items: list[SNode] = [
        SAtomSym("rule", 0, 0),
        SAtomStr(node.id, 0, 0),
        SAtomKw("condition", 0, 0),
        SAtomStr(node.condition, 0, 0),
        SAtomKw("prediction", 0, 0),
        SAtomStr(node.prediction, 0, 0),
    ]
    if node.domain:
        items.extend((SAtomKw("domain", 0, 0), SAtomStr(node.domain, 0, 0)))
    if mode == "exact" or node.confidence != 0.5:
        items.extend((SAtomKw("confidence", 0, 0), SAtomFloat(float(node.confidence), 0, 0)))
    if mode == "exact" or node.source_error_count != 0:
        items.extend(
            (SAtomKw("source-error-count", 0, 0), SAtomInt(int(node.source_error_count), 0, 0))
        )
    if node.applies_to_types:
        items.append(
            SList(
                (SAtomSym("applies-to", 0, 0), *(SAtomStr(t, 0, 0) for t in node.applies_to_types)),
                0,
                0,
            )
        )
    items.extend(_encode_homoiconic_clauses(node, mode=mode))
    if (meta := _encode_meta(node, mode=mode)) is not None:
        items.append(meta)
    return SList(tuple(items), 0, 0)


def _decode_rule_node(node: SList) -> LearnedRuleNode:
    kws, pos = _split_kw_and_pos(node.items[1:])
    if not pos:
        raise HCIRParseError("(rule ...) requires <id>", line=node.line, col=node.col)
    fields: dict[str, Any] = {"id": _decode_str(pos[0])}
    for kw, pk in (("condition", "condition"), ("prediction", "prediction"), ("domain", "domain")):
        if kw in kws:
            fields[pk] = _decode_str(kws[kw])
    if "confidence" in kws:
        fields["confidence"] = _decode_float(kws["confidence"])
    if "source-error-count" in kws:
        fields["source_error_count"] = _decode_int(kws["source-error-count"])
    for sub in pos[1:]:
        if not isinstance(sub, SList):
            raise HCIRParseError("Expected clause inside (rule ...)", line=sub.line, col=sub.col)
        sh = _head_sym(sub)
        if sh == "applies-to":
            fields["applies_to_types"] = [_decode_str(s) for s in sub.items[1:]]
        elif _apply_homoiconic_clause(sh, sub, fields):
            continue
        elif sh == "meta":
            fields.update(_decode_meta(sub))
        else:
            raise HCIRParseError(f"Unknown (rule ...) clause '{sh}'", line=sub.line, col=sub.col)
    return LearnedRuleNode.model_validate(fields)


def _encode_construction_node(
    node: ConstructionNode, *, mode: Literal["canonical", "exact"]
) -> SList:
    items: list[SNode] = [SAtomSym("construction", 0, 0), SAtomStr(node.construction_name, 0, 0)]
    if mode == "canonical" and node.id:
        items.extend((SAtomKw("id", 0, 0), SAtomStr(node.id, 0, 0)))
    items.extend(
        (
            SAtomKw("language", 0, 0),
            SAtomStr(node.language, 0, 0),
            SAtomKw("intent-type", 0, 0),
            SAtomStr(node.intent_type, 0, 0),
            SAtomKw("template", 0, 0),
            SAtomStr(node.syntactic_template, 0, 0),
        )
    )
    if node.trigger_tokens:
        items.append(
            SList(
                (SAtomSym("triggers", 0, 0), *(SAtomStr(t, 0, 0) for t in node.trigger_tokens)),
                0,
                0,
            )
        )
    if node.slot_constraints:
        items.append(
            SList(
                (SAtomSym("slot-constraints", 0, 0), *_encode_map_pairs(node.slot_constraints)),
                0,
                0,
            )
        )
    if node.semantic_mapping:
        items.append(
            SList(
                (SAtomSym("semantic-mapping", 0, 0), *_encode_map_pairs(node.semantic_mapping)),
                0,
                0,
            )
        )
    if node.compiled_hcir_template:
        tpl_ins: list[SNode] = []
        raw_fallback = False
        for raw_i in node.compiled_hcir_template:
            try:
                ins = Instruction.model_validate(dict(raw_i))
            except (ValueError, TypeError):
                raw_fallback = True
                break
            if mode == "exact" and ins.model_dump() != dict(raw_i):
                raw_fallback = True
                break
            tpl_ins.append(_encode_instruction(ins, mode=mode))
        if raw_fallback:
            items.append(
                SList(
                    (
                        SAtomSym("compiled-template-raw", 0, 0),
                        _encode_value(node.compiled_hcir_template),
                    ),
                    0,
                    0,
                )
            )
        else:
            items.append(SList((SAtomSym("compiled-template", 0, 0), *tpl_ins), 0, 0))
    items.extend(_encode_homoiconic_clauses(node, mode=mode))
    extra = {"name": node.name} if mode == "exact" and node.name != node.construction_name else None
    if (meta := _encode_meta(node, mode=mode, extra_fields=extra)) is not None:
        items.append(meta)
    return SList(tuple(items), 0, 0)


def _decode_construction_node(node: SList) -> ConstructionNode:
    kws, pos = _split_kw_and_pos(node.items[1:])
    if not pos:
        raise HCIRParseError(
            "(construction ...) requires <construction_name>", line=node.line, col=node.col
        )
    fields: dict[str, Any] = {"construction_name": _decode_str(pos[0])}
    if "id" in kws:
        fields["id"] = _decode_str(kws["id"])
    if "language" in kws:
        fields["language"] = _decode_str(kws["language"])
    if "intent-type" in kws:
        fields["intent_type"] = _decode_str(kws["intent-type"])
    if "template" in kws:
        fields["syntactic_template"] = _decode_str(kws["template"])
    for sub in pos[1:]:
        if not isinstance(sub, SList):
            raise HCIRParseError(
                "Expected clause inside (construction ...)", line=sub.line, col=sub.col
            )
        sh = _head_sym(sub)
        if sh == "triggers":
            fields["trigger_tokens"] = [_decode_str(s) for s in sub.items[1:]]
        elif sh == "slot-constraints":
            fields["slot_constraints"] = _decode_kv_pairs(sub.items[1:])
        elif sh == "semantic-mapping":
            fields["semantic_mapping"] = _decode_kv_pairs(sub.items[1:])
        elif sh == "compiled-template":
            fields["compiled_hcir_template"] = [
                _decode_instruction(c).model_dump() for c in sub.items[1:]
            ]
        elif sh == "compiled-template-raw":
            if len(sub.items) != 2:
                raise HCIRParseError(
                    "(compiled-template-raw ...) requires 1 list", line=sub.line, col=sub.col
                )
            fields["compiled_hcir_template"] = _decode_value(sub.items[1])
        elif _apply_homoiconic_clause(sh, sub, fields):
            continue
        elif sh == "meta":
            fields.update(_decode_meta(sub))
        else:
            raise HCIRParseError(
                f"Unknown (construction ...) clause '{sh}'", line=sub.line, col=sub.col
            )
    return ConstructionNode.model_validate(fields)


def _encode_concept_node(node: ConceptNode, *, mode: Literal["canonical", "exact"]) -> SList:
    items: list[SNode] = [
        SAtomSym("concept", 0, 0),
        SAtomStr(node.id, 0, 0),
        SAtomKw("label", 0, 0),
        SAtomStr(node.label, 0, 0),
    ]
    if node.domain:
        items.extend((SAtomKw("domain", 0, 0), SAtomStr(node.domain, 0, 0)))
    if node.definition:
        items.extend((SAtomKw("definition", 0, 0), SAtomStr(node.definition, 0, 0)))
    if node.confidence is not None and mode == "canonical":
        items.extend((SAtomKw("confidence", 0, 0), SAtomFloat(float(node.confidence), 0, 0)))
    if (meta := _encode_meta(node, mode=mode)) is not None:
        items.append(meta)
    return SList(tuple(items), 0, 0)


def _decode_concept_node(node: SList) -> ConceptNode:
    kws, pos = _split_kw_and_pos(node.items[1:])
    if not pos:
        raise HCIRParseError("(concept ...) requires <id>", line=node.line, col=node.col)
    fields: dict[str, Any] = {"id": _decode_str(pos[0])}
    for kw, pk in (("label", "label"), ("domain", "domain"), ("definition", "definition")):
        if kw in kws:
            fields[pk] = _decode_str(kws[kw])
    if "confidence" in kws:
        fields["confidence"] = _decode_float(kws["confidence"])
    for sub in pos[1:]:
        if not isinstance(sub, SList) or _head_sym(sub) != "meta":
            raise HCIRParseError("Unexpected clause in (concept ...)", line=sub.line, col=sub.col)
        fields.update(_decode_meta(sub))
    return ConceptNode.model_validate(fields)


def _encode_adaptive_membership_node(
    node: AdaptiveMembershipNode, *, mode: Literal["canonical", "exact"]
) -> SList:
    items: list[SNode] = [
        SAtomSym("adaptive-membership", 0, 0),
        SAtomStr(node.id, 0, 0),
        SAtomKw("model-id", 0, 0),
        SAtomStr(node.model_id, 0, 0),
        SAtomKw("target-id", 0, 0),
        SAtomStr(node.target_id, 0, 0),
    ]
    if mode == "exact" or node.contextual_scope != "default":
        items.extend((SAtomKw("contextual-scope", 0, 0), SAtomStr(node.contextual_scope, 0, 0)))
    items.extend(
        (
            SAtomKw("kernel", 0, 0),
            _encode_ident(node.kernel_family),
            SAtomKw("scale", 0, 0),
            _encode_ident(node.reference_scale_kind),
        )
    )
    if mode == "exact" or node.sample_count != 0:
        items.extend((SAtomKw("sample-count", 0, 0), SAtomInt(int(node.sample_count), 0, 0)))
    if mode == "exact" or node.version != 1:
        items.extend((SAtomKw("version", 0, 0), SAtomInt(int(node.version), 0, 0)))
    if node.parent_version is not None:
        items.extend((SAtomKw("parent-version", 0, 0), SAtomInt(int(node.parent_version), 0, 0)))
    if node.parent_hash is not None:
        items.extend((SAtomKw("parent-hash", 0, 0), SAtomStr(node.parent_hash, 0, 0)))
    if node.learned_parameters:
        items.append(
            SList(
                (SAtomSym("learned-params", 0, 0), *_encode_map_pairs(node.learned_parameters)),
                0,
                0,
            )
        )
    if node.version_snapshots:
        snap_items = tuple(
            SList((SAtomStr(str(ver), 0, 0), SList(_encode_map_pairs(pm), 0, 0)), 0, 0)
            for ver, pm in sorted(node.version_snapshots.items(), key=lambda kv: str(kv[0]))
        )
        items.append(SList((SAtomSym("version-snapshots", 0, 0), *snap_items), 0, 0))
    if (meta := _encode_meta(node, mode=mode)) is not None:
        items.append(meta)
    return SList(tuple(items), 0, 0)


def _decode_adaptive_membership_node(node: SList) -> AdaptiveMembershipNode:
    kws, pos = _split_kw_and_pos(node.items[1:])
    if not pos:
        raise HCIRParseError(
            "(adaptive-membership ...) requires <id>", line=node.line, col=node.col
        )
    fields: dict[str, Any] = {"id": _decode_str(pos[0])}
    for kw, pk in (
        ("model-id", "model_id"),
        ("target-id", "target_id"),
        ("contextual-scope", "contextual_scope"),
        ("kernel", "kernel_family"),
        ("scale", "reference_scale_kind"),
        ("parent-hash", "parent_hash"),
    ):
        if kw in kws:
            fields[pk] = _decode_str(kws[kw])
    for kw, pk in (
        ("sample-count", "sample_count"),
        ("version", "version"),
        ("parent-version", "parent_version"),
    ):
        if kw in kws:
            fields[pk] = _decode_int(kws[kw])
    for sub in pos[1:]:
        if not isinstance(sub, SList):
            raise HCIRParseError(
                "Expected clause in (adaptive-membership ...)", line=sub.line, col=sub.col
            )
        sh = _head_sym(sub)
        if sh == "learned-params":
            fields["learned_parameters"] = _decode_kv_pairs(sub.items[1:])
        elif sh == "version-snapshots":
            snaps: dict[str, dict[str, float]] = {}
            for entry in sub.items[1:]:
                if (
                    not isinstance(entry, SList)
                    or len(entry.items) != 2
                    or not isinstance(entry.items[1], SList)
                ):
                    raise HCIRParseError(
                        "Expected (ver (pairs...)) in (version-snapshots ...)",
                        line=entry.line,
                        col=entry.col,
                    )
                snaps[_decode_str(entry.items[0])] = _decode_kv_pairs(entry.items[1].items)
            fields["version_snapshots"] = snaps
        elif sh == "meta":
            fields.update(_decode_meta(sub))
        else:
            raise HCIRParseError(
                f"Unknown clause '{sh}' in (adaptive-membership ...)", line=sub.line, col=sub.col
            )
    return AdaptiveMembershipNode.model_validate(fields)


def _encode_generic_node(node: HCIRNode, *, mode: Literal["canonical", "exact"]) -> SList:
    dumped = node.model_dump(mode="json")
    payload = {k: v for k, v in dumped.items() if k not in _META_BASE_KEYS}
    items: list[SNode] = [
        SAtomSym("node-record", 0, 0),
        _encode_ident(node.node_type.value),
        SAtomStr(node.id, 0, 0),
    ]
    if payload:
        items.append(SList((SAtomSym("fields", 0, 0), *_encode_map_pairs(payload)), 0, 0))
    if (meta := _encode_meta(node, mode=mode)) is not None:
        items.append(meta)
    return SList(tuple(items), 0, 0)


def _decode_generic_node(node: SList) -> HCIRNode:
    _, pos = _split_kw_and_pos(node.items[1:])
    if len(pos) < 2:
        raise HCIRParseError(
            "(node-record ...) requires <node_type> <id>", line=node.line, col=node.col
        )
    nt = HCIRNodeType(_decode_str(pos[0]))
    cls = NODE_TYPE_REGISTRY[nt]
    fields: dict[str, Any] = {"id": _decode_str(pos[1]), "node_type": nt.value}
    for sub in pos[2:]:
        if not isinstance(sub, SList):
            raise HCIRParseError("Expected clause in (node-record ...)", line=sub.line, col=sub.col)
        sh = _head_sym(sub)
        if sh == "fields":
            fields.update(_decode_kv_pairs(sub.items[1:]))
        elif sh == "meta":
            fields.update(_decode_meta(sub))
        else:
            raise HCIRParseError(
                f"Unknown clause '{sh}' in (node-record ...)", line=sub.line, col=sub.col
            )
    return cls.model_validate(fields)


def _encode_composition_policy(policy: RelationCompositionPolicy) -> SList:
    r_nodes: list[SNode] = []
    for r in policy.rules:
        r_items: list[SNode] = [
            SAtomSym("compose-rule", 0, 0),
            SAtomStr(r.first_relation, 0, 0),
            SAtomStr(r.second_relation, 0, 0),
            SAtomSym("->", 0, 0),
            SAtomNil(0, 0) if r.composed_relation is None else SAtomStr(r.composed_relation, 0, 0),
            SAtomKw("directionality", 0, 0),
            _encode_ident(r.directionality),
            SAtomKw("guard", 0, 0),
            _encode_ident(r.geometric_guard),
        ]
        if r.max_scale_ratio is not None:
            r_items.extend(
                (SAtomKw("max-scale-ratio", 0, 0), SAtomFloat(float(r.max_scale_ratio), 0, 0))
            )
        if r.contextual_scope != "*":
            r_items.extend((SAtomKw("contextual-scope", 0, 0), SAtomStr(r.contextual_scope, 0, 0)))
        r_items.extend(
            (
                SAtomKw("operator", 0, 0),
                _encode_ident(r.composition_operator),
                SAtomKw("attenuation", 0, 0),
                SAtomFloat(float(r.attenuation_factor), 0, 0),
            )
        )
        r_nodes.append(SList(tuple(r_items), 0, 0))
    return SList(
        (SAtomSym("composition-policy", 0, 0), SAtomStr(policy.policy_id, 0, 0), *r_nodes), 0, 0
    )


def _decode_composition_policy(node: SList) -> RelationCompositionPolicy:
    if len(node.items) < 2:
        raise HCIRParseError(
            "(composition-policy ...) requires <policy_id>", line=node.line, col=node.col
        )
    policy_id = _decode_str(node.items[1])
    rules: list[RelationCompositionRule] = []
    for item in node.items[2:]:
        if not isinstance(item, SList) or _head_sym(item) != "compose-rule":
            raise HCIRParseError("Expected (compose-rule ...)", line=item.line, col=item.col)
        kws, pos = _split_kw_and_pos(item.items[1:])
        if len(pos) != 4 or not isinstance(pos[2], SAtomSym) or pos[2].name != "->":
            raise HCIRParseError(
                "(compose-rule ...) requires <r1> <r2> -> <out-or-nil>",
                line=item.line,
                col=item.col,
            )
        r1 = _decode_str(pos[0])
        r2 = _decode_str(pos[1])
        out_rel = None if isinstance(pos[3], SAtomNil) else _decode_str(pos[3])
        r_kw: dict[str, Any] = {
            "first_relation": r1,
            "second_relation": r2,
            "composed_relation": out_rel,
        }
        if "directionality" in kws:
            r_kw["directionality"] = _decode_str(kws["directionality"])
        if "guard" in kws:
            r_kw["geometric_guard"] = _decode_str(kws["guard"])
        if "max-scale-ratio" in kws:
            r_kw["max_scale_ratio"] = _decode_float(kws["max-scale-ratio"])
        if "contextual-scope" in kws:
            r_kw["contextual_scope"] = _decode_str(kws["contextual-scope"])
        if "operator" in kws:
            r_kw["composition_operator"] = _decode_str(kws["operator"])
        if "attenuation" in kws:
            r_kw["attenuation_factor"] = _decode_float(kws["attenuation"])
        rules.append(RelationCompositionRule(**r_kw))
    return RelationCompositionPolicy(policy_id=policy_id, rules=tuple(rules))


def _encode_edge(edge: HCIREdge, *, mode: Literal["canonical", "exact"]) -> SList:
    items: list[SNode] = [
        SAtomSym("edge-def", 0, 0),
        SAtomStr(edge.id, 0, 0),
        SAtomKw("type", 0, 0),
        _encode_ident(edge.edge_type.value),
        SAtomKw("sources", 0, 0),
        SList(tuple(SAtomStr(s, 0, 0) for s in edge.sources), 0, 0),
        SAtomKw("targets", 0, 0),
        SList(tuple(SAtomStr(t, 0, 0) for t in edge.targets), 0, 0),
    ]
    if mode == "exact" or edge.weight != 1.0:
        items.extend((SAtomKw("weight", 0, 0), SAtomFloat(float(edge.weight), 0, 0)))
    if edge.properties:
        items.append(SList((SAtomSym("properties", 0, 0), _encode_value(edge.properties)), 0, 0))
    if mode == "exact":
        items.append(
            SList(
                (
                    SAtomSym("provenance", 0, 0),
                    _encode_value(edge.provenance.model_dump(mode="json")),
                ),
                0,
                0,
            )
        )
    return SList(tuple(items), 0, 0)


def _decode_edge(node: SList) -> HCIREdge:
    kws, pos = _split_kw_and_pos(node.items[1:])
    if not pos or "type" not in kws or "sources" not in kws or "targets" not in kws:
        raise HCIRParseError(
            "(edge-def ...) requires <id> :type :sources :targets", line=node.line, col=node.col
        )
    src_n, tgt_n = kws["sources"], kws["targets"]
    if not isinstance(src_n, SList) or not isinstance(tgt_n, SList):
        raise HCIRParseError(":sources and :targets expect lists", line=node.line, col=node.col)
    fields: dict[str, Any] = {
        "id": _decode_str(pos[0]),
        "edge_type": HCIREdgeType(_decode_str(kws["type"])).value,
        "sources": [_decode_str(s) for s in src_n.items],
        "targets": [_decode_str(t) for t in tgt_n.items],
    }
    if "weight" in kws:
        fields["weight"] = _decode_float(kws["weight"])
    for sub in pos[1:]:
        if not isinstance(sub, SList):
            raise HCIRParseError("Expected clause in (edge-def ...)", line=sub.line, col=sub.col)
        sh = _head_sym(sub)
        if sh in ("properties", "provenance"):
            fields[sh] = _decode_value(sub.items[1])
        else:
            raise HCIRParseError(
                f"Unknown clause '{sh}' in (edge-def ...)", line=sub.line, col=sub.col
            )
    return HCIREdge.model_validate(fields)


def _encode_seed_bundle(bundle: Mapping[str, Any], *, mode: Literal["canonical", "exact"]) -> SList:
    kind = str(bundle.get("kind", "constructions" if "constructions" in bundle else "core-priors"))
    items: list[SNode] = [SAtomSym("seed-bundle", 0, 0), SAtomKw("kind", 0, 0), _encode_ident(kind)]
    for bk, kw in (
        ("version", "seed-version"),
        ("authority", "authority"),
        ("description", "description"),
        ("language", "language"),
    ):
        if bk in bundle and bundle[bk] is not None:
            items.extend((SAtomKw(kw, 0, 0), SAtomStr(str(bundle[bk]), 0, 0)))
    if "lexical_rules" in bundle and isinstance(bundle["lexical_rules"], Mapping):
        items.append(
            SList(
                (SAtomSym("lexical-rules", 0, 0), *_encode_map_pairs(bundle["lexical_rules"])), 0, 0
            )
        )
    if bundle.get("concepts"):
        c_nodes = [
            _encode_concept_node(
                c if isinstance(c, ConceptNode) else ConceptNode.model_validate(dict(c)),
                mode=mode,
            )
            for c in bundle["concepts"]
        ]
        items.append(SList((SAtomSym("concepts", 0, 0), *c_nodes), 0, 0))
    if bundle.get("rules"):
        r_nodes = [
            _encode_rule_node(
                r if isinstance(r, LearnedRuleNode) else LearnedRuleNode.model_validate(dict(r)),
                mode=mode,
            )
            for r in bundle["rules"]
        ]
        items.append(SList((SAtomSym("rules", 0, 0), *r_nodes), 0, 0))
    if bundle.get("skills"):
        s_nodes = [
            _encode_skill_node(
                s if isinstance(s, SkillNode) else SkillNode.model_validate(dict(s)),
                mode=mode,
            )
            for s in bundle["skills"]
        ]
        items.append(SList((SAtomSym("skills", 0, 0), *s_nodes), 0, 0))
    if bundle.get("constructions"):
        cxn_nodes = [
            _encode_construction_node(
                cx
                if isinstance(cx, ConstructionNode)
                else ConstructionNode.model_validate(dict(cx)),
                mode=mode,
            )
            for cx in bundle["constructions"]
        ]
        items.append(SList((SAtomSym("constructions", 0, 0), *cxn_nodes), 0, 0))
    if (pol := bundle.get("relation_composition_policy")) is not None:
        p_obj = (
            pol
            if isinstance(pol, RelationCompositionPolicy)
            else RelationCompositionPolicy.model_validate(dict(pol))
        )
        items.append(_encode_composition_policy(p_obj))
    if bundle.get("edges"):
        e_nodes = [
            _encode_edge(
                e if isinstance(e, HCIREdge) else HCIREdge.model_validate(dict(e)), mode=mode
            )
            for e in bundle["edges"]
        ]
        items.append(SList((SAtomSym("edges", 0, 0), *e_nodes), 0, 0))
    return SList(tuple(items), 0, 0)


def _decode_seed_bundle(node: SList) -> dict[str, Any]:
    kws, pos = _split_kw_and_pos(node.items[1:])
    out: dict[str, Any] = {}
    if "kind" in kws:
        out["kind"] = _decode_str(kws["kind"])
    for kw, bk in (
        ("seed-version", "version"),
        ("authority", "authority"),
        ("description", "description"),
        ("language", "language"),
    ):
        if kw in kws:
            out[bk] = _decode_str(kws[kw])
    for sub in pos:
        if not isinstance(sub, SList):
            raise HCIRParseError(
                "Expected section inside (seed-bundle ...)", line=sub.line, col=sub.col
            )
        sh = _head_sym(sub)
        if sh == "lexical-rules":
            out["lexical_rules"] = _decode_kv_pairs(sub.items[1:])
        elif sh == "concepts":
            out["concepts"] = [
                _decode_concept_node(c) for c in sub.items[1:] if isinstance(c, SList)
            ]
        elif sh == "rules":
            out["rules"] = [_decode_rule_node(r) for r in sub.items[1:] if isinstance(r, SList)]
        elif sh == "skills":
            out["skills"] = [_decode_skill_node(s) for s in sub.items[1:] if isinstance(s, SList)]
        elif sh == "constructions":
            out["constructions"] = [
                _decode_construction_node(cx) for cx in sub.items[1:] if isinstance(cx, SList)
            ]
        elif sh == "composition-policy":
            out["relation_composition_policy"] = _decode_composition_policy(sub)
        elif sh == "edges":
            out["edges"] = [_decode_edge(e) for e in sub.items[1:] if isinstance(e, SList)]
        else:
            raise HCIRParseError(
                f"Unknown section '{sh}' in (seed-bundle ...)", line=sub.line, col=sub.col
            )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Top-Level Encode / Decode & Deterministic Pretty-Printer
# ═══════════════════════════════════════════════════════════════════════════

_MULTILINE_HEADS: frozenset[str] = frozenset(
    {
        "hcir",
        "seed-bundle",
        "stream",
        "pattern",
        "nodes",
        "edges",
        "any-of",
        "none-of",
        "skill",
        "procedure",
        "rule",
        "construction",
        "concept",
        "adaptive-membership",
        "node-record",
        "composition-policy",
        "edge-def",
        "meta",
        "params",
        "body",
        "compiled-template",
        "concepts",
        "rules",
        "skills",
        "constructions",
        "lexical-rules",
        "slot-constraints",
        "semantic-mapping",
    }
)


def _render_atom(node: SNode) -> str:
    if isinstance(node, SAtomNil):
        return "nil"
    if isinstance(node, SAtomBool):
        return "true" if node.value else "false"
    if isinstance(node, SAtomInt):
        return str(node.value)
    if isinstance(node, SAtomFloat):
        return _format_float(node.value)
    if isinstance(node, SAtomStr):
        return _quote_str(node.value)
    if isinstance(node, SAtomReg):
        return node.slot
    if isinstance(node, SAtomVar):
        return node.name
    if isinstance(node, SAtomKw):
        return f":{node.name}"
    if isinstance(node, SAtomSym):
        return node.name
    raise TypeError(f"Not an atom: {node!r}")


def _can_render_inline(node: SNode) -> bool:
    if not isinstance(node, SList):
        return True
    if (
        node.items
        and isinstance(node.items[0], SAtomSym)
        and node.items[0].name in _MULTILINE_HEADS
    ):
        return False
    return all(
        not isinstance(c, SList)
        or (
            not (
                c.items and isinstance(c.items[0], SAtomSym) and c.items[0].name in _MULTILINE_HEADS
            )
        )
        for c in node.items
    )


def _group_list_entries(items: tuple[SNode, ...]) -> list[list[SNode]]:
    """Group head + keyword pairs or positional child forms cleanly onto lines."""
    if not items:
        return []
    groups: list[list[SNode]] = []
    if isinstance(items[0], SAtomSym):
        header: list[SNode] = [items[0]]
        idx = 1
        # Keep short inline label/name right after head on line 0
        if idx < len(items) and not isinstance(items[idx], (SAtomKw, SList)):
            header.append(items[idx])
            idx += 1
        groups.append(header)
    else:
        idx = 0
    while idx < len(items):
        cur = items[idx]
        if (
            isinstance(cur, SAtomKw)
            and idx + 1 < len(items)
            and not (
                isinstance(items[idx], SAtomKw)
                and cur.name in ("no-capability", "no-pattern", "empty")
            )
        ):
            groups.append([cur, items[idx + 1]])
            idx += 2
        else:
            groups.append([cur])
            idx += 1
    return groups


def _render_snode(node: SNode, indent: int = 0) -> str:
    if not isinstance(node, SList):
        return _render_atom(node)
    if not node.items:
        return "()"
    if _can_render_inline(node):
        flat = "(" + " ".join(_render_snode(c, 0) for c in node.items) + ")"
        if indent + len(flat) <= 88:
            return flat
    groups = _group_list_entries(node.items)
    pad = " " * (indent + 2)
    lines: list[str] = ["(" + " ".join(_render_snode(c, indent) for c in groups[0])]
    for grp in groups[1:]:
        if len(grp) == 2 and isinstance(grp[0], SAtomKw):
            kw_txt = _render_atom(grp[0])
            val_txt = _render_snode(grp[1], indent + 2)
            lines.append(f"{pad}{kw_txt} {val_txt}")
        else:
            lines.append(f"{pad}{_render_snode(grp[0], indent + 2)}")
    lines[-1] += ")"
    return "\n".join(lines)


def _encode_top_level(obj: Any, *, mode: Literal["canonical", "exact"]) -> tuple[str, SList]:
    if isinstance(obj, InstructionStream):
        return "stream", _encode_stream_body(obj, mode=mode)
    if isinstance(obj, GraphPattern):
        return "pattern", _encode_graph_pattern(obj)
    if isinstance(obj, SymbolicExpr):
        return "expr", _encode_symbolic_expr(obj)
    if isinstance(obj, FuzzyExpr):
        return "fuzzy-expr", _encode_fuzzy_expr(obj)
    if isinstance(obj, SkillNode):
        return "skill", _encode_skill_node(obj, mode=mode)
    if isinstance(obj, ProcedureNode):
        return "procedure", _encode_procedure_node(obj, mode=mode)
    if isinstance(obj, LearnedRuleNode):
        return "rule", _encode_rule_node(obj, mode=mode)
    if isinstance(obj, ConstructionNode):
        return "construction", _encode_construction_node(obj, mode=mode)
    if isinstance(obj, ConceptNode):
        return "concept", _encode_concept_node(obj, mode=mode)
    if isinstance(obj, AdaptiveMembershipNode):
        return "adaptive-membership", _encode_adaptive_membership_node(obj, mode=mode)
    if isinstance(obj, HCIRNode):
        return "node", _encode_generic_node(obj, mode=mode)
    if isinstance(obj, RelationCompositionPolicy):
        return "composition-policy", _encode_composition_policy(obj)
    if isinstance(obj, HCIREdge):
        return "edge", _encode_edge(obj, mode=mode)
    if isinstance(obj, Mapping) and (
        "kind" in obj or any(k in obj for k in ("concepts", "constructions", "skills", "rules"))
    ):
        return "seed-bundle", _encode_seed_bundle(obj, mode=mode)
    raise TypeError(f"Unsupported object type for format_hcir: {type(obj).__name__}")


def format_hcir(
    obj: Any,
    *,
    mode: Literal["canonical", "exact"] = "canonical",
    envelope: bool = True,
) -> str:
    """Serialize an HCIR object into deterministic `.hcir` text."""
    if mode not in ("canonical", "exact"):
        raise ValueError(f"Unsupported mode {mode!r}; expected 'canonical' or 'exact'")
    doc_type, payload = _encode_top_level(obj, mode=mode)
    root: SNode = (
        SList(
            (
                SAtomSym("hcir", 0, 0),
                SAtomKw("version", 0, 0),
                SAtomInt(1, 0, 0),
                SAtomKw("type", 0, 0),
                _encode_ident(doc_type),
                payload,
            ),
            0,
            0,
        )
        if envelope
        else payload
    )
    return _render_snode(root, 0) + "\n"


_DOC_TYPE_HEADS: dict[str, set[str]] = {
    "stream": {"stream"},
    "pattern": {"pattern"},
    "expr": {"sexpr", *_REV_SEXPR_COMPOUND.keys(), *_REV_SEXPR_LEAF.keys()},
    "fuzzy-expr": {
        "const",
        "var",
        "membership",
        "and",
        "or",
        "not",
        "implies",
        "equiv",
        "hedge",
        "alpha-cut",
        "fuzzy-op",
    },
    "skill": {"skill"},
    "procedure": {"procedure"},
    "rule": {"rule"},
    "construction": {"construction"},
    "concept": {"concept"},
    "adaptive-membership": {"adaptive-membership"},
    "node": {"node-record"},
    "composition-policy": {"composition-policy"},
    "edge": {"edge-def"},
    "seed-bundle": {"seed-bundle"},
}


def parse_hcir(text: str, *, expected_type: str | None = None) -> Any:
    """Parse canonical or exact `.hcir` text back into the corresponding HCIR object."""
    forms = _parse_sexpr_nodes(text)
    if len(forms) != 1:
        raise HCIRParseError(
            f"Expected exactly 1 top-level .hcir form, got {len(forms)}",
            line=forms[1].line if len(forms) > 1 else 1,
            col=forms[1].col if len(forms) > 1 else 1,
        )
    top = forms[0]
    if not isinstance(top, SList):
        raise HCIRParseError(
            "Top-level .hcir document must be a list form", line=top.line, col=top.col
        )

    doc_type: str | None = expected_type
    payload = top
    if _head_sym(top) == "hcir":
        kws, pos = _split_kw_and_pos(top.items[1:])
        if "version" not in kws or _decode_int(kws["version"]) != 1:
            raise HCIRParseError(
                "Unsupported or missing (hcir :version 1 ...)", line=top.line, col=top.col
            )
        if "type" not in kws or len(pos) != 1 or not isinstance(pos[0], SList):
            raise HCIRParseError(
                "(hcir ...) requires :type <doc-type> and 1 payload form",
                line=top.line,
                col=top.col,
            )
        env_type = _decode_str(kws["type"])
        if doc_type is not None and env_type != doc_type:
            raise HCIRParseError(
                f"Expected :type {doc_type!r}, got {env_type!r}",
                line=kws["type"].line,
                col=kws["type"].col,
            )
        doc_type = env_type
        payload = pos[0]

    head = _head_sym(payload)
    if doc_type is not None:
        allowed_heads = _DOC_TYPE_HEADS.get(doc_type)
        if allowed_heads is None:
            raise HCIRParseError(
                f"Unsupported document :type {doc_type!r}", line=payload.line, col=payload.col
            )
        if head not in allowed_heads:
            raise HCIRParseError(
                f"Payload head '{head}' does not match :type {doc_type!r}",
                line=payload.line,
                col=payload.col,
            )

    if doc_type == "fuzzy-expr" or head in {
        "const",
        "var",
        "membership",
        "and",
        "or",
        "implies",
        "equiv",
        "hedge",
        "alpha-cut",
        "fuzzy-op",
    }:
        return _decode_fuzzy_expr(payload)
    if doc_type == "expr" or head in {
        "sexpr",
        *_REV_SEXPR_COMPOUND.keys(),
        *_REV_SEXPR_LEAF.keys(),
    }:
        return _decode_symbolic_expr(payload)
    if head == "stream":
        return _decode_stream(payload)
    if head == "pattern":
        return _decode_graph_pattern(payload)
    if head == "skill":
        return _decode_skill_node(payload)
    if head == "procedure":
        return _decode_procedure_node(payload)
    if head == "rule":
        return _decode_rule_node(payload)
    if head == "construction":
        return _decode_construction_node(payload)
    if head == "concept":
        return _decode_concept_node(payload)
    if head == "adaptive-membership":
        return _decode_adaptive_membership_node(payload)
    if head == "node-record":
        return _decode_generic_node(payload)
    if head == "composition-policy":
        return _decode_composition_policy(payload)
    if head == "edge-def":
        return _decode_edge(payload)
    if head == "seed-bundle":
        return _decode_seed_bundle(payload)
    raise HCIRParseError(f"Unknown top-level form '{head}'", line=payload.line, col=payload.col)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — First-Class BLAKE3 Semantic & Exact Digests
# ═══════════════════════════════════════════════════════════════════════════


def _norm_digest_val(val: Any) -> Any:
    if val is None:
        return ["nil"]
    if isinstance(val, bool):
        return ["bool", val]
    if isinstance(val, int):
        return ["int", val]
    if isinstance(val, float):
        if math.isnan(val):
            raise ValueError("NaN cannot be digested")
        fv = 0.0 if val == 0.0 else float(val)
        return ["float", _format_float(fv)]
    if isinstance(val, str):
        return ["str", val]
    if isinstance(val, tuple):
        return ["tuple", [_norm_digest_val(v) for v in val]]
    if isinstance(val, list):
        return ["list", [_norm_digest_val(v) for v in val]]
    if isinstance(val, Mapping):
        return [
            "map",
            [
                [str(k), _norm_digest_val(v)]
                for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))
            ],
        ]
    if hasattr(val, "value") and isinstance(getattr(val, "value"), str):
        return ["str", str(val.value)]
    if hasattr(val, "model_dump"):
        return _norm_digest_val(val.model_dump())
    return ["str", str(val)]


def _norm_pattern_for_digest(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, GraphPattern):
        return ["graph_pattern", _norm_digest_val(raw.model_dump())]
    if isinstance(raw, Mapping) and any(
        k in {"nodes", "edges", "any_of", "none_of", "distinct_nodes"} for k in raw
    ):
        try:
            gp = GraphPattern.model_validate(dict(raw))
            return ["graph_pattern", _norm_digest_val(gp.model_dump())]
        except (ValueError, TypeError):
            pass
    if (
        isinstance(raw, Mapping)
        and "symbolic_expr" in raw
        and isinstance(raw["symbolic_expr"], Mapping)
    ):
        try:
            se = SymbolicExpr.model_validate(dict(raw["symbolic_expr"]))
            norm = dict(raw)
            norm["symbolic_expr"] = se.model_dump()
            return ["expr_guard", _norm_digest_val(norm)]
        except (ValueError, TypeError):
            pass
    return ["raw_pattern", _norm_digest_val(raw)]


def _norm_stream_for_semantic_digest(raw_or_stream: InstructionStream | Mapping[str, Any]) -> Any:
    stream = (
        raw_or_stream
        if isinstance(raw_or_stream, InstructionStream)
        else InstructionStream.model_validate(dict(raw_or_stream))
    )
    ins_list: list[Any] = []
    for ins in stream.instructions:
        params = dict(ins.params)
        if ins.opcode == Opcode.UNIFY and "pattern" in params:
            params["pattern"] = _norm_pattern_for_digest(params["pattern"])
        elif ins.opcode == Opcode.CALL and "goal_pattern" in params:
            params["goal_pattern"] = _norm_pattern_for_digest(params["goal_pattern"])
        ins_list.append(
            {
                "opcode": ins.opcode.value,
                "label": ins.label,
                "cost_estimate": int(ins.cost_estimate),
                "params": params,
            }
        )
    return _norm_digest_val({"instructions": ins_list})


def _canonical_semantic_projection(obj: Any) -> Any:
    if isinstance(obj, InstructionStream):
        return ["stream", _norm_stream_for_semantic_digest(obj)]
    if isinstance(obj, GraphPattern):
        return ["pattern", _norm_digest_val(obj.model_dump())]
    if isinstance(obj, SymbolicExpr):
        return ["expr", _norm_digest_val(obj.model_dump())]
    if isinstance(obj, FuzzyExpr):
        return ["fuzzy_expr", _norm_digest_val(obj.model_dump())]
    if isinstance(obj, RelationCompositionPolicy):
        return ["composition_policy", _norm_digest_val(obj.model_dump())]
    if isinstance(obj, HCIREdge):
        return [
            "edge",
            _norm_digest_val(
                {
                    "edge_type": obj.edge_type.value,
                    "sources": list(obj.sources),
                    "targets": list(obj.targets),
                    "weight": float(obj.weight),
                    "properties": dict(obj.properties),
                }
            ),
        ]
    if isinstance(obj, HCIRNode):
        base: dict[str, Any] = {
            "node_type": obj.node_type.value,
            "tags": sorted(str(t) for t in obj.tags),
            "properties": dict(obj.properties),
        }
        if isinstance(obj, SkillNode):
            base.update(
                {
                    "skill_name": obj.skill_name,
                    "description": obj.description,
                    "preconditions": list(obj.preconditions),
                    "postconditions": list(obj.postconditions),
                    "parameters": list(obj.parameters),
                    "precondition_pattern": _norm_pattern_for_digest(obj.precondition_pattern),
                    "postcondition_pattern": _norm_pattern_for_digest(obj.postcondition_pattern),
                    "alpha_success": float(obj.alpha_success),
                    "beta_failure": float(obj.beta_failure),
                    "bytecode_stream": (
                        _norm_stream_for_semantic_digest(obj.bytecode_stream)
                        if obj.bytecode_stream is not None
                        else None
                    ),
                }
            )
        elif isinstance(obj, ProcedureNode):
            base.update(
                {
                    "procedure_name": obj.procedure_name,
                    "description": obj.description,
                    "steps": list(obj.steps),
                    "is_atomic": bool(obj.is_atomic),
                    "parameters": list(obj.parameters),
                    "precondition_pattern": _norm_pattern_for_digest(obj.precondition_pattern),
                    "postcondition_pattern": _norm_pattern_for_digest(obj.postcondition_pattern),
                    "alpha_success": float(obj.alpha_success),
                    "beta_failure": float(obj.beta_failure),
                    "bytecode_stream": (
                        _norm_stream_for_semantic_digest(obj.bytecode_stream)
                        if obj.bytecode_stream is not None
                        else None
                    ),
                }
            )
        elif isinstance(obj, LearnedRuleNode):
            base.update(
                {
                    "condition": obj.condition,
                    "prediction": obj.prediction,
                    "confidence": float(obj.confidence),
                    "domain": obj.domain,
                    "applies_to_types": list(obj.applies_to_types),
                    "parameters": list(obj.parameters),
                    "precondition_pattern": _norm_pattern_for_digest(obj.precondition_pattern),
                    "postcondition_pattern": _norm_pattern_for_digest(obj.postcondition_pattern),
                    "alpha_success": float(obj.alpha_success),
                    "beta_failure": float(obj.beta_failure),
                    "bytecode_stream": (
                        _norm_stream_for_semantic_digest(obj.bytecode_stream)
                        if obj.bytecode_stream is not None
                        else None
                    ),
                }
            )
        elif isinstance(obj, ConstructionNode):
            base.update(
                {
                    "construction_name": obj.construction_name,
                    "language": obj.language,
                    "intent_type": obj.intent_type,
                    "syntactic_template": obj.syntactic_template,
                    "trigger_tokens": list(obj.trigger_tokens),
                    "slot_constraints": dict(obj.slot_constraints),
                    "semantic_mapping": dict(obj.semantic_mapping),
                    "compiled_hcir_template": _norm_stream_for_semantic_digest(
                        {"instructions": obj.compiled_hcir_template}
                    ),
                    "parameters": list(obj.parameters),
                    "precondition_pattern": _norm_pattern_for_digest(obj.precondition_pattern),
                    "postcondition_pattern": _norm_pattern_for_digest(obj.postcondition_pattern),
                    "alpha_success": float(obj.alpha_success),
                    "beta_failure": float(obj.beta_failure),
                    "bytecode_stream": (
                        _norm_stream_for_semantic_digest(obj.bytecode_stream)
                        if obj.bytecode_stream is not None
                        else None
                    ),
                }
            )
        elif isinstance(obj, ConceptNode):
            base.update(
                {
                    "label": obj.label,
                    "definition": obj.definition,
                    "domain": obj.domain,
                    "confidence": obj.confidence,
                }
            )
        elif isinstance(obj, AdaptiveMembershipNode):
            base.update(
                {
                    "target_id": obj.target_id,
                    "contextual_scope": obj.contextual_scope,
                    "kernel_family": obj.kernel_family,
                    "reference_scale_kind": obj.reference_scale_kind,
                    "learned_parameters": dict(obj.learned_parameters),
                }
            )
        else:
            dumped = obj.model_dump()
            for k in ("id", "provenance", "attention", "lifecycle", "uncertainty"):
                dumped.pop(k, None)
            base.update(dumped)
        return ["node", _norm_digest_val(base)]
    if isinstance(obj, CognitiveGraph):
        node_projs = [_canonical_semantic_projection(n) for n in obj.nodes.values()]
        edge_projs = [_canonical_semantic_projection(e) for e in obj.edges.values()]
        return [
            "graph",
            [
                sorted(
                    node_projs,
                    key=lambda p: json.dumps(p, separators=(",", ":"), ensure_ascii=False),
                ),
                sorted(
                    edge_projs,
                    key=lambda p: json.dumps(p, separators=(",", ":"), ensure_ascii=False),
                ),
            ],
        ]
    return ["value", _norm_digest_val(obj)]


def _exact_projection(obj: Any) -> Any:
    if isinstance(obj, CognitiveGraph):
        return [
            "exact_graph",
            [_exact_projection(n) for n in sorted(obj.nodes.values(), key=lambda x: x.id)],
            [_exact_projection(e) for e in sorted(obj.edges.values(), key=lambda x: x.id)],
        ]
    if hasattr(obj, "model_dump"):
        return ["exact", type(obj).__name__, _norm_digest_val(obj.model_dump())]
    return ["exact", type(obj).__name__, _norm_digest_val(obj)]


def semantic_digest(obj: Any) -> str:
    """Compute canonical BLAKE3 semantic identity digest independent of ephemeral IDs/timestamps."""
    proj = _canonical_semantic_projection(obj)
    payload = b"HCIR_SEMANTIC_V1\n" + json.dumps(
        proj, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return blake3_hex(payload)


canonical_digest = semantic_digest


def exact_digest(obj: Any) -> str:
    """Compute BLAKE3 exact digest preserving generated IDs, timestamps, and runtime counters."""
    proj = _exact_projection(obj)
    payload = b"HCIR_EXACT_V1\n" + json.dumps(
        proj, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return blake3_hex(payload)
