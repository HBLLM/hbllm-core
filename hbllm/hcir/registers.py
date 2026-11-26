"""HCIR ISA v2 working-memory registers, symbolic variables, and bounded call frames."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from hbllm.hcir.bytecode import InstructionStream

REGISTER_NAMES: frozenset[str] = frozenset(f"$r{i}" for i in range(8))
DEFAULT_SKIP_KEYS: frozenset[str] = frozenset(
    {"bind", "bind_var", "bind_all", "target_label", "register", "var"}
)


class VMRuntimeError(RuntimeError):
    """Base error raised on ISA v2 VM invariant or termination violations."""


class VMFuelExhaustedError(VMRuntimeError):
    """Raised when an instruction stream exhausts its VM execution fuel budget."""


class VMCallDepthExceededError(VMRuntimeError):
    """Raised when nested CALL invocations exceed CallStack.max_depth."""


class VMCallCycleError(VMRuntimeError):
    """Raised when a CALL repeats the exact (callee, args, graph_version) signature."""


def normalize_var_name(name: str) -> str:
    """Return canonical `?name` form for a symbolic variable identifier."""
    stripped = name.strip()
    if not stripped:
        raise ValueError("Symbolic variable name must be non-empty")
    return stripped if stripped.startswith("?") else f"?{stripped}"


def is_register_ref(value: Any) -> bool:
    """Return True if *value* is a RegisterRef or a canonical `$r0`..`$r7` token."""
    return isinstance(value, RegisterRef) or (isinstance(value, str) and value in REGISTER_NAMES)


def is_symbolic_var(value: Any) -> bool:
    """Return True if *value* is a SymbolicVar or a `?name` symbolic variable token."""
    return isinstance(value, SymbolicVar) or (
        isinstance(value, str) and len(value) > 1 and value.startswith("?") and not value.isspace()
    )


class RegisterRef(BaseModel):
    """Reference to one of the 8 VM working-memory registers (`$r0`..`$r7`)."""

    slot: str

    @field_validator("slot")
    @classmethod
    def _validate_slot(cls, value: str) -> str:
        if value not in REGISTER_NAMES:
            raise ValueError(f"Invalid register slot {value!r}; expected one of $r0..$r7")
        return value


class SymbolicVar(BaseModel):
    """Symbolic pattern variable (`?name`) with optional node-type/trait guards."""

    name: str
    node_type: str | None = None
    traits: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return normalize_var_name(value)


class SymbolicParam(BaseModel):
    """Declared formal parameter for a homoiconic `SkillNode` or `ProcedureNode`."""

    name: str
    node_type: str | None = None
    role: str = ""
    required: bool = True
    default: Any = None


class RegisterFile:
    """Bounded 8-slot working-memory register file (`$r0`..`$r7`) per call frame."""

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self._slots: dict[str, Any] = {f"$r{i}": None for i in range(8)}
        self._bound: set[str] = set()
        if initial:
            for slot, val in initial.items():
                self.set(slot, val)

    @staticmethod
    def _check_slot(slot: str) -> str:
        if slot not in REGISTER_NAMES:
            raise ValueError(f"Invalid register slot {slot!r}; expected $r0..$r7")
        return slot

    def is_bound(self, slot: str) -> bool:
        """Return True if *slot* has been explicitly assigned in this frame."""
        return self._check_slot(slot) in self._bound

    def get(self, slot: str, default: Any = None) -> Any:
        """Read *slot* (returning *default* when unbound)."""
        key = self._check_slot(slot)
        return self._slots[key] if key in self._bound else default

    def set(self, slot: str, value: Any) -> None:
        """Bind *slot* to *value* (preserving falsy values like 0, False, '', [])."""
        key = self._check_slot(slot)
        self._slots[key] = value
        self._bound.add(key)

    def unset(self, slot: str) -> None:
        """Clear binding for *slot*."""
        key = self._check_slot(slot)
        self._slots[key] = None
        self._bound.discard(key)

    def clear(self) -> None:
        """Reset all 8 register slots to unbound."""
        for key in self._slots:
            self._slots[key] = None
        self._bound.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return currently bound register values keyed by canonical slot name."""
        return {slot: self._slots[slot] for slot in sorted(self._bound)}

    def to_dict(self) -> dict[str, Any]:
        """Return all 8 register slots in sorted order."""
        return {f"$r{i}": self._slots[f"$r{i}"] for i in range(8)}


def resolve_operands(
    value: Any,
    registers: RegisterFile,
    bindings: Mapping[str, Any] | None = None,
    *,
    preserve_unbound_vars: bool = False,
    skip_keys: frozenset[str] = DEFAULT_SKIP_KEYS,
) -> Any:
    """Recursively substitute `$rN` and `?var` references without mutating inputs."""

    def _rec(item: Any) -> Any:
        if isinstance(item, RegisterRef):
            return registers.get(item.slot)
        if isinstance(item, str) and item in REGISTER_NAMES:
            return registers.get(item)
        if is_symbolic_var(item):
            vname = item.name if isinstance(item, SymbolicVar) else str(item)
            if bindings is not None:
                if vname in bindings:
                    return bindings[vname]
                bare = vname.lstrip("?")
                if bare in bindings:
                    return bindings[bare]
            return item if preserve_unbound_vars else None
        if isinstance(item, dict):
            return {k: (v if k in skip_keys else _rec(v)) for k, v in item.items()}
        if isinstance(item, list):
            return [_rec(x) for x in item]
        if isinstance(item, tuple):
            return tuple(_rec(x) for x in item)
        return item

    return _rec(value)


@dataclass
class CallFrame:
    """Single activation record on the HCIR ISA v2 call stack."""

    callee_id: str
    stream: InstructionStream
    pc: int = 0
    registers: RegisterFile = field(default_factory=RegisterFile)
    bindings: dict[str, Any] = field(default_factory=dict)
    return_bind: str | dict[str, str] | None = None
    branch_name: str | None = None
    call_signature: tuple[str, str, int] = ("", "", 0)
    label_index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label_index:
            self.label_index = {
                ins.label: idx for idx, ins in enumerate(self.stream.instructions) if ins.label
            }


class CallStack:
    """Bounded VM call stack enforcing `max_depth`, cycle detection, and fuel budget."""

    def __init__(self, max_depth: int = 16, max_fuel: int = 1024) -> None:
        if max_depth < 1 or max_fuel < 1:
            raise ValueError("max_depth and max_fuel must be >= 1")
        self.max_depth, self.max_fuel, self.fuel_remaining = max_depth, max_fuel, max_fuel
        self._frames: list[CallFrame] = []

    @property
    def depth(self) -> int:
        return len(self._frames)

    @property
    def is_empty(self) -> bool:
        return not self._frames

    @property
    def current(self) -> CallFrame:
        if not self._frames:
            raise VMRuntimeError("CallStack is empty")
        return self._frames[-1]

    def consume_fuel(self, cost: int = 1) -> None:
        """Consume at least 1 fuel unit per instruction; fail deterministically on 0."""
        step_cost = max(1, int(cost))
        if self.fuel_remaining < step_cost:
            raise VMFuelExhaustedError(f"VM fuel exhausted (remaining={self.fuel_remaining})")
        self.fuel_remaining -= step_cost

    def push(self, frame: CallFrame) -> None:
        """Push an activation frame after validating depth and zero-progress cycle guards."""
        if len(self._frames) >= self.max_depth:
            raise VMCallDepthExceededError(f"CallStack max_depth={self.max_depth} exceeded")
        if frame.callee_id and frame.call_signature[0]:
            for active in self._frames:
                if active.call_signature == frame.call_signature:
                    raise VMCallCycleError(f"Non-progressive call cycle for '{frame.callee_id}'")
        self._frames.append(frame)

    def pop(self) -> CallFrame:
        """Pop and return the active top frame."""
        if not self._frames:
            raise VMRuntimeError("Cannot pop from empty CallStack")
        return self._frames.pop()
