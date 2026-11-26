"""Declarative Neuro-Symbolic Skill Grammar, Serializable SymbolicExpr AST & Compilation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.capabilities.math_capabilities import compute_symmetry_score, compute_topology_cuts
from hbllm.hcir.graph import NodeLifecycle, SkillNode
from hbllm.hcir.primitives import EntityComponent, Grid, detect_background_color, segment_entities
from hbllm.hcir.skills.base import BaseHierarchicalSkill
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.types import Provenance, Scope

logger = logging.getLogger(__name__)


class SkillEvaluationContext:
    """Cached, lifted cognitive context for evaluating symbolic skill predicates."""

    def __init__(
        self, grid: Grid, available_actions: list[int], metadata: dict[str, Any] | None = None
    ) -> None:
        arr = np.asarray(grid)
        self.grid: Grid = arr[-1] if arr.ndim == 3 else arr
        self.available_actions: list[int] = list(available_actions)
        self.available_actions_set: set[int] = set(available_actions)
        self.metadata: dict[str, Any] = metadata or {}
        self._bg_color: int | None = None
        self._entities_4conn: list[EntityComponent] | None = None
        self._entities_8conn: list[EntityComponent] | None = None
        self._color_hist: dict[int, int] | None = None

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def background_color(self) -> int:
        if self._bg_color is None:
            self._bg_color = detect_background_color(self.grid)
        return self._bg_color

    @property
    def color_histogram(self) -> dict[int, int]:
        if self._color_hist is None:
            self._color_hist = dict(Counter(int(x) for x in self.grid.flatten()))
        return self._color_hist

    def get_entities(self, connectivity: int = 4) -> list[EntityComponent]:
        """Return segmented entities on the lattice, lazily computed and cached."""
        attr = "_entities_8conn" if connectivity == 8 else "_entities_4conn"
        cached = getattr(self, attr)
        if cached is None:
            cached = segment_entities(
                self.grid, bg_color=self.background_color, connectivity=connectivity
            )
            setattr(self, attr, cached)
        return cached

    def create_subcontext(
        self, r_min: int, r_max: int, c_min: int, c_max: int
    ) -> SkillEvaluationContext:
        """Create a subcontext for an extracted spatial subgrid (e.g. console panel)."""
        return SkillEvaluationContext(
            self.grid[r_min:r_max, c_min:c_max].copy(), self.available_actions, dict(self.metadata)
        )


def _eval_affordance(a: Mapping[str, Any], ctx: Any) -> bool:
    raw = (
        ctx.available_actions_set
        if hasattr(ctx, "available_actions_set")
        else ctx.get("available_actions", ())
        if isinstance(ctx, Mapping)
        else getattr(ctx, "available_actions", ())
    )
    acts = set(raw)
    if a.get("exact") is not None and acts != set(a["exact"]):
        return False
    req, forb = set(a.get("required") or ()), set(a.get("forbidden") or ())
    return (not req or req.issubset(acts)) and (not forb or forb.isdisjoint(acts))


def _eval_entity_match(
    a: Mapping[str, Any], ctx: SkillEvaluationContext, *, bbox_mode: bool
) -> bool:
    ents = ctx.get_entities(connectivity=int(a.get("connectivity", 4)))
    color, is_frame = a.get("color"), a.get("is_frame")
    mw, xw, mh, xh, ma, xa = (
        a.get(k)
        for k in ("min_width", "max_width", "min_height", "max_height", "min_area", "max_area")
    )
    matched, min_cnt, max_cnt = 0, int(a.get("min_count", 1)), a.get("max_count")
    for e in ents:
        if (color is not None and e.color != color) or (
            is_frame is not None and e.is_frame != is_frame
        ):
            continue
        if (mw is not None and e.width < mw) or (xw is not None and e.width > xw):
            continue
        if (mh is not None and e.height < mh) or (xh is not None and e.height > xh):
            continue
        if (ma is not None and e.area < ma) or (xa is not None and e.area > xa):
            continue
        matched += 1
        if bbox_mode and matched >= min_cnt:
            return True
    return (
        False
        if bbox_mode
        else (matched >= min_cnt and (max_cnt is None or matched <= int(max_cnt)))
    )


def _eval_grid_dim(a: Mapping[str, Any], ctx: SkillEvaluationContext) -> bool:
    h, w, exact = ctx.height, ctx.width, a.get("exact_shape")
    if exact is not None and (h, w) != tuple(exact):
        return False
    bounds = (
        ("min_height", h, False),
        ("max_height", h, True),
        ("min_width", w, False),
        ("max_width", w, True),
    )
    return not any(
        a.get(k) is not None and (val > a[k] if is_max else val < a[k]) for k, val, is_max in bounds
    )


def _eval_region_or_density(
    a: Mapping[str, Any], ctx: SkillEvaluationContext, *, density_mode: bool
) -> bool:
    h, w = ctx.height, ctx.width
    r0, r1 = (
        int(round(a.get("min_row_ratio", 0.0) * h)),
        int(round(a.get("max_row_ratio", 1.0) * h)),
    )
    c0, c1 = (
        int(round(a.get("min_col_ratio", 0.0) * w)),
        int(round(a.get("max_col_ratio", 1.0) * w)),
    )
    if r0 >= r1 or c0 >= c1 or (sub := ctx.grid[r0:r1, c0:c1]).size == 0:
        return False
    if density_mode:
        non_bg = int(np.sum(sub != ctx.background_color))
        return non_bg >= int(a.get("min_count", 1)) and (
            a.get("max_count") is None or non_bg <= int(a["max_count"])
        )
    return bool(
        np.all(sub == (ctx.background_color if a.get("require_background", True) else sub.flat[0]))
    )


def _eval_causal_do(a: Mapping[str, Any], ctx: SkillEvaluationContext) -> bool:
    if not ctx.metadata or (val := ctx.metadata.get(str(a["variable"]))) is None:
        return False
    exp, op = a.get("expected_value"), str(a.get("operator", "=="))
    if op == "==":
        return val == exp
    if op == "!=":
        return val != exp
    if op in (">", "<") and exp is not None:
        return float(val) > float(exp) if op == ">" else float(val) < float(exp)
    return val in exp if op == "in" else False


_SYM_KEYS = ("axis", "max_deviation", "min_entities", "min_area", "max_area", "ignore_top_colors")


class SymbolicExpr(BaseModel):
    """Serializable, inspectable symbolic predicate expression AST backed by Layer-0 ALUs."""

    op: str
    args: dict[str, Any] = Field(default_factory=dict)
    children: list[SymbolicExpr] = Field(default_factory=list)
    alu_syscall: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SymbolicExpr:
        return cls.model_validate(dict(data))

    def evaluate(self, ctx: SkillEvaluationContext | Any) -> bool:
        """Evaluate the AST against a lifted `SkillEvaluationContext` using Layer-0 ALUs."""
        op, a, ch = self.op, self.args, self.children
        if op == "all_of":
            return all(c.evaluate(ctx) for c in ch)
        if op == "any_of":
            return any(c.evaluate(ctx) for c in ch)
        if op == "not":
            return not ch[0].evaluate(ctx) if ch else True
        if op == "affordance":
            return _eval_affordance(a, ctx)
        if op == "grid_dim":
            return _eval_grid_dim(a, ctx)
        if op in ("entity_bbox", "entity_count"):
            return _eval_entity_match(a, ctx, bbox_mode=op == "entity_bbox")
        if op == "color_dist":
            hist = ctx.color_histogram
            d, mn, mx = len(hist), a.get("min_distinct_colors"), a.get("max_distinct_colors")
            return (
                (mn is None or d >= mn)
                and (mx is None or d <= mx)
                and set(a.get("present_colors") or ()).issubset(hist)
                and set(a.get("absent_colors") or ()).isdisjoint(hist)
            )
        if op == "cut_set":
            cuts = compute_topology_cuts(ctx.grid, bg_color=ctx.background_color)
            return int(cuts["partition_count"]) >= int(a.get("min_partitions", 2))
        if op in ("homogeneous_region", "pixel_density"):
            return _eval_region_or_density(a, ctx, density_mode=op == "pixel_density")
        if op == "symmetry":
            kw = {k: a[k] for k in _SYM_KEYS if k in a}
            return bool(
                compute_symmetry_score(ctx.grid, bg_color=ctx.background_color, **kw)["symmetric"]
            )
        if op == "metadata":
            return bool(ctx.metadata) and ctx.metadata.get(str(a["key"])) == a.get("expected_value")
        if op == "causal_do":
            return _eval_causal_do(a, ctx)
        if self.op == "predictive_confidence":
            if not ctx.metadata:
                return True
            fe, conf = (
                ctx.metadata.get("total_free_energy"),
                ctx.metadata.get("calibrated_confidence"),
            )
            return (fe is None or float(fe) <= float(a.get("max_free_energy", 1.0))) and (
                conf is None or float(conf) >= float(a.get("min_confidence", 0.5))
            )
        raise ValueError(f"Unknown SymbolicExpr op: {self.op!r}")


class SymbolicPredicate(ABC):
    """Base contract for declarative predicates (serializable via `to_expr()`)."""

    @abstractmethod
    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        """Evaluate the predicate against the lifted cognitive context."""

    def to_expr(self) -> SymbolicExpr:
        """Return serializable `SymbolicExpr` AST for graph persistence."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement SymbolicExpr conversion"
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_expr().to_dict()

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> ExprPredicate:
        return ExprPredicate(SymbolicExpr.from_dict(data))

    def __and__(self, other: SymbolicPredicate) -> AllOf:
        return AllOf(self, other)

    def __or__(self, other: SymbolicPredicate) -> AnyOf:
        return AnyOf(self, other)

    def __invert__(self) -> Not:
        return Not(self)


def _ser(val: Any) -> Any:
    if isinstance(val, set):
        return sorted(val) if all(isinstance(x, int) for x in val) else sorted(map(str, val))
    return list(val) if isinstance(val, tuple) else val


class _ExprBackedPredicate(SymbolicPredicate):
    _op: ClassVar[str] = "all_of"
    _alu: ClassVar[str | None] = None

    def to_expr(self) -> SymbolicExpr:
        return SymbolicExpr(
            op=self._op, alu_syscall=self._alu, args={k: _ser(v) for k, v in vars(self).items()}
        )

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        return self.to_expr().evaluate(ctx)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in vars(self).items())})"


class ExprPredicate(_ExprBackedPredicate):
    """Concrete predicate wrapping any deserialized `SymbolicExpr` AST."""

    def __init__(self, expr: SymbolicExpr) -> None:
        self.expr = expr

    def to_expr(self) -> SymbolicExpr:
        return self.expr


def _child_expr(p: SymbolicPredicate) -> SymbolicExpr:
    try:
        return p.to_expr()
    except NotImplementedError:
        return SymbolicExpr(op="all_of")


class _NaryPredicate(_ExprBackedPredicate):
    def __init__(self, *predicates: SymbolicPredicate) -> None:
        self.predicates: list[SymbolicPredicate] = list(predicates)

    def to_expr(self) -> SymbolicExpr:
        return SymbolicExpr(op=self._op, children=[_child_expr(p) for p in self.predicates])


class AllOf(_NaryPredicate):
    """Logical conjunction (AND) of multiple symbolic predicates."""

    _op: ClassVar[str] = "all_of"

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        return all(p.evaluate(ctx) for p in self.predicates)


class AnyOf(_NaryPredicate):
    """Logical disjunction (OR) of multiple symbolic predicates."""

    _op: ClassVar[str] = "any_of"

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        return any(p.evaluate(ctx) for p in self.predicates)


class Not(_ExprBackedPredicate):
    """Logical negation (NOT) of a symbolic predicate."""

    def __init__(self, predicate: SymbolicPredicate) -> None:
        self.predicate: SymbolicPredicate = predicate

    def evaluate(self, ctx: SkillEvaluationContext) -> bool:
        return not self.predicate.evaluate(ctx)

    def to_expr(self) -> SymbolicExpr:
        return SymbolicExpr(op="not", children=[_child_expr(self.predicate)])


@dataclass(eq=False, repr=False)
class ActionAffordancePredicate(_ExprBackedPredicate):
    """Evaluates whether available actions satisfy specified affordance constraints."""

    _op: ClassVar[str] = "affordance"
    required: Any = None
    forbidden: Any = None
    exact: Any = None

    def __post_init__(self) -> None:
        self.required = set(self.required) if self.required is not None else set()
        self.forbidden = set(self.forbidden) if self.forbidden is not None else set()
        self.exact = set(self.exact) if self.exact is not None else None

    def evaluate(self, ctx: SkillEvaluationContext | Any) -> bool:
        return _eval_affordance(
            {"required": self.required, "forbidden": self.forbidden, "exact": self.exact}, ctx
        )


@dataclass(eq=False, repr=False)
class GridDimensionPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "grid_dim"
    min_height: int | None = None
    max_height: int | None = None
    min_width: int | None = None
    max_width: int | None = None
    exact_shape: tuple[int, int] | None = None


@dataclass(eq=False, repr=False)
class EntityBoundingBoxPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "entity_bbox"
    _alu: ClassVar[str | None] = "math.connected_components"
    min_width: int | None = None
    max_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None
    min_count: int = 1
    color: int | None = None
    connectivity: int = 4


@dataclass(eq=False, repr=False)
class EntityCountPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "entity_count"
    _alu: ClassVar[str | None] = "math.connected_components"
    min_count: int = 1
    max_count: int | None = None
    color: int | None = None
    is_frame: bool | None = None
    min_area: int | None = None
    max_area: int | None = None
    connectivity: int = 4


@dataclass(eq=False, repr=False)
class ColorDistributionPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "color_dist"
    present_colors: Any = None
    absent_colors: Any = None
    min_distinct_colors: int | None = None
    max_distinct_colors: int | None = None

    def __post_init__(self) -> None:
        self.present_colors = set(self.present_colors) if self.present_colors is not None else set()
        self.absent_colors = set(self.absent_colors) if self.absent_colors is not None else set()


@dataclass(eq=False, repr=False)
class CutSetPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "cut_set"
    _alu: ClassVar[str | None] = "math.topology_cuts"
    min_partitions: int = 2


@dataclass(eq=False, repr=False)
class HomogeneousRegionPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "homogeneous_region"
    min_row_ratio: float = 0.0
    max_row_ratio: float = 1.0
    min_col_ratio: float = 0.0
    max_col_ratio: float = 1.0
    require_background: bool = True


@dataclass(eq=False, repr=False)
class SymmetryPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "symmetry"
    _alu: ClassVar[str | None] = "math.symmetry_score"
    axis: str = "vertical"
    max_deviation: float = 2.0
    min_entities: int = 2
    min_area: int = 9
    max_area: int = 36
    ignore_top_colors: int = 3


@dataclass(eq=False, repr=False)
class MetadataPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "metadata"
    key: str = ""
    expected_value: Any = True


@dataclass(eq=False, repr=False)
class PearlCausalDoPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "causal_do"
    variable: str = ""
    expected_value: Any = None
    operator: str = "=="


@dataclass(eq=False, repr=False)
class PredictiveConfidencePredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "predictive_confidence"
    max_free_energy: float = 1.0
    min_confidence: float = 0.5


@dataclass(eq=False, repr=False)
class PixelDensityPredicate(_ExprBackedPredicate):
    _op: ClassVar[str] = "pixel_density"
    min_count: int = 1
    max_count: int | None = None
    min_row_ratio: float = 0.0
    max_row_ratio: float = 1.0
    min_col_ratio: float = 0.0
    max_col_ratio: float = 1.0


def _subgoal_instructions(
    sg: SymbolicSubgoal, author: str, *, include_assert: bool, fallback_target: str | None = None
) -> list[Instruction]:
    def _ins(op: Opcode, cost: int, **params: Any) -> Instruction:
        return Instruction(opcode=op, params=params, cost_estimate=cost)

    ins: list[Instruction] = []
    if sg.target_query:
        ins.append(_ins(Opcode.UNIFY, 10, pattern=dict(sg.target_query), bind="$r0"))
    if include_assert:
        node_data = {
            "id": f"sg_{author}_{sg.intent.value.lower()}",
            "node_type": "goal",
            "description": sg.intent.value,
            "properties": {
                "intent": sg.intent.value,
                "target_query": dict(sg.target_query),
                "timeout": sg.timeout_steps,
            },
        }
        ins.append(_ins(Opcode.ASSERT, 5, node_data=node_data, author=author, bind="$r1"))
    target = "$r0" if sg.target_query else fallback_target
    ins.append(
        _ins(
            Opcode.EXECUTE,
            25,
            capability=f"motor.{sg.intent.value.lower()}",
            params={"target": target, **dict(sg.parameters)},
        )
    )
    return ins


@dataclass
class SymbolicSubgoal:
    """Declarative subgoal compiling to pure allowlisted ISA v2 bytecode."""

    intent: SpatialActionIntent
    target_query: dict[str, Any] = field(default_factory=dict)
    precondition: SymbolicPredicate | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_steps: int = 50

    def compile_to_instructions(self, author: str = "declarative_skill") -> list[Instruction]:
        """Compile this subgoal into allowlisted ISA v2 instructions (`UNIFY -> ASSERT -> EXECUTE`)."""
        return _subgoal_instructions(self, author, include_assert=True)


class SubgoalSequence:
    """An ordered sequence of declarative subgoals representing a skill program."""

    def __init__(self, *subgoals: SymbolicSubgoal) -> None:
        self.subgoals: list[SymbolicSubgoal] = list(subgoals)

    def __iter__(self):
        return iter(self.subgoals)

    def __len__(self) -> int:
        return len(self.subgoals)

    def __getitem__(self, index: int) -> SymbolicSubgoal:
        return self.subgoals[index]

    def compile_to_stream(self, skill_name: str = "declarative_skill") -> InstructionStream:
        """Compile all subgoals in the sequence into an HCIR InstructionStream."""
        return InstructionStream(
            author=skill_name,
            description=f"Compiled HCIR program for {skill_name}",
            instructions=[
                ins for sg in self.subgoals for ins in sg.compile_to_instructions(author=skill_name)
            ],
        )


class DeclarativeNeuroSymbolicSkill(BaseHierarchicalSkill):
    """Declarative skill compiling to homoiconic `SkillNode` graphs and ISA v2 bytecode."""

    name: str = ""
    description: str = ""
    signature: SymbolicPredicate = AllOf()
    program: SubgoalSequence | None = None
    _compiled_stream: InstructionStream | None = None

    def get_metadata(self) -> dict[str, Any]:
        return {}

    def can_handle(
        self, grid: np.ndarray, available_actions: list[int], metadata: dict[str, Any] | None = None
    ) -> bool:
        ctx = SkillEvaluationContext(
            grid=grid,
            available_actions=available_actions,
            metadata={**self.get_metadata(), **(metadata or {})},
        )
        try:
            return bool(self.signature.evaluate(ctx))
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Declarative skill '%s' evaluation error: %s", self.skill_name, exc)
            return False

    def compile_to_instruction_stream(
        self, grid: np.ndarray, available_actions: list[int], metadata: dict[str, Any] | None = None
    ) -> InstructionStream:
        """Compile this declarative skill into an executable HCIR InstructionStream."""
        if self._compiled_stream is not None:
            return self._compiled_stream.model_copy(deep=True)
        return (
            self.program.compile_to_stream(skill_name=self.skill_name)
            if self.program is not None
            else InstructionStream(
                author=self.skill_name,
                description=f"Empty instruction stream for {self.skill_name}",
            )
        )

    def to_skill_node(
        self,
        *,
        skill_id: str | None = None,
        tenant_id: str = "default",
        alpha_success: float = 2.0,
        beta_failure: float = 1.0,
    ) -> SkillNode:
        """Export this skill as a parameterized homoiconic `SkillNode` executable via `Opcode.CALL`."""
        params_meta: list[dict[str, Any]] = []
        instructions: list[Instruction] = []
        for idx, sg in enumerate(list(self.program.subgoals) if self.program is not None else []):
            pname = f"target_{idx}"
            if sg.target_query:
                params_meta.append(
                    {
                        "name": pname,
                        "role": str(sg.target_query.get("role", "")),
                        "required": False,
                        "default": dict(sg.target_query),
                    }
                )
            instructions.extend(
                _subgoal_instructions(
                    sg, self.skill_name, include_assert=False, fallback_target=f"?{pname}"
                )
            )
        if not instructions:
            instructions.append(
                Instruction(
                    opcode=Opcode.UNIFY,
                    params={"pattern": {"role": "?target"}, "bind": "$r0"},
                    cost_estimate=5,
                )
            )
        instructions.append(Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}))
        stream = InstructionStream(
            author=self.skill_name,
            description=self.description or self.skill_name,
            instructions=instructions,
        )
        try:
            pre_pat: dict[str, Any] = {"symbolic_expr": self.signature.to_dict()}
        except NotImplementedError:
            pre_pat = {"signature": repr(self.signature)}
        return SkillNode(
            id=skill_id or f"skill_{self.skill_name}",
            skill_name=self.skill_name,
            description=self.description,
            parameters=params_meta,
            precondition_pattern=pre_pat,
            postcondition_pattern={
                "intent": self.semantic_intent.value.lower(),
                "skill_name": self.skill_name,
            },
            bytecode_stream=stream.model_dump(),
            alpha_success=alpha_success,
            beta_failure=beta_failure,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by="declarative_skill"),
            scope=Scope(tenant_id=tenant_id),
            tags=["declarative_skill", self.skill_name],
        )

    @classmethod
    def from_skill_node(cls, node: SkillNode) -> DeclarativeNeuroSymbolicSkill:
        """Reconstruct a runnable declarative skill from a graph `SkillNode` (zero subclassing)."""
        skill = cls()
        skill.skill_name, skill.description = node.skill_name or node.name, node.description
        if (
            isinstance(node.precondition_pattern, Mapping)
            and "symbolic_expr" in node.precondition_pattern
        ):
            skill.signature = SymbolicPredicate.from_dict(
                node.precondition_pattern["symbolic_expr"]
            )
        if node.bytecode_stream is not None:
            skill._compiled_stream = InstructionStream.model_validate(node.bytecode_stream)
        return skill

    def solve(
        self, ctx: SkillEvaluationContext, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]] | list[int]:
        return []

    def plan(
        self, grid: np.ndarray, current_level: int = 0, metadata: dict[str, Any] | None = None
    ) -> list[tuple[int, dict[str, int] | None]] | list[int]:
        return self.solve(
            SkillEvaluationContext(
                grid=grid, available_actions=[1, 2, 3, 4, 5, 6, 7], metadata=metadata
            ),
            current_level=current_level,
        )


class CompositeDeclarativeSkill(DeclarativeNeuroSymbolicSkill):
    """Hierarchically composed macro-skill sequencing multiple declarative sub-skills."""

    def __init__(
        self,
        skills: Sequence[DeclarativeNeuroSymbolicSkill],
        skill_name: str = "composite_macro_skill",
        description: str = "",
    ) -> None:
        self.sub_skills = list(skills)
        self.skill_name = skill_name
        self.description = description or f"Macro-skill composed of {len(skills)} sub-skills"
        sigs = [s.signature for s in skills if getattr(s, "signature", None) is not None]
        self.signature = AllOf(*sigs) if sigs else AllOf()
        all_sg = [
            sg for s in skills for sg in (s.program.subgoals if s.program is not None else [])
        ]
        self.program = SubgoalSequence(*all_sg) if all_sg else None

    def solve(
        self, ctx: SkillEvaluationContext, current_level: int = 0
    ) -> list[tuple[int, dict[str, int] | None]] | list[int]:
        actions: list[Any] = []
        for s in self.sub_skills:
            acts = s.solve(ctx, current_level=current_level)
            if acts:
                actions.extend(acts)
        return actions
