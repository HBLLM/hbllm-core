"""Unit and Hypothesis property tests for the canonical `.hcir` codec and BLAKE3 digests."""

from __future__ import annotations

from typing import Any

import hbllm_hcir_graph
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.fuzzy import FuzzyExpr, RelationCompositionPolicy
from hbllm.hcir.graph import (
    ConceptNode,
    GoalNode,
    HCIREdge,
    LearnedRuleNode,
    NodeLifecycle,
    ProcedureNode,
    SkillNode,
)
from hbllm.hcir.nodes import AdaptiveMembershipNode
from hbllm.hcir.registers import REGISTER_NAMES
from hbllm.hcir.seed.loader import build_seeded_constructions, load_core_priors_spec
from hbllm.hcir.sexpr import (
    HCIRParseError,
    _py_blake3_hex,
    blake3_hex,
    canonical_digest,
    exact_digest,
    format_hcir,
    parse_hcir,
    semantic_digest,
)
from hbllm.hcir.skills.declarative import SymbolicExpr
from hbllm.hcir.types import Provenance
from hbllm.hcir.unification import EdgePattern, GraphPattern, NodePattern


def _assert_full_codec_laws(obj: Any) -> None:
    """Verify exact round-trip, canonical idempotence, and semantic digest stability."""
    exact_txt = format_hcir(obj, mode="exact")
    exact_rt = parse_hcir(exact_txt)
    assert exact_rt == obj
    assert exact_rt.exact_digest() == exact_digest(obj)

    canon_txt_1 = format_hcir(obj, mode="canonical")
    canon_rt = parse_hcir(canon_txt_1)
    canon_txt_2 = format_hcir(canon_rt, mode="canonical")
    assert canon_txt_1 == canon_txt_2
    assert canon_rt.semantic_digest() == semantic_digest(obj)
    assert exact_rt.semantic_digest() == semantic_digest(obj)


# ═══════════════════════════════════════════════════════════════════════════
# 1. BLAKE3 Parity (Rust vs. Pure-Python Fallback)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"hello",
        b"x" * 64,
        b"x" * 65,
        b"y" * 1024,
        b"y" * 1025,
        bytes(range(256)) * 17,
    ],
)
def test_blake3_native_python_parity(payload: bytes) -> None:
    native_hex = str(hbllm_hcir_graph.blake3_hash(payload))
    py_hex = _py_blake3_hex(payload)
    assert py_hex == native_hex
    assert blake3_hex(payload) == native_hex
    assert len(py_hex) == 64


# ═══════════════════════════════════════════════════════════════════════════
# 2. Rule 0 & Lexer Diagnostics
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "bad_src",
    [
        "(defmacro foo (x) x)",
        "(eval (quote (1 2)))",
        "(lambda (x) x)",
        "(quote (a b))",
        "(if (empty? $r0) (return))",
        "(hcir :version 2 :type stream (stream))",
        "(stream (execute $r8))",
        "(stream (unify (pattern (nodes (node ?)))))",
        '(stream (execute "unterminated))',
        "(stream (return)",
        "",
        "(hcir :version 1 :type skill (stream))",
    ],
)
def test_rule0_and_lexer_rejections(bad_src: str) -> None:
    with pytest.raises(HCIRParseError) as exc_info:
        parse_hcir(bad_src)
    assert exc_info.value.line >= 1
    assert exc_info.value.col >= 1


def test_nan_rejected_on_format_and_parse() -> None:
    with pytest.raises(HCIRParseError):
        parse_hcir("(const nan)")
    bad_fe = FuzzyExpr(op="const", args={"value": float("nan")})
    with pytest.raises(ValueError, match="NaN"):
        format_hcir(bad_fe)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Exhaustive Constructs & Repository Seed Round-Trips
# ═══════════════════════════════════════════════════════════════════════════


def test_all_12_opcodes_and_15_branch_conditions_roundtrip() -> None:
    conditions = [
        {"condition": "unconditional"},
        {"condition": "if_empty", "register": "$r0"},
        {"condition": "if_not_empty", "var": "?items"},
        {"condition": "if_true"},
        {"condition": "if_false", "register": "$r1"},
        {"condition": "if_equals", "register": "$r2", "value": "done"},
        {"condition": "if_not_equals", "var": "?state", "value": 0},
        {"condition": "if_lt", "register": "$r3", "threshold": 5.0},
        {"condition": "if_lte", "var": "?score", "value": 10},
        {"condition": "if_gt", "register": "$r4", "threshold": 0.25},
        {"condition": "if_gte", "threshold": 1.0},
        {"condition": "if_confidence_lt", "register": "$r5", "threshold": 0.4},
        {"condition": "if_surprise_gt", "threshold": 0.75},
        {"condition": "if_degree_gte", "register": "$r6", "threshold": 0.6},
        {"condition": "if_degree_lt", "var": "?fit", "threshold": 0.3},
    ]
    instructions: list[Instruction] = [
        Instruction(
            opcode=Opcode.ASSERT,
            params={
                "node_data": {"id": "n1", "node_type": "goal"},
                "author": "test",
                "bind": "$r0",
            },
            label="start",
            cost_estimate=5,
        ),
        Instruction(opcode=Opcode.RETRACT, params={"node_id": "n1", "author": "test"}),
        Instruction(
            opcode=Opcode.QUERY,
            params={"node_type": "entity", "min_confidence": 0.5, "limit": 10, "bind": "$r1"},
        ),
        Instruction(
            opcode=Opcode.EXECUTE,
            params={
                "capability": "motor.move",
                "params": {"dx": 1, "flag": True, "label": "true", "empty": []},
                "bind": "$r2",
            },
            cost_estimate=20,
        ),
        Instruction(
            opcode=Opcode.WAIT, params={"event_type": "perception.stabilized", "timeout_ms": 250}
        ),
        Instruction(opcode=Opcode.FORK, params={"branch_name": "sim_cf"}),
        Instruction(
            opcode=Opcode.UNIFY,
            params={
                "pattern": GraphPattern(
                    nodes=[
                        NodePattern(
                            var="?agent",
                            node_type="physical_entity",
                            fuzzy_memberships={"near_goal": 0.7},
                            numeric_tolerances={"dist": {"target": 0.0, "sigma": 1.5}},
                        )
                    ],
                    edges=[
                        EdgePattern(
                            source="?agent",
                            edge_type="located_at",
                            target="?dest",
                            min_weight=0.5,
                            min_semantic_degree=0.6,
                        )
                    ],
                ).model_dump(),
                "fuzzy": True,
                "profile_id": "product",
                "min_degree": 0.5,
                "branch_name": "sim_cf",
                "bind": {"__semantic_degree__": "$r6"},
                "bind_all": "$r7",
            },
        ),
        *[
            Instruction(opcode=Opcode.BRANCH, params={**c, "target_label": "done"})
            for c in conditions
        ],
        Instruction(
            opcode=Opcode.CALL,
            params={
                "goal_pattern": {"intent": "choose_route", "target": "$r0"},
                "args": ["$r0", "?dest"],
                "fuzzy": True,
                "bind": "$r3",
            },
        ),
        Instruction(opcode=Opcode.MERGE, params={"branch_name": "sim_cf", "author": "sim"}),
        Instruction(opcode=Opcode.ROLLBACK, params={"branch_name": "sim_cf", "reason": "low_fit"}),
        Instruction(opcode=Opcode.RETURN, params={"yield": ["$r2", "$r3"]}, label="done"),
        Instruction(opcode=Opcode.RETURN, params={"value": {"ok": True, "count": 1}}),
        Instruction(opcode=Opcode.RETURN, params={}),
    ]
    stream = InstructionStream(
        author="unit_test",
        description="Exhaustive 12-opcode test",
        instructions=instructions,
    )
    _assert_full_codec_laws(stream)
    rt_canon = parse_hcir(format_hcir(stream, mode="canonical"))
    assert rt_canon.semantic_hash() == stream.semantic_hash()


def test_all_symbolic_expr_and_fuzzy_expr_ops_roundtrip() -> None:
    fe = FuzzyExpr(
        op="alpha_cut",
        args={"alpha": 0.65, "strict": True},
        children=[
            FuzzyExpr(
                op="implies",
                children=[
                    FuzzyExpr(
                        op="and",
                        children=[
                            FuzzyExpr(op="const", args={"value": 0.9}),
                            FuzzyExpr(op="var", args={"name": "speed", "default": 0.2}),
                            FuzzyExpr(
                                op="hedge",
                                args={"hedge": "very"},
                                children=[
                                    FuzzyExpr(
                                        op="membership",
                                        args={
                                            "var": "distance",
                                            "kernel_family": "gaussian",
                                            "reference_scale_kind": "entity_extent",
                                            "parameters": {"center": 0.0, "sigma": 1.25},
                                        },
                                    )
                                ],
                            ),
                        ],
                    ),
                    FuzzyExpr(
                        op="equiv",
                        children=[
                            FuzzyExpr(
                                op="or", children=[FuzzyExpr(op="const", args={"value": 0.4})]
                            ),
                            FuzzyExpr(
                                op="not", children=[FuzzyExpr(op="const", args={"value": 0.1})]
                            ),
                        ],
                    ),
                ],
            )
        ],
    )
    _assert_full_codec_laws(fe)

    se = SymbolicExpr(
        op="all_of",
        alu_syscall="math.composite",
        children=[
            SymbolicExpr(
                op="any_of",
                children=[
                    SymbolicExpr(
                        op="not", children=[SymbolicExpr(op="cut_set", args={"min_partitions": 2})]
                    )
                ],
            ),
            SymbolicExpr(op="affordance", args={"required": [1, 2], "forbidden": [5]}),
            SymbolicExpr(op="grid_dim", args={"min_height": 4, "max_width": 32}),
            SymbolicExpr(op="entity_bbox", args={"color": 2, "min_count": 1}),
            SymbolicExpr(op="entity_count", args={"min_count": 2, "max_count": 5}),
            SymbolicExpr(
                op="color_dist", args={"min_distinct_colors": 2, "present_colors": [1, 2]}
            ),
            SymbolicExpr(op="homogeneous_region", args={"require_background": True}),
            SymbolicExpr(op="pixel_density", args={"min_count": 3}),
            SymbolicExpr(op="symmetry", args={"axis": "vertical", "max_deviation": 1.5}),
            SymbolicExpr(op="metadata", args={"key": "phase", "expected_value": "active"}),
            SymbolicExpr(
                op="causal_do", args={"variable": "door", "operator": "==", "expected_value": True}
            ),
            SymbolicExpr(
                op="fuzzy_expr",
                args={"expr": fe.model_dump(), "profile_id": "lukasiewicz", "alpha": 0.5},
            ),
            SymbolicExpr(
                op="predictive_confidence", args={"max_free_energy": 0.8, "min_confidence": 0.7}
            ),
            SymbolicExpr(op="custom_probe", args={"k": "v"}, alu_syscall="math.custom"),
        ],
    )
    _assert_full_codec_laws(se)


def test_cognitive_nodes_and_repo_seeds_roundtrip() -> None:
    proc = ProcedureNode(
        id="proc_1",
        procedure_name="open_and_enter",
        description="Open door then move",
        steps=["open", "enter"],
        is_atomic=False,
        parameters=[{"name": "door", "required": True}],
        bytecode_stream=InstructionStream(
            instructions=[Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]})]
        ).model_dump(),
    )
    amm = AdaptiveMembershipNode(
        id="amm_near",
        model_id="near_v1",
        target_id="near",
        contextual_scope="navigation",
        kernel_family="gaussian",
        reference_scale_kind="navigable_diameter",
        learned_parameters={"center": 0.0, "sigma": 0.35},
        sample_count=12,
        version=2,
        parent_version=1,
        parent_hash="abcd1234",
        version_snapshots={"1": {"center": 0.0, "sigma": 0.5}, "2": {"center": 0.0, "sigma": 0.35}},
    )
    goal = GoalNode(id="g_extra", description="Reach exit", priority=0.9)
    _assert_full_codec_laws(proc)
    _assert_full_codec_laws(amm)
    _assert_full_codec_laws(goal)

    spec = load_core_priors_spec()
    for c in spec["concepts"]:
        _assert_full_codec_laws(ConceptNode(**c))
    for r in spec["rules"]:
        _assert_full_codec_laws(LearnedRuleNode(**r))
    for s in spec["skills"]:
        _assert_full_codec_laws(SkillNode(**s))
    for e in spec["edges"]:
        _assert_full_codec_laws(HCIREdge(**e))
    _assert_full_codec_laws(
        RelationCompositionPolicy.model_validate(spec["relation_composition_policy"])
    )

    for lang in ("en", "si", "ta"):
        for cxn in build_seeded_constructions(lang):
            _assert_full_codec_laws(cxn)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Canonical vs. Exact Identity Proof
# ═══════════════════════════════════════════════════════════════════════════


def test_canonical_vs_exact_digest_separation() -> None:
    stream = InstructionStream(
        author="episode_learner",
        instructions=[
            Instruction(
                opcode=Opcode.UNIFY,
                params={
                    "pattern": {"nodes": [{"var": "?door", "node_type": "physical_entity"}]},
                    "bind": "$r0",
                },
            ),
            Instruction(
                opcode=Opcode.EXECUTE,
                params={"capability": "motor.interact", "params": {"target": "$r0"}, "bind": "$r1"},
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r1"]}),
        ],
    )
    skill_a = SkillNode(
        id="skill_gen_001",
        skill_name="unlock_portal",
        description="Unlock portal with key",
        parameters=[{"name": "door", "required": True}],
        precondition_pattern={"nodes": [{"var": "?door", "node_type": "physical_entity"}]},
        postcondition_pattern={"intent": "unlock", "target": "?door"},
        bytecode_stream=stream.model_dump(),
        alpha_success=4.0,
        beta_failure=1.0,
        invocation_count=1,
        success_rate=0.5,
        provenance=Provenance(created_by="ep_1", timestamp=100.0),
    )
    skill_b = SkillNode(
        id="skill_gen_999",
        skill_name="unlock_portal",
        description="Unlock portal with key",
        parameters=[{"name": "door", "required": True}],
        precondition_pattern=GraphPattern(
            nodes=[NodePattern(var="?door", node_type="physical_entity")]
        ).model_dump(),
        postcondition_pattern={"intent": "unlock", "target": "?door"},
        bytecode_stream=InstructionStream(
            author="different_author",
            description="different description",
            instructions=[
                ins.model_copy(update={"id": f"ins_other_{idx}"})
                for idx, ins in enumerate(stream.instructions)
            ],
        ).model_dump(),
        alpha_success=4.0,
        beta_failure=1.0,
        invocation_count=42,
        success_rate=0.95,
        provenance=Provenance(created_by="ep_2", timestamp=9999.0),
        lifecycle=NodeLifecycle.VALIDATED,
    )

    assert skill_a.exact_digest() != skill_b.exact_digest()
    assert skill_a.semantic_digest() == skill_b.semantic_digest()
    assert canonical_digest(skill_a) == canonical_digest(skill_b)

    rt_a = parse_hcir(format_hcir(skill_a, mode="canonical"))
    assert rt_a.semantic_digest() == skill_a.semantic_digest()
    assert parse_hcir(format_hcir(skill_a, mode="exact")) == skill_a


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hypothesis Property-Based Round-Trip & Canonicalization Suite
# ═══════════════════════════════════════════════════════════════════════════

_IDENT_ST = st.from_regex(r"[a-z][a-z0-9_]{1,12}", fullmatch=True)
_VAR_ST = st.from_regex(r"\?[a-z][a-z0-9_]{0,8}", fullmatch=True)
_REG_ST = st.sampled_from(sorted(REGISTER_NAMES))
_SAFE_FLOAT_ST = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
_UNIT_FLOAT_ST = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_SCALAR_ST = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    _SAFE_FLOAT_ST,
    st.text(alphabet='abcxyz012_-:?$ \n"', min_size=0, max_size=12),
    _REG_ST,
    _VAR_ST,
)
_VALUE_ST = st.recursive(
    _SCALAR_ST,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(_IDENT_ST, children, max_size=3),
    ),
    max_leaves=8,
)


@st.composite
def _fuzzy_expr_st(draw: st.DrawFn) -> FuzzyExpr:
    return draw(
        st.recursive(
            st.one_of(
                _UNIT_FLOAT_ST.map(lambda v: FuzzyExpr(op="const", args={"value": v})),
                _IDENT_ST.flatmap(
                    lambda n: _UNIT_FLOAT_ST.map(
                        lambda d: FuzzyExpr(op="var", args={"name": n, "default": d})
                    )
                ),
                st.tuples(_IDENT_ST, _UNIT_FLOAT_ST).map(
                    lambda nv: FuzzyExpr(
                        op="membership",
                        args={
                            "var": nv[0],
                            "value": nv[1],
                            "kernel_family": "gaussian",
                            "reference_scale_kind": "raw",
                            "parameters": {"center": 0.5, "sigma": 0.5},
                        },
                    )
                ),
            ),
            lambda ch: st.one_of(
                st.lists(ch, min_size=1, max_size=3).map(
                    lambda cs: FuzzyExpr(op="and", children=cs)
                ),
                st.lists(ch, min_size=1, max_size=3).map(
                    lambda cs: FuzzyExpr(op="or", children=cs)
                ),
                ch.map(lambda c: FuzzyExpr(op="not", children=[c])),
                st.tuples(ch, ch).map(lambda ab: FuzzyExpr(op="implies", children=[ab[0], ab[1]])),
                ch.map(lambda c: FuzzyExpr(op="hedge", args={"hedge": "very"}, children=[c])),
                st.tuples(_UNIT_FLOAT_ST, st.booleans(), ch).map(
                    lambda a: FuzzyExpr(
                        op="alpha_cut", args={"alpha": a[0], "strict": a[1]}, children=[a[2]]
                    )
                ),
            ),
            max_leaves=6,
        )
    )


@st.composite
def _graph_pattern_st(draw: st.DrawFn) -> GraphPattern:
    n_nodes = draw(st.integers(min_value=1, max_value=3))
    nodes = [
        NodePattern(
            var=f"?n{i}",
            node_type=draw(st.one_of(st.none(), _IDENT_ST)),
            min_confidence=draw(st.one_of(st.none(), _UNIT_FLOAT_ST)),
            tags=draw(st.lists(_IDENT_ST, max_size=2)),
            attributes=draw(st.dictionaries(_IDENT_ST, _VALUE_ST, max_size=2)),
            fuzzy_memberships=draw(st.dictionaries(_IDENT_ST, _UNIT_FLOAT_ST, max_size=2)),
        )
        for i in range(n_nodes)
    ]
    edges = []
    if n_nodes >= 2 and draw(st.booleans()):
        edges.append(
            EdgePattern(
                source="?n0",
                edge_type=draw(
                    st.one_of(st.none(), st.sampled_from(["located_at", "supports", "causes"]))
                ),
                target="?n1",
                min_weight=draw(st.one_of(st.none(), _UNIT_FLOAT_ST)),
                min_semantic_degree=draw(st.one_of(st.none(), _UNIT_FLOAT_ST)),
            )
        )
    return GraphPattern(
        nodes=nodes,
        edges=edges,
        distinct_nodes=draw(st.booleans()),
    )


@st.composite
def _instruction_st(draw: st.DrawFn) -> Instruction:
    op = draw(st.sampled_from(list(Opcode)))
    label = draw(st.one_of(st.none(), _IDENT_ST))
    cost = draw(st.integers(min_value=0, max_value=100))
    if op == Opcode.EXECUTE:
        params: dict[str, Any] = {
            "capability": draw(st.sampled_from(["motor.move", "math.fuzzy.eval"])),
            "params": draw(st.dictionaries(_IDENT_ST, _VALUE_ST, max_size=2)),
            "bind": draw(_REG_ST),
        }
    elif op == Opcode.UNIFY:
        params = {
            "pattern": draw(_graph_pattern_st()).model_dump(),
            "fuzzy": draw(st.booleans()),
            "min_degree": draw(_UNIT_FLOAT_ST),
            "bind": draw(_REG_ST),
        }
    elif op == Opcode.BRANCH:
        params = {
            "condition": draw(
                st.sampled_from(["if_empty", "if_not_empty", "if_degree_gte", "if_degree_lt"])
            ),
            "register": draw(_REG_ST),
            "threshold": draw(_UNIT_FLOAT_ST),
            "target_label": draw(_IDENT_ST),
        }
    elif op == Opcode.CALL:
        params = {
            "goal_pattern": {"intent": draw(_IDENT_ST), "target": draw(_VAR_ST)},
            "args": draw(st.lists(_REG_ST, max_size=2)),
            "bind": draw(_REG_ST),
        }
    elif op == Opcode.RETURN:
        params = draw(
            st.one_of(
                st.just({}),
                st.lists(_REG_ST, max_size=2).map(lambda r: {"yield": r}),
                _SCALAR_ST.map(lambda v: {"value": v}),
            )
        )
    elif op == Opcode.QUERY:
        params = {
            "node_type": "goal",
            "limit": draw(st.integers(min_value=1, max_value=50)),
            "bind": "$r0",
        }
    else:
        params = draw(st.dictionaries(_IDENT_ST, _VALUE_ST, max_size=2))
    return Instruction(opcode=op, params=params, label=label, cost_estimate=cost)


@settings(max_examples=80, deadline=None)
@given(
    fe=_fuzzy_expr_st(),
    gp=_graph_pattern_st(),
    ins_list=st.lists(_instruction_st(), min_size=1, max_size=5),
)
def test_property_roundtrip_canonicalization_and_digests(
    fe: FuzzyExpr, gp: GraphPattern, ins_list: list[Instruction]
) -> None:
    _assert_full_codec_laws(fe)
    _assert_full_codec_laws(gp)

    stream = InstructionStream(author="prop", instructions=ins_list)
    _assert_full_codec_laws(stream)

    se = SymbolicExpr(
        op="all_of",
        children=[
            SymbolicExpr(op="fuzzy_expr", args={"expr": fe.model_dump(), "alpha": 0.5}),
            SymbolicExpr(op="cut_set", args={"min_partitions": 2}),
        ],
    )
    _assert_full_codec_laws(se)

    skill = SkillNode(
        id="prop_skill",
        skill_name="prop_skill",
        parameters=[{"name": "x", "required": True}],
        precondition_pattern=gp.model_dump(),
        postcondition_pattern={"intent": "prop", "target": "?x"},
        bytecode_stream=stream.model_dump(),
        alpha_success=3.0,
        beta_failure=1.0,
    )
    _assert_full_codec_laws(skill)
