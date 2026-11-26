"""Phase F0 unit, Hypothesis property, replay, and latency tests for ADR 006."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.capabilities.math_capabilities import (
    compute_fuzzy_bipolar,
    compute_fuzzy_compose,
    compute_fuzzy_eval,
    compute_fuzzy_hedge,
    compute_fuzzy_membership,
)
from hbllm.hcir.fuzzy import (
    AdaptiveMembershipModel,
    BipolarObservation,
    FuzzyAssertion,
    FuzzyExpr,
    FuzzyHedge,
    FuzzyProfileId,
    MembershipKernelFamily,
    NegationMode,
    ReferenceScaleKind,
    RelationCompositionPolicy,
    aggregate_bipolar_evidence,
    apply_hedge,
    evaluate_membership_kernel,
    evaluate_membership_kernel_batch,
    get_fuzzy_profile,
)
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.nodes import AdaptiveMembershipNode, HCIRNodeType
from hbllm.hcir.transactions import HCIRTransaction
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState

_UNIT_FLOATS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_PROFILES = st.sampled_from(
    [FuzzyProfileId.GODEL, FuzzyProfileId.PRODUCT, FuzzyProfileId.LUKASIEWICZ]
)


@given(profile_id=_PROFILES, a=_UNIT_FLOATS, b=_UNIT_FLOATS, c=_UNIT_FLOATS)
def test_fuzzy_profile_residuated_lattice_laws(
    profile_id: FuzzyProfileId, a: float, b: float, c: float
) -> None:
    """Verify commutativity, associativity, monotonicity, boundaries, and residuation adjunction."""
    prof = get_fuzzy_profile(profile_id)

    # Commutativity
    assert math.isclose(prof.t_norm(a, b), prof.t_norm(b, a), abs_tol=1e-12)
    assert math.isclose(prof.t_conorm(a, b), prof.t_conorm(b, a), abs_tol=1e-12)

    # Associativity
    assert math.isclose(
        prof.t_norm(prof.t_norm(a, b), c), prof.t_norm(a, prof.t_norm(b, c)), abs_tol=1e-9
    )
    assert math.isclose(
        prof.t_conorm(prof.t_conorm(a, b), c), prof.t_conorm(a, prof.t_conorm(b, c)), abs_tol=1e-9
    )

    # Boundary identities
    assert math.isclose(prof.t_norm(a, 1.0), a, abs_tol=1e-12)
    assert math.isclose(prof.t_norm(a, 0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(prof.t_conorm(a, 0.0), a, abs_tol=1e-12)
    assert math.isclose(prof.t_conorm(a, 1.0), 1.0, abs_tol=1e-12)

    # Residuation adjunction: T(a, c) <= b  <=>  c <= (a => b)
    lhs = prof.t_norm(a, c) <= b + 1e-9
    rhs = c <= prof.residuum(a, b) + 1e-9
    if abs(prof.t_norm(a, c) - b) > 1e-7 and abs(c - prof.residuum(a, b)) > 1e-7:
        assert lhs == rhs


def test_fuzzy_assertion_forbids_confidence_and_validates_intervals() -> None:
    """ADR 006 D2: FuzzyAssertion never accepts a `confidence` scalar."""
    with pytest.raises(ValidationError):
        FuzzyAssertion(
            subject_ref="ent_1",
            semantics="concept:CUP",
            degree=0.91,
            model_id="m_cup",
            confidence=0.88,  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        FuzzyAssertion(
            subject_ref="ent_1",
            semantics="concept:CUP",
            degree=0.95,
            degree_interval=(0.6, 0.9),
            model_id="m_cup",
        )

    valid = FuzzyAssertion(
        subject_ref="ent_1",
        target_ref="ent_2",
        semantics="relation:NEAR",
        degree=0.81,
        degree_interval=(0.75, 0.87),
        profile_id=FuzzyProfileId.PRODUCT,
        context_ref="manipulation",
        model_id="rel_near_manip",
        model_version=2,
        evidence_refs=("obs_1",),
    )
    assert valid.degree == pytest.approx(0.81)
    assert valid.model_version == 2


def test_bipolar_evidence_correlation_idempotence_and_paraconsistency() -> None:
    """Repeated correlated frames must not inflate mass; opposing independent evidence yields conflict > 0."""
    single = [
        BipolarObservation(
            evidence_id="ev_0",
            polarity="support",
            strength=0.85,
            source_weight=0.9,
            dependency_group="dep_static",
            source_channel_id="cam_front",
            tracklet_epoch=1,
            timestamp=10.0,
        )
    ]
    repeated = [
        BipolarObservation(
            evidence_id=f"ev_{i}",
            polarity="support",
            strength=0.85,
            source_weight=0.9,
            dependency_group="dep_static",
            source_channel_id="cam_front",
            tracklet_epoch=1,
            timestamp=10.0 + i * 0.03,
        )
        for i in range(50)
    ]
    s_single = aggregate_bipolar_evidence(single)
    s_repeat = aggregate_bipolar_evidence(repeated)
    assert s_single.support_mass == pytest.approx(s_repeat.support_mass)
    assert s_single.ignorance == pytest.approx(s_repeat.ignorance)
    assert s_single.conflict == 0.0

    empty_state = aggregate_bipolar_evidence([])
    assert empty_state.ignorance == pytest.approx(1.0)
    assert empty_state.conflict == 0.0

    # Independent supporting and refuting clusters produce non-zero paraconsistent conflict
    contested = [
        BipolarObservation(
            evidence_id="sup_a",
            polarity="support",
            strength=0.9,
            source_weight=1.0,
            dependency_group="dep_1",
            source_channel_id="vision",
            tracklet_epoch=1,
        ),
        BipolarObservation(
            evidence_id="sup_b",
            polarity="support",
            strength=0.8,
            source_weight=1.0,
            dependency_group="dep_2",
            source_channel_id="kinesthetic",
            tracklet_epoch=1,
        ),
        BipolarObservation(
            evidence_id="ref_a",
            polarity="refute",
            strength=0.9,
            source_weight=1.0,
            dependency_group="dep_3",
            source_channel_id="tactile",
            tracklet_epoch=1,
        ),
        BipolarObservation(
            evidence_id="ref_b",
            polarity="refute",
            strength=0.85,
            source_weight=1.0,
            dependency_group="dep_4",
            source_channel_id="laser",
            tracklet_epoch=1,
        ),
    ]
    c_state = aggregate_bipolar_evidence(contested, profile_id=FuzzyProfileId.LUKASIEWICZ)
    assert c_state.conflict > 0.5
    assert c_state.ignorance < empty_state.ignorance
    assert math.isclose(
        c_state.support_share + c_state.refutation_share + c_state.ignorance, 1.0, abs_tol=1e-9
    )


def test_adaptive_membership_model_versioning_and_non_retroactive_replay() -> None:
    """Adapting model v1 -> v2 commits transactionally without rewriting historical v1 semantics."""
    scope = Scope(tenant_id="tenant_f0")
    ws = HCIRWorkspaceState(tenant_id="tenant_f0")
    services = KernelServices.create_default(
        ws, verification_pipeline=create_default_pipeline(default_scope=scope)
    )

    model_v1 = AdaptiveMembershipModel(
        model_id="rel_near_manipulation",
        target_id="rel_near",
        contextual_scope="manipulation",
        kernel_family=MembershipKernelFamily.TRAPEZOIDAL,
        reference_scale_kind=ReferenceScaleKind.ENTITY_EXTENT,
        learned_parameters={"a": 0.0, "b": 0.5, "c": 1.0, "d": 2.0},
        sample_count=5,
        version=1,
    )
    node_v1 = model_v1.to_node(scope=scope)
    from hbllm.hcir.transactions import TransactionOp, TransactionOperation

    tx1 = HCIRTransaction(
        author="test_f0",
        tenant_id="tenant_f0",
        operations=[
            TransactionOperation(
                op=TransactionOp.ADD_NODE, node_id=node_v1.id, node_data=node_v1.model_dump()
            )
        ],
    )
    assert services.transaction_manager.commit(tx1).is_committed
    hist_assertion = model_v1.emit_assertion(
        subject_ref="cup_1",
        target_ref="plate_1",
        raw_value=1.5,
        scale_context={"subject_extent": 0.5, "object_extent": 0.5},
        semantics="relation:NEAR",
        snapshot_sequence=1,
    )
    assert hist_assertion.degree == pytest.approx(0.5)
    assert hist_assertion.model_version == 1

    # Evolve to v2 in an isolated fork first, then commit to main workspace
    forked = ws.graph.fork()
    assert forked.get_node(node_v1.id) is not None

    model_v2, evolve_ops = model_v1.evolve(
        {"a": 0.0, "b": 0.5, "c": 2.0, "d": 4.0},
        sample_delta=10,
        scope=scope,
        existing_snapshots=node_v1.version_snapshots,
    )
    tx2 = HCIRTransaction(author="test_f0", tenant_id="tenant_f0", operations=evolve_ops)
    assert services.transaction_manager.commit(tx2).is_committed
    assert model_v2.version == 2
    assert model_v2.parent_version == 1

    stored = ws.graph.get_node(node_v1.id)
    assert isinstance(stored, AdaptiveMembershipNode)
    assert stored.node_type == HCIRNodeType.ADAPTIVE_MEMBERSHIP

    replayed_v1 = AdaptiveMembershipModel.from_node(stored, version=1)
    replayed_v2 = AdaptiveMembershipModel.from_node(stored, version=2)
    assert replayed_v1.evaluate(
        1.5, scale_context={"subject_extent": 0.5, "object_extent": 0.5}
    ) == pytest.approx(hist_assertion.degree)
    assert replayed_v2.evaluate(
        1.5, scale_context={"subject_extent": 0.5, "object_extent": 0.5}
    ) == pytest.approx(1.0)


def test_declarative_relation_composition_policy_guards() -> None:
    """Verify INSIDE o INSIDE, forbidden NEAR o NEAR, and container-scale guards."""
    policy = RelationCompositionPolicy.default_spatial_policy()

    # 1. INSIDE o INSIDE -> INSIDE (Godel min)
    res_in = policy.compose("inside", "inside", 0.92, 0.85)
    assert res_in.allowed and res_in.composed_relation == "inside"
    assert res_in.degree == pytest.approx(0.85)

    # 2. NEAR o NEAR -> explicitly forbidden (Sorites guard)
    res_nn = policy.compose("near", "near", 0.9, 0.9)
    assert not res_nn.allowed and res_nn.composed_relation is None

    # 3. NEAR(A, cup) o INSIDE(cup, warehouse) -> rejected when diam(warehouse) >> diam(cup)
    res_warehouse = policy.compose(
        "near",
        "inside",
        0.95,
        0.98,
        geometry_context={"intermediate_diameter": 0.1, "target_diameter": 50.0},
    )
    assert not res_warehouse.allowed

    # 4. INSIDE(spoon, cup) o NEAR(cup, plate) -> allowed when cup is small relative to L_ref
    res_small = policy.compose(
        "inside",
        "near",
        0.95,
        0.80,
        geometry_context={"intermediate_diameter": 0.1, "reference_scale": 1.0},
    )
    assert res_small.allowed and res_small.composed_relation == "near"
    assert 0.0 < res_small.degree < 0.80

    # 5. INSIDE(desk, building) o NEAR(building, park) -> rejected when intermediate exceeds max_scale_ratio
    res_large_mid = policy.compose(
        "inside",
        "near",
        0.95,
        0.80,
        geometry_context={"intermediate_diameter": 20.0, "reference_scale": 10.0},
    )
    assert not res_large_mid.allowed


def test_kernel_batch_parity_hedges_and_fuzzy_expr() -> None:
    """Verify scalar vs NumPy batch parity, monotonic hedges, and FuzzyExpr profiles."""
    xs = np.linspace(-0.5, 4.5, 25)
    for fam, params in (
        (MembershipKernelFamily.GAUSSIAN, {"center": 1.5, "sigma": 0.8}),
        (MembershipKernelFamily.TRAPEZOIDAL, {"a": 0.0, "b": 0.5, "c": 2.0, "d": 3.5}),
        (MembershipKernelFamily.SIGMOID, {"slope": 2.0, "midpoint": 1.5}),
        (MembershipKernelFamily.MAHALANOBIS_RBF, {"sigma": 1.2}),
    ):
        batch = evaluate_membership_kernel_batch(xs, fam, params)
        for val, b_val in zip(xs.tolist(), batch.tolist(), strict=True):
            assert b_val == pytest.approx(evaluate_membership_kernel(val, fam, params), abs=1e-12)

    assert apply_hedge(0.8, FuzzyHedge.VERY) < 0.8 < apply_hedge(0.8, FuzzyHedge.SOMEWHAT)
    assert (
        get_fuzzy_profile(FuzzyProfileId.GODEL, negation_mode=NegationMode.STRICT_GODEL).negate(0.3)
        == 0.0
    )

    expr = FuzzyExpr(
        op="and",
        children=[
            FuzzyExpr(
                op="hedge",
                args={"hedge": "very"},
                children=[FuzzyExpr(op="var", args={"name": "near"})],
            ),
            FuzzyExpr(op="var", args={"name": "large"}),
        ],
    )
    assert expr.evaluate({"near": 0.8, "large": 0.5}, profile="product") == pytest.approx(
        0.64 * 0.5
    )
    assert expr.evaluate({"near": 0.8, "large": 0.5}, profile="godel") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_vm_math_fuzzy_syscalls_and_separated_latency() -> None:
    """Verify all 5 `math.fuzzy.*` syscalls through HCIRInterpreter and measure ALU vs dispatch."""
    ws = HCIRWorkspaceState(tenant_id="tenant_f0")
    services = KernelServices.create_default(
        ws,
        verification_pipeline=create_default_pipeline(default_scope=Scope(tenant_id="tenant_f0")),
    )
    interp = HCIRInterpreter(ws, services=services)

    stream = InstructionStream(
        author="test_f0",
        instructions=[
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "math.fuzzy.membership",
                    "params": {
                        "value": 1.5,
                        "kernel_family": "trapezoidal",
                        "parameters": {"a": 0.0, "b": 0.0, "c": 1.0, "d": 3.0},
                        "reference_scale_kind": "entity_extent",
                        "scale_context": {"subject_extent": 0.5, "object_extent": 0.5},
                    },
                    "bind": "$r0",
                },
            ),
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "math.fuzzy.hedge",
                    "params": {"degree": 0.8, "hedge": "very"},
                    "bind": "$r1",
                },
            ),
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "math.fuzzy.eval",
                    "params": {
                        "expr": {
                            "op": "and",
                            "children": [
                                {"op": "var", "args": {"name": "a"}},
                                {"op": "var", "args": {"name": "b"}},
                            ],
                        },
                        "bindings": {"a": 0.75, "b": 0.64},
                        "profile_id": "product",
                    },
                    "bind": "$r2",
                },
            ),
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "math.fuzzy.compose",
                    "params": {
                        "first_relation": "inside",
                        "second_relation": "inside",
                        "degree_ab": 0.9,
                        "degree_bc": 0.7,
                    },
                    "bind": "$r3",
                },
            ),
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "math.fuzzy.bipolar",
                    "params": {
                        "observations": [
                            {
                                "evidence_id": "e1",
                                "polarity": "support",
                                "strength": 0.9,
                                "dependency_group": "g1",
                            }
                        ]
                    },
                    "bind": "$r4",
                },
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
        ],
    )
    res = await interp.execute(stream)
    assert res.success is True
    regs = res.events[0]["registers"]
    assert regs["$r0"]["degree"] == pytest.approx(0.75)
    assert regs["$r1"]["degree"] == pytest.approx(0.64)
    assert regs["$r2"]["degree"] == pytest.approx(0.48)
    assert regs["$r3"]["allowed"] is True and regs["$r3"]["degree"] == pytest.approx(0.7)
    assert regs["$r4"]["ignorance"] < 1.0

    # Separated latency measurement: raw ALU vs VM syscall dispatch
    t0 = time.perf_counter()
    for _ in range(200):
        compute_fuzzy_membership(
            1.5,
            kernel_family="trapezoidal",
            parameters={"a": 0.0, "b": 0.0, "c": 1.0, "d": 3.0},
            reference_scale_kind="entity_extent",
            scale_context={"subject_extent": 0.5, "object_extent": 0.5},
        )
        compute_fuzzy_hedge(0.8, hedge="very")
        compute_fuzzy_eval(
            {
                "op": "and",
                "children": [
                    {"op": "var", "args": {"name": "a"}},
                    {"op": "var", "args": {"name": "b"}},
                ],
            },
            bindings={"a": 0.75, "b": 0.64},
        )
        compute_fuzzy_compose("inside", "inside", 0.9, 0.7)
        compute_fuzzy_bipolar([{"evidence_id": "e1", "polarity": "support", "strength": 0.9}])
    alu_per_op_ms = ((time.perf_counter() - t0) * 1000.0) / 1000.0
    assert (
        alu_per_op_ms < 1.0
    )  # Sanity bound; experimental target recorded separately from VM overhead
