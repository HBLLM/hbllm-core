"""Unit tests for Legacy World Modeling Capabilities (W014, W040, W079).

Verifies Criterion A: Isolated domain-general mathematical and algorithmic correctness.
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.causal_discovery import (
    BaseCausalDiscoveryEngine,
    CausalMediationResult,
    MediationIdentifiabilityAssumptions,
)
from hbllm.hcir.world.parietal_coordinate_transform import (
    AffineTransform2D,
    AxonometricIsometricTransform,
    GeometricModelSelector,
    ProjectiveHomography2D,
)

# ── W014: Entity Fission and Fusion Tests ─────────────────────────────────────


# ── W040: Formal Geometric & Perspective Transformation Tests ────────────────


def test_w040_affine_transform_estimation_and_inversion() -> None:
    """Affine estimation and exact analytical inversion (W040)."""
    src = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0]])
    # Translate by (10, 20) and scale x by 2
    true_matrix = np.array([[2.0, 0.0, 10.0], [0.0, 1.0, 20.0]])
    homog = np.hstack([src, np.ones((4, 1))])
    dst = homog @ true_matrix.T

    model, residual = AffineTransform2D.estimate(src, dst)
    assert residual < 1e-6
    pred = model.forward(src)
    np.testing.assert_allclose(pred, dst, atol=1e-5)

    # Invert
    inv_model = model.inverse()
    recovered = inv_model.forward(dst)
    np.testing.assert_allclose(recovered, src, atol=1e-5)


def test_w040_isometric_projection_and_unprojection() -> None:
    """Parallel axonometric/isometric projection (3D -> 2D) and ground-plane unprojection."""
    iso = AxonometricIsometricTransform(alpha_deg=30.0, scale=1.0)
    pts_3d = np.array([[10.0, 20.0, 5.0], [0.0, 0.0, 0.0], [5.0, 5.0, 0.0]])

    pts_2d = iso.project_3d_to_2d(pts_3d)
    assert pts_2d.shape == (3, 2)

    # Unproject point at z=0 ground plane
    ground_pt_2d = iso.project_3d_to_2d(np.array([[5.0, 5.0, 0.0]]))
    recovered_3d = iso.unproject_2d_to_3d(ground_pt_2d, z_plane=0.0)
    np.testing.assert_allclose(recovered_3d, np.array([[5.0, 5.0, 0.0]]), atol=1e-5)


def test_w040_projective_homography_estimation() -> None:
    """Projective homography under non-linear perspective division (W040)."""
    src = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    # Perspective trapezoid
    dst = np.array([[2.0, 3.0], [8.0, 3.0], [9.0, 9.0], [1.0, 9.0]])

    model, residual = ProjectiveHomography2D.estimate(src, dst)
    assert residual < 1e-5
    pred = model.forward(src)
    np.testing.assert_allclose(pred, dst, atol=1e-4)

    # Model selector verifies model fit
    selection = GeometricModelSelector.select_best_model(src, dst)
    assert selection["best_model_type"] in ("projective", "affine")
    assert selection["residual"] < 1.0


# ── W079: Pearl Causal Mediation Analysis Tests ───────────────────────────────


def test_w079_pearl_non_parametric_causal_mediation() -> None:
    """Pearl's mediation formula correctly decomposes Total Effect into NDE and NIE (W079).

    Data Generation Process:
      X ~ Bernoulli(0.5)
      M = 2 * X + noise (discrete mediator)
      Y = 3 * X + 4 * M (treatment has direct effect 3 and indirect effect through M)
      Here:
        Total Effect: changing X from 0 to 1 changes direct by 3, changes M by 2 which changes Y by 8 -> TE = 11.
        Natural Direct Effect (holding M at M(0)): 3.
        Natural Indirect Effect (holding X at 1, changing M from M(0) to M(1)): 8.
    """
    np.random.seed(42)
    observations = []
    for _ in range(400):
        x = int(np.random.rand() > 0.5)
        m = 2 * x if np.random.rand() > 0.1 else 0
        y = 3.0 * x + 4.0 * m
        observations.append({"X": x, "M": m, "Y": y})

    engine = BaseCausalDiscoveryEngine()
    result = engine.compute_causal_mediation(
        observations=observations,
        treatment="X",
        mediator="M",
        outcome="Y",
        baseline_treatment=0,
        active_treatment=1,
    )

    assert isinstance(result, CausalMediationResult)
    assert result.is_causally_identified is True
    # Verify TE = NDE + NIE within floating point tolerance
    assert (
        abs(result.total_effect - (result.natural_direct_effect + result.natural_indirect_effect))
        < 1e-4
    )
    # TE should be near 10-11, NDE near 3, NIE near 7-8
    assert result.natural_direct_effect > 2.0
    assert result.natural_indirect_effect > 5.0
    assert result.proportion_mediated > 0.5


def test_w079_confounding_identifiability_warning() -> None:
    """Engine raises explicit identifiability warnings when Pearl's assumptions are flagged as violated."""
    obs = [{"X": 1, "M": 2, "Y": 5.0}, {"X": 0, "M": 0, "Y": 0.0}]
    violated_assumptions = MediationIdentifiabilityAssumptions(
        no_treatment_outcome_confounding=False,  # Unmeasured confounding
        covariates_controlled=[],
    )

    engine = BaseCausalDiscoveryEngine()
    res = engine.compute_causal_mediation(
        observations=obs,
        treatment="X",
        mediator="M",
        outcome="Y",
        identifiability_assumptions=violated_assumptions,
    )

    assert res.is_causally_identified is False
    assert len(res.unmeasured_confounding_warnings) > 0
    assert res.estimation_regime == "observational_association"


# ── W014 & W040 Advanced Invariant & Lineage Tests ───────────────────────────


def test_w040_occam_selection_affine_vs_projective() -> None:
    """W040 uses geometric invariants to enforce Occam's razor over projective models."""
    # 1. Pure affine transformation: shear + translation
    src = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    affine_mat = np.array([[1.0, 0.5, 3.0], [0.0, 1.0, -2.0]])
    dst_affine = AffineTransform2D(affine_mat).forward(src)

    sel_affine = GeometricModelSelector.select_best_model(src, dst_affine)
    assert sel_affine["best_model_type"] == "affine"
    assert sel_affine["residual"] < 1e-4
    assert sel_affine["parallelism_deviation"] < 1e-4

    # 2. Pure isometric transformation (90° rotation + translation)
    rot_mat = np.array([[0.0, -1.0, 5.0], [1.0, 0.0, 2.0]])
    dst_iso = AffineTransform2D(rot_mat).forward(src)
    sel_iso = GeometricModelSelector.select_best_model(src, dst_iso)
    assert sel_iso["best_model_type"] == "isometric"
    assert sel_iso["is_isometric"] is True
    assert sel_iso["residual"] < 1e-4

    # 3. Collinear degenerate point set
    collinear_src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    collinear_dst = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    sel_degen = GeometricModelSelector.select_best_model(collinear_src, collinear_dst)
    assert sel_degen["best_model_type"] == "degenerate"
    assert sel_degen["ambiguity_score"] == 1.0
