"""Fuzzy Symbolic Cognition & Adaptive Semantics Foundations (ADR 006 Phase F0).

Pure, deterministic, ontology-blind Layer-0 primitives implementing:
- Complete residuated lattice profiles (`FuzzyLogicProfile`: godel, product, lukasiewicz)
- Strict separation of fuzzy degree vs. epistemic probability (`FuzzyAssertion`)
- Correlation-normalized bipolar evidence aggregation (`BipolarEvidenceState`)
- Immutable, versioned adaptive membership models (`AdaptiveMembershipModel`)
- Serializable `FuzzyExpr` ASTs and declarative `RelationCompositionPolicy`
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hbllm.hcir.types import MembershipDegree, NoveltyPolicy, Provenance, Scope, Timestamp

__all__ = [
    "AdaptiveMembershipModel",
    "BipolarEvidenceState",
    "BipolarObservation",
    "FuzzyAssertion",
    "FuzzyExpr",
    "FuzzyHedge",
    "FuzzyLogicProfile",
    "FuzzyProfileId",
    "MembershipKernelFamily",
    "NegationMode",
    "ReferenceScaleKind",
    "RelationCompositionPolicy",
    "RelationCompositionResult",
    "RelationCompositionRule",
    "aggregate_bipolar_evidence",
    "apply_hedge",
    "evaluate_membership_kernel",
    "evaluate_membership_kernel_batch",
    "get_fuzzy_profile",
    "normalize_by_reference_scale",
]


def _clamp01(value: float) -> float:
    x = float(value)
    if math.isnan(x):
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


class FuzzyProfileId(StrEnum):
    """Supported complete residuated lattice fuzzy logic profiles."""

    GODEL = "godel"
    PRODUCT = "product"
    LUKASIEWICZ = "lukasiewicz"


class NegationMode(StrEnum):
    """Negation semantics paired with a residuated lattice profile."""

    STANDARD = "standard"
    STRICT_GODEL = "strict_godel"


class FuzzyHedge(StrEnum):
    """Monotonic linguistic hedge modifiers operating on membership degrees."""

    IDENTITY = "identity"
    VERY = "very"
    EXTREMELY = "extremely"
    SOMEWHAT = "somewhat"
    SLIGHTLY = "slightly"
    INTENSIFY = "intensify"


class MembershipKernelFamily(StrEnum):
    """Dimensionless parametric membership kernel families."""

    GAUSSIAN = "gaussian"
    TRAPEZOIDAL = "trapezoidal"
    SIGMOID = "sigmoid"
    MAHALANOBIS_RBF = "mahalanobis_rbf"


class ReferenceScaleKind(StrEnum):
    """Dimensionless reference normalizers compliant with priors.md."""

    RAW = "raw"
    ENTITY_EXTENT = "entity_extent"
    NAVIGABLE_DIAMETER = "navigable_diameter"
    CLOSING_TTC = "closing_ttc"
    FEATURE_ZSCORE = "feature_zscore"


def apply_hedge(degree: float, hedge: str | FuzzyHedge) -> float:
    """Apply a monotonic linguistic hedge transformation to a degree in [0, 1]."""
    mu = _clamp01(degree)
    kind = hedge if isinstance(hedge, FuzzyHedge) else FuzzyHedge(str(hedge).strip().lower())
    if kind == FuzzyHedge.IDENTITY:
        return mu
    if kind == FuzzyHedge.VERY:
        return mu * mu
    if kind == FuzzyHedge.EXTREMELY:
        return mu * mu * mu
    if kind in (FuzzyHedge.SOMEWHAT, FuzzyHedge.SLIGHTLY):
        return math.sqrt(mu)
    if kind == FuzzyHedge.INTENSIFY:
        if mu <= 0.5:
            return 2.0 * mu * mu
        return 1.0 - 2.0 * ((1.0 - mu) ** 2)
    raise ValueError(f"Unsupported fuzzy hedge: {hedge!r}")


class FuzzyLogicProfile(BaseModel):
    """Coherent residuated-lattice algebra tuple (t-norm, t-conorm, negation, residuum)."""

    model_config = ConfigDict(frozen=True)

    profile_id: FuzzyProfileId = FuzzyProfileId.PRODUCT
    negation_mode: NegationMode = NegationMode.STANDARD

    def t_norm(self, a: float, b: float) -> float:
        x, y = _clamp01(a), _clamp01(b)
        if self.profile_id == FuzzyProfileId.GODEL:
            return min(x, y)
        if self.profile_id == FuzzyProfileId.PRODUCT:
            return x * y
        return max(0.0, x + y - 1.0)

    def t_norm_many(self, values: Iterable[float]) -> float:
        acc = 1.0
        for val in values:
            acc = self.t_norm(acc, val)
            if acc <= 0.0 and self.profile_id != FuzzyProfileId.LUKASIEWICZ:
                return 0.0
        return acc

    def t_conorm(self, a: float, b: float) -> float:
        x, y = _clamp01(a), _clamp01(b)
        if self.profile_id == FuzzyProfileId.GODEL:
            return max(x, y)
        if self.profile_id == FuzzyProfileId.PRODUCT:
            return _clamp01(x + y - x * y)
        return min(1.0, x + y)

    def t_conorm_many(self, values: Iterable[float]) -> float:
        acc = 0.0
        for val in values:
            acc = self.t_conorm(acc, val)
            if acc >= 1.0:
                return 1.0
        return acc

    def negate(self, a: float) -> float:
        x = _clamp01(a)
        if self.negation_mode == NegationMode.STRICT_GODEL:
            return 1.0 if x <= 0.0 else 0.0
        return 1.0 - x

    def residuum(self, a: float, b: float) -> float:
        """Evaluate algebraic residuum `a => b` adjoint to this profile's t-norm."""
        x, y = _clamp01(a), _clamp01(b)
        if self.profile_id == FuzzyProfileId.GODEL:
            return 1.0 if x <= y + 1e-12 else y
        if self.profile_id == FuzzyProfileId.PRODUCT:
            if x <= y + 1e-12 or x <= 0.0:
                return 1.0
            return _clamp01(y / x)
        return min(1.0, 1.0 - x + y)

    def bi_residuum(self, a: float, b: float) -> float:
        """Evaluate fuzzy equivalence `(a => b) AND (b => a)`."""
        return self.t_norm(self.residuum(a, b), self.residuum(b, a))

    def alpha_cut(self, degree: float, alpha: float = 0.5, *, strict: bool = False) -> bool:
        d = _clamp01(degree)
        a = _clamp01(alpha)
        return d > a if strict else d >= a


_PROFILE_CACHE: dict[tuple[FuzzyProfileId, NegationMode], FuzzyLogicProfile] = {}


def get_fuzzy_profile(
    profile_id: str | FuzzyProfileId = FuzzyProfileId.PRODUCT,
    *,
    negation_mode: str | NegationMode = NegationMode.STANDARD,
) -> FuzzyLogicProfile:
    """Return a cached, immutable `FuzzyLogicProfile` instance."""
    pid = (
        profile_id
        if isinstance(profile_id, FuzzyProfileId)
        else FuzzyProfileId(str(profile_id).strip().lower())
    )
    nmode = (
        negation_mode
        if isinstance(negation_mode, NegationMode)
        else NegationMode(str(negation_mode).strip().lower())
    )
    key = (pid, nmode)
    prof = _PROFILE_CACHE.get(key)
    if prof is None:
        prof = FuzzyLogicProfile(profile_id=pid, negation_mode=nmode)
        _PROFILE_CACHE[key] = prof
    return prof


class FuzzyAssertion(BaseModel):
    """Graded concept or relation assertion strictly distinct from epistemic belief confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assertion_id: str = Field(default_factory=lambda: f"fa_{uuid.uuid4().hex[:10]}")
    subject_ref: str
    target_ref: str | None = None
    semantics: str
    degree: MembershipDegree
    degree_interval: tuple[MembershipDegree, MembershipDegree] | None = None
    profile_id: FuzzyProfileId = FuzzyProfileId.PRODUCT
    context_ref: str = "default"
    model_id: str
    model_version: int = Field(default=1, ge=1)
    evidence_refs: tuple[str, ...] = ()
    snapshot_sequence: int = Field(default=0, ge=0)
    timestamp: Timestamp = Field(default_factory=time.time)

    @model_validator(mode="after")
    def _validate_interval(self) -> FuzzyAssertion:
        if self.degree_interval is not None:
            lo, hi = float(self.degree_interval[0]), float(self.degree_interval[1])
            if lo > hi:
                raise ValueError(f"degree_interval lower bound {lo} > upper bound {hi}")
            if float(self.degree) < lo - 1e-9 or float(self.degree) > hi + 1e-9:
                raise ValueError(
                    f"degree {self.degree} must lie inside degree_interval [{lo}, {hi}]"
                )
        return self


class BipolarObservation(BaseModel):
    """A single positive or refuting observation with correlation-cluster provenance."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    polarity: Literal["support", "refute"]
    strength: MembershipDegree
    source_weight: MembershipDegree = 1.0
    dependency_group: str = "default"
    source_channel_id: str = "default"
    tracklet_epoch: int = 0
    timestamp: float = 0.0


class BipolarEvidenceState(BaseModel):
    """Correlation-normalized bipolar support/refutation, ignorance, and paraconsistent conflict."""

    model_config = ConfigDict(frozen=True)

    support_mass: float = Field(ge=0.0)
    refutation_mass: float = Field(ge=0.0)
    support_share: MembershipDegree
    refutation_share: MembershipDegree
    ignorance: MembershipDegree
    conflict: MembershipDegree
    independent_support_clusters: int = Field(ge=0)
    independent_refute_clusters: int = Field(ge=0)
    profile_id: FuzzyProfileId = FuzzyProfileId.LUKASIEWICZ
    ignorance_prior: float = Field(gt=0.0, default=2.0)
    saturation_scale: float = Field(gt=0.0, default=1.0)


def aggregate_bipolar_evidence(
    observations: Sequence[BipolarObservation],
    *,
    profile_id: str | FuzzyProfileId = FuzzyProfileId.LUKASIEWICZ,
    ignorance_prior: float = 2.0,
    saturation_scale: float = 1.0,
    half_life_seconds: float | None = None,
) -> BipolarEvidenceState:
    """Aggregate observations with Stage-1 within-cluster Gödel idempotence and Stage-2 bipolar scoring."""
    if ignorance_prior <= 0.0:
        raise ValueError("ignorance_prior must be > 0")
    if saturation_scale <= 0.0:
        raise ValueError("saturation_scale must be > 0")

    profile = get_fuzzy_profile(profile_id)
    novelty_policy = (
        NoveltyPolicy(half_life_seconds=float(half_life_seconds))
        if half_life_seconds is not None and half_life_seconds > 0.0
        else None
    )

    # Stage 1: Group correlated observations by (polarity, dependency_group, source_channel_id, tracklet_epoch)
    clusters: dict[tuple[str, str, str, int], list[BipolarObservation]] = {}
    for obs in observations:
        key = (obs.polarity, obs.dependency_group, obs.source_channel_id, int(obs.tracklet_epoch))
        clusters.setdefault(key, []).append(obs)

    # Track last timestamp per (polarity, source_channel_id) to discount rapid temporal correlation
    last_channel_ts: dict[tuple[str, str], float] = {}
    support_mass = 0.0
    refutation_mass = 0.0
    sup_clusters = 0
    ref_clusters = 0

    ordered_keys = sorted(
        clusters.keys(),
        key=lambda k: (min(o.timestamp for o in clusters[k]), k[0], k[1], k[2], k[3]),
    )
    for key in ordered_keys:
        polarity, _, channel_id, _ = key
        items = clusters[key]
        # Within-cluster idempotence: repeated identical observations collapse to max strength/weight
        cluster_strength = max(_clamp01(o.strength) for o in items)
        cluster_weight = max(_clamp01(o.source_weight) for o in items)
        cluster_ts = max(float(o.timestamp) for o in items)

        novelty_factor = 1.0
        if novelty_policy is not None and cluster_ts > 0.0:
            ch_key = (polarity, channel_id)
            prev_ts = last_channel_ts.get(ch_key)
            if prev_ts is not None:
                dt = max(0.0, cluster_ts - prev_ts)
                novelty_factor = novelty_policy.compute_temporal_novelty(dt)
            last_channel_ts[ch_key] = max(prev_ts or 0.0, cluster_ts)

        contrib = cluster_weight * cluster_strength * novelty_factor
        if polarity == "support":
            support_mass += contrib
            sup_clusters += 1
        else:
            refutation_mass += contrib
            ref_clusters += 1

    denom = support_mass + refutation_mass + float(ignorance_prior)
    support_share = _clamp01(support_mass / denom)
    refutation_share = _clamp01(refutation_mass / denom)
    ignorance = _clamp01(float(ignorance_prior) / denom)

    sat_s = _clamp01(1.0 - math.exp(-support_mass / float(saturation_scale)))
    sat_r = _clamp01(1.0 - math.exp(-refutation_mass / float(saturation_scale)))
    conflict = profile.t_norm(sat_s, sat_r)

    return BipolarEvidenceState(
        support_mass=float(support_mass),
        refutation_mass=float(refutation_mass),
        support_share=support_share,
        refutation_share=refutation_share,
        ignorance=ignorance,
        conflict=conflict,
        independent_support_clusters=sup_clusters,
        independent_refute_clusters=ref_clusters,
        profile_id=profile.profile_id,
        ignorance_prior=float(ignorance_prior),
        saturation_scale=float(saturation_scale),
    )


def normalize_by_reference_scale(
    raw_value: float,
    scale_kind: str | ReferenceScaleKind,
    scale_context: Mapping[str, float] | None = None,
) -> float:
    """Convert raw physical/feature measurements into a dimensionless quantity per priors.md."""
    kind = (
        scale_kind
        if isinstance(scale_kind, ReferenceScaleKind)
        else ReferenceScaleKind(str(scale_kind).strip().lower())
    )
    val = float(raw_value)
    ctx = scale_context or {}
    eps = max(1e-9, float(ctx.get("epsilon", 1e-6)))

    if kind == ReferenceScaleKind.RAW:
        return val
    if kind == ReferenceScaleKind.ENTITY_EXTENT:
        l_sub = max(0.0, float(ctx.get("subject_extent", 0.5)))
        l_obj = max(0.0, float(ctx.get("object_extent", 0.5)))
        ref = max(eps, float(ctx.get("reference_extent", l_sub + l_obj)))
        return val / ref
    if kind == ReferenceScaleKind.NAVIGABLE_DIAMETER:
        diam = max(eps, float(ctx.get("navigable_diameter", 1.0)))
        return val / diam
    if kind == ReferenceScaleKind.CLOSING_TTC:
        closing_speed = float(ctx.get("closing_speed", 0.0))
        if closing_speed <= eps:
            return float("inf") if val > 0.0 else 0.0
        return max(0.0, val) / closing_speed
    if kind == ReferenceScaleKind.FEATURE_ZSCORE:
        mean = float(ctx.get("mean", 0.0))
        std = max(eps, float(ctx.get("std", 1.0)))
        return abs(val - mean) / std
    raise ValueError(f"Unsupported reference_scale_kind: {scale_kind!r}")


def evaluate_membership_kernel(
    x: float,
    kernel_family: str | MembershipKernelFamily,
    params: Mapping[str, float],
) -> float:
    """Evaluate a dimensionless membership kernel at scalar *x*."""
    fam = (
        kernel_family
        if isinstance(kernel_family, MembershipKernelFamily)
        else MembershipKernelFamily(str(kernel_family).strip().lower())
    )
    xv = float(x)
    if math.isnan(xv):
        return 0.0

    if fam == MembershipKernelFamily.GAUSSIAN:
        if math.isinf(xv):
            return 0.0
        center = float(params.get("center", 0.0))
        sigma = max(1e-6, float(params.get("sigma", 1.0)))
        z = (xv - center) / sigma
        return _clamp01(math.exp(-0.5 * z * z))

    if fam == MembershipKernelFamily.MAHALANOBIS_RBF:
        if math.isinf(xv):
            return 0.0
        sigma = max(1e-6, float(params.get("sigma", 1.0)))
        z = xv / sigma
        return _clamp01(math.exp(-0.5 * z * z))

    if fam == MembershipKernelFamily.TRAPEZOIDAL:
        a = float(params.get("a", 0.0))
        b = float(params.get("b", a))
        c = float(params.get("c", b))
        d = float(params.get("d", c))
        if not (a <= b <= c <= d):
            raise ValueError(
                f"Trapezoidal parameters must satisfy a <= b <= c <= d, got {(a, b, c, d)}"
            )
        if math.isinf(xv):
            return 0.0
        if xv < a or xv > d:
            return 0.0
        if b <= xv <= c:
            return 1.0
        if xv < b:
            return 1.0 if b <= a + 1e-12 else _clamp01((xv - a) / (b - a))
        return 1.0 if d <= c + 1e-12 else _clamp01((d - xv) / (d - c))

    if fam == MembershipKernelFamily.SIGMOID:
        slope = float(params.get("slope", 1.0))
        midpoint = float(params.get("midpoint", 0.0))
        if math.isinf(xv):
            pos_inf = xv > 0.0
            eff = slope if pos_inf else -slope
            return 1.0 if eff > 0.0 else (0.0 if eff < 0.0 else 0.5)
        arg = max(-60.0, min(60.0, -slope * (xv - midpoint)))
        return _clamp01(1.0 / (1.0 + math.exp(arg)))

    raise ValueError(f"Unsupported kernel_family: {kernel_family!r}")


def evaluate_membership_kernel_batch(
    xs: Sequence[float] | np.ndarray,
    kernel_family: str | MembershipKernelFamily,
    params: Mapping[str, float],
) -> np.ndarray:
    """Vectorized NumPy evaluation of a membership kernel over a batch of inputs."""
    arr = np.asarray(xs, dtype=np.float64)
    fam = (
        kernel_family
        if isinstance(kernel_family, MembershipKernelFamily)
        else MembershipKernelFamily(str(kernel_family).strip().lower())
    )
    if fam == MembershipKernelFamily.GAUSSIAN:
        center = float(params.get("center", 0.0))
        sigma = max(1e-6, float(params.get("sigma", 1.0)))
        z = (arr - center) / sigma
        out = np.exp(-0.5 * z * z)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    if fam == MembershipKernelFamily.MAHALANOBIS_RBF:
        sigma = max(1e-6, float(params.get("sigma", 1.0)))
        z = arr / sigma
        out = np.exp(-0.5 * z * z)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    if fam == MembershipKernelFamily.TRAPEZOIDAL:
        a = float(params.get("a", 0.0))
        b = float(params.get("b", a))
        c = float(params.get("c", b))
        d = float(params.get("d", c))
        if not (a <= b <= c <= d):
            raise ValueError(
                f"Trapezoidal parameters must satisfy a <= b <= c <= d, got {(a, b, c, d)}"
            )
        out = np.zeros_like(arr, dtype=np.float64)
        finite = np.isfinite(arr)
        plateau = finite & (arr >= b) & (arr <= c)
        out[plateau] = 1.0
        if b > a + 1e-12:
            left = finite & (arr >= a) & (arr < b)
            out[left] = (arr[left] - a) / (b - a)
        if d > c + 1e-12:
            right = finite & (arr > c) & (arr <= d)
            out[right] = (d - arr[right]) / (d - c)
        return np.clip(out, 0.0, 1.0)

    if fam == MembershipKernelFamily.SIGMOID:
        slope = float(params.get("slope", 1.0))
        midpoint = float(params.get("midpoint", 0.0))
        arg = np.clip(-slope * (arr - midpoint), -60.0, 60.0)
        out = 1.0 / (1.0 + np.exp(arg))
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

    raise ValueError(f"Unsupported kernel_family: {kernel_family!r}")


class AdaptiveMembershipModel(BaseModel):
    """Immutable, versioned adaptive membership parameterization persisted in HCIR (ADR 006 §4.4)."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    target_id: str
    contextual_scope: str = "default"
    kernel_family: MembershipKernelFamily = MembershipKernelFamily.GAUSSIAN
    reference_scale_kind: ReferenceScaleKind = ReferenceScaleKind.RAW
    learned_parameters: dict[str, float] = Field(default_factory=dict)
    sample_count: int = Field(default=0, ge=0)
    version: int = Field(default=1, ge=1)
    parent_version: int | None = None
    parent_hash: str | None = None
    created_at: Timestamp = Field(default_factory=time.time)
    provenance: Provenance = Field(default_factory=Provenance)

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "model_id": self.model_id,
                "target_id": self.target_id,
                "contextual_scope": self.contextual_scope,
                "kernel_family": self.kernel_family.value,
                "reference_scale_kind": self.reference_scale_kind.value,
                "learned_parameters": {
                    str(k): float(v) for k, v in sorted(self.learned_parameters.items())
                },
                "sample_count": int(self.sample_count),
                "version": int(self.version),
                "parent_version": self.parent_version,
                "parent_hash": self.parent_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def evaluate(self, raw_value: float, scale_context: Mapping[str, float] | None = None) -> float:
        scaled = normalize_by_reference_scale(raw_value, self.reference_scale_kind, scale_context)
        return evaluate_membership_kernel(scaled, self.kernel_family, self.learned_parameters)

    def evaluate_batch(
        self,
        raw_values: Sequence[float] | np.ndarray,
        scale_context: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        arr = np.asarray(raw_values, dtype=np.float64)
        if self.reference_scale_kind == ReferenceScaleKind.RAW:
            return evaluate_membership_kernel_batch(
                arr, self.kernel_family, self.learned_parameters
            )
        scaled = np.array(
            [
                normalize_by_reference_scale(float(v), self.reference_scale_kind, scale_context)
                for v in arr.flat
            ],
            dtype=np.float64,
        ).reshape(arr.shape)
        return evaluate_membership_kernel_batch(scaled, self.kernel_family, self.learned_parameters)

    def emit_assertion(
        self,
        *,
        subject_ref: str,
        raw_value: float,
        semantics: str,
        target_ref: str | None = None,
        scale_context: Mapping[str, float] | None = None,
        profile_id: str | FuzzyProfileId = FuzzyProfileId.PRODUCT,
        evidence_refs: tuple[str, ...] = (),
        snapshot_sequence: int = 0,
        degree_interval: tuple[float, float] | None = None,
    ) -> FuzzyAssertion:
        deg = self.evaluate(raw_value, scale_context=scale_context)
        pid = (
            profile_id
            if isinstance(profile_id, FuzzyProfileId)
            else FuzzyProfileId(str(profile_id).strip().lower())
        )
        return FuzzyAssertion(
            subject_ref=subject_ref,
            target_ref=target_ref,
            semantics=semantics,
            degree=deg,
            degree_interval=degree_interval,
            profile_id=pid,
            context_ref=self.contextual_scope,
            model_id=self.model_id,
            model_version=self.version,
            evidence_refs=tuple(evidence_refs),
            snapshot_sequence=snapshot_sequence,
        )

    def to_node(
        self,
        *,
        scope: Scope | None = None,
        existing_snapshots: Mapping[str, Mapping[str, float]] | None = None,
    ) -> Any:
        from hbllm.hcir.nodes import AdaptiveMembershipNode

        snapshots: dict[str, dict[str, float]] = {
            str(ver): {str(pk): float(pv) for pk, pv in params.items()}
            for ver, params in (existing_snapshots or {}).items()
        }
        snapshots[str(self.version)] = {
            str(k): float(v) for k, v in self.learned_parameters.items()
        }
        return AdaptiveMembershipNode(
            id=f"amm_{self.model_id}",
            model_id=self.model_id,
            target_id=self.target_id,
            contextual_scope=self.contextual_scope,
            kernel_family=self.kernel_family.value,
            reference_scale_kind=self.reference_scale_kind.value,
            learned_parameters={str(k): float(v) for k, v in self.learned_parameters.items()},
            sample_count=self.sample_count,
            version=self.version,
            parent_version=self.parent_version,
            parent_hash=self.parent_hash,
            version_snapshots=snapshots,
            scope=scope or Scope(),
            provenance=self.provenance,
            tags=["adaptive_membership", self.contextual_scope, self.target_id],
        )

    @classmethod
    def from_node(cls, node: Any, *, version: int | None = None) -> AdaptiveMembershipModel:
        target_ver = int(version) if version is not None else int(node.version)
        snapshots = getattr(node, "version_snapshots", {}) or {}
        if str(target_ver) in snapshots:
            params = {str(k): float(v) for k, v in snapshots[str(target_ver)].items()}
        elif target_ver == int(node.version):
            params = {str(k): float(v) for k, v in (node.learned_parameters or {}).items()}
        else:
            raise KeyError(
                f"AdaptiveMembershipModel '{node.model_id}' has no snapshot for version {target_ver}"
            )
        return cls(
            model_id=str(node.model_id or node.id.removeprefix("amm_")),
            target_id=str(node.target_id),
            contextual_scope=str(node.contextual_scope),
            kernel_family=MembershipKernelFamily(str(node.kernel_family)),
            reference_scale_kind=ReferenceScaleKind(str(node.reference_scale_kind)),
            learned_parameters=params,
            sample_count=int(node.sample_count),
            version=target_ver,
            parent_version=node.parent_version if target_ver == int(node.version) else None,
            parent_hash=node.parent_hash if target_ver == int(node.version) else None,
            provenance=node.provenance,
        )

    def evolve(
        self,
        new_parameters: Mapping[str, float],
        *,
        sample_delta: int = 1,
        learning_rate: float = 0.25,
        trigger_error_ids: Sequence[str] = (),
        scope: Scope | None = None,
        existing_snapshots: Mapping[str, Mapping[str, float]] | None = None,
        author: str = "adaptive_membership",
    ) -> tuple[AdaptiveMembershipModel, list[Any]]:
        """Create a new versioned model and emit atomic TransactionOperations without mutating history."""
        from hbllm.hcir.graph import HCIREdge
        from hbllm.hcir.nodes import AdaptationEventNode, HCIREdgeType
        from hbllm.hcir.transactions import TransactionOp, TransactionOperation

        next_ver = self.version + 1
        next_params = {str(k): float(v) for k, v in new_parameters.items()}
        next_model = AdaptiveMembershipModel(
            model_id=self.model_id,
            target_id=self.target_id,
            contextual_scope=self.contextual_scope,
            kernel_family=self.kernel_family,
            reference_scale_kind=self.reference_scale_kind,
            learned_parameters=next_params,
            sample_count=self.sample_count + max(0, int(sample_delta)),
            version=next_ver,
            parent_version=self.version,
            parent_hash=self.content_hash(),
            provenance=Provenance(
                created_by=author,
                source_type="inferred",
                reason=f"Adapted {self.model_id} v{self.version}->v{next_ver}",
            ),
        )

        merged_snapshots: dict[str, dict[str, float]] = {
            str(ver): {str(pk): float(pv) for pk, pv in params.items()}
            for ver, params in (existing_snapshots or {}).items()
        }
        merged_snapshots.setdefault(
            str(self.version), {str(k): float(v) for k, v in self.learned_parameters.items()}
        )
        merged_snapshots[str(next_ver)] = dict(next_params)

        eff_scope = scope or Scope()
        node = next_model.to_node(scope=eff_scope, existing_snapshots=merged_snapshots)
        adapt_event = AdaptationEventNode(
            id=f"adapt_{self.model_id}_v{next_ver}_{uuid.uuid4().hex[:6]}",
            model_id=self.model_id,
            adaptation_type="membership_parameter_update",
            parameters_before=dict(self.learned_parameters),
            parameters_after=dict(next_params),
            learning_rate_used=float(learning_rate),
            evidence_count=max(0, int(sample_delta)),
            trigger_error_ids=list(trigger_error_ids),
            scope=eff_scope,
            provenance=next_model.provenance,
        )
        adapts_edge = HCIREdge(
            id=f"edge_adapts_{adapt_event.id}",
            edge_type=HCIREdgeType.ADAPTS,
            sources=[adapt_event.id],
            targets=[node.id],
            properties={"from_version": self.version, "to_version": next_ver},
            provenance=next_model.provenance,
        )
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node.id,
                node_data=node.model_dump(),
            ),
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id=adapt_event.id,
                node_data=adapt_event.model_dump(),
            ),
            TransactionOperation(
                op=TransactionOp.ADD_EDGE,
                edge_id=adapts_edge.id,
                edge_data=adapts_edge.model_dump(),
            ),
        ]
        return next_model, ops


class FuzzyExpr(BaseModel):
    """Serializable homoiconic fuzzy expression AST evaluated under a `FuzzyLogicProfile`."""

    model_config = ConfigDict(frozen=True)

    op: Literal[
        "const",
        "var",
        "membership",
        "and",
        "or",
        "not",
        "implies",
        "equiv",
        "hedge",
        "alpha_cut",
    ]
    args: dict[str, Any] = Field(default_factory=dict)
    children: list[FuzzyExpr] = Field(default_factory=list)

    def evaluate(
        self,
        bindings: Mapping[str, Any] | None = None,
        *,
        profile: str | FuzzyProfileId | FuzzyLogicProfile = FuzzyProfileId.PRODUCT,
        models: Mapping[str, AdaptiveMembershipModel] | None = None,
    ) -> float:
        prof = profile if isinstance(profile, FuzzyLogicProfile) else get_fuzzy_profile(profile)
        bmap = bindings or {}
        if self.op == "const":
            return _clamp01(float(self.args.get("value", 0.0)))
        if self.op == "var":
            name = str(self.args["name"])
            return _clamp01(float(bmap.get(name, self.args.get("default", 0.0))))
        if self.op == "membership":
            raw_val = self.args.get("value")
            if raw_val is None and "var" in self.args:
                raw_val = bmap.get(str(self.args["var"]), 0.0)
            scale_ctx = self.args.get("scale_context")
            if isinstance(scale_ctx, str):
                scale_ctx = bmap.get(scale_ctx)
            model_id = self.args.get("model_id")
            if model_id and models and str(model_id) in models:
                return models[str(model_id)].evaluate(
                    float(raw_val or 0.0),
                    scale_context=scale_ctx if isinstance(scale_ctx, Mapping) else None,
                )
            kernel = str(self.args.get("kernel_family", MembershipKernelFamily.GAUSSIAN.value))
            scale_kind = str(self.args.get("reference_scale_kind", ReferenceScaleKind.RAW.value))
            params = self.args.get("parameters", {})
            scaled = normalize_by_reference_scale(
                float(raw_val or 0.0),
                scale_kind,
                scale_ctx if isinstance(scale_ctx, Mapping) else None,
            )
            return evaluate_membership_kernel(scaled, kernel, params)
        if self.op == "and":
            return prof.t_norm_many(
                c.evaluate(bmap, profile=prof, models=models) for c in self.children
            )
        if self.op == "or":
            return prof.t_conorm_many(
                c.evaluate(bmap, profile=prof, models=models) for c in self.children
            )
        if self.op == "not":
            if not self.children:
                return 1.0
            return prof.negate(self.children[0].evaluate(bmap, profile=prof, models=models))
        if self.op == "implies":
            if len(self.children) < 2:
                raise ValueError("FuzzyExpr 'implies' requires 2 children")
            a = self.children[0].evaluate(bmap, profile=prof, models=models)
            b = self.children[1].evaluate(bmap, profile=prof, models=models)
            return prof.residuum(a, b)
        if self.op == "equiv":
            if len(self.children) < 2:
                raise ValueError("FuzzyExpr 'equiv' requires 2 children")
            a = self.children[0].evaluate(bmap, profile=prof, models=models)
            b = self.children[1].evaluate(bmap, profile=prof, models=models)
            return prof.bi_residuum(a, b)
        if self.op == "hedge":
            if not self.children:
                raise ValueError("FuzzyExpr 'hedge' requires 1 child")
            deg = self.children[0].evaluate(bmap, profile=prof, models=models)
            return apply_hedge(deg, str(self.args.get("hedge", FuzzyHedge.IDENTITY.value)))
        if self.op == "alpha_cut":
            if not self.children:
                raise ValueError("FuzzyExpr 'alpha_cut' requires 1 child")
            deg = self.children[0].evaluate(bmap, profile=prof, models=models)
            alpha = float(self.args.get("alpha", 0.5))
            strict = bool(self.args.get("strict", False))
            return 1.0 if prof.alpha_cut(deg, alpha, strict=strict) else 0.0
        raise ValueError(f"Unsupported FuzzyExpr op: {self.op!r}")


class RelationCompositionRule(BaseModel):
    """Declarative rule governing whether and how two graded relations compose (ADR 006 §4.5)."""

    model_config = ConfigDict(frozen=True)

    first_relation: str
    second_relation: str
    composed_relation: str | None = None
    directionality: Literal["forward", "symmetric", "directional_axis"] = "forward"
    geometric_guard: Literal[
        "none",
        "topological_closure",
        "triangle_inequality_bound",
        "extent_bounded",
        "directional_alignment",
    ] = "none"
    max_scale_ratio: float | None = None
    contextual_scope: str = "*"
    composition_operator: Literal["godel", "product", "lukasiewicz", "attenuated_product"] = (
        "product"
    )
    attenuation_factor: MembershipDegree = 1.0


class RelationCompositionResult(BaseModel):
    """Deterministic outcome of composing two graded relations under `RelationCompositionPolicy`."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    composed_relation: str | None = None
    degree: MembershipDegree = 0.0
    reason: str = ""
    rule_matched: bool = False


class RelationCompositionPolicy(BaseModel):
    """Declarative, context- and scale-guarded spatial/relational composition policy."""

    model_config = ConfigDict(frozen=True)

    policy_id: str = "default_spatial_policy"
    rules: tuple[RelationCompositionRule, ...] = ()

    def resolve_rule(
        self, first_relation: str, second_relation: str, contextual_scope: str = "default"
    ) -> RelationCompositionRule | None:
        r1 = first_relation.strip().lower()
        r2 = second_relation.strip().lower()
        ctx = contextual_scope.strip().lower()
        exact: RelationCompositionRule | None = None
        wildcard: RelationCompositionRule | None = None
        for rule in self.rules:
            if (
                rule.first_relation.strip().lower() != r1
                or rule.second_relation.strip().lower() != r2
            ):
                continue
            r_ctx = rule.contextual_scope.strip().lower()
            if r_ctx == ctx:
                exact = rule
                break
            if r_ctx == "*" and wildcard is None:
                wildcard = rule
        return exact or wildcard

    def compose(
        self,
        first_relation: str,
        second_relation: str,
        degree_ab: float,
        degree_bc: float,
        *,
        contextual_scope: str = "default",
        geometry_context: Mapping[str, Any] | None = None,
    ) -> RelationCompositionResult:
        rule = self.resolve_rule(first_relation, second_relation, contextual_scope)
        if rule is None:
            return RelationCompositionResult(
                allowed=False,
                composed_relation=None,
                degree=0.0,
                reason=f"No composition rule for {first_relation} o {second_relation}",
                rule_matched=False,
            )
        if rule.composed_relation is None:
            return RelationCompositionResult(
                allowed=False,
                composed_relation=None,
                degree=0.0,
                reason=f"Composition {first_relation} o {second_relation} explicitly forbidden",
                rule_matched=True,
            )

        gctx = geometry_context or {}
        slack = 1.0
        guard = rule.geometric_guard

        if guard == "directional_alignment":
            if gctx.get("same_axis_aligned") is False:
                return RelationCompositionResult(
                    allowed=False,
                    composed_relation=None,
                    degree=0.0,
                    reason="Directional axis alignment guard failed",
                    rule_matched=True,
                )
        elif guard == "triangle_inequality_bound" and rule.max_scale_ratio is not None:
            diam_b = float(gctx.get("intermediate_diameter", 0.0))
            ref_c = max(1e-9, float(gctx.get("reference_scale", 1.0)))
            ratio = diam_b / ref_c
            if ratio > float(rule.max_scale_ratio):
                return RelationCompositionResult(
                    allowed=False,
                    composed_relation=None,
                    degree=0.0,
                    reason=(
                        f"Scale ratio diam(B)/L_ref(C)={ratio:.3f} exceeds "
                        f"max_scale_ratio={rule.max_scale_ratio:.3f}"
                    ),
                    rule_matched=True,
                )
            slack = max(0.0, 1.0 - ratio / max(1e-9, float(rule.max_scale_ratio) * 2.0))
        elif guard == "extent_bounded" and rule.max_scale_ratio is not None:
            diam_b = max(1e-9, float(gctx.get("intermediate_diameter", 1.0)))
            diam_c = float(gctx.get("target_diameter", diam_b))
            ratio = diam_c / diam_b
            if ratio > float(rule.max_scale_ratio):
                return RelationCompositionResult(
                    allowed=False,
                    composed_relation=None,
                    degree=0.0,
                    reason=(
                        f"Container scale diam(C)/diam(B)={ratio:.3f} exceeds "
                        f"max_scale_ratio={rule.max_scale_ratio:.3f}"
                    ),
                    rule_matched=True,
                )

        d1, d2 = _clamp01(degree_ab), _clamp01(degree_bc)
        if rule.composition_operator == "godel":
            raw = min(d1, d2)
        elif rule.composition_operator == "lukasiewicz":
            raw = max(0.0, d1 + d2 - 1.0)
        elif rule.composition_operator == "attenuated_product":
            raw = d1 * d2 * float(rule.attenuation_factor) * slack
        else:
            raw = d1 * d2

        return RelationCompositionResult(
            allowed=True,
            composed_relation=rule.composed_relation,
            degree=_clamp01(raw),
            reason=f"Composed via {rule.composition_operator} ({rule.geometric_guard})",
            rule_matched=True,
        )

    @classmethod
    def default_spatial_policy(cls) -> RelationCompositionPolicy:
        """Return the canonical ADR 006 §4.5 spatial relation composition policy."""
        return cls(
            policy_id="default_spatial_policy",
            rules=(
                RelationCompositionRule(
                    first_relation="inside",
                    second_relation="inside",
                    composed_relation="inside",
                    directionality="forward",
                    geometric_guard="topological_closure",
                    composition_operator="godel",
                ),
                RelationCompositionRule(
                    first_relation="contains",
                    second_relation="contains",
                    composed_relation="contains",
                    directionality="forward",
                    geometric_guard="topological_closure",
                    composition_operator="godel",
                ),
                RelationCompositionRule(
                    first_relation="above",
                    second_relation="above",
                    composed_relation="above",
                    directionality="directional_axis",
                    geometric_guard="directional_alignment",
                    composition_operator="product",
                ),
                RelationCompositionRule(
                    first_relation="below",
                    second_relation="below",
                    composed_relation="below",
                    directionality="directional_axis",
                    geometric_guard="directional_alignment",
                    composition_operator="product",
                ),
                RelationCompositionRule(
                    first_relation="left_of",
                    second_relation="left_of",
                    composed_relation="left_of",
                    directionality="directional_axis",
                    geometric_guard="directional_alignment",
                    composition_operator="product",
                ),
                RelationCompositionRule(
                    first_relation="right_of",
                    second_relation="right_of",
                    composed_relation="right_of",
                    directionality="directional_axis",
                    geometric_guard="directional_alignment",
                    composition_operator="product",
                ),
                RelationCompositionRule(
                    first_relation="inside",
                    second_relation="near",
                    composed_relation="near",
                    directionality="forward",
                    geometric_guard="triangle_inequality_bound",
                    max_scale_ratio=0.5,
                    composition_operator="attenuated_product",
                    attenuation_factor=0.90,
                ),
                RelationCompositionRule(
                    first_relation="near",
                    second_relation="inside",
                    composed_relation=None,
                    directionality="forward",
                    geometric_guard="extent_bounded",
                    max_scale_ratio=1.5,
                    composition_operator="attenuated_product",
                ),
                RelationCompositionRule(
                    first_relation="near",
                    second_relation="near",
                    composed_relation=None,
                    directionality="symmetric",
                    geometric_guard="none",
                    composition_operator="product",
                ),
                RelationCompositionRule(
                    first_relation="far",
                    second_relation="far",
                    composed_relation=None,
                    directionality="symmetric",
                    geometric_guard="none",
                    composition_operator="product",
                ),
            ),
        )
