from __future__ import annotations

"""Multi-Modal Feature Fusion — fuses features from multiple modalities for single entities.

Provides ``MultiModalFeatureBundle`` to group features by entity across modalities,
and ``FeatureFusion`` to produce unified cross-modal embeddings while preserving
per-modality provenance.

All operations are deterministic and LLM-free.
"""

from dataclasses import dataclass, field

from hbllm.hcir.grounding.contracts.enums import ModalitySource
from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector


@dataclass
class MultiModalFeatureBundle:
    """Groups perceptual feature vectors by entity across modalities.

    A single entity (e.g. "red cube on table") may have:
    - Vision features from spatial grid extraction
    - Kinematic features from proprioceptive observation
    - Semantic features from text description
    """

    entity_id: str
    features_by_modality: dict[ModalitySource, list[PerceptualFeatureVector]] = field(
        default_factory=dict,
    )

    def add_feature(self, feature: PerceptualFeatureVector) -> None:
        """Adds a feature vector, grouped by its modality."""
        if feature.modality not in self.features_by_modality:
            self.features_by_modality[feature.modality] = []
        self.features_by_modality[feature.modality].append(feature)

    @property
    def modality_count(self) -> int:
        """Number of distinct modalities contributing features."""
        return len(self.features_by_modality)

    @property
    def total_features(self) -> int:
        return sum(len(fs) for fs in self.features_by_modality.values())

    @property
    def is_multimodal(self) -> bool:
        return self.modality_count > 1


@dataclass(frozen=True)
class FusedFeatureVector:
    """A unified embedding fused from multiple modality-specific feature vectors.

    Preserves provenance of contributing features and modalities.
    """

    entity_id: str
    fused_embedding: tuple[float, ...]
    contributing_modalities: frozenset[ModalitySource]
    contributing_feature_ids: tuple[str, ...]
    fusion_method: str  # "concatenation" | "mean_pool" | "weighted_mean"
    cross_modal_confidence: float  # 0.0–1.0, higher if multiple modalities agree


class FeatureFusion:
    """Fuses per-modality feature vectors into unified cross-modal embeddings.

    Fusion strategies:
    - ``mean_pool``: Element-wise mean across modalities (requires same-dim embeddings)
    - ``concatenation``: Concatenates embeddings from all modalities
    - ``weighted_mean``: Weighted by number of features per modality
    """

    def __init__(
        self,
        default_strategy: str = "concatenation",
        target_dim: int | None = None,
    ) -> None:
        self._strategy = default_strategy
        self._target_dim = target_dim

    def fuse_bundle(
        self,
        bundle: MultiModalFeatureBundle,
        strategy: str | None = None,
    ) -> FusedFeatureVector:
        """Fuses all features in a bundle into a single embedding."""
        method = strategy or self._strategy

        all_features: list[PerceptualFeatureVector] = []
        for modality_features in bundle.features_by_modality.values():
            all_features.extend(modality_features)

        if not all_features:
            return FusedFeatureVector(
                entity_id=bundle.entity_id,
                fused_embedding=(),
                contributing_modalities=frozenset(),
                contributing_feature_ids=(),
                fusion_method=method,
                cross_modal_confidence=0.0,
            )

        embeddings = [f.embedding for f in all_features]
        modalities = frozenset(bundle.features_by_modality.keys())
        feature_ids = tuple(f.feature_id for f in all_features)

        if method == "mean_pool":
            fused = self._mean_pool(embeddings)
        elif method == "weighted_mean":
            weights = self._compute_modality_weights(bundle)
            fused = self._weighted_mean(all_features, weights)
        else:
            # Default: concatenation
            fused = self._concatenate(embeddings)

        # Cross-modal confidence: higher when multiple modalities contribute
        cross_modal_confidence = min(1.0, bundle.modality_count / 3.0)
        if bundle.modality_count == 1:
            cross_modal_confidence = 0.3

        return FusedFeatureVector(
            entity_id=bundle.entity_id,
            fused_embedding=tuple(fused),
            contributing_modalities=modalities,
            contributing_feature_ids=feature_ids,
            fusion_method=method,
            cross_modal_confidence=cross_modal_confidence,
        )

    def fuse_bundles(
        self,
        bundles: list[MultiModalFeatureBundle],
        strategy: str | None = None,
    ) -> list[FusedFeatureVector]:
        """Fuses multiple entity bundles."""
        return [self.fuse_bundle(b, strategy=strategy) for b in bundles]

    # ------------------------------------------------------------------
    # Fusion implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(embeddings: list[tuple[float, ...]]) -> list[float]:
        """Element-wise mean across embeddings.  Pads shorter embeddings with 0."""
        if not embeddings:
            return []
        max_dim = max(len(e) for e in embeddings)
        result = [0.0] * max_dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                result[i] += v
        n = len(embeddings)
        return [v / n for v in result]

    @staticmethod
    def _concatenate(embeddings: list[tuple[float, ...]]) -> list[float]:
        """Concatenates all embeddings into one long vector."""
        result: list[float] = []
        for emb in embeddings:
            result.extend(emb)
        return result

    @staticmethod
    def _weighted_mean(
        features: list[PerceptualFeatureVector],
        weights: dict[ModalitySource, float],
    ) -> list[float]:
        """Weighted element-wise mean using modality weights."""
        if not features:
            return []
        max_dim = max(len(f.embedding) for f in features)
        result = [0.0] * max_dim
        total_w = 0.0
        for f in features:
            w = weights.get(f.modality, 1.0)
            for i, v in enumerate(f.embedding):
                result[i] += v * w
            total_w += w

        if total_w > 0:
            result = [v / total_w for v in result]
        return result

    @staticmethod
    def _compute_modality_weights(
        bundle: MultiModalFeatureBundle,
    ) -> dict[ModalitySource, float]:
        """Computes per-modality weights based on feature count.

        Modalities with more features get proportionally less weight per feature
        (to avoid dominating the fused embedding).
        """
        total = bundle.total_features
        if total == 0:
            return {}
        weights: dict[ModalitySource, float] = {}
        for modality, features in bundle.features_by_modality.items():
            # Inverse-count weighting: fewer features = higher weight per feature
            weights[modality] = 1.0 / max(1, len(features))
        # Normalize
        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {k: v / w_sum for k, v in weights.items()}
        return weights


def group_features_by_entity(
    features: list[PerceptualFeatureVector],
    entity_feature_map: dict[str, list[str]] | None = None,
) -> list[MultiModalFeatureBundle]:
    """Groups feature vectors into per-entity multi-modal bundles.

    If ``entity_feature_map`` is provided, uses it to map feature IDs
    to entity IDs.  Otherwise groups by ``observation_id``.
    """
    bundles: dict[str, MultiModalFeatureBundle] = {}

    for fv in features:
        # Determine entity ID
        entity_id: str | None = None
        if entity_feature_map:
            for eid, fids in entity_feature_map.items():
                if fv.feature_id in fids:
                    entity_id = eid
                    break

        if entity_id is None:
            # Fallback: group by observation_id
            entity_id = fv.observation_id

        if entity_id not in bundles:
            bundles[entity_id] = MultiModalFeatureBundle(entity_id=entity_id)
        bundles[entity_id].add_feature(fv)

    return list(bundles.values())
