from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from hbllm.hcir.grounding.contracts.enums import ModalitySource
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope


@dataclass(frozen=True)
class PerceptualFeatureVector:
    """Normalized feature vector extracted from a raw observation.

    Level 1 in the 5-Layer Abstraction Ladder: sits between raw sensor
    captures (Level 0 ObservationEnvelope) and uncommitted perceptual
    candidates (Level 2 GroundingClaim).

    Feature vectors are immutable, carry provenance back to their source
    observation, and expose an explicit modality tag for dependency-group
    aware Bayesian fusion downstream.
    """

    feature_id: str
    observation_id: str  # References source ObservationEnvelope
    modality: ModalitySource
    channel_id: str
    embedding: tuple[float, ...]  # Dense normalised feature vector
    spatial_extent: tuple[float, ...] | None = None  # bbox (x, y, w, h) or spatial coords
    semantic_labels: dict[str, float] = field(default_factory=dict)  # soft label -> confidence
    feature_metadata: dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.feature_id:
            raise ValueError("feature_id must be non-empty")
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        if not isinstance(self.embedding, tuple):
            object.__setattr__(self, "embedding", tuple(self.embedding))
        if len(self.embedding) == 0:
            raise ValueError("embedding must be non-empty")

        # Auto-generate provenance hash if not supplied
        if not self.provenance_hash:
            seed = f"{self.feature_id}:{self.observation_id}:{self.channel_id}"
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            object.__setattr__(self, "provenance_hash", digest)

    @property
    def dimensionality(self) -> int:
        return len(self.embedding)


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for projecting raw ObservationEnvelopes into normalised feature vectors.

    Domain-specific implementations register themselves at runtime.
    The cognitive core never depends on a concrete extractor.
    """

    @property
    def extractor_name(self) -> str:
        """Human-readable identifier for this extractor."""
        ...

    def can_extract(self, envelope: ObservationEnvelope) -> bool:
        """Returns True if this extractor handles the envelope's modality/channel."""
        ...

    def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        """Projects a raw observation into one or more normalised feature vectors."""
        ...


# ---------------------------------------------------------------------------
#  Concrete extractors: domain-agnostic defaults
# ---------------------------------------------------------------------------


class SpatialGridFeatureExtractor:
    """Extracts normalised feature vectors from 2-D grid spatial observations.

    Operates deterministically on grid metadata (cell values, positions, bounding
    boxes) embedded in ``envelope.metadata``.  Produces one feature vector per
    detected spatial entity segment.

    Expected ``metadata`` layout:
        {
            "grid": list[list[int]],          # optional raw grid
            "entities": [                      # pre-segmented entity list
                {
                    "id": str,
                    "position": (row, col),
                    "color": int,
                    "size": int,
                    "bbox": (x, y, w, h),     # optional
                },
                ...
            ],
        }
    """

    @property
    def extractor_name(self) -> str:
        return "SpatialGridFeatureExtractor"

    def can_extract(self, envelope: ObservationEnvelope) -> bool:
        return envelope.source_modality in (
            ModalitySource.VISION,
            ModalitySource.ENVIRONMENTAL_SENSOR,
        ) and ("entities" in envelope.metadata or "grid" in envelope.metadata)

    def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        features: list[PerceptualFeatureVector] = []
        entities = envelope.metadata.get("entities", [])

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("seg")))
            pos = ent.get("position", (0.0, 0.0))
            color = float(ent.get("color", 0))
            size = float(ent.get("size", 1))
            bbox = ent.get("bbox")

            # Normalised embedding: (norm_row, norm_col, norm_color, norm_size)
            embedding = (
                float(pos[0]) / max(1.0, float(ent.get("grid_h", 30))),
                float(pos[1]) / max(1.0, float(ent.get("grid_w", 30))),
                color / 10.0,  # ARC uses 0-9 colour palette
                math.log1p(size),
            )

            spatial = tuple(float(x) for x in bbox) if bbox else (float(pos[0]), float(pos[1]))

            labels: dict[str, float] = {}
            if "label" in ent:
                labels[str(ent["label"])] = float(ent.get("label_confidence", 1.0))

            features.append(
                PerceptualFeatureVector(
                    feature_id=_gen_id("feat_grid"),
                    observation_id=envelope.observation_id,
                    modality=envelope.source_modality,
                    channel_id=envelope.source_channel_id,
                    embedding=embedding,
                    spatial_extent=spatial,
                    semantic_labels=labels,
                    feature_metadata={"entity_id": ent_id, "raw_color": int(color)},
                    provenance_hash=envelope.payload_hash,
                    timestamp=envelope.wall_timestamp,
                )
            )

        return features


class KinematicFeatureExtractor:
    """Extracts normalised feature vectors from proprioceptive / kinesthetic streams.

    Expected ``metadata`` layout:
        {
            "entities": [
                {
                    "id": str,
                    "position": (x, y) | (x, y, z),
                    "velocity": (vx, vy) | (vx, vy, vz),
                    "mass": float,         # optional
                },
                ...
            ],
        }
    """

    @property
    def extractor_name(self) -> str:
        return "KinematicFeatureExtractor"

    def can_extract(self, envelope: ObservationEnvelope) -> bool:
        return (
            envelope.source_modality
            in (
                ModalitySource.KINESTHETIC,
                ModalitySource.PROPRIOCEPTIVE,
            )
            and "entities" in envelope.metadata
        )

    def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        features: list[PerceptualFeatureVector] = []
        entities = envelope.metadata.get("entities", [])

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("kin")))
            pos = tuple(float(x) for x in ent.get("position", (0.0, 0.0)))
            vel = tuple(float(x) for x in ent.get("velocity", (0.0, 0.0)))
            mass = float(ent.get("mass", 1.0))

            speed = math.sqrt(sum(v * v for v in vel))

            embedding = pos + vel + (speed, math.log1p(mass))

            features.append(
                PerceptualFeatureVector(
                    feature_id=_gen_id("feat_kin"),
                    observation_id=envelope.observation_id,
                    modality=envelope.source_modality,
                    channel_id=envelope.source_channel_id,
                    embedding=embedding,
                    spatial_extent=pos,
                    feature_metadata={"entity_id": ent_id, "speed": speed},
                    provenance_hash=envelope.payload_hash,
                    timestamp=envelope.wall_timestamp,
                )
            )

        return features


class SemanticTextFeatureExtractor:
    """Extracts feature vectors from text / dialogue observation metadata.

    This extractor operates on pre-tokenised or pre-embedded text
    representations stored in ``envelope.metadata``.  It does **not**
    invoke an LLM — any neural encoding must have already been applied
    by the perception adapter before the observation reached SGE.

    Expected ``metadata`` layout:
        {
            "tokens": list[str],                   # optional raw tokens
            "token_embeddings": list[list[float]],  # optional pre-computed
            "sentence_embedding": list[float],      # optional aggregate
            "named_entities": [                     # optional NER output
                {"text": str, "type": str, "confidence": float},
                ...
            ],
        }
    """

    @property
    def extractor_name(self) -> str:
        return "SemanticTextFeatureExtractor"

    def can_extract(self, envelope: ObservationEnvelope) -> bool:
        return envelope.source_modality in (
            ModalitySource.TEXT,
            ModalitySource.AUDIO,
            ModalitySource.SEMANTIC_LLM_ADAPTER,
        ) and (
            "sentence_embedding" in envelope.metadata
            or "named_entities" in envelope.metadata
            or "tokens" in envelope.metadata
        )

    def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        features: list[PerceptualFeatureVector] = []

        # Sentence-level embedding
        sent_emb = envelope.metadata.get("sentence_embedding")
        if sent_emb:
            features.append(
                PerceptualFeatureVector(
                    feature_id=_gen_id("feat_txt_sent"),
                    observation_id=envelope.observation_id,
                    modality=envelope.source_modality,
                    channel_id=envelope.source_channel_id,
                    embedding=tuple(float(x) for x in sent_emb),
                    feature_metadata={"type": "sentence_embedding"},
                    provenance_hash=envelope.payload_hash,
                    timestamp=envelope.wall_timestamp,
                )
            )

        # Named entity features (one per entity)
        for ne in envelope.metadata.get("named_entities", []):
            text = str(ne.get("text", ""))
            ne_type = str(ne.get("type", "UNKNOWN"))
            conf = float(ne.get("confidence", 1.0))

            # Simple bag-of-characters hash for a deterministic embedding stub
            char_hash = tuple(float(ord(c) % 100) / 100.0 for c in text[:8].ljust(8))

            features.append(
                PerceptualFeatureVector(
                    feature_id=_gen_id("feat_txt_ne"),
                    observation_id=envelope.observation_id,
                    modality=envelope.source_modality,
                    channel_id=envelope.source_channel_id,
                    embedding=char_hash,
                    semantic_labels={ne_type: conf},
                    feature_metadata={"text": text, "ne_type": ne_type},
                    provenance_hash=envelope.payload_hash,
                    timestamp=envelope.wall_timestamp,
                )
            )

        return features


class FeatureExtractorRegistry:
    """Runtime registry of modality-specific feature extractors.

    Dispatches each ObservationEnvelope to every extractor that can handle it,
    collecting and deduplicating the produced feature vectors.
    """

    def __init__(self, extractors: list[FeatureExtractor] | None = None) -> None:
        self._extractors: list[FeatureExtractor] = list(extractors or [])

    def register(self, extractor: FeatureExtractor) -> None:
        """Register a new feature extractor."""
        self._extractors.append(extractor)

    def extract_all(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        """Run all compatible extractors on the envelope and collect results."""
        results: list[PerceptualFeatureVector] = []
        for ext in self._extractors:
            if ext.can_extract(envelope):
                results.extend(ext.extract(envelope))
        return results

    @classmethod
    def default(cls) -> FeatureExtractorRegistry:
        """Returns a registry pre-populated with all built-in domain-agnostic extractors."""
        return cls(
            extractors=[
                SpatialGridFeatureExtractor(),
                KinematicFeatureExtractor(),
                SemanticTextFeatureExtractor(),
            ]
        )
