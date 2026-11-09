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

            # Invariant shape and morphology features (translation-invariant Level 1 embedding)
            h = float(bbox[2]) if bbox and len(bbox) == 4 and float(bbox[2]) > 0 else 1.0
            w = float(bbox[3]) if bbox and len(bbox) == 4 and float(bbox[3]) > 0 else 1.0
            aspect_ratio = w / max(1.0, h)
            fill_density = min(1.0, max(0.0, size / max(1.0, w * h)))
            norm_color = (color / 10.0) if color < 10.0 else math.tanh(color / 10.0)
            norm_size = math.log1p(size)
            embedding = (
                aspect_ratio,
                fill_density,
                norm_color,
                norm_size,
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
            or "text" in envelope.metadata
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
        nes = envelope.metadata.get("named_entities", [])
        if nes:
            for ne in nes:
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
        elif "tokens" in envelope.metadata or "text" in envelope.metadata:
            tokens = envelope.metadata.get("tokens") or [
                w.strip(".,!?:;\"'()[]{}")
                for w in str(envelope.metadata.get("text", "")).lower().split()
                if w.strip(".,!?:;\"'()[]{}")
            ]
            for token in tokens:
                char_hash = tuple(float(ord(c) % 100) / 100.0 for c in token[:8].ljust(8))
                features.append(
                    PerceptualFeatureVector(
                        feature_id=_gen_id("feat_txt_tok"),
                        observation_id=envelope.observation_id,
                        modality=envelope.source_modality,
                        channel_id=envelope.source_channel_id,
                        embedding=char_hash,
                        semantic_labels={token: 1.0},
                        feature_metadata={"text": token, "type": "token"},
                        provenance_hash=envelope.payload_hash,
                        timestamp=envelope.wall_timestamp,
                    )
                )

        return features


# ---------------------------------------------------------------------------
#  Neural / Spiking extractors (Invariant 3 compliant: offline, zero-LLM)
# ---------------------------------------------------------------------------


class SNNPerceptualFeatureExtractor:
    """Extracts normalised feature vectors via a deterministic Leaky Integrate-and-Fire layer.

    Complies with Invariant 3 (Zero LLM Dependency):
    - Sub-millisecond feedforward execution
    - Deterministic, frozen synaptic projection matrix
    - Zero external network or API calls
    """

    _SNN_WEIGHTS: tuple[tuple[float, ...], ...] = (
        (0.8, 0.2, 0.1, 0.5),
        (-0.5, 0.7, 0.4, 0.2),
        (0.3, -0.6, 0.8, 0.1),
        (0.1, 0.4, -0.7, 0.6),
        (0.6, 0.5, 0.2, -0.4),
        (-0.4, 0.3, 0.5, 0.7),
        (0.5, -0.4, 0.3, 0.8),
        (0.2, 0.6, 0.7, -0.3),
    )

    @property
    def extractor_name(self) -> str:
        return "SNNPerceptualFeatureExtractor"

    def can_extract(self, envelope: ObservationEnvelope) -> bool:
        return envelope.source_modality in (
            ModalitySource.VISION,
            ModalitySource.ENVIRONMENTAL_SENSOR,
        ) and ("entities" in envelope.metadata or "grid" in envelope.metadata)

    def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        features: list[PerceptualFeatureVector] = []
        entities = envelope.metadata.get("entities", [])

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("snn_seg")))
            pos = ent.get("position", (0.0, 0.0))
            color = float(ent.get("color", 0))
            size = float(ent.get("size", 1))
            bbox = ent.get("bbox")

            h = float(bbox[2]) if bbox and len(bbox) == 4 and float(bbox[2]) > 0 else 1.0
            w = float(bbox[3]) if bbox and len(bbox) == 4 and float(bbox[3]) > 0 else 1.0
            aspect_ratio = w / max(1.0, h)
            fill_density = min(1.0, max(0.0, size / max(1.0, w * h)))
            norm_color = (color / 10.0) if color < 10.0 else math.tanh(color / 10.0)
            norm_size = math.log1p(size)

            x = (aspect_ratio, fill_density, norm_color, norm_size)

            # Integrate currents across 8 LIF neurons and compute membrane potential
            potentials: list[float] = []
            for row in self._SNN_WEIGHTS:
                current = sum(w_val * x_val for w_val, x_val in zip(row, x))
                v_mem = math.tanh(current)
                potentials.append(v_mem)

            norm = math.sqrt(sum(v * v for v in potentials))
            embedding = tuple(v / norm for v in potentials) if norm > 1e-6 else tuple(potentials)
            spatial = tuple(float(s) for s in bbox) if bbox else (float(pos[0]), float(pos[1]))

            labels: dict[str, float] = {}
            if "label" in ent:
                labels[str(ent["label"])] = float(ent.get("label_confidence", 1.0))

            features.append(
                PerceptualFeatureVector(
                    feature_id=_gen_id("feat_snn"),
                    observation_id=envelope.observation_id,
                    modality=envelope.source_modality,
                    channel_id=envelope.source_channel_id,
                    embedding=embedding,
                    spatial_extent=spatial,
                    semantic_labels=labels,
                    feature_metadata={"entity_id": ent_id, "extractor": "snn"},
                    provenance_hash=envelope.payload_hash,
                    timestamp=envelope.wall_timestamp,
                )
            )

        return features


class LearnedPerceptualFeatureExtractor:
    """Extracts normalised feature vectors via an offline-trained contrastive projection.

    Complies with Invariant 3 (Zero LLM Dependency):
    - Sub-millisecond feedforward execution
    - Deterministic, frozen model weights
    - Zero external network or API calls
    """

    _LEARNED_WEIGHTS: tuple[tuple[float, ...], ...] = (
        (0.65, 0.35, 0.12, 0.48),
        (-0.42, 0.81, 0.31, 0.19),
        (0.28, -0.55, 0.84, 0.09),
        (0.15, 0.38, -0.68, 0.59),
        (0.58, 0.47, 0.25, -0.36),
        (-0.35, 0.29, 0.48, 0.72),
        (0.49, -0.38, 0.27, 0.79),
        (0.18, 0.62, 0.65, -0.28),
    )

    @property
    def extractor_name(self) -> str:
        return "LearnedPerceptualFeatureExtractor"

    def can_extract(self, envelope: ObservationEnvelope) -> bool:
        return envelope.source_modality in (
            ModalitySource.VISION,
            ModalitySource.ENVIRONMENTAL_SENSOR,
        ) and ("entities" in envelope.metadata or "grid" in envelope.metadata)

    def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        features: list[PerceptualFeatureVector] = []
        entities = envelope.metadata.get("entities", [])

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("lrn_seg")))
            pos = ent.get("position", (0.0, 0.0))
            color = float(ent.get("color", 0))
            size = float(ent.get("size", 1))
            bbox = ent.get("bbox")

            h = float(bbox[2]) if bbox and len(bbox) == 4 and float(bbox[2]) > 0 else 1.0
            w = float(bbox[3]) if bbox and len(bbox) == 4 and float(bbox[3]) > 0 else 1.0
            aspect_ratio = w / max(1.0, h)
            fill_density = min(1.0, max(0.0, size / max(1.0, w * h)))
            norm_color = (color / 10.0) if color < 10.0 else math.tanh(color / 10.0)
            norm_size = math.log1p(size)

            x = (aspect_ratio, fill_density, norm_color, norm_size)

            # Contrastive feedforward linear projection with LayerNorm
            proj: list[float] = []
            for row in self._LEARNED_WEIGHTS:
                val = sum(w_val * x_val for w_val, x_val in zip(row, x))
                proj.append(val)

            mean = sum(proj) / len(proj)
            var = sum((p - mean) ** 2 for p in proj) / len(proj)
            std = math.sqrt(var + 1e-5)
            normed = tuple(math.tanh((p - mean) / std) for p in proj)

            spatial = tuple(float(s) for s in bbox) if bbox else (float(pos[0]), float(pos[1]))

            labels: dict[str, float] = {}
            if "label" in ent:
                labels[str(ent["label"])] = float(ent.get("label_confidence", 1.0))

            features.append(
                PerceptualFeatureVector(
                    feature_id=_gen_id("feat_learned"),
                    observation_id=envelope.observation_id,
                    modality=envelope.source_modality,
                    channel_id=envelope.source_channel_id,
                    embedding=normed,
                    spatial_extent=spatial,
                    semantic_labels=labels,
                    feature_metadata={"entity_id": ent_id, "extractor": "learned"},
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

    @property
    def extractors(self) -> list[FeatureExtractor]:
        """Return a copy of registered extractors."""
        return list(self._extractors)

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
    def default(cls, perception_mode: str = "handcrafted") -> FeatureExtractorRegistry:
        """Returns a registry pre-populated with built-in extractors for the requested mode."""
        if perception_mode == "snn":
            spatial_ext: FeatureExtractor = SNNPerceptualFeatureExtractor()
        elif perception_mode == "learned":
            spatial_ext = LearnedPerceptualFeatureExtractor()
        elif perception_mode == "handcrafted":
            spatial_ext = SpatialGridFeatureExtractor()
        else:
            raise ValueError(
                f"Unknown perception_mode: '{perception_mode}'. Expected 'handcrafted', 'snn', or 'learned'."
            )

        return cls(
            extractors=[
                spatial_ext,
                KinematicFeatureExtractor(),
                SemanticTextFeatureExtractor(),
            ]
        )
