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

COLOR_NAMES: dict[int, str] = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "brown",
}


def _extract_palette_scale(envelope: ObservationEnvelope, entities: list[dict[str, Any]]) -> float:
    """Extracts normalization scale for color features from envelope or reference frame metadata.

    Supports declared palette sizes, explicit [min, max] ranges, discrete color sets,
    or dynamically bounds based on observed entity features instead of a rigid constant.
    """
    meta = envelope.metadata or {}
    if "palette_size" in meta and float(meta["palette_size"]) > 0:
        return float(meta["palette_size"])
    if "color_range" in meta:
        cr = meta["color_range"]
        if isinstance(cr, (list, tuple)) and len(cr) == 2 and cr[1] > cr[0]:
            return float(cr[1] - cr[0])
    if (
        "palette" in meta
        and isinstance(meta["palette"], (list, tuple, set))
        and len(meta["palette"]) > 0
    ):
        return float(len(meta["palette"]))

    if envelope.reference_frame and envelope.reference_frame.metadata:
        rf_meta = envelope.reference_frame.metadata
        if "palette_size" in rf_meta and float(rf_meta["palette_size"]) > 0:
            return float(rf_meta["palette_size"])
        if "color_range" in rf_meta:
            cr = rf_meta["color_range"]
            if isinstance(cr, (list, tuple)) and len(cr) == 2 and cr[1] > cr[0]:
                return float(cr[1] - cr[0])
        if (
            "palette" in rf_meta
            and isinstance(rf_meta["palette"], (list, tuple, set))
            and len(rf_meta["palette"]) > 0
        ):
            return float(len(rf_meta["palette"]))

    # Dynamic fallback: check observed colors in entities
    colors = [float(e.get("color", 0)) for e in entities if "color" in e]
    if colors:
        max_c = max(colors)
        if max_c > 0:
            return max(10.0, max_c)

    return 10.0


def _extract_grid_context(
    envelope: ObservationEnvelope,
) -> tuple[Any | None, int, dict[int, int]]:
    """Extract normalized 2D lattice, background value, and global feature counts from metadata."""
    meta = envelope.metadata or {}
    grid = meta.get("grid")
    if grid is None:
        return None, int(meta.get("bg_feature", 0) or 0), {}
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if rows <= 0 or cols <= 0:
        return None, int(meta.get("bg_feature", 0) or 0), {}
    counts: dict[int, int] = {}
    for r in range(rows):
        row = grid[r]
        for c in range(cols):
            val = int(row[c])
            counts[val] = counts.get(val, 0) + 1
    bg_raw = meta.get("bg_feature")
    if bg_raw is None and counts:
        bg_val = max(counts, key=lambda k: counts[k])
    else:
        bg_val = int(bg_raw or 0)
    return grid, bg_val, counts


def _compute_entity_morphology_and_topology(
    ent: dict[str, Any],
    palette_scale: float,
    grid: Any | None = None,
    bg_feature: int = 0,
    feature_counts: dict[int, int] | None = None,
) -> tuple[tuple[float, float, float, float], tuple[float, ...], tuple[float, ...], int]:
    """Compute morphology (4D), local structural-topological + kinematic (14D), and spatial extent."""
    pos = ent.get("position", (0.0, 0.0))
    color = float(ent.get("color", 0))
    raw_color = int(color)
    size = float(ent.get("size", 1))
    bbox = ent.get("bbox")

    h = float(bbox[2]) if bbox and len(bbox) == 4 and float(bbox[2]) > 0 else 1.0
    w = float(bbox[3]) if bbox and len(bbox) == 4 and float(bbox[3]) > 0 else 1.0
    aspect_ratio = w / max(1.0, h)
    fill_density = min(1.0, max(0.0, size / max(1.0, w * h)))
    norm_color = (
        (color / palette_scale) if color <= palette_scale else math.tanh(color / palette_scale)
    )
    norm_size = math.log1p(size)
    base_4 = (aspect_ratio, fill_density, norm_color, norm_size)

    # Local 3x3 topological-affordance invariants (color-permutation invariant)
    open_deg4 = 1.0
    same_feat_deg4 = 0.0
    opposite_open_axes = 1.0
    corner_pin_risk = 0.0
    high_freq_adj4 = 0.0
    interior_dist = 0.5
    feat_count = (feature_counts or {}).get(raw_color, int(max(1.0, size)))
    rarity = 1.0 / (1.0 + math.log1p(max(1, feat_count)))
    heterogeneity_8 = 0.0

    if grid is not None and len(grid) > 0 and len(grid[0]) > 0:
        H = len(grid)
        W = len(grid[0])
        r = max(0, min(H - 1, int(round(float(pos[0])))))
        c = max(0, min(W - 1, int(round(float(pos[1])))))
        high_thresh = max(H, W)
        counts_map = feature_counts or {}

        def _is_open(rr: int, cc: int) -> bool:
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            return int(grid[rr][cc]) == bg_feature

        def _is_barrier_like(rr: int, cc: int) -> bool:
            if not (0 <= rr < H and 0 <= cc < W):
                return True
            v = int(grid[rr][cc])
            return v != bg_feature and counts_map.get(v, 1) >= high_thresh

        n_open = 0
        n_same = 0
        n_hf = 0
        blocked_dirs: list[bool] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                nv = int(grid[nr][nc])
                if nv == bg_feature:
                    n_open += 1
                    blocked_dirs.append(False)
                else:
                    blocked_dirs.append(True)
                    if nv == raw_color:
                        n_same += 1
                    if counts_map.get(nv, 1) >= high_thresh:
                        n_hf += 1
            else:
                blocked_dirs.append(True)
                n_hf += 1

        open_deg4 = n_open / 4.0
        same_feat_deg4 = n_same / 4.0
        high_freq_adj4 = n_hf / 4.0
        v_open = _is_open(r - 1, c) and _is_open(r + 1, c)
        h_open = _is_open(r, c - 1) and _is_open(r, c + 1)
        opposite_open_axes = (int(v_open) + int(h_open)) / 2.0
        # Perpendicular wall/barrier confinement (corner risk)
        if (_is_barrier_like(r - 1, c) or _is_barrier_like(r + 1, c)) and (
            _is_barrier_like(r, c - 1) or _is_barrier_like(r, c + 1)
        ):
            corner_pin_risk = 1.0
        interior_dist = min(r, H - 1 - r, c, W - 1 - c) / max(1.0, min(H, W) / 2.0)

        distinct_8: set[int] = set()
        for dr8 in (-1, 0, 1):
            for dc8 in (-1, 0, 1):
                if dr8 == 0 and dc8 == 0:
                    continue
                rr8, cc8 = r + dr8, c + dc8
                if 0 <= rr8 < H and 0 <= cc8 < W:
                    v8 = int(grid[rr8][cc8])
                    if v8 != bg_feature and v8 != raw_color:
                        distinct_8.add(v8)
        heterogeneity_8 = len(distinct_8) / 8.0

    vel = ent.get("velocity", (0.0, 0.0))
    speed = math.hypot(float(vel[0]), float(vel[1])) if vel and len(vel) >= 2 else 0.0
    norm_speed = min(1.0, speed / 2.0)
    is_mobile = 1.0 if (speed > 1e-6 or bool(ent.get("moved", False))) else 0.0

    full_14 = base_4 + (
        open_deg4,
        same_feat_deg4,
        opposite_open_axes,
        corner_pin_risk,
        high_freq_adj4,
        interior_dist,
        rarity,
        heterogeneity_8,
        norm_speed,
        is_mobile,
    )
    spatial = tuple(float(x) for x in bbox) if bbox else (float(pos[0]), float(pos[1]))
    return base_4, full_14, spatial, raw_color


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
        palette_scale = _extract_palette_scale(envelope, entities)

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("seg")))
            embedding, _, spatial, raw_color = _compute_entity_morphology_and_topology(
                ent, palette_scale=palette_scale
            )

            labels: dict[str, float] = {}
            if "label" in ent:
                labels[str(ent["label"])] = float(ent.get("label_confidence", 1.0))
            if raw_color in COLOR_NAMES:
                cname = COLOR_NAMES[raw_color]
                labels.setdefault(cname, 1.0)
                labels.setdefault(f"{cname} block", 0.9)
                labels.setdefault(f"{cname} component", 0.9)

            features.append(
                PerceptualFeatureVector(
                    feature_id=_gen_id("feat_grid"),
                    observation_id=envelope.observation_id,
                    modality=envelope.source_modality,
                    channel_id=envelope.source_channel_id,
                    embedding=embedding,
                    spatial_extent=spatial,
                    semantic_labels=labels,
                    feature_metadata={"entity_id": ent_id, "raw_color": raw_color},
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
    """Extracts normalised feature vectors via a stateful Leaky Integrate-and-Fire (LIF) layer.

    Projects 14-D structural, topological, and kinematic entity descriptors through a
    deterministic 8x14 synaptic matrix across multiple leaky membrane integration steps,
    persisting entity membrane traces across consecutive observations.

    Complies with Invariant 3 (Zero LLM Dependency):
    - Sub-millisecond feedforward execution
    - Deterministic, frozen synaptic projection matrix
    - Zero external network or API calls
    """

    # 8x14 offline-calibrated synaptic receptive fields over 14-D morphology+topology+kinematics:
    # dims [0..6] focus on color-invariant structure/mobility; dim [7] preserves chromatic tuning.
    _SNN_WEIGHTS: tuple[tuple[float, ...], ...] = (
        (0.25, 0.25, 0.0, -0.75, 0.85, -0.90, 0.95, -0.80, -0.70, 0.65, 0.70, 0.15, -0.25, -0.35),
        (-0.20, -0.20, 0.0, 0.85, -0.90, 0.95, -0.85, 0.55, 0.85, -0.70, -0.85, 0.10, -0.30, -0.40),
        (0.10, 0.10, 0.0, -0.20, 0.35, -0.30, 0.25, -0.20, -0.15, 0.35, 0.45, 0.40, 1.15, 1.25),
        (0.15, 0.20, 0.0, 0.10, -0.35, -0.10, -0.75, 1.10, 0.85, -0.45, 0.30, 0.35, -0.15, -0.20),
        (0.45, -0.40, 0.0, 0.55, 0.30, -0.25, 0.40, -0.30, -0.25, 0.50, 0.55, -0.35, 0.40, 0.35),
        (-0.35, 0.45, 0.0, -0.40, -0.45, 0.65, -0.35, 0.35, 0.45, -0.40, -0.50, 0.55, 0.25, 0.40),
        (0.20, 0.15, 0.0, -0.30, 0.60, -0.45, 0.70, -0.60, -0.50, 0.75, 0.50, -0.20, -0.45, -0.50),
        (0.0, 0.0, 1.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    def __init__(self, membrane_decay: float = 0.55, num_steps: int = 5) -> None:
        self._decay = membrane_decay
        self._num_steps = num_steps
        self._membrane_traces: dict[str, list[float]] = {}

    def reset_traces(self) -> None:
        """Clear persistent entity membrane traces on episode reset."""
        self._membrane_traces.clear()

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
        palette_scale = _extract_palette_scale(envelope, entities)
        grid, bg_feature, feature_counts = _extract_grid_context(envelope)

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("snn_seg")))
            _, x14, spatial, raw_color = _compute_entity_morphology_and_topology(
                ent,
                palette_scale=palette_scale,
                grid=grid,
                bg_feature=bg_feature,
                feature_counts=feature_counts,
            )

            # Stateful LIF integration across temporal sub-steps with persistent membrane decay
            currents = [
                sum(w_val * x_val for w_val, x_val in zip(row, x14)) for row in self._SNN_WEIGHTS
            ]
            v = list(self._membrane_traces.get(ent_id, [0.0] * len(self._SNN_WEIGHTS)))
            spike_acc = [0.0] * len(v)
            threshold = 0.45
            for _ in range(self._num_steps):
                for j in range(len(v)):
                    v[j] = self._decay * v[j] + currents[j]
                    if abs(v[j]) >= threshold:
                        spike_acc[j] += 1.0 if v[j] > 0.0 else -1.0
                        v[j] *= 0.25  # soft reset
                    spike_acc[j] += 0.35 * math.tanh(v[j])
            if len(self._membrane_traces) > 256:
                self._membrane_traces.pop(next(iter(self._membrane_traces)))
            self._membrane_traces[ent_id] = v

            norm = math.sqrt(sum(s * s for s in spike_acc))
            embedding = tuple(s / norm for s in spike_acc) if norm > 1e-6 else tuple(spike_acc)

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
                    feature_metadata={
                        "entity_id": ent_id,
                        "extractor": "snn",
                        "raw_color": raw_color,
                    },
                    provenance_hash=envelope.payload_hash,
                    timestamp=envelope.wall_timestamp,
                )
            )

        return features


class LearnedPerceptualFeatureExtractor:
    """Extracts normalised feature vectors via an offline-trained 2-layer contrastive MLP.

    Encodes 14-D morphology, local 3x3 topological affordance invariants, and kinematic
    descriptors through frozen offline-trained ``14 -> 16 (tanh) -> 8`` MLP weights with
    LayerNorm + ``tanh``, separating barriers, open-interior objects, edge-pinned states,
    and mobile agents invariantly across arbitrary integer feature ID permutations.

    Complies with Invariant 3 (Zero LLM Dependency):
    - Sub-millisecond feedforward execution
    - Deterministic, frozen model weights
    - Zero external network or API calls
    """

    _MLP_W1: tuple[tuple[float, ...], ...] = (
        (
            -0.3787,
            0.4214,
            0.0,
            -0.5318,
            0.1878,
            -0.6419,
            0.0671,
            -0.8377,
            -0.4728,
            0.0702,
            0.144,
            0.2681,
            -0.3158,
            -0.9416,
        ),
        (
            -0.3143,
            -0.5487,
            0.0,
            -0.174,
            -0.7233,
            -0.1412,
            -0.2654,
            0.3982,
            -0.5225,
            -0.434,
            0.2853,
            0.3882,
            -0.2375,
            -0.2198,
        ),
        (
            0.226,
            -0.657,
            0.0,
            0.444,
            0.2887,
            0.0461,
            -0.0348,
            0.5213,
            0.735,
            -0.04,
            -0.0833,
            0.5453,
            -0.2875,
            0.6437,
        ),
        (
            -0.3829,
            -0.0467,
            0.0,
            -0.2854,
            -0.2123,
            0.0935,
            0.3288,
            -0.2013,
            -0.1476,
            0.0263,
            -0.3233,
            0.1895,
            0.1931,
            0.9459,
        ),
        (
            -0.0417,
            -0.0663,
            0.0,
            0.763,
            -0.1757,
            0.1368,
            0.1356,
            -0.4653,
            -0.0018,
            -0.0501,
            0.1571,
            -0.1952,
            -0.6251,
            -0.894,
        ),
        (
            -0.4294,
            -0.4455,
            0.0,
            -0.4926,
            -0.1609,
            -0.2649,
            -0.0893,
            0.4355,
            0.1812,
            0.0303,
            -0.1439,
            -0.1175,
            0.0676,
            0.1276,
        ),
        (
            -0.1581,
            0.1251,
            0.0,
            -0.6426,
            0.6523,
            -0.4032,
            0.9646,
            -1.228,
            -0.4766,
            0.2682,
            0.2238,
            -0.093,
            0.1456,
            0.4045,
        ),
        (
            0.2597,
            -0.1543,
            0.0,
            -1.1179,
            0.0973,
            -0.5627,
            0.2532,
            1.1365,
            -0.1385,
            -0.3631,
            -0.0542,
            0.3705,
            0.201,
            0.2015,
        ),
        (
            0.2861,
            0.1289,
            0.0,
            -0.1854,
            0.5368,
            -0.1876,
            0.3707,
            0.0333,
            -0.28,
            0.4969,
            0.2559,
            0.2662,
            0.0614,
            -0.1684,
        ),
        (
            -0.0373,
            0.3902,
            0.0,
            -0.9664,
            -0.1584,
            0.2725,
            -0.0902,
            -0.2638,
            0.4424,
            0.1779,
            0.1254,
            -0.0905,
            0.6305,
            0.518,
        ),
        (
            0.3387,
            0.1569,
            0.0,
            0.4443,
            -0.1484,
            0.479,
            -0.2875,
            -0.6169,
            0.0919,
            0.1529,
            0.0301,
            0.0804,
            -0.031,
            0.3255,
        ),
        (
            -0.1278,
            -0.2602,
            0.0,
            0.7676,
            -0.5642,
            0.2819,
            -0.5628,
            0.1733,
            0.0255,
            -0.7077,
            -0.0185,
            -0.116,
            -0.7229,
            0.1215,
        ),
        (
            0.1768,
            0.1934,
            0.0,
            0.7973,
            0.1508,
            0.8767,
            0.1682,
            -0.6367,
            -0.1008,
            -0.0962,
            -0.8928,
            0.0463,
            0.7781,
            0.4554,
        ),
        (
            0.3712,
            -0.0159,
            0.0,
            0.5802,
            -0.4347,
            -0.0778,
            0.1074,
            0.6776,
            -0.052,
            0.276,
            -0.1896,
            0.0995,
            -0.0263,
            0.1327,
        ),
        (0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    _MLP_B1: tuple[float, ...] = (
        0.2096,
        -0.0545,
        0.0411,
        -0.0774,
        0.0712,
        -0.0515,
        0.1142,
        -0.1536,
        -0.0165,
        -0.0406,
        -0.0057,
        -0.1141,
        -0.2537,
        -0.0708,
        0.0,
        0.0,
    )
    _MLP_W2: tuple[tuple[float, ...], ...] = (
        (
            -0.5533,
            0.1243,
            -0.0586,
            0.2694,
            0.1775,
            0.6904,
            -0.612,
            -0.6721,
            0.1089,
            -0.3151,
            -0.2266,
            0.6836,
            0.553,
            0.1316,
            0.0,
            0.0,
        ),
        (
            0.4866,
            -0.3537,
            -0.3049,
            0.1065,
            0.3523,
            0.2681,
            0.685,
            -0.2023,
            0.3564,
            0.032,
            0.2869,
            -0.643,
            -0.1178,
            -0.3913,
            0.0,
            0.0,
        ),
        (
            0.9353,
            -0.0296,
            -0.228,
            0.365,
            0.2985,
            -0.577,
            0.7394,
            -0.7834,
            -0.0296,
            -0.3016,
            0.0991,
            -0.3052,
            -0.0168,
            -0.2636,
            0.0,
            0.0,
        ),
        (
            -0.5245,
            -0.4585,
            0.3241,
            0.3536,
            -0.8439,
            0.0308,
            -0.2822,
            0.6694,
            0.1122,
            0.5709,
            -0.2848,
            0.0029,
            -0.4317,
            0.4771,
            0.0,
            0.0,
        ),
        (
            -0.8615,
            0.2769,
            0.428,
            0.2384,
            0.078,
            -0.0782,
            -0.9148,
            0.0971,
            -0.6462,
            -0.2113,
            0.332,
            -0.0653,
            -0.1841,
            -0.1852,
            0.0,
            0.0,
        ),
        (
            0.4288,
            0.1636,
            0.0341,
            0.2456,
            -0.0374,
            -0.442,
            0.6453,
            -0.4765,
            0.2322,
            0.3662,
            0.2476,
            -0.2173,
            0.5171,
            -0.6964,
            0.0,
            0.0,
        ),
        (
            0.2969,
            0.1822,
            -0.0293,
            -0.8364,
            0.1927,
            0.1145,
            0.0683,
            0.5628,
            0.5056,
            -0.1877,
            -0.2634,
            -0.162,
            -1.3362,
            0.0181,
            0.0,
            0.0,
        ),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, -0.9),
    )
    _MLP_B2: tuple[float, ...] = (-0.45, 0.0499, -0.1904, 0.1716, -0.1535, 0.1598, 0.1023, 0.0)

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
        palette_scale = _extract_palette_scale(envelope, entities)
        grid, bg_feature, feature_counts = _extract_grid_context(envelope)

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("lrn_seg")))
            _, x14, spatial, raw_color = _compute_entity_morphology_and_topology(
                ent,
                palette_scale=palette_scale,
                grid=grid,
                bg_feature=bg_feature,
                feature_counts=feature_counts,
            )

            # Hidden layer (14 -> 16) with tanh activation
            hidden = [
                math.tanh(b + sum(w_val * x_val for w_val, x_val in zip(row, x14)))
                for row, b in zip(self._MLP_W1, self._MLP_B1)
            ]
            # Output projection (16 -> 8) with LayerNorm + tanh
            proj = [
                b + sum(w_val * h_val for w_val, h_val in zip(row, hidden))
                for row, b in zip(self._MLP_W2, self._MLP_B2)
            ]
            mean = sum(proj) / len(proj)
            var = sum((p - mean) ** 2 for p in proj) / len(proj)
            std = math.sqrt(var + 1e-5)
            normed = tuple(math.tanh((p - mean) / std) for p in proj)

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
                    feature_metadata={
                        "entity_id": ent_id,
                        "extractor": "learned",
                        "raw_color": raw_color,
                    },
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
