from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from hbllm.hcir.grounding.contracts.enums import ModalitySource
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.features.calibrated_weights import (
    INIT_B1,
    INIT_B2,
    INIT_CONV_B,
    INIT_CONV_W,
    INIT_SYN_BIAS,
    INIT_SYN_DELTA,
    INIT_SYN_KIN,
    INIT_SYN_PATCH,
    INIT_W1,
    INIT_W2,
)


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

_COMPATIBLE_PRIMARY_ROLES: frozenset[str] = frozenset({"goal", "receptacle"})


def _roles_conflict_pair(role_a: str, role_b: str) -> bool:
    """Return True if two physical affordance roles should separate in representation space."""
    if role_a == role_b:
        return False
    return not (role_a in _COMPATIBLE_PRIMARY_ROLES and role_b in _COMPATIBLE_PRIMARY_ROLES)


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
    """Compute handcrafted morphology (4D), topological invariants (14D), and spatial extent."""
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
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                nv = int(grid[nr][nc])
                if nv == bg_feature:
                    n_open += 1
                else:
                    if nv == raw_color:
                        n_same += 1
                    if counts_map.get(nv, 1) >= high_thresh:
                        n_hf += 1
            else:
                n_hf += 1

        open_deg4 = n_open / 4.0
        same_feat_deg4 = n_same / 4.0
        high_freq_adj4 = n_hf / 4.0
        v_open = _is_open(r - 1, c) and _is_open(r + 1, c)
        h_open = _is_open(r, c - 1) and _is_open(r, c + 1)
        opposite_open_axes = (int(v_open) + int(h_open)) / 2.0
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


def _extract_raw_patch_and_scalars(
    ent: dict[str, Any],
    grid: Any | None = None,
    bg_feature: int = 0,
    feature_counts: dict[int, int] | None = None,
    prev_grid: list[list[int]] | None = None,
) -> tuple[
    list[list[int]],
    list[list[float]],
    tuple[float, float, float, float, float],
    tuple[float, ...],
    int,
]:
    """Extract a palette-invariant 5x5x4 raw sensory patch (channel index 0..3),
    a 5x5 temporal change map, and 5 raw physical/census/kinematic scalars.

    Channel map at each cell of the 5x5 receptive field:
    - ``0``: background
    - ``1``: same symbol as target entity
    - ``2``: foreign non-background symbol
    - ``3``: out-of-bounds image border
    """
    pos = ent.get("position", (0.0, 0.0))
    sample_pos = ent.get("sample_pos") or pos
    raw_color = int(float(ent.get("color", 0)))
    size = max(1.0, float(ent.get("size", 1)))
    bbox = ent.get("bbox")
    h = float(bbox[2]) if bbox and len(bbox) == 4 and float(bbox[2]) > 0 else 1.0
    w = float(bbox[3]) if bbox and len(bbox) == 4 and float(bbox[3]) > 0 else 1.0

    patch_ch: list[list[int]] = [[0] * 5 for _ in range(5)]
    delta_patch: list[list[float]] = [[0.0] * 5 for _ in range(5)]

    if grid is not None and len(grid) > 0 and len(grid[0]) > 0:
        H = len(grid)
        W = len(grid[0])
        r0 = max(0, min(H - 1, int(round(float(sample_pos[0])))))
        c0 = max(0, min(W - 1, int(round(float(sample_pos[1])))))
        has_prev = prev_grid is not None and len(prev_grid) == H and len(prev_grid[0]) == W

        for py, dy in enumerate(range(-2, 3)):
            rr = r0 + dy
            row_ch = patch_ch[py]
            row_dp = delta_patch[py]
            if 0 <= rr < H:
                grid_row = grid[rr]
                prev_row = prev_grid[rr] if has_prev and prev_grid is not None else None
                for px, dx in enumerate(range(-2, 3)):
                    cc = c0 + dx
                    if 0 <= cc < W:
                        v = int(grid_row[cc])
                        if v == bg_feature:
                            row_ch[px] = 0
                        elif v == raw_color:
                            row_ch[px] = 1
                        else:
                            row_ch[px] = 2
                        if prev_row is not None and int(prev_row[cc]) != v:
                            row_dp[px] = 1.0
                    else:
                        row_ch[px] = 3
            else:
                for px in range(5):
                    row_ch[px] = 3

        same_count = (feature_counts or {}).get(raw_color, int(size))
        grid_area = max(1, H * W)
    else:
        patch_ch[2][2] = 1
        same_count = int(size)
        grid_area = max(16, int(size) * 4)

    vel = ent.get("velocity", (0.0, 0.0))
    speed = math.hypot(float(vel[0]), float(vel[1])) if vel and len(vel) >= 2 else 0.0
    norm_speed = min(1.0, speed / 2.0)
    moved_flag = 1.0 if (speed > 1e-6 or bool(ent.get("moved", False))) else 0.0
    if moved_flag > 0.0 and prev_grid is None:
        delta_patch[2][2] = 1.0

    scalars5 = (
        math.tanh(math.log1p(size) / 3.0),
        math.tanh((w - h) / max(1.0, w + h)),
        math.tanh(math.log1p(same_count) / max(1.0, math.log1p(grid_area))),
        norm_speed,
        moved_flag,
    )
    spatial = tuple(float(x) for x in bbox) if bbox else (float(pos[0]), float(pos[1]))
    return patch_ch, delta_patch, scalars5, spatial, raw_color


def _encode_text_ngrams(text: str, dim: int = 8) -> tuple[float, ...]:
    """Deterministic L2-normalized subword character n-gram projection (zero-LLM)."""
    clean = " ".join(text.lower().split())
    if not clean:
        val = 1.0 / math.sqrt(float(dim))
        return tuple(val for _ in range(dim))

    padded = f"<{clean}>"
    grams: list[str] = [clean]
    for n in (1, 2, 3):
        if len(padded) >= n:
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))

    acc = [0.0] * dim
    for g in grams:
        digest = hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest()
        h_int = int.from_bytes(digest, "little")
        bin_idx = h_int % dim
        sign = 1.0 if ((h_int >> 8) & 1) == 0 else -1.0
        acc[bin_idx] += sign

    norm = math.sqrt(sum(v * v for v in acc))
    if norm <= 1e-8:
        val = 1.0 / math.sqrt(float(dim))
        return tuple(val for _ in range(dim))
    return tuple(v / norm for v in acc)


class SpatialGridFeatureExtractor:
    """Extracts normalised handcrafted feature vectors from 2-D grid spatial observations."""

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
    """Extracts normalised feature vectors from proprioceptive / kinesthetic streams."""

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
    """Extracts feature vectors from text / dialogue observation metadata."""

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
                ngram_emb = _encode_text_ngrams(text)

                features.append(
                    PerceptualFeatureVector(
                        feature_id=_gen_id("feat_txt_ne"),
                        observation_id=envelope.observation_id,
                        modality=envelope.source_modality,
                        channel_id=envelope.source_channel_id,
                        embedding=ngram_emb,
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
                ngram_emb = _encode_text_ngrams(str(token))
                features.append(
                    PerceptualFeatureVector(
                        feature_id=_gen_id("feat_txt_tok"),
                        observation_id=envelope.observation_id,
                        modality=envelope.source_modality,
                        channel_id=envelope.source_channel_id,
                        embedding=ngram_emb,
                        semantic_labels={token: 1.0},
                        feature_metadata={"text": token, "type": "token"},
                        provenance_hash=envelope.payload_hash,
                        timestamp=envelope.wall_timestamp,
                    )
                )

        return features


# ---------------------------------------------------------------------------
#  Neural / Spiking raw-patch extractors (Invariant 3 compliant: offline, zero-LLM)
# ---------------------------------------------------------------------------


class SNNPerceptualFeatureExtractor:
    """Spatiotemporal Leaky Integrate-and-Fire (LIF) encoder over raw 5x5x4 visual patches.

    Operates directly on raw 5x5 local occupancy/self/foreign/oob patches, 5x5
    frame-difference temporal maps, and 5 raw physical/kinematic scalars through
    an 8-neuron spiking receptive field with persistent membrane traces and online
    three-factor role-modulated Hebbian synaptic plasticity.
    """

    def __init__(self, membrane_decay: float = 0.65, num_steps: int = 4) -> None:
        self._decay = membrane_decay
        self._num_steps = num_steps
        self._syn_patch: list[list[list[list[float]]]] = [
            [[[float(c) for c in row] for row in mat] for mat in neuron]
            for neuron in INIT_SYN_PATCH
        ]
        self._syn_delta: list[list[list[float]]] = [
            [[float(c) for c in row] for row in neuron] for neuron in INIT_SYN_DELTA
        ]
        self._syn_kin: list[list[float]] = [[float(c) for c in row] for row in INIT_SYN_KIN]
        self._syn_bias: list[float] = [float(b) for b in INIT_SYN_BIAS]

        self._membrane_traces: dict[str, list[float]] = {}
        self._prev_grid: list[list[int]] | None = None
        self._raw_input_cache: dict[
            str, tuple[list[list[int]], list[list[float]], tuple[float, ...]]
        ] = {}
        self._role_targets: dict[str, list[float]] = {}

    def reset_traces(self) -> None:
        """Clear persistent traces and short-term frame buffer on episode reset.

        Retains online-adapted synaptic weights and role attractors across episodes.
        """
        self._membrane_traces.clear()
        self._prev_grid = None
        self._raw_input_cache.clear()

    @property
    def extractor_name(self) -> str:
        return "SNNPerceptualFeatureExtractor"

    def can_extract(self, envelope: ObservationEnvelope) -> bool:
        return envelope.source_modality in (
            ModalitySource.VISION,
            ModalitySource.ENVIRONMENTAL_SENSOR,
        ) and ("entities" in envelope.metadata or "grid" in envelope.metadata)

    def _forward_snn(
        self,
        patch_ch: list[list[int]],
        delta_patch: list[list[float]],
        scalars5: tuple[float, ...],
        init_v: list[float] | None = None,
    ) -> tuple[tuple[float, ...], list[float]]:
        m_dim = len(self._syn_bias)
        currents = [0.0] * m_dim
        for m in range(m_dim):
            sp_m = self._syn_patch[m]
            sd_m = self._syn_delta[m]
            sk_m = self._syn_kin[m]
            acc = self._syn_bias[m]
            for py in range(5):
                sp_row = sp_m[py]
                sd_row = sd_m[py]
                ch_row = patch_ch[py]
                dp_row = delta_patch[py]
                for px in range(5):
                    acc += sp_row[px][ch_row[px]]
                    d_val = dp_row[px]
                    if d_val != 0.0:
                        acc += sd_row[px] * d_val
            for idx, s_val in enumerate(scalars5):
                acc += sk_m[idx] * s_val
            currents[m] = acc

        v = list(init_v) if init_v is not None else [0.0] * m_dim
        rate = [0.0] * m_dim
        threshold = 0.85
        step_inv = 1.0 / max(1, self._num_steps)
        for _ in range(self._num_steps):
            for m in range(m_dim):
                v[m] = self._decay * v[m] + 0.35 * currents[m]
                if v[m] >= threshold:
                    rate[m] += step_inv
                    v[m] -= threshold

        raw = [0.6 * rate[m] + 0.4 * math.tanh(v[m]) for m in range(m_dim)]
        mean = sum(raw) / m_dim
        var = sum((r - mean) ** 2 for r in raw) / m_dim
        std = math.sqrt(var + 1e-5)
        normed = tuple(math.tanh((r - mean) / std) for r in raw)
        return normed, v

    def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        features: list[PerceptualFeatureVector] = []
        entities = envelope.metadata.get("entities", [])
        grid, bg_feature, feature_counts = _extract_grid_context(envelope)

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("snn_seg")))
            patch_ch, delta_patch, scalars5, spatial, raw_color = _extract_raw_patch_and_scalars(
                ent,
                grid=grid,
                bg_feature=bg_feature,
                feature_counts=feature_counts,
                prev_grid=self._prev_grid,
            )

            has_motion = scalars5[3] > 1e-6 or scalars5[4] > 0.5
            prev_trace = self._membrane_traces.get(ent_id) if has_motion else None
            init_v = [0.35 * val for val in prev_trace] if prev_trace is not None else None
            normed, final_v = self._forward_snn(patch_ch, delta_patch, scalars5, init_v=init_v)

            if has_motion:
                if len(self._membrane_traces) >= 256:
                    self._membrane_traces.pop(next(iter(self._membrane_traces)))
                self._membrane_traces[ent_id] = final_v
            else:
                self._membrane_traces.pop(ent_id, None)

            labels: dict[str, float] = {}
            if "label" in ent:
                labels[str(ent["label"])] = float(ent.get("label_confidence", 1.0))

            feat_id = _gen_id("feat_snn")
            if len(self._raw_input_cache) >= 128:
                self._raw_input_cache.pop(next(iter(self._raw_input_cache)))
            self._raw_input_cache[feat_id] = (patch_ch, delta_patch, scalars5)

            features.append(
                PerceptualFeatureVector(
                    feature_id=feat_id,
                    observation_id=envelope.observation_id,
                    modality=envelope.source_modality,
                    channel_id=envelope.source_channel_id,
                    embedding=normed,
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

        if grid is not None and len(grid) > 0 and len(grid[0]) > 0:
            self._prev_grid = [[int(val) for val in row] for row in grid]

        return features

    def adapt_role(
        self,
        feature_id: str,
        role_name: str,
        lr: float = 0.04,
    ) -> tuple[float, ...] | None:
        """Apply online three-factor Hebbian plasticity from empirical role confirmation."""
        cached = self._raw_input_cache.get(feature_id)
        if cached is None:
            return None
        patch_ch, delta_patch, scalars5 = cached
        z, _ = self._forward_snn(patch_ch, delta_patch, scalars5, init_v=None)
        m_dim = len(z)

        clean_role = role_name.strip().lower()
        mu_pos = self._role_targets.get(clean_role)
        if mu_pos is None:
            self._role_targets[clean_role] = list(z)
            mu_pos = self._role_targets[clean_role]
        else:
            for m in range(m_dim):
                mu_pos[m] = 0.8 * mu_pos[m] + 0.2 * z[m]

        mod = [0.0] * m_dim
        has_conflict = False
        margin = 0.65
        for other_role, mu_neg in self._role_targets.items():
            if not _roles_conflict_pair(clean_role, other_role):
                continue
            d_neg = math.sqrt(sum((z[m] - mu_neg[m]) ** 2 for m in range(m_dim)) / m_dim)
            if d_neg < margin:
                has_conflict = True
                inv_d = 1.0 / max(d_neg, 1e-3)
                weight = (margin - d_neg) * inv_d * 0.75
                for m in range(m_dim):
                    mod[m] += weight * (z[m] - mu_neg[m])

        d_pos = math.sqrt(sum((z[m] - mu_pos[m]) ** 2 for m in range(m_dim)) / m_dim)
        if has_conflict or d_pos > 0.35:
            pull_w = 0.5 if d_pos > 0.08 else 0.0
            for m in range(m_dim):
                mod[m] += pull_w * (mu_pos[m] - z[m])

        if max(abs(val) for val in mod) < 1e-5:
            return z

        eff_lr = min(lr, 0.02)
        for m in range(m_dim):
            delta_m = max(-0.15, min(0.15, eff_lr * mod[m]))
            self._syn_bias[m] = max(-2.5, min(2.5, self._syn_bias[m] + 0.25 * delta_m))
            sp_m = self._syn_patch[m]
            sd_m = self._syn_delta[m]
            sk_m = self._syn_kin[m]
            for py in range(5):
                for px in range(5):
                    ch = patch_ch[py][px]
                    sp_m[py][px][ch] = max(-2.5, min(2.5, sp_m[py][px][ch] + delta_m * 0.12))
                    d_val = delta_patch[py][px]
                    if d_val != 0.0:
                        sd_m[py][px] = max(-2.5, min(2.5, sd_m[py][px] + delta_m * 0.15 * d_val))
            for idx, s_val in enumerate(scalars5):
                sk_m[idx] = max(-2.5, min(2.5, sk_m[idx] + delta_m * 0.35 * s_val))

        updated, _ = self._forward_snn(patch_ch, delta_patch, scalars5, init_v=None)
        return updated


class LearnedPerceptualFeatureExtractor:
    """2D Spatial Conv + Contrastive MLP encoder over raw 5x5x4 visual patches.

    Convolves 6 learned 3x3x4 receptive fields over raw 5x5 local occupancy/self/foreign/oob
    patches with global average + max pooling, concatenates 5 raw physical/kinematic
    scalars (17-D), and projects through a trainable ``17 -> 16 (tanh) -> 8`` MLP head
    with online supervised contrastive backpropagation on causal role binding.
    """

    def __init__(self) -> None:
        self._conv_w: list[list[list[list[float]]]] = [
            [[[float(c) for c in px] for px in row] for row in filt] for filt in INIT_CONV_W
        ]
        self._conv_b: list[float] = [float(b) for b in INIT_CONV_B]
        self._w1: list[list[float]] = [[float(v) for v in row] for row in INIT_W1]
        self._b1: list[float] = [float(b) for b in INIT_B1]
        self._w2: list[list[float]] = [[float(v) for v in row] for row in INIT_W2]
        self._b2: list[float] = [float(b) for b in INIT_B2]

        self._raw_input_cache: dict[str, tuple[list[list[int]], tuple[float, ...]]] = {}
        self._role_centroids: dict[str, list[float]] = {}

    @property
    def extractor_name(self) -> str:
        return "LearnedPerceptualFeatureExtractor"

    def can_extract(self, envelope: ObservationEnvelope) -> bool:
        return envelope.source_modality in (
            ModalitySource.VISION,
            ModalitySource.ENVIRONMENTAL_SENSOR,
        ) and ("entities" in envelope.metadata or "grid" in envelope.metadata)

    def _forward_with_cache(
        self,
        patch_ch: list[list[int]],
        scalars5: tuple[float, ...],
    ) -> tuple[
        tuple[float, ...],
        list[float],
        float,
        list[float],
        list[float],
        list[list[list[float]]],
        list[tuple[int, int]],
    ]:
        num_filters = len(self._conv_b)
        feat_map: list[list[list[float]]] = [
            [[0.0] * num_filters for _ in range(3)] for _ in range(3)
        ]
        for k in range(num_filters):
            cw_k = self._conv_w[k]
            cb_k = self._conv_b[k]
            for oy in range(3):
                for ox in range(3):
                    acc = cb_k
                    for ky in range(3):
                        ch_row = patch_ch[oy + ky]
                        w_row = cw_k[ky]
                        for kx in range(3):
                            acc += w_row[kx][ch_row[ox + kx]]
                    feat_map[oy][ox][k] = math.tanh(acc)

        avg_p = [0.0] * num_filters
        max_p = [-2.0] * num_filters
        max_coords: list[tuple[int, int]] = [(0, 0)] * num_filters
        for k in range(num_filters):
            s_k = 0.0
            best_v = -2.0
            best_coord = (0, 0)
            for oy in range(3):
                for ox in range(3):
                    val = feat_map[oy][ox][k]
                    s_k += val
                    if val > best_v:
                        best_v = val
                        best_coord = (oy, ox)
            avg_p[k] = s_k / 9.0
            max_p[k] = best_v
            max_coords[k] = best_coord

        x17 = avg_p + max_p + list(scalars5)
        h = [
            math.tanh(b + sum(w_val * x_val for w_val, x_val in zip(row, x17)))
            for row, b in zip(self._w1, self._b1)
        ]
        proj = [
            b + sum(w_val * h_val for w_val, h_val in zip(row, h))
            for row, b in zip(self._w2, self._b2)
        ]
        mean = sum(proj) / len(proj)
        var = sum((p - mean) ** 2 for p in proj) / len(proj)
        std = math.sqrt(var + 1e-5)
        u = [(p - mean) / std for p in proj]
        normed = tuple(math.tanh(val) for val in u)
        return normed, u, std, x17, h, feat_map, max_coords

    def extract(self, envelope: ObservationEnvelope) -> list[PerceptualFeatureVector]:
        features: list[PerceptualFeatureVector] = []
        entities = envelope.metadata.get("entities", [])
        grid, bg_feature, feature_counts = _extract_grid_context(envelope)

        for ent in entities:
            ent_id = str(ent.get("id", _gen_id("lrn_seg")))
            patch_ch, _, scalars5, spatial, raw_color = _extract_raw_patch_and_scalars(
                ent,
                grid=grid,
                bg_feature=bg_feature,
                feature_counts=feature_counts,
            )
            normed, _, _, _, _, _, _ = self._forward_with_cache(patch_ch, scalars5)

            labels: dict[str, float] = {}
            if "label" in ent:
                labels[str(ent["label"])] = float(ent.get("label_confidence", 1.0))

            feat_id = _gen_id("feat_learned")
            if len(self._raw_input_cache) >= 128:
                self._raw_input_cache.pop(next(iter(self._raw_input_cache)))
            self._raw_input_cache[feat_id] = (patch_ch, scalars5)

            features.append(
                PerceptualFeatureVector(
                    feature_id=feat_id,
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

    def adapt_role(
        self,
        feature_id: str,
        role_name: str,
        lr: float = 0.04,
    ) -> tuple[float, ...] | None:
        """Apply online supervised contrastive backpropagation from empirical role binding."""
        cached = self._raw_input_cache.get(feature_id)
        if cached is None:
            return None
        patch_ch, scalars5 = cached
        z, u, std, x17, h, feat_map, max_coords = self._forward_with_cache(patch_ch, scalars5)
        out_dim = len(z)

        clean_role = role_name.strip().lower()
        mu_pos = self._role_centroids.get(clean_role)
        if mu_pos is None:
            self._role_centroids[clean_role] = list(z)
            mu_pos = self._role_centroids[clean_role]
        else:
            for j in range(out_dim):
                mu_pos[j] = 0.8 * mu_pos[j] + 0.2 * z[j]

        dz = [0.0] * out_dim
        has_conflict = False
        margin = 0.65
        for other_role, mu_neg in self._role_centroids.items():
            if not _roles_conflict_pair(clean_role, other_role):
                continue
            d_neg = math.sqrt(sum((z[j] - mu_neg[j]) ** 2 for j in range(out_dim)) / out_dim)
            if d_neg < margin:
                has_conflict = True
                inv_d = 1.0 / max(d_neg, 1e-3)
                coeff = -1.6 * (margin - d_neg) * inv_d
                for j in range(out_dim):
                    dz[j] += coeff * (z[j] - mu_neg[j])

        d_pos = math.sqrt(sum((z[j] - mu_pos[j]) ** 2 for j in range(out_dim)) / out_dim)
        if has_conflict or d_pos > 0.35:
            pull_w = 0.6 if d_pos > 0.08 else 0.0
            for j in range(out_dim):
                dz[j] += pull_w * (z[j] - mu_pos[j])

        if max(abs(val) for val in dz) < 1e-5:
            return z

        lr = min(lr, 0.015)

        # Backprop through z = tanh(u) and LayerNorm
        du = [dz[j] * (1.0 - z[j] * z[j]) for j in range(out_dim)]
        du_mean = sum(du) / out_dim
        du_u_mean = sum(du[j] * u[j] for j in range(out_dim)) / out_dim
        d_proj = [(du[j] - du_mean - u[j] * du_u_mean) / std for j in range(out_dim)]

        def _clip(g: float) -> float:
            return max(-0.5, min(0.5, g))

        # Backprop into W2, b2, and hidden activation h
        hid_dim = len(h)
        dh = [0.0] * hid_dim
        for j in range(out_dim):
            dp_j = d_proj[j]
            self._b2[j] -= lr * _clip(dp_j)
            row_w2 = self._w2[j]
            for i in range(hid_dim):
                dh[i] += dp_j * row_w2[i]
                row_w2[i] = row_w2[i] * (1.0 - 1e-4) - lr * _clip(dp_j * h[i])

        # Backprop into W1, b1, and pooled conv input x17
        in_dim = len(x17)
        dx17 = [0.0] * in_dim
        for i in range(hid_dim):
            d_pre1 = dh[i] * (1.0 - h[i] * h[i])
            self._b1[i] -= lr * _clip(d_pre1)
            row_w1 = self._w1[i]
            for k in range(in_dim):
                dx17[k] += d_pre1 * row_w1[k]
                row_w1[k] = row_w1[k] * (1.0 - 1e-4) - lr * _clip(d_pre1 * x17[k])

        # Backprop into 2D Spatial Conv stem (6x3x3x4), normalized over 3x3 positions
        stem_lr = lr * 0.2 / 9.0
        num_filters = len(self._conv_b)
        for k in range(num_filters):
            d_avg_k = dx17[k] / 9.0
            d_max_k = dx17[num_filters + k]
            max_oy, max_ox = max_coords[k]
            cw_k = self._conv_w[k]
            db_acc = 0.0
            for oy in range(3):
                for ox in range(3):
                    d_fm = d_avg_k + (d_max_k if (oy == max_oy and ox == max_ox) else 0.0)
                    f_val = feat_map[oy][ox][k]
                    d_pre_conv = d_fm * (1.0 - f_val * f_val)
                    db_acc += d_pre_conv
                    g_clip = _clip(d_pre_conv)
                    for ky in range(3):
                        ch_row = patch_ch[oy + ky]
                        w_row = cw_k[ky]
                        for kx in range(3):
                            ch_idx = ch_row[ox + kx]
                            w_row[kx][ch_idx] -= stem_lr * g_clip
            self._conv_b[k] -= stem_lr * _clip(db_acc)

        updated, _, _, _, _, _, _ = self._forward_with_cache(patch_ch, scalars5)
        return updated


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
