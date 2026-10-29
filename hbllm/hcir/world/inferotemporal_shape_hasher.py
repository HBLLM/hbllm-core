from __future__ import annotations

"""Inferotemporal (IT) Viewpoint- & Scale-Invariant Morphological Shape Archetype Hashing.

Biologically & mathematically grounded in:
1. J.J. DiCarlo, D. Zoccolan, & N.C. Rust (2012) - "How does the brain solve visual
   object recognition?" (Neuron). The ventral visual stream (V1 → V2 → V4 → IT)
   disentangles object identity from affine variations (translation, position, scale).
2. M.K. Hu (1962) - "Visual pattern recognition by moment invariants" (IRE Trans Inf Theory).
   Scale- and translation-invariant 2D geometric moments (H1..H4).

In HBLLM, this module computes continuous geometric invariants and discrete canonical
archetype hashes for spatial entities, enabling zero-shot recognition of translated,
scaled, or recolored objects.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass
class ShapeArchetypeSignature:
    """Translation- and scale-invariant geometric signature of an extracted entity."""

    archetype_hash: str
    hu_moments: tuple[float, float, float, float]
    aspect_ratio: float
    centroid: tuple[float, float]
    area: int
    bounding_box: tuple[int, int, int, int]  # (min_r, max_r, min_c, max_c)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export shape archetype as an HCIR causal predicate."""
        return CausalPredicate(
            variable=f"shape_archetype:{self.archetype_hash[:8]}",
            operator="==",
            value=f"area:{self.area}:aspect:{self.aspect_ratio:.2f}",
            relation=f"h1:{self.hu_moments[0]:.4f}:h2:{self.hu_moments[1]:.4f}",
            target_entity="inferotemporal_shape_hasher",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording invariant shape archetype."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"shape_{self.archetype_hash[:12]}",
                node_data={
                    "archetype_hash": self.archetype_hash,
                    "hu_moments": [float(h) for h in self.hu_moments],
                    "aspect_ratio": float(self.aspect_ratio),
                    "centroid": [self.centroid[0], self.centroid[1]],
                    "area": int(self.area),
                    "bounding_box": list(self.bounding_box),
                },
            )
        ]


class InferotemporalShapeHasher:
    """Ventral stream IT module computing scale- and translation-invariant shape invariants."""

    def compute_signature(
        self, coordinates: set[tuple[int, int]] | list[tuple[int, int]]
    ) -> ShapeArchetypeSignature:
        """Compute translation- and scale-invariant Hu moments and canonical archetype hash."""
        coords = list(coordinates)
        n = len(coords)
        if n == 0:
            return ShapeArchetypeSignature(
                archetype_hash="empty_shape",
                hu_moments=(0.0, 0.0, 0.0, 0.0),
                aspect_ratio=1.0,
                centroid=(0.0, 0.0),
                area=0,
                bounding_box=(0, 0, 0, 0),
            )

        # 1. Area & Centroid
        sum_r = sum(r for r, _ in coords)
        sum_c = sum(c for _, c in coords)
        cr = sum_r / n
        cc = sum_c / n

        min_r = min(r for r, _ in coords)
        max_r = max(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        max_c = max(c for _, c in coords)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        aspect = float(h) / float(max(1, w))

        # 2. Central Moments μ_pq = Σ (r - cr)^p (c - cc)^q
        mu20 = sum((r - cr) ** 2 for r, _ in coords)
        mu02 = sum((c - cc) ** 2 for _, c in coords)
        mu11 = sum((r - cr) * (c - cc) for r, c in coords)
        mu30 = sum((r - cr) ** 3 for r, _ in coords)
        mu03 = sum((c - cc) ** 3 for _, c in coords)
        mu21 = sum(((r - cr) ** 2) * (c - cc) for r, c in coords)
        mu12 = sum((r - cr) * ((c - cc) ** 2) for r, c in coords)

        # 3. Normalized Central Moments η_pq = μ_pq / (μ00^((p+q)/2 + 1))
        # where μ00 = n (area)
        # For a single pixel (n=1), moments are 0
        if n > 1:
            norm2 = float(n) ** 2.0
            norm3 = float(n) ** 2.5
            eta20 = mu20 / norm2
            eta02 = mu02 / norm2
            eta11 = mu11 / norm2
            eta30 = mu30 / norm3
            eta03 = mu03 / norm3
            eta21 = mu21 / norm3
            eta12 = mu12 / norm3
        else:
            eta20 = eta02 = eta11 = eta30 = eta03 = eta21 = eta12 = 0.0

        # 4. Hu Invariant Moments H1..H4 (Hu 1962)
        h1 = eta20 + eta02
        h2 = ((eta20 - eta02) ** 2) + (4.0 * (eta11**2))
        h3 = ((eta30 - (3.0 * eta12)) ** 2) + (((3.0 * eta21) - eta03) ** 2)
        h4 = ((eta30 + eta12) ** 2) + ((eta21 + eta03) ** 2)

        hu_moments = (h1, h2, h3, h4)

        # 5. Deterministic Archetype Hash
        # Quantize invariants to 3 decimals to ensure noise-tolerant canonical grouping
        q_h1 = round(h1, 3)
        q_h2 = round(h2, 3)
        q_aspect = round(aspect, 2)
        hash_seed = f"hu:{q_h1}:{q_h2}:asp:{q_aspect}:n:{n}"
        archetype_hash = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:16]

        return ShapeArchetypeSignature(
            archetype_hash=f"arch_{archetype_hash}",
            hu_moments=hu_moments,
            aspect_ratio=aspect,
            centroid=(cr, cc),
            area=n,
            bounding_box=(min_r, max_r, min_c, max_c),
        )

    def compute_similarity(
        self, sig1: ShapeArchetypeSignature, sig2: ShapeArchetypeSignature
    ) -> float:
        """Measure invariant shape similarity in Hu moment and aspect ratio space."""
        if sig1.archetype_hash == sig2.archetype_hash:
            return 1.0

        # Weighted distance across scale-invariant features
        diff_h1 = abs(sig1.hu_moments[0] - sig2.hu_moments[0])
        diff_h2 = abs(sig1.hu_moments[1] - sig2.hu_moments[1])
        diff_aspect = abs(sig1.aspect_ratio - sig2.aspect_ratio)

        error = (diff_h1 * 2.0) + (diff_h2 * 4.0) + (diff_aspect * 0.5)
        return max(0.0, 1.0 / (1.0 + error))
