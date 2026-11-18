from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpatialDyad:
    """Instantaneous spatial and kinematic relationship metrics between two entities."""

    subject_id: str
    object_id: str
    distance: float
    dx: float
    dy: float
    dz: float = 0.0
    relative_speed: float = 0.0
    centroid_speed: float = 0.0  # Speed of the pair's centroid in global coordinates
    contact: bool = False
    containment_ratio: float = 0.0  # Fraction of subject contained within object
    support_ratio: float = 0.0  # Vertical support / resting normal force ratio
    bearing_rad: float = 0.0  # Polar angle from subject to object in radians
    occlusion_ratio: float = 0.0  # Mutual visual occlusion fraction
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.subject_id or not self.object_id:
            raise ValueError("subject_id and object_id must be non-empty")
        if self.subject_id == self.object_id:
            raise ValueError("subject_id and object_id cannot be identical for dyad")
        if self.distance < 0.0:
            raise ValueError(f"distance cannot be negative, got {self.distance}")


class KinematicDyadHistory:
    """Maintains a bounded temporal sliding window of spatial dyads for an entity pair.

    Calculates emergent kinematic invariants:
    - Distance stability (low variance)
    - Trajectory synchronization (high velocity correlation)
    - Approach vs. recession rates (d(dist)/dt)
    - Sustained mechanical contact
    """

    def __init__(self, subject_id: str, object_id: str, max_history: int = 50) -> None:
        if not subject_id or not object_id:
            raise ValueError("subject_id and object_id must be non-empty")
        self._subject_id = subject_id
        self._object_id = object_id
        self._max_history = max(2, max_history)
        self._samples: collections.deque[SpatialDyad] = collections.deque(maxlen=self._max_history)

    @property
    def subject_id(self) -> str:
        return self._subject_id

    @property
    def object_id(self) -> str:
        return self._object_id

    @property
    def pair_key(self) -> tuple[str, str]:
        return (self._subject_id, self._object_id)

    def add_sample(self, dyad: SpatialDyad) -> None:
        """Appends a new dyad observation to the rolling history."""
        if (dyad.subject_id, dyad.object_id) != self.pair_key:
            raise ValueError(
                f"Dyad pair ({dyad.subject_id}, {dyad.object_id}) does not match history pair {self.pair_key}"
            )
        self._samples.append(dyad)

    def sample_count(self) -> int:
        return len(self._samples)

    def latest(self) -> SpatialDyad | None:
        return self._samples[-1] if self._samples else None

    def mean_distance(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.distance for s in self._samples) / len(self._samples)

    def distance_variance(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        m = self.mean_distance()
        return sum((s.distance - m) ** 2 for s in self._samples) / len(self._samples)

    def mean_relative_speed(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.relative_speed for s in self._samples) / len(self._samples)

    def mean_centroid_speed(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s.centroid_speed for s in self._samples) / len(self._samples)

    def contact_ratio(self) -> float:
        """Fraction of observations where entities maintained physical contact."""
        if not self._samples:
            return 0.0
        return sum(1.0 for s in self._samples if s.contact) / len(self._samples)

    def approaching_rate(self) -> float:
        """Computes rate of change of distance: negative = approaching, positive = receding."""
        if len(self._samples) < 2:
            return 0.0
        first = self._samples[0]
        last = self._samples[-1]
        dt = last.timestamp - first.timestamp
        if dt <= 1e-6:
            return 0.0
        return (last.distance - first.distance) / dt

    def is_stable_contact(self, min_ratio: float = 0.8) -> bool:
        """Checks if entities have maintained stable physical contact across history."""
        return len(self._samples) >= 3 and self.contact_ratio() >= min_ratio

    def is_synchronous_motion(
        self,
        max_dist_variance: float = 0.5,
        min_centroid_speed: float = 0.1,
        max_rel_speed: float = 0.5,
    ) -> bool:
        """Detects whether two entities are moving together with constant relative distance.

        E.g., A following B, cargo carried by avatar, or co-patrolling entities.
        """
        if len(self._samples) < 4:
            return False
        return (
            self.mean_centroid_speed() >= min_centroid_speed
            and self.mean_relative_speed() <= max_rel_speed
            and self.distance_variance() <= max_dist_variance
            and self.mean_distance() > 0.0
        )
