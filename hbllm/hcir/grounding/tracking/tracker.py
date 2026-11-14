from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedEntityRef,
    GroundedRef,
    TrackletRef,
)
from hbllm.hcir.grounding.tracking.operations import (
    EntityAliasOp,
    EntityAssociationOp,
    EntityMergeOp,
    EntityReidentificationOp,
    EntitySplitOp,
    IdentityOperation,
)


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@dataclass
class TrackletState:
    """State of an active or historical tracklet in a single sensor stream."""

    tracklet_ref: TrackletRef
    channel_id: str
    start_time: float
    last_update_time: float
    last_spatial_coords: tuple[float, ...] = ()
    feature_vector: list[float] = field(default_factory=list)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateEntityState:
    """State of a provisional persistent entity hypothesis in SGE Working Store."""

    candidate_ref: EntityCandidateRef
    associated_tracklet_ids: set[str] = field(default_factory=set)
    first_observed_time: float = field(default_factory=time.time)
    last_observed_time: float = field(default_factory=time.time)
    spatial_trajectory: list[tuple[float, ...]] = field(default_factory=list)
    feature_centroid: list[float] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    grounded_entity_ref: GroundedEntityRef | None = None
    aliases: set[str] = field(default_factory=set)
    is_active: bool = True

    @property
    def metadata(self) -> dict[str, Any]:
        return self.properties


class ThreeTierEntityTracker:
    """Manages the lifecycle of Tracklets, Entity Candidates, and Grounded Entities.

    Supports spatio-temporal association, occlusion bridging, and reversible
    merge and split operations with full provenance tracking.
    """

    def __init__(self, spatial_distance_threshold: float = 2.0) -> None:
        self._spatial_distance_threshold = spatial_distance_threshold
        self._tracklets: dict[str, TrackletState] = {}
        self._candidates: dict[str, CandidateEntityState] = {}
        self._tracklet_to_candidate: dict[str, str] = {}
        self._audit_log: list[IdentityOperation] = []
        self._lock = threading.RLock()

    def update_tracklet(
        self,
        tracklet_ref: TrackletRef,
        channel_id: str,
        spatial_coords: tuple[float, ...] = (),
        feature_vector: list[float] | None = None,
        is_active: bool = True,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrackletState:
        """Register or update an observation point for a sensor-local tracklet."""
        t = timestamp if timestamp is not None else time.time()
        with self._lock:
            trk_id = tracklet_ref.id
            if trk_id not in self._tracklets:
                state = TrackletState(
                    tracklet_ref=tracklet_ref,
                    channel_id=channel_id,
                    start_time=t,
                    last_update_time=t,
                    last_spatial_coords=spatial_coords,
                    feature_vector=feature_vector or [],
                    is_active=is_active,
                    metadata=metadata or {},
                )
                self._tracklets[trk_id] = state
            else:
                state = self._tracklets[trk_id]
                state.last_update_time = t
                state.last_spatial_coords = spatial_coords
                if feature_vector:
                    state.feature_vector = feature_vector
                state.is_active = is_active
                if metadata:
                    state.metadata.update(metadata)

            # If tracklet is associated with a candidate, update candidate state
            cand_id = self._tracklet_to_candidate.get(trk_id)
            if cand_id and cand_id in self._candidates:
                cand = self._candidates[cand_id]
                cand.last_observed_time = t
                cand.is_active = is_active
                if spatial_coords:
                    cand.spatial_trajectory.append(spatial_coords)
                    if len(cand.spatial_trajectory) > 50:
                        cand.spatial_trajectory = cand.spatial_trajectory[-50:]
                if metadata:
                    cand.properties.update(metadata)

            return state

    def auto_associate(
        self,
        tracklet_ref: TrackletRef,
        spatial_coords: tuple[float, ...] = (),
        max_distance: float | None = None,
    ) -> EntityCandidateRef:
        """Associates a tracklet with the nearest spatial candidate or spawns a new one."""
        threshold = max_distance if max_distance is not None else self._spatial_distance_threshold
        with self._lock:
            # If already associated, return existing
            cand_id = self._tracklet_to_candidate.get(tracklet_ref.id)
            if cand_id and cand_id in self._candidates:
                return self._candidates[cand_id].candidate_ref

            best_cand_id: str | None = None
            min_dist = float("inf")

            if spatial_coords:
                trk_state = self._tracklets.get(tracklet_ref.id)
                trk_color = trk_state.metadata.get("color") if trk_state is not None else None
                for cid, cand in self._candidates.items():
                    cand_color = cand.properties.get("color")
                    if (
                        trk_color is not None
                        and cand_color is not None
                        and int(trk_color) != int(cand_color)
                    ):
                        continue
                    if cand.spatial_trajectory:
                        last_pos = cand.spatial_trajectory[-1]
                        dist = self._euclidean_distance(spatial_coords, last_pos)
                        if dist < min_dist and dist <= threshold:
                            min_dist = dist
                            best_cand_id = cid

            if best_cand_id is not None:
                candidate_ref = self._candidates[best_cand_id].candidate_ref
                self.associate_tracklet_to_candidate(
                    tracklet_ref=tracklet_ref,
                    candidate_ref=candidate_ref,
                    confidence=max(0.5, 1.0 - (min_dist / threshold)),
                    rationale=f"Spatial proximity match (dist={min_dist:.2f} <= {threshold})",
                )
                return candidate_ref
            else:
                # Spawn new candidate entity
                new_candidate_ref = EntityCandidateRef(id=_gen_id("ecand"))
                self.associate_tracklet_to_candidate(
                    tracklet_ref=tracklet_ref,
                    candidate_ref=new_candidate_ref,
                    confidence=1.0,
                    rationale="New unassociated tracklet initialized as candidate entity",
                )
                return new_candidate_ref

    def associate_tracklet_to_candidate(
        self,
        tracklet_ref: TrackletRef,
        candidate_ref: EntityCandidateRef,
        confidence: float = 1.0,
        rationale: str = "",
        evidence_ids: tuple[str, ...] = (),
    ) -> EntityAssociationOp:
        """Explicitly binds a tracklet to an entity candidate and records an association operation."""
        with self._lock:
            cand_id = candidate_ref.id
            trk_id = tracklet_ref.id

            if cand_id not in self._candidates:
                self._candidates[cand_id] = CandidateEntityState(
                    candidate_ref=candidate_ref,
                    associated_tracklet_ids={trk_id},
                    first_observed_time=time.time(),
                    last_observed_time=time.time(),
                )
            else:
                self._candidates[cand_id].associated_tracklet_ids.add(trk_id)

            self._tracklet_to_candidate[trk_id] = cand_id

            # Sync spatial trajectory if tracklet has coordinates
            if trk_id in self._tracklets:
                t_state = self._tracklets[trk_id]
                coords = t_state.last_spatial_coords
                if coords:
                    self._candidates[cand_id].spatial_trajectory.append(coords)
                    if len(self._candidates[cand_id].spatial_trajectory) > 50:
                        self._candidates[cand_id].spatial_trajectory = self._candidates[
                            cand_id
                        ].spatial_trajectory[-50:]
                if t_state.metadata:
                    self._candidates[cand_id].properties.update(t_state.metadata)

            op = EntityAssociationOp(
                operation_id=_gen_id("op_assoc"),
                tracklet_ref=tracklet_ref,
                candidate_ref=candidate_ref,
                confidence=confidence,
                rationale=rationale or "Explicit tracklet-to-candidate association",
                evidence_ids=evidence_ids,
            )
            self._record_op(op)
            return op

    def _record_op(self, op: IdentityOperation) -> None:
        """Appends an identity operation to audit log with bounded retention."""
        self._audit_log.append(op)
        if len(self._audit_log) > 200:
            self._audit_log = self._audit_log[-200:]

    def merge_candidates(
        self,
        source_candidates: tuple[EntityCandidateRef, ...],
        merged_candidate: EntityCandidateRef | None = None,
        confidence: float = 0.9,
        rationale: str = "",
        evidence_ids: tuple[str, ...] = (),
    ) -> EntityMergeOp:
        """Merges two or more entity candidates into a unified candidate.

        Preserves prior tracklet bindings in the merge operation record to enable clean splitting.
        """
        if len(source_candidates) < 2:
            raise ValueError("Merging requires at least two source candidates")

        with self._lock:
            target_ref = merged_candidate or EntityCandidateRef(id=_gen_id("ecand_merged"))
            target_id = target_ref.id

            prior_tracklet_map: dict[str, tuple[str, ...]] = {}
            merged_tracklets: set[str] = set()
            merged_trajectory: list[tuple[float, ...]] = []
            merged_aliases: set[str] = set()
            merged_props: dict[str, Any] = {}
            first_time = float("inf")
            last_time = float("-inf")

            for sc in source_candidates:
                sc_id = sc.id
                if sc_id in self._candidates:
                    state = self._candidates[sc_id]
                    prior_tracklet_map[sc_id] = tuple(sorted(state.associated_tracklet_ids))
                    merged_tracklets.update(state.associated_tracklet_ids)
                    merged_trajectory.extend(state.spatial_trajectory)
                    merged_aliases.update(state.aliases)
                    merged_props.update(state.properties)
                    first_time = min(first_time, state.first_observed_time)
                    last_time = max(last_time, state.last_observed_time)
                    # Remove old candidate
                    del self._candidates[sc_id]

            # Re-point tracklets to new merged candidate
            for trk_id in merged_tracklets:
                self._tracklet_to_candidate[trk_id] = target_id

            # Create merged candidate state
            self._candidates[target_id] = CandidateEntityState(
                candidate_ref=target_ref,
                associated_tracklet_ids=merged_tracklets,
                first_observed_time=first_time if first_time != float("inf") else time.time(),
                last_observed_time=last_time if last_time != float("-inf") else time.time(),
                spatial_trajectory=merged_trajectory,
                aliases=merged_aliases,
                properties=merged_props,
                is_active=True,
            )

            op = EntityMergeOp(
                operation_id=_gen_id("op_merge"),
                source_candidates=source_candidates,
                merged_candidate=target_ref,
                prior_tracklet_map=prior_tracklet_map,
                confidence=confidence,
                rationale=rationale or "Provisional candidate entity merge",
                evidence_ids=evidence_ids,
            )
            self._record_op(op)
            return op

    def split_candidate(
        self,
        candidate_to_split: EntityCandidateRef,
        merge_operation_ref: str | None = None,
        resulting_candidates: tuple[EntityCandidateRef, ...] | None = None,
        tracklet_partition: dict[str, str] | None = None,
        confidence: float = 1.0,
        rationale: str = "",
    ) -> EntitySplitOp:
        """Splits an entity candidate back into distinct entities, reversing a prior merge."""
        with self._lock:
            split_id = candidate_to_split.id
            if split_id not in self._candidates:
                raise KeyError(f"Candidate {split_id} not found for split")

            # If reversing a prior merge
            target_candidates: list[EntityCandidateRef] = []
            partition: dict[str, str] = {}

            if merge_operation_ref:
                # Find matching merge op
                merge_op = next(
                    (
                        op
                        for op in self._audit_log
                        if op.operation_id == merge_operation_ref and isinstance(op, EntityMergeOp)
                    ),
                    None,
                )
                if merge_op is None:
                    raise ValueError(f"Merge operation {merge_operation_ref} not found")

                target_candidates = list(merge_op.source_candidates)
                for sc_ref in target_candidates:
                    sc_id = sc_ref.id
                    trk_ids = merge_op.prior_tracklet_map.get(sc_id, ())
                    for trk_id in trk_ids:
                        partition[trk_id] = sc_id
            elif resulting_candidates and tracklet_partition:
                target_candidates = list(resulting_candidates)
                partition = tracklet_partition
            else:
                raise ValueError("split_candidate requires either merge_operation_ref or partition")

            # Remove current merged candidate
            del self._candidates[split_id]

            # Recreate restored candidates
            for res_cand in target_candidates:
                res_id = res_cand.id
                assigned_tracklets = {trk_id for trk_id, cid in partition.items() if cid == res_id}
                for trk_id in assigned_tracklets:
                    self._tracklet_to_candidate[trk_id] = res_id

                self._candidates[res_id] = CandidateEntityState(
                    candidate_ref=res_cand,
                    associated_tracklet_ids=assigned_tracklets,
                    first_observed_time=time.time(),
                    last_observed_time=time.time(),
                    is_active=True,
                )

            split_op = EntitySplitOp(
                operation_id=_gen_id("op_split"),
                split_candidate=candidate_to_split,
                resulting_candidates=tuple(target_candidates),
                merge_operation_ref=merge_operation_ref,
                tracklet_partition=partition,
                confidence=confidence,
                rationale=rationale or "Reversible entity split executed",
            )
            self._record_op(split_op)
            return split_op

    def reidentify_entity(
        self,
        candidate_ref: EntityCandidateRef,
        grounded_ref: GroundedEntityRef,
        confidence: float = 0.95,
        similarity_metrics: dict[str, float] | None = None,
        rationale: str = "",
        evidence_ids: tuple[str, ...] = (),
    ) -> EntityReidentificationOp:
        """Binds a candidate entity to an authoritative GroundedEntityRef."""
        with self._lock:
            cand_id = candidate_ref.id
            if cand_id not in self._candidates:
                raise KeyError(f"Candidate {cand_id} not found for re-identification")

            self._candidates[cand_id].grounded_entity_ref = grounded_ref

            op = EntityReidentificationOp(
                operation_id=_gen_id("op_reid"),
                candidate_ref=candidate_ref,
                grounded_ref=grounded_ref,
                confidence=confidence,
                rationale=rationale or "Candidate re-identified as known canonical entity",
                evidence_ids=evidence_ids,
                similarity_metrics=similarity_metrics or {},
            )
            self._record_op(op)
            return op

    def add_alias(
        self,
        target_ref: GroundedRef,
        alias_label: str,
        alias_context: str = "",
        confidence: float = 1.0,
    ) -> EntityAliasOp:
        """Binds an alias name or descriptive tag without duplicating graph state."""
        with self._lock:
            target_id = target_ref.id
            if target_id in self._candidates:
                self._candidates[target_id].aliases.add(alias_label)

            op = EntityAliasOp(
                operation_id=_gen_id("op_alias"),
                target_ref=target_ref,
                alias_label=alias_label,
                alias_context=alias_context,
                confidence=confidence,
                rationale=f"Bound alias '{alias_label}' to {target_id}",
            )
            self._record_op(op)
            return op

    def get_candidate(self, candidate_id: str) -> CandidateEntityState | None:
        with self._lock:
            return self._candidates.get(candidate_id)

    def get_candidate_for_tracklet(self, tracklet_id: str) -> EntityCandidateRef | None:
        with self._lock:
            cand_id = self._tracklet_to_candidate.get(tracklet_id)
            if cand_id and cand_id in self._candidates:
                return self._candidates[cand_id].candidate_ref
            return None

    def get_active_candidates(self) -> list[CandidateEntityState]:
        with self._lock:
            return [cand for cand in self._candidates.values() if cand.is_active]

    def get_audit_log(self) -> list[IdentityOperation]:
        with self._lock:
            return list(self._audit_log)

    @staticmethod
    def _euclidean_distance(p1: tuple[float, ...], p2: tuple[float, ...]) -> float:
        if len(p1) != len(p2):
            return float("inf")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2, strict=False)))
