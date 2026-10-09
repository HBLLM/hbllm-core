"""Parieto-Occipital Mental Imagery (Area V6 / MST / Precuneus) Optical Ray Projection.

Modeled on human visual mental imagery (Kosslyn's mental imagery buffer) and optical ray optics:
1. Ray Emitter & Direction Vector Induction: Identifies ray sources and emission vectors
   vec{v} in {(-1, 0), (1, 0), (0, -1), (0, 1)}.
2. Specular Reflection Physics: Simulates ray reflections off angled mirrors:
   - Forward slash mirror ('/'): (dr', dc') = (-dc, -dr)
   - Backslash mirror ('\\'): (dr', dc') = (dc, dr)
3. Bidirectional Ray Intersection: Traces forward rays from active emitters and backward
   rays from target receptors to discover critical mirror placement coordinates (r*, c*).
4. Alignment Subgoal Synthesis: Generates structured spatial subgoals to place or rotate
   mirrors into the computed optical path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class MirrorOrientation(StrEnum):
    """Mirror reflector angle and orientation."""

    FORWARD_SLASH = "/"  # Angle 135 deg: (dr, dc) -> (-dc, -dr)
    BACKSLASH = "\\"  # Angle 45 deg: (dr, dc) -> (dc, dr)
    OMNI_REFLECTOR = "+"  # Rotatable or 4-way prism


@dataclass
class RayStep:
    """A single spatial segment along an optical beam trajectory."""

    coord: tuple[int, int]
    direction: tuple[int, int]  # (dr, dc)
    step_index: int


@dataclass
class OpticalRayPath:
    """Complete simulated optical trajectory from emitter to termination."""

    emitter_pos: tuple[int, int]
    steps: list[RayStep] = field(default_factory=list)
    terminated_at: tuple[int, int] | None = None
    termination_reason: str = ""  # 'receptor', 'barrier', 'boundary', 'cycle'
    hit_receptor: bool = False


@dataclass
class MirrorPlacementHypothesis:
    """Predicted mirror placement required to redirect ray into receptor."""

    mirror_pos: tuple[int, int]
    required_orientation: MirrorOrientation
    incoming_dir: tuple[int, int]
    outgoing_dir: tuple[int, int]
    confidence: float = 1.0


class OpticalRayProjector:
    """Biologically-modeled mental imagery engine for optical ray tracing and mirror reflection."""

    @staticmethod
    def reflect_vector(
        incoming_dir: tuple[int, int],
        mirror_type: MirrorOrientation,
    ) -> tuple[int, int] | None:
        """Compute reflected direction vector following specular reflection physics.

        Args:
            incoming_dir: (dr, dc) normalized velocity vector.
            mirror_type: Forward slash ('/') or backslash ('\\').

        Returns:
            Reflected (dr, dc) vector, or None if direct normal collision absorption.
        """
        dr, dc = incoming_dir
        if mirror_type == MirrorOrientation.FORWARD_SLASH:
            # (0, 1) -> (-1, 0), (1, 0) -> (0, -1), (0, -1) -> (1, 0), (-1, 0) -> (0, 1)
            return (-dc, -dr)
        elif mirror_type == MirrorOrientation.BACKSLASH:
            # (0, 1) -> (1, 0), (-1, 0) -> (0, -1), (0, -1) -> (-1, 0), (1, 0) -> (0, 1)
            return (dc, dr)
        return None

    @staticmethod
    def trace_ray(
        start_pos: tuple[int, int],
        initial_dir: tuple[int, int],
        grid_shape: tuple[int, int],
        barriers: set[tuple[int, int]],
        mirrors: dict[tuple[int, int], MirrorOrientation],
        receptors: set[tuple[int, int]],
        max_steps: int = 256,
    ) -> OpticalRayPath:
        """Simulate forward mental imagery ray tracing through the optical environment.

        Args:
            start_pos: (r, c) emitter coordinate.
            initial_dir: (dr, dc) unit direction vector.
            grid_shape: (H, W) grid dimensions.
            barriers: Set of impassable or absorbing coordinates.
            mirrors: Coordinate -> MirrorOrientation mapping.
            receptors: Target collector coordinates.
            max_steps: Maximum propagation horizon.

        Returns:
            OpticalRayPath containing the complete simulated beam.
        """
        H, W = grid_shape
        curr_r, curr_c = start_pos
        dr, dc = initial_dir

        path = OpticalRayPath(emitter_pos=start_pos)
        visited_states: set[tuple[int, int, int, int]] = set()

        for step_idx in range(max_steps):
            next_r = curr_r + dr
            next_c = curr_c + dc

            # Out of bounds
            if not (0 <= next_r < H and 0 <= next_c < W):
                path.terminated_at = (curr_r, curr_c)
                path.termination_reason = "boundary"
                break

            # Receptor hit
            if (next_r, next_c) in receptors:
                path.steps.append(
                    RayStep(coord=(next_r, next_c), direction=(dr, dc), step_index=step_idx)
                )
                path.terminated_at = (next_r, next_c)
                path.termination_reason = "receptor"
                path.hit_receptor = True
                break

            # Mirror reflection
            if (next_r, next_c) in mirrors:
                m_type = mirrors[(next_r, next_c)]
                refl_dir = OpticalRayProjector.reflect_vector((dr, dc), m_type)
                if refl_dir is None:
                    path.terminated_at = (next_r, next_c)
                    path.termination_reason = "barrier"
                    break
                path.steps.append(
                    RayStep(coord=(next_r, next_c), direction=(dr, dc), step_index=step_idx)
                )
                curr_r, curr_c = next_r, next_c
                dr, dc = refl_dir
                state = (curr_r, curr_c, dr, dc)
                if state in visited_states:
                    path.terminated_at = (curr_r, curr_c)
                    path.termination_reason = "cycle"
                    break
                visited_states.add(state)
                continue

            # Barrier / Obstacle collision
            if (next_r, next_c) in barriers:
                path.terminated_at = (next_r, next_c)
                path.termination_reason = "barrier"
                break

            # Clean propagation
            path.steps.append(
                RayStep(coord=(next_r, next_c), direction=(dr, dc), step_index=step_idx)
            )
            curr_r, curr_c = next_r, next_c
            state = (curr_r, curr_c, dr, dc)
            if state in visited_states:
                path.terminated_at = (curr_r, curr_c)
                path.termination_reason = "cycle"
                break
            visited_states.add(state)

        return path

    @staticmethod
    def solve_mirror_placement(
        emitter_pos: tuple[int, int],
        emitter_dir: tuple[int, int],
        receptor_pos: tuple[int, int],
        grid_shape: tuple[int, int],
        barriers: set[tuple[int, int]],
    ) -> list[MirrorPlacementHypothesis]:
        """Compute the intersection coordinate (r*, c*) where a single 90-degree mirror connects emitter to receptor.

        Solves the bidirectional ray intersection problem:
        Forward ray from emitter along (dr_e, dc_e) creates ray line L_e.
        Backward ray from receptor along incoming vectors creates ray line L_r.
        The intersection of L_e and L_r determines the required mirror position and angle.
        """
        H, W = grid_shape
        er, ec = emitter_pos
        tr, tc = receptor_pos
        edr, edc = emitter_dir

        hypotheses: list[MirrorPlacementHypothesis] = []

        # Candidate mirror positions along the forward emitter beam
        forward_cells: list[tuple[int, int]] = []
        cr, cc = er, ec
        while True:
            cr += edr
            cc += edc
            if not (0 <= cr < H and 0 <= cc < W) or (cr, cc) in barriers:
                break
            forward_cells.append((cr, cc))

        for mr, mc in forward_cells:
            # Vector from mirror candidate to receptor
            del_r = tr - mr
            del_c = tc - mc

            # Check if receptor lies directly along one cardinal axis from this mirror
            if del_r == 0 and del_c != 0:
                out_dir = (0, 1 if del_c > 0 else -1)
            elif del_c == 0 and del_r != 0:
                out_dir = (1 if del_r > 0 else -1, 0)
            else:
                continue

            # Verify line of sight from mirror to receptor is unobstructed
            step_r = 0 if del_r == 0 else (1 if del_r > 0 else -1)
            step_c = 0 if del_c == 0 else (1 if del_c > 0 else -1)
            curr_check = (mr + step_r, mc + step_c)
            obstructed = False
            while curr_check != (tr, tc):
                if curr_check in barriers or not (
                    0 <= curr_check[0] < H and 0 <= curr_check[1] < W
                ):
                    obstructed = True
                    break
                curr_check = (curr_check[0] + step_r, curr_check[1] + step_c)

            if obstructed:
                continue

            # Determine required mirror orientation that maps edr, edc -> out_dir
            for m_orient in (MirrorOrientation.FORWARD_SLASH, MirrorOrientation.BACKSLASH):
                reflected = OpticalRayProjector.reflect_vector((edr, edc), m_orient)
                if reflected == out_dir:
                    hypotheses.append(
                        MirrorPlacementHypothesis(
                            mirror_pos=(mr, mc),
                            required_orientation=m_orient,
                            incoming_dir=(edr, edc),
                            outgoing_dir=out_dir,
                            confidence=0.95,
                        )
                    )

        return hypotheses

    @staticmethod
    def detect_and_solve_optical_paths(
        grid: np.ndarray,
        background_feature: int,
        receptors: list[tuple[int, int]],
        barriers: set[tuple[int, int]],
        min_beam_length: int = 2,
    ) -> list[MirrorPlacementHypothesis]:
        """Detect active optical ray beams in grid and solve required mirror placement for receptors."""
        H, W = grid.shape
        hypotheses: list[MirrorPlacementHypothesis] = []
        if not receptors:
            return hypotheses

        visited_beams: set[tuple[int, int, int, int]] = set()

        for r in range(H):
            for c in range(W):
                feat = int(grid[r, c])
                if feat == background_feature or (r, c) in barriers:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (r, c, dr, dc) in visited_beams:
                        continue
                    beam_cells: list[tuple[int, int]] = [(r, c)]
                    cr, cc = r + dr, c + dc
                    while (
                        0 <= cr < H
                        and 0 <= cc < W
                        and int(grid[cr, cc]) == feat
                        and (cr, cc) not in barriers
                    ):
                        beam_cells.append((cr, cc))
                        cr += dr
                        cc += dc
                    if len(beam_cells) >= min_beam_length:
                        for br, bc in beam_cells:
                            visited_beams.add((br, bc, dr, dc))
                        emitter_pos = beam_cells[-1]
                        for rec_pos in receptors:
                            solved = OpticalRayProjector.solve_mirror_placement(
                                emitter_pos=emitter_pos,
                                emitter_dir=(dr, dc),
                                receptor_pos=rec_pos,
                                grid_shape=(H, W),
                                barriers=barriers,
                            )
                            hypotheses.extend(solved)
        return hypotheses
