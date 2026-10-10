"""Unit tests for Advanced Morphology, Pattern Stamping, Scaling, and Wallpaper Tiling (W036, W054, W056, W106).

Verifies:
- W036: Repeated motif detection and lattice induction
- W054: Fractional scaling & anisotropic resizing
- W056: Generative pattern stamping & replication
- W106: Planar wallpaper group symmetry classification
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.world.cortex_morphology import (
    FractionalScaleTransformer,
    GenerativePatternStamper,
    ScaleInferenceResult,
    WallpaperSymmetryGroup,
    WallpaperTilingInducer,
    WallpaperTilingModel,
)


def test_w054_fractional_and_anisotropic_scaling() -> None:
    """Fractional non-integer scaling and scale factor inference (W054)."""
    # 2x2 source kernel
    kernel = np.array([[1, 2], [3, 4]], dtype=int)

    # Scale 2.5x along rows, 1.5x along columns -> target shape (5, 3)
    scaled = FractionalScaleTransformer.scale_grid(kernel, scale_r=2.5, scale_c=1.5, order=0)
    assert scaled.shape == (5, 3)
    # Token identity preservation (discrete values only: 1, 2, 3, 4)
    assert set(np.unique(scaled)).issubset({1, 2, 3, 4})

    # Infer scale factors
    res = FractionalScaleTransformer.infer_scale_factors(kernel, scaled)
    assert isinstance(res, ScaleInferenceResult)
    assert res.scale_r == 2.5
    assert res.scale_c == 1.5
    assert res.is_isotropic is False
    assert res.residual_error == 0.0


def test_w056_generative_pattern_stamping() -> None:
    """Discovers repeated stamps of a source kernel and reconstructs canvas (W056)."""
    # 3x3 stamp kernel (a plus sign)
    kernel = np.array(
        [
            [0, 5, 0],
            [5, 5, 5],
            [0, 5, 0],
        ],
        dtype=int,
    )

    # Canvas of size 12x12 with two stamps placed at (1, 1) and (6, 7)
    canvas = np.zeros((12, 12), dtype=int)
    canvas[1:4, 1:4] = kernel
    canvas[6:9, 7:10] = kernel

    # Find stamp placements
    placements = GenerativePatternStamper.find_stamp_placements(canvas, kernel, bg=0)
    assert len(placements) == 2

    coords = {(p.row, p.col) for p in placements}
    assert (1, 1) in coords
    assert (6, 7) in coords

    # Synthesize canvas from placements
    synth = GenerativePatternStamper.synthesize_stamped_canvas((12, 12), kernel, placements, bg=0)
    np.testing.assert_array_equal(synth, canvas)


def test_w036_w106_wallpaper_tiling_group_induction() -> None:
    """Discovers 2D translation lattice, fits wallpaper group, and tiles canvas (W036, W106)."""
    # Fundamental unit cell (2x2 check pattern)
    unit_cell = np.array(
        [
            [1, 2],
            [2, 1],
        ],
        dtype=int,
    )

    # 8x8 tiled canvas
    canvas = np.tile(unit_cell, (4, 4))

    # Fit wallpaper model
    model = WallpaperTilingInducer.fit_wallpaper_group(canvas, bg=0)
    assert isinstance(model, WallpaperTilingModel)
    assert model.period_r == 2
    assert model.period_c == 2
    assert model.concordance_score == 1.0
    # Symmetric 2x2 check has 180° rotation -> p2 or higher
    assert model.group in (
        WallpaperSymmetryGroup.P2,
        WallpaperSymmetryGroup.P4,
        WallpaperSymmetryGroup.P4M,
        WallpaperSymmetryGroup.PM,
    )

    # Forward tiling generation
    tiled_recon = WallpaperTilingInducer.tile_canvas(model.unit_cell, (8, 8), group=model.group)
    np.testing.assert_array_equal(tiled_recon, canvas)
