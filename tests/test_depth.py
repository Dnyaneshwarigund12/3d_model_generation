"""Tests for the Tier 3 arithmetic: metric point cloud -> millimetres per pixel.

The depth *model* needs a GPU and multi-gigabyte weights, but the maths that turns
its output into a scale factor does not, so a synthetic point cloud stands in for
the model here.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import subject_mask

from app import depth as depth_module
from app.errors import ScaleError
from app.scale import TIER_MONOCULAR, from_monocular_depth
from app.segment import segmentation_from_mask


def _flat_prediction(mask_shape, distance_m=1.0, focal_px=1000.0):
    """A fronto-parallel plane at a known distance, seen by a known camera.

    At 1 m with a 1000 px focal length, one pixel is exactly one millimetre.
    """
    height, width = mask_shape
    intrinsics = np.array(
        [[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1]], dtype=float
    )
    depth = np.full(mask_shape, distance_m, dtype=float)
    points = depth_module._backproject(depth, intrinsics)
    return depth_module.DepthPrediction(
        depth_m=depth, points_m=points, intrinsics=intrinsics, model="synthetic"
    )


@pytest.fixture
def patched_depth(monkeypatch):
    def _apply(prediction):
        monkeypatch.setattr(
            depth_module, "predict_depth", lambda image, settings=None: prediction
        )

    return _apply


def test_backprojection_matches_the_pinhole_model():
    intrinsics = np.array([[1000.0, 0, 50], [0, 1000.0, 50], [0, 0, 1]])
    depth = np.full((100, 100), 2.0)
    points = depth_module._backproject(depth, intrinsics)
    # A pixel 10 to the right of the principal point, 2 m away: X = 10*2/1000.
    assert points[0, 50, 60] == pytest.approx(0.02)
    assert points[1, 60, 50] == pytest.approx(0.02)
    assert points[2, 50, 50] == pytest.approx(2.0)


def test_scale_from_depth_recovers_the_known_ratio(patched_depth):
    mask = subject_mask(width_px=300, height_px=150, canvas=(500, 700))
    image = np.full((*mask.shape, 3), 200, dtype=np.uint8)
    patched_depth(_flat_prediction(mask.shape))

    result = from_monocular_depth(image, segmentation_from_mask(image, mask))

    assert result.tier == TIER_MONOCULAR
    assert result.is_estimate
    # 1 mm per pixel, less ~2% from trimming outliers off the mask edges - a
    # rounding error next to this tier's 20% uncertainty.
    assert result.mm_per_px == pytest.approx(1.0, rel=0.05)
    assert result.estimated_error_pct >= 20.0


def test_scale_from_depth_tracks_distance(patched_depth):
    """Twice as far away means each pixel covers twice as much."""
    mask = subject_mask(width_px=300, height_px=150, canvas=(500, 700))
    image = np.full((*mask.shape, 3), 200, dtype=np.uint8)
    segmentation = segmentation_from_mask(image, mask)

    patched_depth(_flat_prediction(mask.shape, distance_m=2.0))
    far = from_monocular_depth(image, segmentation)
    assert far.mm_per_px == pytest.approx(2.0, rel=0.05)


def test_tiny_subject_reports_more_error(patched_depth):
    image = np.full((500, 700, 3), 200, dtype=np.uint8)
    patched_depth(_flat_prediction((500, 700)))

    big = from_monocular_depth(
        image, segmentation_from_mask(image, subject_mask(300, 150, (500, 700)))
    )
    small = from_monocular_depth(
        image, segmentation_from_mask(image, subject_mask(30, 20, (500, 700)))
    )
    assert small.estimated_error_pct > big.estimated_error_pct


def test_mismatched_depth_shape_raises(patched_depth):
    mask = subject_mask(width_px=300, height_px=150, canvas=(500, 700))
    image = np.full((*mask.shape, 3), 200, dtype=np.uint8)
    patched_depth(_flat_prediction((200, 200)))

    with pytest.raises(ScaleError, match="does not match"):
        from_monocular_depth(image, segmentation_from_mask(image, mask))


def test_unsupported_depth_model_raises():
    from app.config import Settings

    settings = Settings()
    settings.depth_model = "some-other-net"
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    with pytest.raises(ScaleError, match="Unsupported depth model"):
        depth_module.predict_depth(image, settings)
