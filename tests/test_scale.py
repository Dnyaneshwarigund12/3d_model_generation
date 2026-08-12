"""Tests for the millimetres-per-pixel math.

These are the numbers every reported dimension is multiplied by, so they are
checked against synthetic images whose true geometry is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import card_image, marker_image

from app.errors import ScaleError
from app.scale import (
    TIER_CARD,
    TIER_MANUAL,
    TIER_MARKER,
    from_aruco,
    from_card,
    from_manual,
    lsq_mm_per_px,
    mm_per_px_from_quad,
    quad_edges,
    resolve_scale,
)
from app.segment import segmentation_from_mask


def test_quad_edges_of_a_square():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    assert np.allclose(quad_edges(square), [10, 10, 10, 10])


def test_lsq_ratio_is_exact_when_consistent():
    # 4 edges of 50 px that are really 100 mm each.
    assert lsq_mm_per_px([50, 50, 50, 50], [100, 100, 100, 100]) == pytest.approx(2.0)


def test_lsq_ratio_averages_inconsistent_edges():
    ratio = lsq_mm_per_px([50, 60], [100, 100])
    assert 1.6 < ratio < 2.0  # between 100/60 and 100/50


def test_quad_ratio_uses_the_least_foreshortened_edge():
    """Perspective only ever shortens an edge, so the longest one is the honest sample."""
    # Two edges at full length, two foreshortened by 40%.
    ratio = mm_per_px_from_quad([100, 60, 100, 60], [50, 50, 50, 50])
    assert ratio == pytest.approx(0.5)
    # Averaging would have inflated it.
    assert lsq_mm_per_px([100, 60, 100, 60], [50, 50, 50, 50]) > ratio


def _tilt(image, squash, box=(300, 200, 500, 400)):
    """Foreshorten a region vertically, as if the marker were tilted away."""
    import cv2

    x0, y0, x1, y1 = box
    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    dy = (y1 - y0) * (1 - squash) / 2
    dst = np.float32([[x0, y0 + dy], [x1, y0 + dy], [x1, y1 - dy], [x0, y1 - dy]])
    return cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(src, dst),
        (image.shape[1], image.shape[0]),
        borderValue=(235, 235, 235),
    )


@pytest.mark.parametrize("squash", [1.0, 0.9, 0.75, 0.6, 0.45])
def test_tilted_marker_still_measures_correctly(squash):
    """A tilted marker used to cost up to 20% of error; it must stay near zero now."""
    flat = marker_image(marker_side_px=200, canvas=(700, 900), origin=(200, 300))
    tilted = _tilt(flat, squash)

    result = from_aruco(tilted, marker_mm=50.0)
    assert result.mm_per_px == pytest.approx(50.0 / 200.0, rel=0.02)


def test_tilt_is_reported_as_extra_uncertainty():
    flat = marker_image(marker_side_px=200, canvas=(700, 900), origin=(200, 300))
    square_on = from_aruco(flat, marker_mm=50.0)
    tilted = from_aruco(_tilt(flat, 0.6), marker_mm=50.0)
    assert tilted.estimated_error_pct > square_on.estimated_error_pct
    assert tilted.detail["edge_ratio"] > square_on.detail["edge_ratio"]


@pytest.mark.parametrize("marker_side_px,marker_mm", [(200, 50.0), (120, 80.0), (400, 25.0)])
def test_aruco_recovers_known_scale(marker_side_px, marker_mm):
    image = marker_image(marker_side_px=marker_side_px)
    result = from_aruco(image, marker_mm=marker_mm)

    assert result.tier == TIER_MARKER
    assert not result.is_estimate
    expected = marker_mm / marker_side_px
    # Within 1%: corner localisation is sub-pixel on a synthetic image.
    assert result.mm_per_px == pytest.approx(expected, rel=0.01)
    assert result.detail["markers_found"] == 1


def test_aruco_converts_pixel_lengths_to_millimetres():
    image = marker_image(marker_side_px=200)
    result = from_aruco(image, marker_mm=50.0)
    # An object spanning 400 px is twice the marker, so 100 mm.
    assert result.size_mm(400) == pytest.approx(100.0, rel=0.01)


def test_aruco_error_grows_for_a_small_marker():
    big = from_aruco(marker_image(marker_side_px=400), marker_mm=50.0)
    small = from_aruco(marker_image(marker_side_px=35), marker_mm=50.0)
    assert small.estimated_error_pct > big.estimated_error_pct


def test_aruco_missing_marker_raises():
    blank = np.full((400, 400, 3), 200, dtype=np.uint8)
    with pytest.raises(ScaleError, match="No ArUco marker"):
        from_aruco(blank, marker_mm=50.0)


def test_aruco_rejects_nonpositive_size():
    with pytest.raises(ScaleError):
        from_aruco(marker_image(), marker_mm=0.0)


def test_card_recovers_known_scale():
    # A card 317 px along its 85.60 mm side.
    result = from_card(card_image(long_px=317, short_px=200))
    assert result.tier == TIER_CARD
    assert result.mm_per_px == pytest.approx(85.60 / 317, rel=0.03)


def test_card_missing_raises():
    blank = np.full((400, 400, 3), 40, dtype=np.uint8)
    with pytest.raises(ScaleError, match="No card-shaped"):
        from_card(blank)


def test_manual_scale_uses_the_named_axis(rectangular_subject):
    image, mask = rectangular_subject
    segmentation = segmentation_from_mask(image, mask)
    assert (segmentation.width_px, segmentation.height_px) == (300, 150)

    by_height = from_manual(150.0, "height", segmentation)
    assert by_height.tier == TIER_MANUAL
    assert by_height.mm_per_px == pytest.approx(1.0)

    by_width = from_manual(600.0, "width", segmentation)
    assert by_width.mm_per_px == pytest.approx(2.0)


def test_manual_scale_rejects_bad_input(rectangular_subject):
    image, mask = rectangular_subject
    segmentation = segmentation_from_mask(image, mask)
    with pytest.raises(ScaleError):
        from_manual(-5.0, "height", segmentation)
    with pytest.raises(ScaleError):
        from_manual(100.0, "diagonal", segmentation)


def test_resolve_scale_prefers_the_dimension_the_user_supplied(rectangular_subject):
    image, mask = rectangular_subject
    segmentation = segmentation_from_mask(image, mask)
    result = resolve_scale(
        image, segmentation, source="auto", known_mm=150.0, known_axis="height"
    )
    assert result.tier == TIER_MANUAL


def test_resolve_scale_falls_back_to_the_marker(rectangular_subject):
    _, mask = rectangular_subject
    image = marker_image(marker_side_px=200, canvas=mask.shape)
    segmentation = segmentation_from_mask(image, mask)
    result = resolve_scale(image, segmentation, source="auto", marker_mm=50.0)
    assert result.tier == TIER_MARKER


def test_resolve_scale_rejects_unknown_source(rectangular_subject):
    image, mask = rectangular_subject
    segmentation = segmentation_from_mask(image, mask)
    with pytest.raises(ScaleError, match="Unknown scale source"):
        resolve_scale(image, segmentation, source="telepathy")
