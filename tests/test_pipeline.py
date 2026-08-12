"""End-to-end tests.

They run on CPU with the `silhouette` placeholder backend and a caller-supplied
mask, so the orchestration, scaling, export and JSON contract are all covered
without a GPU or any model weights.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh
from conftest import marker_image, subject_mask

from app.config import Settings
from app.pipeline import _expand_quad, run
from app.segment import _zero_polygons, segmentation_from_mask


@pytest.fixture
def scene():
    """A 300x150 px object plus a 150 px marker, not overlapping."""
    mask = subject_mask(width_px=300, height_px=150, canvas=(500, 700))
    image = marker_image(marker_side_px=150, canvas=(500, 700), origin=(300, 480))
    image[mask] = (60, 110, 180)
    return image, mask


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.generator = "silhouette"
    s.output_dir = tmp_path / "outputs"
    s.subject_size = 256
    return s


def test_marker_scale_end_to_end(scene, settings):
    image, mask = scene
    result = run(
        image,
        settings=settings,
        scale_source="marker",
        marker_mm=50.0,
        segmentation=segmentation_from_mask(image, mask),
    )

    # A 150 px marker that is really 50 mm gives 1/3 mm per pixel, so the
    # 300 x 150 px object is 100 x 50 mm.
    assert result.measurements["measurement_tier"] == "reference_marker"
    assert result.measurements["length_mm"] == pytest.approx(100.0, rel=0.02)
    assert result.measurements["width_mm"] == pytest.approx(50.0, rel=0.02)
    assert result.scale.mm_per_px == pytest.approx(50.0 / 150.0, rel=0.02)


def test_manual_scale_end_to_end(scene, settings):
    image, mask = scene
    result = run(
        image,
        settings=settings,
        scale_source="manual",
        known_mm=300.0,
        known_axis="width",
        segmentation=segmentation_from_mask(image, mask),
    )
    assert result.measurements["measurement_tier"] == "manual_reference"
    assert result.measurements["length_mm"] == pytest.approx(300.0, rel=0.02)
    assert result.measurements["width_mm"] == pytest.approx(150.0, rel=0.02)


def test_outputs_are_written_to_the_run_directory(scene, settings):
    image, mask = scene
    result = run(
        image,
        settings=settings,
        scale_source="manual",
        known_mm=300.0,
        known_axis="width",
        segmentation=segmentation_from_mask(image, mask),
    )

    assert result.run_dir.is_dir()
    for name in ("model.glb", "measurements.json", "run.json", "input.png", "cutout.png"):
        assert (result.run_dir / name).exists(), name

    saved = json.loads(result.measurements_path.read_text(encoding="utf-8"))
    assert saved == result.measurements

    metadata = json.loads((result.run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["generator"] == "silhouette"
    assert metadata["scale_tier_used"] == "manual_reference"
    assert "segmentation" not in metadata["timings_s"]  # mask was supplied
    assert metadata["timings_s"]["generation"] >= 0


def test_exported_glb_is_in_metres(scene, settings):
    """glTF's unit is the metre while every reported number is a millimetre."""
    image, mask = scene
    result = run(
        image,
        settings=settings,
        scale_source="manual",
        known_mm=300.0,
        known_axis="width",
        segmentation=segmentation_from_mask(image, mask),
    )

    loaded = trimesh.load(str(result.glb_path), force="mesh")
    longest_m = float(np.sort(loaded.extents)[::-1][0])
    assert longest_m == pytest.approx(result.measurements["length_mm"] / 1000.0, rel=0.02)


def test_placeholder_backend_is_flagged(scene, settings):
    image, mask = scene
    result = run(
        image,
        settings=settings,
        scale_source="manual",
        known_mm=300.0,
        known_axis="width",
        segmentation=segmentation_from_mask(image, mask),
    )
    assert any("placeholder" in w for w in result.warnings)


def test_inferred_depth_is_reported_separately(scene, settings):
    """Depth away from the camera is the generator's guess, so it is labelled."""
    image, mask = scene
    result = run(
        image,
        settings=settings,
        scale_source="manual",
        known_mm=300.0,
        known_axis="width",
        segmentation=segmentation_from_mask(image, mask),
    )
    detail = result.measurements["detail"]
    assert detail["inferred_depth_mm"] > 0
    assert "inferred" in detail["inferred_depth_note"]


def test_marker_region_is_removed_from_the_mask(scene):
    """The marker must not survive into the mask, or it gets reconstructed too."""
    image, _ = scene
    from app.scale import detect_markers

    markers = detect_markers(image)
    assert markers, "the fixture must contain a detectable marker"

    everything = np.ones(image.shape[:2], dtype=bool)
    polygons = [_expand_quad(m.corners, 6) for m in markers]
    trimmed = _zero_polygons(everything, polygons)

    centre = markers[0].corners.mean(axis=0).astype(int)
    assert everything[centre[1], centre[0]]
    assert not trimmed[centre[1], centre[0]]
    assert trimmed.sum() < everything.sum()


def test_expand_quad_grows_outward():
    square = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=float)
    grown = _expand_quad(square, 5.0)
    assert grown[:, 0].min() < square[:, 0].min()
    assert grown[:, 0].max() > square[:, 0].max()
