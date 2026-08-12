"""Tests for the UI's data path.

The page itself needs a browser, but the function behind the button does not, and
it is the part that can silently return the wrong shape of data.
"""

from __future__ import annotations

import pytest
from conftest import marker_image, subject_mask

from app.config import Settings

gr = pytest.importorskip("gradio")

from app.ui import _dimension_rows, _tier_badge, build_ui, process_upload  # noqa: E402


@pytest.fixture
def scene():
    mask = subject_mask(width_px=400, height_px=200, canvas=(600, 900))
    image = marker_image(marker_side_px=100, canvas=(600, 900), origin=(430, 700))
    image[mask] = (70, 120, 190)
    return image


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.generator = "silhouette"
    s.output_dir = tmp_path / "outputs"
    s.subject_size = 256
    return s


def test_the_page_builds():
    s = Settings()
    s.generator = "silhouette"
    assert build_ui(s) is not None


def test_process_returns_every_output_the_page_expects(scene, settings):
    outputs = process_upload(
        scene,
        "marker",
        50.0,
        "DICT_4X4_50",
        None,
        "height",
        "silhouette",
        settings,
    )
    model_path, badge, rows, json_file, glb_file, gallery, raw = outputs

    assert model_path.endswith(".glb")
    assert json_file.endswith("measurements.json")
    assert glb_file == model_path
    assert "Measured" in badge
    assert any("Length" in row[0] for row in rows)
    assert all(len(row) == 3 for row in rows)
    assert gallery, "the debug gallery should not be empty"
    # 400 px wide at 0.5 mm/px is 200 mm.
    assert raw["length_mm"] == pytest.approx(200.0, rel=0.03)


def test_missing_image_is_a_friendly_error(settings):
    with pytest.raises(gr.Error):
        process_upload(None, "auto", 50.0, "DICT_4X4_50", None, "height", "silhouette", settings)


def test_badge_distinguishes_measured_from_estimated():
    measured = _tier_badge("reference_marker", 4.0, [])
    estimated = _tier_badge("monocular_estimate", 20.0, ["add a marker"])
    assert "Measured" in measured and "Estimated" not in measured
    assert "Estimated" in estimated
    assert "add a marker" in estimated


def test_dimension_rows_include_the_inferred_depth():
    rows = _dimension_rows(
        {
            "length_mm": 200.0,
            "width_mm": 100.0,
            "height_mm": 50.0,
            "volume_cm3": 1000.0,
            "surface_area_cm2": 700.0,
            "detail": {"inferred_depth_mm": 50.0},
        }
    )
    labels = [row[0] for row in rows]
    assert "Length" in labels
    assert any("inferred" in label for label in labels)
