"""Exercises the validation harness on a synthetic scene of known geometry.

This checks the measuring chain end to end - printed marker to millimetres per
pixel to mesh scale to reported dimensions - and that `tools/validate.py`
computes its error statistics correctly.

It is not a substitute for the tape-measure run on real objects. Synthetic images
have perfect edges, no lens distortion and a marker exactly coplanar with the
subject, so they validate the arithmetic, not how the models behave on a real
photo.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
from conftest import marker_image, subject_mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.imaging import save_image

# A 100 px marker that is really 50 mm gives 0.5 mm per pixel, so the 400 x 200 px
# object below is exactly 200 x 100 mm.
MARKER_MM = 50.0
MARKER_PX = 100
OBJECT_PX = (400, 200)
TRUTH_MM = (200.0, 100.0)


@pytest.fixture
def synthetic_manifest(tmp_path):
    mask = subject_mask(width_px=OBJECT_PX[0], height_px=OBJECT_PX[1], canvas=(600, 900))
    image = marker_image(marker_side_px=MARKER_PX, canvas=(600, 900), origin=(420, 700))
    image[mask] = (70, 120, 190)

    photo = tmp_path / "synthetic_box.png"
    save_image(image, photo)

    # The third value is the slab depth the placeholder extruder produces
    # (0.203 of the long side); only the first two are set by the marker.
    depth_mm = TRUTH_MM[0] * 13 / 64
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "truth_length_mm",
                "truth_width_mm",
                "truth_height_mm",
                "scale_source",
                "marker_mm",
                "known_mm",
                "known_axis",
                "notes",
            ]
        )
        writer.writerow(
            [
                photo.name,
                TRUTH_MM[0],
                TRUTH_MM[1],
                round(depth_mm, 2),
                "marker",
                MARKER_MM,
                "",
                "",
                "synthetic",
            ]
        )
    return manifest


def test_harness_reports_small_error_on_a_known_scene(synthetic_manifest, tmp_path):
    from tools.validate import evaluate

    settings = Settings()
    settings.generator = "silhouette"
    settings.output_dir = tmp_path / "outputs"
    settings.subject_size = 256

    report = evaluate(synthetic_manifest, settings, generator="silhouette")

    assert "reference_marker" in report["summary"]
    summary = report["summary"]["reference_marker"]
    assert summary["objects"] == 1
    assert summary["mean_abs_error_pct"] < 3.0

    record = report["records"][0]
    assert "error" not in record
    # The two dimensions the marker actually determines.
    assert record["predicted_mm"][0] == pytest.approx(TRUTH_MM[0], rel=0.02)
    assert record["predicted_mm"][1] == pytest.approx(TRUTH_MM[1], rel=0.02)
    assert Path(record["run_dir"]).is_dir()


def test_harness_records_failures_without_stopping(tmp_path):
    """A photo with no marker must be reported, not crash the whole run."""
    from tools.validate import evaluate

    blank = tmp_path / "blank.png"
    save_image(
        __import__("numpy").full((300, 300, 3), 210, dtype="uint8"), blank
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image,truth_length_mm,truth_width_mm,truth_height_mm,scale_source,marker_mm\n"
        f"{blank.name},100,50,25,marker,50\n",
        encoding="utf-8",
    )

    settings = Settings()
    settings.generator = "silhouette"
    settings.output_dir = tmp_path / "outputs"

    report = evaluate(manifest, settings, generator="silhouette")
    assert report["summary"] == {}
    assert "error" in report["records"][0]
