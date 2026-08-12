"""Tests for the bridging step: mesh units -> millimetres -> dimensions."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.errors import MeasurementError
from app.measure import apply_scale, measure, scale_and_measure, solve_scale_factor

TIER = "manual_reference"


def _box(extents=(0.2, 0.1, 0.05)):
    return trimesh.creation.box(extents=extents)


def test_scale_factor_is_exact_for_a_matching_box():
    mesh = _box((0.2, 0.1, 0.05))
    solution = solve_scale_factor(mesh, target_width_mm=200.0, target_height_mm=100.0)
    assert solution.factor == pytest.approx(1000.0)
    assert solution.aspect_error == pytest.approx(0.0, abs=1e-9)


def test_measured_dimensions_match_the_target():
    mesh = _box((0.2, 0.1, 0.05))
    _, m, _ = scale_and_measure(
        mesh,
        target_width_mm=200.0,
        target_height_mm=100.0,
        measurement_tier=TIER,
        estimated_error_pct=1.5,
    )
    assert (m.length_mm, m.width_mm, m.height_mm) == pytest.approx((200.0, 100.0, 50.0))
    # 200 x 100 x 50 mm = 1000 cm3, surface area 700 cm2.
    assert m.volume_cm3 == pytest.approx(1000.0)
    assert m.surface_area_cm2 == pytest.approx(700.0)
    assert m.watertight is True
    assert m.volume_basis == "mesh"


def test_dimensions_are_sorted_largest_first():
    mesh = _box((0.05, 0.2, 0.1))
    _, m, _ = scale_and_measure(
        mesh,
        target_width_mm=50.0,
        target_height_mm=200.0,
        measurement_tier=TIER,
        estimated_error_pct=1.5,
    )
    assert m.length_mm >= m.width_mm >= m.height_mm


@pytest.mark.parametrize(
    "angles",
    [
        (np.pi / 2, 0.0, 0.0),
        (0.0, np.pi / 2, 0.0),
        (np.pi / 2, np.pi / 2, 0.0),
        (0.0, 0.0, np.pi / 2),
    ],
)
def test_result_is_invariant_to_quarter_turns(angles):
    """Generators differ in canonical pose, so the solver must not depend on it."""
    reference = _box((0.2, 0.1, 0.05))
    _, expected, _ = scale_and_measure(
        reference,
        target_width_mm=200.0,
        target_height_mm=100.0,
        measurement_tier=TIER,
        estimated_error_pct=1.5,
    )

    rotated = reference.copy()
    for angle, axis in zip(angles, ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        if angle:
            rotated.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))

    _, actual, _ = scale_and_measure(
        rotated,
        target_width_mm=200.0,
        target_height_mm=100.0,
        measurement_tier=TIER,
        estimated_error_pct=1.5,
    )
    assert (actual.length_mm, actual.width_mm, actual.height_mm) == pytest.approx(
        (expected.length_mm, expected.width_mm, expected.height_mm)
    )


def test_scale_factor_is_independent_of_the_meshs_own_units():
    """A generator that normalises to a unit cube and one that does not must agree."""
    small = _box((0.2, 0.1, 0.05))
    large = _box((200.0, 100.0, 50.0))
    a = solve_scale_factor(small, 400.0, 200.0)
    b = solve_scale_factor(large, 400.0, 200.0)
    assert a.factor * 0.2 == pytest.approx(b.factor * 200.0)


def test_ambiguous_aspect_prefers_the_largest_facing_pair():
    """Extents of 4:2:1 match a 2:1 outline two ways; the flatter reading wins."""
    mesh = _box((0.2, 0.1, 0.05))
    solution = solve_scale_factor(mesh, 200.0, 100.0)
    assert solution.mesh_axes == (0, 1)
    assert solution.factor == pytest.approx(1000.0)


def test_aspect_mismatch_is_reported():
    """A mesh whose proportions disagree with the silhouette must not hide it."""
    mesh = _box((0.2, 0.1, 0.05))
    good = solve_scale_factor(mesh, 200.0, 100.0)
    bad = solve_scale_factor(mesh, 200.0, 300.0)
    assert good.aspect_error < 0.01
    assert bad.aspect_error > 0.2


def test_non_watertight_volume_falls_back_to_the_hull():
    mesh = _box((0.2, 0.1, 0.05))
    mesh.faces = mesh.faces[:-2]  # punch a hole
    m = measure(mesh, measurement_tier=TIER, estimated_error_pct=1.5)
    assert m.watertight is False
    assert m.volume_basis == "convex_hull"
    assert m.volume_cm3 > 0


def test_invalid_targets_and_factors_raise():
    mesh = _box()
    with pytest.raises(MeasurementError):
        solve_scale_factor(mesh, 0.0, 100.0)
    with pytest.raises(MeasurementError):
        apply_scale(mesh, -1.0)


def test_measurements_serialise_to_the_documented_contract():
    mesh = _box((0.2, 0.1, 0.05))
    _, m, _ = scale_and_measure(
        mesh,
        target_width_mm=200.0,
        target_height_mm=100.0,
        measurement_tier="reference_marker",
        estimated_error_pct=4.0,
    )
    payload = m.to_dict()
    for key in (
        "length_mm",
        "width_mm",
        "height_mm",
        "volume_cm3",
        "measurement_tier",
        "estimated_error_pct",
    ):
        assert key in payload
    assert payload["measurement_tier"] == "reference_marker"
    assert payload["estimated_error_pct"] == 4.0


def test_silhouette_generator_produces_a_measurable_mesh():
    from app.generators.silhouette import SilhouetteGenerator

    rgba = np.zeros((400, 500, 4), dtype=np.uint8)
    rgba[100:250, 100:400] = (200, 120, 60, 255)  # 300 x 150 px
    mesh = SilhouetteGenerator().generate(rgba)

    assert mesh.is_watertight
    assert len(mesh.faces) < 50_000
    assert mesh.extents[0] / mesh.extents[1] == pytest.approx(2.0, rel=0.02)

    _, m, _ = scale_and_measure(
        mesh,
        target_width_mm=300.0,
        target_height_mm=150.0,
        measurement_tier=TIER,
        estimated_error_pct=1.5,
    )
    assert (m.length_mm, m.width_mm) == pytest.approx((300.0, 150.0), rel=0.02)
    # A slab: volume is width x height x the extruded depth.
    assert m.volume_cm3 == pytest.approx(
        30.0 * 15.0 * (m.height_mm / 10.0), rel=0.02
    )
