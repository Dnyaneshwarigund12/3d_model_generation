"""Stage 4: put the mesh into millimetres, then measure it.

The generators emit a mesh normalised into an arbitrary canonical scale, so the
bridging step from `04-measurement-methodology.md` section 4 is what turns a
shape into dimensions: work out how many millimetres one mesh unit is worth, scale
uniformly, then read the numbers off the result.

Matching mesh units to millimetres needs to know which mesh axes correspond to
the image's horizontal and vertical directions. Rather than hard-coding a
convention per generator - which silently breaks whenever a model updates its
canonical pose - `solve_scale_factor` works it out from the data: the subject's
silhouette has a known pixel aspect ratio, so the pair of mesh axes whose aspect
ratio matches it is the pair facing the camera, and the third is depth. That test
is unaffected by the quarter-turn rotations generators differ by, because
rotating the mesh only permutes its extents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np

from .errors import MeasurementError

_EPS = 1e-9


@dataclass(frozen=True)
class ScaleSolution:
    """How many millimetres one mesh unit is worth, and how it was decided."""

    factor: float
    mesh_axes: tuple[int, int]
    aspect_error: float
    target_width_mm: float
    target_height_mm: float
    mesh_extents: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mm_per_mesh_unit": round(self.factor, 6),
            "matched_mesh_axes": list(self.mesh_axes),
            "aspect_error": round(self.aspect_error, 4),
            "target_width_mm": round(self.target_width_mm, 2),
            "target_height_mm": round(self.target_height_mm, 2),
            "mesh_extents_units": [round(v, 6) for v in self.mesh_extents],
        }


@dataclass(frozen=True)
class Measurements:
    """The numbers we report, in the contract from the measurement doc section 3."""

    length_mm: float
    width_mm: float
    height_mm: float
    volume_cm3: float
    surface_area_cm2: float
    watertight: bool
    volume_basis: str
    measurement_tier: str
    estimated_error_pct: float
    obb_extents_mm: tuple[float, float, float]
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "length_mm": round(self.length_mm, 1),
            "width_mm": round(self.width_mm, 1),
            "height_mm": round(self.height_mm, 1),
            "volume_cm3": round(self.volume_cm3, 1),
            "surface_area_cm2": round(self.surface_area_cm2, 1),
            "watertight": self.watertight,
            "volume_basis": self.volume_basis,
            "measurement_tier": self.measurement_tier,
            "estimated_error_pct": round(self.estimated_error_pct, 1),
            "obb_extents_mm": [round(v, 1) for v in self.obb_extents_mm],
            "detail": self.detail,
        }


def mesh_extents(mesh) -> np.ndarray:
    """Axis-aligned extents of the mesh in its own canonical frame."""
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if extents.shape != (3,) or not np.isfinite(extents).all():
        raise MeasurementError("The mesh has no usable bounding box.")
    return extents


def solve_scale_factor(
    mesh, target_width_mm: float, target_height_mm: float
) -> ScaleSolution:
    """Find the uniform millimetres-per-unit factor for this mesh.

    Args:
        target_width_mm: the subject's real width, from the silhouette's pixel
            width times the resolved millimetres-per-pixel.
        target_height_mm: likewise for height.
    """
    if target_width_mm <= 0 or target_height_mm <= 0:
        raise MeasurementError("Target dimensions must be greater than zero.")

    extents = mesh_extents(mesh)
    target_ratio = target_width_mm / target_height_mm

    best: ScaleSolution | None = None
    best_key: tuple[float, float] | None = None
    for i, j in permutations(range(3), 2):
        w_units, h_units = extents[i], extents[j]
        if w_units <= _EPS or h_units <= _EPS:
            continue
        # Compare in log space so a 2x error scores the same either direction.
        aspect_error = abs(np.log((w_units / h_units) / target_ratio))
        # Uniform factor minimising squared error against both target dimensions.
        factor = (target_width_mm * w_units + target_height_mm * h_units) / (
            w_units**2 + h_units**2
        )
        # Aspect ratio alone can be ambiguous: an object with extents in a 4:2:1
        # ratio matches a 2:1 outline through two different axis pairs. Break the
        # tie toward the pair spanning the most area, i.e. treat the smallest
        # axis as depth - a flat object facing the camera is far more common than
        # a long one seen end-on. Rounding keeps near-ties from being decided by
        # floating-point noise, and both criteria only depend on the magnitudes
        # of the extents, so the result survives any rotation of the mesh.
        key = (round(float(aspect_error), 6), -float(w_units + h_units))
        if best_key is None or key < best_key:
            best_key = key
            best = ScaleSolution(
                factor=float(factor),
                mesh_axes=(i, j),
                aspect_error=float(aspect_error),
                target_width_mm=float(target_width_mm),
                target_height_mm=float(target_height_mm),
                mesh_extents=tuple(float(v) for v in extents),
            )

    if best is None:
        raise MeasurementError(
            "The mesh is degenerate (no two axes have non-zero extent)."
        )
    return best


def apply_scale(mesh, factor: float):
    """Return a copy of the mesh scaled uniformly into millimetres."""
    if factor <= 0 or not np.isfinite(factor):
        raise MeasurementError(f"Invalid scale factor: {factor}.")
    scaled = mesh.copy()
    scaled.apply_scale(float(factor))
    return scaled


def _quantised(points: np.ndarray, steps: int = 128) -> np.ndarray:
    """Snap points to a coarse grid and drop duplicates."""
    span = float((points.max(axis=0) - points.min(axis=0)).max())
    if span <= 0:
        return points
    step = span / steps
    return np.unique(np.round(points / step).astype(np.int64), axis=0) * step


def _oriented_extents(mesh) -> np.ndarray:
    """Extents of the tightest oriented box around the mesh.

    Oriented rather than axis-aligned because a mesh sitting at an angle in its
    canonical frame would otherwise report a box bigger than the object it
    contains.

    trimesh searches every convex-hull facet normal for the best orientation,
    which costs about 15 seconds on the ~80k-face meshes these generators emit -
    far more than generating the mesh in the first place. So the orientation is
    found from a quantised hull, then the extents are measured by projecting the
    full hull onto that orientation. Only the choice of angle is approximated,
    and the extents stay exact for the angle chosen.
    """
    import trimesh

    try:
        hull_points = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
        candidate = hull_points if len(hull_points) <= 2000 else _quantised(hull_points)
        to_origin, _ = trimesh.bounds.oriented_bounds(candidate)
        rotation = np.asarray(to_origin, dtype=np.float64)[:3, :3]
        projected = hull_points @ rotation.T
        extents = projected.max(axis=0) - projected.min(axis=0)
    except Exception:
        extents = mesh_extents(mesh)

    extents = np.asarray(extents, dtype=np.float64)
    if extents.shape != (3,) or not np.isfinite(extents).all() or (extents <= 0).any():
        extents = mesh_extents(mesh)
    return extents


def measure(
    mesh,
    *,
    measurement_tier: str,
    estimated_error_pct: float,
    detail: dict[str, Any] | None = None,
) -> Measurements:
    """Extract dimensions from a mesh that is already scaled to millimetres.

    Length, width and height are the oriented bounding box's extents sorted
    largest to smallest - the mesh has no notion of which way is up, so calling
    the tallest extent "height" would be a guess.
    """
    extents = np.sort(_oriented_extents(mesh))[::-1]
    length_mm, width_mm, height_mm = (float(v) for v in extents)

    watertight = bool(getattr(mesh, "is_watertight", False))
    if watertight:
        volume_mm3 = abs(float(mesh.volume))
        volume_basis = "mesh"
    else:
        # Generated meshes are often not watertight; a convex hull is an
        # over-estimate for a concave object, so it is labelled as such.
        try:
            volume_mm3 = abs(float(mesh.convex_hull.volume))
            volume_basis = "convex_hull"
        except Exception:
            volume_mm3 = float(length_mm * width_mm * height_mm)
            volume_basis = "bounding_box"

    try:
        area_mm2 = abs(float(mesh.area))
    except Exception:
        area_mm2 = 0.0

    return Measurements(
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        volume_cm3=volume_mm3 / 1000.0,
        surface_area_cm2=area_mm2 / 100.0,
        watertight=watertight,
        volume_basis=volume_basis,
        measurement_tier=measurement_tier,
        estimated_error_pct=estimated_error_pct,
        obb_extents_mm=(length_mm, width_mm, height_mm),
        detail=detail or {},
    )


def scale_and_measure(
    mesh,
    *,
    target_width_mm: float,
    target_height_mm: float,
    measurement_tier: str,
    estimated_error_pct: float,
    detail: dict[str, Any] | None = None,
) -> tuple[Any, Measurements, ScaleSolution]:
    """Solve the scale factor, apply it, and measure - the whole bridging step."""
    solution = solve_scale_factor(mesh, target_width_mm, target_height_mm)
    scaled = apply_scale(mesh, solution.factor)
    combined = dict(detail or {})
    combined["scale_solution"] = solution.to_dict()
    measurements = measure(
        scaled,
        measurement_tier=measurement_tier,
        estimated_error_pct=estimated_error_pct,
        detail=combined,
    )
    return scaled, measurements, solution
