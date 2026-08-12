"""The whole flow: one photo in, a scaled GLB and its dimensions out.

Runs synchronously in-process. There is no queue, worker pool or database - a
single user waiting on a progress bar does not need one, and the plan's async
architecture only earns its keep under concurrent traffic.

Stage order matters in one non-obvious way: reference objects are detected
*before* segmentation, so the marker can be cut out of the subject mask. Left in,
the 3D generator happily fuses the marker into the object's geometry.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import DEFAULT_ARUCO_DICT, DEFAULT_MARKER_MM, Settings
from .errors import PipelineError
from .generators import get_generator, upright_transform
from .imaging import as_rgb, draw_bbox, draw_polygon, load_image, save_image
from .measure import Measurements, scale_and_measure
from .scale import SOURCE_AUTO, ScaleResult, resolve_scale
from .segment import Segmentation, remove_background

# Beyond this log-space mismatch between the mesh's aspect ratio and the
# silhouette's, the mesh does not really match the photo and the dimensions
# deserve a warning.
_ASPECT_WARN = 0.2

# Pad detected reference quads before deleting them from the mask, to catch the
# marker's white quiet zone and its shadow.
_EXCLUSION_PAD_PX = 6


@dataclass
class PipelineResult:
    run_id: str
    run_dir: Path
    glb_path: Path
    measurements_path: Path
    measurements: dict[str, Any]
    scale: ScaleResult
    generator: str
    timings_s: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Path] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        m = self.measurements
        return (
            f"{m['length_mm']:.0f} x {m['width_mm']:.0f} x {m['height_mm']:.0f} mm "
            f"({self.scale.label}, +/- {m['estimated_error_pct']:.0f}%)"
        )


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _expand_quad(corners: np.ndarray, pad: float) -> np.ndarray:
    """Grow a quad outward from its centre by roughly `pad` pixels."""
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    centre = pts.mean(axis=0)
    offsets = pts - centre
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1e-6
    return centre + offsets * (1.0 + pad / norms)


def _reference_polygons(image: np.ndarray, source: str, aruco_dict: str) -> list[np.ndarray]:
    """Find reference objects so they can be removed from the subject mask."""
    from .scale import SOURCE_CARD, detect_card, detect_markers

    polygons: list[np.ndarray] = []
    try:
        for marker in detect_markers(image, aruco_dict):
            polygons.append(_expand_quad(marker.corners, _EXCLUSION_PAD_PX))
    except Exception:
        pass  # a missing aruco module must not stop the run

    if not polygons and source in {SOURCE_CARD, SOURCE_AUTO}:
        try:
            card = detect_card(image)
            if card is not None:
                polygons.append(_expand_quad(card.corners, _EXCLUSION_PAD_PX))
        except Exception:
            pass
    return polygons


def _export_mesh(mesh, run_dir: Path, in_millimetres: bool) -> Path:
    """Write the mesh as GLB.

    glTF's unit is the metre, so the exported file is metres even though every
    reported number is millimetres. Viewers that trust the unit then show the
    object at its real size.
    """
    export = mesh.copy()
    if not in_millimetres:
        export.apply_scale(0.001)
    path = run_dir / "model.glb"
    export.export(str(path))
    return path


def run(
    image: str | Path | np.ndarray,
    *,
    settings: Settings | None = None,
    generator: str | None = None,
    scale_source: str = SOURCE_AUTO,
    marker_mm: float = DEFAULT_MARKER_MM,
    aruco_dict: str = DEFAULT_ARUCO_DICT,
    known_mm: float | None = None,
    known_axis: str = "height",
    run_id: str | None = None,
    segmentation: Segmentation | None = None,
) -> PipelineResult:
    """Turn a photo into a scaled 3D model plus measurements.

    Args:
        image: path to an image, or an RGB/RGBA array.
        scale_source: one of marker, card, manual, estimate, auto.
        marker_mm: printed side length of the ArUco marker, in millimetres.
        known_mm: a dimension of the object the user already knows.
        known_axis: which dimension `known_mm` refers to, width or height.
        segmentation: a mask computed elsewhere, skipping background removal.
            Used by tests and by anything with a better segmenter to hand.
    """
    settings = settings or Settings()
    generator_name = generator or settings.generator
    run_id = run_id or _new_run_id()
    run_dir = Path(settings.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    warnings: list[str] = []
    debug: dict[str, Path] = {}

    def _timed(name: str, fn):
        start = time.perf_counter()
        value = fn()
        timings[name] = round(time.perf_counter() - start, 3)
        return value

    rgb = as_rgb(load_image(image) if isinstance(image, (str, Path)) else image)
    if settings.save_debug:
        debug["input"] = save_image(rgb, run_dir / "input.png")

    # 1. Reference objects first, so the marker can be kept out of the mask.
    polygons = _timed(
        "reference_detection", lambda: _reference_polygons(rgb, scale_source, aruco_dict)
    )

    # 2. Segmentation.
    if segmentation is None:
        segmentation = _timed(
            "segmentation",
            lambda: remove_background(rgb, settings, exclude_polygons=polygons),
        )
    subject = segmentation.subject(
        padding=settings.subject_padding, size=settings.subject_size
    )
    if settings.save_debug:
        debug["cutout"] = save_image(segmentation.rgba, run_dir / "cutout.png")
        debug["subject"] = save_image(subject, run_dir / "subject.png")

    # 3. Scale, independent of the mesh.
    scale: ScaleResult = _timed(
        "scale",
        lambda: resolve_scale(
            rgb,
            segmentation,
            source=scale_source,
            marker_mm=marker_mm,
            aruco_dict=aruco_dict,
            known_mm=known_mm,
            known_axis=known_axis,
            settings=settings,
        ),
    )

    # 4. Generation.
    backend = get_generator(generator_name, settings)
    mesh = _timed("generation", lambda: backend.generate(subject))
    if getattr(backend, "is_placeholder", False):
        warnings.append(
            f"The {backend.name} backend is a placeholder that extrudes the "
            "outline - it does not reconstruct real depth. Switch to triposr or "
            "hunyuan3d for an actual 3D model."
        )
    warnings.extend(getattr(backend, "last_warnings", []) or [])
    if settings.upright_output:
        mesh = mesh.copy()
        mesh.apply_transform(upright_transform())

    # 5. Bridge scale onto the mesh, then measure.
    target_width_mm = scale.size_mm(segmentation.width_px)
    target_height_mm = scale.size_mm(segmentation.height_px)

    def _measure() -> tuple[Any, Measurements, Any]:
        return scale_and_measure(
            mesh,
            target_width_mm=target_width_mm,
            target_height_mm=target_height_mm,
            measurement_tier=scale.tier,
            estimated_error_pct=scale.estimated_error_pct,
            detail={
                "scale_source": scale.tier,
                "mm_per_px": round(scale.mm_per_px, 6),
                "subject_bbox_px": list(segmentation.bbox),
                "scale_detail": scale.detail,
            },
        )

    scaled_mesh, measurements, solution = _timed("measurement", _measure)

    # The third mesh axis is depth the generator invented; nothing in the photo
    # constrains it. Report it separately rather than hiding it among the dims.
    depth_axis = next(a for a in range(3) if a not in solution.mesh_axes)
    inferred_depth_mm = solution.mesh_extents[depth_axis] * solution.factor
    measurements.detail["inferred_depth_mm"] = round(float(inferred_depth_mm), 1)
    measurements.detail["inferred_depth_note"] = (
        "Depth away from the camera is inferred by the 3D model, not measured "
        "from the photo."
    )

    if solution.aspect_error > _ASPECT_WARN:
        warnings.append(
            "The generated mesh's proportions do not match the photo's outline "
            f"(aspect mismatch {solution.aspect_error:.2f}), so these dimensions "
            "are less reliable than the accuracy tier suggests."
        )
    if measurements.volume_basis == "convex_hull":
        warnings.append(
            "The mesh is not watertight, so volume is a convex-hull "
            "over-estimate for anything concave."
        )
    if scale.is_estimate:
        warnings.append(
            "Scale came from a monocular estimate. Add a printed marker or a "
            "bank card to the photo for a real measurement."
        )

    # 6. Outputs.
    glb_path = _timed(
        "export", lambda: _export_mesh(scaled_mesh, run_dir, in_millimetres=False)
    )

    if settings.save_debug:
        overlay = rgb
        corners = scale.detail.get("corners")
        if corners:
            overlay = draw_polygon(overlay, np.asarray(corners), label="size reference")
        overlay = draw_bbox(overlay, segmentation.bbox, label="subject")
        debug["overlay"] = save_image(overlay, run_dir / "reference_overlay.png")

    payload = measurements.to_dict()
    measurements_path = run_dir / "measurements.json"
    measurements_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "app_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "generator": generator_name,
                "scale_source_requested": scale_source,
                "scale_tier_used": scale.tier,
                "measurements": payload,
                "scale_detail": scale.detail,
                "timings_s": timings,
                "warnings": warnings,
                "settings": {
                    "mc_resolution": settings.mc_resolution,
                    "subject_size": settings.subject_size,
                    "rembg_model": settings.rembg_model,
                    "upright_output": settings.upright_output,
                    "device": settings.resolve_device(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PipelineResult(
        run_id=run_id,
        run_dir=run_dir,
        glb_path=glb_path,
        measurements_path=measurements_path,
        measurements=payload,
        scale=scale,
        generator=generator_name,
        timings_s=timings,
        warnings=warnings,
        debug=debug,
    )


def run_or_raise(*args, **kwargs) -> PipelineResult:
    """`run` with unexpected failures wrapped as PipelineError for the UI."""
    try:
        return run(*args, **kwargs)
    except PipelineError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise PipelineError(f"{type(exc).__name__}: {exc}") from exc
