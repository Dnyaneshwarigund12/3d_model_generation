"""Tier 3 scale: a monocular metric-depth model, used when the photo has no reference.

This is the fallback that makes "just upload a photo" always return a number, and
it is the weakest link in the system. These models predict absolute distance in
metres per pixel from a single RGB image, which sounds like it solves the scale
problem - it does not. They are statistical priors learned mostly from everyday
indoor and street scenes, so they work on furniture and vehicles and generalise
badly to unusual objects, extreme close-ups and odd framing. Published zero-shot
error is commonly 5-20% and worse out of distribution
(`04-measurement-methodology.md` section 2, Tier 3).

Results from this path are labelled an estimate everywhere they surface. Do not
quietly promote them.

The output feeds the same code as every other scale source: the metric point
cloud gives the subject's real width and height, which divided by its pixel width
and height is millimetres per pixel.

UniDepth v2 is the backend because it predicts its own camera intrinsics, so it
needs nothing from EXIF - uploaded photos are frequently stripped of it. Metric3D
v2 is the documented alternative if UniDepth's licence or quality disappoints;
it would slot in as another `_load_*` function here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings
from .errors import ScaleError
from .imaging import as_rgb
from .segment import Segmentation

_UNIDEPTH_CHECKPOINT = "lpiccinelli/unidepth-v2-vitl14"

# Trim this fraction from each end of the masked point cloud before measuring
# its extent. Mask edges pick up background depth, and a single stray pixel at
# the wrong distance would otherwise set the object's size.
_OUTLIER_PERCENTILE = 1.0

_MODELS: dict[str, object] = {}


@dataclass(frozen=True)
class DepthPrediction:
    """Metric depth and the point cloud back-projected from it."""

    depth_m: np.ndarray  # (H, W) metres
    points_m: np.ndarray  # (3, H, W) metres, camera frame
    intrinsics: np.ndarray | None  # (3, 3) if the model reported them
    model: str


def _load_unidepth(device: str):
    if _UNIDEPTH_CHECKPOINT in _MODELS:
        return _MODELS[_UNIDEPTH_CHECKPOINT]
    try:
        from unidepth.models import UniDepthV2
    except ImportError as exc:
        raise ScaleError(
            "UniDepth is not installed, so a photo with no reference object "
            "cannot be measured. Add a printed marker or a bank card to the "
            "photo, or enter a known dimension. (The Colab notebook has an "
            "optional cell that installs UniDepth.)"
        ) from exc

    model = UniDepthV2.from_pretrained(_UNIDEPTH_CHECKPOINT)
    model = model.to(device).eval()
    _MODELS[_UNIDEPTH_CHECKPOINT] = model
    return model


def _backproject(depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Pinhole back-projection, the same maths as the Tier 1 sensor path.

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    height, width = depth_m.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    x = (us - cx) * depth_m / max(fx, 1e-6)
    y = (vs - cy) * depth_m / max(fy, 1e-6)
    return np.stack([x, y, depth_m], axis=0)


def predict_depth(
    image: np.ndarray, settings: Settings | None = None
) -> DepthPrediction:
    """Run the metric depth model over an RGB image."""
    settings = settings or Settings()
    if settings.depth_model != "unidepth":
        raise ScaleError(
            f"Unsupported depth model {settings.depth_model!r}. Only 'unidepth' "
            "is wired up; see this module's docstring for how to add another."
        )

    import torch

    device = settings.resolve_device()
    model = _load_unidepth(device)

    rgb = as_rgb(image)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device)
    with torch.no_grad():
        prediction = model.infer(tensor)

    depth = prediction["depth"].squeeze().float().cpu().numpy()
    intrinsics = prediction.get("intrinsics")
    if intrinsics is not None:
        intrinsics = intrinsics.squeeze().float().cpu().numpy()

    points = prediction.get("points")
    if points is not None:
        points = points.squeeze().float().cpu().numpy()
    elif intrinsics is not None:
        points = _backproject(depth, intrinsics)
    else:
        raise ScaleError(
            "The depth model returned neither a point cloud nor camera "
            "intrinsics, so pixels cannot be converted to millimetres."
        )

    return DepthPrediction(
        depth_m=depth, points_m=points, intrinsics=intrinsics, model="unidepth-v2"
    )


def _robust_extent(values: np.ndarray) -> float:
    low = np.percentile(values, _OUTLIER_PERCENTILE)
    high = np.percentile(values, 100.0 - _OUTLIER_PERCENTILE)
    return float(high - low)


def scale_from_metric_depth(
    image: np.ndarray,
    segmentation: Segmentation,
    settings: Settings | None = None,
) -> "object":
    """Turn a predicted metric point cloud into millimetres per pixel."""
    from .scale import TIER_MONOCULAR, ScaleResult, lsq_mm_per_px

    settings = settings or Settings()
    prediction = predict_depth(image, settings)

    mask = segmentation.mask
    points = prediction.points_m
    if points.shape[1:] != mask.shape:
        raise ScaleError(
            f"Depth map shape {points.shape[1:]} does not match the image "
            f"{mask.shape}."
        )

    xs = points[0][mask]
    ys = points[1][mask]
    zs = points[2][mask]
    finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & (zs > 0)
    if finite.sum() < 50:
        raise ScaleError("The depth model produced no usable depth on the subject.")

    width_mm = _robust_extent(xs[finite]) * 1000.0
    height_mm = _robust_extent(ys[finite]) * 1000.0
    if width_mm <= 0 or height_mm <= 0:
        raise ScaleError("The predicted point cloud has no extent on the subject.")

    mm_per_px = lsq_mm_per_px(
        [segmentation.width_px, segmentation.height_px], [width_mm, height_mm]
    )

    subject_fraction = float(mask.sum()) / float(mask.size)
    error = 20.0
    if subject_fraction < 0.02:
        # A subject filling almost none of the frame gives the model very little
        # to work with.
        error += 10.0

    return ScaleResult(
        mm_per_px=mm_per_px,
        tier=TIER_MONOCULAR,
        estimated_error_pct=error,
        detail={
            "depth_model": prediction.model,
            "median_distance_m": round(float(np.median(zs[finite])), 3),
            "predicted_width_mm": round(width_mm, 1),
            "predicted_height_mm": round(height_mm, 1),
            "subject_frame_fraction": round(subject_fraction, 4),
            "intrinsics_source": (
                "predicted by the model" if prediction.intrinsics is not None else "none"
            ),
            "error_basis": "heuristic; this tier is a learned guess",
        },
    )
