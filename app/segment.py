"""Stage 1: separate the subject from its background.

Uses rembg (U^2-Net) because it runs on CPU in about a second and needs no GPU
budget, leaving the whole card free for generation. `01-research-notes.md`
section 2 lists the quality upgrades (SAM2/SAM3, RMBG-2.0, BiRefNet) - they slot
in behind `remove_background` without touching any caller.

This stage also owns two jobs the rest of the pipeline depends on:

* it produces the subject's pixel bounding box, which is what the scale factor
  gets multiplied by to obtain a real-world size;
* it removes the reference marker from the mask, so the 3D generator never sees
  the marker and cannot fuse it into the object's geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import Settings
from .errors import SegmentationError
from .imaging import as_rgb

_SESSIONS: dict[str, object] = {}


@dataclass(frozen=True)
class Segmentation:
    """Result of background removal.

    Attributes:
        rgba: full-frame cutout, background alpha zeroed.
        mask: boolean subject mask, same height/width as the input.
        bbox: subject bounds as (x0, y0, x1, y1), half-open on x1/y1.
    """

    rgba: np.ndarray
    mask: np.ndarray
    bbox: tuple[int, int, int, int]

    @property
    def width_px(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height_px(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())

    def subject(
        self,
        padding: float = 0.06,
        size: int | None = 768,
        square: bool = True,
    ) -> np.ndarray:
        """Crop to the subject and centre it on a square transparent canvas.

        The generators expect a centred subject with margin around it; framing
        the object edge-to-edge degrades their output noticeably.
        """
        from PIL import Image

        x0, y0, x1, y1 = self.bbox
        pad = int(round(max(x1 - x0, y1 - y0) * padding))
        h, w = self.mask.shape
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
        crop = self.rgba[cy0:cy1, cx0:cx1]

        if square:
            side = max(crop.shape[0], crop.shape[1])
            canvas = np.zeros((side, side, 4), dtype=np.uint8)
            oy = (side - crop.shape[0]) // 2
            ox = (side - crop.shape[1]) // 2
            canvas[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
            crop = canvas

        if size:
            if square:
                target = (size, size)
            else:
                longest = max(crop.shape[0], crop.shape[1])
                ratio = size / float(longest)
                target = (
                    max(1, int(round(crop.shape[1] * ratio))),
                    max(1, int(round(crop.shape[0] * ratio))),
                )
            img = Image.fromarray(crop, mode="RGBA").resize(target, Image.LANCZOS)
            crop = np.asarray(img, dtype=np.uint8)
        return crop


def _import_rembg():
    """Import rembg, distinguishing "missing" from "present but broken".

    rembg pulls in pymatting/numba/cv2, and a version clash in any of those also
    surfaces as ImportError. Reporting "not installed" for every failure sends
    people to reinstall a package pip already considers satisfied.
    """
    try:
        from rembg import new_session, remove
    except Exception as exc:  # pragma: no cover - depends on environment
        import importlib.util

        try:
            installed = importlib.util.find_spec("rembg") is not None
        except Exception:
            installed = True
        if not installed:
            raise SegmentationError(
                "rembg is not installed. Run: pip install rembg onnxruntime"
            ) from exc
        raise SegmentationError(
            "rembg is installed but failed to import - this is a dependency "
            f"version clash, not a missing package: {type(exc).__name__}: {exc}. "
            "If a pip install ran after this kernel started (Colab steps 4-6), "
            "restart the runtime and run the cells again; the installs will be "
            "no-ops the second time."
        ) from exc
    return new_session, remove


def _get_session(model_name: str):
    if model_name not in _SESSIONS:
        new_session, _ = _import_rembg()
        _SESSIONS[model_name] = new_session(model_name)
    return _SESSIONS[model_name]


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest blob, dropping specks and stray background objects."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 2:  # background plus at most one component
        return mask
    # Row 0 is the background label.
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _zero_polygons(mask: np.ndarray, polygons: Iterable[np.ndarray]) -> np.ndarray:
    import cv2

    out = mask.astype(np.uint8).copy()
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(out, pts, 0)
    return out.astype(bool)


def remove_background(
    image: np.ndarray,
    settings: Settings | None = None,
    *,
    exclude_polygons: Sequence[np.ndarray] | None = None,
    keep_largest: bool = True,
) -> Segmentation:
    """Cut the subject out of `image`.

    Args:
        image: RGB or RGBA uint8 array.
        exclude_polygons: regions to force out of the mask - used to delete the
            reference marker so it is not reconstructed as part of the object.
        keep_largest: keep only the largest connected component.
    """
    settings = settings or Settings()
    rgb = as_rgb(image)

    from PIL import Image

    session = _get_session(settings.rembg_model)
    _, remove = _import_rembg()

    cutout = remove(Image.fromarray(rgb), session=session)
    rgba = np.asarray(cutout.convert("RGBA"), dtype=np.uint8)

    mask = rgba[:, :, 3] > settings.alpha_threshold
    if exclude_polygons:
        mask = _zero_polygons(mask, exclude_polygons)
    if not mask.any():
        raise SegmentationError(
            "No subject found in the photo. Try a plainer background, or make "
            "sure the object is not the same colour as what is behind it."
        )
    if keep_largest:
        mask = _largest_component(mask)

    rgba = rgba.copy()
    rgba[~mask] = 0
    return Segmentation(rgba=rgba, mask=mask, bbox=_mask_to_bbox(mask))


def segmentation_from_mask(image: np.ndarray, mask: np.ndarray) -> Segmentation:
    """Build a Segmentation from a mask computed elsewhere (used by tests)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise SegmentationError("Mask is empty.")
    rgb = as_rgb(image)
    rgba = np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])
    rgba[~mask] = 0
    return Segmentation(rgba=rgba, mask=mask, bbox=_mask_to_bbox(mask))
