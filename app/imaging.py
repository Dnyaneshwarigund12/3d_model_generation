"""Small image helpers shared by the pipeline stages.

Everything in the pipeline passes numpy arrays in RGB or RGBA order (not BGR),
so conversions to OpenCV's BGR convention happen at the call site.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array."""
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def as_rgb(image: np.ndarray) -> np.ndarray:
    """Drop an alpha channel by compositing over white, or expand greyscale."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    return arr[:, :, :3].astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(as_rgb(image), cv2.COLOR_RGB2GRAY)


def save_image(image: np.ndarray, path: str | Path) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else None
    Image.fromarray(arr.astype(np.uint8), mode=mode).save(path)
    return path


def composite_on_gray(rgba: np.ndarray, value: float = 0.5) -> np.ndarray:
    """Flatten an RGBA cutout onto a flat grey background.

    The single-image-to-3D models were trained on cutouts composited this way,
    so feeding them a transparent PNG or a white background measurably changes
    the result.
    """
    arr = np.asarray(rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3].astype(np.float32) / 255.0
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        out = rgb * alpha + value * (1.0 - alpha)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return as_rgb(arr)


def draw_polygon(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int] = (0, 220, 90),
    thickness: int = 3,
    label: str | None = None,
) -> np.ndarray:
    """Return a copy of `image` with a closed polygon (and optional label) drawn."""
    import cv2

    out = as_rgb(image).copy()
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        scale, weight = 0.6, 2
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, weight
        )
        x, y = pts.reshape(-1, 2).min(axis=0)
        # Keep the text inside the frame, so labels on a shape near an edge stay
        # readable instead of running off it.
        x = int(np.clip(x, 0, max(0, out.shape[1] - text_w)))
        y = int(np.clip(y - 8, text_h + 2, out.shape[0] - 2))
        cv2.putText(
            out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA
        )
    return out


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int] = (255, 170, 0),
    thickness: int = 3,
    label: str | None = None,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return draw_polygon(image, corners, color=color, thickness=thickness, label=label)
