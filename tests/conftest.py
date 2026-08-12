import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def marker_image(
    marker_side_px: int = 200,
    canvas: tuple[int, int] = (900, 1200),
    dictionary: str = "DICT_4X4_50",
    marker_id: int = 0,
    origin: tuple[int, int] = (120, 150),
) -> np.ndarray:
    """A marker of an exact pixel size pasted onto a plain canvas."""
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(d, marker_id, marker_side_px)
    else:
        marker = cv2.aruco.drawMarker(d, marker_id, marker_side_px)

    image = np.full((canvas[0], canvas[1], 3), 235, dtype=np.uint8)
    y, x = origin
    image[y : y + marker_side_px, x : x + marker_side_px] = marker[:, :, None]
    return image


def card_image(
    long_px: int = 317,
    short_px: int = 200,
    canvas: tuple[int, int] = (700, 900),
    origin: tuple[int, int] = (200, 300),
) -> np.ndarray:
    """A light card-shaped rectangle on a dark background."""
    image = np.full((canvas[0], canvas[1], 3), 40, dtype=np.uint8)
    y, x = origin
    image[y : y + short_px, x : x + long_px] = 240
    return image


def subject_mask(
    width_px: int = 300, height_px: int = 150, canvas: tuple[int, int] = (500, 700)
) -> np.ndarray:
    mask = np.zeros(canvas, dtype=bool)
    mask[100 : 100 + height_px, 150 : 150 + width_px] = True
    return mask


@pytest.fixture
def rectangular_subject():
    """An RGB photo plus the matching mask of a 300x150 px rectangular object."""
    mask = subject_mask()
    image = np.full((*mask.shape, 3), 220, dtype=np.uint8)
    image[mask] = (60, 110, 180)
    return image, mask
