"""Stage 3: establish real-world scale, which cannot come from the pixels.

A single photo is scale-ambiguous: a toy car close up and a real car far away
project to identical pixels (`04-measurement-methodology.md` section 1). So scale
has to be supplied from outside the image content. This module implements the
sources that are possible for an uploaded photo, best first:

    1. a printed ArUco marker of known side length      (~4% error)
    2. a bank card at the standard ISO/IEC 7810 ID-1 size (~10% error)
    3. a dimension the user already knows and types in    (their own accuracy)
    4. a monocular metric-depth model                     (20%+, an estimate)

Tier 1 from the measurement doc (phone LiDAR / ARCore depth) is not reachable
here because an uploaded photo carries no depth map.

Every source reduces to one number - **millimetres per pixel** - so a single
piece of code applies the result to the mesh. The dominant error in the marker
and card paths is depth mismatch: the ratio is exact only in the reference
object's own plane, and degrades as the subject sits nearer or further than it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import (
    CREDIT_CARD_ASPECT,
    CREDIT_CARD_MM,
    DEFAULT_ARUCO_DICT,
    DEFAULT_MARKER_MM,
    Settings,
)
from .errors import ScaleError
from .imaging import to_gray
from .segment import Segmentation

TIER_MARKER = "reference_marker"
TIER_CARD = "reference_card"
TIER_MANUAL = "manual_reference"
TIER_MONOCULAR = "monocular_estimate"

TIER_LABELS: dict[str, str] = {
    TIER_MARKER: "Measured with a printed reference marker",
    TIER_CARD: "Measured with a reference card",
    TIER_MANUAL: "Measured from a dimension you provided",
    TIER_MONOCULAR: "Estimated from the photo alone",
}

TIER_IS_ESTIMATE: dict[str, bool] = {
    TIER_MARKER: False,
    TIER_CARD: False,
    TIER_MANUAL: False,
    TIER_MONOCULAR: True,
}

# Starting points for the reported error, before per-photo penalties. These are
# the published/expected ranges from the measurement doc, not error bars we have
# measured ourselves - `tools/validate.py` is what replaces them with real
# numbers once the tape-measure run is done.
_BASE_ERROR_PCT: dict[str, float] = {
    TIER_MARKER: 4.0,
    TIER_CARD: 10.0,
    TIER_MANUAL: 1.5,
    TIER_MONOCULAR: 20.0,
}


@dataclass(frozen=True)
class ScaleResult:
    """A resolved pixel-to-millimetre ratio and how much to trust it."""

    mm_per_px: float
    tier: str
    estimated_error_pct: float
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return TIER_LABELS.get(self.tier, self.tier)

    @property
    def is_estimate(self) -> bool:
        return TIER_IS_ESTIMATE.get(self.tier, True)

    def size_mm(self, pixels: float) -> float:
        return float(pixels) * self.mm_per_px


@dataclass(frozen=True)
class QuadDetection:
    """A four-cornered reference object found in the image."""

    corners: np.ndarray  # (4, 2) float, in pixel coordinates
    edges_px: np.ndarray  # (4,) edge lengths, in corner order
    marker_id: int | None = None

    @property
    def min_edge_px(self) -> float:
        return float(self.edges_px.min())

    @property
    def edge_ratio(self) -> float:
        return float(self.edges_px.max() / max(self.edges_px.min(), 1e-6))


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def quad_edges(corners: np.ndarray) -> np.ndarray:
    """Side lengths of a quadrilateral, in corner order."""
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    rolled = np.roll(pts, -1, axis=0)
    return np.linalg.norm(rolled - pts, axis=1)


def lsq_mm_per_px(pixel_lengths: Sequence[float], mm_lengths: Sequence[float]) -> float:
    """Least-squares ratio k minimising sum((mm_i - k * px_i)^2).

    Averages several independent measurements of the same ratio. Used where the
    inputs are unbiased, e.g. fitting a point cloud's width and height together.
    Not used for reference-object edges - see `mm_per_px_from_quad`.
    """
    px = np.asarray(pixel_lengths, dtype=np.float64)
    mm = np.asarray(mm_lengths, dtype=np.float64)
    denom = float((px**2).sum())
    if denom <= 0:
        raise ScaleError("Reference object has zero size in pixels.")
    return float((mm * px).sum() / denom)


def mm_per_px_from_quad(
    pixel_lengths: Sequence[float], mm_lengths: Sequence[float]
) -> float:
    """Millimetres per pixel from a reference quad, taking its least foreshortened edge.

    Averaging the four edges is the obvious choice and it is measurably worse. A
    reference object tilted away from the sensor images *smaller* than it is, and
    only smaller - perspective can shorten an edge but never lengthen it. So the
    four edges are not noisy samples of one ratio, they are one good sample and
    three that are biased in a known direction, and averaging them drags the
    estimate toward the bias.

    Taking the smallest millimetres-per-pixel ratio, which is the longest edge
    relative to its real length, picks the least foreshortened measurement. On
    synthetic tilt tests this removes essentially all of the bias when the tilt is
    about one axis (20.7% down to 0.1% at a 55% squash) and roughly halves it under
    two-axis tilt (15.5% down to 6.6%), while being no worse on small or blurred
    markers.
    """
    px = np.asarray(pixel_lengths, dtype=np.float64)
    mm = np.asarray(mm_lengths, dtype=np.float64)
    if px.size == 0 or (px <= 0).all():
        raise ScaleError("Reference object has zero size in pixels.")
    usable = px > 0
    return float(np.min(mm[usable] / px[usable]))


def _size_penalty_pct(min_edge_px: float) -> float:
    """Extra error for a reference object that occupies too few pixels.

    Corner localisation error is roughly constant in pixels, so its effect on
    the ratio grows as the reference gets smaller in frame.
    """
    if min_edge_px >= 120:
        return 0.0
    if min_edge_px >= 80:
        return 1.0
    if min_edge_px >= 50:
        return 3.0
    if min_edge_px >= 30:
        return 8.0
    return 15.0


def _tilt_penalty_pct(edge_ratio: float, expected_ratio: float = 1.0) -> float:
    """Extra error when the reference is not facing the camera squarely.

    `mm_per_px_from_quad` already removes most foreshortening bias, so this covers
    what is left: the residual under two-axis tilt (measured at 4-7% on synthetic
    tests) plus the fact that a tilted reference object is unlikely to be sitting
    in the same plane as the subject, which is the assumption the whole ratio rests
    on.
    """
    deviation = abs(edge_ratio - expected_ratio) / expected_ratio
    if deviation <= 0.08:
        return 0.0
    if deviation <= 0.2:
        return 2.0
    if deviation <= 0.4:
        return 6.0
    return 12.0


# --------------------------------------------------------------------------- #
# ArUco marker (best available source for an uploaded photo)
# --------------------------------------------------------------------------- #


def _aruco_dictionary(name: str):
    import cv2

    if not hasattr(cv2, "aruco"):
        raise ScaleError(
            "This OpenCV build has no aruco module. Install opencv-python "
            "(>=4.8) or opencv-contrib-python."
        )
    if not hasattr(cv2.aruco, name):
        raise ScaleError(f"Unknown ArUco dictionary {name!r}.")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _detector_params():
    """Detector parameters, with sub-pixel corner refinement enabled.

    Refinement matters here: the marker's pixel size divides straight into every
    reported dimension.
    """
    import cv2

    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters_create"):  # OpenCV <= 4.6
        params = aruco.DetectorParameters_create()
    else:  # OpenCV >= 4.7
        params = aruco.DetectorParameters()
    try:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    except Exception:  # pragma: no cover - older builds without refinement
        pass
    return params


def detect_markers(
    image: np.ndarray, dictionary_name: str = DEFAULT_ARUCO_DICT
) -> list[QuadDetection]:
    """Find every ArUco marker in the image."""
    import cv2

    gray = to_gray(image)
    dictionary = _aruco_dictionary(dictionary_name)
    params = _detector_params()

    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # OpenCV <= 4.6
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    found: list[QuadDetection] = []
    if corners is None or len(corners) == 0:
        return found
    id_list = ids.flatten().tolist() if ids is not None else [None] * len(corners)
    for quad, marker_id in zip(corners, id_list):
        pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
        found.append(
            QuadDetection(
                corners=pts,
                edges_px=quad_edges(pts),
                marker_id=None if marker_id is None else int(marker_id),
            )
        )
    return found


def from_aruco(
    image: np.ndarray,
    marker_mm: float = DEFAULT_MARKER_MM,
    dictionary_name: str = DEFAULT_ARUCO_DICT,
) -> ScaleResult:
    """Resolve scale from one or more printed ArUco markers of known side length."""
    if marker_mm <= 0:
        raise ScaleError("Marker size must be greater than zero.")

    markers = detect_markers(image, dictionary_name)
    if not markers:
        raise ScaleError(
            "No ArUco marker found. Check that the whole marker is visible, in "
            "focus and not glared out, or pick a different scale source."
        )

    ratios = [mm_per_px_from_quad(m.edges_px, [marker_mm] * 4) for m in markers]
    # Median across markers: one badly angled marker cannot drag the result.
    mm_per_px = float(np.median(ratios))
    best = min(
        markers,
        key=lambda m: abs(mm_per_px_from_quad(m.edges_px, [marker_mm] * 4) - mm_per_px),
    )

    error = (
        _BASE_ERROR_PCT[TIER_MARKER]
        + _size_penalty_pct(best.min_edge_px)
        + _tilt_penalty_pct(best.edge_ratio)
    )
    return ScaleResult(
        mm_per_px=mm_per_px,
        tier=TIER_MARKER,
        estimated_error_pct=round(error, 1),
        detail={
            "marker_mm": marker_mm,
            "dictionary": dictionary_name,
            "markers_found": len(markers),
            "marker_ids": [m.marker_id for m in markers],
            "min_edge_px": round(best.min_edge_px, 1),
            "edge_ratio": round(best.edge_ratio, 3),
            "per_marker_mm_per_px": [round(r, 5) for r in ratios],
            "corners": best.corners.round(1).tolist(),
            "error_basis": "heuristic",
        },
    )


# --------------------------------------------------------------------------- #
# Bank card fallback
# --------------------------------------------------------------------------- #


def detect_card(
    image: np.ndarray,
    aspect_tolerance: float = 0.18,
    min_area_fraction: float = 0.002,
    max_area_fraction: float = 0.5,
) -> QuadDetection | None:
    """Look for a card-shaped quadrilateral (ID-1 aspect ratio, ~1.586:1).

    Much weaker than marker detection: it finds *a* rectangle of about the right
    proportions, with no identity check, so it can lock onto a book, a phone or a
    tile. The overlay in the run's debug output exists so this can be eyeballed.
    """
    import cv2

    gray = to_gray(image)
    image_area = float(gray.shape[0] * gray.shape[1])
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, QuadDetection]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = abs(cv2.contourArea(approx))
        if not (min_area_fraction * image_area <= area <= max_area_fraction * image_area):
            continue

        pts = approx.reshape(4, 2).astype(np.float64)
        edges_px = quad_edges(pts)
        long_side = float(np.sort(edges_px)[-2:].mean())
        short_side = float(np.sort(edges_px)[:2].mean())
        if short_side <= 1e-6:
            continue
        aspect_error = abs(long_side / short_side - CREDIT_CARD_ASPECT)
        if aspect_error > aspect_tolerance * CREDIT_CARD_ASPECT:
            continue
        candidates.append(
            (aspect_error, -area, QuadDetection(corners=pts, edges_px=edges_px))
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def from_card(
    image: np.ndarray, card_mm: tuple[float, float] = CREDIT_CARD_MM
) -> ScaleResult:
    """Resolve scale from a bank card lying in the same plane as the subject."""
    detection = detect_card(image)
    if detection is None:
        raise ScaleError(
            "No card-shaped rectangle found. Lay the card flat next to the "
            "object with all four corners visible, or use a printed marker."
        )

    # Opposite sides of the quad are the ones that share a real length, so pair
    # them by position (0 with 2, 1 with 3) rather than by sorting all four - a
    # tilted card can image with a long edge shorter than a short one.
    edges = detection.edges_px
    mm_lengths = np.empty(4, dtype=np.float64)
    if edges[0] + edges[2] >= edges[1] + edges[3]:
        mm_lengths[[0, 2]], mm_lengths[[1, 3]] = card_mm[0], card_mm[1]
    else:
        mm_lengths[[0, 2]], mm_lengths[[1, 3]] = card_mm[1], card_mm[0]
    mm_per_px = mm_per_px_from_quad(edges, mm_lengths)

    long_side = float(np.sort(detection.edges_px)[-2:].mean())
    short_side = float(np.sort(detection.edges_px)[:2].mean())
    error = (
        _BASE_ERROR_PCT[TIER_CARD]
        + _size_penalty_pct(detection.min_edge_px)
        + _tilt_penalty_pct(long_side / max(short_side, 1e-6), CREDIT_CARD_ASPECT)
    )
    return ScaleResult(
        mm_per_px=mm_per_px,
        tier=TIER_CARD,
        estimated_error_pct=round(error, 1),
        detail={
            "card_mm": list(card_mm),
            "aspect_detected": round(long_side / max(short_side, 1e-6), 3),
            "aspect_expected": round(CREDIT_CARD_ASPECT, 3),
            "min_edge_px": round(detection.min_edge_px, 1),
            "corners": detection.corners.round(1).tolist(),
            "error_basis": "heuristic",
        },
    )


# --------------------------------------------------------------------------- #
# User-supplied dimension
# --------------------------------------------------------------------------- #


def from_manual(
    known_mm: float, axis: str, segmentation: Segmentation
) -> ScaleResult:
    """Resolve scale from one dimension of the object that the user already knows.

    Also the ground-truth path for validation: give it a tape-measured height and
    every other reported dimension becomes checkable.
    """
    if known_mm <= 0:
        raise ScaleError("The known dimension must be greater than zero.")
    axis = axis.lower().strip()
    if axis in {"width", "w"}:
        pixels = segmentation.width_px
    elif axis in {"height", "h"}:
        pixels = segmentation.height_px
    else:
        raise ScaleError(f"Unknown axis {axis!r}; expected 'width' or 'height'.")
    if pixels <= 0:
        raise ScaleError("The subject has no extent along that axis.")

    return ScaleResult(
        mm_per_px=known_mm / float(pixels),
        tier=TIER_MANUAL,
        estimated_error_pct=_BASE_ERROR_PCT[TIER_MANUAL],
        detail={
            "known_mm": known_mm,
            "axis": axis,
            "axis_px": int(pixels),
            "error_basis": "assumes your measurement is accurate",
        },
    )


# --------------------------------------------------------------------------- #
# Monocular metric depth
# --------------------------------------------------------------------------- #


def from_monocular_depth(
    image: np.ndarray,
    segmentation: Segmentation,
    settings: Settings | None = None,
) -> ScaleResult:
    """Estimate scale with a metric monocular depth model. Genuinely a guess."""
    from .depth import scale_from_metric_depth

    return scale_from_metric_depth(image, segmentation, settings)


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #

SOURCE_MARKER = "marker"
SOURCE_CARD = "card"
SOURCE_MANUAL = "manual"
SOURCE_ESTIMATE = "estimate"
SOURCE_AUTO = "auto"

SOURCES = (SOURCE_MARKER, SOURCE_CARD, SOURCE_MANUAL, SOURCE_ESTIMATE, SOURCE_AUTO)


def resolve_scale(
    image: np.ndarray,
    segmentation: Segmentation,
    *,
    source: str = SOURCE_AUTO,
    marker_mm: float = DEFAULT_MARKER_MM,
    aruco_dict: str = DEFAULT_ARUCO_DICT,
    known_mm: float | None = None,
    known_axis: str = "height",
    settings: Settings | None = None,
) -> ScaleResult:
    """Resolve scale from the requested source.

    `auto` walks the sources from most to least accurate and takes the first that
    works, which is what makes "just upload a photo" always return something.
    """
    source = (source or SOURCE_AUTO).lower().strip()
    if source not in SOURCES:
        raise ScaleError(f"Unknown scale source {source!r}; expected one of {SOURCES}.")

    if source == SOURCE_MARKER:
        return from_aruco(image, marker_mm, aruco_dict)
    if source == SOURCE_CARD:
        return from_card(image)
    if source == SOURCE_MANUAL:
        if known_mm is None:
            raise ScaleError("Enter the known dimension in millimetres.")
        return from_manual(known_mm, known_axis, segmentation)
    if source == SOURCE_ESTIMATE:
        return from_monocular_depth(image, segmentation, settings)

    attempts: list[str] = []
    if known_mm:
        try:
            return from_manual(known_mm, known_axis, segmentation)
        except ScaleError as exc:
            attempts.append(f"manual: {exc}")
    for attempt in (
        lambda: from_aruco(image, marker_mm, aruco_dict),
        lambda: from_card(image),
        lambda: from_monocular_depth(image, segmentation, settings),
    ):
        try:
            return attempt()
        except ScaleError as exc:
            attempts.append(str(exc))
        except Exception as exc:  # a backend that is not installed
            attempts.append(f"{type(exc).__name__}: {exc}")
    raise ScaleError(
        "Could not establish scale from this photo. Tried:\n- "
        + "\n- ".join(attempts)
    )
