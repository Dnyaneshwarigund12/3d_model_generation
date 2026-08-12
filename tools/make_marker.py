"""Generate a printable ArUco marker at an exact physical size.

The marker's printed side length is the constant the whole measurement chain
divides by, so two things matter more than they look:

* print at 100% scale - any "fit to page" or "shrink to fit" silently changes the
  number and every dimension is then wrong by that factor;
* measure the printed black square with a ruler afterwards and use the measured
  value, not the requested one, if they differ.

Usage:
    python tools/make_marker.py --mm 50 --out assets/markers/marker_50mm.png
    python tools/make_marker.py --mm 80 --pdf --out assets/markers/marker_80mm.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_ARUCO_DICT, DEFAULT_MARKER_MM  # noqa: E402

MM_PER_INCH = 25.4


def render_marker(
    marker_mm: float,
    dictionary_name: str = DEFAULT_ARUCO_DICT,
    marker_id: int = 0,
    dpi: int = 300,
    quiet_zone_ratio: float = 0.2,
) -> Image.Image:
    if not hasattr(cv2.aruco, dictionary_name):
        raise SystemExit(f"Unknown ArUco dictionary {dictionary_name!r}.")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    side_px = int(round(marker_mm / MM_PER_INCH * dpi))
    if hasattr(cv2.aruco, "generateImageMarker"):  # OpenCV >= 4.7
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    else:  # OpenCV <= 4.6
        marker = cv2.aruco.drawMarker(dictionary, marker_id, side_px)

    quiet = int(round(side_px * quiet_zone_ratio))
    font_size = max(12, int(side_px * 0.075))
    caption_px = int(font_size * 1.6)
    ruler_px = int(round(side_px * 0.09))

    # Nothing black may touch the marker's quiet zone: the detector traces the
    # marker's outer contour, so an adjacent mark merges into it and inflates the
    # measured side length. Hence the generous white gaps below.
    marker_bottom = quiet + side_px
    caption_y = marker_bottom + quiet
    ruler_y = caption_y + caption_px + quiet
    canvas = Image.new(
        "RGB",
        (side_px + 2 * quiet, ruler_y + ruler_px + quiet),
        "white",
    )
    canvas.paste(Image.fromarray(marker).convert("RGB"), (quiet, quiet))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    caption = (
        f"{marker_mm:g} mm side  -  {dictionary_name} id {marker_id}  -  "
        "print at 100% scale, do not fit to page"
    )
    draw.text((quiet, caption_y), caption, fill="black", font=font)

    # A ruler well clear of the marker, to check the print came out at scale.
    step_mm = 10
    draw.line(
        [(quiet, ruler_y + ruler_px), (quiet + side_px, ruler_y + ruler_px)],
        fill="black",
        width=2,
    )
    for mm in range(0, int(marker_mm) + 1, step_mm):
        x = quiet + int(round(mm / marker_mm * side_px))
        draw.line(
            [(x, ruler_y), (x, ruler_y + ruler_px)],
            fill="black",
            width=2,
        )

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mm", type=float, default=DEFAULT_MARKER_MM)
    parser.add_argument("--dict", dest="dictionary", default=DEFAULT_ARUCO_DICT)
    parser.add_argument("--id", dest="marker_id", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf", action="store_true", help="Also write a PDF next to the PNG.")
    parser.add_argument(
        "--out", type=Path, default=Path("assets/markers/marker.png"), help="Output PNG path"
    )
    args = parser.parse_args()

    image = render_marker(args.mm, args.dictionary, args.marker_id, args.dpi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, dpi=(args.dpi, args.dpi))
    print(f"Wrote {args.out} ({image.width}x{image.height}px at {args.dpi}dpi)")

    if args.pdf:
        pdf_path = args.out.with_suffix(".pdf")
        image.save(pdf_path, "PDF", resolution=float(args.dpi))
        print(f"Wrote {pdf_path}")

    # Round-trip check: the generated marker must be detectable and measure back
    # to the size requested.
    from app.scale import from_aruco

    result = from_aruco(np.asarray(image), args.mm, args.dictionary)
    measured = result.mm_per_px * (args.mm / MM_PER_INCH * args.dpi)
    print(
        f"Self-check: detected, {result.mm_per_px:.5f} mm/px, "
        f"round-trips to {measured:.2f} mm (asked for {args.mm:g} mm)"
    )


if __name__ == "__main__":
    main()
