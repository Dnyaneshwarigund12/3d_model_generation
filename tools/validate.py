"""Compare reported dimensions against tape-measured ground truth.

The acceptance criterion for this project is not "the pipeline runs" - it is "the
numbers are right". This script is what turns the error percentages in
`app/scale.py` from published estimates into measured ones.

How to use it:

 1. Pick 10-15 objects you can measure with a tape or calipers, covering the sizes
    and shapes you actually care about, including a few awkward ones (dark,
    shiny, thin, cluttered background).
 2. Photograph each with a printed marker lying flat beside it, in the same plane.
 3. Record the truth in a CSV (see validation_manifest.example.csv).
 4. Run:  python tools/validate.py --manifest my_objects.csv

Errors are compared dimension by dimension after sorting both the truth and the
prediction largest-first, because the pipeline reports extents in that order and
has no notion of which way up the object was.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.errors import PipelineError  # noqa: E402
from app.pipeline import run  # noqa: E402


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("image")]
    if not rows:
        raise SystemExit(f"No usable rows in {path}.")
    return rows


def _truth(row: dict[str, str]) -> list[float]:
    values = [
        float(row[key])
        for key in ("truth_length_mm", "truth_width_mm", "truth_height_mm")
        if row.get(key)
    ]
    if len(values) != 3:
        raise SystemExit(
            f"Row for {row['image']} needs all three truth dimensions in mm."
        )
    return sorted(values, reverse=True)


def evaluate(manifest: Path, settings: Settings, generator: str | None) -> dict:
    rows = _read_manifest(manifest)
    records: list[dict] = []

    for row in rows:
        image_path = (manifest.parent / row["image"]).resolve()
        if not image_path.exists():
            image_path = Path(row["image"]).resolve()
        try:
            result = run(
                image_path,
                settings=settings,
                generator=generator,
                scale_source=row.get("scale_source") or "auto",
                marker_mm=float(row["marker_mm"]) if row.get("marker_mm") else 50.0,
                known_mm=float(row["known_mm"]) if row.get("known_mm") else None,
                known_axis=row.get("known_axis") or "height",
            )
        except PipelineError as exc:
            print(f"FAILED {row['image']}: {exc}")
            records.append({"image": row["image"], "error": str(exc)})
            continue

        truth = _truth(row)
        predicted = sorted(
            [
                result.measurements["length_mm"],
                result.measurements["width_mm"],
                result.measurements["height_mm"],
            ],
            reverse=True,
        )
        errors_pct = [
            100.0 * (p - t) / t for p, t in zip(predicted, truth)
        ]
        worst = max(abs(e) for e in errors_pct)
        mean_abs = statistics.fmean(abs(e) for e in errors_pct)

        records.append(
            {
                "image": row["image"],
                "notes": row.get("notes", ""),
                "tier": result.measurements["measurement_tier"],
                "claimed_error_pct": result.measurements["estimated_error_pct"],
                "truth_mm": [round(v, 1) for v in truth],
                "predicted_mm": [round(v, 1) for v in predicted],
                "errors_pct": [round(e, 1) for e in errors_pct],
                "mean_abs_error_pct": round(mean_abs, 1),
                "worst_abs_error_pct": round(worst, 1),
                "run_dir": str(result.run_dir),
            }
        )
        print(
            f"{row['image']:<28} {result.measurements['measurement_tier']:<18} "
            f"mean {mean_abs:5.1f}%  worst {worst:5.1f}%"
        )

    per_tier: dict[str, dict] = {}
    for record in records:
        if "error" in record:
            continue
        tier = per_tier.setdefault(
            record["tier"], {"count": 0, "errors": [], "claimed": []}
        )
        tier["count"] += 1
        tier["errors"].append(record["mean_abs_error_pct"])
        tier["claimed"].append(record["claimed_error_pct"])

    summary = {}
    for tier, data in per_tier.items():
        errors = data["errors"]
        summary[tier] = {
            "objects": data["count"],
            "mean_abs_error_pct": round(statistics.fmean(errors), 1),
            "median_abs_error_pct": round(statistics.median(errors), 1),
            "worst_abs_error_pct": round(max(errors), 1),
            "claimed_error_pct": round(statistics.fmean(data["claimed"]), 1),
        }

    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/validation_report.json"))
    args = parser.parse_args()

    settings = Settings.from_env()
    report = evaluate(args.manifest, settings, args.generator)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nPer-tier summary")
    print(f"{'tier':<20}{'n':>4}{'mean':>8}{'median':>8}{'worst':>8}{'claimed':>9}")
    for tier, stats in report["summary"].items():
        print(
            f"{tier:<20}{stats['objects']:>4}{stats['mean_abs_error_pct']:>7.1f}%"
            f"{stats['median_abs_error_pct']:>7.1f}%{stats['worst_abs_error_pct']:>7.1f}%"
            f"{stats['claimed_error_pct']:>8.1f}%"
        )
    print(f"\nWrote {args.out}")
    print(
        "If measured error is worse than claimed, raise the base values in "
        "app/scale.py rather than leaving the badge optimistic."
    )


if __name__ == "__main__":
    main()
