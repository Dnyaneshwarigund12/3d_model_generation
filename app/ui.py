"""Gradio front end: upload a photo, get a 3D model and its dimensions.

This is the whole client. Gradio serves the page, renders the GLB in a WebGL
viewer and, with `--share`, hands back a public URL, which is what makes a Colab
notebook usable as the app's host.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_ARUCO_DICT, DEFAULT_MARKER_MM, Settings
from .errors import PipelineError, ScaleError
from .generators import available as available_generators
from .pipeline import run_or_raise
from .scale import TIER_LABELS

_SCALE_CHOICES = [
    ("Auto: use whatever the photo offers", "auto"),
    ("Printed ArUco marker in the photo (most accurate)", "marker"),
    ("Bank card in the photo", "card"),
    ("I know one dimension already", "manual"),
    ("Estimate from the photo alone (roughest)", "estimate"),
]

_INTRO = """
# Photo to 3D model + measurements

Upload one photo of an object. You get back a 3D model you can spin around, plus
its length, width, height and volume.

**For measurements that mean anything, put a size reference in the photo.** A
single photo cannot reveal scale on its own - a toy car close up and a real car
far away make identical pixels. A printed marker or a bank card lying beside the
object, roughly in the same plane, is what turns the shape into millimetres.
Without one, the numbers are a model's guess and are labelled as such.
"""

_MARKER_HELP = """
### Getting a marker

Generate a printable one at an exact size:

```bash
python tools/make_marker.py --mm 50 --out assets/markers/marker_50mm.png
```

Print it **without** any "fit to page" scaling, lay it flat next to the object in
the same plane, and keep all four corners visible and in focus. Then enter the
same millimetre value in the box on the left.

A bank card works as a fallback: every card is 85.60 x 53.98 mm by the ID-1
standard. It is less accurate because detection can lock onto any card-shaped
rectangle, so check the reference overlay in the debug images.
"""


def _tier_badge(tier: str, error_pct: float, warnings: list[str]) -> str:
    label = TIER_LABELS.get(tier, tier)
    estimate = tier == "monocular_estimate"
    headline = (
        f"### Estimated, +/- {error_pct:.0f}%" if estimate else f"### Measured, +/- {error_pct:.0f}%"
    )
    lines = [headline, f"{label}."]
    if warnings:
        lines.append("")
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines)


def _no_scale_notice(message: str) -> str:
    return f"### Needs a size reference\n\n{message}"


def _dimension_rows(measurements: dict) -> list[list[str]]:
    m = measurements
    rows = [
        ["Length", f"{m['length_mm']:.1f} mm", f"{m['length_mm'] / 10:.1f} cm"],
        ["Width", f"{m['width_mm']:.1f} mm", f"{m['width_mm'] / 10:.1f} cm"],
        ["Height", f"{m['height_mm']:.1f} mm", f"{m['height_mm'] / 10:.1f} cm"],
        ["Volume", f"{m['volume_cm3']:.1f} cm3", f"{m['volume_cm3'] / 1000:.3f} L"],
        ["Surface area", f"{m['surface_area_cm2']:.1f} cm2", ""],
    ]
    depth = m.get("detail", {}).get("inferred_depth_mm")
    if depth is not None:
        rows.append(["Depth (inferred by the model)", f"{depth:.1f} mm", ""])
    return rows


def _failure_notice(title: str, message: str) -> str:
    return f"### {title}\n\n{message}"


def _empty_outputs(notice: str, raw: dict):
    """Clear every output box and put the explanation in the badge.

    Raising `gr.Error` makes Gradio paint every component red with a generic
    "Error" badge and hides the real message - that is what the screenshots of
    empty red boxes are. Returning None/[] instead leaves the page usable and
    puts the actual cause where the user is looking.
    """
    return None, notice, [], None, None, [], raw


def process_upload(
    image,
    scale_source: str,
    marker_mm: float,
    aruco_dict: str,
    known_mm: float | None,
    known_axis: str,
    generator: str,
    settings: Settings,
):
    """Run the pipeline and shape the result into the UI's outputs.

    Kept out of `build_ui` so it can be tested without standing up a server.
    Returns, in order: model path, tier badge, dimension rows, measurements file,
    mesh file, debug gallery, raw result.
    """
    if image is None:
        return _empty_outputs(
            _failure_notice("No photo", "Upload a photo first, then press Generate."),
            {"error": "no_photo"},
        )
    try:
        result = run_or_raise(
            image,
            settings=settings,
            generator=generator,
            scale_source=scale_source,
            marker_mm=float(marker_mm or DEFAULT_MARKER_MM),
            aruco_dict=aruco_dict or DEFAULT_ARUCO_DICT,
            known_mm=float(known_mm) if known_mm else None,
            known_axis=known_axis,
        )
    except ScaleError as exc:
        # A refusal, not a fault: the photo contains nothing of known size.
        return _empty_outputs(
            _no_scale_notice(str(exc)),
            {"needs_size_reference": str(exc)},
        )
    except PipelineError as exc:
        return _empty_outputs(
            _failure_notice(f"Stopped: {type(exc).__name__}", str(exc)),
            {"pipeline_error": type(exc).__name__, "message": str(exc)},
        )
    except Exception as exc:  # pragma: no cover - last resort for unexpected faults
        return _empty_outputs(
            _failure_notice(
                "Unexpected failure",
                f"{type(exc).__name__}: {exc}\n\n"
                "Re-run step 5 (`python tools/doctor.py`) in the notebook. "
                "If that passes, try generator `silhouette` with "
                "'I know one dimension already' to confirm the UI path.",
            ),
            {"unexpected_error": type(exc).__name__, "message": str(exc)},
        )

    gallery = [
        (str(path), name.replace("_", " "))
        for name, path in result.debug.items()
        if Path(path).exists()
    ]
    return (
        str(result.glb_path),
        _tier_badge(
            result.measurements["measurement_tier"],
            result.measurements["estimated_error_pct"],
            result.warnings,
        ),
        _dimension_rows(result.measurements),
        str(result.measurements_path),
        str(result.glb_path),
        gallery,
        result.measurements,
    )


def build_ui(settings: Settings | None = None):
    import gradio as gr

    settings = settings or Settings.from_env()
    generators = available_generators()
    default_generator = (
        settings.generator if settings.generator in generators else generators[0]
    )

    def process(
        image,
        scale_source: str,
        marker_mm: float,
        aruco_dict: str,
        known_mm: float,
        known_axis: str,
        generator: str,
    ):
        return process_upload(
            image,
            scale_source,
            marker_mm,
            aruco_dict,
            known_mm,
            known_axis,
            generator,
            settings,
        )

    with gr.Blocks(title="Photo to 3D + measurements") as demo:
        gr.Markdown(_INTRO)

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="Photo", type="numpy", height=320)
                scale_source = gr.Radio(
                    choices=_SCALE_CHOICES,
                    value="auto",
                    label="Where should the size come from?",
                )
                with gr.Group(visible=True) as marker_group:
                    marker_mm = gr.Number(
                        value=DEFAULT_MARKER_MM,
                        label="Printed marker side length (mm)",
                        info="Measure the printed black square's edge to be sure.",
                    )
                    aruco_dict = gr.Dropdown(
                        choices=[
                            "DICT_4X4_50",
                            "DICT_5X5_50",
                            "DICT_6X6_250",
                            "DICT_APRILTAG_36h11",
                        ],
                        value=DEFAULT_ARUCO_DICT,
                        label="Marker dictionary",
                    )
                with gr.Group(visible=False) as manual_group:
                    known_mm = gr.Number(
                        value=None, label="Known dimension (mm)", precision=1
                    )
                    known_axis = gr.Dropdown(
                        choices=["height", "width"],
                        value="height",
                        label="Which dimension is that?",
                    )
                generator = gr.Dropdown(
                    choices=generators,
                    value=default_generator,
                    label="3D generator",
                    info="triposr is fast and low quality; hunyuan3d is slower and better.",
                )
                go = gr.Button("Generate 3D model", variant="primary")

            with gr.Column(scale=1):
                model_out = gr.Model3D(label="3D model", height=340)
                badge = gr.Markdown()
                table = gr.Dataframe(
                    headers=["Dimension", "Value", "Also"],
                    label="Measurements",
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    json_file = gr.File(label="measurements.json")
                    glb_file = gr.File(label="model.glb (metres)")

        with gr.Accordion("What the pipeline saw", open=False):
            gallery = gr.Gallery(label="Debug images", columns=4, height=220)
            raw_json = gr.JSON(label="Raw result")

        with gr.Accordion("How to get accurate numbers", open=False):
            gr.Markdown(_MARKER_HELP)

        def _toggle(source: str):
            return (
                gr.update(visible=source in {"marker", "auto"}),
                gr.update(visible=source in {"manual", "auto"}),
            )

        scale_source.change(
            _toggle, inputs=scale_source, outputs=[marker_group, manual_group]
        )
        go.click(
            process,
            inputs=[
                image_in,
                scale_source,
                marker_mm,
                aruco_dict,
                known_mm,
                known_axis,
                generator,
            ],
            outputs=[
                model_out,
                badge,
                table,
                json_file,
                glb_file,
                gallery,
                raw_json,
            ],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Photo to 3D + measurements UI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (needed when running in Colab).",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--generator", default=None, help="triposr, hunyuan3d, silhouette")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.generator:
        settings.generator = args.generator

    demo = build_ui(settings)
    demo.queue().launch(
        share=args.share, server_name=args.host, server_port=args.port, show_error=True
    )


if __name__ == "__main__":
    main()
