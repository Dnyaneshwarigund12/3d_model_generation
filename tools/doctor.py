"""Check the environment and the pipeline before spending GPU time.

Run this as a script, never imported into a notebook kernel: a fresh process is
the only place where version numbers and imports reflect what is actually on
disk. A kernel reports whatever it imported first, and package metadata can
disagree with the files sitting next to it.

    python tools/doctor.py

Exit status is 0 only if every required check passed. Optional checks - the GPU
and the two generation backends - report as WARN, because the measurement
pipeline is verifiable without either.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Doctor:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.failed: list[str] = []
        self.warned: list[str] = []

    def check(self, label: str, function, *, required: bool = True) -> bool:
        try:
            detail = function() or ""
        except Exception as exc:
            status = FAIL if required else WARN
            (self.failed if required else self.warned).append(label)
            print(f"{status}  {label:34} {type(exc).__name__}: {exc}")
            if self.verbose:
                traceback.print_exc()
            return False
        print(f"{PASS}  {label:34} {detail}")
        return True

    def section(self, title: str) -> None:
        print(f"\n{title}\n{'-' * len(title)}")


def version_of(module_name: str):
    def check():
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")

    return check


def check_opencv_aruco():
    import cv2

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "this OpenCV build has no aruco module; install opencv-contrib-python"
        )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = (
        cv2.aruco.ArucoDetector(dictionary)
        if hasattr(cv2.aruco, "ArucoDetector")
        else None
    )
    api = "ArucoDetector" if detector else "detectMarkers (pre-4.7 API)"
    return f"{cv2.__version__}, {api}"


def check_rembg_session():
    from app.segment import _import_rembg

    new_session, _ = _import_rembg()
    new_session("u2net")
    return "u2net session created (weights cached)"


def marker_scene():
    """A 300x150 px object beside a 150 px marker, on one canvas."""
    import cv2
    import numpy as np

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = 150
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, 0, side)
    else:
        marker = cv2.aruco.drawMarker(dictionary, 0, side)

    image = np.full((500, 700, 3), 235, dtype=np.uint8)
    image[300 : 300 + side, 480 : 480 + side] = marker[:, :, None]

    mask = np.zeros((500, 700), dtype=bool)
    mask[100:250, 150:450] = True
    image[mask] = (60, 110, 180)
    return image, mask


def check_pipeline_end_to_end():
    """The whole chain on a scene whose true size is known exactly.

    A 150 px marker that is really 50 mm means 1/3 mm per pixel, so the
    300 x 150 px object is 100 x 50 mm. Uses the CPU placeholder generator and a
    supplied mask, so it is deterministic and needs no GPU and no weights.
    """
    import tempfile

    from app.config import Settings
    from app.pipeline import run
    from app.segment import segmentation_from_mask

    image, mask = marker_scene()
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings()
        settings.generator = "silhouette"
        settings.output_dir = Path(tmp) / "outputs"
        settings.subject_size = 256

        result = run(
            image,
            settings=settings,
            scale_source="marker",
            marker_mm=50.0,
            segmentation=segmentation_from_mask(image, mask),
        )
        length = result.measurements["length_mm"]
        width = result.measurements["width_mm"]
        if not (98.0 <= length <= 102.0 and 49.0 <= width <= 51.0):
            raise RuntimeError(
                f"expected about 100 x 50 mm, measured {length:.1f} x {width:.1f}"
            )
        if not result.glb_path.exists():
            raise RuntimeError("no GLB was written")
    return f"{length:.1f} x {width:.1f} mm on a true 100 x 50 mm object"


def check_rembg_on_an_image():
    """Does background removal actually run? Quality is not the question here."""
    from app.errors import SegmentationError
    from app.segment import remove_background

    image, _ = marker_scene()
    try:
        segmentation = remove_background(image)
    except SegmentationError as exc:
        # u2net is trained on photographs; a flat synthetic rectangle is a hard
        # case for it, and an empty mask here does not mean a broken install.
        raise RuntimeError(f"ran but found no subject ({exc})") from exc
    coverage = float(segmentation.mask.mean())
    return f"mask covers {coverage:.1%} of the frame"


def check_torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"torch {torch.__version__} sees no GPU - set Runtime > Change runtime "
            "type > T4 GPU"
        )
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"torch {torch.__version__}, {name}, {total:.1f} GB"


def check_triposr():
    from app.config import Settings
    from app.generators.triposr import _ensure_marching_cubes

    repo = Settings().triposr_repo
    if not (repo / "tsr").is_dir():
        raise RuntimeError(f"not cloned at {repo}; run tools/colab_setup.py")
    sys.path.insert(0, str(repo))
    source = _ensure_marching_cubes()
    from tsr.system import TSR  # noqa: F401

    return f"imports cleanly, marching cubes from {source}"


def check_unidepth():
    """Only needed for the 'estimate' scale source; a marker beats it comfortably."""
    from unidepth.models import UniDepthV2  # noqa: F401

    return "available, so the 'estimate' scale source works"


def check_hunyuan3d():
    from app.config import Settings

    repo = Settings().hunyuan_repo
    if not (repo / "hy3dshape").is_dir():
        raise RuntimeError(f"not cloned at {repo}; run tools/colab_setup.py")
    for sub in ("hy3dshape", "hy3dpaint"):
        path = repo / sub
        if path.is_dir():
            sys.path.insert(0, str(path))
    sys.path.insert(0, str(repo))
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401

    return "shape pipeline imports cleanly"


def main() -> int:
    verbose = "--verbose" in sys.argv
    doctor = Doctor(verbose=verbose)

    print(f"python {sys.version.split()[0]} at {sys.executable}")

    doctor.section("Libraries the pipeline needs")
    for module in ("numpy", "PIL", "scipy", "trimesh", "onnxruntime", "gradio"):
        doctor.check(module, version_of(module))
    doctor.check("cv2 + aruco", check_opencv_aruco)
    doctor.check("rembg", check_rembg_session)

    doctor.section("The pipeline itself")
    doctor.check("scale, mesh, measure, export", check_pipeline_end_to_end)
    doctor.check("background removal runs", check_rembg_on_an_image, required=False)

    doctor.section("GPU and generation backends (optional)")
    doctor.check("torch sees a GPU", check_torch_cuda, required=False)
    doctor.check("triposr backend", check_triposr, required=False)
    doctor.check("hunyuan3d backend", check_hunyuan3d, required=False)
    doctor.check("unidepth (no-marker scale)", check_unidepth, required=False)

    print()
    if doctor.failed:
        print(f"{len(doctor.failed)} required check(s) failed: {', '.join(doctor.failed)}")
        print("Run tools/colab_setup.py, restart the runtime, then run this again.")
        print("Re-run with --verbose for full tracebacks.")
        return 1

    if doctor.warned:
        print(f"Everything required passed. Optional, unavailable: {', '.join(doctor.warned)}")
        print("You can still measure objects; pick a backend that passed above.")
    else:
        print("Everything passed. Launch the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
