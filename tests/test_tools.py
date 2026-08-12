"""Tests for the Colab setup and the TripoSR marching-cubes fallback.

Neither needs a GPU. The fallback is checked with a stub torch module so its
geometry can be verified on a machine that has no torch at all.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import colab_setup  # noqa: E402


@pytest.fixture
def fake_torch(monkeypatch):
    """Just enough torch for the fallback: from_numpy returning the array."""
    module = types.ModuleType("torch")
    module.from_numpy = lambda array: array
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


@pytest.fixture
def no_torchmcubes(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchmcubes", None)
    yield
    sys.modules.pop("torchmcubes", None)


def test_freeze_baseline_pins_only_what_is_installed(tmp_path, monkeypatch):
    present = {"numpy": "2.5.2", "pillow": "12.3.0"}
    monkeypatch.setattr(colab_setup, "installed", lambda name: present.get(name))

    path = tmp_path / "constraints.txt"
    pins = colab_setup.freeze_baseline(path)

    assert pins == present
    body = path.read_text(encoding="utf-8")
    assert "numpy==2.5.2" in body
    assert "pillow==12.3.0" in body
    # Absent packages must not appear, or pip would go and install them.
    assert "torch" not in body.replace("# ", "")


def test_freeze_baseline_writes_an_explanation(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_setup, "installed", lambda name: "1.0" if name == "numpy" else None)
    path = tmp_path / "constraints.txt"
    colab_setup.freeze_baseline(path)
    assert path.read_text(encoding="utf-8").startswith("#")


def test_backend_choices_cover_the_repos():
    assert set(colab_setup.REPOS) == set(colab_setup.BACKEND_PACKAGES)


def fake_clone(monkeypatch, returncode: int, *, writes: Path | None = None):
    """Stand in for git, optionally creating what a real clone would have."""

    def run(command, **kwargs):
        if writes is not None:
            writes.mkdir(parents=True, exist_ok=True)
        assert "--depth" in command, "the history is never read; clone shallow"
        return types.SimpleNamespace(
            returncode=returncode, stdout="", stderr="error: Filename too long"
        )

    monkeypatch.setattr(colab_setup.subprocess, "run", run)


def test_clone_skips_a_repo_that_is_already_there(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("git was called for a repo already present")

    monkeypatch.setattr(colab_setup.subprocess, "run", refuse)
    (tmp_path / "repo" / "tsr").mkdir(parents=True)

    assert colab_setup.clone("url", tmp_path / "repo", "tsr") == "already present"


def test_clone_accepts_a_checkout_that_skipped_files(tmp_path, monkeypatch):
    """Hunyuan3D ships paths longer than Windows allows; the package still arrives."""
    destination = tmp_path / "repo"
    fake_clone(monkeypatch, 128, writes=destination / "hy3dshape")

    state = colab_setup.clone("url", destination, "hy3dshape")

    assert "cloned" in state and "could not be written" in state


def test_clone_failure_explains_itself_instead_of_raising_a_traceback(
    tmp_path, monkeypatch
):
    destination = tmp_path / "repo"
    fake_clone(monkeypatch, 128)  # nothing written: a genuine failure

    with pytest.raises(RuntimeError) as failure:
        colab_setup.clone("https://example.invalid/repo", destination, "hy3dshape")

    message = str(failure.value)
    assert "Filename too long" in message  # git's own words, not a summary
    assert str(destination) in message  # and what to delete before retrying


def test_pymeshlab_is_installed_for_hunyuan():
    """hy3dshape imports it, and its absence is not obvious from the traceback."""
    assert "pymeshlab" in colab_setup.BACKEND_PACKAGES["hunyuan3d"]


def test_marching_cubes_fallback_registers_a_module(no_torchmcubes, fake_torch):
    from app.generators.triposr import _ensure_marching_cubes

    source = _ensure_marching_cubes()

    assert "fallback" in source
    assert callable(sys.modules["torchmcubes"].marching_cubes)


def test_marching_cubes_fallback_returns_vertices_in_index_space(
    no_torchmcubes, fake_torch
):
    """TripoSR divides the vertices by resolution, so they must be voxel indices."""
    from app.generators.triposr import _ensure_marching_cubes

    _ensure_marching_cubes()
    marching_cubes = sys.modules["torchmcubes"].marching_cubes

    resolution = 32
    grid = np.indices((resolution,) * 3).astype(np.float32)
    centre = (resolution - 1) / 2.0
    radius = np.sqrt(((grid - centre) ** 2).sum(axis=0))
    # Positive inside a sphere of radius 10, negative outside.
    volume = 10.0 - radius

    vertices, faces = marching_cubes(volume, 0.0)

    assert len(vertices) > 0 and len(faces) > 0
    assert vertices.min() >= 0.0
    assert vertices.max() <= resolution - 1
    # The surface of a radius-10 sphere, in voxel units, about the centre.
    distances = np.sqrt(((vertices - centre) ** 2).sum(axis=1))
    assert distances.mean() == pytest.approx(10.0, abs=0.5)


def test_marching_cubes_prefers_the_cuda_extension(monkeypatch):
    stub = types.ModuleType("torchmcubes")
    stub.marching_cubes = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "torchmcubes", stub)

    from app.generators.triposr import _ensure_marching_cubes

    assert _ensure_marching_cubes() == "torchmcubes (CUDA)"


@pytest.mark.parametrize("photographed_side_px", [120, 150, 240])
def test_the_printable_sheet_measures_back_through_the_pipeline(
    tmp_path, photographed_side_px
):
    """The whole chain, on the real printable asset rather than a bare marker.

    The sheet carries ruler ticks and a caption. When those sat too close to the
    marker the detector traced them as part of it, and every measurement came out 3%
    large - a bias invisible to any test that pastes a bare ArUco square.
    """
    from PIL import Image

    from app.config import Settings
    from app.pipeline import run
    from app.segment import segmentation_from_mask
    from tools.make_marker import MM_PER_INCH, render_marker

    object_px = (300, 150)
    marker_mm = 50.0

    # The sheet as printed, then shrunk to the size it would occupy in a photograph.
    sheet = render_marker(marker_mm)
    factor = photographed_side_px / (marker_mm / MM_PER_INCH * 300)
    patch = np.asarray(
        sheet.resize(
            (round(sheet.width * factor), round(sheet.height * factor)), Image.LANCZOS
        ).convert("RGB"),
        dtype=np.uint8,
    )

    # Canvas sized to hold the whole sheet: a cropped marker is not detectable, which
    # would make this pass or fail for the wrong reason.
    left = 60 + object_px[0] + 40
    height = max(80 + object_px[1] + 40, 40 + patch.shape[0] + 40)
    image = np.full((height, left + patch.shape[1] + 40, 3), 232, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[80 : 80 + object_px[1], 60 : 60 + object_px[0]] = True
    image[mask] = (60, 110, 180)
    image[40 : 40 + patch.shape[0], left : left + patch.shape[1]] = patch

    settings = Settings()
    settings.generator = "silhouette"
    settings.output_dir = tmp_path / "outputs"
    result = run(
        image,
        settings=settings,
        scale_source="marker",
        marker_mm=marker_mm,
        segmentation=segmentation_from_mask(image, mask),
    )

    mm_per_px = marker_mm / photographed_side_px
    assert result.measurements["length_mm"] == pytest.approx(
        object_px[0] * mm_per_px, rel=0.03
    )
    assert result.measurements["width_mm"] == pytest.approx(
        object_px[1] * mm_per_px, rel=0.03
    )
