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
