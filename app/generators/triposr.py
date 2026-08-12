"""TripoSR backend - the fast, low-VRAM generator used to get the pipeline running.

MIT licensed, roughly 4GB of VRAM, a second or two per image on a T4. Quality is
the lowest of the candidates in `01-research-notes.md`, so it is the starting
point rather than the destination: switch to `hunyuan3d` once the flow works.

TripoSR ships as a repo rather than a pip package, so the notebook clones it into
`third_party/TripoSR` and this adapter puts that directory on sys.path.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from ..config import Settings
from ..errors import GeneratorError
from ..imaging import composite_on_gray

_REPO_URL = "https://github.com/VAST-AI-Research/TripoSR"


def _ensure_marching_cubes() -> str:
    """Make `import torchmcubes` succeed, with a CPU implementation if it cannot.

    TripoSR imports torchmcubes, a CUDA extension that compiles from source and
    routinely fails to build on Colab. scikit-image's marching cubes produces the
    same surface, more slowly. Its axis order may differ, which rotates the model
    in the viewer but cannot change the measurements: those come from the oriented
    bounding box, which is invariant under rotation.

    Returns the name of whichever implementation is in play, for the doctor script.
    """
    try:
        import torchmcubes  # noqa: F401

        return "torchmcubes (CUDA)"
    except Exception:
        pass

    try:
        from skimage.measure import marching_cubes as skimage_marching_cubes
    except Exception as exc:
        raise GeneratorError(
            "TripoSR needs marching cubes, and neither torchmcubes nor "
            "scikit-image is importable. Install scikit-image, or use the "
            "hunyuan3d backend, which does not need either."
        ) from exc

    def marching_cubes(volume, isolevel: float = 0.0):
        import torch

        array = (
            volume.detach().cpu().numpy()
            if hasattr(volume, "detach")
            else np.asarray(volume)
        )
        vertices, faces, _, _ = skimage_marching_cubes(array, level=float(isolevel))
        return (
            torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int64)),
        )

    def grid_interp(*_args, **_kwargs):
        raise GeneratorError(
            "torchmcubes.grid_interp has no CPU stand-in. Build torchmcubes, or "
            "use the hunyuan3d backend."
        )

    module = types.ModuleType("torchmcubes")
    module.marching_cubes = marching_cubes
    module.grid_interp = grid_interp
    sys.modules["torchmcubes"] = module
    return "scikit-image (CPU fallback)"


class TripoSRGenerator:
    name = "triposr"
    is_placeholder = False

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model = None
        self._device: str | None = None
        self._marching_cubes: str | None = None

    def _ensure_repo_on_path(self) -> Path:
        repo = self.settings.triposr_repo
        if not (repo / "tsr").is_dir():
            raise GeneratorError(
                f"TripoSR source not found at {repo}. Clone it first:\n"
                f"  git clone {_REPO_URL} {repo}"
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        return repo

    def _load(self):
        if self._model is not None:
            return self._model

        import torch

        self._ensure_repo_on_path()
        self._marching_cubes = _ensure_marching_cubes()
        try:
            from tsr.system import TSR
        except ImportError as exc:
            raise GeneratorError(
                "Could not import TripoSR. Its extra dependencies (omegaconf, "
                "einops, jaxtyping) are installed by tools/colab_setup.py."
            ) from exc

        device = self.settings.resolve_device()
        model = TSR.from_pretrained(
            self.settings.triposr_model,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        # Bounds the renderer's memory use; the value comes from TripoSR's own
        # reference script.
        model.renderer.set_chunk_size(8192)
        model.to(device)
        if device == "cuda":
            torch.cuda.empty_cache()

        self._model = model
        self._device = device
        return model

    def _extract_mesh(self, model, scene_codes):
        resolution = self.settings.mc_resolution
        try:
            return model.extract_mesh(scene_codes, True, resolution=resolution)
        except TypeError:
            # Older checkpoints of the repo lack the vertex-colour argument.
            return model.extract_mesh(scene_codes, resolution=resolution)

    def generate(self, image: np.ndarray):
        import torch
        from PIL import Image

        model = self._load()
        # TripoSR was trained on cutouts flattened onto mid-grey; a transparent
        # or white background visibly changes its output.
        rgb = composite_on_gray(np.asarray(image), value=0.5)
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            scene_codes = model([pil], device=self._device)
            meshes = self._extract_mesh(model, scene_codes)

        if not meshes:
            raise GeneratorError("TripoSR returned no mesh for this image.")
        mesh = meshes[0]
        if self._marching_cubes and "fallback" in self._marching_cubes:
            # The fallback's axis order can invert face winding, which shows up as
            # an inside-out model in the viewer. Measurements are unaffected.
            try:
                mesh.fix_normals()
            except Exception:
                pass
        if mesh.faces is None or len(mesh.faces) == 0:
            raise GeneratorError(
                "TripoSR produced an empty mesh. This usually means the subject "
                "was cut out badly - check the cutout in the debug output."
            )
        return mesh
