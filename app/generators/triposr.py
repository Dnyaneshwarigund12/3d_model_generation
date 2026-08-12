"""TripoSR backend - the fast, low-VRAM generator used to get the pipeline running.

MIT licensed, roughly 4GB of VRAM, a second or two per image on a T4. Quality is
the lowest of the candidates in `01-research-notes.md`, so it is the starting
point rather than the destination: switch to `hunyuan3d` once the flow works.

TripoSR ships as a repo rather than a pip package, so the notebook clones it into
`third_party/TripoSR` and this adapter puts that directory on sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..config import Settings
from ..errors import GeneratorError
from ..imaging import composite_on_gray

_REPO_URL = "https://github.com/VAST-AI-Research/TripoSR"


class TripoSRGenerator:
    name = "triposr"
    is_placeholder = False

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._model = None
        self._device: str | None = None

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
        try:
            from tsr.system import TSR
        except ImportError as exc:
            raise GeneratorError(
                "Could not import TripoSR. Its extra dependencies (torchmcubes, "
                "omegaconf, einops) are installed by the Colab notebook."
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
        if mesh.faces is None or len(mesh.faces) == 0:
            raise GeneratorError(
                "TripoSR produced an empty mesh. This usually means the subject "
                "was cut out badly - check the cutout in the debug output."
            )
        return mesh
