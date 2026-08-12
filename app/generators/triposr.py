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


def _remap_triposr_vit_keys(state_dict: dict) -> dict:
    """Map transformers 4.x ViT keys onto transformers 5.x names.

    TripoSR's published checkpoint and its own requirements pin
    ``transformers==4.35.0``, where ViT weights live under
    ``encoder.layer.*.attention.attention.query``. Gradio 6 (and therefore this
    Colab stack) pulls transformers 5.x, which renamed those to
    ``layers.*.attention.q_proj``. Without the remap, ``load_state_dict`` fails
    with a wall of Missing/Unexpected keys - exactly the step-8 traceback.

    We cannot pin transformers 4.35.0: it needs ``huggingface-hub<1``, which
    Gradio 6 refuses. Remapping the handful of ViT keys is the compatible fix.
    """
    # Already on the new naming, or not a TripoSR checkpoint.
    if not any(
        key.startswith("image_tokenizer.model.encoder.layer.") for key in state_dict
    ):
        return state_dict

    remapped: dict = {}
    for key, value in state_dict.items():
        if not key.startswith("image_tokenizer.model."):
            remapped[key] = value
            continue
        rest = key[len("image_tokenizer.model.") :]
        rest = rest.replace("encoder.layer.", "layers.", 1)
        rest = rest.replace("attention.attention.query", "attention.q_proj")
        rest = rest.replace("attention.attention.key", "attention.k_proj")
        rest = rest.replace("attention.attention.value", "attention.v_proj")
        # Attention output must be rewritten before the MLP's output.dense.
        rest = rest.replace("attention.output.dense", "attention.o_proj")
        rest = rest.replace("intermediate.dense", "mlp.fc1")
        rest = rest.replace("output.dense", "mlp.fc2")
        remapped["image_tokenizer.model." + rest] = value
    return remapped


def _load_triposr_checkpoint(model_id: str):
    """Load TripoSR the way its repo does, plus the transformers-5 key remap."""
    import os

    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from tsr.system import TSR

    if os.path.isdir(model_id):
        config_path = os.path.join(model_id, "config.yaml")
        weight_path = os.path.join(model_id, "model.ckpt")
    else:
        config_path = hf_hub_download(repo_id=model_id, filename="config.yaml")
        weight_path = hf_hub_download(repo_id=model_id, filename="model.ckpt")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = TSR(cfg)

    # torch>=2.4 defaults weights_only=True, which rejects OmegaConf pickles in
    # some side paths; TripoSR's ckpt is a plain state_dict, but be explicit.
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")
    except Exception:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise GeneratorError(
            f"TripoSR checkpoint at {weight_path} is not a state dict."
        )

    ckpt = _remap_triposr_vit_keys(ckpt)
    try:
        model.load_state_dict(ckpt, strict=True)
    except RuntimeError as exc:
        raise GeneratorError(
            "TripoSR's weights do not match the installed transformers package. "
            "This project remaps the ViT keys for transformers 5.x automatically; "
            "if you still see this, clear the Hugging Face cache and retry, or "
            "switch the generator to 'hunyuan3d' / 'silhouette'.\n\n"
            f"Underlying error: {exc}"
        ) from exc
    return model


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
        try:
            model = _load_triposr_checkpoint(self.settings.triposr_model)
        except GeneratorError:
            raise
        except Exception as exc:
            raise GeneratorError(
                f"Failed to load TripoSR ({type(exc).__name__}: {exc}). "
                "Check the network / Hugging Face cache, or switch generator to "
                "'silhouette' to confirm the rest of the pipeline."
            ) from exc
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
