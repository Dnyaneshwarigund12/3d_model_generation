"""Hunyuan3D 2.1 backend - the quality path.

Chosen over TRELLIS-2 for the free-Colab T4 specifically. The T4 is a Turing card
with no bf16 and no flash-attention, which is exactly what TRELLIS-2 loads by
default; getting it running there means converting weights to fp16, swapping the
attention backend and patching dtype mismatches. Hunyuan3D 2.1 instead ships a
supported low-VRAM mode that offloads to CPU between stages.

VRAM, from the model's own documentation:

    shape only          10 GB
    texture only        21 GB
    both, no offload    29 GB
    low_vram_mode      ~15 GB, sequentially, 35-50 s

A free T4 has about 15 GB, so shape generation is comfortable and texture
generation is marginal - hence `hunyuan_texture` defaults to off. Turn it on only
on a bigger card.

The repo needs two custom CUDA/C++ extensions compiled before it will import; the
Colab notebook does that. If the API here has drifted since, this adapter is the
only file to touch.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ..config import Settings
from ..errors import GeneratorError

_REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1"


class Hunyuan3DGenerator:
    name = "hunyuan3d"
    is_placeholder = False

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._shape = None
        self._paint = None
        self.last_warnings: list[str] = []

    def _ensure_repo_on_path(self) -> Path:
        repo = self.settings.hunyuan_repo
        if not (repo / "hy3dshape").is_dir():
            raise GeneratorError(
                f"Hunyuan3D 2.1 source not found at {repo}. Clone it first:\n"
                f"  git clone {_REPO_URL} {repo}"
            )
        for sub in ("hy3dshape", "hy3dpaint"):
            path = repo / sub
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        return repo

    def _load_shape(self):
        if self._shape is not None:
            return self._shape

        self._ensure_repo_on_path()
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as exc:
            raise GeneratorError(
                "Could not import Hunyuan3D's shape pipeline. Its custom CUDA "
                "extensions have to be compiled first - see the Colab notebook's "
                "install cell."
            ) from exc

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.settings.hunyuan_model,
            subfolder=self.settings.hunyuan_shape_subfolder,
            use_safetensors=True,
        )
        if self.settings.low_vram:
            # Keeps one component on the GPU at a time. Slower, but the
            # difference between running and an out-of-memory crash on a T4.
            for method in ("enable_sequential_cpu_offload", "enable_model_cpu_offload"):
                if hasattr(pipeline, method):
                    try:
                        getattr(pipeline, method)()
                        break
                    except Exception:
                        continue
        elif hasattr(pipeline, "to"):
            pipeline.to(self.settings.resolve_device())

        self._shape = pipeline
        return pipeline

    def _clean(self, mesh):
        """Drop floating fragments and degenerate faces, then decimate.

        Generative meshes routinely come with detached specks, which would
        inflate the bounding box and therefore the reported dimensions.
        """
        try:
            from hy3dshape.postprocessors import (
                DegenerateFaceRemover,
                FaceReducer,
                FloaterRemover,
            )
        except Exception:
            self.last_warnings.append(
                "Hunyuan3D post-processors unavailable; mesh returned unclean."
            )
            return mesh

        for step in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
            try:
                mesh = step(mesh)
            except Exception as exc:
                self.last_warnings.append(f"{type(step).__name__} failed: {exc}")
        return mesh

    def _texture(self, mesh, image_rgba: np.ndarray):
        """Optional PBR texture pass. Needs ~21GB of VRAM on its own."""
        from PIL import Image

        self._ensure_repo_on_path()
        try:
            from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
        except ImportError as exc:
            raise GeneratorError(f"Texture pipeline unavailable: {exc}") from exc

        if self._paint is None:
            config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
            self._paint = Hunyuan3DPaintPipeline(config)

        # The 2.1 paint pipeline works on files rather than objects.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            mesh_path = tmp_dir / "shape.obj"
            image_path = tmp_dir / "image.png"
            mesh.export(str(mesh_path))
            Image.fromarray(image_rgba).save(image_path)

            out_path = self._paint(
                mesh_path=str(mesh_path),
                image_path=str(image_path),
                output_mesh_path=str(tmp_dir / "textured.glb"),
            )
            import trimesh

            loaded = trimesh.load(str(out_path), force="mesh")
        return loaded

    def generate(self, image: np.ndarray):
        from PIL import Image

        self.last_warnings = []
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise GeneratorError(
                "Hunyuan3D needs an RGBA cutout - with an opaque background it "
                "reconstructs the background too."
            )

        pipeline = self._load_shape()
        pil = Image.fromarray(arr, mode="RGBA")

        try:
            output = pipeline(
                image=pil,
                num_inference_steps=self.settings.hunyuan_steps,
                octree_resolution=self.settings.hunyuan_octree_resolution,
            )
        except TypeError:
            output = pipeline(image=pil)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise GeneratorError(
                    "The GPU ran out of memory. Lower octree_resolution, keep "
                    "hunyuan_texture off, or use the triposr backend."
                ) from exc
            raise

        mesh = output[0] if isinstance(output, (list, tuple)) else output
        if mesh is None or len(getattr(mesh, "faces", [])) == 0:
            raise GeneratorError("Hunyuan3D returned an empty mesh.")
        mesh = self._clean(mesh)

        if self.settings.hunyuan_texture:
            try:
                mesh = self._texture(mesh, arr)
            except Exception as exc:
                self.last_warnings.append(
                    f"Texture pass failed, returning untextured shape: {exc}"
                )
        return mesh
