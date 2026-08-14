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

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..config import Settings
from ..errors import GeneratorError

_REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1"
_CKPT_NAME = "model.fp16.ckpt"
_MIN_CKPT_BYTES = 1_000_000_000
_COLAB_DRIVE_HY3DGEN = Path("/content/drive/MyDrive/p3d-cache/hy3dgen")


def resolve_hy3dgen_models_dir() -> Path:
    """Directory Hunyuan reads via HY3DGEN_MODELS.

    After a Colab runtime restart the env is cleared. If Drive is mounted we
    re-point at the persistent cache so a finished step-6b download is found.
    """
    raw = os.environ.get("HY3DGEN_MODELS")
    if raw:
        path = Path(os.path.expanduser(raw))
        path.mkdir(parents=True, exist_ok=True)
        return path
    if _COLAB_DRIVE_HY3DGEN.parent.is_dir():
        _COLAB_DRIVE_HY3DGEN.mkdir(parents=True, exist_ok=True)
        os.environ["HY3DGEN_MODELS"] = str(_COLAB_DRIVE_HY3DGEN)
        return _COLAB_DRIVE_HY3DGEN
    default = Path(os.path.expanduser("~/.cache/hy3dgen"))
    default.mkdir(parents=True, exist_ok=True)
    os.environ["HY3DGEN_MODELS"] = str(default)
    return default


def shape_checkpoint_path(settings: Settings | None = None) -> Path:
    settings = settings or Settings()
    base = resolve_hy3dgen_models_dir()
    return (
        base
        / settings.hunyuan_model
        / settings.hunyuan_shape_subfolder
        / _CKPT_NAME
    )


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

    def _ensure_shape_weights(self) -> Path:
        """Download ``model.fp16.ckpt`` (~7 GB) into HY3DGEN_MODELS if missing.

        Hugging Face does not ship ``model.fp16.safetensors`` for Hunyuan3D-2.1.
        We never ask for that name.
        """
        from huggingface_hub import snapshot_download

        ckpt = shape_checkpoint_path(self.settings)
        repo_dir = ckpt.parent.parent
        sub = self.settings.hunyuan_shape_subfolder
        if ckpt.is_file() and ckpt.stat().st_size > _MIN_CKPT_BYTES:
            return ckpt

        print(
            f"Downloading Hunyuan3D shape weights into {ckpt.parent} "
            "(~7 GB, first time only) ..."
        )
        snapshot_download(
            repo_id=self.settings.hunyuan_model,
            allow_patterns=[f"{sub}/*"],
            local_dir=str(repo_dir),
        )
        if not ckpt.is_file():
            found = sorted(p.name for p in ckpt.parent.glob("*")) if ckpt.parent.is_dir() else []
            raise GeneratorError(
                f"Download finished but {ckpt} is still missing.\n"
                f"Files present: {found}\n"
                "Check Hugging Face access / disk space, or switch to `triposr`."
            )
        if ckpt.stat().st_size < _MIN_CKPT_BYTES:
            raise GeneratorError(
                f"{ckpt} looks incomplete ({ckpt.stat().st_size} bytes). "
                "Delete it and re-run notebook step 6b."
            )
        return ckpt

    def _load_shape(self):
        if self._shape is not None:
            return self._shape

        self._ensure_repo_on_path()
        try:
            import torch
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as exc:
            raise GeneratorError(
                "Could not import Hunyuan3D's shape pipeline. Its custom CUDA "
                "extensions have to be compiled first - see the Colab notebook's "
                "install cell."
            ) from exc

        try:
            ckpt = self._ensure_shape_weights()
        except GeneratorError:
            raise
        except Exception as exc:
            raise GeneratorError(
                f"Could not download Hunyuan3D weights ({type(exc).__name__}: {exc}). "
                "Switch to `triposr`, or set HF_TOKEN if Hugging Face rate-limited you."
            ) from exc

        config_path = ckpt.parent / "config.yaml"
        if not config_path.is_file():
            raise GeneratorError(
                f"Hunyuan3D config missing next to the checkpoint:\n  {config_path}\n"
                "Re-run notebook step 6b (`python tools/download_hunyuan.py`)."
            )

        # Load the local .ckpt directly. Do NOT use from_pretrained here:
        # older Hunyuan defaults ask for model.fp16.safetensors, which HF never
        # ships for 2.1, and smart_load_model also ignores Drive if HY3DGEN_MODELS
        # was cleared by a Colab restart.
        #
        # On Colab T4, load onto CPU first then enable offload. Loading the ~7 GB
        # fp16 weights straight onto CUDA spikes VRAM and kills the runtime; Gradio
        # then only shows "Connection to the server was lost".
        import gc

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        load_device = "cpu" if self.settings.low_vram else self.settings.resolve_device()
        print(f"Loading Hunyuan3D shape weights from {ckpt} onto {load_device} ...")
        try:
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                str(ckpt),
                str(config_path),
                device=load_device,
                dtype=torch.float16,
                use_safetensors=False,
            )
        except FileNotFoundError as exc:
            raise GeneratorError(
                "Hunyuan3D could not find its shape weights.\n"
                "\n"
                f"Expected checkpoint:\n  {ckpt}\n"
                f"Exists: {ckpt.is_file()}  "
                f"HY3DGEN_MODELS={os.environ.get('HY3DGEN_MODELS')}\n"
                "\n"
                "Fix: re-run notebook step 2, then step 6b, wait for OK, then "
                "launch with GENERATOR = \"hunyuan3d\".\n"
                "Or switch the generator to `triposr` for now.\n"
                "\n"
                f"Underlying: {exc}"
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise GeneratorError(
                    "GPU/RAM ran out while loading Hunyuan3D. "
                    "Runtime > Disconnect and delete runtime, then re-run steps "
                    "2 → 5 → 7 with GENERATOR = \"triposr\". "
                    "Hunyuan needs a fresh T4 session with nothing else loaded."
                ) from exc
            raise GeneratorError(
                f"Hunyuan3D failed to load ({type(exc).__name__}: {exc}). "
                "Switch the generator to `triposr` to continue, or re-run "
                "notebook step 6b."
            ) from exc
        except Exception as exc:
            raise GeneratorError(
                f"Hunyuan3D failed to load ({type(exc).__name__}: {exc}). "
                "Switch the generator to `triposr` to continue, or re-run "
                "notebook step 6b."
            ) from exc
        if self.settings.low_vram:
            # Keeps one component on the GPU at a time. Slower, but the
            # difference between running and an out-of-memory crash on a T4.
            for method in ("enable_sequential_cpu_offload", "enable_model_cpu_offload"):
                if hasattr(pipeline, method):
                    try:
                        getattr(pipeline, method)()
                        print(f"Hunyuan3D low-VRAM: {method} enabled")
                        break
                    except Exception:
                        continue
        elif hasattr(pipeline, "to"):
            pipeline.to(self.settings.resolve_device())

        print("Hunyuan3D shape pipeline ready")
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
