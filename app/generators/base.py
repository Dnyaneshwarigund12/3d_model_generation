"""The seam between the pipeline and whichever image-to-3D model is in use.

Every backend takes one RGBA cutout and returns a `trimesh.Trimesh` in
arbitrary units. Nothing downstream knows or cares which model produced it:
real-world scale is solved separately and applied afterwards, per
`04-measurement-methodology.md` section 4.
"""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import Settings
from ..errors import GeneratorError


@runtime_checkable
class MeshGenerator(Protocol):
    name: str

    def generate(self, image: np.ndarray) -> "object":
        """Turn an RGBA cutout into a trimesh.Trimesh in arbitrary units."""


# name -> (module, class) so importing this module never pulls in torch.
_BACKENDS: dict[str, tuple[str, str]] = {
    "silhouette": ("app.generators.silhouette", "SilhouetteGenerator"),
    "triposr": ("app.generators.triposr", "TripoSRGenerator"),
    "hunyuan3d": ("app.generators.hunyuan3d", "Hunyuan3DGenerator"),
}

_CACHE: dict[tuple[str, int], MeshGenerator] = {}


def available() -> list[str]:
    return list(_BACKENDS)


def get_generator(name: str, settings: Settings | None = None) -> MeshGenerator:
    """Instantiate a backend by name, caching it so weights load only once.

    Reloading multi-gigabyte weights per request is the single biggest avoidable
    latency cost in this pipeline (`02-architecture.md` section 6).
    """
    settings = settings or Settings()
    key = (name, id(settings))
    if key in _CACHE:
        return _CACHE[key]

    if name not in _BACKENDS:
        raise GeneratorError(
            f"Unknown generator {name!r}. Available: {', '.join(available())}"
        )
    module_name, class_name = _BACKENDS[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise GeneratorError(
            f"Generator {name!r} could not be imported: {exc}. "
            "Its dependencies are installed by the Colab notebook."
        ) from exc
    generator = getattr(module, class_name)(settings)
    _CACHE[key] = generator
    return generator


def upright_transform() -> np.ndarray:
    """Rotate a generator's canonical output so it stands upright in viewers.

    Purely cosmetic. The reported dimensions come from the oriented bounding
    box, which is invariant to rotation, so this cannot change any measurement.
    """
    import trimesh

    rx = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    ry = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    return ry @ rx


def register(name: str, module: str, class_name: str) -> None:
    _BACKENDS[name] = (module, class_name)


def clear_cache() -> None:
    _CACHE.clear()


__all__ = [
    "MeshGenerator",
    "available",
    "get_generator",
    "register",
    "clear_cache",
    "upright_transform",
]
