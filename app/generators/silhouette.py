"""CPU-only placeholder generator: extrudes the subject's silhouette into a slab.

This is deliberately not a reconstruction - it invents no depth at all. It exists
so the rest of the pipeline (scale solving, measurement, export, UI) can be
exercised and tested on a machine with no GPU, and so that a broken generation
backend can be told apart from a broken pipeline. The two real backends are
`triposr` and `hunyuan3d`.

The mesh is built by emitting only the exposed faces of the filled cells of a
downsampled mask, which keeps it watertight and low-poly without needing a
polygon triangulation library.
"""

from __future__ import annotations

import numpy as np

from ..config import Settings
from ..errors import GeneratorError

# Slab depth as a fraction of the silhouette's shorter side.
_DEPTH_RATIO = 0.4
_MAX_GRID = 64


def _extrude_mask(mask: np.ndarray, depth_cells: int):
    """Build a watertight slab from a boolean mask, one cube per filled cell.

    Cells share a single vertex lattice, so every interior edge is used by
    exactly two faces and the result closes up.
    """
    import trimesh

    height, width = mask.shape
    stride = width + 1

    def vertex_index(i: int, j: int, k: int) -> int:
        return k * (stride * (height + 1)) + j * stride + i

    xs, ys, zs = np.meshgrid(
        np.arange(width + 1, dtype=np.float64),
        np.arange(height + 1, dtype=np.float64),
        np.array([0.0, float(depth_cells)]),
        indexing="xy",
    )
    # Image rows run top-down; flip so +y is up in the mesh.
    vertices = np.stack(
        [
            xs.transpose(2, 0, 1).reshape(-1),
            (height - ys).transpose(2, 0, 1).reshape(-1),
            zs.transpose(2, 0, 1).reshape(-1),
        ],
        axis=1,
    )

    filled = np.argwhere(mask)
    faces: list[tuple[int, int, int]] = []
    for j, i in filled:
        j, i = int(j), int(i)
        b00, b10 = vertex_index(i, j, 0), vertex_index(i + 1, j, 0)
        b11, b01 = vertex_index(i + 1, j + 1, 0), vertex_index(i, j + 1, 0)
        f00, f10 = vertex_index(i, j, 1), vertex_index(i + 1, j, 1)
        f11, f01 = vertex_index(i + 1, j + 1, 1), vertex_index(i, j + 1, 1)

        faces.append((f01, f11, f10))
        faces.append((f01, f10, f00))
        faces.append((b00, b10, b11))
        faces.append((b00, b11, b01))

        if i == 0 or not mask[j, i - 1]:
            faces.append((b00, b01, f01))
            faces.append((b00, f01, f00))
        if i == width - 1 or not mask[j, i + 1]:
            faces.append((b10, f10, f11))
            faces.append((b10, f11, b11))
        if j == 0 or not mask[j - 1, i]:
            faces.append((b00, f00, f10))
            faces.append((b00, f10, b10))
        if j == height - 1 or not mask[j + 1, i]:
            faces.append((b01, b11, f11))
            faces.append((b01, f11, f01))

    if not faces:
        raise GeneratorError("The silhouette has no filled cells.")

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


class SilhouetteGenerator:
    name = "silhouette"
    is_placeholder = True

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def generate(self, image: np.ndarray):
        import cv2

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise GeneratorError("The silhouette generator needs an RGBA cutout.")

        mask = arr[:, :, 3] > 127
        if not mask.any():
            raise GeneratorError("The cutout is empty.")

        ys, xs = np.nonzero(mask)
        cropped = mask[
            int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1
        ]

        h, w = cropped.shape
        ratio = _MAX_GRID / float(max(h, w))
        grid_w = max(2, int(round(w * ratio)))
        grid_h = max(2, int(round(h * ratio)))
        small = (
            cv2.resize(
                cropped.astype(np.float32),
                (grid_w, grid_h),
                interpolation=cv2.INTER_AREA,
            )
            >= 0.5
        )
        if not small.any():
            small = np.ones((grid_h, grid_w), dtype=bool)

        depth_cells = max(1, int(round(min(grid_w, grid_h) * _DEPTH_RATIO)))
        mesh = _extrude_mask(small, depth_cells)

        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        longest = float(mesh.extents.max())
        if longest > 0:
            mesh.apply_scale(1.0 / longest)
        return mesh
