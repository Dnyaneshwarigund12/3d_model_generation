"""Configuration and physical constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

# ISO/IEC 7810 ID-1, the standard bank-card size: long side, short side.
CREDIT_CARD_MM: tuple[float, float] = (85.60, 53.98)
CREDIT_CARD_ASPECT = CREDIT_CARD_MM[0] / CREDIT_CARD_MM[1]

DEFAULT_MARKER_MM = 50.0
DEFAULT_ARUCO_DICT = "DICT_4X4_50"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime knobs. Every field is overridable by environment variable so the
    Colab notebook can configure the app without editing code."""

    generator: str = "triposr"
    device: str = "auto"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    third_party_dir: Path = DEFAULT_THIRD_PARTY_DIR

    # Segmentation
    rembg_model: str = "u2net"
    alpha_threshold: int = 127
    subject_padding: float = 0.06
    subject_size: int = 768

    # TripoSR
    triposr_model: str = "stabilityai/TripoSR"
    mc_resolution: int = 256

    # Hunyuan3D 2.1
    hunyuan_model: str = "tencent/Hunyuan3D-2.1"
    hunyuan_shape_subfolder: str = "hunyuan3d-dit-v2-1"
    hunyuan_steps: int = 30
    hunyuan_octree_resolution: int = 256
    # Texture generation needs ~21GB VRAM on its own; off by default because the
    # free Colab T4 has ~15GB.
    hunyuan_texture: bool = False
    low_vram: bool = True

    # Monocular metric depth (Tier 3)
    depth_model: str = "unidepth"

    # Cosmetic only: rotate generator output so it stands upright in viewers.
    # Measurements are unaffected - they come from the oriented bounding box,
    # which is rotation invariant.
    upright_output: bool = True
    save_debug: bool = True

    @property
    def triposr_repo(self) -> Path:
        return self.third_party_dir / "TripoSR"

    @property
    def hunyuan_repo(self) -> Path:
        return self.third_party_dir / "Hunyuan3D-2.1"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.generator = os.environ.get("P3D_GENERATOR", s.generator)
        s.device = os.environ.get("P3D_DEVICE", s.device)
        s.output_dir = Path(os.environ.get("P3D_OUTPUT_DIR", str(s.output_dir)))
        s.third_party_dir = Path(
            os.environ.get("P3D_THIRD_PARTY_DIR", str(s.third_party_dir))
        )
        s.rembg_model = os.environ.get("P3D_REMBG_MODEL", s.rembg_model)
        s.mc_resolution = int(os.environ.get("P3D_MC_RESOLUTION", s.mc_resolution))
        s.subject_size = int(os.environ.get("P3D_SUBJECT_SIZE", s.subject_size))
        s.hunyuan_texture = _env_bool("P3D_HUNYUAN_TEXTURE", s.hunyuan_texture)
        s.low_vram = _env_bool("P3D_LOW_VRAM", s.low_vram)
        s.depth_model = os.environ.get("P3D_DEPTH_MODEL", s.depth_model)
        s.upright_output = _env_bool("P3D_UPRIGHT", s.upright_output)
        s.save_debug = _env_bool("P3D_SAVE_DEBUG", s.save_debug)
        return s

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
