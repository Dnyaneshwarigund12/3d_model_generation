"""Download Hunyuan3D-2.1 shape weights into the cache Hunyuan expects.

Hugging Face ships ``model.fp16.ckpt`` (~7 GB) under
``tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/``. There is no
``model.fp16.safetensors`` for this model. Hunyuan's loader reads from
``HY3DGEN_MODELS`` (default ``~/.cache/hy3dgen``; on Colab with Drive,
``/content/drive/MyDrive/p3d-cache/hy3dgen``).

    python tools/download_hunyuan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download

from app.generators.hunyuan3d import (  # noqa: E402
    _CKPT_NAME,
    _MIN_CKPT_BYTES,
    resolve_hy3dgen_models_dir,
    shape_checkpoint_path,
)

REPO = "tencent/Hunyuan3D-2.1"
SUB = "hunyuan3d-dit-v2-1"


def main() -> int:
    base = resolve_hy3dgen_models_dir()
    dest = base / REPO
    sub_dir = dest / SUB
    ckpt = shape_checkpoint_path()
    wrong = sub_dir / "model.fp16.safetensors"

    print(f"HY3DGEN_MODELS = {base}")
    print(f"Target checkpoint = {ckpt}")

    if ckpt.is_file() and ckpt.stat().st_size > _MIN_CKPT_BYTES:
        print(f"Already present ({ckpt.stat().st_size / 1e9:.2f} GB). Nothing to do.")
        return 0

    if wrong.exists() and not ckpt.exists():
        print(f"Removing leftover wrong filename: {wrong}")
        wrong.unlink()

    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading ~7 GB from Hugging Face (first time only). Leave this cell running...")
    snapshot_download(
        repo_id=REPO,
        allow_patterns=[f"{SUB}/*"],
        local_dir=str(dest),
    )

    if not ckpt.is_file():
        found = sorted(p.name for p in sub_dir.glob("*")) if sub_dir.is_dir() else []
        print(
            f"ERROR: expected {_CKPT_NAME} after download.\n"
            f"Files in {sub_dir}: {found}\n"
            "Check disk space / network. Optional: set HF_TOKEN for higher rate limits.",
            file=sys.stderr,
        )
        return 1

    size = ckpt.stat().st_size
    if size < _MIN_CKPT_BYTES:
        print(
            f"ERROR: {ckpt} is only {size} bytes — download looks incomplete. Delete it and re-run.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {ckpt} ({size / 1e9:.2f} GB)")
    print("Next: in step 7 set GENERATOR = \"hunyuan3d\" and launch the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
