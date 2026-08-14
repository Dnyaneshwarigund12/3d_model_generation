"""Download Hunyuan3D-2.1 shape weights into the cache Hunyuan expects.

Hugging Face ships ``model.fp16.ckpt`` (~7 GB) under
``tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/``. Hunyuan's loader reads from
``~/.cache/hy3dgen/tencent/Hunyuan3D-2.1/...`` (or ``HY3DGEN_MODELS``).

    python tools/download_hunyuan.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "tencent/Hunyuan3D-2.1"
SUB = "hunyuan3d-dit-v2-1"


def main() -> int:
    base = Path(os.path.expanduser(os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")))
    dest = base / REPO
    ckpt = dest / SUB / "model.fp16.ckpt"

    print(f"Target: {ckpt}")
    if ckpt.is_file() and ckpt.stat().st_size > 1_000_000_000:
        print(f"Already present ({ckpt.stat().st_size / 1e9:.1f} GB). Nothing to do.")
        return 0

    # Clear a broken partial that only has the wrong filename leftover.
    wrong = dest / SUB / "model.fp16.safetensors"
    if wrong.exists() and not ckpt.exists():
        print(f"Removing leftover wrong file: {wrong}")
        wrong.unlink()

    print("Downloading ~7 GB (first time). Leave this running...")
    snapshot_download(repo_id=REPO, allow_patterns=[f"{SUB}/*"], local_dir=str(dest))

    if not ckpt.is_file():
        print(
            f"ERROR: expected {ckpt} after download.\n"
            "Check disk space and Hugging Face access. Optional: export HF_TOKEN=...",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {ckpt} ({ckpt.stat().st_size / 1e9:.1f} GB)")
    print("Now launch the app and select generator = hunyuan3d.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
