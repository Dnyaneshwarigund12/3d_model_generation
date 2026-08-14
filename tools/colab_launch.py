"""Launch the Gradio app so it actually works on Google Colab.

Why this exists
---------------
Gradio 6.4+ renders the app inside Colab's iframe when `inline` is left at its
default. The page HTML loads (so it looks fine), but every subsequent API call -
upload, Generate, even the queue heartbeat - goes to an internal Colab hostname
the browser cannot reach. The page then shows exactly:

    Connection errored out. Failed to fetch

That is not a pipeline failure. It is the frontend talking to a dead address.

What this does instead
----------------------
1. Never embeds the UI in the notebook (`inline=False`).
2. Ensures Gradio's share-tunnel binary (`frpc`) is present, then asks for a
   public `*.gradio.live` link.
3. Always prints Colab's own port-forward URL, which does not depend on Gradio's
   share servers at all.
4. If the Gradio share tunnel still fails, starts a Cloudflare tunnel against the
   same local port and prints that URL.

Usage from the notebook:

    from tools.colab_launch import launch
    launch(generator="triposr")
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def free_port(preferred: int = 7860) -> int:
    for port in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free port between 7860 and 7899.")


def ensure_frpc() -> str:
    """Pre-download Gradio's share-tunnel binary using Gradio's own code.

    Gradio 6 keeps it under HF_HOME/gradio/frpc (not inside the package). Letting
    Gradio download it itself keeps the checksum and filename in sync across
    versions. Doing it before launch means a missing binary shows up as a clear
    error here, not as a silent fall-back to the broken Colab iframe.
    """
    from gradio.tunneling import BINARY_PATH, Tunnel

    Tunnel.download_binary()
    path = Path(BINARY_PATH)
    if not path.exists():
        raise RuntimeError(f"frpc binary missing after download: {path}")
    path.chmod(path.stat().st_mode | 0o111)
    return f"ready at {path}"


def ensure_cloudflared() -> str:
    binary = shutil.which("cloudflared")
    if binary:
        return binary
    deb = Path("/tmp/cloudflared-linux-amd64.deb")
    print("Installing cloudflared (fallback public tunnel) ...")
    subprocess.run(
        [
            "wget",
            "-q",
            "-O",
            str(deb),
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb",
        ],
        check=True,
    )
    subprocess.run(["dpkg", "-i", str(deb)], check=True)
    binary = shutil.which("cloudflared")
    if not binary:
        raise RuntimeError("cloudflared installed but not on PATH")
    return binary


def colab_proxy_url(port: int) -> str | None:
    if not in_colab():
        return None
    from google.colab.output import eval_js

    # Opens as a real browser tab origin, not the broken notebook iframe.
    return eval_js(f"google.colab.kernel.proxyPort({port})")


def start_cloudflare_tunnel(port: int) -> tuple[subprocess.Popen, str | None]:
    binary = ensure_cloudflared()
    process = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url: str | None = None
    deadline = time.time() + 45
    assert process.stdout is not None
    while time.time() < deadline and url is None:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        print(line.rstrip())
        if "trycloudflare.com" in line:
            for token in line.split():
                if token.startswith("https://") and "trycloudflare.com" in token:
                    url = token.strip()
                    break
    return process, url


def launch_kwargs(port: int, *, share: bool, allowed_paths: list[str]) -> dict:
    """Arguments Gradio must receive on Colab. Kept as a function so tests can
    assert the iframe stays off without standing up a server.
    """
    return {
        "server_name": "0.0.0.0",
        "server_port": port,
        "share": share,
        "inline": False,  # the whole point: no Colab iframe
        "inbrowser": False,
        "show_error": True,
        "debug": False,  # debug=True enables a reloader that drops connections
        "ssr_mode": False,
        "quiet": False,
        "max_file_size": "50mb",
        "allowed_paths": allowed_paths,
        "prevent_thread_lock": True,  # return so we can print URLs / fall back
    }


def launch(
    generator: str = "triposr",
    *,
    port: int | None = None,
    low_vram: bool = True,
    texture: bool = False,
) -> None:
    """Block forever serving the app. Prints every URL that is known to work."""
    import gradio as gr

    from app.config import Settings
    from app.ui import build_ui

    # A previous failed launch leaves a dead server on the port; Gradio then
    # either binds elsewhere or serves the corpse. Start clean.
    try:
        gr.close_all()
    except Exception:
        pass

    settings = Settings.from_env()
    settings.generator = generator
    settings.low_vram = low_vram
    settings.hunyuan_texture = texture
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    port = port or free_port()
    on_colab = in_colab()

    # Colab restarts wipe env vars. Re-attach the Drive weight cache before any
    # generator looks under ~/.cache/hy3dgen and misses a finished download.
    from app.generators.hunyuan3d import resolve_hy3dgen_models_dir, shape_checkpoint_path

    hy_cache = resolve_hy3dgen_models_dir()
    print(f"HY3DGEN_MODELS = {hy_cache}")
    if generator == "hunyuan3d":
        ckpt = shape_checkpoint_path(settings)
        if not ckpt.is_file() or ckpt.stat().st_size < 1_000_000_000:
            raise SystemExit(
                "Hunyuan3D checkpoint not ready.\n"
                f"  expected: {ckpt}\n"
                "Re-run notebook step 2, then step 6b (`python tools/download_hunyuan.py`),\n"
                "wait until it prints OK, then launch again.\n"
                "Or set GENERATOR = \"triposr\" to continue without Hunyuan."
            )
        print(f"Hunyuan checkpoint OK ({ckpt.stat().st_size / 1e9:.2f} GB)")

    if on_colab:
        try:
            print("Share tunnel binary:", ensure_frpc())
        except Exception as exc:
            print(f"Could not pre-install frpc ({exc}); will try Gradio's own download.")

    demo = build_ui(settings)
    # One job at a time: TripoSR/Hunyuan on a T4 cannot share the GPU cleanly,
    # and a second request mid-download is a common way to kill the process.
    demo = demo.queue(default_concurrency_limit=1)

    print()
    print("=" * 72)
    print(f"Starting on 0.0.0.0:{port}  generator={generator}")
    print("The UI is NOT embedded in this notebook - that path is broken on")
    print("Gradio 6 + Colab and is what produces 'Failed to fetch'.")
    print("Open one of the URLs printed below in a new browser tab.")
    print("=" * 72)

    share_requested = on_colab
    # Launch in a thread so we can still print the Colab proxy URL and, if
    # needed, start cloudflared after Gradio has bound the port.
    launch_error: list[BaseException] = []
    kwargs = launch_kwargs(
        port,
        share=share_requested,
        allowed_paths=[str(settings.output_dir), str(PROJECT_ROOT)],
    )

    def _serve() -> None:
        try:
            demo.launch(**kwargs)
        except BaseException as exc:  # noqa: BLE001 - surface anything to the notebook
            launch_error.append(exc)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    # Wait until something is listening, or the launch thread died.
    for _ in range(60):
        if launch_error:
            raise launch_error[0]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.5)
    else:
        raise RuntimeError(
            f"Gradio did not bind port {port}. Scroll up for the launch traceback."
        )

    urls: list[tuple[str, str]] = []

    if on_colab:
        try:
            proxy = colab_proxy_url(port)
        except Exception as exc:
            print(f"Colab proxy URL unavailable: {exc}")
            proxy = None
        if proxy:
            urls.append(("Colab port forward (most reliable)", proxy))

    # Gradio stashes the share URL on the Blocks object once the tunnel is up.
    share_url = None
    for _ in range(40):
        share_url = getattr(demo, "share_url", None)
        if share_url:
            break
        if not thread.is_alive() and launch_error:
            break
        time.sleep(0.5)
    if share_url:
        urls.append(("Gradio share link", share_url))

    tunnel_process = None
    if on_colab and not share_url:
        print()
        print("Gradio share link did not come up. Starting Cloudflare tunnel ...")
        try:
            tunnel_process, cf_url = start_cloudflare_tunnel(port)
            if cf_url:
                urls.append(("Cloudflare tunnel", cf_url))
            else:
                print("Cloudflare tunnel started but no URL was parsed; see log above.")
        except Exception as exc:
            print(f"Cloudflare tunnel failed: {exc}")

    print()
    print("=" * 72)
    if urls:
        print("Open ONE of these in a new browser tab:")
        for label, url in urls:
            print(f"  [{label}]")
            print(f"    {url}")
    else:
        print(f"Local only: http://127.0.0.1:{port}")
        print("No public URL could be created. Check the errors above.")
    print()
    print("Leave this cell running. Stopping it kills every URL above.")
    print("First TripoSR / Hunyuan request downloads weights - wait for it.")
    print("=" * 72)

    try:
        # Keep the kernel busy so Colab does not recycle the runtime, and so the
        # daemon Gradio thread is not the only thing holding the process open.
        while thread.is_alive():
            time.sleep(1)
        if launch_error:
            raise launch_error[0]
        # Gradio returned; keep the cell alive while the server serves.
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", port)) != 0:
                    print("Server port closed; exiting.")
                    break
            time.sleep(2)
    finally:
        if tunnel_process is not None and tunnel_process.poll() is None:
            tunnel_process.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", default="triposr")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    launch(generator=args.generator, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
