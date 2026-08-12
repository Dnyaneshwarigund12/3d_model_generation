"""Tests for the Colab Gradio launcher helpers."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import colab_launch  # noqa: E402


def test_free_port_returns_a_bindable_port():
    port = colab_launch.free_port(7900)
    assert 7900 <= port < 7940
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_in_colab_is_false_outside_colab():
    assert colab_launch.in_colab() is False


def test_launch_kwargs_turn_the_colab_iframe_off():
    """This is the setting that stops 'Failed to fetch' on Gradio 6 + Colab."""
    kwargs = colab_launch.launch_kwargs(
        7860, share=True, allowed_paths=["/tmp/out"]
    )
    assert kwargs["inline"] is False
    assert kwargs["share"] is True
    assert kwargs["server_name"] == "0.0.0.0"
    assert kwargs["ssr_mode"] is False
    assert kwargs["debug"] is False
    assert kwargs["prevent_thread_lock"] is True
    assert kwargs["allowed_paths"] == ["/tmp/out"]
