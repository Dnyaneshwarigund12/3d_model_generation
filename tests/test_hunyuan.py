"""Hunyuan weight path / loader wiring (no GPU, no real download)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.errors import GeneratorError
from app.generators import hunyuan3d as hy


def test_resolve_hy3dgen_models_dir_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HY3DGEN_MODELS", str(tmp_path / "cache"))
    path = hy.resolve_hy3dgen_models_dir()
    assert path == tmp_path / "cache"
    assert path.is_dir()


def test_shape_checkpoint_path_is_ckpt_not_safetensors(tmp_path, monkeypatch):
    monkeypatch.setenv("HY3DGEN_MODELS", str(tmp_path))
    ckpt = hy.shape_checkpoint_path(Settings())
    assert ckpt.name == "model.fp16.ckpt"
    assert "safetensors" not in str(ckpt)
    assert ckpt.parent.name == "hunyuan3d-dit-v2-1"


def test_load_shape_calls_from_single_file_with_ckpt(tmp_path, monkeypatch):
    monkeypatch.setenv("HY3DGEN_MODELS", str(tmp_path))
    settings = Settings()
    settings.third_party_dir = tmp_path / "third_party"
    repo = settings.hunyuan_repo
    (repo / "hy3dshape").mkdir(parents=True)

    ckpt = hy.shape_checkpoint_path(settings)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"x" * (hy._MIN_CKPT_BYTES + 1))
    (ckpt.parent / "config.yaml").write_text("vae: {}\n", encoding="utf-8")

    calls: list[dict] = []

    class FakePipeline:
        @classmethod
        def from_single_file(cls, ckpt_path, config_path, **kwargs):
            calls.append({"ckpt": ckpt_path, "config": config_path, **kwargs})
            return cls()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # pragma: no cover
            raise AssertionError("from_pretrained must not be used")

    fake_mod = types.ModuleType("hy3dshape")
    fake_pipelines = types.ModuleType("hy3dshape.pipelines")
    fake_pipelines.Hunyuan3DDiTFlowMatchingPipeline = FakePipeline
    fake_mod.pipelines = fake_pipelines
    monkeypatch.setitem(sys.modules, "hy3dshape", fake_mod)
    monkeypatch.setitem(sys.modules, "hy3dshape.pipelines", fake_pipelines)

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    gen = hy.Hunyuan3DGenerator(settings)
    pipeline = gen._load_shape()

    assert isinstance(pipeline, FakePipeline)
    assert len(calls) == 1
    assert calls[0]["ckpt"] == str(ckpt)
    assert calls[0]["use_safetensors"] is False
    assert calls[0]["ckpt"].endswith(".ckpt")


def test_ensure_shape_weights_rejects_missing_ckpt(tmp_path, monkeypatch):
    monkeypatch.setenv("HY3DGEN_MODELS", str(tmp_path))
    settings = Settings()

    def fake_download(**kwargs):
        (tmp_path / settings.hunyuan_model / settings.hunyuan_shape_subfolder).mkdir(
            parents=True, exist_ok=True
        )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_download),
    )

    gen = hy.Hunyuan3DGenerator(settings)
    with pytest.raises(GeneratorError, match="still missing"):
        gen._ensure_shape_weights()
