"""Tests for the segmentation stage's dependency reporting.

A version clash inside rembg's own import chain raises ImportError just like a
missing package does, and telling the user to reinstall something pip already
considers satisfied wastes real time. These pin the two messages apart.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from app import segment
from app.errors import SegmentationError


@pytest.fixture
def unimportable_rembg(monkeypatch):
    """Make `from rembg import ...` raise, whatever is really installed."""
    monkeypatch.setitem(sys.modules, "rembg", None)


def test_absent_rembg_says_how_to_install_it(unimportable_rembg, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SegmentationError, match="not installed"):
        segment._import_rembg()


def test_broken_rembg_reports_the_clash_not_a_missing_package(
    unimportable_rembg, monkeypatch
):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    with pytest.raises(SegmentationError) as err:
        segment._import_rembg()

    message = str(err.value)
    assert "failed to import" in message
    assert "not installed" not in message
    assert "restart the runtime" in message.lower()
