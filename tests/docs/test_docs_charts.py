# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The docs site's charts render in both themes, byte-identically from run to run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("matplotlib", reason="charts need the docs dependency group")


def _load_docs_charts() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "docs" / "site" / "docs_charts.py"
    spec = importlib.util.spec_from_file_location("docs_charts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_chart_renders_in_both_themes_deterministically() -> None:
    charts = _load_docs_charts()
    first = charts.render_all()
    assert sorted(first) == sorted(
        f"{name}-{theme}.svg"
        for name in (*charts.CHARTS, *charts.RAW_CHARTS)
        for theme in ("light", "dark")
    )
    assert first == charts.render_all()
    for svg in first.values():
        assert b"<dc:date>" not in svg
        assert b"http://" not in svg.replace(b"http://www.w3.org/", b"").replace(
            b"http://purl.org/dc/", b""
        ).replace(b"http://creativecommons.org/", b"")
