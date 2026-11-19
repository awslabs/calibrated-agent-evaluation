# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the normalizer suite — paths into the real captured corpus.

The normalizer tests run against the genuinely-captured ``fixtures/raw/`` corpus (produced by the
corpus companion), not hand-shaped placeholders ( design "assume nothing"). These fixtures
expose the raw-file directory and the hand-authored ``fixtures/normalized/`` oracle so tests
can assert structural equality against an independently-authored gold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW = _REPO_ROOT / "fixtures" / "raw"
_NORMALIZED = _REPO_ROOT / "fixtures" / "normalized"


@pytest.fixture
def raw_dir() -> Path:
    """The frozen real-capture corpus directory (``fixtures/raw/``)."""
    return _RAW


@pytest.fixture
def oracle() -> Any:
    """Loader for the hand-authored normalized oracle (``fixtures/normalized/<id>.json``)."""

    def _load(trace_id: str) -> dict[str, Any]:
        return json.loads((_NORMALIZED / f"{trace_id}.json").read_text(encoding="utf-8"))

    return _load
