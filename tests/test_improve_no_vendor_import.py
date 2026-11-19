# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary: improve/ and stats/mcnemar.py import no vendor SDK.

Contract test -- asserts a *public boundary*, not implementation internals (rubric rule 2 blessed).
Uses ``ast`` to inspect the ACTUAL import statements (not substrings, which would false-match a word
in a docstring). The methodology plane (``improve/`` + the ``stats/mcnemar.py`` block) speaks
the normalized schema +
``calibrated_agent_evaluation.{stats,calibrate,run,attribution,sampling,review}`` only; unlike it
has NO ``backends/`` adapter and no vendor write path at all. ``statsmodels``/``scipy`` are the
numeric engine (permissive, methodology-plane maths -- not a trace-store/executor SDK in the sense),
allowed in ``stats/mcnemar.py`` exactly as the ``stats/`` use them; a trace-store / observability /
executor SDK is forbidden anywhere in this plane.
"""

from __future__ import annotations

import pathlib

from _imports import imported_roots

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "calibrated_agent_evaluation"
_IMPROVE_DIR = _SRC / "improve"
_MCNEMAR = _SRC / "stats" / "mcnemar.py"
# A trace-store / executor / observability vendor SDK must never appear in the methodology plane.
_FORBIDDEN_ROOTS = {"langfuse", "inspect_ai", "strands", "autoevals", "boto3"}


def test_improve_plane_imports_no_vendor_sdk() -> None:
    """No module under improve/ imports a trace-store/executor vendor SDK."""
    offenders: list[str] = []
    for path in _IMPROVE_DIR.rglob("*.py"):
        forbidden = imported_roots(path) & _FORBIDDEN_ROOTS
        offenders += [f"improve/{path.name}: imports {mod}" for mod in sorted(forbidden)]
    assert offenders == [], f"vendor imports leaked into improve/: {offenders}"


def test_mcnemar_block_imports_no_vendor_sdk() -> None:
    """Stats/mcnemar.py uses statsmodels/scipy maths but no trace-store vendor SDK."""
    forbidden = imported_roots(_MCNEMAR) & _FORBIDDEN_ROOTS
    assert forbidden == set(), f"vendor imports leaked into stats/mcnemar.py: {sorted(forbidden)}"
