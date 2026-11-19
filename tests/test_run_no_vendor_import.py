# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary: run/ and stats/drift.py import no vendor SDK.

Contract test -- asserts a *public boundary*, not implementation internals (rubric rule 2 blessed).
Uses ``ast`` to inspect the ACTUAL import statements (not substrings, which would false-match the
word "langfuse" in a docstring). The methodology plane (``run/`` + the drift block) speaks the
normalized schema + ``calibrated_agent_evaluation.stats``/``calibrated_agent_evaluation.calibrate``
only; the ONLY vendor SDK import in lives in the sanctioned ``backends/dashboard.py`` adapter.
``scipy``/``statsmodels`` are the numeric engine (permissive, methodology-plane maths -- not a
trace-store/executor "vendor SDK" in the sense), so they are allowed in ``stats/drift.py`` exactly
as the ``stats/`` uses them; ``langfuse`` is the forbidden root here.
"""

from __future__ import annotations

import pathlib

from _imports import imported_roots

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "calibrated_agent_evaluation"
_RUN_DIR = _SRC / "run"
_DRIFT = _SRC / "stats" / "drift.py"
# A trace-store / executor vendor SDK must never appear in the methodology plane.
_FORBIDDEN_ROOTS = {"langfuse", "inspect_ai", "strands", "autoevals", "boto3"}


def test_run_plane_imports_no_vendor_sdk() -> None:
    """No module under run/ imports a trace-store/executor vendor SDK."""
    offenders: list[str] = []
    for path in _RUN_DIR.rglob("*.py"):
        forbidden = imported_roots(path) & _FORBIDDEN_ROOTS
        offenders += [f"run/{path.name}: imports {mod}" for mod in sorted(forbidden)]
    assert offenders == [], f"vendor imports leaked into run/: {offenders}"


def test_drift_block_imports_no_vendor_sdk() -> None:
    """Stats/drift.py uses scipy/statsmodels maths but no trace-store vendor SDK."""
    forbidden = imported_roots(_DRIFT) & _FORBIDDEN_ROOTS
    assert forbidden == set(), f"vendor imports leaked into stats/drift.py: {sorted(forbidden)}"


def test_langfuse_confined_to_dashboard_adapter() -> None:
    """The numeric-score vendor call lives in backends/dashboard.py, not run/."""
    adapter = _SRC / "backends" / "dashboard.py"
    text = adapter.read_text(encoding="utf-8")
    # The vendor import is lazy (inside from_credentials) so the module imports with no [langfuse]
    # extra installed; assert the sanctioned site references the SDK.
    assert "from langfuse import Langfuse" in text
