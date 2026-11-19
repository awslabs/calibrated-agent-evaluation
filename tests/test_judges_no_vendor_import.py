# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary test: no judges/ module imports a vendor SDK.

lives in the methodology plane and must reach Inspect only through the execute/ seam. This
static check parses every module under src/calibrated_agent_evaluation/judges/ and asserts none
imports inspect_ai (or boto3) — the vendor stays behind execute/judge_model.py.
"""

from __future__ import annotations

from pathlib import Path

from _imports import imported_roots

_JUDGES_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "calibrated_agent_evaluation" / "judges"
)
_FORBIDDEN = {"inspect_ai", "boto3"}


def test_no_vendor_import_in_judges_package() -> None:
    """Every judges/ module is free of vendor-SDK imports."""
    offenders: dict[str, set[str]] = {}
    modules = list(_JUDGES_DIR.glob("*.py"))
    assert modules, "expected judges/ modules to scan"
    for module in modules:
        bad = imported_roots(module) & _FORBIDDEN
        if bad:
            offenders[module.name] = bad
    assert not offenders, (
        f"vendor SDK imported inside judges/ (vendor-boundary violation): {offenders}"
    )
