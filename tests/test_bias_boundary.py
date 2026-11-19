# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Methodology-plane boundary.

``bias/`` is a methodology-plane package: it must import no Langfuse/Inspect SDK type, and the
battery must obtain judge verdicts only through the injected ``ProbeRunner`` port whose signature
takes/returns normalized types. The concrete runner lives in ``judges/`` (and this suite's stub),
never in shipped ``bias/``.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import typing
from pathlib import Path

import calibrated_agent_evaluation.bias as bias_pkg
from calibrated_agent_evaluation.bias import ProbeInput, ProbeVerdict
from calibrated_agent_evaluation.bias.probe import ProbeRunner

_VENDOR_ROOTS = {"langfuse", "inspect_ai"}


def _vendor_imports_in_source(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {
                a.name.split(".")[0] for a in node.names if a.name.split(".")[0] in _VENDOR_ROOTS
            }
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] in _VENDOR_ROOTS
        ):
            found.add(node.module.split(".")[0])
    return found


def test_bias_package_imports_no_vendor_sdk() -> None:
    modules = [
        info.name
        for info in pkgutil.walk_packages(
            bias_pkg.__path__, prefix="calibrated_agent_evaluation.bias."
        )
    ]
    assert modules, "bias/ package must expose modules to scan"
    offenders: dict[str, set[str]] = {}
    for name in modules:
        module = importlib.import_module(name)
        mod_file = getattr(module, "__file__", None)
        if mod_file is None:
            continue
        vendor = _vendor_imports_in_source(Path(mod_file))
        if vendor:
            offenders[name] = vendor
    assert not offenders, f"vendor SDK imported under bias/: {offenders}"


def test_probe_runner_port_signature_is_normalized_typed() -> None:
    # The port the battery depends on takes a ProbeInput and returns a ProbeVerdict — both
    # normalized types, no vendor type in the contract. (get_type_hints resolves the
    # stringified `from __future__ import annotations` forms.)
    hints = typing.get_type_hints(ProbeRunner.run)
    assert hints["probe_input"] is ProbeInput
    assert hints["return"] is ProbeVerdict


def test_scan_catches_a_planted_vendor_import(tmp_path: Path) -> None:
    planted = tmp_path / "leak.py"
    planted.write_text("from inspect_ai import Task\n")
    assert _vendor_imports_in_source(planted) == {"inspect_ai"}
