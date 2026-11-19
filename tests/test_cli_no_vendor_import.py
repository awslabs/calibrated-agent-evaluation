# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary: the CLI's offline import path pulls no vendor SDK.

Contract test (rubric rule 2 blessed): asserts a *public boundary* via ``ast`` — no module under
``cli/`` imports a trace-store/executor/observability vendor SDK at **module top level**. The CLI
reaches vendor SDKs (boto3, inspect_ai, strands, fasthtml, langfuse) only through lazy imports
inside functions, so ``import calibrated_agent_evaluation.cli`` on a bare install stays clean. The
adapters themselves live in ``execute/`` (the exemption package), not ``cli/``.
"""

from __future__ import annotations

import ast
import pathlib

_CLI_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "src" / "calibrated_agent_evaluation" / "cli"
)
_FORBIDDEN_ROOTS = {
    "langfuse",
    "inspect_ai",
    "strands",
    "autoevals",
    "boto3",
    "botocore",
    "fasthtml",
}


def _top_level_imported_roots(path: pathlib.Path) -> set[str]:
    """Import roots at MODULE TOP LEVEL only — lazy imports inside functions are allowed
    ."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for (
        node
    ) in tree.body:  # module body only — not ast.walk (which would catch in-function imports)
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_cli_top_level_imports_no_vendor_sdk() -> None:
    """No cli/ module imports a vendor SDK at top level (offline path stays clean)."""
    offenders: list[str] = []
    for path in _CLI_DIR.rglob("*.py"):
        forbidden = _top_level_imported_roots(path) & _FORBIDDEN_ROOTS
        rel = path.relative_to(_CLI_DIR.parent.parent)
        offenders += [f"{rel}: top-level imports {mod}" for mod in sorted(forbidden)]
    assert offenders == [], f"vendor imports leaked into the CLI offline path: {offenders}"
