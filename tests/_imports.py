# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The static import scan the package-boundary tests share."""

from __future__ import annotations

import ast
from pathlib import Path


def imported_roots(path: Path) -> set[str]:
    """The top-level package of every absolute import in the Python source at ``path``.

    Relative imports name the file's own package, so they are left out.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots
