# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve an ``--agent module:callable`` the user names on the command line."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.errors import CliError, UsageError

__all__ = ["load_agent"]


def load_agent(spec: str, *, flag: str = "--agent") -> Any:
    """Import ``module`` and return its ``callable``, searching the current directory first.

    A console script puts its own ``bin/`` directory on ``sys.path``, not the current directory,
    so ``--agent myagent:main`` could not find ``./myagent.py``. The current directory goes first
    on ``sys.path``, as ``python -m`` puts it.

    Args:
        spec: ``module:callable``.
        flag: The option name, for the error message.

    Raises:
        CliError: ``spec`` is not ``module:callable``, or it does not resolve.
    """
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise UsageError(f"{flag} must be 'module:callable', got {spec!r}")
    cwd = str(Path.cwd())
    if cwd not in sys.path and "" not in sys.path:
        sys.path.insert(0, cwd)
    try:
        # nosemgrep: non-literal-import  -- the operator's own agent, named on their command line
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise CliError(f"cannot load agent {spec!r}: {exc}") from exc
