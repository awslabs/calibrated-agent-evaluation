# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrated Agent Evaluation: rigorous, statistically honest evaluation of AI agents.

The distribution is ``calibrated-agent-evaluation`` (not yet on PyPI); the command-line entry point
is ``calibrated-agent-evaluation``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("calibrated-agent-evaluation")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
