# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrated Agent Evaluation: error analysis and judge calibration for AI agents.

The distribution and the command-line entry point are both ``calibrated-agent-evaluation``.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("calibrated-agent-evaluation")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
