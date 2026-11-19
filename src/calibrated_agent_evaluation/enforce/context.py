# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The enforcement mode/capability context.

A frozen capability object the CLI maps flags onto (``--ci`` -> ``ci``,
``--allow-likert`` -> ``allow_likert``). Modeled as an explicit object rather
than process-global state so it is testable and thread-safe by construction (no
shared mutable global). Defaults are safe: CI off (generic metrics allowed-but-flagged, not a hard
refusal) and Likert disabled (a Likert axis is refused unless opted in).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnforcementContext:
    """The mode/capabilities a lint or gate call runs under.

    ``ci`` -- while true, generic generation-quality metrics are hard-refused.
    ``allow_likert`` -- while true, a single Likert axis is permitted with a warning;
    disabled by default.
    """

    ci: bool = False
    allow_likert: bool = False
