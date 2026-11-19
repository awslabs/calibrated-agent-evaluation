# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which human label counts per rater, and whether raters agree on the expected output.

Labels are kept in the order they were written, so each rater's newest label is the last one from
that rater. ``calibrated-agent-evaluation score`` and ``calibrated-agent-evaluation dataset build``
both use it to pick each rater's newest label that records an expected output, and to treat raters
who record different ones as conflicting.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from calibrated_agent_evaluation.schema.types import Label

__all__ = ["ExpectedResolution", "newest_per_rater", "resolve_expected"]


def newest_per_rater(labels: Iterable[Label]) -> dict[str | None, Label]:
    """Each rater's newest label among ``labels``, keyed by ``rater_id``."""
    newest: dict[str | None, Label] = {}
    for label in labels:
        newest[label.rater_id] = label
    return newest


@dataclass(frozen=True)
class ExpectedResolution:
    """The one expected output the labels agree on, or why there isn't one."""

    value: dict[str, Any] | None
    conflicting: bool = False


def resolve_expected(labels: Iterable[Label]) -> ExpectedResolution:
    """The ``expected`` the labels' non-null values agree on; conflicting when two differ."""
    values = {
        json.dumps(lb.expected, sort_keys=True): lb.expected
        for lb in labels
        if lb.expected is not None
    }
    if len(values) > 1:
        return ExpectedResolution(value=None, conflicting=True)
    return ExpectedResolution(value=next(iter(values.values()), None))
