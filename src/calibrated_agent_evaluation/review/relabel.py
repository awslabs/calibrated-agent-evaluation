# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The relabel rule: a rater's newest label on a unit is the one that counts.

Shared by the agreement gate and ``calibrated-agent-evaluation cluster --ratify``, and kept free of
the statistics engine so the ratification path imports without numpy.
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation.schema import Label

__all__ = ["newest_per_rater_unit", "unit_of"]


def unit_of(label: Label) -> tuple[str, str]:
    """The ``(trace_id, span_id)`` unit a label codes; a trace-level label sorts first."""
    return (label.trace_id, label.span_id or "")


def newest_per_rater_unit(labels: Sequence[Label]) -> tuple[list[Label], int]:
    """Keep each rater's newest label per ``(trace_id, span_id)`` unit, and the drop count.

    ``labels`` must be oldest first. Labels with no ``rater_id`` pass through, since nothing says
    two of them came from one person. Order is kept.
    """
    newest: dict[tuple[str, str, str], int] = {}
    for index, label in enumerate(labels):
        if label.rater_id is not None:
            newest[(label.rater_id, *unit_of(label))] = index
    keep = set(newest.values())
    kept = [lb for i, lb in enumerate(labels) if lb.rater_id is None or i in keep]
    return kept, len(labels) - len(kept)
