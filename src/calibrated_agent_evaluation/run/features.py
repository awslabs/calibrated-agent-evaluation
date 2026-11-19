# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trace input features that input drift is measured on, and the taxonomy-cell counts.

``calibrated-agent-evaluation run --drift-baseline`` compares one of these features between a
baseline store and the traces in scope, and always compares how the traces fall across taxonomy
cells.

  * ``input-chars``: the number of characters in the trace's input, rendered as JSON.
  * ``turns``: the number of agent steps (spans) in the trace.
  * ``tool-calls``: the number of tool calls across the trace's spans.

A probed trace has no spans, so it has no step or tool-call count to report; it is left out of
``turns`` and ``tool-calls`` and counted, never read as zero.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Literal, get_args

from calibrated_agent_evaluation.run.errors import RunError
from calibrated_agent_evaluation.schema.types import Trace

InputFeature = Literal["input-chars", "turns", "tool-calls"]
INPUT_FEATURES: tuple[str, ...] = get_args(InputFeature)

NO_CELL = "(no cell)"


def _input_chars(trace: Trace) -> float | None:
    return float(len(json.dumps(trace.input, sort_keys=True, ensure_ascii=False)))


def _turns(trace: Trace) -> float | None:
    return float(len(trace.spans)) if trace.spans else None


def _tool_calls(trace: Trace) -> float | None:
    if not trace.spans:
        return None
    return float(sum(len(span.tool_calls) for span in trace.spans))


_EXTRACTORS: dict[str, Callable[[Trace], float | None]] = {
    "input-chars": _input_chars,
    "turns": _turns,
    "tool-calls": _tool_calls,
}


def feature_values(traces: Sequence[Trace], feature: str) -> tuple[list[float], int]:
    """``feature`` for each trace that has it, and how many traces did not.

    Raises :class:`RunError` for a feature name outside :data:`INPUT_FEATURES`.
    """
    extract = _EXTRACTORS.get(feature)
    if extract is None:
        raise RunError(f"unknown input feature {feature!r}; expected one of {list(INPUT_FEATURES)}")
    values: list[float] = []
    missing = 0
    for trace in traces:
        value = extract(trace)
        if value is None:
            missing += 1
        else:
            values.append(value)
    return values, missing


def cell_key(trace: Trace) -> str:
    """The trace's taxonomy cell as one category name; a trace with no cell is its own category."""
    if not trace.taxonomy_cell:
        return NO_CELL
    return ", ".join(f"{k}={v}" for k, v in sorted(trace.taxonomy_cell.items()))


def cell_counts(
    baseline: Sequence[Trace], fresh: Sequence[Trace]
) -> tuple[tuple[str, ...], list[int], list[int]]:
    """Each side's count per taxonomy cell, over the union of cells, in sorted cell order."""
    base = Counter(cell_key(t) for t in baseline)
    new = Counter(cell_key(t) for t in fresh)
    cells = tuple(sorted(set(base) | set(new)))
    return cells, [base[c] for c in cells], [new[c] for c in cells]
