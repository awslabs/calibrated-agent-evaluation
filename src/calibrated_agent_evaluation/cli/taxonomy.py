# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a stage's taxonomy comes from: a ``--taxonomy`` file, a ``--frame`` file, or the store.

A taxonomy file is one JSON object of dimension -> list of values,
``{"feature": ["refund", "exchange"], "persona": ["new", "returning"]}``. A frame file is what
``calibrated-agent-evaluation define --out`` writes: the rows of a discovery frame, each a taxonomy
cell. Without either, the grid is the union of the stored traces' ``taxonomy_cell`` values, which is
empty for a corpus whose traces were never tagged; the refusal then names both ways to supply one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.schema.types import Trace

__all__ = [
    "frame_from_file",
    "frame_to_doc",
    "grid_from_file",
    "grid_from_traces",
]

NO_TAXONOMY = (
    "the stored traces carry no taxonomy_cell values, so there is no taxonomy to build a frame "
    "from. Tag traces at instrumentation time (calibrated_agent_evaluation.taxonomy.<dimension> "
    "span attributes), or "
    'pass --taxonomy FILE, a JSON object of dimension -> values, e.g. {"feature": ["refund", '
    '"exchange"], "persona": ["new", "returning"]}'
)


def _read_json(path: str | Path, what: str) -> Any:
    src = Path(path)
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read {what} {src}: {exc.strerror or exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError(f"{what} {src} is not valid JSON: {exc}") from exc


def grid_from_file(path: str | Path) -> Any:
    """A ``TaxonomyGrid`` from a taxonomy file; a malformed file is a ``CliError``."""
    from calibrated_agent_evaluation.sampling import TaxonomyGrid

    doc = _read_json(path, "taxonomy file")
    if not isinstance(doc, dict) or not doc:
        raise CliError(
            f"taxonomy file {path} must be a non-empty JSON object of dimension -> values"
        )
    dims: dict[str, list[str]] = {}
    for dim, values in doc.items():
        if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
            raise CliError(f"taxonomy file {path}: dimension {dim!r} must map to a list of strings")
        if len(set(values)) != len(values):
            raise CliError(f"taxonomy file {path}: dimension {dim!r} repeats a value")
        dims[dim] = list(values)
    return TaxonomyGrid(dimensions=dims)


def grid_from_traces(traces: Sequence[Trace]) -> tuple[Any, list[str]]:
    """The grid the stored traces' cells span, and the dimensions padded to two values.

    A dimension seen with one value gets a placeholder second value, ``<dim>_synthetic_2``, so the
    covering array has two values to cross. The padded dimensions are returned so the stage can
    say so: a frame row on a placeholder value is a cell no trace was seen in.
    """
    from calibrated_agent_evaluation.sampling import TaxonomyGrid

    dimensions: dict[str, set[str]] = {}
    for trace in traces:
        for dim, val in trace.taxonomy_cell.items():
            dimensions.setdefault(dim, set()).add(val)
    if not dimensions:
        raise CliError(NO_TAXONOMY)
    padded = []
    for dim, vals in dimensions.items():
        if len(vals) < 2:
            vals.add(f"{dim}_synthetic_2")
            padded.append(dim)
    return TaxonomyGrid(dimensions={k: sorted(v) for k, v in dimensions.items()}), sorted(padded)


def frame_to_doc(frame: Any) -> dict[str, Any]:
    """The JSON document ``define --out`` writes for a ``FrameResult``."""
    return {
        "frame": frame.frame,
        "coverage_strength": frame.coverage_strength,
        "rows": [{"cell": dict(r.cell), "frame": r.frame, "weight": r.weight} for r in frame.rows],
    }


def frame_from_file(path: str | Path) -> Any:
    """The ``FrameResult`` a frame file holds; a malformed file is a ``CliError``."""
    from calibrated_agent_evaluation.sampling import FrameResult, FrameRow

    doc = _read_json(path, "frame file")
    try:
        frame = doc["frame"]
        rows = [
            FrameRow(cell=dict(r["cell"]), frame=r["frame"], weight=float(r["weight"]))
            for r in doc["rows"]
        ]
        strength = int(doc["coverage_strength"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise CliError(
            f"frame file {path} is not a calibrated-agent-evaluation define frame: {exc!r}"
        ) from exc
    if frame not in ("discovery", "production-representative") or not rows:
        raise CliError(f"frame file {path} has no rows or an unknown frame {frame!r}")
    if any(r.frame != frame for r in rows):
        raise CliError(f"frame file {path} mixes rows from more than one frame")
    return FrameResult(rows=rows, frame=frame, coverage_strength=strength)
