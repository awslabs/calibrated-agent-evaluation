# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--trace``: the one-trace scope every pipeline stage accepts, and the whole-store read.

A stage reads ``read_scope`` instead of ``backend.query()``, so ``--trace ID`` narrows it to that
trace and an id the store does not hold is the backend's typed ``BackendError`` (exit 1), never a
silent run over the whole store. Without ``--trace`` the whole store is scanned
(``calibrated_agent_evaluation.backends.scan_traces``): a trace Calibrated Agent Evaluation did not
write, in a Langfuse project shared with another client, is left out, counted and named in the
stage's output by ``skip_lines``, rather than aborting the read or vanishing from it.
"""

from __future__ import annotations

import argparse

from calibrated_agent_evaluation.backends import TraceScan, TraceStoreBackend, scan_traces
from calibrated_agent_evaluation.cli.errors import UsageError
from calibrated_agent_evaluation.schema.types import Trace

__all__ = ["in_cell", "parse_cells", "read_scope", "scope_note", "skip_lines"]

# Skipped ids named in a stage's output, at most this many per reason; the count is always full.
_SHOW = 5


def read_scope(backend: TraceStoreBackend, args: argparse.Namespace) -> TraceScan:
    """The stored traces the stage runs over: the ``--trace`` one, else a scan of the store."""
    trace_id = getattr(args, "trace_id", None)
    if trace_id:
        return TraceScan(traces=[backend.get_trace(trace_id)])
    return scan_traces(backend)


def skip_lines(scan: TraceScan) -> list[str]:
    """One output line per reason a trace was left out of the scan, naming the traces."""
    by_reason: dict[str, list[str]] = {}
    for skipped in scan.skipped:
        by_reason.setdefault(skipped.reason, []).append(skipped.trace_id)
    lines = []
    for reason, ids in by_reason.items():
        shown = ", ".join(repr(i) for i in ids[:_SHOW])
        more = f" and {len(ids) - _SHOW} more" if len(ids) > _SHOW else ""
        lines.append(f"  skipped {len(ids)} trace(s), {reason}: {shown}{more}")
    return lines


def scope_note(args: argparse.Namespace, n: int) -> str:
    """``trace 'X'`` for a ``--trace`` run, else ``N stored trace(s)``."""
    trace_id = getattr(args, "trace_id", None)
    return f"trace {trace_id!r}" if trace_id else f"{n} stored trace(s)"


def parse_cells(raw: list[str] | None) -> dict[str, str] | None:
    """``["feature=refund", "persona=P1"]`` -> ``{"feature": "refund", "persona": "P1"}``."""
    if not raw:
        return None
    cells: dict[str, str] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise UsageError(f"--cell expects KEY=VALUE, got {item!r}")
        cells[key.strip()] = value.strip()
    return cells


def in_cell(trace: Trace, cell: dict[str, str] | None) -> bool:
    """Whether ``trace``'s taxonomy cell has every ``KEY=VALUE`` in ``cell`` (all, for ``None``)."""
    return not cell or all(trace.taxonomy_cell.get(k) == v for k, v in cell.items())
