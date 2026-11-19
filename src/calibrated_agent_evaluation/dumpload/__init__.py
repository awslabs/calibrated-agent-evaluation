# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extensible dump/load registry for Calibrated Agent Evaluation artifacts.

This layer owns the round-trip *mechanism* and the core ``Trace`` / ``Span`` / ``Label``
codecs; other packages call ``register(artifact_type, to_row, from_row)`` to make their own
artifact types take part in ``dump`` / ``load``. Every artifact round-trips losslessly through
CSV and JSON, so nothing is locked into one store.

The results table (``Estimate``, ``results_table``, ``write_results_csv``) is served lazily: it
imports the record types of ``calibrate``/``run``/``improve``/``bias``/``flagship``, whose stats
chain pulls ``scipy``/``pyarrow``, and ``calibrated-agent-evaluation dump``/``load`` must stay
importable on a bare offline install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.dumpload.io import (
    MANIFEST_NAME,
    CorruptDumpError,
    DumpFileEntry,
    DumpManifest,
    DumpMismatchError,
    IncompleteDumpError,
    UnlistedDumpFileError,
    dump,
    load,
    register_core_codecs,
    verify_dump,
)
from calibrated_agent_evaluation.dumpload.registry import register

if TYPE_CHECKING:
    from calibrated_agent_evaluation.dumpload.results import (
        RESULTS_COLUMNS,
        Estimate,
        ResultsTableError,
        results_table,
        write_results_csv,
    )

_LAZY = frozenset(
    {"RESULTS_COLUMNS", "Estimate", "ResultsTableError", "results_table", "write_results_csv"}
)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from calibrated_agent_evaluation.dumpload import results

        return getattr(results, name)
    raise AttributeError(f"module 'calibrated_agent_evaluation.dumpload' has no attribute {name!r}")


__all__ = [
    "MANIFEST_NAME",
    "RESULTS_COLUMNS",
    "CorruptDumpError",
    "DumpFileEntry",
    "DumpManifest",
    "DumpMismatchError",
    "Estimate",
    "IncompleteDumpError",
    "ResultsTableError",
    "UnlistedDumpFileError",
    "dump",
    "load",
    "register",
    "register_core_codecs",
    "results_table",
    "verify_dump",
    "write_results_csv",
]
