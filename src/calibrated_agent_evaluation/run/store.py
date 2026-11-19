# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV+JSON round-trip for the RunRecord.

``dump_run_records`` writes ``run_record.csv`` and ``run_record.json`` under a directory and
``load_run_records`` reads them back with every nested part (trend points and their intervals,
input and judge drift, multiplicity results, decisions) rebuilt. A stored ``schema_version`` other
than the running ``RUN_SCHEMA_VERSION``, and every read or parse failure, raises
:class:`~calibrated_agent_evaluation.run.errors.RunStoreError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.run.errors import RunStoreError
from calibrated_agent_evaluation.run.report import RUN_SCHEMA_VERSION, RunRecord

__all__ = ["dump_run_records", "load_run_records"]

_STORE = RecordStore(
    RunRecord,
    stem="run_record",
    schema_version=RUN_SCHEMA_VERSION,
    error=RunStoreError,
    noun="run record",
    rerun="calibrated-agent-evaluation run",
)


def dump_run_records(records: Sequence[RunRecord], path: str | Path, fmt: str = "csv") -> None:
    """Write ``records`` to ``run_record.{csv,json}`` under ``path``.

    Args:
        records: The run records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_run_records` reads back (both are always written).

    Raises:
        RunStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_run_records(path: str | Path, fmt: str = "csv") -> list[RunRecord]:
    """Read run records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_run_records` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        RunStoreError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
