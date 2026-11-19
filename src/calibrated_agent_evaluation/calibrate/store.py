# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV+JSON round-trip for the CalibrationRecord.

A calibration record spans a hunted set and a separate rate sample, so the trace-store ``dump()``
(which collects only Trace/Span/Label) cannot carry it. ``dump_records`` writes
``calibration_record.csv`` and ``calibration_record.json`` under a directory and ``load_records``
reads them back field for field. A stored ``schema_version`` other than the running
``CALIB_SCHEMA_VERSION`` raises :class:`CalibrationVersionError` before any record is returned;
every other failure is a :class:`CalibrationStoreError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.calibrate.errors import (
    CalibrationStoreError,
    CalibrationVersionError,
)
from calibrated_agent_evaluation.calibrate.record import CALIB_SCHEMA_VERSION, CalibrationRecord

__all__ = ["dump_records", "load_records"]

_STORE = RecordStore(
    CalibrationRecord,
    stem="calibration_record",
    schema_version=CALIB_SCHEMA_VERSION,
    error=CalibrationStoreError,
    version_error=CalibrationVersionError,
    noun="calibration record",
    rerun="calibrated-agent-evaluation calibrate",
)


def dump_records(records: Sequence[CalibrationRecord], path: str | Path, fmt: str = "csv") -> None:
    """Write ``records`` to ``calibration_record.{csv,json}`` under ``path``.

    Args:
        records: The records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_records` reads back (both are always written).

    Raises:
        CalibrationStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_records(path: str | Path, fmt: str = "csv") -> list[CalibrationRecord]:
    """Read calibration records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_records` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        CalibrationVersionError: A stored ``schema_version`` differs from the running one.
        CalibrationStoreError: Any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
